"""Read strided vLLM pages into oneDNN head order and merge partial attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_bits(raw):
    bits = raw.to(tl.uint16)
    exponent = (bits >> 3) & 15
    fraction = bits & 7
    normal = ((exponent + 8) << 10) | (fraction << 7)
    subnormal = (fraction.to(tl.float32) * 0.001953125).to(tl.float16).to(tl.uint16, bitcast=True)
    value = tl.where(exponent == 0, subnormal, normal)
    value = tl.where((bits & 127) == 127, 0x7E00, value)
    return (value | ((bits & 128) << 8)).to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _read(
    K,
    V,
    Blocks,
    KH,
    VH,
    first,
    L,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    stride_page,
    stride_t,
    stride_h,
    BT: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    pos = first + row
    page = tl.load(Blocks + pos // PAGE, row < L, 0)
    src = page.to(tl.int64) * stride_page + pos % PAGE * stride_t + h * stride_h
    valid = row[:, None] < L
    k = tl.load(K + src[:, None] + d[None, :], valid, 0)
    v = tl.load(V + src[:, None] + d[None, :], valid, 0)
    if FP8:
        k = _e4m3_bits(k)
        v = _e4m3_bits(v)
    dst = (h * L + row[:, None]) * D + d[None, :]
    tl.store(KH + dst, k, valid)
    tl.store(VH + dst, v, valid)


def read_heads(key, value, blocks, first, end):
    _, page_tokens, heads, width = key.shape
    n = end - first
    k = torch.empty((heads, n, width), device=key.device, dtype=torch.float16)
    v = torch.empty_like(k)
    fp8 = key.dtype == torch.float8_e4m3fn
    if fp8:
        key, value = key.view(torch.uint8), value.view(torch.uint8)
    _read[(triton.cdiv(n, 8), heads)](
        key, value, blocks, k, v, first, n, heads, width, page_tokens, *key.stride()[:3], 8, fp8, num_warps=4
    )
    return k, v


@triton.jit
def _merge(Out, Lse, Part, OlderLse, T, H: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    old = (i % H) * T + i // H
    live = i < T * H
    a = tl.load(Lse + i, live, 0)
    b = tl.load(OlderLse + old, live, 0)
    maximum = tl.maximum(a, b)
    normalizer = maximum + tl.log(tl.exp(a - maximum) + tl.exp(b - maximum))
    d = tl.arange(0, D)
    x = tl.load(Out + i[:, None] * D + d[None, :], live[:, None], 0)
    y = tl.load(Part + old[:, None] * D + d[None, :], live[:, None], 0).to(tl.float32)
    merged = x * tl.exp(a - normalizer)[:, None] + y * tl.exp(b - normalizer)[:, None]
    tl.store(Out + i[:, None] * D + d[None, :], merged, live[:, None])
    tl.store(Lse + i, normalizer, live)


def merge(out, lse, part, older_lse):
    rows, heads, width = out.shape
    _merge[(triton.cdiv(rows * heads, 8),)](out, lse, part, older_lse, rows, heads, width, 8, num_warps=4)
