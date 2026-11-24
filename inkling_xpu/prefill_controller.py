"""Prefill controllers with BF16 projections and FP32 normalization and mixing."""

import torch
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice


@triton.jit
def _controller_input(A, B, Out, sa, sb, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    side = tl.program_id(1)
    d = tl.arange(0, BD)
    if side == 0:
        x = tl.load(A + row * sa + d, d < D, 0)
    else:
        x = tl.load(B + row * sb + d, d < D, 0)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
    tl.store(Out + (row * 2 + side) * D + d, x * inv, d < D)


@triton.jit
def _controller_output(A, B, GF, W, Out, Out16, sa, sb, D: tl.constexpr, BD: tl.constexpr, STORE32: tl.constexpr,
                       STORE16: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    live = d < D
    a = tl.load(A + row * sa + d, live, 0)
    b = tl.load(B + row * sb + d, live, 0)
    g = tl.load(GF + row * 2 * D + d, live, 0).to(tl.float32)
    f = tl.load(GF + (row * 2 + 1) * D + d, live, 0).to(tl.float32)
    y = (1.0 - libdevice.tanh(f)) * a + libdevice.tanh(g) * b
    inv = tl.rsqrt(tl.sum(y * y, 0) / D + 1e-6)
    norm = tl.load(W + d, live, 0).to(tl.float32)
    y = y * inv * norm
    if STORE32:
        tl.store(Out + row * D + d, y, live)
    if STORE16:
        tl.store(Out16 + row * D + d, y.to(tl.bfloat16), live)


def controller(cur, msg, norm, down, up, dtypes=(torch.float32,)):
    """dtypes as DepthChunk.mix: FP32 and/or BF16 (RNE of the FP32 result) copies of the output."""
    rows, width = cur.shape
    block = triton.next_power_of_2(width)
    joined = torch.empty((rows, 2 * width), dtype=torch.bfloat16, device=cur.device)
    _controller_input[(rows, 2)](cur, msg, joined, cur.stride(0), msg.stride(0), width, block, num_warps=16)
    hidden = torch.nn.functional.linear(joined, down)
    gate = torch.nn.functional.linear(hidden, up)
    out = torch.empty((rows, width), device=cur.device, dtype=torch.float32) if torch.float32 in dtypes else None
    out16 = torch.empty((rows, width), device=cur.device, dtype=torch.bfloat16) if torch.bfloat16 in dtypes else None
    _controller_output[(rows,)](
        cur,
        msg,
        gate,
        norm,
        out if out is not None else out16,
        out16 if out16 is not None else out,
        cur.stride(0),
        msg.stride(0),
        width,
        block,
        out is not None,
        out16 is not None,
        num_warps=16,
        enable_fp_fusion=False,
    )
    if len(dtypes) == 1:
        return out if out is not None else out16
    return tuple(out if t == torch.float32 else out16 for t in dtypes)


@triton.jit
def _bank_message(A, B, Prob, O, T, layer, D: tl.constexpr, Q: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BC + tl.arange(0, BC)
    a = tl.load(A + row * D + d, d < D, 0)
    b = tl.load(B + row * D + d, d < D, 0)
    for j in tl.static_range(4):
        pair = (row * Q + layer * 4 + j) * 2
        wa = tl.load(Prob + pair)
        wb = tl.load(Prob + pair + 1)
        tl.store(O + (j * T + row) * D + d, a * wa + b * wb, d < D)


class PrefillBank:
    """Cache normalized states and all query probabilities for a two-state repeat bank."""

    def __init__(self, first, last, queries):
        from inkling_trellis.inkling2.fused import rms_norm

        self.a, self.b = rms_norm(first), rms_norm(last)
        self.queries = queries.shape[0]
        scores = torch.stack(
            [torch.nn.functional.linear(state, queries) for state in (self.a, self.b)], dim=-1
        )
        self.probabilities = scores.softmax(-1)

    def __getitem__(self, layer):
        rows, width = self.a.shape
        out = torch.empty((4, rows, width), device=self.a.device, dtype=self.a.dtype)
        _bank_message[(rows, triton.cdiv(width, 256))](
            self.a, self.b, self.probabilities, out, rows, layer, width, self.queries, 256, num_warps=4
        )
        return out
