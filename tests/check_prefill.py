"""Numerical regression checks for the B70 prefill kernels. Run inside the coding runtime."""

import math

import torch
from inkling_trellis.inkling2.fused import mix, rms_norm

from inkling_xpu.prefill_kernels import local_bias, prefill_conv, prefill_prep

DEVICE = "xpu"


def check_convolution():
    for rows, start in ((1, 0), (7, 65), (63, 70), (251, 0), (257, 3137), (4096, 4096)):
        for dtype, dynamic_taps in ((torch.bfloat16, False), (torch.float32, False), (torch.float32, True)):
            width, offset, ring_size = 1024, 1024, 64
            source = torch.randn(rows, width + 32, device=DEVICE, dtype=dtype)
            x = source[:, :width]
            dynamic = source[:, width:] if dynamic_taps else None
            weights = torch.randn(width, 4, device=DEVICE)
            history = torch.randn(ring_size, 3072, device=DEVICE)
            tags = torch.full((ring_size,), -1, dtype=torch.int64, device=DEVICE)
            past = torch.arange(max(0, start - ring_size), start, device=DEVICE)
            tags[past % ring_size] = past
            tags[(start - 2) % ring_size] = -1
            previous = torch.arange(start - 3, start, device=DEVICE)
            valid = (previous >= 0) & (tags[previous % ring_size] == previous)
            prior = history[previous % ring_size, offset : offset + width] * valid[:, None]
            window = torch.cat([prior, x.float()]).unfold(0, 4, 1)
            filtered = (window * weights).sum(-1)
            if dynamic_taps:
                lag = dynamic.reshape(rows, 8, 4).flip(-1)
                filtered += (window.reshape(rows, 8, 128, 4) * lag[:, :, None]).sum(-1).reshape_as(x)
            expected = (x.float() + filtered) / math.sqrt(9 if dynamic_taps else 5)
            retained = min(rows, ring_size)
            expected_ring = history.clone()
            slots = torch.arange(start + rows - retained, start + rows, device=DEVICE) % ring_size
            expected_ring[slots, offset : offset + width] = x[-retained:].float()
            ring = history.clone()
            actual = prefill_conv(x, weights, ring, tags, start, offset, dynamic)
            assert torch.equal(actual, expected) and torch.equal(ring, expected_ring)


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


def check_attention():
    rows, heads = 257, 24
    source = torch.randn(rows, 6552, device=DEVICE).bfloat16().float()
    k, v = [torch.randn(rows, 1024, device=DEVICE) for _ in range(2)]
    qs = torch.randn(heads, 128, device=DEVICE)
    rs = torch.randn(heads, 16, device=DEVICE)
    projection = torch.randn(16, 1024, device=DEVICE).bfloat16().float()
    q_limit, r_limit = math.sqrt(512), 8.0

    def clip(x, limit):
        return x * (limit / (x.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)

    for start, alpha in ((0, None), (65728, 0.1)):
        pos = torch.arange(start, start + rows, device=DEVICE)
        q, r, g, s = source.split([3072, 384, 3072, 24], -1)
        tau = (
            1 if alpha is None else (1 + alpha * ((pos.float() + 1) / 8192).clamp(min=1).log())[:, None, None]
        )
        expect = (
            (clip(q.reshape(rows, heads, 128) * qs, q_limit) * tau).half(),
            clip(r.reshape(rows, heads, 16) * rs, r_limit) * tau,
            g.sigmoid(),
            s,
            clip(k.reshape(rows, 8, 128), q_limit).bfloat16(),
            rms_norm(v.reshape(rows, 8, 128)).bfloat16(),
        )
        got = prefill_prep(source, k, v, pos, qs, rs, q_limit, r_limit, alpha, 8192)
        assert max(relative_error(a, b) for a, b in zip(got, expect, strict=True)) < 0.00015
        table = local_bias(got[1], projection)
        assert relative_error(table, torch.einsum("thr,re->the", got[1], projection).bfloat16()) < 0.0002


def check_control():
    from inkling_xpu.prefill_controller import controller

    rows, width = 251, 2304
    a, b = [torch.randn(rows, width, device=DEVICE) for _ in range(2)]
    scale = torch.randn(width, device=DEVICE)
    down = (torch.randn(64, 2 * width, device=DEVICE) / math.sqrt(2 * width)).bfloat16()
    up = (torch.randn(2 * width, 64, device=DEVICE) / 8).bfloat16()
    hidden = torch.nn.functional.linear(torch.cat([rms_norm(a), rms_norm(b)], -1).bfloat16(), down)
    gain, forget = torch.nn.functional.linear(hidden, up).float().tanh().chunk(2, dim=-1)
    expected = rms_norm((1 - forget) * a + gain * b, scale)
    assert relative_error(controller(a, b, scale, down, up), expected) < 0.001


def check_depth():
    from inkling_xpu.prefill_depth import DepthQueries

    rows, width = 251, 2304
    bank = torch.randn(6, rows, width, device=DEVICE)
    cur = torch.randn_like(bank[0])
    qs = torch.randn(3, width, device=DEVICE) / math.sqrt(width)
    cache = DepthQueries({"attn_res.test": qs}, width).start(bank)
    for i in range(len(bank)):
        cache.save(i, bank[i])
    for slots in ([], [4], [2, 5, 3]):
        indices = torch.tensor(slots, device=DEVICE, dtype=torch.int32)
        for count, norm in ((1, None), (1, qs[:1]), (3, qs)):
            args = bank, indices, cur, qs[:count], norm
            assert relative_error(cache.mix(*args), mix(*args)) < 3e-5


def check_prefix_io():
    from inkling_xpu.prefix_io import merge, read_heads

    # Every E4M3 encoding in both logical cache views and a shuffled page table.
    words = torch.arange(256, device=DEVICE, dtype=torch.int32).to(torch.uint8)
    raw = torch.empty((3, 8, 16, 256), device=DEVICE, dtype=torch.uint8)
    raw[:] = words
    raw[1] = words.flip(0)
    key, value = raw.view(torch.float8_e4m3fn).transpose(1, 2).split(128, -1)
    blocks = torch.tensor([1, 0, 2], device=DEVICE, dtype=torch.int32)
    start, end = 7, 45
    pos = torch.arange(start, end, device=DEVICE)
    output = read_heads(key, value, blocks, start, end)
    for src, actual in zip((key, value), output, strict=True):
        ref = src.view(torch.uint8)[blocks[pos // 16].long(), pos % 16].view(key.dtype).half().transpose(0, 1)
        assert torch.equal(torch.isnan(ref), torch.isnan(actual))
        live = ~torch.isnan(ref)
        assert torch.equal(ref[live].view(torch.int16), actual[live].view(torch.int16))

    out = torch.randn(71, 24, 128, device=DEVICE)
    lse = torch.randn(71, 24, device=DEVICE) * 40
    part = torch.randn(24, 71, 128, device=DEVICE, dtype=torch.float16)
    partial_lse = torch.randn(24, 71, device=DEVICE) * 40
    normalizer = torch.logaddexp(lse, partial_lse.T)
    expected = out * (lse - normalizer).exp()[:, :, None]
    expected.addcmul_(part.transpose(0, 1), (partial_lse.T - normalizer).exp()[:, :, None])
    merge(out, lse, part, partial_lse)
    assert relative_error(out, expected) < 1e-6 and relative_error(lse, normalizer) < 1e-6


if __name__ == "__main__":
    torch.manual_seed(211)
    check_convolution()
    check_prefix_io()
    check_attention()
    check_control()
    check_depth()
    print(
        "B70 prefill convolution and history, attention preparation, bias, controller and depth checks passed"
    )
