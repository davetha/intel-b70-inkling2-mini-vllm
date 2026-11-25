"""Merge normalized key splits on Intel XPU without materializing a weighted output tensor."""

import os

import torch
import triton
import triton.language as tl


NATIVE_MERGE = os.environ.get("INKLING2_NATIVE_MERGE", "0") == "1"
LONG_VERIFY_SPLITS = os.environ.get("INKLING2_DSPARK_LONG_SPLITS", "0") == "1"
# The same decode XMX kernels built AOT at 256 GRF (no spills, bit-identical; scripts/build-decode-xmx.sh).
DECODE_XMX256 = os.environ.get("INKLING2_DECODE_XMX256", "0") == "1"
_XMX_OPS = None
# INKLING2_DECODE_CASTS=1 (with the 256-GRF library): BF16 q goes to the kernel, which converts it (same values).
DECODE_CASTS = os.environ.get("INKLING2_DECODE_CASTS", "0") == "1"


def _q16(q):
    if DECODE_CASTS and DECODE_XMX256 and q.dtype == torch.bfloat16 and q.is_contiguous():
        return q
    return q.half().contiguous()


def xmx_ops():
    """paged_xmx / local_xmx / draft_global_xmx: inkling2_xpu (JIT, 128 GRF) or inkling2_dxmx (INKLING2_DECODE_XMX_LIB)."""
    global _XMX_OPS
    if _XMX_OPS is None:
        if DECODE_XMX256:
            torch.ops.load_library(os.environ.get("INKLING2_DECODE_XMX_LIB", "/lab/.work/build/inkling2-decode-xmx.so"))
            _XMX_OPS = torch.ops.inkling2_dxmx
        else:
            _XMX_OPS = torch.ops.inkling2_xpu
    return _XMX_OPS


@triton.jit
def _merge(PART, LSE, OUT, heads: tl.constexpr, parts: tl.constexpr, BP: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.program_id(1) * BD + tl.arange(0, BD)
    p = tl.arange(0, BP)
    lse = tl.load(LSE + p * heads + row, mask=p < parts, other=-float("inf"))
    weights = tl.exp(lse - tl.max(lse, 0))
    block = tl.load(
        PART + ((p * heads + row) * 128)[:, None] + dim[None, :], mask=(p < parts)[:, None], other=0.0
    )
    result = tl.sum(block * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(OUT + row * 128 + dim, result)


def merge_splits(partial, lse):
    if NATIVE_MERGE:
        return torch.ops.inkling2_xpu.merge_attention(partial, lse)
    parts, rows, heads, dim = partial.shape
    if dim != 128:
        raise ValueError("Inkling split attention requires head dimension 128")
    result = partial.new_empty((rows, heads, dim))
    _merge[(rows * heads, 8)](
        partial, lse, result, rows * heads, parts, triton.next_power_of_2(parts), 16, num_warps=8
    )
    return result


def paged_splits(tokens, rows=1):
    # B70 single-row XMX decode benefits from more independent key tiles on
    # short prompts and fewer partial results to merge at very long lengths.
    if LONG_VERIFY_SPLITS and rows == 4 and tokens >= 196608:
        return 128
    if rows == 1 and (tokens <= 8192 or tokens >= 131072):
        tile_count = (max(1, tokens) + 15) // 16
        partitions = 1 << (tile_count - 1).bit_length()
        return min(192, max(8, partitions))
    return min(512, max(16, 1 << (max(1, (tokens - 1) // 256).bit_length())))


def paged_xmx(q, key_cache, value_cache, pages, length, pos, rel, proj, sink, parts, packed=False):
    rows, heads, width = q.shape
    partial = q.new_empty((parts, rows, heads, width), dtype=torch.float32)
    lse = q.new_empty((parts, rows, heads), dtype=torch.float32)
    xmx_ops().paged_xmx(
        _q16(q),
        key_cache,
        value_cache,
        pages,
        length,
        pos.int().contiguous(),
        (rel.float() @ proj.float()).contiguous(),
        sink.contiguous(),
        partial,
        lse,
        *([True] if packed else []),
    )
    return merge_splits(partial, lse)


def local_xmx(q, k, v, ring_positions, length, pos, rel, proj, sink):
    rows, heads, width = q.shape
    parts = 16
    partial = q.new_empty((parts, rows, heads, width), dtype=torch.float32)
    lse = q.new_empty((parts, rows, heads), dtype=torch.float32)
    xmx_ops().local_xmx(
        _q16(q),
        k,
        v,
        ring_positions,
        length,
        pos.int().contiguous(),
        (rel.float() @ proj.float()).contiguous(),
        sink.contiguous(),
        partial,
        lse,
    )
    return merge_splits(partial, lse)
