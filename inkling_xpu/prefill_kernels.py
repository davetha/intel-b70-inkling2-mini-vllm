"""Tiled prefill operations for the single-sequence Inkling adapter."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _short_conv(
    X,
    W,
    Dynamic,
    History,
    Tags,
    Res,
    Out,
    T,
    start,
    offset,
    ring_stride,
    stride_x,
    stride_dyn,
    RING: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    HAS_DYNAMIC: tl.constexpr,
    HAS_RES: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    live = (row[:, None] < T) & (col[None, :] < WIDTH)
    now = tl.load(X + row[:, None] * stride_x + col[None, :], live, 0).to(tl.float32)
    acc = tl.full((ROWS, COLS), 0, tl.float32)
    dyn_acc = tl.full((ROWS, COLS), 0, tl.float32)
    for tap in tl.static_range(4):
        rel = row - (3 - tap)
        value = tl.load(X + rel[:, None] * stride_x + col[None, :], live & (rel[:, None] >= 0), 0).to(
            tl.float32
        )
        if tap != 3 and tl.program_id(0) == 0:
            pos = start + rel
            slot = pos % RING
            use = (rel < 0) & (pos >= 0)
            tag = tl.load(Tags + slot, use, -1)
            past = tl.load(
                History + slot[:, None] * ring_stride + offset + col[None, :],
                live & (use & (tag == pos))[:, None],
                0,
            )
            value = tl.where(rel[:, None] < 0, past, value)
        weight = tl.load(W + col * 4 + tap, col < WIDTH, 0)
        acc += value * weight[None, :]
        if HAS_DYNAMIC:
            dyn = tl.load(
                Dynamic + row[:, None] * stride_dyn + (col[None, :] // 128) * 4 + (3 - tap), live, 0
            )
            dyn_acc += value * dyn.to(tl.float32)
    if HAS_DYNAMIC:
        result = (now + (acc + dyn_acc)) * (1.0 / 3.0)
    else:
        result = (now + acc) * (1.0 / 2.23606797749979)
    if HAS_RES:
        # The caller's residual add (residual + conv), fused; FP32 addition is commutative, so this is bit-exact.
        result = tl.load(Res + row[:, None] * WIDTH + col[None, :], live, 0) + result
    tl.store(Out + row[:, None] * WIDTH + col[None, :], result, live)


def prefill_conv(x, weight, history, tags, start, offset, dynamic=None, residual=None):
    rows, channels = x.shape
    if residual is not None and (residual.dtype != torch.float32 or residual.shape != x.shape
                                 or not residual.is_contiguous()):
        raise ValueError("prefill_conv residual must be a contiguous FP32 tensor shaped like x")
    ring_size = history.shape[0]
    out = torch.empty((rows, channels), device=x.device, dtype=torch.float32)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if dynamic is not None and dynamic.stride(-1) != 1:
        dynamic = dynamic.contiguous()
    weight = weight.contiguous()
    _short_conv[(triton.cdiv(rows, 8), triton.cdiv(channels, 128))](
        x,
        weight,
        dynamic if dynamic is not None else x,
        history,
        tags,
        residual if residual is not None else out,
        out,
        rows,
        start,
        offset,
        history.stride(0),
        x.stride(0),
        dynamic.stride(0) if dynamic is not None else 0,
        ring_size,
        channels,
        8,
        128,
        dynamic is not None,
        residual is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # All workgroups must finish reading the old history before any new slot is written.
    retained = min(rows, ring_size)
    slots = torch.arange(start + rows - retained, start + rows, device=x.device) % ring_size
    history[slots, offset : offset + channels] = x[-retained:].float()
    return out


@triton.jit
def _local_bias(
    R, P, B, T, H: tl.constexpr, E: tl.constexpr, DR: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr
):
    flat = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, DR)
    distance = tl.program_id(1) * COLS + tl.arange(0, COLS)
    r = tl.load(R + flat[:, None] * DR + d[None, :], flat[:, None] < T * H, 0)
    p = tl.load(P + d[:, None] * E + distance[None, :])
    # Accumulate the FP16 high product and first-order residuals in FP32.
    # Write BF16 directly so the full FP32 distance table is never allocated.
    rh = r.to(tl.float16)
    rl = (r - rh.to(tl.float32)).to(tl.float16)
    ph = p.to(tl.float16)
    pl = (p - ph.to(tl.float32)).to(tl.float16)
    correction = tl.dot(rl, ph) + tl.dot(rh, pl)
    bias = tl.dot(rh, ph) + correction
    addr = flat[:, None] * E + distance[None, :]
    tl.store(B + addr, bias, flat[:, None] < T * H)


def local_bias(r, projection):
    rows, heads, width = r.shape
    extent = projection.shape[-1]
    if (heads, width, extent) != (24, 16, 1024):
        raise ValueError("Unexpected Inkling local relative projection geometry")
    table = torch.empty((rows, heads, extent), device=r.device, dtype=torch.bfloat16)
    # Tile (rows, cols, warps): every tile computes each element with the same three K=16 dots, so the table is bitwise
    # identical for any tile; 64x256x8 writes it ~1.9x faster than 32x64x4 on the B70 (results/IQR-prefill-profile.md).
    tr, tc, tw = _LOCAL_BIAS_TILE
    _local_bias[(triton.cdiv(rows * heads, tr), extent // tc)](
        r.contiguous(),
        projection.contiguous(),
        table,
        rows,
        heads,
        extent,
        width,
        tr,
        tc,
        num_warps=tw,
    )
    return table


_LOCAL_BIAS_TILE = tuple(int(v) for v in (os.environ.get("INKLING2_LOCAL_BIAS_TILE") or "32,64,4").split(","))


@triton.jit
def _prefill_prep(
    QRGS,
    K,
    V,
    Pos,
    QS,
    RS,
    QO,
    RO,
    GO,
    SO,
    KO,
    VO,
    stride,
    sk,
    sv,
    qlim,
    rlim,
    alpha,
    floor,
    TAU: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1) * 4 + tl.arange(0, 4)
    d = tl.arange(0, 128)
    dr = tl.arange(0, 16)
    if TAU:
        p = tl.load(Pos + t).to(tl.float32)
        tau = 1.0 + alpha * tl.log(tl.maximum((p + 1.0) / floor, 1.0))
    else:
        tau = 1.0
    qoff = h[:, None] * 128 + d[None, :]
    roff = h[:, None] * 16 + dr[None, :]
    q = tl.load(QRGS + t * stride + qoff) * tl.load(QS + qoff)
    r = tl.load(QRGS + t * stride + H * 128 + roff) * tl.load(RS + roff)
    q *= tl.minimum(qlim / (tl.sqrt(tl.sum(q * q, 1)) + 1e-6), 1.0)[:, None]
    r *= tl.minimum(rlim / (tl.sqrt(tl.sum(r * r, 1)) + 1e-6), 1.0)[:, None]
    tl.store(QO + t * H * 128 + qoff, q * tau)
    tl.store(RO + t * H * 16 + roff, r * tau)
    gate = tl.load(QRGS + t * stride + H * 144 + qoff).to(tl.float32)
    tl.store(GO + t * H * 128 + qoff, tl.sigmoid(gate))
    tl.store(SO + t * H + h, tl.load(QRGS + t * stride + H * 272 + h))
    if tl.program_id(1) < HK // 4:
        k = tl.load(K + t * sk + qoff)
        v = tl.load(V + t * sv + qoff)
        k *= tl.minimum(qlim / (tl.sqrt(tl.sum(k * k, 1)) + 1e-6), 1.0)[:, None]
        v *= tl.rsqrt(tl.sum(v * v, 1) / 128.0 + 1e-6)[:, None]
        tl.store(KO + t * HK * 128 + qoff, k)
        tl.store(VO + t * HK * 128 + qoff, v)


def prefill_prep(qrgs, k, v, pos, qscale, rscale, qlim, rlim, alpha, floor):
    rows = qrgs.shape[0]
    heads, kv_heads = 24, 8

    def alloc(shape, dtype):
        return torch.empty((rows, *shape), device=k.device, dtype=dtype)

    q = alloc((heads, 128), torch.float16)
    r = alloc((heads, 16), torch.float32)
    gate = alloc((heads * 128,), torch.float32)
    sink = alloc((heads,), torch.float32)
    key = alloc((kv_heads, 128), torch.bfloat16)
    value = torch.empty_like(key)
    _prefill_prep[(rows, heads // 4)](
        qrgs,
        k,
        v,
        pos,
        qscale,
        rscale,
        q,
        r,
        gate,
        sink,
        key,
        value,
        qrgs.stride(0),
        k.stride(0),
        v.stride(0),
        qlim,
        rlim,
        alpha if alpha is not None else 0.0,
        float(floor),
        alpha is not None,
        heads,
        kv_heads,
        num_warps=4,
    )
    return q, r, gate, sink, key, value


_PACKED_PROJECTIONS = {}


def packed_projection(proj):
    """FP16 high/low split of the [16, E] relative projection as prefill_attn_rel reads it: int32 [8, E] K-pair packed
    (low half = even row), distance axis reversed. Same split as the Triton table kernel (RNE casts)."""
    key = (proj.data_ptr(), tuple(proj.shape))
    packed = _PACKED_PROJECTIONS.get(key)
    if packed is None:
        width, extent = proj.shape
        if width != 16 or extent % 16:
            raise ValueError("Unexpected Inkling local relative projection geometry")
        hi = proj.half()
        lo = (proj - hi.float()).half()
        packed = tuple(t.flip(-1).reshape(8, 2, extent).permute(0, 2, 1).contiguous().view(torch.int32).reshape(8, extent)
                       for t in (hi, lo))
        _PACKED_PROJECTIONS[key] = packed
    return packed


_REL_OPS = None


def rel_ops():
    """torch.ops.inkling2_rel (INKLING2_REL_LIB, scripts/build-rel-attention.sh): the 256-GRF AOT prefill attention."""
    global _REL_OPS
    if _REL_OPS is None:
        torch.ops.load_library(os.environ.get("INKLING2_REL_LIB", "/lab/.work/build/inkling2-rel-attention.so"))
        _REL_OPS = torch.ops.inkling2_rel
    return _REL_OPS


def attention_prefill_rel(q, k, v, qpos0, kpos0, r, proj, sink, *, window, qtw):
    """attention_prefill_esimd with the local relative bias computed in the kernel from r [n, H, 16] and proj [16, E]."""
    hi, lo = packed_projection(proj)
    o = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    rel_ops().prefill_attn_rel(q.to(torch.float16).contiguous(), k.to(torch.float16).contiguous(),
                                            v.to(torch.float16).contiguous(), o, r.float().contiguous(), hi, lo,
                                            sink.float().contiguous(), int(qpos0), int(kpos0), int(window), int(qtw))
    return o


@triton.jit
def _global_bias(R, P, B, N, E: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    flat = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    live = flat < N
    acc = tl.zeros((ROWS, COLS), tl.float32)
    # One FMA chain over the 16 relative channels in order: bitwise what the FP32 einsum (oneDNN) computes here.
    for d in tl.static_range(16):
        a = tl.load(R + flat * 16 + d, live, 0)
        acc = tl.fma(a[:, None], tl.load(P + d * E + col)[None, :], acc)
    tl.store(B + flat[:, None] * E + col[None, :], acc.to(tl.bfloat16), live[:, None])


def global_bias(r, projection):
    """torch.einsum("thr,re->the", r, projection).to(bfloat16) in one pass (no FP32 table)."""
    rows, heads, width = r.shape
    extent = projection.shape[-1]
    if width != 16 or extent % 128:
        raise ValueError("Unexpected Inkling relative projection geometry")
    r, projection = r.float().contiguous(), projection.float().contiguous()
    table = torch.empty((rows, heads, extent), device=r.device, dtype=torch.bfloat16)
    _global_bias[(triton.cdiv(rows * heads, 64), extent // 128)](r, projection, table, rows * heads, extent, 64, 128,
                                                                 num_warps=8, enable_fp_fusion=False)
    return table


def attention_prefill_table(q, k, v, qpos0, kpos0, table, sink, *, window=0, qtw=4, lse=None):
    """attention_prefill_esimd on the 256-GRF build (inkling2_rel.prefill_attn); same outputs, bit for bit."""
    o = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    rel_ops().prefill_attn(q.to(torch.float16).contiguous(), k.to(torch.float16).contiguous(), v.to(torch.float16).contiguous(),
                           o, table.to(torch.bfloat16).contiguous(), sink.float().contiguous(), int(qpos0), int(kpos0),
                           int(window), int(qtw), lse)
    return o


@triton.jit
def _gate_bf16(O, G, Out, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < N
    o = tl.load(O + i, live, 0)
    g = tl.load(G + i, live, 0)
    tl.store(Out + i, (o * g).to(tl.bfloat16), live)


def gate_bf16(o, gate):
    """bf16(o * gate) in one pass: the attention output gate and the cast in front of the BF16 output projection."""
    if o.dtype != torch.float32 or gate.dtype != torch.float32 or o.shape != gate.shape:
        raise ValueError("gate_bf16 expects FP32 tensors of one shape")
    o, gate = o.contiguous(), gate.contiguous()
    out = torch.empty(o.shape, device=o.device, dtype=torch.bfloat16)
    n = o.numel()
    _gate_bf16[(triton.cdiv(n, 4096),)](o, gate, out, n, 4096, num_warps=16, enable_fp_fusion=False)
    return out


def bf16_prefill_linear(linear, x):
    if x.shape[0] <= linear.max_gemv_rows or linear.w is None or linear.w8a8 is not None:
        return linear(x)
    return torch.nn.functional.linear(x.to(torch.bfloat16), linear.w)
