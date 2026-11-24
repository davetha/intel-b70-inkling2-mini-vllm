"""Reuse immutable residual-bank query scores during an Inkling prefill chunk."""

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_norm(X, Out, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    x = tl.load(X + row * D + d, d < D, 0)
    tl.store(Out + row, tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6))


@triton.jit
def _mix(
    Buf,
    Idx,
    Cur,
    Q,
    Norm,
    Scores,
    Inv,
    Out,
    Out16,
    T,
    offset,
    stride_c,
    stride_q,
    stride_n,
    QCOUNT: tl.constexpr,
    S: tl.constexpr,
    NS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NORMALIZE: tl.constexpr,
    STORE32: tl.constexpr,
    STORE16: tl.constexpr,
):
    row, query = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    live = d < D
    x = tl.load(Cur + row * stride_c + d, live, 0)
    w = tl.load(Q + query * stride_q + d, live, 0)
    inv_cur = tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
    if S > 0:
        si = tl.arange(0, NS)
        slots = tl.load(Idx + si, si < S, 0)
        row_ids = slots * T + row
        inv = tl.load(Inv + row_ids, si < S, 0)
        cached = tl.load(Scores + row_ids * QCOUNT + offset + query, si < S, 0)
        cur_score = tl.sum(x * w, 0) * inv_cur
        logits = tl.where(si < S, cached * inv, tl.where(si == S, cur_score, -float("inf")))
        exp = tl.exp(logits - tl.max(logits, 0))
        factor = exp / tl.sum(exp, 0) * tl.where(si == S, inv_cur, inv)
        y = tl.full((BD,), 0, tl.float32)
        for j in tl.static_range(S):
            slot = tl.load(Idx + j)
            value = tl.load(Buf + (slot * T + row) * D + d, live, 0)
            y += value * tl.sum(tl.where(si == j, factor, 0), 0)
        y += x * tl.sum(tl.where(si == S, factor, 0), 0)
    else:
        y = x * inv_cur
    if NORMALIZE:
        y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-6)
        y *= tl.load(Norm + query * stride_n + d, live, 0)
    if STORE32:
        tl.store(Out + (query * T + row) * D + d, y, live)
    if STORE16:
        # Same RNE rounding as torch's .to(bfloat16) of the FP32 result; saves the separate cast in the consumer. The
        # index is written so Triton sees 4-wide contiguity (as for the FP32 store): an 8-wide BF16 store changes the
        # kernel's layout and with it the reduction order of every sum above (iqr/fuse/mixv.py).
        d4 = (d // 4) * 4 + d % 4
        tl.store(Out16 + (query * T + row) * D + d4, y.to(tl.bfloat16), d4 < D)


class DepthQueries:
    def __init__(self, weights, width):
        self.offsets = {}
        matrices = []
        offset = 0
        for name, value in weights.items():
            if name.startswith(("attn_res.", "middle_cycle_cross_pass")):
                rows = value.reshape(-1, width)
                for j in range(len(rows)):
                    self.offsets[rows[j].data_ptr()] = offset + j
                matrices.append(rows)
                offset += len(rows)
        self.weights = torch.cat(matrices)

    def start(self, bank):
        return DepthChunk(self, bank)


class DepthChunk:
    def __init__(self, queries, bank):
        self.queries = queries
        self.scores = bank.new_empty((*bank.shape[:2], len(queries.weights)))
        self.inv = bank.new_empty(bank.shape[:2])

    def save(self, slot, state):
        state = state.contiguous()
        rows, width = state.shape
        torch.mm(state, self.queries.weights.T, out=self.scores[slot])
        _inverse_norm[(rows,)](state, self.inv[slot], width, triton.next_power_of_2(width), num_warps=16)

    def mix(self, bank, indices, current, queries, norm=None, dtypes=(torch.float32,)):
        """dtypes: output copies to return, (float32,), (bfloat16,) or (float32, bfloat16); one tensor per dtype."""
        if current.stride(-1) != 1:
            current = current.contiguous()
        rows, width = current.shape
        offset = self.queries.offsets[queries.data_ptr()]
        shape = (len(queries), rows, width)
        out = torch.empty(shape, device=current.device, dtype=torch.float32) if torch.float32 in dtypes else None
        out16 = torch.empty(shape, device=current.device, dtype=torch.bfloat16) if torch.bfloat16 in dtypes else None
        count = indices.numel()
        _mix[(rows, len(queries))](
            bank,
            indices if count else self.inv,
            current,
            queries,
            norm if norm is not None else current,
            self.scores,
            self.inv,
            out if out is not None else out16,
            out16 if out16 is not None else out,
            rows,
            offset,
            current.stride(0),
            queries.stride(0),
            norm.stride(0) if norm is not None else 0,
            len(self.queries.weights),
            count,
            triton.next_power_of_2(count + 1),
            width,
            triton.next_power_of_2(width),
            norm is not None,
            out is not None,
            out16 is not None,
            num_warps=16,
            enable_fp_fusion=False,
        )
        if len(dtypes) == 1:
            return out if out is not None else out16
        return tuple(out if t == torch.float32 else out16 for t in dtypes)
