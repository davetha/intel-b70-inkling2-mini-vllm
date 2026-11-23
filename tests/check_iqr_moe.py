"""IQR expert kernel (iqr_moe) checks and microbenchmark.

check: real layer weights (q file = gptq.py output or an iqr-layer-XX.safetensors) and real layer inputs (calib visit
  dump), real router. (1) kernel int8 activations == exact-arithmetic reference (bit-exact), (2) kernel output vs an
  fp64 reference of the same quantized arithmetic, for decode (M<=8) and prefill M, (3) decode vs prefill rows,
  (4) vs the dense fake-quant (+a8, evalkl math) on the dense/<var> bf16 weights, (5) graph replay == eager (decode
  M=8, prefill M=9/16/48), (6) forced routing (all 8 rows on the same 8 experts: decode RC=8 branch), (7) invalid ids
  -> NaN rows (or a RuntimeError with INKLING2_IQR_CHECK_IDS=1).
bench: random arenas cycled past L2; IQR vs EXL3 (prod knobs: prefill_min 48, pair_max 8, tiny DPAS 4/9/18) for decode
  (graph-replayed 48-call sequences) and prefill M.
usage: check_iqr_moe.py check --layer L --q FILE --dense FILE --x FILE [--shard NVFP4_SHARD]
       check_iqr_moe.py bench [--rows ...] [--pool P] [--layers N] [--no-exl3]
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from inkling_xpu.iqr_experts import ops, pack_layer

H, I, E, TOPK = 2304, 1152, 128, 8
dev = torch.device("xpu")
torch.set_grad_enabled(False)
INV_SQRT128 = torch.tensor(0.08838834764831845, dtype=torch.float32)


def fwht(v):
    """Unnormalized WHT over the last dim (128), same butterfly order and fp32 ops as the kernel."""
    shp = v.shape
    h = 1
    while h < 128:
        w = v.reshape(-1, 128 // (2 * h), 2, h)
        v = torch.stack([w[:, :, 0] + w[:, :, 1], w[:, :, 0] - w[:, :, 1]], 2).reshape(shp)
        h *= 2
    return v


def rotq(v):
    """[N, 128] fp32 -> (q int, scale fp32, sum) exactly as rotq_store."""
    r = fwht(v.float())
    amax = r.abs().amax(-1)
    inv = torch.where(amax > 0, torch.tensor(127.0) / amax, torch.zeros_like(amax))
    q = torch.round(r * inv[:, None])
    s = (amax * torch.tensor(1.0 / 127.0, dtype=torch.float32)) * INV_SQRT128
    return q, s, q.sum(-1)


def quant_rows_ref(x):
    M = x.shape[0]
    q, s, su = rotq(x.float().reshape(-1, 128))
    return q.reshape(M, -1), s.reshape(M, -1), su.reshape(M, -1)


def act(g, u):
    cg = 4.0 * (2.0 / (1.0 + torch.exp(-g * 0.5)) - 1.0)
    cu = 25.0 * (2.0 / (1.0 + torch.exp(-u * 0.08)) - 1.0)
    return cg / (1.0 + torch.exp(-g)) * cu


def unpack(packed):
    E_, R, Kh = packed.shape
    return torch.stack([packed & 15, packed >> 4], -1).reshape(E_, R, 2 * Kh).to(torch.int16) - 8


def proj_ref(codes, scales, q, s, su):
    """codes [R, K] int (value), scales [R, G] bf16, q [K], s [G] -> [R] fp64 (kernel arithmetic, zero folded)."""
    R, K = codes.shape
    G = K // 128
    dots = (codes.double().view(R, G, 128) * q.double().view(1, G, 128)).sum(-1)       # exact integers
    return (dots * scales.double() * s.double().view(1, G)).sum(-1)


def iqr_ref(x, ids, w, L):
    """fp64 reference of the kernel arithmetic. L = dict of unpacked codes/scales (CPU)."""
    xq, xs, _ = quant_rows_ref(x.cpu())
    M = x.shape[0]
    y = torch.zeros(M, H, dtype=torch.float64)
    for m in range(M):
        for k in range(TOPK):
            e = int(ids[m, k])
            gu = proj_ref(L["w13"][e], L["w13_scales"][e], xq[m], xs[m], None)
            h = act(gu[0::2].float(), gu[1::2].float())
            hq, hs, _ = rotq(h.view(-1, 128))
            dn = proj_ref(L["w2"][e], L["w2_scales"][e], hq.reshape(-1), hs, None)
            y[m] += float(w[m, k]) * dn
    return y


def a8(x):
    shp = x.shape
    xr = (x.float().reshape(-1, 128) @ HAD).reshape(*shp[:-1], -1, 128)
    s = xr.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 127.0
    q = torch.round(xr / s).clamp(-127, 127) * s
    return (q.reshape(-1, 128) @ HAD).reshape(shp)


def dense_ref(x, ids, w, d13, d2):
    """evalkl's +a8 expert math on the dense fake-quant weights (fp32 out)."""
    out = torch.zeros(x.shape[0], H, device=x.device)
    xin = a8(x)
    for ex in ids.unique().tolist():
        rows, slot = (ids == ex).nonzero(as_tuple=True)
        hh = F.linear(xin[rows].to(torch.bfloat16), d13[ex]).float()
        gte, up = hh[:, 0::2], hh[:, 1::2]
        a = 4.0 * torch.tanh(gte / 4.0) * torch.sigmoid(gte) * (25.0 * torch.tanh(up / 25.0))
        y = F.linear(a8(a).to(torch.bfloat16), d2[ex]).float()
        out.index_add_(0, rows, y * w[rows, slot, None])
    return out


def rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm()).item()


def route(x, gw, gb):
    probs = torch.sigmoid(F.linear(x.float(), gw.float()))
    topi = (probs + gb.float()).topk(TOPK, -1).indices
    topw = probs.gather(1, topi)
    return topi.to(torch.int32), (topw / topw.sum(-1, keepdim=True) * 8.0).float()


def load_q(path, layer):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        t = load_file(path)
        p = f"model.llm.layers.{layer}.mlp.experts."
        return {k: t[p + k] for k in ("w13_codes", "w13_scales", "w2_codes", "w2_scales")}
    d = torch.load(path)
    return {k: d[k] for k in ("w13_codes", "w13_scales", "w2_codes", "w2_scales")}


def mode_check(a):
    global HAD
    import numpy as np
    from safetensors import safe_open
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    HAD = (h / math.sqrt(128)).float().to(dev)
    qd = load_q(a.q, a.layer)
    arena = pack_layer(*(qd[k].to(dev) for k in ("w13_codes", "w13_scales", "w2_codes", "w2_scales"))).contiguous()
    assert arena.shape == (E, ops().blob_bytes()), arena.shape
    L = {"w13": unpack(qd["w13_codes"]), "w2": unpack(qd["w2_codes"]), "w13_scales": qd["w13_scales"], "w2_scales": qd["w2_scales"]}
    f = safe_open(a.shard, "pt", device="cpu")
    gw = f.get_tensor(f"model.llm.layers.{a.layer}.mlp.gate.weight").to(dev)
    gb = f.get_tensor(f"model.llm.layers.{a.layer}.mlp.gate.bias").to(dev)
    mm = np.memmap(a.x, dtype=np.int16, mode="r").reshape(-1, H)
    xs_all = torch.from_numpy(np.ascontiguousarray(mm[a.row0:a.row0 + 4096])).view(torch.bfloat16).to(dev)
    ok = True
    # (1) activation quantization bit-exact
    q, s, su = ops().quant_rows(xs_all[:512].contiguous())
    rq, rs, rsu = quant_rows_ref(xs_all[:512].cpu())
    eq = (q.cpu().float() == rq).all().item() and torch.equal(s.cpu(), rs) and torch.equal(su.cpu(), rsu.int())
    print(f"quant_rows bit-exact vs reference: {eq}  (code mismatches {(q.cpu().float() != rq).sum().item()})")
    ok &= eq
    # (2) kernel vs fp64 reference
    for M in a.check_rows:
        x = xs_all[:M].contiguous()
        ids, w = route(x, gw, gb)
        out = ops().moe_forward(x, ids, w, arena)
        torch.xpu.synchronize()
        nref = M if M == a.ref_full else min(M, a.ref_rows)
        y = iqr_ref(x[:nref], ids[:nref].cpu(), w[:nref].cpu(), L)
        r = rel(out[:nref].float(), y)
        same = (out[:nref].cpu() == y.to(torch.bfloat16)).float().mean().item()
        good = r < 3e-3
        ok &= good
        print(f"M={M:5d} ({'decode' if M <= 8 else 'prefill'}) vs fp64 ref ({nref} rows): rel_l2 {r:.2e}, bf16-identical "
              f"{100 * same:.2f}%  {'PASS' if good else 'FAIL'}")
    # (3) decode rows vs the same rows inside a prefill call, (4) vs dense fake-quant
    x8 = xs_all[:8].contiguous(); ids8, w8 = route(x8, gw, gb)
    xp = xs_all[:256].contiguous(); idsp, wp = route(xp, gw, gb)
    od = ops().moe_forward(x8, ids8, w8, arena)
    op = ops().moe_forward(xp, idsp, wp, arena)
    r = rel(od.float(), op[:8].float())
    print(f"decode M=8 vs the same rows in a prefill M=256 call: rel_l2 {r:.2e}, identical {torch.equal(od, op[:8])}")
    ok &= r < 2e-3
    if a.dense:
        dd = torch.load(a.dense)
        d13, d2 = dd["w13"].to(dev), dd["w2"].to(dev)
        for M in (8, 1024):
            x = xs_all[:M].contiguous(); ids, w = route(x, gw, gb)
            out = ops().moe_forward(x, ids, w, arena).float()
            r = rel(out, dense_ref(x, ids.long(), w, d13, d2))
            print(f"M={M}: vs dense fake-quant +a8 (evalkl math): rel_l2 {r:.2e}")
            ok &= r < 2e-2
    # (5) graph replay == eager: capture on rows [0, M), replay on rows [M, 2M); the id check syncs, so it refuses capture
    strict = os.environ.get("INKLING2_IQR_CHECK_IDS") == "1"
    for M in (() if strict else (8, 9, 16, 48)):
        x1 = xs_all[:M].contiguous(); i1, w1 = route(x1, gw, gb)
        g = torch.xpu.XPUGraph()
        xg = x1.clone(); ig = i1.clone(); wg = w1.clone()
        with torch.xpu.graph(g):
            og = ops().moe_forward(xg, ig, wg, arena)
        x2 = xs_all[M:2 * M].contiguous(); i2, w2_ = route(x2, gw, gb)
        xg.copy_(x2); ig.copy_(i2); wg.copy_(w2_)
        g.replay(); torch.xpu.synchronize()
        oe = ops().moe_forward(x2, i2, w2_, arena)
        same = torch.equal(og, oe)
        print(f"graph replay M={M} (new inputs) == eager: {same}")
        ok &= same
    # (6) forced routing: every row on row 0's 8 experts (8 pairs per expert -> decode RC=8 branch)
    xf = xs_all[:9].contiguous(); idf, wf = route(xf, gw, gb)
    idf = idf[:1].expand(9, TOPK).contiguous()
    od = ops().moe_forward(xf[:8].contiguous(), idf[:8].contiguous(), wf[:8].contiguous(), arena)
    op9 = ops().moe_forward(xf, idf, wf, arena)
    torch.xpu.synchronize()
    r = rel(od.float(), iqr_ref(xf[:8], idf[:8].cpu(), wf[:8].cpu(), L))
    r9 = rel(od.float(), op9[:8].float())
    good = r < 3e-3 and r9 < 2e-3
    print(f"forced routing (8 rows x same 8 experts): decode vs fp64 ref rel_l2 {r:.2e}, vs prefill M=9 rows {r9:.2e}  "
          f"{'PASS' if good else 'FAIL'}")
    ok &= good
    if strict:
        try:
            with torch.xpu.graph(torch.xpu.XPUGraph()):
                ops().moe_forward(x8, ids8, w8, arena)
            good = False
        except RuntimeError as ex:
            good = "graph capture" in str(ex)
        print(f"INKLING2_IQR_CHECK_IDS=1 refuses graph capture with a clear error: {good}")
        ok &= good
    # (7) invalid ids: out-of-range expert (decode, prefill) and > 8 pairs on one expert (decode) -> NaN rows only
    for M, name in ((4, "range"), (16, "range"), (8, "dup")):
        x1 = xs_all[:M].contiguous(); i1, w1 = route(x1, gw, gb)
        i1 = i1.clone(); bad_rows = [1]
        if name == "range":
            i1[1, 3] = E
        else:
            e0 = int(i1[2, 0]); i1[1] = e0; bad_rows = sorted({1} | set((i1 == e0).any(1).nonzero().flatten().tolist()))
        try:
            o = ops().moe_forward(x1, i1, w1, arena); torch.xpu.synchronize()
            nan = o.float().isnan().any(1).cpu()
            good = not strict and nan.nonzero().flatten().tolist() == bad_rows
            print(f"invalid ids ({name}, M={M}): NaN rows {nan.nonzero().flatten().tolist()} (expected {bad_rows})  "
                  f"{'PASS' if good else 'FAIL'}")
        except RuntimeError as ex:
            good = strict
            print(f"invalid ids ({name}, M={M}): raised {str(ex).splitlines()[0][:80]!r}  {'PASS' if good else 'FAIL'}")
        ok &= good
    # determinism
    o1 = ops().moe_forward(xp, idsp, wp, arena); o2 = ops().moe_forward(xp, idsp, wp, arena)
    det = torch.equal(o1, o2) and torch.equal(o1, op)
    print(f"prefill M=256 repeat-identical: {det}")
    ok &= det
    print("CHECK", "ALL PASS" if ok else "FAILED")


def mode_bench(a):
    blob = ops().blob_bytes()
    arenas = []
    for i in range(a.layers):
        arenas.append(torch.randint(0, 256, (E, blob), dtype=torch.uint8, device=dev))
    # random bytes can be inf/nan as bf16: make every scale finite (planes at [GU_CODES, +GU_SC), [.. + DN_CODES, +DN_SC))
    gu_c, gu_s, dn_c, dn_s = 144 * 18 * 1024, 144 * 18 * 32, 144 * 9 * 1024, 144 * 9 * 32
    for t in arenas:
        for off, n in ((gu_c, gu_s), (gu_c + gu_s + dn_c, dn_s)):
            v = t[:, off:off + n].view(torch.int16)
            v.copy_((v & 0x7F) | 0x3B80)
    exl3 = None
    if not a.no_exl3:
        from inkling_trellis.moe_offload import ops as eops, s64
        o = eops()
        o.moe_set_prefill_min_m(48); o.moe_set_pair_max_m(8); o.moe_set_tiny_dpas(4, 9, 18)
        eb = o.blob_bytes(H, I, 4)
        exl3 = []
        for _ in range(a.layers):
            t = torch.randint(0, 256, (E, eb), dtype=torch.uint8, device=dev)
            exl3.append((t, torch.arange(E, dtype=torch.int64, device=dev) * eb + s64(t.data_ptr())))
    torch.xpu.synchronize()
    print(f"IQR blob {blob / 1e6:.3f} MB, {a.layers} arenas" + ("" if exl3 is None else f"; EXL3 blob {eb / 1e6:.3f} MB"))
    g = torch.Generator().manual_seed(0)
    for M in a.rows:
        if a.pool and M <= 8:
            pool = torch.randperm(E, generator=g)[: a.pool]
            idx = torch.stack([pool[torch.randperm(len(pool), generator=g)[:TOPK]] for _ in range(M)])
        else:
            idx = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)])
        distinct = len(torch.unique(idx))
        x = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        w = torch.softmax(torch.randn(M, TOPK, generator=g), -1).float().to(dev)
        ids = idx.to(torch.int32).to(dev)
        res = []
        for name, fn in (("iqr", lambda c: ops().moe_forward(x, ids, w, arenas[c % a.layers])),
                         ("exl3", None if exl3 is None else lambda c: eops().moe_forward(x, ids, w, exl3[c % a.layers][1], I, 4, E))):
            if fn is None:
                continue
            calls = 48 if M <= 8 else 12
            for c in range(a.layers):
                fn(c)
            torch.xpu.synchronize()
            if M <= 8:
                gr = torch.xpu.XPUGraph()
                with torch.xpu.graph(gr):
                    for c in range(calls):
                        fn(c)
                gr.replay(); torch.xpu.synchronize()
                best = 1e9
                for _ in range(5):
                    t0 = time.perf_counter()
                    for _ in range(4):
                        gr.replay()
                    torch.xpu.synchronize(); best = min(best, (time.perf_counter() - t0) / (4 * calls))
            else:
                best = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    for c in range(calls):
                        fn(c)
                    torch.xpu.synchronize(); best = min(best, (time.perf_counter() - t0) / calls)
            res.append((name, best))
        line = f"M={M:5d} distinct={distinct:3d}:"
        for name, dt in res:
            byts = distinct * (blob if name == "iqr" else eb)
            tf = 2 * M * TOPK * 3 * I * H / dt / 1e12
            line += f"  {name} {dt * 1e6:8.1f} us ({byts / dt / 1e9:4.0f} GB/s, {tf:5.1f} TFLOPS)"
        if len(res) == 2:
            line += f"  iqr/exl3 {res[0][1] / res[1][1]:.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "bench"])
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--q"); p.add_argument("--dense"); p.add_argument("--x")
    p.add_argument("--row0", type=int, default=100000)
    p.add_argument("--shard", default="/storage/inkling2/danmoreng-034/models/nvfp4-unc/model-00001-of-00001.safetensors")
    p.add_argument("--check-rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 9, 16, 47, 300, 2048])
    p.add_argument("--ref-rows", type=int, default=12)
    p.add_argument("--ref-full", type=int, default=47, help="check-rows size whose fp64 reference covers every row")
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 48, 256, 2048, 4096])
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--no-exl3", action="store_true")
    a = p.parse_args()
    mode_check(a) if a.mode == "check" else mode_bench(a)
