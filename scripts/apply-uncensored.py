"""Make an uncensored copy of the Inkling 2 Mini EXL3 checkpoint.

OpenUncensored/Inkling-2-Mini-Uncensored changes only the bf16 attention output projections
(attn.wo_ud) of layers 4-23; experts and everything else are byte-identical to the original. The
EXL3 checkpoint stores those projections as bf16 too, so the edit transfers exactly: every file is
hard-linked except the dense shards holding an edited wo_ud, which are copied and patched in place
(same name, dtype and shape, so byte offsets are unchanged).

usage: apply_uncensored.py <exl3_dir> <uncensored_dir> <out_dir>
"""
import glob
import json
import os
import shutil
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def index(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        n, h = header(p)
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (p, n, v)
    return out


def main(exl3, unc, out):
    if os.path.exists(out):
        sys.exit(f"{out} exists; refusing to overwrite")
    src, edit = index(exl3), index(unc)
    changed = {}
    for k, (up, un, uv) in edit.items():
        if ".attn.wo_ud." not in k:
            continue
        ep, en, ev = src[k]
        if (ev["dtype"], ev["shape"]) != (uv["dtype"], uv["shape"]):
            sys.exit(f"{k}: {ev['dtype']}{ev['shape']} vs {uv['dtype']}{uv['shape']}")
        with open(up, "rb") as f:
            a, b = uv["data_offsets"]; f.seek(8 + un + a); new = f.read(b - a)
        with open(ep, "rb") as f:
            a, b = ev["data_offsets"]; f.seek(8 + en + a); old = f.read(b - a)
        if new != old:
            changed.setdefault(ep, []).append((k, 8 + en + ev["data_offsets"][0], new))

    os.makedirs(out)
    for root, _, files in os.walk(exl3):
        rel = os.path.relpath(root, exl3)
        if rel.split(os.sep)[0] == ".cache":
            continue
        os.makedirs(os.path.join(out, rel), exist_ok=True)
        for fn in files:
            p = os.path.join(root, fn)
            q = os.path.join(out, rel, fn)
            if p in changed:
                shutil.copyfile(p, q)
                with open(q, "r+b") as f:
                    for _, off, data in changed[p]:
                        f.seek(off); f.write(data)
            else:
                os.link(p, q)
    n = sum(len(v) for v in changed.values())
    print(f"patched {n} wo_ud tensors in {len(changed)} shard(s):",
          sorted(int(k.split(".layers.")[1].split(".")[0]) for v in changed.values() for k, _, _ in v))


if __name__ == "__main__":
    main(*sys.argv[1:4])
