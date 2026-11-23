"""IQR 4-bit routed experts (INKLING2_EXPERTS=iqr4): GPTQ codes in the WHT-128-rotated basis, bf16 scale per
(row, 128 of K), served by the iqr_moe SYCL library (inkling_xpu/csrc/iqr_moe.sycl).

Checkpoint layout (<model>/iqr-layer-XX.safetensors, described in iqr_config.json), per layer L, prefix
model.llm.layers.L.mlp.experts.:
  w13_codes uint8 [E, 2304, 1152]  two u4 codes per byte along K (low nibble = even k), value = code - 8; rows as the
                                   original checkpoint (row 2j = gate j, row 2j+1 = up j)
  w13_scales bf16 [E, 2304, 18]    W13_r[r, k] = scale[r, k // 128] * (code[r, k] - 8), W13 = W13_r R (R = block WHT)
  w2_codes  uint8 [E, 2304, 576]   down, same encoding (rows = output columns)
  w2_scales bf16 [E, 2304, 9]
At load time each layer is repacked on the device into the kernel's tiled arena [E, blob].
"""

import os

import torch
from safetensors import safe_open

_ops = None


def ops():
    global _ops
    if _ops is None:
        torch.ops.load_library(os.environ.get("INKLING2_IQR_MOE_LIB", "/lab/.work/build/inkling2-iqr-moe.so"))
        _ops = torch.ops.iqr_moe
    return _ops


def gate_up_tile_rows(device=None):
    """Kernel tile order -> w13 row: tile t, lane l < 8 = gate row of intermediate 8t+l, lane >= 8 = its up row."""
    t = torch.arange(144, device=device).view(144, 1)
    lane = torch.arange(16, device=device).view(1, 16)
    return (2 * (8 * t + lane % 8) + (lane >= 8)).reshape(-1)


def pack_codes(packed, rows=None):
    """uint8 [E, R, K/2] (checkpoint encoding) -> uint8 [E, R/16 * K/128 * 1024] in the kernel's DPAS-B tile layout:
    byte (d*64 + q*16 + lane)*4 + i of block (tile, group) = k 32d+4q+i (low nibble) | k 32d+16+4q+i (high nibble)."""
    E, R, Kh = packed.shape
    K = 2 * Kh
    c = torch.stack([packed & 15, packed >> 4], -1).reshape(E, R, K)
    if rows is not None:
        c = c[:, rows]
    c = c.view(E, R // 16, 16, K // 128, 4, 2, 4, 4).permute(0, 1, 3, 4, 6, 2, 7, 5)   # e t g d q lane i h
    return (c[..., 0] | (c[..., 1] << 4)).reshape(E, -1)


def pack_scales(scales, rows=None):
    """bf16 [E, R, G] -> uint8 [E, R/16 * G * 32] ([tile][group][16 lanes] bf16)."""
    E, R, G = scales.shape
    if rows is not None:
        scales = scales[:, rows]
    return scales.view(E, R // 16, 16, G).permute(0, 1, 3, 2).contiguous().view(torch.uint8).reshape(E, -1)


def pack_layer(w13_codes, w13_scales, w2_codes, w2_scales):
    rows = gate_up_tile_rows(w13_codes.device)
    return torch.cat([pack_codes(w13_codes, rows), pack_scales(w13_scales, rows), pack_codes(w2_codes),
                      pack_scales(w2_scales)], 1)


class InklingIQRExperts:
    def __init__(self, checkpoint, layer, device, num_experts=128, chunk=16):
        self.kernel = ops().moe_forward
        blob = ops().blob_bytes()
        self.arena = torch.empty((num_experts, blob), dtype=torch.uint8, device=device)
        p = f"model.llm.layers.{layer}.mlp.experts."
        names = ("w13_codes", "w13_scales", "w2_codes", "w2_scales")
        with safe_open(str(checkpoint / f"iqr-layer-{layer:02d}.safetensors"), framework="pt", device="cpu") as f:
            if f.get_slice(p + "w13_codes").get_shape()[0] != num_experts:
                raise ValueError(f"IQR layer {layer}: expert count mismatch")
            for e0 in range(0, num_experts, chunk):
                t = [f.get_slice(p + n)[e0:e0 + chunk].to(device) for n in names]
                self.arena[e0:e0 + chunk].copy_(pack_layer(*t))
        if self.arena.shape[1] != blob:
            raise ValueError("IQR arena size mismatch")

    def __call__(self, hidden, indices, routing):
        return self.kernel(hidden.to(torch.bfloat16).contiguous(), indices.to(torch.int32).contiguous(),
                           routing.float().contiguous(), self.arena)
