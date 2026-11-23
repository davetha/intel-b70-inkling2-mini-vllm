"""Batch-one vLLM adapter for the Intel XPU Inkling EXL3 model."""

import json
from functools import partial
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from inkling_trellis.inkling2.attn_triton import attention_prefill_esimd
from inkling_trellis.inkling2.sgl_model import CONV_RING, RING, Inkling2LoopForConditionalGeneration
from inkling_trellis.moe_offload import ops as moe_ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

from . import packed_kv
from .exl3_experts import InklingEXL3Experts
from .iqr_experts import InklingIQRExperts
from .paged_attention import decode_paged
from .prefix_attention import hybrid_prefill
from .split_attention import local_xmx, paged_splits, paged_xmx


class Inkling2B70ForCausalLM(Inkling2LoopForConditionalGeneration):
    # The dense tensors go through vLLM's weight iterator. EXL3 expert shards
    # are loaded one expert at a time into their final packed device arenas.
    enable_vision = False
    # EXL3 checkouts keep dense weights in dense-*.safetensors; native NVFP4 checkpoints ship model-*.safetensors.
    allow_patterns_overrides = (["model-*.safetensors"] if os.environ.get("INKLING2_EXPERTS", "exl3") == "native"
                                else ["dense-*.safetensors"])

    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.scheduler_config.max_num_seqs != 1 or not vllm_config.model_config.enforce_eager:
            raise ValueError("Inkling XPU currently requires --max-num-seqs 1 --enforce-eager")
        cache = vllm_config.cache_config
        if cache.enable_prefix_caching != (os.environ.get("INKLING2_RING_PREFIX", "0") == "1"):
            raise ValueError("Prefix caching requires the single-slot ring history scheduler")
        if cache.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise ValueError("Inkling XPU requires a BF16 or E4M3 KV cache")
        checkpoint = Path(vllm_config.model_config.model)
        raw = json.loads((checkpoint / "config.json").read_text())
        super().__init__(SimpleNamespace(inkling_text=raw["text_config"], inkling_vision=raw["vision_config"] if self.enable_vision else None), external_slots=1)
        self.prefill_controller = os.environ.get("INKLING2_PREFILL_CONTROLLER", "0") == "1"
        self.prefill_bank = os.environ.get("INKLING2_PREFILL_BANK", "0") == "1"
        self.prefill_dense_bf16 = os.environ.get("INKLING2_PREFILL_DENSE_BF16", "0") == "1"
        self.prefill_prep = os.environ.get("INKLING2_PREFILL_PREP", "0") == "1"
        self.prefill_local_bias = os.environ.get("INKLING2_PREFILL_LOCAL_BIAS", "0") == "table"
        self.prefill_kv_conv = os.environ.get("INKLING2_PREFILL_KV_CONV", "0") == "1"
        self.prefill_output_conv = os.environ.get("INKLING2_PREFILL_OUTPUT_CONV", "0") == "1"
        self.enable_ring_prefix = cache.enable_prefix_caching
        self.ring_high = self.saved_frontier = 0
        self.saved_rings = []
        self.use_decode_graph = os.environ.get("INKLING2_DECODE_GRAPH", "0") == "1"
        self.decode_graphs = {}
        self.local_xmx = local_xmx if os.environ.get("INKLING2_LOCAL_XMX", "0") == "1" else None
        self.use_paged_xmx = os.environ.get("INKLING2_PAGED_XMX", "0") == "1"
        self.packed_kv = os.environ.get("INKLING2_PACKED_KV", "0") == "1"
        if self.packed_kv and not (self.use_paged_xmx and cache.cache_dtype.startswith("fp8")
                                   and (cache.block_size or 16) % 16 == 0):
            raise ValueError("The XMX-packed cache requires paged XMX and FP8 KV with 16-token tiles")
        self.warmed_shapes = set()
        self.compare_graph = os.environ.get("INKLING2_COMPARE_GRAPH", "0") == "1"
        spec = vllm_config.speculative_config
        self.want_dspark = spec is not None and spec.custom_method == "inkling_dspark"
        self.arbitrary_rows = self.local_xmx is not None and self.use_paged_xmx and cache.cache_dtype.startswith("fp8")
        self.page_tokens = cache.block_size or 16
        self.arbitrary_rows = self.arbitrary_rows and self.page_tokens % 16 == 0
        if self.want_dspark and spec.num_speculative_tokens not in (1, 3, 7) and not self.arbitrary_rows:
            raise ValueError("This DSpark width requires FP8 KV and both target XMX attention kernels")
        self.capture_rows = {1, spec.num_speculative_tokens + 1} if self.want_dspark else {1}
        variable_width = (os.environ.get("INKLING2_DSPARK_ADAPTIVE", "0") == "1"
                          or os.environ.get("INKLING2_NGRAM_DRAFT", "0") == "1")
        if self.want_dspark and variable_width:
            if vllm_config.scheduler_config.async_scheduling:
                raise ValueError("Adaptive DSpark requires synchronous scheduling")
            if not self.arbitrary_rows:
                raise ValueError("Adaptive DSpark requires FP8 KV and both target XMX attention kernels")
            self.capture_rows.update(range(2, spec.num_speculative_tokens + 1))
        self.max_context = vllm_config.model_config.max_model_len
        self.checkpoint = checkpoint
        self.logit_cap = self.tc["final_logit_softcapping"]
        self.cache_dtype = torch.float8_e4m3fn if cache.cache_dtype.startswith("fp8") else torch.bfloat16
        self.prefill_prefix_io = os.environ.get("INKLING2_PREFILL_PREFIX_IO", "0") == "1"
        self.tight_prefix = os.environ.get("INKLING2_PREFILL_TIGHT_PREFIX", "0") == "1"
        self.prefix_block = int(os.environ.get("INKLING2_ONEDNN_PREFIX_BLOCK", "0"))
        if self.prefix_block:
            if self.prefix_block < 0 or self.prefix_block % self.page_tokens:
                raise ValueError("The oneDNN prefix block must be a positive multiple of the KV page size")
            torch.ops.load_library("/opt/exl3xpu/exl3xpu/_C.so")
        root = f"{prefix}." if prefix else ""
        self.kv_layers = nn.ModuleList([
            # Reuse the metadata builder and allocator contract. This model
            # calls the SYCL attention directly using the bound page views.
            Attention(self.H, self.hd, self.hd ** -0.5, self.Hkv, cache_config=cache,
                      prefix=f"{root}global_attn.{index}", attn_backend=TritonAttentionBackend)
            for index in range(self.n_global)
        ])

    def load_weights(self, weights):
        if os.environ.get("INKLING2_ONEDNN_DETERMINISTIC") == "1":
            # oneDNN otherwise picks reduction orders per call, so identical bf16 GEMMs can differ run to run.
            torch.backends.mkldnn.deterministic = True
        device = torch.device("xpu", torch.xpu.current_device())
        min_dpas = int(os.environ.get("INKLING2_EXL3_PREFILL_MIN", "1025"))
        moe_ops().moe_set_prefill_min_m(min_dpas)
        moe_ops().moe_set_pair_max_m(int(os.environ.get("INKLING2_EXL3_PAIR_MAX", "8")))
        if os.environ.get("INKLING2_EXL3_SMALL_DPAS") == "1":
            moe_ops().moe_set_tiny_dpas(4, 9, 18)
        # INKLING2_EXPERTS=native keeps the checkpoint's NVFP4/MXFP8 experts (the model's training precision);
        # the base loader then builds Trellis' SYCL experts from them. The default re-packs EXL3 experts.
        # INKLING2_EXPERTS=iqr4 loads GPTQ 4-bit IQR experts (iqr-layer-*.safetensors, iqr_moe library).
        kind = os.environ.get("INKLING2_EXPERTS", "exl3")
        if kind == "native":
            experts = None
        elif kind in ("exl3", "iqr4"):
            cls = InklingIQRExperts if kind == "iqr4" else InklingEXL3Experts
            experts = {layer: cls(self.checkpoint, layer, device, self.tc["n_routed_experts"])
                       for layer in range(self.n_layers)}
        else:
            raise ValueError(f"INKLING2_EXPERTS must be exl3, iqr4 or native, not {kind!r}")
        super().load_weights(weights, experts=experts)
        self.prefill_depth = None
        if os.environ.get("INKLING2_PREFILL_DEPTH", "0") == "1":
            from .prefill_depth import DepthQueries
            self.prefill_depth = DepthQueries(self.w, self.D)
        if self.want_dspark:
            from .dspark import InklingDraft, bind_target
            self.draft = InklingDraft(self.checkpoint, device, self.max_context)
            self.draft.bind_target_head(self.lm_q8)
            bind_target(self)

    def embed_input_ids(self, input_ids):
        return self.get_input_embeddings()(input_ids)

    def compute_logits(self, hidden_states):
        logits = self.lm_q8(hidden_states)
        return torch.tanh(logits / self.logit_cap) * self.logit_cap

    def _pages(self, visit, batch):
        cache = batch.caches[self.visit_slot[visit]]
        if cache.ndim != 4 or cache.shape[1:] != (self.Hkv, self.page_tokens, 2 * self.hd):
            raise ValueError(f"Unexpected vLLM cache view {tuple(cache.shape)}")
        # vLLM's allocator exposes FP8 cache storage as bytes. The kernel
        # dispatches on the logical E4M3 type and needs the original byte bits.
        if self.cache_dtype == torch.float8_e4m3fn and cache.dtype in (torch.uint8, torch.int8):
            cache = cache.view(self.cache_dtype)
        if cache.dtype != self.cache_dtype:
            raise ValueError(f"Expected {self.cache_dtype} in the vLLM KV cache, got {cache.dtype}")
        return cache.transpose(1, 2).split(self.hd, dim=-1)

    def _store_kv(self, key_cache, value_cache, key, value, rows):
        if self.packed_kv:
            packed_kv.store(key_cache, rows, key, value)
            return
        pages = rows // self.page_tokens
        positions = rows % self.page_tokens
        if key_cache.dtype == torch.float8_e4m3fn:
            key_cache.view(torch.uint8)[pages, positions] = key.to(key_cache.dtype).view(torch.uint8)
            value_cache.view(torch.uint8)[pages, positions] = value.to(value_cache.dtype).view(torch.uint8)
        else:
            key_cache[pages, positions] = key.to(key_cache.dtype)
            value_cache[pages, positions] = value.to(value_cache.dtype)

    def _decode_global(self, visit, q, k, v, pos, rel, proj, sink, batch):
        key_cache, value_cache = self._pages(visit, batch)
        self._store_kv(key_cache, value_cache, k, v, batch.out_cache_loc)
        if self.use_paged_xmx and key_cache.dtype == torch.float8_e4m3fn and self.page_tokens % 16 == 0:
            return paged_xmx(q, key_cache, value_cache, batch.block_table, batch.seq_lens,
                             pos, rel, proj, sink, paged_splits(batch.total_tokens, q.shape[0]), self.packed_kv)
        splits = min(64, max(8, (batch.total_tokens + 1023) // 1024))
        return decode_paged(q, key_cache, value_cache, batch.block_table, batch.seq_lens, pos, rel, proj, sink,
                            queries_per_request=q.shape[0], num_splits=splits)

    def _control_norm(self, il, module, cur, msg, weight):
        if self.prefill_controller and not self._static and cur.shape[0] > 8:
            down = self.lin[f"layers.{il}.controllers.{module}.down_weight"]
            up, _ = self.fused[il, "gf" + module]
            if down.w8a8 is None and up.w8a8 is None:
                from .prefill_controller import controller
                return controller(cur, msg, weight, down.w, up.w)
        return super()._control_norm(il, module, cur, msg, weight)

    def _conv(self, vi, name, x, weight, dynamic, reqs, pos):
        use_prefill_conv = self.prefill_output_conv if name in ("attn_out", "mlp_out") else self.prefill_kv_conv
        if use_prefill_conv and not self._static:
            from .prefill_kernels import prefill_conv
            slot, _, _, start = reqs[0]
            offset, _ = self.conv_ch[name]
            return prefill_conv(x, weight, self.conv[vi, slot], self.conv_pos[slot], start, offset, dynamic)
        return super()._conv(vi, name, x, weight, dynamic, reqs, pos)

    def _prefill_global(self, visit, q, k, v, table, sink, reqs, batch):
        key_cache, value_cache = self._pages(visit, batch)
        self._store_kv(key_cache, value_cache, k, v, batch.out_cache_loc)

        def read_kv(start, end):
            page_start = start // self.page_tokens
            page_end = (end + self.page_tokens - 1) // self.page_tokens
            pages = batch.block_table[0, page_start:page_end].long()
            if self.packed_kv:
                return packed_kv.read(key_cache, pages, end - start)
            if key_cache.dtype == torch.float8_e4m3fn:
                return tuple(t.view(torch.uint8).index_select(0, pages).flatten(0, 1)[:end - start].view(t.dtype)
                             for t in (key_cache, value_cache))
            return tuple(t.index_select(0, pages).flatten(0, 1)[:end - start] for t in (key_cache, value_cache))

        if batch.first_position == 0:
            # The first chunk already has every key and value in logical order.
            keys, values = k, v
        elif self.prefix_block and batch.first_position + 1 - table.shape[-1] >= (self.page_tokens if self.tight_prefix else self.prefix_block):
            read_heads = None
            if self.prefill_prefix_io and not self.packed_kv:
                from .prefix_io import read_heads as gather_heads
                read_heads = partial(gather_heads, key_cache, value_cache, batch.block_table[0])
            return [hybrid_prefill(q, table, sink, batch.first_position, batch.total_tokens,
                                   self.prefix_block, read_kv, split_alignment=self.page_tokens if self.tight_prefix else None,
                                   read_heads=read_heads)]
        else:
            keys, values = read_kv(0, batch.total_tokens)
        return [attention_prefill_esimd(q, keys, values, batch.first_position, 0, table, sink, qtw=4)]

    def _metadata(self, size, device):
        metadata = get_forward_context().attn_metadata
        if metadata is not None:
            md = metadata[self.kv_layers[0].layer_name]
            if md.seq_lens.numel() != 1 or md.num_actual_tokens != size:
                raise ValueError("Inkling XPU requires one unpadded sequence per forward")
            self.page_tokens = self.kv_layers[0].kv_cache.shape[2]
            if not getattr(self, "_kv_layout_logged", False):
                self._kv_layout_logged = True
                logging.getLogger(__name__).warning("Inkling KV page_tokens=%d kv_cache.shape=%s",
                                                    self.page_tokens, tuple(self.kv_layers[0].kv_cache.shape))
            total = int(md.seq_lens[0].item())
            return SimpleNamespace(caches=[layer.kv_cache for layer in self.kv_layers], block_table=md.block_table,
                                   seq_lens=md.seq_lens, out_cache_loc=md.slot_mapping[:size].long(),
                                   total_tokens=total, first_position=total - size)

        # Run the model during vLLM's memory profiling before it binds a KV arena.
        num_pages = (size + self.page_tokens - 1) // self.page_tokens
        pages = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
        caches = [torch.empty((num_pages, self.Hkv, self.page_tokens, 2 * self.hd), dtype=self.cache_dtype, device=device)
                  for _ in self.kv_layers]
        return SimpleNamespace(caches=caches, block_table=pages,
                               seq_lens=torch.tensor([size], dtype=torch.int32, device=device),
                               out_cache_loc=torch.arange(size, device=device), total_tokens=size, first_position=0)

    def _static_forward(self, ids, positions, batch, embeds=None):
        size = positions.numel()
        self._st_nq = size
        self._st_req_slots = torch.zeros(1, dtype=torch.int32, device=positions.device)
        self._st_slot = torch.zeros(size, dtype=torch.long, device=positions.device)
        self._st_pos = positions
        self.swa_pos[0, positions.remainder(RING)] = positions.int()
        self.conv_pos[0, positions.remainder(CONV_RING)] = positions
        return self._forward_body(ids, positions, batch, None, embeds, return_hidden=True)

    def _graph_forward(self, ids, positions, batch):
        # Capture only after a real eager decode initialized Triton and the
        # layer-specific GEMV plans. Ring positions and page IDs remain dynamic.
        size = positions.numel()
        if size not in self.warmed_shapes:
            out = self._static_forward(ids, positions, batch)
            self.warmed_shapes.add(size)
            return out
        splits = min(64, max(8, (batch.total_tokens + 1023) // 1024))
        if self.use_paged_xmx:
            splits = (paged_splits(batch.total_tokens, size) if self.cache_dtype == torch.float8_e4m3fn
                      and self.page_tokens % 16 == 0 else splits)
        key = (splits, tuple(batch.block_table.shape), ids.dtype,
               tuple(cache.data_ptr() for cache in batch.caches))
        state = self.decode_graphs.get(size)
        if state is None or state.key != key:
            self.decode_graphs.pop(size, None)
            staged = SimpleNamespace(**vars(batch))
            staged.block_table = batch.block_table.clone()
            staged.seq_lens = batch.seq_lens.clone()
            staged.out_cache_loc = batch.out_cache_loc.clone()
            state = SimpleNamespace(key=key, batch=staged, ids=ids.clone(), positions=positions.clone(),
                                    graph=torch.xpu.XPUGraph(), comparisons=0)
            with torch.xpu.graph(state.graph):
                state.output = self._static_forward(state.ids, state.positions, staged)
            self.decode_graphs[size] = state
        else:
            state.ids.copy_(ids)
            state.positions.copy_(positions)
            state.batch.seq_lens.copy_(batch.seq_lens)
            state.batch.out_cache_loc.copy_(batch.out_cache_loc)
            state.batch.block_table.copy_(batch.block_table)
        state.graph.replay()
        if self.compare_graph and state.comparisons < 8:
            # A second pass at identical positions can overwrite those same slots.
            # No committed history row is evicted by repeating the current block.
            actual = state.output.clone()
            reference = self._static_forward(ids, positions, batch)
            diff = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
            matches = torch.equal(self.compute_logits(actual).argmax(-1), self.compute_logits(reference).argmax(-1))
            logging.getLogger(__name__).warning("B70 replay parity rows=%d pos=%d rel_l2=%.8g same_argmax=%s",
                                                size, batch.first_position, diff.item(), matches)
            if diff.item() > 0.003 or not matches:
                repeat = self._static_forward(ids, positions, batch)
                eager_delta = (repeat.float()-reference.float()).norm() / reference.float().norm()
                raise RuntimeError(f"XPU replay mismatch. Relative L2 on eager repetition: {eager_delta.item():.8g}")
            state.comparisons += 1
        return state.output

    def _ring_buffers(self):
        yield from (self.swa_k, self.swa_v, self.swa_pos, self.conv, self.conv_pos)
        if self.want_dspark:
            yield self.draft.f_ring
            for index in sorted(self.draft.local):
                yield self.draft.kv[index]

    def _restore_history(self, start):
        if start == 0:
            self.ring_high = self.saved_frontier = 0
        elif start < self.ring_high - 32:
            if not self.saved_frontier - 32 <= start <= self.saved_frontier:
                raise RuntimeError("Prefix hit exceeds both retained ring histories")
            for live, saved in zip(self._ring_buffers(), self.saved_rings, strict=True):
                live.copy_(saved)
            self.ring_high = start

    def _save_history(self, end):
        if self.saved_rings:
            for live, saved in zip(self._ring_buffers(), self.saved_rings, strict=True):
                saved.copy_(live)
        else:
            self.saved_rings = [live.clone() for live in self._ring_buffers()]
        self.saved_frontier = end

    @torch.no_grad()
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        size = positions.numel()
        positions = positions.long()
        batch = self._metadata(size, positions.device)
        real_history = self.enable_ring_prefix and get_forward_context().attn_metadata is not None
        if real_history:
            self._restore_history(batch.first_position)
            self.ring_high = batch.total_tokens
        if self.want_dspark:
            self.active_draft = self.draft if get_forward_context().attn_metadata is not None else None
            self.draft_end = batch.total_tokens
        self._static = batch.first_position > 0 and (size in (1, 2, 4, 8) or self.arbitrary_rows and 1 <= size <= 8)
        if self._static:
            if self.use_decode_graph and size in self.capture_rows and inputs_embeds is None:
                return self._graph_forward(input_ids, positions, batch)
            return self._static_forward(input_ids, positions, batch, inputs_embeds)

        reqs = [(0, 0, size, batch.first_position)]
        self._reqs = reqs
        self._conv_pos_update = []
        self._prepare(batch, reqs, positions, kv_pool=True)
        hidden = self._forward_body(input_ids, positions, batch, reqs, inputs_embeds, return_hidden=True)
        if real_history and size > 8:
            self._save_history(batch.total_tokens)
        return hidden
