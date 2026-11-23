"""Batch-one Inkling 2 Mini DSpark for the Intel XPU target and vLLM V2 runner."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from .split_attention import merge_splits

from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

# vLLM 0.34 has no runner hook to hand a custom speculator the target model,
# so the Inkling model binds itself here once its draft is built.
_TARGET = None


def bind_target(model):
    global _TARGET
    _TARGET = model


def norm(x, weight):
    values = x.float()
    return (values * (values.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight.float()).to(
        torch.bfloat16
    )


class InklingDraft:
    def __init__(self, checkpoint, device, max_context):
        config = json.loads((checkpoint / "config.json").read_text())
        self.config = config["mtp_config"]
        self.width = self.config["hidden_size"]
        self.head = self.config["head_dim"]
        self.heads = self.config["num_attention_heads"]
        self.groups = self.config["num_key_value_heads"]
        self.block = self.config["block_size"]
        self.rank = self.config["markov_rank"]
        self.window = self.config["sliding_window_size"]
        self.local = set(self.config["local_layer_ids"])
        self.taps = self.config["target_layer_ids"]
        self.cap = config["text_config"]["final_logit_softcapping"]
        self.vocab = config["text_config"]["unpadded_vocab_size"]
        # The shared projection uses the target's group-32 INT8 vocabulary weights.
        self.shared_head = os.environ.get("INKLING2_DRAFT_Q8_HEAD", "0") == "1"
        self.q8_head = None
        self.shortlist = int(os.environ.get("INKLING2_DSPARK_SHORTLIST", "0"))
        if not 0 <= self.shortlist <= self.vocab:
            raise ValueError("Invalid DSpark candidate count")
        self.w = {k: v.to(device) for k, v in load_file(checkpoint / "mtp.safetensors").items()}
        self.mix = self.w["fuse_weight"].float().softmax(-1)
        self.freq = self.config["rope_theta"] ** (
            -torch.arange(0, self.head, 2, device=device).float() / self.head
        )
        self.offsets = torch.arange(self.block, dtype=torch.long, device=device)
        self.split_attention = os.environ.get("INKLING2_DSPARK_BLOCK", "0") == "1"
        self.fp8_global = self.split_attention and os.environ.get("INKLING2_DRAFT_FP8_GLOBAL", "0") == "1"
        self.graph_enabled = os.environ.get("INKLING2_DSPARK_GRAPH", "0") == "1"
        if self.graph_enabled and not self.split_attention:
            raise ValueError("The DSpark graph requires the fixed-layout split attention kernel")
        self.reference_math = os.environ.get("INKLING2_DRAFT_REFERENCE_MATH", "0") == "1"
        self.debug_replay = os.environ.get("INKLING2_COMPARE_DRAFT_GRAPH", "0") == "1"
        self.sampling = None
        self.graph_checks = 0
        self.graph_state = None
        if self.reference_math:
            self.mix = self.mix.to(self.w["fuse_weight"].dtype).float()
        self.draft_mix_norm = getattr(torch.ops.inkling2_xpu, "draft_mix_norm", None)
        fuse_taps = os.environ.get("INKLING2_DRAFT_FUSED_TAPS", "1") == "1"
        if fuse_taps and self.draft_mix_norm is not None and self.mix.shape == (4, 4) and self.width == 2304:
            norms = [self.w[f"layers.{i}.dflare_norm.weight"] for i in range(3)]
            self.draft_norms = torch.stack([*norms, self.w["markov_head.dflare_norm.weight"]]).float()
        else:
            self.draft_mix_norm = None
        self.warmed = set()
        self.kv = []
        self.capacities = []
        slack = 64 if os.environ.get("INKLING2_RING_PREFIX", "0") == "1" else self.block * 2
        self.f_ring = torch.empty(slack, self.width, device=device, dtype=torch.bfloat16)
        for index in range(self.config["num_hidden_layers"]):
            p = f"layers.{index}."
            for group, stems in (
                ("self_attn.qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
                ("self_attn.context_kv", ["self_attn.k_proj_ctx", "self_attn.v_proj_ctx"]),
                ("mlp.gate_up", ["mlp.gate_proj", "mlp.up_proj"]),
            ):
                self.w[p + group] = torch.cat([self.w.pop(p + stem + ".weight") for stem in stems])
            length = self.window + slack if index in self.local else max_context
            self.capacities.append(length if index in self.local else 0)
            rows = (
                (2 * length if index in self.local else length + self.block)
                if self.split_attention
                else length
            )
            dtype = torch.float16 if self.split_attention else torch.bfloat16
            if self.fp8_global and index not in self.local:
                rows = (rows + 15) // 16 * 16
                dtype = torch.float8_e4m3fn
            self.kv.append(torch.empty((2, rows, self.groups, self.head), device=device, dtype=dtype))
        shard = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"][
            "model.llm.unembed.weight"
        ]
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
            self.lm = reader.get_tensor("model.llm.unembed.weight")[: self.vocab].to(device).contiguous()
        candidate_count = self.lm.shape[0]
        if self.shortlist > candidate_count:
            raise ValueError("DSpark shortlist exceeds the draft projection vocabulary")
        w = self.w.pop("markov_head.hidden_out.weight").float()
        b = self.w.pop("markov_head.hidden_out.bias").float()
        self.fold = torch.empty(candidate_count, self.rank, device=device, dtype=torch.bfloat16)
        self.bias = torch.empty(candidate_count, device=device, dtype=torch.float32)
        for first in range(0, candidate_count, 2048):
            rows = self.lm[first : first + 2048].float()
            self.fold[first : first + 2048] = rows @ w
            self.bias[first : first + 2048] = rows @ b
        if self.shared_head:
            del self.lm
        weight = self.w.pop("markov_head.rnn_proj.weight").float()
        self.recurrent, self.context = (
            v.contiguous() for v in weight.split([2 * self.rank, 2 * self.width], dim=1)
        )

    def bind_target_head(self, head):
        if self.shared_head:
            self.q8_head = head

    def rotation(self, positions):
        angle = positions.float()[:, None] * self.freq
        return angle.cos()[:, None, :], angle.sin()[:, None, :]

    def rotate(self, x, rotation):
        a, b = x.float().chunk(2, -1)
        cos, sin = rotation
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).to(torch.bfloat16)

    def store(self, features, positions):
        # The local and recurrent rings have slack for a whole verification
        # block. Rejected rows are past the next committed boundary and get overwritten.
        fused = None
        if self.draft_mix_norm is not None and len(positions) <= 8:
            fused = self.draft_mix_norm(*features, self.mix, self.draft_norms)
        else:
            sources = torch.stack(features, dim=1).float()
        rotation = self.rotation(positions)
        for i, kv in enumerate(self.kv):
            p = f"layers.{i}."
            limit = self.capacities[i] or len(positions)
            if fused is not None:
                mixed = fused[i, -limit:]
            else:
                src = sources[-limit:]
                mixed = norm((src * self.mix[i, :, None]).sum(1), self.w[p + "dflare_norm.weight"])
            k, v = (
                F.linear(mixed, self.w[p + "self_attn.context_kv"])
                .view(len(mixed), 2, self.groups, self.head)
                .unbind(1)
            )
            k = self.rotate(
                norm(k, self.w[p + "self_attn.k_norm_ctx.weight"]), tuple(x[-limit:] for x in rotation)
            )
            rows = positions[-limit:]
            self.write_rows(i, rows, k, v)
        ring = self.f_ring.size(0)
        if fused is not None:
            self.f_ring[positions[-ring:].remainder(ring)] = fused[-1, -ring:]
            return
        context = (sources[-ring:] * self.mix[-1, :, None]).sum(1)
        self.f_ring[positions[-ring:].remainder(ring)] = norm(
            context, self.w["markov_head.dflare_norm.weight"]
        )

    def write_rows(self, i, rows, k, v):
        kv = self.kv[i]
        capacity = self.capacities[i]
        k, v = k.to(kv.dtype), v.to(kv.dtype)
        if kv.dtype == torch.float8_e4m3fn:
            kv = kv.view(torch.uint8)
            k, v = k.view(torch.uint8), v.view(torch.uint8)
        if capacity:
            rows = rows.remainder(capacity)
            if self.split_attention:
                kv[0, rows + capacity], kv[1, rows + capacity] = k, v
        kv[0, rows], kv[1, rows] = k, v

    def attend_block(self, i, q, k, v, positions, start, splits):
        self.write_rows(i, positions, k, v)
        kv = self.kv[i]
        capacity = self.capacities[i]
        parts = 16 if capacity else splits
        out = q.new_empty((parts, self.block, self.heads, self.head), dtype=torch.float32)
        lse = torch.empty_like(out[:, :, :, 0], memory_format=torch.contiguous_format)
        query = q.to(torch.float16).contiguous()
        if self.fp8_global and not capacity:
            torch.ops.inkling2_xpu.draft_global_xmx(query, kv[0], kv[1], start, out, lse)
        else:
            torch.ops.inkling2_xpu.dspark_attention(
                query, kv[0], kv[1], start, out, lse, capacity, self.window if capacity else 0
            )
        return merge_splits(out, lse).reshape(self.block, -1).to(torch.bfloat16)

    def propose(self, target, anchor, start, steps, upper):
        if steps >= self.block:
            raise ValueError("DSpark needs an anchor and one noise row per proposed token")
        splits = min(512, max(16, 1 << max(0, (max(upper, 1) - 1).bit_length() - 7)))
        if not self.split_attention:
            return self._block(target, anchor, start, steps, splits)
        position = start
        if not self.graph_enabled or splits not in self.warmed:
            self.warmed.add(splits)
            return self._block(target, anchor, position, steps, splits)
        if self.graph_state is None or self.graph_state[0] != splits:
            buf_id, buf_pos = anchor.clone(), position.clone()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                output = self._block(target, buf_id, buf_pos, steps, splits)
            self.graph_state = (splits, graph, buf_id, buf_pos, output)
        _, graph, buf_id, buf_pos, output = self.graph_state
        buf_id.copy_(anchor)
        buf_pos.copy_(position)
        graph.replay()
        if self.debug_replay and self.graph_checks < 8:
            reference = self._block(target, anchor, position, steps, splits)
            if not torch.equal(output, reference):
                raise RuntimeError("DSpark graph proposals differ from an eager pass at the same position")
            self.graph_checks += 1
        return output

    def _block(self, target, anchor, start, steps, splits):
        positions = self.offsets + start
        rotation = self.rotation(positions)
        x = torch.cat(
            [
                target.get_input_embeddings()(anchor).to(torch.bfloat16),
                self.w["mask_embed"].expand(self.block - 1, -1),
            ]
        )
        for i, kv in enumerate(self.kv):
            p = f"layers.{i}."
            h = norm(x, self.w[p + "input_layernorm.weight"])
            q, k, v = (
                F.linear(h, self.w[p + "self_attn.qkv"])
                .view(self.block, -1, self.head)
                .split([self.heads, self.groups, self.groups], dim=1)
            )
            q = self.rotate(norm(q, self.w[p + "self_attn.q_norm.weight"]), rotation)
            k = self.rotate(norm(k, self.w[p + "self_attn.k_norm.weight"]), rotation)
            if self.split_attention:
                h = self.attend_block(i, q, k, v, positions, start, splits)
            else:
                if i in self.local:
                    rows = torch.arange(max(0, start - self.window), start, device=x.device).remainder(
                        kv.size(1)
                    )
                    before_k, before_v = kv[:, rows].unbind()
                else:
                    before_k, before_v = kv[:, :start].unbind()
                keys = torch.cat([before_k, k]).transpose(0, 1).unsqueeze(1)
                values = torch.cat([before_v, v]).transpose(0, 1).unsqueeze(1)
                q = q.transpose(0, 1).reshape(self.groups, -1, self.block, self.head)
                h = F.scaled_dot_product_attention(q, keys, values)
                h = h.reshape(self.heads, self.block, self.head).transpose(0, 1).reshape(self.block, -1)
            projected = F.linear(h, self.w[p + "self_attn.o_proj.weight"])
            if self.reference_math:
                x = x.to(torch.bfloat16).float() + projected.float()
            else:
                x = x + projected
            h = norm(x, self.w[p + "post_attention_layernorm.weight"])
            gate, up = F.linear(h, self.w[p + "mlp.gate_up"]).chunk(2, -1)
            if self.reference_math:
                activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
                x = (
                    x.to(torch.bfloat16).float()
                    + F.linear(activated, self.w[p + "mlp.down_proj.weight"]).float()
                )
            else:
                x = x + F.linear(F.silu(gate) * up, self.w[p + "mlp.down_proj.weight"])
        hidden = norm(x[1 : steps + 1], self.w["norm.weight"])
        if self.q8_head is None:
            base = F.linear(hidden, self.lm).float()
        else:
            base = self.q8_head(hidden)[:, : self.vocab]
        f_rnn = self.f_ring[(start - 1) % self.f_ring.size(0)].expand(len(hidden), -1)
        context = F.linear(
            torch.cat([hidden, f_rnn], dim=1).float(),
            self.context,
            self.w["markov_head.rnn_proj.bias"].float(),
        )
        if self.shortlist:
            base, indices = base.topk(self.shortlist, dim=-1)
            folded = F.embedding(indices, self.fold).float()
            biases = self.bias[indices]
        state = hidden.new_zeros((1, self.rank))
        prev = anchor
        steps = []
        for i in range(len(hidden)):
            token = F.embedding(prev, self.w["markov_head.markov_w1.weight"])
            raw = F.linear(torch.cat([state, token], dim=-1).float(), self.recurrent) + context[i]
            gate, candidate, latent = raw.to(torch.bfloat16).chunk(3, -1)
            gate = gate.sigmoid()
            state = gate * state + (1 - gate) * candidate.tanh()
            if self.shortlist:
                correction = (folded[i] @ latent.tanh().float().T).T + biases[i]
            else:
                correction = (
                    F.linear(latent.tanh().float(), self.fold.float()).float() + self.bias
                    if self.reference_math
                    else (F.linear(latent.tanh(), self.fold).float() + self.bias)
                )
            if self.reference_math:
                combined = (base[i] + correction.to(torch.bfloat16).float()).to(torch.bfloat16)
                logits = (self.cap * torch.tanh(combined.float() / self.cap)).to(torch.bfloat16)
            else:
                logits = self.cap * torch.tanh((base[i] + correction) / self.cap)
            if self.sampling is not None:
                # Rejection compares p and q in the target vocabulary. Retain
                # the exact pre-temperature proposal logits, including zero
                # probability outside this step's candidate set.
                ids = indices[i] if self.shortlist else None
                prev = self.sampling.sample(logits, ids, start + i, i)
            else:
                if self.shortlist:
                    prev = torch.where(logits == logits.amax(-1, keepdim=True), indices[i], self.vocab).amin(
                        -1
                    )
                else:
                    prev = logits.argmax(dim=-1)
            steps.append(prev)
        if self.sampling is not None and self.sampling.adaptive:
            # A low-confidence token may be offered, but no subsequent token is.
            keep = self.sampling.confidence[0] >= 0.75
            for i in range(1, len(steps)):
                steps[i] = torch.where(keep, steps[i], -1)
                keep = keep & (self.sampling.confidence[i] >= 0.75)
        return torch.stack(steps, dim=1)


class DraftSampling:
    """Use vLLM's position-keyed draft RNG and its matching rejection sampler."""

    def __init__(self, width, vocab, device, fp64):
        self.cache = DraftLogitsCache(1, width, vocab, dtype=torch.float32, fill=-float("inf"), device=device)
        self.logits = self.cache.values
        self.adaptive = os.environ.get("INKLING2_DSPARK_ADAPTIVE", "0") == "1"
        self.confidence = [None] * width if self.adaptive else []
        self.temperature = torch.zeros(1, device=device)
        self.seed = torch.zeros(1, dtype=torch.int64, device=device)
        self.slot = torch.zeros_like(self.seed)
        self.columns = [torch.tensor(i, dtype=torch.int32, device=device) for i in range(width)]
        self.fp64 = fp64
        self.topk = int(os.environ.get("INKLING2_DRAFT_SAMPLE_TOPK", "0"))
        if not 0 <= self.topk <= vocab:
            raise ValueError("Invalid proposal top-k")

    def sample(self, values, ids, position, step):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        if 0 < self.topk < values.shape[-1]:
            values, indices = values.topk(self.topk, dim=-1)
            ids = indices[0] if ids is None else ids[indices[0]]
        if self.adaptive:
            scaled = values.float() / self.temperature.clamp_min(0.01)
            self.confidence[step] = scaled.softmax(-1).amax(-1)
        if ids is not None:
            full = torch.full_like(self.logits[:, step], -float("inf"))
            full.scatter_(1, ids.reshape(1, -1), values.float().reshape(1, -1))
        else:
            full = values.float().reshape(1, -1)
        return gumbel_sample(
            full,
            self.slot,
            self.temperature,
            self.seed,
            position.reshape(1),
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.logits,
            logits_cache_col=self.columns[step],
            use_fp64=self.fp64,
        )


def ngram_continuation(tokens, low, high, depth, with_length=False):
    """Continuation after the longest (then most recent) earlier match of the trailing n-gram."""
    n = len(tokens)
    ends = np.flatnonzero(tokens[: n - 1] == tokens[n - 1])
    if not len(ends):
        return (None, 0) if with_length else None
    length = np.ones(len(ends), dtype=np.int64)
    alive = np.ones(len(ends), dtype=bool)
    for j in range(1, min(high, n)):
        back = ends - j
        alive &= back >= 0
        alive[alive] = tokens[back[alive]] == tokens[n - 1 - j]
        if not alive.any():
            break
        length += alive
    good = np.flatnonzero(length >= low)
    if not len(good):
        return (None, int(length.max())) if with_length else None
    best = good[np.lexsort((ends[good], length[good]))[-1]]
    first = int(ends[best]) + 1
    copied = tokens[first : first + depth].tolist()
    return (copied, int(length[best])) if with_length else copied


class InklingDSparkSpeculator(BaseSpeculator):
    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, config, device, req_states=None):
        self.req_states = req_states
        self.width = config.speculative_config.num_speculative_tokens
        if not 1 <= self.width < 8 or config.scheduler_config.max_num_seqs != 1:
            raise ValueError("Inkling DSpark supports one sequence and between one and seven proposals")
        self.empty = torch.zeros(1, self.width, device=device, dtype=torch.long)
        self.sampling = None
        if config.speculative_config.draft_sample_method == "probabilistic":
            self.sampling = DraftSampling(
                self.width, config.model_config.get_vocab_size(), device, config.model_config.use_fp64_gumbel
            )
            self.draft_logits = self.sampling.cache
        # INKLING2_NGRAM_DRAFT copies the continuation of the longest earlier
        # match of the trailing n-gram (num_speculative_tokens deep) and falls
        # back to INKLING2_DRAFT_MAX DSpark tokens, padded with -1.
        self.ngram_max = 0
        if os.environ.get("INKLING2_NGRAM_DRAFT", "0") == "1":
            self.ngram_min = int(os.environ.get("INKLING2_NGRAM_MIN", "3"))
            self.ngram_max = int(os.environ.get("INKLING2_NGRAM_MAX", "8"))
            self.dspark_steps = int(os.environ.get("INKLING2_DRAFT_MAX", "3"))
            if not 1 <= self.ngram_min <= self.ngram_max or not 1 <= self.dspark_steps <= self.width:
                raise ValueError("Invalid n-gram draft settings")
            self.padding = torch.full(
                (1, self.width - self.dspark_steps), -1, device=device, dtype=torch.long
            )
            self.ngram_rounds = 0
            # INKLING2_NGRAM_GATE=1: per request, keep an EMA of tokens accepted per n-gram round. While it is below
            # GATE_ACCEPT only matches of at least GATE_LONG tokens are copied (DSpark drafts the rest), so fresh text
            # stops paying for rejected copy rows; a long match that is accepted re-enables short matches.
            self.gate = os.environ.get("INKLING2_NGRAM_GATE", "0") == "1"
            self.gate_accept = float(os.environ.get("INKLING2_NGRAM_GATE_ACCEPT", "2.0"))
            self.gate_long = int(os.environ.get("INKLING2_NGRAM_GATE_LONG", "12"))
            self.gate_decay = float(os.environ.get("INKLING2_NGRAM_GATE_DECAY", "0.5"))
            # Matches shorter than GATE_LONG copy at most SHORT_DEPTH tokens (default: full width).
            self.short_depth = int(os.environ.get("INKLING2_NGRAM_SHORT_DEPTH", str(self.width)))
            if not 0 <= self.gate_decay < 1 or not 1 <= self.short_depth <= self.width:
                raise ValueError("Invalid n-gram gate settings")
            self.gate_req = None
            self.gate_ema = float(self.width)
            self.last_round = None
            trace = os.environ.get("INKLING2_NGRAM_TRACE")
            self.trace = open(trace, "a", buffering=1 << 16) if trace else None

    @property
    def target(self):
        if _TARGET is None:
            raise RuntimeError("The Inkling model has not bound its DSpark target")
        return _TARGET

    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        *,
        dummy_run=False,
        is_profile=False,
        **kwargs,
    ):
        if dummy_run or is_profile or not input_batch.num_reqs:
            return self.empty[: input_batch.num_reqs]
        if self.sampling is not None:
            self.target.draft.sampling = self.sampling
            self.sampling.temperature.copy_(temperature[input_batch.idx_mapping[:1]])
            self.sampling.seed.copy_(seeds[input_batch.idx_mapping[:1]])
        upper = self.target.draft_end
        if self.target.draft.split_attention:
            start = input_batch.seq_lens[:1].int() - num_rejected[:1].int()
        else:
            start = upper - int(num_rejected[0].item())
        if self.ngram_max:
            copied = self.ngram_propose(input_batch, num_sampled)
            if copied is not None:
                return copied
        anchor = last_sampled[input_batch.idx_mapping[:1]].reshape(1).long()
        draft = self.target.draft
        if not self.ngram_max:
            return draft.propose(self.target, anchor, start, self.width, upper)
        steps = draft.propose(self.target, anchor, start, self.dspark_steps, upper)
        return torch.cat([steps, self.padding], dim=1)

    def ngram_propose(self, input_batch, num_sampled):
        # The target forward stores draft KV and f_ring for every verified row,
        # so a round without the DSpark block leaves the next block's inputs intact.
        req = int(input_batch.idx_mapping_np[0])
        length, sampled = torch.stack(
            [self.req_states.total_len.gpu[req].long(), num_sampled[0].long()]
        ).tolist()
        if input_batch.req_ids[0] != self.gate_req:
            self.gate_req = input_batch.req_ids[0]
            self.gate_ema = float(self.width)
            self.last_round = None
            if self.trace is not None:
                self.trace.write("req\n")
                self.trace.flush()
        if self.last_round is not None and sampled > 0:
            kind, match, count = self.last_round
            accepted = min(sampled - 1, count)
            if kind == "ngram":
                self.gate_ema = self.gate_decay * self.gate_ema + (1 - self.gate_decay) * accepted
            if self.trace is not None:
                self.trace.write(f"{kind} {match} {count} {accepted}\n")
        self.last_round = ("dspark", 0, self.dspark_steps)
        if sampled <= 0 or length <= self.ngram_min:
            return None
        history = getattr(self.req_states.all_token_ids, "_uva_buf", None)
        if history is not None:
            tokens = history.np[req, :length]
        else:
            tokens = self.req_states.all_token_ids.gpu[req, :length].cpu().numpy()
        copied, match = ngram_continuation(tokens, self.ngram_min, self.ngram_max, self.width, with_length=True)
        if copied is None:
            return None
        if match < self.gate_long:
            if self.gate and self.gate_ema < self.gate_accept:
                self.last_round = ("skip", match, self.dspark_steps)
                return None
            copied = copied[: self.short_depth]
        self.last_round = ("ngram", match, len(copied))
        self.ngram_rounds += 1
        count = len(copied)
        row = torch.tensor(
            [list(copied) + [-1] * (self.width - count)], dtype=torch.long
        ).to(self.empty.device)
        if self.sampling is not None:
            # One-hot proposal distribution keeps rejection sampling exact.
            logits = self.sampling.logits[0, :count]
            logits.fill_(-float("inf"))
            logits.scatter_(1, row[0, :count, None], 0.0)
        return row
