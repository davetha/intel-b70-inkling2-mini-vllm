"""G0 CPU smoke for the vLLM 0.34 Inkling plugin port (no GPU, no weights)."""
import os
from types import SimpleNamespace

import numpy as np
import torch

os.environ.update(INKLING2_RING_PREFIX="1", INKLING2_NGRAM_DRAFT="1")
from inkling_xpu import vllm_plugin, dspark
from inkling_xpu.dspark_config import InklingDSparkSpeculatorConfig
from inkling_xpu.ring_prefix import RingPrefix

vllm_plugin.register()
vllm_plugin.register()  # idempotent in every process
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import registered_speculative_methods

assert registered_speculative_methods() == ["inkling_dspark"], registered_speculative_methods()
spec = SpeculativeConfig(method="inkling_dspark", num_speculative_tokens=7, draft_sample_method="probabilistic",
                         rejection_sample_method="block")
assert spec.custom_method == "inkling_dspark" and spec.method is None, (spec.custom_method, spec.method)
assert type(spec.speculator_config) is InklingDSparkSpeculatorConfig
for bad in (0, 8):
    try:
        SpeculativeConfig(method="inkling_dspark", num_speculative_tokens=bad)
    except Exception as e:  # validation must reject widths outside 1..7
        print("rejects width", bad, type(e).__name__)
    else:
        raise AssertionError(f"width {bad} accepted")

cfg = SimpleNamespace(speculative_config=spec, scheduler_config=SimpleNamespace(max_num_seqs=1),
                      model_config=SimpleNamespace(get_vocab_size=lambda: 64, use_fp64_gumbel=False))
req_states = object()
spec_obj = init_speculator(cfg, torch.device("cpu"), req_states)
assert type(spec_obj) is dspark.InklingDSparkSpeculator and spec_obj.req_states is req_states
assert spec_obj.draft_logits is spec_obj.sampling.cache and spec_obj.draft_logits.row_mapping is None
assert tuple(spec_obj.sampling.logits.shape) == (1, 7, 64) and spec_obj.ngram_max == 8
try:
    spec_obj.target
except RuntimeError:
    pass
else:
    raise AssertionError("unbound target")
marker = object()
dspark.bind_target(marker)
assert spec_obj.target is marker

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
h = object.__new__(DraftTokensHandler)
h._copy_pending, h._deferred_req_ids = False, None
h._draft_tokens_by_req = {"a": np.array([5, 6, -1, -1]), "b": np.array([-1, -1, -1, -1])}
out = h.get_draft_tokens()
assert out.req_ids == ["a", "b"] and out.draft_token_ids == [[5, 6], []], out

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import RequestStatus
assert KVCacheManager._inkling_ring_prefix


class Coord:
    def __init__(self, cached): self.cached, self.freed = cached, []
    def find_longest_cache_hit(self, hashes, bound):
        n = 0
        while n < len(hashes) and n < self.cached and (n + 1) * 16 <= bound:
            n += 1
        return list(range(n)), n * 16, 0
    def free(self, rid): self.freed.append(rid)
    def reset_prefix_cache(self): return True


m = object.__new__(KVCacheManager)
m.enable_caching, m.enable_kv_cache_events, m.log_stats, m.max_model_len = True, False, False, 1 << 20
m.coordinator, m.empty_kv_cache_blocks = Coord(1000), "EMPTY"
m.create_kv_cache_blocks = lambda b: ("BLOCKS", len(b))
m.inkling_history = RingPrefix(16)
hashes = list(range(1, 202))
h_ = m.inkling_history
h_.scheduled("a", 2048, 2048)
for end in range(2052, 2177, 4):
    h_.scheduled("a", end, 4)
req = SimpleNamespace(request_id="a", num_tokens=2174, num_computed_tokens=2174, block_hashes=hashes,
                      status=RequestStatus.FINISHED_STOPPED, prefix_cache_hit_limit=None)
m.free(req)
assert m.coordinator.freed == ["a"]
req2 = SimpleNamespace(request_id="b", num_tokens=3001, block_hashes=hashes, prefix_cache_hit_limit=None)
req2.get_max_cache_hit_length = lambda bound=None: min(x for x in (req2.num_tokens - 1, bound) if x is not None)
blocks, hits, _ = m.get_computed_blocks(req2)
assert hits == 2160 and blocks == ("BLOCKS", 135), (hits, blocks)
assert m.get_num_cached_tokens(req2) == 2160
blocks, hits, _ = m.get_computed_blocks(req2, 2015)
assert hits == 0, hits
req3 = SimpleNamespace(request_id="c", num_tokens=3001, block_hashes=hashes[:128] + [-1] * 73, prefix_cache_hit_limit=None)
req3.get_max_cache_hit_length = req2.get_max_cache_hit_length
req3.get_max_cache_hit_length = lambda bound=None: min(x for x in (3000, bound) if x is not None)
assert m.get_computed_blocks(req3)[1] == 2048
assert m.reset_prefix_cache() and m.get_computed_blocks(req2)[1] == 0
aborted = SimpleNamespace(request_id="a", num_tokens=10, num_computed_tokens=10, block_hashes=hashes,
                          status=RequestStatus.FINISHED_ABORTED)
m.free(aborted)
assert m.get_num_cached_tokens(req2) == 0
print("G0 plugin smoke PASS")
