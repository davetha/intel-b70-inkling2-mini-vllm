"""Guard vLLM prefix hits with the single slot's resident or saved ring history."""


class RingPrefix:
    # Both the convolution and local-attention rings retain this many extra
    # positions beyond their receptive fields. One frontier represents live
    # decode state, the other a snapshot from the last sizeable prompt chunk.
    slack = 32

    def __init__(self, hash_tokens):
        self.hash_tokens = hash_tokens
        self.owner = None
        self.high = 0
        self.snapshot = 0
        self.hashes = ()

    def scheduled(self, owner, end, rows):
        if self.owner != owner:
            self.high = self.snapshot = 0
        self.owner = owner
        self.high = max(self.high, end)
        if rows > 8:
            self.snapshot = end
        self.hashes = ()

    def finish(self, owner, committed, hashes):
        if owner == self.owner:
            self.hashes = tuple(hashes[: min(committed, self.high) // self.hash_tokens])

    def covers(self, length):
        return length > 0 and any(
            end - self.slack <= length <= end for end in (self.high, self.snapshot) if end > 0
        )

    def limit(self, hashes, allowed):
        """Block hashes include the full prefix and the caller's cache salt."""
        available = min(len(self.hashes), len(hashes), allowed // self.hash_tokens)
        for end in (self.high, self.snapshot):
            latest = min(available, end // self.hash_tokens)
            for index in range(latest, 0, -1):
                if index * self.hash_tokens < end - self.slack:
                    break
                if self.hashes[index - 1] == hashes[index - 1]:
                    return index * self.hash_tokens
        return 0


def install():
    """Bound vLLM 0.34 KVCacheManager prefix hits by the single slot's ring history.

    Plugin-side replacement for the 0.33 vllm-single-slot-ring-prefix.patch.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import RequestStatus

    if getattr(KVCacheManager, "_inkling_ring_prefix", False):
        return
    init, computed, alloc, free, reset = (KVCacheManager.__init__, KVCacheManager.get_computed_blocks,
                                          KVCacheManager.allocate_slots, KVCacheManager.free,
                                          KVCacheManager.reset_prefix_cache)

    def __init__(self, kv_cache_config, max_model_len, scheduler_block_size, hash_block_size, *args, **kwargs):
        self.inkling_history = RingPrefix(hash_block_size)
        init(self, kv_cache_config, max_model_len, scheduler_block_size, hash_block_size, *args, **kwargs)

    def get_computed_blocks(self, request, max_cache_hit_length=None):
        if not self.prefix_cache_lookup_enabled(request):
            return computed(self, request, max_cache_hit_length)
        bound = self.inkling_history.limit(request.block_hashes, request.get_max_cache_hit_length(max_cache_hit_length))
        blocks, hits, boundary = computed(self, request, bound)
        if hits and not self.inkling_history.covers(hits):
            return self.empty_kv_cache_blocks, 0, 0
        return blocks, hits, boundary

    def get_num_cached_tokens(self, request):
        if not self.prefix_cache_lookup_enabled(request):
            return 0
        bound = self.inkling_history.limit(request.block_hashes, request.num_tokens - 1)
        _, hits, _ = self.coordinator.find_longest_cache_hit(request.block_hashes, bound)
        return hits if hits and self.inkling_history.covers(hits) else 0

    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                       num_lookahead_tokens=0, num_external_computed_tokens=0, *args, **kwargs):
        blocks = alloc(self, request, num_new_tokens, num_new_computed_tokens, new_computed_blocks,
                       num_lookahead_tokens, num_external_computed_tokens, *args, **kwargs)
        if blocks is not None:
            total = min(request.num_computed_tokens + num_new_computed_tokens + num_external_computed_tokens,
                        self.max_model_len)
            self.inkling_history.scheduled(request.request_id, total + num_new_tokens, num_new_tokens)
        return blocks

    def free_(self, request):
        committed = min(request.num_tokens, request.num_computed_tokens) if request.status in (
            RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED) else 0
        self.inkling_history.finish(request.request_id, committed, request.block_hashes)
        free(self, request)

    def reset_prefix_cache(self):
        self.inkling_history.hashes = ()
        return reset(self)

    def get_computed_blocks_for_connector(self, *args, **kwargs):
        raise NotImplementedError("Inkling ring prefix reuse does not support KV connectors")

    KVCacheManager.__init__ = __init__
    KVCacheManager.get_computed_blocks = get_computed_blocks
    KVCacheManager.get_num_cached_tokens = get_num_cached_tokens
    KVCacheManager.allocate_slots = allocate_slots
    KVCacheManager.free = free_
    KVCacheManager.reset_prefix_cache = reset_prefix_cache
    KVCacheManager.get_computed_blocks_for_connector = get_computed_blocks_for_connector
    KVCacheManager._inkling_ring_prefix = True
