"""Register the separate Inkling model in each vLLM process."""

import os


def install_draft_prefix_handler():
    """Hand the scheduler only the valid (non -1) prefix of variable-width drafts.

    Plugin-side replacement for the 0.33 adaptive-draft patch to DraftTokensHandler:
    always copy drafts back (not only for structured outputs) and drop -1 padding.
    """
    from vllm.v1.outputs import DraftTokenIds
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    if getattr(DraftTokensHandler, "_inkling_prefix", False):
        return
    set_tokens, get_tokens = DraftTokensHandler.set_draft_tokens, DraftTokensHandler.get_draft_tokens

    class AlwaysCopy:
        has_structured_output_reqs = True

        def __init__(self, batch):
            self.req_ids = batch.req_ids

    def set_draft_tokens(self, input_batch, draft_tokens):
        set_tokens(self, AlwaysCopy(input_batch), draft_tokens)

    def get_draft_tokens(self):
        drafts = get_tokens(self)
        if drafts is None:
            return None
        return DraftTokenIds(drafts.req_ids, [[t for t in row if t >= 0] for row in drafts.draft_token_ids])

    DraftTokensHandler.set_draft_tokens = set_draft_tokens
    DraftTokensHandler.get_draft_tokens = get_draft_tokens
    DraftTokensHandler._inkling_prefix = True


def register():
    from vllm import ModelRegistry
    from vllm.v1.worker.gpu.spec_decode.speculator_config import register_speculative_method

    ModelRegistry.register_model("Inkling2B70ForCausalLM", "inkling_xpu.vllm_model:Inkling2B70ForCausalLM")
    ModelRegistry.register_model(
        "Inkling2B70ForConditionalGeneration", "inkling_xpu.vision:Inkling2B70ForConditionalGeneration"
    )
    register_speculative_method("inkling_dspark", "inkling_xpu.dspark_config.InklingDSparkSpeculatorConfig")
    if os.environ.get("INKLING2_RING_PREFIX", "0") == "1":
        from .ring_prefix import install

        install()
    if "1" in (os.environ.get("INKLING2_NGRAM_DRAFT", "0"), os.environ.get("INKLING2_DSPARK_ADAPTIVE", "0")):
        install_draft_prefix_handler()
