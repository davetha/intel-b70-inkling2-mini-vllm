"""vLLM image inputs for the Inkling 2 Mini EXL3 target on Intel XPU."""

from functools import cached_property
from PIL import Image
import torch
from transformers import BatchFeature
from inkling_trellis.inkling2.vendor.image_processing import Inkling2ImageProcessor
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.models.inkling.common.mm_preprocess import (
    IMAGE_MARKER_ID, IMAGE_TOKEN_ID, InklingMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import BaseDummyInputsBuilder, BaseProcessingInfo
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from .vllm_model import Inkling2B70ForCausalLM

# The encoder runs four native 20x20 patches per 40x40 language token.
# 2048 tokens cover the original Pi desktop and mobile screenshots at full resolution.
MAX_PATCHES = 2048
PATCH = 40


def bounded_image(image):
    width, height = image.size
    def count(w, h):
        return ((w + PATCH - 1) // PATCH) * ((h + PATCH - 1) // PATCH)
    if count(width, height) <= MAX_PATCHES:
        return image
    edge = max(width, height)
    lo, hi = 1, edge
    def dimensions(n):
        return max(1, width * n // edge), max(1, height * n // edge)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(*dimensions(mid)) <= MAX_PATCHES:
            lo = mid
        else:
            hi = mid - 1
    return image.resize(dimensions(lo), Image.Resampling.LANCZOS)


class Inkling2ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {"image": None}

    def get_supported_mm_processor_kwargs(self):
        return {}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": MAX_PATCHES}

    @cached_property
    def image_processor(self):
        return Inkling2ImageProcessor()


class Inkling2DummyImages(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return "<|content_image|>" * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {"image": self._get_dummy_images(
            width=64 * PATCH, height=32 * PATCH, num_images=mm_counts.get("image", 0),
            overrides=mm_options.get("image"),
        )}


class Inkling2ImageInputs(InklingMultiModalProcessor):
    # Both Inkling checkpoints use identical rendered image marker and placeholder IDs.
    # Reuse vLLM's marker expansion, including the per-position embedding mask.
    def _apply_hf_processor_main(self, mm_items, hf_kwargs):
        data, _, passthrough = self._get_hf_mm_inputs(mm_items, hf_kwargs)
        images = [bounded_image(im) for im in data.get("images", [])]
        inputs = {"input_ids": [[IMAGE_MARKER_ID] * len(images)], **passthrough}
        if images:
            features = self.info.image_processor.preprocess(images)
            grid = features["vision_grid_thw"]
            inputs.update(vision_patches_bthwc=features["vision_patches_bthwc"],
                          vision_grid_thw=grid, num_patches=grid.prod(-1))
        return BatchFeature(inputs)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        sizes = hf_inputs.get("num_patches", torch.empty(0, dtype=torch.int64))
        return {
            "vision_patches_bthwc": MultiModalFieldConfig.flat_from_sizes("image", sizes),
            "vision_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "num_patches": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }


class Inkling2VisionModelState(DefaultModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if self.supports_mm_inputs:
            # Normalized Inkling text embeddings are FP32. The engine's default
            # BF16 staging buffer would round every text row of a mixed prompt.
            self.encoder_runner.inputs_embeds = self.encoder_runner.inputs_embeds.float()

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        if self.supports_mm_inputs and self.prompt_embeds_state is None and not scheduled_encoder_inputs:
            finished_prefill = (input_batch.num_computed_tokens_np >= input_batch.prefill_len_np).all()
            if finished_prefill or not any(self.encoder_cache.mm_features.get(req)
                                           for req in input_batch.req_ids):
                # The ordinary text model performs embedding lookup inside its
                # native decode graph. All visual rows were consumed at prefill.
                return None
        return super().prepare_inputs_embeds(scheduled_encoder_inputs, input_batch, req_states)


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2ImageInputs, info=Inkling2ProcessingInfo, dummy_inputs=Inkling2DummyImages,
)
class Inkling2B70ForConditionalGeneration(Inkling2B70ForCausalLM, SupportsMultiModal):
    enable_vision = True
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality != "image":
            raise ValueError(f"Unsupported Inkling input modality: {modality}")
        return "<|content_image|>"

    @staticmethod
    def get_model_state_cls():
        return Inkling2VisionModelState

    def get_language_model(self):
        return self

    def embed_multimodal(self, vision_patches_bthwc=None, vision_grid_thw=None, **kwargs):
        if vision_patches_bthwc is None:
            return []
        grids = vision_grid_thw.reshape(-1, 3).cpu()
        counts = grids.prod(-1).tolist()
        device = self.swa_pos.device
        return [self.visual(patches.to(device=device, dtype=torch.bfloat16), grid.reshape(1, 3))
                for patches, grid in zip(vision_patches_bthwc.split(counts), grids)]

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return embeds
        if is_multimodal is None:
            raise ValueError("Missing Inkling image embedding mask")
        return _merge_multimodal_embeddings(embeds, multimodal_embeddings, is_multimodal)

    def load_weights(self, weights):
        expected = {name: param.shape for name, param in self.visual.named_parameters()}

        def checked():
            for name, tensor in weights:
                if name.startswith("model.visual."):
                    shape = expected.pop(name.removeprefix("model.visual."), None)
                    if shape != tensor.shape:
                        raise ValueError(f"Unexpected vision tensor or shape: {name}, {tensor.shape}")
                yield name, tensor
            if expected:
                raise ValueError(f"Missing Inkling vision weights: {sorted(expected)[:5]}")
        return super().load_weights(checked())
