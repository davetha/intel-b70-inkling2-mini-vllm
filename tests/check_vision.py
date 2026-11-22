"""Check the serving processor and checkpoint encoder using two images on CPU."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from inkling_trellis.inkling2.vendor.inkling2_vision import Inkling2VisionEncoder
from inkling_xpu.vision import (
    IMAGE_MARKER_ID,
    IMAGE_TOKEN_ID,
    Inkling2B70ForConditionalGeneration,
)
from inkling_xpu.vllm_plugin import register

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/exl3"


def main():
    register()
    torch.set_num_threads(1)
    serving_config = json.loads((ROOT / ".work/vllm-inkling/config/config.json").read_text())
    serving_config["architectures"] = ["Inkling2B70ForConditionalGeneration"]
    with TemporaryDirectory() as directory:
        Path(directory, "config.json").write_text(json.dumps(serving_config))
        config = ModelConfig(
            model=str(CHECKPOINT), tokenizer=str(CHECKPOINT), hf_config_path=directory,
            tokenizer_mode="inkling", dtype="bfloat16", limit_mm_per_prompt={"image": 64},
            max_model_len=262144, enforce_eager=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tokenizer)
        images = [Image.new("RGB", (81, 41), "red"), Image.new("RGB", (40, 40), "blue")]
        items = processor.info.parse_mm_data({"image": images})
        prompt = [27, IMAGE_MARKER_ID, 31, IMAGE_MARKER_ID, 47]
        result = processor.apply(ProcessorInputs(prompt, items), TimingContext(enabled=False))
        expected = [27, IMAGE_MARKER_ID] + [IMAGE_TOKEN_ID] * 6 + [31, IMAGE_MARKER_ID, IMAGE_TOKEN_ID, 47]
        assert result["prompt_token_ids"] == expected
        positions = result["mm_placeholders"]["image"]
        assert [(p.offset, p.length) for p in positions] == [(1, 7), (9, 2)]
        features = processor.info.image_processor.preprocess(images)

    checkpoint_config = json.loads((CHECKPOINT / "config.json").read_text())
    encoder = Inkling2VisionEncoder(checkpoint_config["vision_config"], checkpoint_config["text_config"]["hidden_size"])
    weights = {}
    for shard in sorted(CHECKPOINT.glob("dense-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith("model.visual."):
                    weights[key.removeprefix("model.visual.")] = reader.get_tensor(key)
    encoder.load_state_dict(weights, strict=True, assign=True)
    encoder.eval()
    target = SimpleNamespace(visual=encoder, swa_pos=torch.tensor([0]))
    patches = features["vision_patches_bthwc"]
    grids = features["vision_grid_thw"]
    with torch.inference_mode():
        outputs = Inkling2B70ForConditionalGeneration.embed_multimodal(target, patches, grids)
        for output, part, grid in zip(outputs, patches.split([6, 1]), grids, strict=True):
            reference = encoder(part, grid.reshape(1, 3))
            assert torch.isfinite(output).all() and torch.equal(output, reference)
        assert not torch.equal(outputs[0][0], outputs[1][0])
    print(f"Verified {len(weights)} vision tensors, mixed prompt positions and individual encoder features.")


if __name__ == "__main__":
    main()
