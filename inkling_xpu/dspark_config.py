"""vLLM 0.34 speculative method "inkling_dspark" (registered from vllm_plugin.register)."""

from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig


class InklingDSparkSpeculatorConfig(SpeculatorConfig):
    speculator_class = "inkling_xpu.dspark.InklingDSparkSpeculator"

    @classmethod
    def configure(cls, spec):
        super().configure(spec)
        if not 1 <= spec.num_speculative_tokens <= 7:
            raise ValueError("Inkling DSpark supports between one and seven proposals")
