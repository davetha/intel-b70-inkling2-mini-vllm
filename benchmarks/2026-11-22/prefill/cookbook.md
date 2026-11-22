# Comparing the original Inkling B70 recipe with EXL3

The Intel Arc Pro B70 cookbook uses the original Inkling 2 Mini weights. Its first-token latency is appreciably lower in the published 32K test. The comparable quantity here is time from the client request to the first output, including reasoning. The workloads, quantizations, cache types and power caps differ. [Cookbook recipe at the reviewed commit](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/d3ef55afc899e85b7e8cb4c29e6830dfdd453ed1/docs/inkling2-mini/INKLING2-MINI-VLLM-XPU.md)

| B70, one cold request | Prompt tokens | Median first output | Prompt tokens / first-output seconds |
| --- | ---: | ---: | ---: |
| Original, stock kernels and output-head fix, 230 W | 32,768 target | 11.696 s | 2,802 |
| Original, CUTLASS Xe2 grouped experts and DSpark k=7, 230 W | 32,767–32,768 | 9.842 s | 3,330 |
| Local EXL3, 4,096-token chunk and adaptive DSpark, 180 W | 31,376 | 17.77 s | 1,766 |

The original recipe used three entropy-first prompts, each with one generated token. The EXL3 row uses four Pi requests on frozen Django source, across two engine starts. Our engine-only counter gave 1,773 tokens/s on those same inputs. The original recipe's stock-to-CUTLASS comparison also enabled the drafter, so the 18.9% gain measures that combination. [Cookbook raw receipts](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/tree/d3ef55afc899e85b7e8cb4c29e6830dfdd453ed1/results/inkling2-mini/vllm-xpu-20261122) · [Local measurements](measurements.json)

## What carries over

- **Weight loading and grouped GEMMs.** The cookbook retains NVFP4 experts in layers 0–19 and MXFP8 in layers 20–23. Its kernels consume these packed formats and their per-block scales, sort routed rows, run grouped Xe2 GEMMs, and apply the Inkling activation. EXL3 stores MCG trellises and Hadamard scales. Its kernel reconstructs weights inside XMX tiles and fuses the gated activation. Using the donor GEMM would require an EXL3-specific weight loader or an unpacking step whose memory cost and latency must be measured. [Cookbook expert dispatch](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/d3ef55afc899e85b7e8cb4c29e6830dfdd453ed1/patches/inkling2-mini-vllm-xpu/moe_cutlass.py) · [EXL3 packing and activation changes](../../../patches/trellis-inkling-exl3-mcg-situ.patch)
- **Vocabulary projection.** The cookbook avoids rebuilding the output head in FP32. Our target already projects through its resident group-32 INT8 head, which the drafter can share. The cookbook's measured memory saving applies to its original baseline. [Local logits path](../../../inkling_xpu/vllm_model.py)
- **Compilation and prefetch.** The donor builds specifically for BMG-G31 with a large register file and prefetches ahead in the GEMM reduction loop. Our grouped expert kernels already request 256 GRFs for the main DPAS launches and load the next packed weight row into registers. We can isolate the other two changes with the EXL3 weights fixed. One experiment compiles device binaries ahead of time for BMG-G31. The other prefetches packed weights farther ahead during prefill. [Donor build](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/d3ef55afc899e85b7e8cb4c29e6830dfdd453ed1/patches/inkling2-mini-vllm-xpu/build-nvfp4-cutlass-xe2.sh) · [Donor kernel](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/d3ef55afc899e85b7e8cb4c29e6830dfdd453ed1/patches/inkling2-mini-vllm-xpu/nvfp4_xe2_ext.cpp)

## Six-row prefetch on real EXL3 experts

The experimental kernel prefetched packed gate/up and down weights into L2 six reduction rows ahead, while retaining the existing one-row register pipeline. The change applied only to the large 32-row prefill tiles. We loaded all 128 real experts from layer zero and routed each BF16 input row to eight experts with a fixed skew. Four fresh processes ran in reference, candidate, candidate, reference order. After warmup, each observation timed five complete MoE calls including routing and recombination. Each process collected nine observations per length. All four processes produced identical output hashes at every length.

| Input rows | Existing kernel | Six-row prefetch | Change in throughput |
| ---: | ---: | ---: | ---: |
| 512 | 2.623 ms | 2.723 ms | −3.65% |
| 2,048 | 9.732 ms | 10.049 ms | −3.16% |
| 4,096 | 18.876 ms | 19.463 ms | −3.02% |

These are medians of the two process medians per implementation. We kept the existing kernel for serving. [All timings, output hashes and source hashes](expert-prefetch.json)

## Checkpoint and context

| Weight files in bytes | Original | EXL3 |
| --- | ---: | ---: |
| Target, including the vision encoder | 19,526,019,168 | 16,154,070,151 |
| Identical DSpark drafter | 635,801,416 | 635,801,416 |
| Total | 20,161,820,584 | 16,789,871,567 |

EXL3 saves 3.14 GiB on disk. The quantizer reports 97.47% teacher-forced next-token agreement on code and overall perplexity of 4.773 for the original versus 4.784 for EXL3. Our Pi quality study compared EXL3 with Qwen. A comparison of complete agent repairs between the two Inkling weight formats remains open. [Original checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7) · [EXL3 evaluation](https://huggingface.co/grant-ai/Inkling-2-Mini-EXL3-4bpw/blob/c266f76b68429255e1d6ec151d0aa15f48f6abcc/README.md)

The original used BF16 KV. It processed 160,000 input plus 128 output tokens with DSpark and 256,000 plus 128 without a drafter in separate text-only capacity runs. Its 194,128-token KV pool with DSpark served a configured maximum of 163,840. Our EXL3 text profile uses FP8 global KV, completed 262,016 input plus 128 output tokens with DSpark, and passed two real Pi repairs starting at 240,518 input tokens. A run with an active image encoder, a real screenshot and accepted draft tokens is needed to qualify that combination. [Cookbook capacity receipts](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/d3ef55afc899e85b7e8cb4c29e6830dfdd453ed1/results/inkling2-mini/vllm-xpu-20261122/capacity-profiles.json) · [Local capacity receipts](../../2026-11-21/agent-decode/context.json) · [Long Pi repairs](../quality/long-pi.md)
