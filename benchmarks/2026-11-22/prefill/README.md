# Inkling prefill on one Intel Arc Pro B70

The [original-checkpoint B70 comparison](cookbook.md) covers the cookbook’s 9.84-second cold 32K prefill, transferable kernel changes and measured context limits with DSpark.

## What Strata v0.2.19 measured

The highest published RTX 5090 prompt result is 11,306 tokens/s with the SQ-20 pack. The explicitly labeled 2K result is 8,722 tokens/s. SQ-20 coding decode is 198.9 tokens/s for one request and output throughput is 4,089.7 tokens/s across 64 requests. The release lists 159.2 tokens/s for B70 DSpark output. It gives no B70 prefill timing, prompt length for the B70 entry or equivalent Pi coding run. The model card describes thinking-disabled measurements. The release also optimizes Qwen3.8-Flash-Next, a separate model from the local Qwen3.8-27B reference. [Release](https://github.com/Niko1221/Strata/releases/tag/v0.2.19) · [Model card and timing table](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata)

The tagged Inkling server normalizes the requested reasoning effort to `none` and sets `enable_thinking=False`. The local B70 profile uses Pi, medium effort and an 8,192-token reasoning budget. [Strata server dialect](https://github.com/Niko1221/Strata/blob/v0.2.19/terminal/serve/inkling_dialect.py#L270-L287)

## Native Intel paths worth comparing

- With XMX enabled and at least 32 input rows, Strata groups routed pairs by expert on the host. Experts receiving at least 12 rows by default use a gather, FP16 weight unpack, oneMKL GEMMs and a scatter. Smaller expert groups share SG16 kernels. This suggests using the occupancy of each expert tile as a dispatch signal in our device-side EXL3 grouping. Our experimental change skips fully empty eight-row DPAS subtiles in the gate/up and down projections. [Routing and dispatch](https://github.com/Niko1221/Strata/blob/v0.2.19/engine/src/inkling2/sycl/ink_sy_runner.h#L576-L610) · [XMX threshold](https://github.com/Niko1221/Strata/blob/v0.2.19/engine/src/inkling2/sycl/ink_sy_xmx.h#L35-L41) · [oneMKL expert](https://github.com/Niko1221/Strata/blob/v0.2.19/engine/src/inkling2/sycl/ink_sy_xmx.h#L459-L489)
- Strata's SG16 attention computes the relative-position term, attention sink, softmax and output sigmoid gate together, reading FP16 KV. Gate fusion is a candidate for our near-context prefill path. Our recipe uses FP8 KV storage and oneDNN for the older global prefix. That existing split raised 196,608-token prefill from 953 to 1,393 tokens/s in the earlier fixed-prompt test. [Strata attention](https://github.com/Niko1221/Strata/blob/v0.2.19/engine/src/inkling2/sycl/ink_sy_ops.h#L246-L327) · [Earlier local 192K comparison](../../2026-11-21/README.md#vllm-exl3)

The local Qwen recipe also points to a different experiment. Its INT8 prefill optimization applies to large EXL3 linear projections. The oneDNN attention operator used for our global prefix accepts FP16 Q/K/V. Qwen's INT8 linear fast path checks the codebook and uses a fixed weight scale for its supported format. Our 4-bit Inkling experts use MCG and fused FP16 DPAS. Testing INT8 there would require MCG-specific scaling, expert gathers and checks against real expert outputs and model predictions. The optional W8A8 switch in the current Inkling recipe applies to dense projections only and had comparable throughput to BF16 at 16K. [Qwen EXL3 implementation](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/0356a370da5f73f6b0a4eab613d7103a2356b513/engine/direct-xpu/runtime/csrc/exl3_ops.sycl) · [Local hybrid attention and W8A8 experiment](../../../README.md)

At 262,144 tokens in one slot, the tagged Intel implementation allocates 1 GiB per full-context target visit for the FP16 keys and values. The eight global visits therefore need 8 GiB. Three full-context DSpark KV buffers add 3 GiB. This is a source-derived estimate before sliding-window caches, weights and scratch allocations. Keeping our FP8 global cache matters when testing 4-bit weights on a 32 GB card. [Target KV allocations](https://github.com/Niko1221/Strata/blob/v0.2.19/engine/src/inkling2/sycl/ink_sy_runner.h#L210-L217) · [Draft allocations](https://github.com/Niko1221/Strata/blob/v0.2.19/engine/src/inkling2/sycl/ink_sy_runner.h#L259-L264)

## Same input text through Pi

The medium probe placed the same first 55,000 characters of the frozen source corpus into both agents. With zero cached input tokens, Qwen needed 6.5–6.7 seconds for 14,347 tokens. Inkling needed 12.0–12.1 seconds in four observations and 10.2 seconds in one additional warmed observation for 16,635 tokens. The long repair presented the same 69 files to both agents. Inkling processed 240,518 tokens in 184.4–184.7 seconds. Qwen processed 198,579 in 197.1 seconds on its completed attempt. Compare seconds for identical input text across tokenizers. These are first-request engine prefill times, independent of subsequent reasoning, tools and test outcomes. [Corpus manifest](../quality/long-source.json)

## Cold prefill on the B70

The table gives medians across four requests per cell. Each server processed the three lengths twice after a short warmup. The six server starts ran in the order reference, 4K chunk, tail kernel, tail kernel, 4K chunk, reference. All runs used adaptive DSpark, EXL3 4-bit experts, FP8 KV, a 262,144-token context, medium thinking and one Pi process at a time. Every request had zero cached input tokens. The 4K-chunk server reported a 263,168-token KV capacity, equal to the reference. [Timings and input digests](measurements.json)

| Input tokens | Released 2,048 chunk, tok/s | 4,096 chunk, tok/s | Relative throughput | 2,048 chunk with experimental expert kernel, tok/s | First output, released → 4,096 chunk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,844 | 1,810 | 1,872 | +3.5% | 1,753 | 2.74 → 2.66 s |
| 16,635 | 1,574 | 1,630 | +3.6% | 1,533 | 10.66 → 10.30 s |
| 31,376 | 1,708 | 1,773 | +3.8% | 1,661 | 18.44 → 17.77 s |

Larger chunks improved throughput in both server runs at all three lengths. The density-aware expert kernel reduced throughput by 2.6–3.1%. Its real-checkpoint FP16 and BF16 outputs were bitwise identical to the reference at eight row counts from 8 through 2,048, using skewed routing over 32 experts. Scheduling gains were measured with the released expert kernel.

## Qwen on the same text

The production Qwen3.8-27B server ran the same Pi probe twice per length on the same B70. The prompt text hashes match across models. Qwen's tokenizer used 14–18% fewer tokens on these inputs. Both the per-token rate and the token count contribute to the wall-time difference. [Individual Qwen and Inkling samples](measurements.json)

| Source characters | Qwen / Inkling input tokens | Qwen / Inkling 4,096-chunk prefill, tok/s | Qwen / Inkling 4,096-chunk time to first output |
| ---: | ---: | ---: | ---: |
| 12,000 | 4,068 / 4,844 | 2,109 / 1,872 | 1.95 / 2.66 s |
| 55,000 | 14,347 / 16,635 | 2,160 / 1,630 | 6.68 / 10.30 s |
| 105,000 | 25,773 / 31,376 | 2,084 / 1,773 | 12.41 / 17.77 s |

## Near-full coding qualification

For the same 69-file Django repair, seed 42 and 4,096-token Pi retention, increasing only the vLLM scheduler chunk reduced initial cold prefill from 184.44 to 175.50 seconds at 240,518 Inkling tokens. Throughput rose from 1,304 to 1,370 tokens/s (+5.1%). The production Qwen control took 197.10 seconds on the same input text (198,579 tokens). These three runs all produced patches that passed the independent official tests. [Reference runs](../quality/long-pi.md) · [4,096-chunk run](measurements.json)

Pi compacted successfully at 245,826 tokens with the larger chunk. Its next actual request used 7,722 tokens. End to end, the 4,096-chunk session took 874.9 seconds, compared with 611.2 seconds for the 2,048-chunk session and 585.0 seconds for Qwen. The larger-chunk trajectory included long-running Django test commands. This single paired repair supports the prefill improvement and provides no evidence of faster task completion. The larger chunk is an opt-in setting for prompt processing.

## Reproducing the Pi measurement

Prepare the frozen Django source corpus and Pi client described in the [replay instructions](../quality/replay/README.md). Give the benchmark exclusive access to one already running endpoint. Each execution of the probe starts a fresh Pi session per observation, warms with a short prompt, then submits three lengths twice with independent cache salts. It verifies zero input cache hits. Engine throughput uses the vLLM prefill timer. Client TTFT counts time from forwarding the HTTP request to the first nonempty output or reasoning delta. Runner startup, model loading and Pi process startup are outside both timers.

```bash
INKLING2_DSPARK_ADAPTIVE=1 bash scripts/serve-coding-agent.sh dspark --max-num-batched-tokens 4096
```

With the server ready and the Pi environment variables set, run the following command from the repository root. For a control run, restart the server with a 2,048-token chunk and use a new output directory.

```bash
python benchmarks/2026-11-22/prefill/probe.py long.txt chunk4096-run \
  --url http://127.0.0.1:8093 --model inkling2-mini-exl3-b70
```
