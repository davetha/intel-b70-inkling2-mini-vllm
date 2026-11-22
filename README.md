# Inkling 2 Mini on Intel Arc Pro B70

Serve [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on a single **32 GB Intel Arc Pro B70** using vLLM 0.33, 4-bit EXL3
experts and native SYCL kernels. The vLLM endpoint supports ordinary decoding,
the checkpoint's DSpark drafter, streamed tool calls and single-sequence
prefix reuse. A second recipe runs the Unsloth 4-bit GGUF in llama.cpp.

For a Pi Coding Agent session that starts close to the 262,144-token limit, set `compaction.keepRecentTokens` to `4096` in the session profile and leave `reserveTokens` at `16384`. On one blind Django repair with 240,518 initial Inkling tokens, both seeds compacted and passed the official tests. With the default 20,000-token retention, both seeds exhausted the window. Medium reasoning and the 8,192-token thinking budget were active in all four runs. [Measured Pi settings and outcomes](benchmarks/2026-11-22/quality/long-pi.md)

For faster prompt processing with adaptive DSpark, append `--max-num-batched-tokens 4096` to the coding launcher. On the B70 this improved cold prefill by 3.5–3.8% across 4.8K, 16.6K and 31.4K input tokens through Pi. At 31.4K, median time to first output fell from 18.44 to 17.77 seconds. [Qwen comparison, Strata Intel kernels and replay](benchmarks/2026-11-22/prefill/README.md)

## Validated profile

| Setting | Reference configuration |
| --- | --- |
| GPU | 1 × Intel Arc Pro B70, 32 GB, 180 W |
| Checkpoint | [Inkling 2 Mini EXL3 4bpw](https://huggingface.co/grant-ai/Inkling-2-Mini-EXL3-4bpw/tree/c266f76b68429255e1d6ec151d0aa15f48f6abcc), MCG expert weights, BF16 dense weights |
| Runtime | vLLM 0.33.0, PyTorch 2.15.0+xpu, Level Zero |
| Context | 262,144 input and output tokens combined |
| Prefill | Grouped XMX expert GEMM, 2,048-token chunks, oneDNN prefix SDPA |
| KV cache | FP8 E4M3, 4 GiB + 16 MiB |
| Serving | Text and tools, one sequence, TP1, native XPU decode graphs, validated prefix reuse |
| API | `http://127.0.0.1:8093/v1`, model `inkling2-mini-exl3-b70` |

## Coding-agent performance

These measurements use one 32 GB Arc Pro B70, one active request and the full 262,144-token server profile with tool support and prefix reuse. The target uses four-bit EXL3 expert weights. DSpark proposes three tokens and vLLM verifies them as a block.

With medium reasoning and a cumulative 8,192-token budget, two 5K code prompts reached 59.16 and 58.82 client tok/s in ordinary mode, up from 58.14 and 57.77. The rebuilt DSpark profile with fused draft taps reached 84.59 and 83.66 on those prompts. At 196,608 input tokens, ordinary measured 1,394 prefill / 43.22 decode tok/s and fused DSpark measured 1,389 / 64.71. Both cold requests retrieved the code. On two cold source prompts of about 232K tokens, fused DSpark reached 76.73 and 68.21 client decode tok/s with 1,312–1,316 prefill tok/s. The coding profile uses 128 target partitions for four verification rows from 196,608 tokens. Set `INKLING2_DSPARK_LONG_SPLITS=0` to compare the old schedule and `INKLING2_DRAFT_FUSED_TAPS=0` to compare the tensor implementation of the draft input mix. [Five-experiment study, task times and reproduction commands](benchmarks/2026-11-21/agent-tuning/README.md)

On the four original editing cases with medium reasoning, fused DSpark and the matched tensor implementation both solved all four, in 190.27 and 206.34 s respectively. For the measured coding panel, start `INKLING2_DSPARK_ADAPTIVE=1 bash scripts/serve-coding-agent.sh dspark`. The adaptive verifier completed the same four in 174.19 s and all four harder cases in 452.83 s. Fixed-width fused DSpark solved three of the four harder cases in 566.55 s. The 192K and 232K figures above use fixed-width verification. The archived Qwen run solved them in 145.88 s. Fused Inkling generated 16,219 tokens across tool turns versus 8,130 for Qwen. [Matched task runs and thinking-budget checks](benchmarks/2026-11-21/agent-tuning/README.md)

### Earlier reference runs

| Frozen source review | Ordinary client tok/s | DSpark client tok/s | Archived Qwen client tok/s |
| --- | ---: | ---: | ---: |
| ~5K Inkling input tokens | 58.1 | 82.6 | 60.6 |
| ~77K Inkling input tokens | 50.6 | 64.9 | 53.3 |
| ~155K Inkling input tokens | 44.3 | 56.7 | 44.6 |

Each median covers three reviews of the same source text as Qwen’s archived 4K, 64K and 128K cases. Each run produced 1,024 outputs at temperature 1, top-p 0.95 and top-k 20. Client decode timing starts with the first streamed output. The source-review text has no correctness grade.

| Coding tasks (four runs per row) | Hidden suites passed | Total wall time (s) | Generated tokens |
| --- | ---: | ---: | ---: |
| Inkling ordinary, original fixtures, none | 4/4 | 154.8 | 8,149 |
| Inkling DSpark, original fixtures, none | 4/4 | 113.7 | 9,454 |
| Inkling ordinary, original fixtures, medium | 4/4 | 304.3 | 17,139 |
| Inkling DSpark, original fixtures, medium | 4/4 | 194.2 | 16,361 |
| Archived Qwen, original fixtures, medium | 4/4 | 145.9 | 8,130 |
| Inkling ordinary, held-out fixtures, none | 1/4 | 312.1 | 17,182 |
| Inkling DSpark, held-out fixtures, none | 1/4 | 232.3 | 19,145 |

The original fixtures cover range-set editing and dependency planning. The held-out fixtures cover routing and a streaming codec. Task times include tool calls and failed attempts. Medium runs request an 8,192-token thinking budget and a 16,384-token output limit. [Per-task results and test settings](benchmarks/2026-11-21/agent-decode/README.md)

| Cold logbook prompt | Ordinary prefill / decode (tok/s) | DSpark prefill / decode (tok/s) |
| --- | ---: | ---: |
| 4,096 input + 128 output | 1866.7–1868.1 / 59.9–59.9 | 1844.5–1844.7 / 106.7–114.8 |
| 196,608 input + 128 output | 1,390 / 42.5 | 1,381 / 57.3 |
| 262,016 input + 128 output | 1,290 / 37.3 | 1,285 / 40.6 |

The logbook timings come from vLLM’s native counters. Each 4K row spans two warm-engine requests with separate cold prompts. The 262,016-token prompt plus 128 outputs fills the advertised context. Each prompt asks for an embedded seven-digit code.

Ordinary reused 196,608 prompt tokens after the client discarded its first continuation. First output arrived in 0.40 s with reuse and 143.4 s with a fresh cache salt. The reuse run missed the code. The fresh-salt follow-up missed it.

DSpark reused 196,608 prompt tokens after the client discarded its first continuation. First output arrived in 0.88 s with reuse and 142.2 s with a fresh cache salt. The reuse run retrieved the code. The fresh-salt follow-up missed it.

Retrieval on the initial cold 192K and 256K requests (ordinary: 2/2, dspark: 2/2).

[Raw context measurements](benchmarks/2026-11-21/agent-decode/context.json) · [Source-review measurements](benchmarks/2026-11-21/agent-decode/source-review.json) · [Previous public checkpoint](benchmarks/2026-11-21/agent-decode/checkpoint-smoke.json)

## Original prefill measurements

Measured on November 21, 2026 with one request at a time and 128 completion
tokens. The A/B rows use the same tokenized prompts. The 4K row is the second
control request. The 256K row fills the context with 262,016 input tokens
and 128 output tokens.

| Context | ESIMD prefill tok/s | Hybrid oneDNN prefill tok/s | Hybrid decode tok/s |
| --- | ---: | ---: | ---: |
| 4K | 1,847 | 1,866 | 20.1 |
| 32K | 1,652–1,661 | 1,718–1,755 | 20.0 |
| 192K | 953 | **1,393** | 19.3 |
| 256K | — | **1,290** | 19.2 |

At 192K, hybrid attention reduced prefill time from 206.33 s to 141.15 s
(+46.2% throughput). It was inactive at 4K because that prompt had no eligible
8,192-token prefix block. The 192K and 256K requests retrieved a validation
code inserted in the middle of a synthetic prompt. [Method and raw measurements](benchmarks/2026-11-21/README.md)

## Build and serve

Start with the [setup instructions](docs/SETUP.md) for container images, device
selection and checkpoint downloads. The exact reference image IDs correspond
to the local artifacts from the
[Qwen B70 recipe](https://github.com/Danmoreng/intel-b70-qwen38-vllm/tree/0356a370da5f73f6b0a4eab613d7103a2356b513).
The setup also describes a source-build option for a new machine.

From the repository root, after preparing the images and checkpoint:

```bash
bash scripts/check-sycl.sh
bash scripts/check-sycl.sh exl3
INKLING2_EXL3_TILE_ROWS=32 bash scripts/check-sycl.sh exl3 --tokens 256 512

bash scripts/build-loop-control.sh
bash scripts/serve-coding-agent.sh dspark
```

In a second terminal, check readiness and run a short benchmark:

```bash
curl -fsS http://127.0.0.1:8093/health
python3 scripts/bench-sycl-baseline.py --url http://127.0.0.1:8093 \
  --vllm-model inkling2-mini-exl3-b70 --lengths 4096 --replicates 2 \
  --timeout 90 --output results/vllm-4k.jsonl
```

Use `bash scripts/serve-coding-agent.sh ordinary` to run the same endpoint
with ordinary decoding. Both modes support the OpenAI-compatible chat and
completion endpoints, including automatic tool calls on chat requests.
The 262,144-token limit includes the generated continuation.

For the earlier GGUF comparison, the llama.cpp recipe measured 33.3 tok/s
with DSpark and 26.5 tok/s with ordinary decode at 4K.
[llama.cpp build and launch commands](docs/SETUP.md#llamacpp-and-dspark)

## How the kernels work

- The [EXL3 patch](patches/trellis-inkling-exl3-mcg-situ.patch) adds the MCG
  codebook and Inkling's SiTU activation to Trellis. The grouped prefill
  kernel decodes 4-bit expert weights into XMX tiles. The tuned recipe uses
  32 activation rows per tile.
- The [paged attention patch](patches/trellis-vllm-block-table.patch) lets
  the SYCL decode kernel read vLLM's physical KV pages in BF16 or FP8 E4M3.
  The vLLM adapter keeps the per-visit sliding-window and convolution state
  in device rings. Some supporting operations use XPU Triton.
- The [hybrid prefill](inkling_xpu/prefix_attention.py) reuses the
  [Qwen oneDNN operator](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/0356a370da5f73f6b0a4eab613d7103a2356b513/engine/direct-xpu/runtime/csrc/exl3_ops.sycl)
  for older keys that are visible to every query and outside Inkling's
  relative-bias band. The ESIMD kernel handles the causal tail, relative
  bias and attention sink. The adapter merges the outputs using FP32
  softmax normalizers. Set `INKLING2_ONEDNN_PREFIX_BLOCK=0` for the full ESIMD
  control. The optional W8A8 experiment targets dense projections and gave
  comparable throughput to BF16 at 16K.

Kernel sources are pinned to [Trellis Serve](https://github.com/sybil-solutions/trellis-serve/tree/8f8e2fccf0c82c9715b774812f30d9d52f78a5fb)
and [Unsloth llama.cpp](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8).
Model files and built images are obtained separately. [Third-party notices](NOTICE.md)
