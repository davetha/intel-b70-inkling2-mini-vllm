# b70-prod: Inkling 2 Mini on one Intel Arc Pro B70, vLLM 0.34

This branch is the serving stack we run in production. It is
[Danmoreng's intel-b70-inkling2-mini-vllm](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm)
with our additions, ported to vLLM 0.34.0. Danmoreng wrote the original stack: the EXL3 and SYCL kernels,
the vLLM adapter, DSpark drafting, the prefill path and all the measurements in his README (below).
Our changes are opt-in knobs on top of his code. Several are open as PRs to his repo
([#1](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm/pull/1),
[#2](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm/pull/2),
[#4](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm/pull/4)).

Hardware: a single Intel Arc Pro B70 (32 GB). One request at a time, 262,144-token context.

Our additions:
- vLLM 0.34.0 port of the plugin: DSpark registered as a speculative method, plugin-side ring-prefix reuse.
  The runtime image is built on the official `vllm/vllm-openai-xpu` v0.34.0 image.
- IQR GPTQ 4-bit routed experts (`INKLING2_EXPERTS=iqr4`). This is a SYCL decode library plus an int8-DPAS prefill library.
- N-gram copy drafting inside the DSpark speculator, with the width-only draft handoff (`NOCOPY`).
- Prefill fusions: glue fusion, relative bias computed in the attention kernel, a one-pass global table and an unrolled IQR prefill GEMM.
  All are bit-exact against the unfused path.
- Decode: 256-GRF AOT build of the XMX decode attention (no register spills), and 185 fewer standalone cast kernels per step.
  Both are bit-exact.

Community: [Launch80 Discord](https://discord.gg/launch80). Most members run AMD R9700s, and a small group runs Intel Arc.

## Models

| Repo | Experts | Use |
| --- | --- | --- |
| [davetha/Inkling-2-Mini-Uncensored-IQR4-B70](https://huggingface.co/davetha/Inkling-2-Mini-Uncensored-IQR4-B70) | IQR GPTQ 4-bit | this branch (default) |
| [davetha/Inkling-2-Mini-Uncensored-EXL3-B70](https://huggingface.co/davetha/Inkling-2-Mini-Uncensored-EXL3-B70) | EXL3 4bpw | `INKLING2_EXPERTS=exl3` |

Both apply the [OpenUncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored) refusal edit to
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). The model cards list the quality measurements.

## Build and serve

You need Linux x86-64 with the B70's DRM render node, Docker, Git, Python 3 and the Hugging Face `hf` CLI.
Give the scripts exclusive use of the GPU.

```bash
git clone -b b70-prod https://github.com/davetha/intel-b70-inkling2-mini-vllm.git
cd intel-b70-inkling2-mini-vllm

# 1. Images: vLLM 0.34.0 runtime + oneAPI builder (1-2 h; skip if you already have them and set
#    INKLING2_RUNTIME / INKLING2_BUILDER to their tags)
bash scripts/build-images-034.sh

# 2. Model (about 16 GB)
hf download davetha/Inkling-2-Mini-Uncensored-IQR4-B70 \
  --revision 8301907a28db7a4e71beeeb59406e7ddbbeebf95 --local-dir models/iqr4-unc

# 3. Kernels into .work/build/ (the first step checks one library on the GPU)
export INKLING2_RENDER_NODE=/dev/dri/by-path/pci-0000:03:00.0-render   # your B70: ls -l /dev/dri/by-path/
bash scripts/build-b70-prod.sh

# 4. Serve (OpenAI-compatible, model name inkling2-mini-exl3-b70)
bash scripts/serve-b70-prod.sh
```

In a second terminal:

```bash
curl -fsS http://127.0.0.1:8093/health
curl -s http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling2-mini-exl3-b70", "reasoning_effort": "medium", "max_tokens": 2048,
  "messages": [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]}'
```

The first start compiles JIT kernels and takes a few minutes. The server runs in a container named
`inkling2-vllm-exl3`. Stop it with Ctrl-C or `docker stop inkling2-vllm-exl3`.

## Settings

`scripts/serve-b70-prod.sh` sets these values, and each one can be overridden from the environment.
`scripts/serve-coding-agent.sh dspark` and `scripts/serve-vllm-exl3.sh` set the rest of the profile.

| Variable | Value | Meaning |
| --- | --- | --- |
| `INKLING2_MODEL_DIR` | `models/iqr4-unc` | Checkpoint directory, relative to the checkout |
| `INKLING2_EXPERTS` | `iqr4` | Routed experts from the IQR GPTQ-4 library (`exl3` for the EXL3 checkpoint) |
| `INKLING2_HOST` / `INKLING2_VLLM_PORT` | `127.0.0.1` / `8093` | API bind address and port (our server uses `0.0.0.0` / `8202`) |
| `INKLING2_DEADLINE_EPOCH` | `4102444800` | Container stop time. The upstream default is 22:44 Europe/Berlin each day |
| `INKLING2_RENDER_NODE` | upstream default `pci-0000:03:00.0` | B70 render node, set it for your machine |
| `INKLING2_RUNTIME` / `INKLING2_BUILDER` | `local/b70-inkling-vllm:034-runtime` / `local/inkling2-oneapi-034:2026.1.1` | Images from `scripts/build-images-034.sh` |
| `INKLING2_CXX_STD` | `c++20` | C++ standard for kernel builds (the 0.34 builder needs C++20) |
| `INKLING2_ONEDNN_DETERMINISTIC` | `1` | Fixed oneDNN reduction order, so the same GEMM gives the same bits on every run |
| `INKLING2_NGRAM_DRAFT` | `1` | Copy drafts: continue the longest earlier match of the trailing n-gram. DSpark drafts otherwise |
| `INKLING2_NGRAM_MIN` / `INKLING2_NGRAM_MAX` | `3` / `32` | Shortest and longest n-gram matched |
| `INKLING2_NGRAM_DEPTH` | `7` | Copied draft length and vLLM `num_speculative_tokens` |
| `INKLING2_NGRAM_NOCOPY` | `1` | Pass only each draft's valid width to the scheduler, not a copy of the tokens. Output is exact |
| `INKLING2_DRAFT_MAX` | `3` | DSpark draft tokens per step when there is no n-gram match |
| `INKLING2_PREFILL_PROFILE` | `optimized` | Danmoreng's accelerated prefill flag set (see `serve-coding-agent.sh`) |
| `INKLING2_VLLM_BATCH` | `4096` | Prefill chunk (`--max-num-batched-tokens`) |
| `INKLING2_LOCAL_BIAS_TILE` | `64,256,8` | Tile (rows, cols, warps) of the local relative-bias table kernel. Output is bit-identical for any tile |
| `INKLING2_PREFILL_GLUE` | `1` | Fused prefill glue kernels, bit-exact |
| `INKLING2_PREFILL_REL_BIAS` | `1` | Relative bias computed inside the 256-GRF prefill attention kernel (`inkling2-rel-attention.so`) |
| `INKLING2_PREFILL_GLOBAL_TABLE` | `1` | One-pass global-attention bias table |
| `INKLING2_DECODE_XMX256` | `1` | Decode XMX attention from the 256-GRF AOT library (`inkling2-decode-xmx.so`), no spills |
| `INKLING2_DECODE_CASTS` | `1` | Cached FP32 norm weights, int32 positions once per step, BF16 q converted in the XMX kernel |
| `INKLING2_DECODE_XMX_LIB` | `/lab/.work/build/inkling2-decode-xmx.so` | Decode attention library (path inside the container) |

Fixed by the launchers: DSpark with probabilistic drafts (top-k 20) and block verification, FP8 E4M3 KV cache
(4,311,744,512 bytes), 262,144-token context, one sequence, prefix reuse for the most recent sequence, and XPU decode graphs.
Adaptive DSpark (`INKLING2_DSPARK_ADAPTIVE=1`) cannot be combined with `NOCOPY`.

## Measurements

These numbers come from our tuning log, on one B70 with one request at a time and the full 262,144-token profile.
We report each value as the range over two back-to-back runs. Our column is this branch's settings (runs of 2026-11-24/25).

| Benchmark | This branch | Danmoreng main, same harness |
| --- | ---: | ---: |
| Prefill, 4K cold prompt (tok/s) | 5,842–5,875 | 3,736–3,800 |
| Prefill, 64K cold prompt (tok/s) | 4,723–4,726 ¹ | 3,271–3,288 |
| Prefill, 192K cold prompt (tok/s) | 3,364–3,395 | 2,571–2,574 |
| Decode after 4K, greedy (tok/s) | 159.5–160.3 | 64.3–66.5 ² |
| Decode after 192K, greedy (tok/s) | 112.7–113.1 | 49.0–50.4 ² |
| Fresh generation, sampled (tok/s) | 105.3–105.4 | 82.1 ³ |
| Copy-heavy code edits, greedy (tok/s) | 327.8–328.2 | 159.5 ³ |
| Copy-heavy code edits, sampled (tok/s) | 308.6–309.3 ⁴ | not measured |

How we measured:
- Prefill and decode use `scripts/bench-sycl-baseline.py`, which reports vLLM's own prompt and decode rates.
  Each prompt is a cold synthetic logbook with an embedded code to retrieve, followed by 128 greedy output tokens.
  4K is the median of three prompts (seeds 37-39) after one warm-up request. Individual 4K prompts decode at
  157-237 tok/s, depending on how well the drafts match. 64K and 192K are single prompts. Retrieval passed in every row.
- Fresh generation uses six write-new-code and explain prompts at the model's default sampling, with thinking off and a
  natural stop (at most 1,024 tokens). The rate is completion tokens divided by wall time.
- Copy-heavy runs paste eight 120-250 line Python files and ask for each whole file back with one small edit (thinking off).
- These two harnesses are local scripts and are not in this repo.
- Danmoreng column: his main at `92e2347` on vLLM 0.33 with the EXL3 4bpw checkpoint and his default coding profile
  (4 slots, adaptive DSpark up to 7 drafts, `INKLING2_PREFILL_PROFILE=optimized`), measured on 2026-11-24.

¹ Measured with the current prefill settings, before the two decode changes (those changes do not touch prefill).
² Per-prompt 4K decode was 63.5 / 111.9 / 66.5 tok/s. In our tests, adaptive DSpark stops proposing on two of the
  three random-digit logbook prompts, so this benchmark understates his decode on ordinary text.
³ EXL3 uncensored checkpoint on his main plus our IQR PR branch code (the EXL3 path is unchanged), same 4-slot
  adaptive profile, 2026-11-23. Single runs.
⁴ Last measured with `INKLING2_DECODE_XMX256=1` before `INKLING2_DECODE_CASTS=1` was added. Not re-measured since.

Quality checks, run when each change was adopted: a teacher-forced logprob probe matched the previous build
(top-1 100%, KL 0.0000), and the output text was identical at 4K and 192K. Tool calls, multi-turn tool results and
a coding check passed. There were 0/16 refusals on our refusal prompt set.
Known issue: `reasoning_effort: "none"` still produces reasoning text, because the chat template has no no-think path.

---

# Upstream README (Danmoreng)

# Inkling 2 Mini on Intel Arc Pro B70

Serve [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on a single **32 GB Intel Arc Pro B70** using vLLM 0.33, 4-bit EXL3
experts and native SYCL kernels. The vLLM endpoint supports ordinary decoding,
the checkpoint's DSpark drafter, streamed tool calls and single-sequence
prefix reuse. A second recipe runs the Unsloth 4-bit GGUF in llama.cpp.

On ten blind Pi repairs (six Python and four frontend TypeScript issues), adaptive Inkling passed 7/10 and returned all seven before the deadline. A focused instruction gave 8/10 passing patches, six returned in time. Qwen 27B passed and returned 3/10. On the four TypeScript issues, the two Inkling profiles each passed three and Qwen passed one. On the separate streaming-codec task Qwen passed one of two seeds and Inkling passed neither. [Paired results, deadlines, decode measurements and long-context repairs](benchmarks/2026-11-22/quality/README.md)

For a Pi Coding Agent session that starts close to the 262,144-token limit, set `compaction.keepRecentTokens` to `4096` in the session profile and leave `reserveTokens` at `16384`. On one blind Django repair with 240,518 initial Inkling tokens, both seeds compacted and passed the official tests. With the default 20,000-token retention, both seeds exhausted the window. Medium reasoning and the 8,192-token thinking budget were active in all four runs. [Measured Pi settings and outcomes](benchmarks/2026-11-22/quality/long-pi.md)

For faster prompt processing with adaptive DSpark, append `--max-num-batched-tokens 4096` to the coding launcher. On the B70 this improved cold prefill by 3.5–3.8% across 4.8K, 16.6K and 31.4K input tokens through Pi. At 31.4K, median time to first output fell from 18.44 to 17.77 seconds. At 240,518 tokens it saved 8.94 seconds of prefill. The long coding repair passed with both settings and took longer overall with the larger chunk. [Qwen comparison, Strata Intel kernels and replay](benchmarks/2026-11-22/prefill/README.md)

For text sessions with frequent medium-length cold prompts, set `INKLING2_VLLM_BATCH=6144` when starting the coding launcher. At 16,635 input tokens this reduced median first-output latency from 10.30 to 9.23 seconds compared with 4,096. At 31,376 tokens the difference was 0.15 seconds, with 1.31 GiB less free VRAM. The 4,096 setting has the 240K repair qualification. [Scheduler measurements and Pi results](benchmarks/2026-11-22/prefill/chunk-sweep.md)

For browser screenshots, use the optional image-enabled coding profile below. A real Pi browser session on the B70 correctly read a randomized image and used DSpark to verify the response. [Image handling, screenshot and draft counters](benchmarks/2026-11-22/vision/README.md)

```bash
INKLING2_VISION=1 INKLING2_DSPARK_ADAPTIVE=1 bash scripts/serve-coding-agent.sh dspark --max-num-batched-tokens 4096
```

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
