# Inkling 2 Mini prefill kernels on the 180 W B70

A single 32 GB Arc Pro B70 ran the EXL3 checkpoint with FP8 global KV, adaptive DSpark, medium reasoning and an 8,192-token thinking budget. The opt-in coding profile uses 6,144 tokens per scheduler chunk. The previous engine source is `f84741fb79a4b42e68ff6a738c8c9c1e98deceae`. [Boot settings, individual samples and independent grades](prefill-kernel-sweep.json)

## Uncached text through the vision-enabled Pi endpoint

Each observation starts the real Pi Coding Agent with automatic compaction and a fresh cache salt. The engine reported zero cached tokens for every first request. The table reports medians after one shape warmup. Prefill throughput uses the engine timer. First output measures the client interval to the first reasoning or answer delta. The selected profile has two observations at each of the three original lengths on one server start.

| Input tokens | Previous 4,096 chunk (tok/s) | Previous 6,144 chunk (tok/s) | Selected (tok/s) | Gain over faster previous chunk | Selected first output |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,844 | 1,860 | 1,925 | **2,700** | +40.2% | 1.86 s |
| 16,635 | 1,687 | 1,851 | **2,582** | +39.5% | 6.53 s |
| 31,376 | 1,771 | 1,794 | **2,487** | +38.6% | 12.69 s |

On the same source text, the archived Qwen 27B control reached 2,109, 2,160 and 2,084 engine tok/s. Its first outputs arrived at 1.95, 6.68 and 12.41 seconds. The tokenizers differ. At the longest input Qwen processed 25,773 tokens and Inkling processed 31,376. [Qwen control and source digests](measurements.json)

## Isolated changes and combinations

These operator experiments used the text-only architecture. All three original prompt lengths use identical source digests throughout the study. The baseline and candidate operator profiles use 4,096 scheduler tokens. Each row gives median engine tokens per second.

| Experiment | Compared with | 4,844 | 16,635 | 31,376 |
| --- | --- | ---: | ---: | ---: |
| Previous engine | Repeat control | 1,862 | 1,687 | 1,770 |
| Attention and MLP output convolution | Previous | 1,982 | 1,792 | 1,885 |
| All four convolutions | Output convolutions | 2,058 | 1,842 | 1,942 |
| Expert DPAS from 48 input rows | Previous | 1,859 | 1,803 | 1,769 |
| BF16 relative table written directly | Previous | 1,884 | 1,706 | 1,792 |
| Sheared table and native ESIMD block reads | Previous | 1,873 | 1,694 | 1,777 |
| Sheared table and Triton sliding attention | Previous | — | 1,407 | — |
| Four convolutions, threshold 48 and direct bias table | Previous | 2,090 | 2,014 | 1,973 |
| Fused attention preparation | Three changes above | 2,136 | 2,052 | 2,013 |
| Fused controller input and output | Three changes above | 2,262 | 2,176 | 2,128 |
| Cross-pass probability reuse | Three changes above | 2,132 | 2,047 | 2,003 |
| Page-aligned global prefix boundary | Three changes above | 2,086 | 2,044 | 2,010 |
| Joined local QKV weights for shared inputs | Three changes above | 2,085 | 2,016 | 1,978 |
| Selected individual changes together | Three changes above | 2,378 | 2,315 | 2,272 |
| Cached residual query scores | Combined profile | 2,558 | 2,480 | 2,429 |
| Single convolution history snapshot per chunk | Best prior combination | 2,545 | 2,481 | 2,431 |
| Pass BF16 dense outputs into fused preparation | Best prior combination | 2,591 | 2,518 | 2,463 |

The 16,635-token prompt ends in a 251-row chunk at a scheduler width of 4,096. In a warmed diagnostic replay the original expert calls took 718 ms of that chunk’s 923 ms. The isolated checkpoint layer required 12.55 ms for grouped execution and 2.59 ms for DPAS at 251 rows. In a warmed full first chunk of 4,096 rows, the 48 expert visits totaled 840 ms, the short convolutions 253 ms, and depth mixing 237 ms. Synchronization at measured boundaries adds overhead to the diagnostic.

## Scheduler and memory

All rows used the same combined kernels in the vision-enabled model. Free VRAM is the minimum whole-card DRM Xe reading during the two 31,376-token prefills, sampled once per second. The final selection also received the multimodal capacity checks below.

| Scheduler chunk | 4,844 tok/s | 16,635 tok/s | 31,376 tok/s | Lowest sampled free VRAM at 31,376 |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2,508 | 2,438 | 2,371 | 5.94 GiB |
| 4,096 | 2,591 | 2,510 | 2,459 | 4.59 GiB |
| 6,144 | 2,702 | 2,568 | 2,471 | 3.33 GiB |
| 7,168 | 2,702 | 2,571 | 2,483 | 2.56 GiB |
| 8,192 | 2,693 | 2,549 | 2,497 | 2.03 GiB |

With the scheduler fixed at 6,144, direct FP8 page reads and the fused partial-output reduction increased throughput from 2,222 to 2,258 tok/s on 92,170 input tokens and from 2,471 to 2,486 tok/s on 31,376 tokens. Each rate is the median of two cold Pi requests.

## Qualification

- The Django repair (96K corpus excerpt) started with 96,001 uncached tokens. Engine prefill took 42.68 s (2,250 tok/s). Pi finished in 358.7 s with compaction enabled and 55 tool calls. The patch passed both issue regression tests and 115 of 117 preservation tests. The official judge rejected the patch on two file-field tests. This run used the first 332,917 characters of the frozen Django source corpus and seed 42.
- The separate 240,518-token Django Pi repair processed the cold prompt in 130.53 s (1,843 tok/s) and compacted at 245,949 tokens. The next actual request contained 8,088 tokens. The operator stopped the session before grading.
- The fresh Chromium screenshot and text retrieval check at 12,291 tokens passed. Cold prefill took 4.75 s (2,586 engine tok/s). Client first output arrived at 4.83 s with 3.36 GiB of sampled free VRAM.
- The fresh Chromium screenshot and text retrieval check at 196,616 tokens passed. Cold prefill took 101.41 s (1,939 engine tok/s). Client first output arrived at 101.67 s with 3.27 GiB of sampled free VRAM.
- The fresh Chromium screenshot and text retrieval check at 240,006 tokens passed. Cold prefill took 130.21 s (1,843 engine tok/s). Client first output arrived at 130.53 s with 3.26 GiB of sampled free VRAM.
- Pi’s read tool supplied a screenshot confirmed by its provider payload hash. Pixel identification passed and DSpark accepted 158 tokens on image-bearing requests.
- `finalqual-routefix` ran 10 hidden tests. The generated patch passed the suite. Pi completed in 223.1 s.
- `visionbase-routefix` ran 10 hidden tests. The generated patch passed the suite. Pi completed in 202.2 s.
- `visionbase-streamcodec-911701` ran 10 hidden tests. The generated patch passed the suite. Pi completed in 93.4 s.

Target next-token distributions were sampled at matching prefix hashes through Pi.

| Comparison | Matched positions | Same top token | Median total variation | Maximum total variation | Mean KL (nats) |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference repeat | 30 | 30 | 0.00000 | 0.00002 | 0.00000 |
| candidate | 30 | 29 | 0.00732 | 0.99860 | 0.22786 |

The reference used 4,096 tokens per scheduler chunk and the candidate used 6,144. The largest difference occurred at position 16,512 of the 31,376-token input. The campaign ended before the same-chunk logits control, the matched 96K reference repair, the candidate codec repair and the final speed sweep. The measured quality difference therefore has no single established cause.

## Reproduction and rollback

Start the selected vision-enabled coding profile with the following command.

```bash
INKLING2_PREFILL_PROFILE=optimized INKLING2_VISION=1 INKLING2_DSPARK_ADAPTIVE=1 bash scripts/serve-coding-agent.sh dspark
```

For a same-chunk control set `INKLING2_PREFILL_PROFILE=reference` and `INKLING2_VLLM_BATCH=6144`. The default `reference` profile uses the previous 2,048-token launcher default and 256-row expert crossover. Use the existing Pi replay on the frozen corpus. [Pi probe and corpus instructions](README.md#reproducing-the-pi-measurement)
