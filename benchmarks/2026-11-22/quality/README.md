# Pi Coding Agent on one Intel Arc Pro B70

Pi 1.5.0 drove each coding task through its built-in read, bash, edit and write tools. Its automatic compaction was enabled with the standard 16,384-token reserve and 20,000 recent tokens. Both providers declared a 262,144-token context and used medium reasoning with an 8,192-token budget per response. Each task started a new Pi session in a fresh, offline checkout. The first model request and the measured patch both belong to that session. [Client and compaction settings](pi-protocol.json) · [Model and GPU settings](runtime.json)

## Six Python repairs and four TypeScript frontend repairs

All three arms saw the same issue and base commit. The focused instruction asked the model to reproduce, inspect affected callers, test a nearby case and finish promptly. The 480 s deadline included the Pi process, inference and tool calls. Each frozen patch was evaluated in a separate official image. Evaluation required the original base to fail and the reference to pass.

| Pi profile | Official passes / 10 | Passed and returned / 10 | All task seconds | Median seconds per returned pass | Output tokens from completed requests | Tool calls |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 27B MTP | 3 | 3 | 3574.6 | 215.8 | 178436 | 355 |
| Inkling adaptive DSpark | 7 | 7 | 2613.1 | 246.4 | 186519 | 562 |
| Inkling adaptive + focused instruction | 8 | 6 | 2764.5 | 175.0 | 197272 | 594 |

| Agents A / B | Both pass | Only A | Only B | Both fail | Both returned a pass | Seconds on those cases (A / B) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 27B MTP / Inkling adaptive DSpark | 3 | 0 | 4 | 3 | 3 | 633.2 / 567.0 |
| Inkling adaptive DSpark / Inkling adaptive + focused instruction | 7 | 0 | 1 | 2 | 5 | 1016.4 / 812.2 |

| Issue | Qwen 27B MTP | Inkling adaptive DSpark | Inkling adaptive + focused instruction |
| --- | --- | --- | --- |
| django__django-10880 | fail · 480.1 s (time limit) | pass · 138.4 s | pass · 126.6 s |
| django__django-13344 | fail · 480.1 s (time limit) | pass · 442.8 s | pass · 480.1 s (time limit) |
| django__django-16938 | fail · 414.8 s | fail · 153.3 s | pass · 436.6 s |
| django__django-14238 | pass · 184.6 s | pass · 64.3 s | pass · 37.8 s |
| sympy__sympy-24443 | pass · 215.8 s | pass · 274.1 s | pass · 164.3 s |
| sympy__sympy-21930 | fail · 480.0 s (time limit) | fail · 274.1 s | fail · 75.5 s |
| vuejs__core-11739 | fail · 480.0 s (time limit) | fail · 480.1 s (time limit) | fail · 480.1 s (time limit) |
| facebook__docusaurus-10309 | fail · 480.1 s (time limit) | pass · 246.4 s | pass · 480.1 s (time limit) |
| vuejs__core-11915 | pass · 232.8 s | pass · 228.6 s | pass · 297.9 s |
| facebook__docusaurus-9183 | fail · 126.2 s | pass · 311.0 s | pass · 185.6 s |

The first six cases come from SWE-bench Verified (Django and SymPy). At the user’s request, the last four were replaced before their Pi runs by TypeScript web cases from SWE-bench Multilingual (Vue and Docusaurus). One seed per task estimates performance on this panel. Timed-out requests can lack final usage counters. Token counts use each model’s tokenizer. [Case selection and official grader revision](selection.json) · [Measured Pi attempts and test counts](pi-results.json)

Docusaurus 10309 was rescored on the exact base commit for both controls and all three frozen patches because the image build had changed the tracked Yarn manifest. On Docusaurus 9183 the official reference passed all 36 scored checks. Its evaluation script then exited 1 while trying to restore paths absent at the base revision. Scores use the official test report.

## Python and frontend TypeScript subsets

| Subset | Profile | Cases | Official passes | Passed and returned | All task seconds |
| --- | --- | --- | --- | --- | --- |
| Python | Qwen 27B MTP | 6 | 2 | 2 | 2255.5 |
| Python | Inkling adaptive DSpark | 6 | 4 | 4 | 1347.0 |
| Python | Inkling adaptive + focused instruction | 6 | 5 | 4 | 1320.8 |
| TypeScript web | Qwen 27B MTP | 4 | 1 | 1 | 1319.1 |
| TypeScript web | Inkling adaptive DSpark | 4 | 3 | 3 | 1266.1 |
| TypeScript web | Inkling adaptive + focused instruction | 4 | 3 | 2 | 1443.7 |

## Where the four frontend repairs spent their time

Each row sums the four Pi sessions. The wall clock includes tool execution and model requests. Times for prefill and decode come from the vLLM worker counters. At a deadline these counters can omit an unfinished request.

| Profile | Timeouts / 4 | Tool calls | Wall time (s) | Recorded prefill (s) | Recorded decode (s) |
| --- | --- | --- | --- | --- | --- |
| Qwen 27B MTP | 2 | 180 | 1319.1 | 177.7 | 1047.9 |
| Inkling adaptive DSpark | 1 | 291 | 1266.1 | 179.9 | 983.9 |
| Inkling adaptive + focused instruction | 2 | 325 | 1443.7 | 175.7 | 1157.2 |

## Decode during actual Pi tasks

The throughput here divides total timed output tokens by total vLLM decode seconds across completed sessions with full usage records. The numerator excludes each HTTP request’s first output token. Each session’s generated-token counter had to match the sum of the request usage. This measures decode separately from prefill, Pi tools and process startup. Coverage varies when a session reaches its time limit.

| Profile | Sessions with complete counters | Timed output tokens | Engine decode seconds | Decode tok/s |
| --- | --- | --- | --- | --- |
| Qwen 27B MTP | 5 | 60394 | 981.93 | 61.5 |
| Inkling adaptive DSpark | 9 | 153660 | 1701.87 | 90.3 |
| Inkling adaptive + focused instruction | 7 | 96399 | 1061.34 | 90.8 |

## Reported reasoning use

Each Pi request carried an 8,192-token reasoning budget. These counters include responses for which vLLM returned reasoning-token details. The budget applies per response, so a session with many tool calls can spend more than 8,192 reasoning tokens in total. An interrupted HTTP request may end before vLLM sends its final usage.

| Pi profile | Requests reporting reasoning / all | Total reported reasoning tokens | Most in one response | Responses at or above 8,192 |
| --- | --- | --- | --- | --- |
| Qwen 27B MTP | 336 / 339 | 134112 | 8191 | 0 |
| Inkling adaptive DSpark | 568 / 569 | 108421 | 5854 | 0 |
| Inkling adaptive + focused instruction | 602 / 604 | 116691 | 4359 | 0 |

## Fresh-seed instruction comparison

Two retained Python cases and one of the four new TypeScript cases were selected before their respective runs for another session with seed 101. The sample is separate from the ten first-seed tasks.

| Issue | Inkling default | Inkling focused |
| --- | --- | --- |
| django__django-14238 | pass · 27.0 s | pass · 50.4 s |
| sympy__sympy-24443 | pass · 370.1 s | pass · 245.1 s |
| vuejs__core-11739 | fail · 480.1 s (time limit) | fail · 480.1 s (time limit) |

## Cross-file Django forms repair

The long trials prepended 69 files from the same base checkout to the issue text. Both models received identical source bytes. Each had a 900 s wall limit and a fresh cache salt. Short controls used 480 s. The first long Qwen trial reached its deadline during a harness cleanup failure and lacked a recoverable final patch. The recorded Qwen completion is the independent rerun with container-enforced timing and persistently mounted Pi sessions. The cold rates are from the first request. Post-compaction counts come from the next actual tool-enabled request. Pi can summarize the initial source as it continues. [Corpus contents and SHA-256](long-source.json)

| Pi profile | First prompt | Cached | Cold prefill tok/s | Cold decode tok/s | Compaction starts | Observed continuations | Prompt tokens after compaction | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adaptive Inkling short | 1868 | 0 | 1882.15 | 96.66 | 0 | 0 | — | pass · 166.0 s |
| Adaptive Inkling long | 240518 | 0 | 1304.26 | 68.51 | 0 | 0 | — | fail · 310.8 s (incomplete) |
| Fixed-width Inkling long | 240518 | 0 | 1304.19 | 76.62 | 0 | 0 | — | fail · 302.3 s (incomplete) |
| Qwen short | 1753 | 0 | 2092.1 | 71.42 | 0 | 0 | — | pass · 480.1 s (time limit) |
| Qwen long | 198579 | 0 | 1007.51 | 44.26 | 0 | 0 | — | pass · 585.0 s |

## Pi compaction with a large initial message

The adaptive seed-42 baseline left the large source message in one original turn. Replaying the saved session through Pi 1.5.0 prepareCompaction produced zero eligible boundaries above threshold at keepRecentTokens 20,000 and the first eligible boundary at 246,314 tokens with keepRecentTokens 4,096. The following fresh Pi runs changed only that setting within each seed. The model retained medium reasoning and the 8,192-token per-response budget.

| Pi profile | First prompt | Cached | Cold prefill tok/s | Cold decode tok/s | Compaction starts | Observed continuations | Actual next prompt tokens | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adaptive Inkling long | 240518 | 0 | 1304.26 | 68.51 | 0 | 0 | — | fail · 310.8 s (incomplete) |
| Adaptive, keep 4096, seed 42 | 240518 | 0 | 1304.03 | 68.49 | 1 | 1 | 6997 | pass · 611.2 s |
| Adaptive default, seed 101 | 240518 | 0 | 1304.13 | 66.65 | 0 | 0 | — | fail · 265.5 s (incomplete) |
| Adaptive, keep 4096, seed 101 | 240518 | 0 | 1302.42 | 66.72 | 1 | 1 | 6285 | pass · 607.2 s |

## Routing and streaming codec

The two fixtures used the same sealed specifications, independent acceptance tests and two test seeds on each model. The independent grader applies the whole saved patch, including changes to visible tests, and runs ten separately mounted acceptance tests. The deadline was 240 s per session. The Intel Xe driver reported a copy-engine memory fault at 18:25:40 during the first Qwen routing session. The health endpoint kept returning HTTP 200 while generation stalled. That trial and the following two affected attempts were archived as infrastructure failures. The comparison uses fresh sessions after restarting the same production engine.

| Pi model | Fixture | Seed | Hidden pass | Passed and returned | Seconds |
| --- | --- | --- | --- | --- | --- |
| inkling2-mini-exl3-b70 | mtp-next-routefix-v2 | 911701 | True | True | 129.6 |
| inkling2-mini-exl3-b70 | mtp-next-routefix-v2 | 912101 | True | True | 167.2 |
| inkling2-mini-exl3-b70 | mtp-next-streamcodec-v2 | 911701 | False | False | 124.1 |
| inkling2-mini-exl3-b70 | mtp-next-streamcodec-v2 | 912101 | False | False | 76.5 |
| Qwen3.8-27B | mtp-next-routefix-v2 | 911701 | True | False | 240.1 |
| Qwen3.8-27B | mtp-next-routefix-v2 | 912101 | True | True | 187.9 |
| Qwen3.8-27B | mtp-next-streamcodec-v2 | 911701 | True | True | 165.0 |
| Qwen3.8-27B | mtp-next-streamcodec-v2 | 912101 | False | False | 76.1 |

Only the Pi runs above measure this agent. The archived direct-client comparison used the earlier ten Python cases. [Fixture baseline and independent reference controls](pi-fixture-calibration.json) · [Pi fixture grades](pi-control.json) · [Replay a Pi task](replay/README.md) · [Earlier direct-client controls](direct-controls.md)
