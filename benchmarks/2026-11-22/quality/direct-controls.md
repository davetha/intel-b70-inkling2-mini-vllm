# Direct vLLM controls on one Arc Pro B70

These controls use a custom Python tool loop. Pi Coding Agent has its own system prompt, tools and automatic compaction. The direct-client panel used ten Python issues. The later Pi panel retained the first six and replaced the final four with frontend TypeScript issues before those sessions began. Each panel has its own denominator.

## Routing and streaming codec

With two frozen tasks and two seeds per task, adaptive Inkling returned four correct patches in 452.83 s. The current production Qwen service returned two correct patches in 580.51 s. A third Qwen patch passed the hidden checks on disk at its 240 s deadline. All-attempt totals include deadlines. The Inkling observations are from November 21 and the matched Qwen replay is from November 22. [Control trials](control.json)

## Ten SWE-bench Verified tasks

The ten issues were frozen before model requests. Each agent saw the same issue and had an offline shell in a fresh base checkout. The focused arm added instructions to reproduce the failure, inspect callers, check an adjacent case and finish promptly. Every submitted patch was evaluated in a separate official image after all three solutions were frozen. Each evaluation setup rejected the base and accepted the reference. The sampling settings were medium reasoning, 8,192 thinking tokens per response, up to 16,384 output tokens per response, temperature 1, top-p 0.95, top-k 20 and first-turn seed 42. Short tasks allowed 480 s and up to 64 turns. [Selection and harness revision](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm/blob/94f021ed5cbdfb283d8306b7cb87d139c4653182/benchmarks/2026-11-22/quality/selection.json) · [Runtime settings](runtime.json) · [Official image digests](https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm/blob/94f021ed5cbdfb283d8306b7cb87d139c4653182/benchmarks/2026-11-22/quality/evaluation-images.json)

| Agent | Patches passing / 10 | Passing and returned / 10 | Time for all attempts (s) | Median time for returned passes (s) | Output tokens from completed responses |
| --- | --- | --- | --- | --- | --- |
| Qwen 27B, MTP | 7 | 6 | 2699.7 | 131.0 | 128817 |
| Inkling, adaptive DSpark | 7 | 7 | 2012.5 | 44.9 | 150000 |
| Inkling, adaptive DSpark with focused instructions | 7 | 7 | 1617.2 | 100.2 | 120055 |

| Paired agents (A / B) | Both pass | Only A | Only B | Both fail | Both returned a pass | Total seconds on shared returned passes (A / B) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 27B, MTP / Inkling, adaptive DSpark | 7 | 0 | 0 | 3 | 6 | 897.0 / 618.3 |
| Inkling, adaptive DSpark / Inkling, adaptive DSpark with focused instructions | 7 | 0 | 0 | 3 | 7 | 957.9 / 691.6 |

| Issue | Qwen 27B, MTP | Inkling, adaptive DSpark | Inkling, adaptive DSpark with focused instructions |
| --- | --- | --- | --- |
| django__django-10880 | pass · 197.0 s | pass · 120.8 s | pass · 100.2 s |
| django__django-13344 | fail · 480.0 s (wall_limit) | fail · 367.5 s | fail · 330.1 s |
| django__django-16938 | fail · 480.0 s (wall_limit) | fail · 318.0 s (turn_limit) | fail · 174.8 s |
| django__django-14238 | pass · 196.4 s | pass · 44.9 s | pass · 38.8 s |
| sympy__sympy-24443 | pass · 479.3 s (wall_limit) | pass · 339.6 s | pass · 200.7 s |
| sympy__sympy-21930 | fail · 363.4 s | fail · 369.0 s | fail · 420.7 s (turn_limit) |
| sympy__sympy-24539 | pass · 20.6 s | pass · 17.0 s | pass · 22.4 s |
| sympy__sympy-17139 | pass · 65.6 s | pass · 26.3 s | pass · 47.5 s |
| psf__requests-5414 | pass · 366.3 s | pass · 375.5 s | pass · 177.0 s |
| psf__requests-1142 | pass · 51.2 s | pass · 33.7 s | pass · 105.0 s |

Wall time includes model requests and shell actions. It starts after checkout and ends before independent grading. A pass requires all official regression and preservation checks. The time and tokens spent on failed tasks count towards the all-attempt totals. An interrupted final response has no client output-token count. Tokenizers differ across models. Case order alternated which engine or instruction variant ran first. This sample covers three Python repositories.

## Direct-client long-context observations

Adaptive Inkling evaluated 239,234 cold prompt tokens at 1,305 prefill tok/s and 62.65 first-response decode tok/s. The adaptive and fixed-width direct clients exhausted the context while appending tool history. The last direct Qwen coding run was explicitly stopped when the benchmark switched to Pi. These earlier trials do not measure Pi compaction.

Across the 30 short trials, the largest observed reasoning count in a completed API response was 7,825 tokens. 0 trial(s) recorded a response exceeding the requested 8,192 tokens. The budget resets on each API request.

Full per-attempt results include hidden-test counts, tool use, reasoning-token counts, patch hashes and completion status. [Paired results](paired-results.json)
