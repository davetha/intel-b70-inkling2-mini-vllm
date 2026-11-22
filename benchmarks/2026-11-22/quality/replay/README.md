# Replaying a task with Pi Coding Agent

Pi 1.5.0 owns the session, the built-in read, bash, edit and write tools, and automatic compaction. The benchmark starts the real CLI inside an unprivileged official case container. A Unix socket exposes the selected local vLLM chat endpoint to that container. Each trial uses a fresh cache salt and a new checkout. The host records the effective request settings, first-response timing, usage, Pi events, the saved session and the resulting patch. [Pi protocol](../pi-protocol.json)

Install the Pi package and supply a Linux x64 Node executable compatible with the case image. The published measurements use Node v24.10.0. The profile supplied here contains the sampled model settings. The client reads the dataset, agent checkout and private grading record from separate paths. Run the following commands from this directory with Docker, Python and a running B70 endpoint.

```bash
npm install --prefix "$PI_INSTALL" @earendil-works/pi-coding-agent@1.5.0
export B70_PI_MODULES="$PI_INSTALL/node_modules"
export B70_PI_NODE="$PORTABLE_LINUX_NODE"
export B70_PI_PROFILE="$PWD/provider-profile.json"

python prepare_case.py "$SWE_VERIFIED_PARQUET" django__django-10880 case
python run_pi.py --public case/public.json --archive case/source.tgz \
  --url http://127.0.0.1:8093 --model inkling2-mini-exl3-b70 --output ink-run
python run_pi.py --public case/public.json --archive case/source.tgz \
  --url http://127.0.0.1:8081 --model Qwen3.8-27B --output qwen-run
```

Load one model at a time. The default task deadline is 480 seconds. Add `--focused` to test the instruction to reproduce, inspect callers, test an adjacent case and finish promptly. `--seed` changes the provider's session seed. `wire.jsonl` records completed HTTP requests and `events.jsonl` records Pi compaction results. For a successful compaction, compare its event timestamp with the prompt-token count from the next tool-enabled HTTP request. An initial response at 239k tokens is a separate measurement from the continuation after compaction.

The preparation command requires PyArrow. The grading commands use the SWE-bench harness revision and dataset digest in the selection file. Each patch runs in an independent official image. [Frozen selection](../selection.json) · [Official image digests](../evaluation-images.json)

```bash
python grade_patch.py case/private.json grades/base BASE
python grade_patch.py case/private.json grades/gold GOLD
python grade_patch.py case/private.json grades/ink ink-run/candidate.patch
python grade_patch.py case/private.json grades/qwen qwen-run/candidate.patch
```

For the long repair, prepare `django__django-14631` and run `build_corpus.py case/source.tgz "$QWEN_TOKENIZER_JSON" "$INKLING_TOKENIZER_JSON" long.txt`. It verifies the file order and corpus digest. Pass `--prefix long.txt --wall 900` to Pi for the long case. For the measured 4,096-token retention configuration add `--keep-recent 4096`. Use `--seed 101` for the second trial. The short control receives only the issue. The campaign warms Pi, checks cold prefill using a 55,000-character excerpt sent through Pi, and uses the repository's `scripts/guard-prefill.py` with a 650 tok/s floor for the long first request.

For Requests 5414 in the archived all-Python panel, the grader creates a temporary isolated Docker bridge. Four scored preservation tests expect a 100 ms TCP connect timeout to 10.255.255.1. The bridge reserves the address without assigning it to a peer and has no external route. The agent's checkout remains offline. Both the original base and the official reference were checked with this setup.

To test Pi's history retention, add `--keep-recent 4096` on the long case. This passes `compaction.keepRecentTokens` through the documented Pi settings file. The compaction threshold, 262,144-token context, 8,192-token reasoning budget and original source text stay fixed. [Cut points obtained from the original Pi session](../pi-cutpoints.jsonl)

### TypeScript web cases

The amended ten-case panel uses SWE-bench Verified for the first six Python cases and SWE-bench Multilingual for the last four frontend cases. Supply the corresponding parquet to `prepare_case.py`. The selection records the upstream revision, input digests and all four replacement IDs. The Docker image digest is pinned per case. [Amended selection](../selection.json) · [Multilingual dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual)

For Vue and Docusaurus the runner creates a fresh disposable Docker volume from the official case image. It checks out the exact base commit, removes the upstream Git history and verifies the new single-commit checkout against the original Git tree. The volume retains the project's installed Node dependencies so Pi can run the repository's frontend tests offline. For Docusaurus the independent grader also restores the tracked files to the frozen base before applying a candidate patch. The image build can change the root package manifest to pin Yarn. The grader retains the installed offline dependencies and runs the upstream evaluation script. Only the independent grader receives the test patch and reference fix. It must reproduce a failing baseline and a passing reference before each case enters the model comparison.

| Position | Repository and repair | Frontend behavior |
| ---: | --- | --- |
| 7 | Vue 11739 | Hydrate server-rendered components whose children bind styles and use CSS variables from Vue expressions. |
| 8 | Docusaurus 10309 | Switch documentation versions when the current version is served at the root URL. |
| 9 | Vue 11915 | Preserve literal double braces inside an HTML textarea marked with `v-pre`. |
| 10 | Docusaurus 9183 | Highlight fenced code blocks regardless of capitalization in the language identifier. |
