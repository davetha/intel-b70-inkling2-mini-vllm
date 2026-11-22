# Inkling EXL3 with browser vision on the B70

The Pi Coding Agent launched a local preview and opened it in Chromium. Its task was to identify two randomly chosen colors and a three-digit number. The agent used the shell only to start the HTTP server. The browser supplied an 820 × 480 screenshot. Its SHA-256 hash matched the exact image forwarded in the subsequent model request.

Inkling correctly identified the orange rectangle, red circle and number 288. The image-bearing turn had one image cache miss and 41 accepted DSpark tokens. The three-request Pi session took 24.57 seconds, including first-use warmup. It used medium reasoning, an 8,192-token thinking budget and automatic compaction. [Pi and engine counters](pi-screenshot.json)

## Serving configuration

Set `INKLING2_VISION=1` and `INKLING2_DSPARK_ADAPTIVE=1` when running the coding launcher with `dspark --max-num-batched-tokens 4096`. This one-sequence profile exposes 262,144 total tokens. The XPU worker reported 263,168 KV-cache tokens and 17.85 GiB for model loading. The encoder uses the BF16 vision weights included in the EXL3 checkpoint. Its 422 parameters matched the checkpoint shapes. Two small color panels also produced the same features as independent calls to the upstream encoder in the CPU reference test.

Each 40 × 40 image tile occupies one language token. The processor preserves image resolution up to 2,048 tiles per image and scales down larger images proportionally. The launch accepts up to 64 images per prompt, subject to the total context and cache budgets. The measured prefill chunk size is 4,096. The text path retains FP32 normalized embeddings when mixing text and pixels and uses the ordinary decode graph after prefill.
