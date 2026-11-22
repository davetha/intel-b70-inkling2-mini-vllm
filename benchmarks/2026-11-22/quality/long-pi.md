# One long repair through Pi Coding Agent

The frozen Django forms repair supplied 69 files from the base checkout with the issue. Pi 1.5.0 ran its own tools, session and automatic compaction. The judge checked both regression and preservation tests against the resulting patch in the official image. The same input text occupied 240,518 tokens for Inkling and 198,579 for Qwen. Both used medium thinking, an 8,192-token thinking budget per reply, a 262,144-token window and 16,384 reserved tokens. [Corpus and order](long-source.json) · [Pi protocol](pi-protocol.json) · [Request and grade summary](pi-results.json)

| Model and profile | Seed | Retain recent tokens | First cold prefill, tok/s | Pi exit | Official tests | Seconds |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Inkling, adaptive DSpark | 42 | 20,000 | 1,304 | Context exhausted | Failed | 311 |
| Inkling, adaptive DSpark | 101 | 20,000 | 1,304 | Context exhausted | Failed | 266 |
| Inkling, adaptive DSpark | 42 | 4,096 | 1,304 | Completed | Passed | 611 |
| Inkling, adaptive DSpark | 101 | 4,096 | 1,302 | Completed | Passed | 607 |
| Inkling, fixed-width DSpark | 42 | 20,000 | 1,304 | Context exhausted | Failed | 302 |
| Qwen3.8-27B production | 42 | 20,000 | 1,008 | Completed | Passed | 585 |

The 4,096-token setting let Pi select a valid boundary earlier. Its compaction events reported 246,241 and 246,065 tokens before the summary. The next tool-enabled requests actually used 6,997 and 6,285 tokens. Both sessions continued with tool calls and produced passing patches. A replay of the default trajectory found zero eligible boundaries at 20,000 recent tokens. This is evidence from one near-full initial prompt and two seeds. [Boundary replay](pi-cutpoints.jsonl)

In the issue-only controls Inkling finished and passed in 166 s. Qwen's patch passed after the 480 s timeout. An earlier Qwen long attempt exceeded 900 s during cleanup, so its patch was unavailable for judging. The table reports the completed long rerun with bounded cleanup and persisted sessions. [Replay instructions and grading procedure](replay/README.md)

The tested near-full Inkling workload used this Pi profile.

```json
{
  "thinkingBudgets": {"medium": 8192},
  "compaction": {"enabled": true, "reserveTokens": 16384, "keepRecentTokens": 4096}
}
```
