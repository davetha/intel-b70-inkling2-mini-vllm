"""Measure cold-prefix engine prefill and client TTFT through Pi Coding Agent."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPLAY = Path(__file__).resolve().parents[1] / "quality" / "replay" / "run_pi.py"
RUNTIME = "sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default=RUNTIME)
    args = parser.parse_args()
    source = args.corpus.read_text()
    if len(source) < 105000:
        parser.error("The corpus needs at least 105,000 characters")
    args.output.mkdir(parents=True, exist_ok=False)

    archive = args.output / "source.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("README.txt")
        info.size = 6
        tar.addfile(info, io.BytesIO(b"Ready\n"))

    rows = []
    for repeat, chars in [(0, 0)] + [
        (rep, n) for rep in (1, 2) for n in (12000, 55000, 105000)
    ]:
        dest = args.output / f"r{repeat}-{chars}"
        issue = "The environment is ready. Reply with the single word READY."
        if chars:
            issue = "Reference text:\n" + source[:chars] + "\n\n" + issue
        meta = args.output / f"issue-{chars}.json"
        meta.write_text(
            json.dumps(
                {"instance_id": "pi-prefill-check", "issue": issue, "image": args.image}
            )
        )
        subprocess.run(
            [
                sys.executable,
                REPLAY,
                "--public",
                meta,
                "--archive",
                archive,
                "--output",
                dest,
                "--url",
                args.url,
                "--model",
                args.model,
                "--wall",
                "100",
            ],
            check=True,
            timeout=140,
        )
        result = json.loads((dest / "result.json").read_text())
        wire = json.loads((dest / "wire.jsonl").read_text().splitlines()[0])
        usage = wire["usage"]
        if (
            not result["completed"]
            or wire["status"] != 200
            or usage["prompt_tokens_details"]["cached_tokens"]
        ):
            raise RuntimeError(f"Incomplete session or cache hit in {dest}")
        if chars:
            elapsed = wire["engine_delta"]["request_prefill_time_seconds_sum"]
            rows.append(
                {
                    "repeat": repeat,
                    "chars": chars,
                    "tokens": usage["prompt_tokens"],
                    "prefill_s": elapsed,
                    "prefill_tps": usage["prompt_tokens"] / elapsed,
                    "ttft_s": wire["first_seconds"],
                    "prompt_sha256": hashlib.sha256(issue.encode()).hexdigest(),
                }
            )
            (args.output / "measurements.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
