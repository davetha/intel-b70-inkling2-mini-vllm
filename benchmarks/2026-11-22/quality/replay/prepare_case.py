"""Prepare one pinned public issue for the agent and its private official evaluation record."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, help="Original SWE-bench Verified parquet"
    )
    parser.add_argument("instance")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    selection = json.loads(
        (Path(__file__).resolve().parents[1] / "selection.json").read_text()
    )
    if (
        hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        != selection["parquet_sha256"]
    ):
        parser.error("Dataset digest differs from the frozen panel")
    matches = [
        task for task in selection["tasks"] if task["instance_id"] == args.instance
    ]
    if not matches:
        parser.error("The requested case is outside the frozen panel")
    meta = matches[0]
    data = pq.read_table(
        args.dataset, filters=[("instance_id", "=", args.instance)]
    ).to_pylist()
    if len(data) != 1:
        parser.error("Expected one matching dataset row")
    row = data[0]
    if (
        hashlib.sha256(row["problem_statement"].encode()).hexdigest()
        != meta["issue_sha256"]
    ):
        parser.error("Issue digest differs from the frozen panel")
    if row["base_commit"] != meta["base_commit"] or row["image"] != meta["image"]:
        parser.error("Evaluation image or base commit differs from the frozen panel")
    images_path = Path(__file__).resolve().parents[1] / "evaluation-images.json"
    image = json.loads(images_path.read_text())[args.instance]["image"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "public.json").write_text(
        json.dumps({**meta, "image": image, "issue": row["problem_statement"]})
    )
    (args.output / "private.json").write_text(json.dumps({**row, "image": image}))
    subprocess.run(["docker", "pull", image], check=True)
    with (args.output / "source.tgz").open("wb") as out:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "git",
                image,
                "-C",
                "/testbed",
                "archive",
                "--format=tar.gz",
                meta["base_commit"],
            ],
            stdout=out,
            check=True,
        )


if __name__ == "__main__":
    main()
