"""Apply a frozen patch to an independent official image and run its dataset evaluation script."""

import argparse
import json
import subprocess as sp
from pathlib import Path
from uuid import uuid4

from swebench.harness.grading import get_eval_report
from swebench.harness.utils import make_test_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("private", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("patch", help="candidate file, BASE or GOLD")
    a = p.parse_args()
    r = json.loads(a.private.read_text())
    spec = make_test_spec(r)
    a.out.mkdir(exist_ok=True, parents=True)
    patch = (
        ""
        if a.patch == "BASE"
        else r["patch"]
        if a.patch == "GOLD"
        else Path(a.patch).read_text()
    )
    path = (a.out / "input.patch").resolve()
    path.write_text(patch)
    script = (a.out / "evaluation.sh").resolve()
    script.write_text(spec.eval_script)
    log = a.out / "test.log"
    name = "b70-quality-grade-" + uuid4().hex[:10]
    network = "none"
    if r["instance_id"] == "psf__requests-5414":
        # Four scored tests need a real 100 ms TCP connect timeout. Reserve their
        # fixture IP on an isolated link with no gateway route or external peers.
        network = name + "-timeout"
        sp.run(
            [
                "docker",
                "network",
                "create",
                "--internal",
                "--subnet",
                "10.255.255.0/29",
                "--gateway",
                "10.255.255.6",
                "--aux-address",
                "tarpit=10.255.255.1",
                "--opt",
                "com.docker.network.bridge.gateway_mode_ipv4=isolated",
                network,
            ],
            check=True,
            stdout=sp.DEVNULL,
        )
    try:
        with log.open("w") as stream:
            proc = sp.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--init",
                    "--name",
                    name,
                    "--network",
                    network,
                    "--cpus",
                    "3",
                    "--memory",
                    "5g",
                    "--pids-limit",
                    "512",
                    "--security-opt",
                    "no-new-privileges",
                    "--mount",
                    f"type=bind,src={path},dst=/input.patch,readonly",
                    "--mount",
                    f"type=bind,src={script},dst=/evaluation.sh,readonly",
                    "--env",
                    "PIP_NO_INDEX=1",
                    "--entrypoint",
                    "/bin/bash",
                    r["image"],
                    "-c",
                    "set -e; cd /testbed; git config --global --add safe.directory /testbed; if test -s /input.patch; then git apply --whitespace=nowarn /input.patch; fi; echo B70_PATCH_APPLIED; exec /bin/bash /evaluation.sh",
                ],
                check=False,
                stdout=stream,
                stderr=sp.STDOUT,
                timeout=450,
            )
    finally:
        sp.run(["docker", "rm", "-f", name], check=False, capture_output=True)
        if network != "none":
            sp.run(["docker", "network", "rm", network], check=True, stdout=sp.DEVNULL)
    predicted = {
        "instance_id": r["instance_id"],
        "model_name_or_path": "b70",
        "model_patch": patch,
    }
    report = get_eval_report(spec, predicted, str(log), True)[r["instance_id"]]
    report["docker_exit"] = proc.returncode
    report["applied"] = "B70_PATCH_APPLIED" in log.read_text(errors="replace")
    report["fail_to_pass_total"] = len(spec.FAIL_TO_PASS)
    report["pass_to_pass_total"] = len(spec.PASS_TO_PASS)
    (a.out / "grade.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        r["instance_id"],
        a.out.name,
        report["resolved"],
        report["infra_failure"],
        report["docker_exit"],
        flush=True,
    )


if __name__ == "__main__":
    main()
