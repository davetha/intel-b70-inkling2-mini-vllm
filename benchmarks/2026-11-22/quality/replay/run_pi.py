"""Run the installed Pi CLI, its built-in tools and native session management inside a clean case image."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess as sp
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

import pi_wire as wire


def metrics(url):
    names = (
        "request_prefill_time_seconds_sum",
        "request_decode_time_seconds_sum",
        "spec_decode_num_drafts_total",
        "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total",
        "generation_tokens_total",
        "prompt_tokens_cached_total",
        "prompt_tokens_total",
    )
    with urlopen(url + "/metrics", timeout=5) as resp:
        body = resp.read().decode()
    return {
        name: sum(
            map(
                float,
                re.findall(
                    r"^vllm:" + name + r"\{[^\n]+?\} ([0-9.eE+-]+)", body, re.MULTILINE
                ),
            )
        )
        for name in names
    }


pkg = Path(os.environ["B70_PI_MODULES"]).resolve()
node = Path(os.environ["B70_PI_NODE"]).resolve()
profile = Path(os.environ["B70_PI_PROFILE"]).resolve()
net_script = """const n=require('node:net');n.createServer(c=>{const u=n.connect('/wire/model.sock');c.pipe(u).pipe(c);u.on('error',()=>c.destroy());c.on('error',()=>u.destroy());}).listen(8177,'127.0.0.1')"""


def run(*cmd, **kw):
    return sp.run(list(map(str, cmd)), check=True, **kw)


def event_reader(proc, path, state):
    with path.open("w") as out:
        for line in proc.stdout:
            if line.startswith(
                (b'{"type":"message_update"', b'{"type":"tool_execution_update"')
            ):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            typ = event.get("type")
            row = {"type": typ, "at": time.time()}
            if typ == "message_end":
                m = event["message"]
                if m.get("role") != "assistant":
                    continue
                row.update(
                    stop=m.get("stopReason"),
                    error=m.get("errorMessage"),
                    usage=m.get("usage"),
                )
                state["last_stop"] = row["stop"]
                state["error"] = row["error"] or ""
            elif typ in ("compaction_start", "compaction_end"):
                row.update(
                    reason=event.get("reason"),
                    aborted=event.get("aborted"),
                    error=event.get("errorMessage"),
                )
                result = event.get("result") or {}
                row.update(
                    {
                        k: result.get(k)
                        for k in (
                            "tokensBefore",
                            "estimatedTokensAfter",
                            "firstKeptEntryId",
                        )
                        if k in result
                    }
                )
                state["compactions"].append(row)
            elif typ == "tool_execution_start":
                row["tool"] = event.get("toolName")
                state["tools"] += 1
            elif typ == "agent_settled":
                state["settled"] = not event.get("aborted", False)
            elif typ not in (
                "agent_start",
                "agent_end",
                "summarization_retry_attempt_start",
            ):
                continue
            print(json.dumps(row), file=out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    for name in ("public", "archive", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix", type=Path)
    ap.add_argument("--wall", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focused", action="store_true")
    ap.add_argument("--reserve", type=int)
    ap.add_argument("--keep-recent", type=int)
    a = ap.parse_args()
    d = a.output.resolve()
    d.mkdir(parents=True, exist_ok=False)
    (d / "sessions").mkdir(mode=0o700)
    task = json.loads(a.public.read_text())
    uid = os.getuid()
    gid = os.getgid()
    box = "b70-quality-pi-" + uuid4().hex[:10]
    state = {
        "tools": 0,
        "compactions": [],
        "last_stop": None,
        "error": "",
        "settled": False,
    }
    source = (
        a.prefix.read_text() + "\n\nIssue to resolve in this checkout:\n"
        if a.prefix
        else ""
    ) + task["issue"]
    original = json.loads(profile.read_text())["providers"]["b70-vllm"]["models"][0]
    model = {
        **{
            k: original[k]
            for k in ("reasoning", "input", "cost", "thinkingLevelMap", "compat")
            if k in original
        },
        "id": a.model,
        "name": a.model,
        "contextWindow": 262144,
        "maxTokens": 16384,
        "samplingParams": {
            **original["samplingParams"],
            "thinking_token_budget": 8192,
            "seed": a.seed,
        },
    }
    if a.model.startswith("inkling"):
        model["compat"] = {**model["compat"], "thinkingFormat": "openai"}
        model["compat"].pop("chatTemplateKwargs", None)
    cfg = {
        "providers": {
            "b70-vllm": {
                "api": "openai-completions",
                "apiKey": "local",
                "baseUrl": "http://127.0.0.1:8177/v1",
                "models": [model],
            }
        }
    }
    settings = {"thinkingBudgets": {"medium": 8192}, "compaction": {"enabled": True}}
    if a.reserve or a.keep_recent:
        settings["compaction"] = {"enabled": True}
        if a.reserve:
            settings["compaction"]["reserveTokens"] = a.reserve
        if a.keep_recent:
            settings["compaction"]["keepRecentTokens"] = a.keep_recent
    (d / "models.json").write_text(json.dumps(cfg))
    (d / "settings.json").write_text(json.dumps(settings))
    with tempfile.TemporaryDirectory(prefix="b70-pi-wire-") as tmp:
        server = wire.start(
            Path(tmp) / "model.sock", a.url, a.model, uuid4().hex, d / "wire.jsonl"
        )
        server.metrics_fn = metrics
        proc = None
        try:
            run(
                "docker",
                "run",
                "--rm",
                "-d",
                "--init",
                "--name",
                box,
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                f"{uid}:{gid}",
                "--read-only",
                "--pids-limit",
                "240",
                "--memory",
                "4g",
                "--cpus",
                "3",
                "--tmpfs",
                f"/testbed:rw,exec,size=1g,uid={uid},gid={gid},mode=755",
                "--tmpfs",
                "/tmp:rw,exec,size=650m,mode=1777",
                "-v",
                f"{tmp}:/wire:ro",
                "-v",
                f"{node}:/opt/pi-node:ro",
                "-v",
                f"{pkg}:/opt/pi-modules:ro",
                "-v",
                f"{a.archive.resolve()}:/input/source.tgz:ro",
                "-v",
                f"{d}/sessions:/pi-sessions:rw",
                "-v",
                f"{d}/models.json:/input/models.json:ro",
                "-v",
                f"{d}/settings.json:/input/settings.json:ro",
                "-e",
                "HOME=/tmp",
                "-e",
                "PI_CODING_AGENT_DIR=/tmp/pi-profile",
                "-e",
                "PI_OFFLINE=1",
                "-e",
                "PI_SKIP_VERSION_CHECK=1",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "-e",
                "PYTHONPATH=/testbed",
                "-e",
                "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin",
                "--entrypoint",
                "sleep",
                task["image"],
                "infinity",
                stdout=sp.DEVNULL,
            )

            def execute(*args, **kw):
                return run("docker", "exec", "-w", "/testbed", box, *args, **kw)

            initial = (
                execute(
                    "sh",
                    "-ec",
                    "tar xzf /input/source.tgz; git init -q; git add -f .; git -c user.name=bench -c user.email=bench@local commit -qm base; mkdir -p /tmp/pi-profile; cp /input/models.json /input/settings.json /tmp/pi-profile; git rev-parse HEAD",
                    capture_output=True,
                    timeout=100,
                )
                .stdout.decode()
                .strip()
                .splitlines()[-1]
            )
            run("docker", "exec", "-d", box, "/opt/pi-node", "-e", net_script)
            args = [
                "/opt/pi-node",
                "/opt/pi-modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js",
                "--mode",
                "json",
                "--print",
                "--provider",
                "b70-vllm",
                "--model",
                a.model,
                "--thinking",
                "medium",
                "--no-extensions",
                "--no-mcp",
                "--no-skills",
                "--no-prompt-templates",
                "--session-dir",
                "/pi-sessions",
            ]
            if a.focused:
                args += [
                    "--append-system-prompt",
                    "Reproduce the reported behavior with a small focused check. Inspect the affected callers and related implementations before editing. Verify the changed code on that check and on an adjacent case. Once the change passes these checks, finish promptly.",
                ]
            before = metrics(a.url)
            start = time.monotonic()
            with (
                (d / "stderr.log").open("w") as err,
                tempfile.TemporaryFile() as input_file,
            ):
                input_file.write(source.encode())
                input_file.seek(0)
                bounded = [
                    "timeout",
                    "--signal=KILL",
                    str(a.wall),
                    "sh",
                    "-c",
                    'echo $$ > /tmp/pi-main.pid; exec "$@"',
                    "pi",
                    *args,
                ]
                proc = sp.Popen(
                    ["docker", "exec", "-i", "-w", "/testbed", box, *bounded],
                    stdin=input_file,
                    stdout=sp.PIPE,
                    stderr=err,
                )
                reader = threading.Thread(
                    target=event_reader,
                    args=(proc, d / "events.jsonl", state),
                    daemon=True,
                )
                reader.start()
                try:
                    code = proc.wait(timeout=a.wall + 9)
                    timedout = (
                        code in (124, 137, -9)
                        and time.monotonic() - start >= a.wall - 0.25
                    )
                except sp.TimeoutExpired:
                    execute(
                        "sh",
                        "-c",
                        'pid=$(cat /tmp/pi-main.pid); kill -KILL "$pid" 2>/dev/null || true',
                        timeout=5,
                    )
                    try:
                        code = proc.wait(timeout=3)
                    except sp.TimeoutExpired:
                        proc.kill()
                        code = proc.wait(timeout=2)
                    timedout = True
                reader.join(timeout=3)
            elapsed = time.monotonic() - start
            execute("git", "add", "-A")
            patch = execute(
                "git", "diff", "--cached", "--binary", initial, capture_output=True
            ).stdout
            (d / "candidate.patch").write_bytes(patch)
            listing = sorted(
                str(f.relative_to(d)) for f in (d / "sessions").rglob("*.jsonl")
            )
            (d / "session-paths.json").write_text(json.dumps(listing))
            after = metrics(a.url)
            delta = {k: after[k] - before[k] for k in before}
            saved = {
                "id": task["instance_id"],
                "model": a.model,
                "seed": a.seed,
                "focused": a.focused,
                "long": bool(a.prefix),
                "reserve_override": a.reserve,
                "keep_recent_override": a.keep_recent,
                "wall_seconds": elapsed,
                "wall_limit": a.wall,
                "timed_out": timedout,
                "exit_code": code,
                "completed": not timedout
                and code == 0
                and state["settled"]
                and state["last_stop"] == "stop",
                "prompt_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "patch_bytes": len(patch),
                "engine_delta": delta,
                **state,
            }
            (d / "result.json").write_text(json.dumps(saved, indent=2) + "\n")
            print(
                task["instance_id"],
                a.model,
                saved["completed"],
                state["last_stop"],
                round(elapsed, 1),
                len(state["compactions"]),
                len(patch),
                flush=True,
            )
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
            sp.run(["docker", "rm", "-f", box], check=False, capture_output=True)
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
