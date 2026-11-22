"""Give Pi the frozen TypeScript checkout and its offline project dependencies."""

import subprocess

DATASET = "SWE-bench/SWE-bench_Multilingual"
PREPARE = """
set -eu
cd /testbed
git -c safe.directory=/testbed checkout -q --detach "$1"
git -c safe.directory=/testbed clean -ffd -q
git -c safe.directory=/testbed diff --quiet "$1" --
expected=$(git -c safe.directory=/testbed rev-parse "$1^{tree}")
git -c safe.directory=/testbed ls-files -z > /tmp/benchmark-files
rm -rf .git
git init -q
git --literal-pathspecs add --force --pathspec-from-file=/tmp/benchmark-files --pathspec-file-nul
test "$(git write-tree)" = "$expected"
git -c user.name=bench -c user.email=bench@local commit -qm base
test -d node_modules
if test -d /root/.cache/node/corepack; then
    cp -a /root/.cache/node/corepack node_modules/.b70-corepack
    chmod -R a+rX node_modules/.b70-corepack
fi
find . -type d -name node_modules -prune -o -exec chown -h "$2" {} +
# Vite, Vitest and Jest can create package-local caches during offline tests.
find . -name node_modules -type d -prune -exec sh -c '
    owner=$1; shift
    for modules do
        for cache in .cache .vite .vite-temp; do
            mkdir -p "$modules/$cache"; chown -R "$owner" "$modules/$cache"
        done
    done' sh "$2" {} +
"""


def prepare(volume, task, uid, gid):
    subprocess.run(
        [
            "docker",
            "volume",
            "create",
            "--label",
            "b70-purpose=pi-typescript-benchmark",
            volume,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            volume + "-init",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw",
            "--pids-limit",
            "128",
            "--memory",
            "2g",
            "--cpus",
            "3",
            "--mount",
            f"type=volume,src={volume},dst=/testbed",
            "--entrypoint",
            "/bin/sh",
            task["image"],
            "-ec",
            PREPARE,
            "prepare",
            task["base_commit"],
            f"{uid}:{gid}",
        ],
        check=True,
        timeout=150,
    )
    return [
        "--mount",
        f"type=volume,src={volume},dst=/testbed,volume-nocopy",
        "--env",
        "COREPACK_HOME=/testbed/node_modules/.b70-corepack",
        "--env",
        "COREPACK_ENABLE_NETWORK=0",
        "--env",
        "CI=1",
    ]


def remove(volume):
    subprocess.run(["docker", "rm", "-f", volume + "-init"], check=False, capture_output=True)
    subprocess.run(
        ["docker", "volume", "rm", volume], check=True, stdout=subprocess.DEVNULL
    )
