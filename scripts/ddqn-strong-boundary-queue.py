"""Wait for a free T4, run two fixed boundary treatments, collect before release."""

import json
import subprocess
import sys
import time
from pathlib import Path

SESSION = "dodge-strong-boundary-v1"
ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(
        ["colab", "--auth", "adc", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )


def main():
    for _ in range(180):
        result = cli("new", "--session", SESSION, "--gpu", "T4")
        if result.returncode == 0:
            print("T4_READY", flush=True)
            break
        if "TooManyAssignmentsError" not in result.stdout:
            raise RuntimeError(result.stdout)
        print("QUEUED_WAITING_FOR_FREE_T4", flush=True)
        time.sleep(60)
    else:
        raise TimeoutError("No free T4 after three hours; active jobs untouched")
    uploads = (
        ("/tmp/ddqn-reward-source-v2.tar.gz", "reward-source-v2.tar.gz"),
        (
            "/tmp/ddqn-reward-wheels-v2/dodge_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl",
            "dodge_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl",
        ),
        ("/tmp/ddqn-strong-release/ddqn-t4-screen.py", "ddqn-t4-screen.py"),
        (
            "/tmp/ddqn-strong-release/reward_profiles.py",
            "reward_profiles.py",
        ),
        (
            "history/dodge/gymnasium/cnn-image-ddqn/t4-nstep3s43boundary-200k-v1/config.json",
            "best-config.json",
        ),
    )
    for local, remote in uploads:
        result = cli("upload", "--session", SESSION, local, f"/content/{remote}")
        if result.returncode:
            raise RuntimeError(result.stdout)
    result = cli(
        "exec",
        "--session",
        SESSION,
        "--file",
        "/tmp/ddqn-strong-boundary-setup.py",
        "--timeout",
        "60",
    )
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("Setup failed; inspect allocated session before release")
    from colab_cli.common import state

    endpoint = state.store.get(SESSION).endpoint
    record = {
        "session": SESSION,
        "endpoint": endpoint,
        "runs": ["t4-boundary5-200k-v1", "t4-boundary10-200k-v1"],
    }
    (ROOT / "analysis/DDQN_STRONG_BOUNDARY_DEPLOYMENT.json").write_text(
        json.dumps(record, indent=2)
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/ddqn-colab-collect.py",
            "--endpoint",
            endpoint,
            "--destination",
            "history/dodge/gymnasium/cnn-image-ddqn",
            "--remote-history",
            "content/t4-history",
            "--run-id",
            "t4-boundary5-200k-v1",
            "--run-id",
            "t4-boundary10-200k-v1",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
