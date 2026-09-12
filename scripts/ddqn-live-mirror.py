"""Read Colab artifacts into a separate dashboard cache; never control training."""

import json
import time
from pathlib import Path

import requests
from colab_cli.auth import AuthProvider
from colab_cli.common import state

ROOT = Path("history/dodge/gymnasium/live-cnn-image-ddqn")
CANONICAL = Path("history/dodge/gymnasium/cnn-image-ddqn")
LANES = {
    "dodge-opt-best-resume-v2": [
        "t4-best-resume500k-opt8-v1",
        "t4-boundary10-resume500k-opt8-v1",
        "t4-best-lane8-baseline-200k-s43-v1",
        "t4-best-lane1-opt-200k-s43-v1",
        "t4-best-lane1-baseline-200k-s43-v1",
    ],
    "dodge-opt-gate-v1": [
        "t4-best-opt8-200k-s43-v1",
        "t4-best-resume500k-opt8-v1",
    ],
    "dodge-opt-continuation-v1": ["t4-boundary10-resume500k-opt8-v1"],
    "dodge-strong-boundary-v1": ["t4-boundary5-200k-v1", "t4-boundary10-200k-v1"],
    "dodge-sync-1000-v1": [
        f"t4-boundarysync{interval}-200k-v1" for interval in (1000, 100, 10000)
    ],
    "dodge-sync-2500-v1": [
        f"t4-boundarysync{interval}-200k-v1" for interval in (2500, 5000)
    ],
    "dodge-best-boundary-v1": ["t4-nstep3s43boundary-200k-v1"],
    "dodge-reward-control-death-v1": ["t4-rgbcontrol-200k-v1", "t4-rgbdeath-200k-v1"],
    "dodge-reward-boundary-events-v1": [
        "t4-rgbboundary-200k-v1",
        "t4-rgbevents-200k-v1",
    ],
}


def remote_history(session):
    return "opt-history" if session.startswith("dodge-opt-") else "t4-history"


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    state.auth_provider = AuthProvider.OAUTH2
    while True:
        try:
            assignments = {a.endpoint: a for a in state.client.list_assignments()}
            for session, runs in LANES.items():
                try:
                    entry = state.store.get(session)
                except (KeyError, ValueError):
                    continue
                if entry is None:
                    continue
                assignment = assignments.get(entry.endpoint)
                if assignment is None:
                    continue
                proxy = assignment.runtime_proxy_info
                headers = {
                    "X-Colab-Runtime-Proxy-Token": proxy.token,
                    "X-Colab-Client-Agent": "colab-cli",
                }
                for run in runs:
                    if (CANONICAL / run / "report.json").exists():
                        continue
                    destination = ROOT / run
                    destination.mkdir(parents=True, exist_ok=True)
                    updated = False
                    for name in (
                        "manifest.json",
                        "config.json",
                        "metrics.jsonl",
                        "status.json",
                        "report.json",
                        "evaluation.json",
                    ):
                        response = requests.get(
                            f"{proxy.url.rstrip('/')}/files/content/{remote_history(session)}/cnn-image-ddqn/{run}/{name}",
                            headers=headers,
                            timeout=15,
                        )
                        if response.status_code == 404:
                            continue
                        response.raise_for_status()
                        data = response.content
                        if len(data) > 8 * 1024 * 1024:
                            raise ValueError("live artifact exceeds 8 MiB limit")
                        if name.endswith(".json"):
                            json.loads(data)
                        else:
                            # A live JSONL append may end with an incomplete line.
                            data = data[: data.rfind(b"\n") + 1]
                        atomic(destination / name, data)
                        updated = updated or name == "status.json"
                    if updated:
                        atomic(
                            destination / "live_mirror.json",
                            json.dumps(
                                {
                                    "session": session,
                                    "fetched_at": time.time(),
                                    "source": "read-only Colab artifact mirror",
                                }
                            ).encode(),
                        )
                        print("MIRRORED", run, flush=True)
        except Exception as error:
            print("MIRROR_RETRY", type(error).__name__, flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
