"""Collect this seed campaign with refreshed Colab proxy credentials.

Run with the Colab CLI's Python interpreter. Only releases the named assignment
after both expected archives have been downloaded and safely extracted.
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
import time
from pathlib import Path

import requests
from colab_cli.auth import AuthProvider
from colab_cli.common import state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--remote-history", default="content/seed-history")
    parser.add_argument("--collect-failure", action="store_true")
    args = parser.parse_args()
    state.auth_provider = AuthProvider.ADC
    expected = set(
        args.run_id or [f"seedpool-{pool}-500k-s42-v1" for pool in (700, 5000)]
    )
    if any(
        not name
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for c in name
        )
        or name in (".", "..")
        for name in expected
    ):
        parser.error("run IDs must be safe path components")
    collected = set()
    deadline = time.monotonic() + 10800
    args.destination.mkdir(parents=True, exist_ok=True)
    while time.monotonic() < deadline:
        try:
            assignments = state.client.list_assignments()
            assignment = next(a for a in assignments if a.endpoint == args.endpoint)
            proxy = assignment.runtime_proxy_info
            headers = {
                "X-Colab-Runtime-Proxy-Token": proxy.token,
                "X-Colab-Client-Agent": "colab-cli",
            }
            base = proxy.url.rstrip("/")
            if args.collect_failure:
                failure_remote = (
                    f"{args.remote_history.strip('/')}/campaign-failure.tar.gz"
                )
                failure = requests.get(
                    f"{base}/api/contents/{failure_remote}",
                    headers=headers,
                    params={"content": 0},
                    timeout=30,
                )
                if failure.status_code != 404:
                    failure.raise_for_status()
                    rejected = args.destination.parent / "rejected-campaigns"
                    rejected.mkdir(parents=True, exist_ok=True)
                    final = rejected / f"{sorted(expected)[0]}-failure.tar.gz"
                    if final.exists():
                        raise FileExistsError(final)
                    partial = final.with_suffix(".partial")
                    with requests.get(
                        f"{base}/files/{failure_remote}",
                        headers=headers,
                        stream=True,
                        timeout=120,
                    ) as download:
                        download.raise_for_status()
                        with partial.open("wb") as stream:
                            for chunk in download.iter_content(1024 * 1024):
                                stream.write(chunk)
                    with tarfile.open(partial) as archive:
                        archive.getmembers()
                    partial.rename(final)
                    state.client.unassign(args.endpoint)
                    print(
                        "COLLECTED_FAILED_CAMPAIGN_AND_RELEASED_ASSIGNMENT",
                        final,
                        flush=True,
                    )
                    return
            for run_id in sorted(expected - collected):
                remote = f"{args.remote_history.strip('/')}/{run_id}.tar.gz"
                response = requests.get(
                    f"{base}/api/contents/{remote}",
                    headers=headers,
                    params={"content": 0},
                    timeout=30,
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                # Archives are created only after training and snapshot eval.
                # Tar validation catches a file observed before gzip is closed;
                # retry without publishing a partially extracted run.
                archive_path = args.destination / f".{run_id}.tar.gz"
                with requests.get(
                    f"{base}/files/{remote}", headers=headers, stream=True, timeout=120
                ) as download:
                    download.raise_for_status()
                    with archive_path.open("wb") as stream:
                        for chunk in download.iter_content(1024 * 1024):
                            stream.write(chunk)
                with tarfile.open(archive_path) as archive:
                    members = archive.getmembers()
                    if not all(
                        m.name == run_id or m.name.startswith(run_id + "/")
                        for m in members
                    ):
                        raise ValueError("unexpected archive root")
                    # Do not overwrite historical run artifacts.
                    if (args.destination / run_id).exists():
                        raise FileExistsError(args.destination / run_id)
                    with tempfile.TemporaryDirectory(
                        prefix=".ddqn-collect-", dir=args.destination
                    ) as temporary:
                        archive.extractall(temporary, filter="data")
                        (Path(temporary) / run_id).rename(args.destination / run_id)
                collected.add(run_id)
                print(f"COLLECTED {run_id}", flush=True)
            if collected == expected:
                state.client.unassign(args.endpoint)
                print("COLLECTED_ALL_AND_RELEASED_ASSIGNMENT", flush=True)
                return
            print(f"WAITING {sorted(expected - collected)}", flush=True)
        except (
            requests.RequestException,
            StopIteration,
            tarfile.TarError,
            EOFError,
        ) as error:
            print(f"RETRY {type(error).__name__}", flush=True)
        time.sleep(30)
    raise TimeoutError("Campaign not collected within 3 hours; inspect assignment")


if __name__ == "__main__":
    main()
