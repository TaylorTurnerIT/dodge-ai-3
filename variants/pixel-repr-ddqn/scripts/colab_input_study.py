"""Launch one bounded T4 for the paired pixel-input study and retrieve evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from colab_large_probe import ROOT, _validate_run_id, cli, digest, upload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    dataset = args.dataset.resolve()
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    protocol = {
        "experiment": "lewm-input-encoding-v1",
        "run_id": args.run_id,
        "data_sha256": digest(dataset / "manifest.json"),
        "world_steps": 1024,
        "world_milestones": [512, 1024],
        "world_batch_size": 32,
        "world_init_seed": 42,
        "world_sampling_seed": 43,
        "world_stochastic_seed": 44,
        "decoder_milestones": [512, 2048, 8192],
        "decoder_init_seed": 904,
        "decoder_sampling_seed": 903,
        "loss_kind": "palette-ce",
        "representation": "cls",
        "arms": ["rgb", "palette"],
        "worker_timeout_seconds": 7200,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (job / "input_protocol.json").write_text(json.dumps(protocol, indent=2))
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in (
            "src",
            "native/Cargo.toml",
            "native/Cargo.lock",
            "native/crates",
            "third_party",
            "variants/pixel-repr-ddqn",
            "tests/variant_pixel_repr_ddqn",
            "references/manifest.json",
            "references/le-wm/module.py",
        ):
            output.add(
                ROOT / name,
                arcname=name,
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(dataset, arcname="dataset")
        output.add(job / "input_protocol.json", arcname="input_protocol.json")
    source_hash = digest(archive)
    (job / "source.sha256").write_text(source_hash + "\n")
    remote = job / "remote.py"
    remote.write_text(
        f"import os\nos.environ['LEWM_SOURCE_HASH']={source_hash!r}\n"
        f"os.environ['LEWM_RUN_ID']={args.run_id!r}\n"
        + Path(__file__)
        .with_name("colab_remote.py")
        .read_text()
        .replace("colab_worker.py", "colab_input_worker.py")
    )
    session = f"dodge-{args.run_id}"
    cli("new", "--session", session, "--gpu", "T4")
    (job / "session.json").write_text(json.dumps({"session": session}))
    upload(archive, session, job)
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    names = [
        f"{args.run_id}-{arm}{suffix}"
        for arm in ("rgb", "palette")
        for suffix in ("", "-cls")
    ]
    for name in names:
        (history / name).mkdir(exist_ok=False)
    with (job / "remote.log").open("w") as log:
        process = subprocess.Popen(
            [
                "colab",
                "--auth",
                "adc",
                "exec",
                "--session",
                session,
                "--file",
                str(remote),
                "--timeout",
                "7200",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            for name in names:
                for artifact in (
                    "manifest.json",
                    "config.json",
                    "status.json",
                    "metrics.jsonl",
                    "visualizations.json",
                ):
                    target = history / name / artifact
                    temporary = target.with_name(".mirror-" + artifact)
                    try:
                        result = subprocess.run(
                            [
                                "colab",
                                "--auth",
                                "adc",
                                "download",
                                f"/content/lewm-work/history/{name}/{artifact}",
                                str(temporary),
                                "--session",
                                session,
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=20,
                        )
                        if result.returncode == 0 and temporary.is_file():
                            temporary.replace(target)
                    except subprocess.TimeoutExpired:
                        pass
            time.sleep(15)
        if process.returncode:
            raise RuntimeError("Input worker failed; session retained for recovery")
    cli(
        "download",
        "/content/lewm-input-results.sha256",
        str(job / "results.sha256"),
        "--session",
        session,
    )
    cli(
        "download",
        "/content/lewm-input-results.tar.gz",
        str(job / "results.tar.gz"),
        "--session",
        session,
    )
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("Retrieved archive checksum mismatch; session retained")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    shutil.copytree(job / "results/history", history, dirs_exist_ok=True)
    print(
        "INPUT_RESULTS_RETRIEVED: verify checkpoints before releasing session",
        flush=True,
    )


if __name__ == "__main__":
    main()
