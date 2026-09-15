"""Ship a frozen LeWM and data to one T4 for a bounded decoder-only study."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from colab_mvp import ROOT, cli
from colab_normalization_audit import upload_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, choices=(256, 512, 2048), default=256)
    parser.add_argument("--resume-decoder", type=Path)
    args = parser.parse_args()
    if (args.steps > 256) != (args.resume_decoder is not None):
        parser.error("512/2048 require --resume-decoder; 256 must start fresh")
    if not args.run_id.replace("-", "").isalnum():
        raise ValueError("run id must be alphanumeric with hyphens")
    protocol = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "data_hash": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "world_model_updates": 0,
        "decoder_kind": "query",
        "decoder_total_steps": args.steps,
        "new_decoder_updates": args.steps - {256: 0, 512: 256, 2048: 512}[args.steps],
        "retained_steps": [args.steps],
        "resume_decoder_sha256": (
            hashlib.sha256(args.resume_decoder.read_bytes()).hexdigest()
            if args.resume_decoder
            else None
        ),
        "batch_size": 8,
        "output_size": 128,
        "initialization_seed": 904,
        "sampling_seed": 903,
    }
    job = (
        ROOT
        / "history/dodge/gymnasium/pixel-repr-ddqn-jobs"
        / f"{args.run_id}-{args.steps}"
    )
    job.mkdir(parents=True, exist_ok=False)
    (job / "decoder_protocol.json").write_text(json.dumps(protocol, indent=2))
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in [
            "src",
            "native/Cargo.toml",
            "native/Cargo.lock",
            "native/crates",
            "third_party",
            "variants/pixel-repr-ddqn",
            "tests/variant_pixel_repr_ddqn",
            "references/manifest.json",
            "references/le-wm/module.py",
        ]:
            output.add(
                ROOT / name,
                arcname=name,
                filter=lambda i: None if "__pycache__" in i.name else i,
            )
        output.add(args.checkpoint, arcname="checkpoint.pt")
        output.add(args.dataset, arcname="dataset")
        output.add(job / "decoder_protocol.json", arcname="decoder_protocol.json")
        if args.resume_decoder:
            output.add(args.resume_decoder, arcname="resume_decoder.pt")
    if archive.stat().st_size > 1024**3:
        raise ValueError("source archive exceeds1GiB")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (job / "source.sha256").write_text(digest + "\n")
    remote = job / "remote.py"
    remote.write_text(
        f"import os\nos.environ['LEWM_SOURCE_HASH']={digest!r}\n"
        f"os.environ['LEWM_RUN_ID']={args.run_id!r}\n"
        + Path(__file__)
        .with_name("colab_remote.py")
        .read_text()
        .replace("colab_worker.py", "colab_decoder_worker.py")
    )
    session = f"dodge-{args.run_id}-{args.steps}"
    cli("new", "--session", session, "--gpu", "T4", timeout=180)
    upload_archive(archive, session, job, remote_path="/content/lewm-source.tar.gz")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    names = [f"{args.run_id}-query{args.steps}"]
    for name in names:
        (history / name).mkdir(parents=True, exist_ok=False)
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
                "3000",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        while process.poll() is None:
            for name in names:
                for artifact in (
                    "manifest.json",
                    "status.json",
                    "metrics.jsonl",
                    "visualizations.json",
                ):
                    temporary = history / name / (".mirror-" + artifact)
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
                        timeout=60,
                    )
                    if result.returncode == 0 and temporary.exists():
                        temporary.replace(history / name / artifact)
            time.sleep(5)
        if (
            process.returncode
            or "LEWM_DECODER_STUDY_COMPLETE" not in (job / "remote.log").read_text()
        ):
            raise RuntimeError(
                f"decoder study failed; retained session {session}; "
                f"see {job / 'remote.log'}"
            )
    cli(
        "download",
        "/content/lewm-decoder-results.tar.gz",
        str(job / "results.tar.gz"),
        "--session",
        session,
    )
    with tarfile.open(job / "results.tar.gz") as downloaded:
        downloaded.extractall(job / "results", filter="data")
    for name in names:
        shutil.copytree(
            job / "results/history" / name, history / name, dirs_exist_ok=True
        )
    cli("stop", "--session", session)
    print(f"Decoder study artifacts: {job}", flush=True)


if __name__ == "__main__":
    main()
