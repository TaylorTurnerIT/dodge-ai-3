"""Ship frozen MVP source to a fresh Colab T4, run phases, retrieve artifacts.

Run from this worktree: python3 variants/pixel-repr-ddqn/scripts/colab_mvp.py
An interrupted/failed session is retained for diagnosis; successful runs release it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def cli(*args: str, timeout: int = 3600) -> None:
    subprocess.run(
        ["colab", "--auth", "adc", *args], check=True, cwd=ROOT, timeout=timeout
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="lewm-t4-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Scenario TOML to freeze into the T4 source archive",
    )
    parser.add_argument(
        "--practice-dataset",
        type=Path,
        help="Frozen imported corpus for practice-overfit-v1",
    )
    args = parser.parse_args()
    if args.practice_dataset is not None and args.scenario is not None:
        raise ValueError("practice dataset and scenario collection are exclusive")
    practice = None
    if args.practice_dataset is not None:
        corpus = args.practice_dataset.resolve()
        raw = (corpus / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        if not manifest.get("practice_import") or not (corpus / "READY").is_file():
            raise ValueError("expected a published practice corpus")
        practice = {
            "experiment": "practice-overfit-v1",
            "model_updates": 512,
            "decoder_updates": 256,
            "batch_size": 8,
            "seed": 42,
            "data_hash": hashlib.sha256(raw).hexdigest(),
        }
    if args.scenario is not None:
        # Parse before creating a job or allocating a GPU. Native/schema validation
        # runs in the frozen worker before collection.
        tomllib.loads(args.scenario.read_text())
    if not args.run_id.replace("-", "").isalnum():
        raise ValueError("run id must be alphanumeric with hyphens")
    session = "dodge-" + args.run_id
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    if practice is not None:
        (job / "practice_protocol.json").write_text(json.dumps(practice, indent=2))
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for relative in [
            "src",
            "native/Cargo.toml",
            "native/Cargo.lock",
            "native/crates",
            "variants/pixel-repr-ddqn",
            "variants/cnn-image-ddqn/config.toml",
            "tests/variant_pixel_repr_ddqn",
            "references/manifest.json",
            "references/le-wm/module.py",
            "third_party",
        ]:
            output.add(
                ROOT / relative,
                arcname=relative,
                filter=lambda info: None if "__pycache__" in info.name else info,
            )
        if practice is not None:
            output.add(corpus, arcname="dataset")
            output.add(job / "practice_protocol.json", arcname="practice_protocol.json")
        if args.scenario is not None:
            output.add(args.scenario, arcname="scenario.toml")
    if archive.stat().st_size > 2 * 1024**3:
        raise ValueError("source archive exceeds 2GiB protocol cap")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (job / "job.json").write_text(
        json.dumps(
            {
                "session": session,
                "source_sha256": digest,
                "gpu_required": "T4",
                "scenario_source": str(args.scenario)
                if args.scenario is not None
                else None,
                "run_id": args.run_id,
                "practice_protocol": practice,
            },
            indent=2,
        )
    )
    remote = job / "remote.py"
    remote.write_text(
        "import os\n"
        + f"os.environ['LEWM_SOURCE_HASH'] = {digest!r}\n"
        + f"os.environ['LEWM_RUN_ID'] = {args.run_id!r}\n"
        + (Path(__file__).with_name("colab_remote.py")).read_text()
    )

    cli("new", "--session", session, "--gpu", "T4", timeout=180)
    cli("upload", str(archive), "/content/lewm-source.tar.gz", "--session", session)
    destination = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    command = [
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
    ]
    with (job / "remote.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        while process.poll() is None:
            for name in [
                "manifest.json",
                "status.json",
                "metrics.jsonl",
                "visualization.json",
                "visualizations.json",
                "report.json",
            ]:
                temporary = destination / (".mirror-" + name)
                result = subprocess.run(
                    [
                        "colab",
                        "--auth",
                        "adc",
                        "download",
                        f"/content/lewm-work/history/{args.run_id}/{name}",
                        str(temporary),
                        "--session",
                        session,
                    ],
                    cwd=ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
                if result.returncode == 0 and temporary.exists():
                    temporary.replace(destination / name)
            time.sleep(5)
        if (
            process.returncode
            or "LEWM_MVP_COMPLETE" not in (job / "remote.log").read_text()
        ):
            raise RuntimeError(
                f"Remote run failed; retained session {session}; "
                f"see {job / 'remote.log'}"
            )

    results = job / "results.tar.gz"
    cli("download", "/content/lewm-results.tar.gz", str(results), "--session", session)
    with tarfile.open(results) as downloaded:
        downloaded.extractall(job / "results", filter="data")
    import shutil

    shutil.copytree(
        job / "results/history" / args.run_id, destination, dirs_exist_ok=True
    )
    cli("stop", "--session", session)
    print(f"Artifacts: {destination}", flush=True)


if __name__ == "__main__":
    main()
