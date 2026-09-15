"""Run the frozen local spatial readout diagnosis on one T4 and retrieve it."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path

from colab_large_probe import ROOT, _validate_run_id, cli, digest, upload

EXPERIMENT = "lewm-spatial-readout-v1"
ARMS = ("cls", "patch", "pixels")
MILESTONES = (512, 2048, 8192)
RUN_ARTIFACTS = (
    "manifest.json",
    "config.json",
    "status.json",
    "metrics.jsonl",
    "report.json",
    "visualizations.json",
)
TRAIN_FRAMES = 16_384
VALIDATION_FRAMES = 2_048
BANK_PROVENANCE = "bank-provenance"


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _protocol(
    *,
    checkpoint_sha256: str,
    data_sha256: str,
    run_id: str,
    source_commit: str,
) -> dict[str, object]:
    """Return the immutable worker contract for the three matched arms."""

    return {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "checkpoint_name": "world.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "data_sha256": data_sha256,
        "world_updates": 0,
        "world_model_updates": 0,
        "milestones": list(MILESTONES),
        "batch_size": 32,
        "decoder_init_seed": 904,
        "decoder_sampling_seed": 903,
        "loss_kind": "palette-ce",
        "decoder_kind": "local-patch",
        "arms": list(ARMS),
        "worker_timeout_seconds": 7200,
        "source_commit": source_commit,
    }


def _validate_run_artifacts(
    run: Path, *, arm: str, protocol: Mapping[str, object]
) -> None:
    for artifact in RUN_ARTIFACTS:
        if not (run / artifact).is_file():
            raise RuntimeError(f"missing {run.name}/{artifact}; session retained")
    checkpoint = run / "checkpoint.pt"
    if not checkpoint.is_file() or digest(checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError(f"{run.name} world checkpoint mismatch; session retained")
    manifest = _read_json(run / "manifest.json", f"{run.name}/manifest.json")
    config = _read_json(run / "config.json", f"{run.name}/config.json")
    report = _read_json(run / "report.json", f"{run.name}/report.json")
    status = _read_json(run / "status.json", f"{run.name}/status.json")
    for label, payload in (
        ("manifest", manifest),
        ("config", config),
        ("report", report),
    ):
        if payload.get("experiment") != EXPERIMENT:
            raise RuntimeError(f"{run.name} {label} experiment mismatch")
        if payload.get("representation") != arm:
            raise RuntimeError(f"{run.name} {label} arm mismatch")
        if payload.get("world_model_sha256") != protocol["checkpoint_sha256"]:
            raise RuntimeError(f"{run.name} {label} checkpoint provenance mismatch")
        if payload.get("data_sha256") != protocol["data_sha256"]:
            raise RuntimeError(f"{run.name} {label} dataset provenance mismatch")
        if payload.get("world_model_updates") != 0:
            raise RuntimeError(f"{run.name} {label} changed the world model")
        if payload.get("loss_kind") != protocol["loss_kind"]:
            raise RuntimeError(f"{run.name} {label} loss provenance mismatch")
        if payload.get("milestones") != protocol["milestones"]:
            raise RuntimeError(f"{run.name} {label} milestone provenance mismatch")
        if payload.get("palette_source_split") != "train":
            raise RuntimeError(f"{run.name} {label} palette is not train-only")
    if status.get("state") != "completed" or status.get("step") != MILESTONES[-1]:
        raise RuntimeError(f"{run.name} did not complete the declared schedule")
    if not (run / "metrics.jsonl").read_text().strip():
        raise RuntimeError(f"{run.name} metrics are empty")
    for line in (run / "metrics.jsonl").read_text().splitlines():
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{run.name} contains invalid metrics JSON") from error
    for step in protocol["milestones"]:
        decoder = run / f"decoder-{step}.pt"
        evaluation_path = run / f"evaluation-{step}.json"
        visualizations = run / f"visualizations-{step}.json"
        if not decoder.is_file() or decoder.stat().st_size == 0:
            raise RuntimeError(f"missing {run.name}/decoder-{step}.pt")
        if not evaluation_path.is_file() or not visualizations.is_file():
            raise RuntimeError(f"missing evaluation artifacts for {run.name}/{step}")
        evaluation = _read_json(evaluation_path, f"{run.name}/evaluation-{step}.json")
        if (
            evaluation.get("step") != step
            or evaluation.get("representation") != arm
            or evaluation.get("world_model_sha256") != protocol["checkpoint_sha256"]
            or evaluation.get("data_sha256") != protocol["data_sha256"]
            or evaluation.get("world_model_updates") != 0
            or evaluation.get("palette_source_split") != "train"
            or evaluation.get("splits", {}).get("train", {}).get("frame_count")
            != TRAIN_FRAMES
            or evaluation.get("splits", {})
            .get("validation", {})
            .get("frame_count")
            != VALIDATION_FRAMES
        ):
            raise RuntimeError(
                f"incomplete provenance in {run.name}/evaluation-{step}.json"
            )


def _validate_bank_provenance(
    history: Path, run_id: str, protocol: Mapping[str, object]
) -> None:
    root = history / f"{run_id}-{BANK_PROVENANCE}"
    world = root / "world.pt"
    if not world.is_file() or digest(world) != protocol["checkpoint_sha256"]:
        raise RuntimeError(
            "retrieved world checkpoint bundle is invalid; session retained"
        )
    received_protocol = _read_json(root / "spatial_protocol.json", "spatial protocol")
    if received_protocol != dict(protocol):
        raise RuntimeError("retrieved spatial protocol differs; session retained")
    required = {
        "standard": {
            split: ("metadata.json", "index.json", "READY")
            for split in ("train", "validation")
        },
        "spatial": {
            split: ("metadata.json", "READY")
            for split in ("train", "validation")
        },
    }
    for kind, splits in required.items():
        for split, names in splits.items():
            for name in names:
                path = root / kind / split / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(
                        f"missing bank provenance {kind}/{split}/{name}; "
                        "session retained"
                    )
    for relative in (
        "spatial/validation/patches.npy",
        "standard/validation/cls.npy",
    ):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing bank audit array {relative}; session retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        "--world-checkpoint",
        dest="checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if not (dataset / "manifest.json").is_file():
        parser.error(f"dataset manifest does not exist: {dataset / 'manifest.json'}")
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    protocol = _protocol(
        checkpoint_sha256=digest(checkpoint),
        data_sha256=digest(dataset / "manifest.json"),
        run_id=args.run_id,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    (job / "spatial_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
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
            "references/paper/lewm-v3.md",
        ):
            output.add(
                ROOT / name,
                arcname=name,
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(checkpoint, arcname="world.pt")
        output.add(dataset, arcname="dataset")
        output.add(job / "spatial_protocol.json", arcname="spatial_protocol.json")
    if archive.stat().st_size > 1024**3:
        raise ValueError("source archive exceeds 1 GiB")
    source_hash = digest(archive)
    (job / "source.sha256").write_text(source_hash + "\n")
    remote = job / "remote.py"
    remote.write_text(
        f"import os\nos.environ['LEWM_SOURCE_HASH']={source_hash!r}\n"
        f"os.environ['LEWM_RUN_ID']={args.run_id!r}\n"
        + Path(__file__)
        .with_name("colab_remote.py")
        .read_text()
        .replace("colab_worker.py", "colab_spatial_worker.py")
    )
    session = f"dodge-{args.run_id}"
    cli("new", "--session", session, "--gpu", "T4")
    (job / "session.json").write_text(json.dumps({"session": session}) + "\n")
    upload(archive, session, job)
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    names = [f"{args.run_id}-{arm}" for arm in ARMS]
    for name in names:
        (history / name).mkdir(parents=True, exist_ok=False)
    comparison = history / f"{args.run_id}-spatial-comparison.json"
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
                for artifact in RUN_ARTIFACTS:
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
                            timeout=30,
                        )
                        if result.returncode == 0 and temporary.is_file():
                            temporary.replace(target)
                    except subprocess.TimeoutExpired:
                        pass
            temporary = comparison.with_name("." + comparison.name)
            try:
                result = subprocess.run(
                    [
                        "colab",
                        "--auth",
                        "adc",
                        "download",
                        f"/content/lewm-work/history/{comparison.name}",
                        str(temporary),
                        "--session",
                        session,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                if result.returncode == 0 and temporary.is_file():
                    temporary.replace(comparison)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(15)
    remote_log = (job / "remote.log").read_text()
    if process.returncode or "SPATIAL_COMPLETE" not in remote_log:
        raise RuntimeError(f"Spatial study failed; session retained: {session}")
    cli(
        "download",
        "/content/lewm-spatial-results.sha256",
        str(job / "results.sha256"),
        "--session",
        session,
    )
    cli(
        "download",
        "/content/lewm-spatial-results.tar.gz",
        str(job / "results.tar.gz"),
        "--session",
        session,
        timeout=300,
    )
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("retrieved archive checksum mismatch; session retained")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    shutil.copytree(job / "results/history", history, dirs_exist_ok=True)
    environment = _read_json(
        history / f"{args.run_id}-environment.json",
        f"{args.run_id}-environment.json",
    )
    if (
        environment.get("source_sha256") != source_hash
        or environment.get("protocol") != protocol
    ):
        raise RuntimeError("retrieved provenance mismatch; session retained")
    for arm, name in zip(ARMS, names, strict=True):
        _validate_run_artifacts(history / name, arm=arm, protocol=protocol)
    comparison_payload = _read_json(comparison, comparison.name)
    gallery = history / f"{args.run_id}-comparison.html"
    if not gallery.is_file() or gallery.stat().st_size == 0:
        raise RuntimeError("spatial comparison gallery is missing; session retained")
    if (
        comparison_payload.get("experiment") != EXPERIMENT
        or comparison_payload.get("run_id") != args.run_id
        or comparison_payload.get("world_model_sha256") != protocol["checkpoint_sha256"]
        or comparison_payload.get("data_sha256") != protocol["data_sha256"]
        or comparison_payload.get("world_model_updates") != 0
        or comparison_payload.get("milestones") != protocol["milestones"]
        or comparison_payload.get("arms") != list(ARMS)
    ):
        raise RuntimeError("spatial comparison provenance mismatch; session retained")
    _validate_bank_provenance(history, args.run_id, protocol)
    print(
        f"Spatial artifacts retrieved; verify checkpoints before releasing T4: {job}",
        flush=True,
    )


if __name__ == "__main__":
    main()
