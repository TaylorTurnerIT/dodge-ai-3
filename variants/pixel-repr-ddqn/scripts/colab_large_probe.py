"""Run the frozen large-corpus CLS/projected reconstruction screen on one T4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODES = ("cls", "projected")
MILESTONES = (512, 2048, 8192)
LOSS_KINDS = ("mse", "balanced-bright", "palette-ce", "palette-bce")
PALETTE_LOSS_KINDS = ("palette-ce", "palette-bce")
PALETTE_SUFFIXES = {"palette-ce": "ce", "palette-bce": "bce"}
RUN_ARTIFACTS = (
    "manifest.json",
    "config.json",
    "status.json",
    "metrics.jsonl",
    "report.json",
    "visualizations.json",
)
TRAIN_FRAMES = 16384
VALIDATION_FRAMES = 2048


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cli(*args: str, timeout: int = 180):
    result = subprocess.run(
        ["colab", "--auth", "adc", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = re.sub(
        r"colab-runtime-proxy-token=[^\s)]+",
        "colab-runtime-proxy-token=REDACTED",
        result.stdout + result.stderr,
    )
    if output.strip():
        print(output.strip(), flush=True)
    if result.returncode:
        raise RuntimeError(f"Colab command failed: {args[0]}")
    return result


def upload(archive: Path, session: str, job: Path):
    count = 0
    with archive.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            part = job / f"upload-{count:03d}"
            part.write_bytes(block)
            cli(
                "upload",
                str(part),
                f"/content/large-probe.part-{count:03d}",
                "--session",
                session,
            )
            part.unlink()
            count += 1
    assembly = job / "assemble.py"
    assembly.write_text(
        "from pathlib import Path\nimport hashlib\n"
        "target=Path('/content/lewm-source.tar.gz')\n"
        "with target.open('wb') as output:\n"
        f" for i in range({count}):\n"
        "  part=Path(f'/content/large-probe.part-{i:03d}')\n"
        "  output.write(part.read_bytes())\n  part.unlink()\n"
        f"assert hashlib.sha256(target.read_bytes()).hexdigest()=={digest(archive)!r}\n"
        "print('SOURCE_ARCHIVE_VERIFIED')\n"
    )
    cli("exec", "--session", session, "--file", str(assembly), "--timeout", "120")


def _validate_run_id(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,70}", value):
        raise ValueError(f"invalid {label}")


def _loss_kinds(loss_kind: str, palette_comparison: bool) -> tuple[str, ...]:
    """Return the ordered condition list for one worker invocation."""

    if loss_kind not in LOSS_KINDS:
        raise ValueError(f"unsupported loss kind: {loss_kind}")
    if palette_comparison:
        if loss_kind != "mse":
            raise ValueError(
                "--palette-comparison cannot be combined with a single "
                "--loss-kind"
            )
        return PALETTE_LOSS_KINDS
    return (loss_kind,)


def _condition_run_id(run_id: str, loss_kind: str, palette_comparison: bool) -> str:
    if not palette_comparison:
        return run_id
    try:
        suffix = PALETTE_SUFFIXES[loss_kind]
    except KeyError as error:
        raise ValueError(f"palette comparison does not support {loss_kind}") from error
    return f"{run_id}-{suffix}"


def _run_names(
    run_id: str, *, loss_kinds: tuple[str, ...], palette_comparison: bool
) -> tuple[str, ...]:
    """List all per-head artifact directories in worker execution order."""

    return tuple(
        f"{_condition_run_id(run_id, loss_kind, palette_comparison)}-{mode}"
        for loss_kind in loss_kinds
        for mode in MODES
    )


def _protocol(
    *,
    checkpoint_sha256: str,
    data_hash: str,
    run_id: str,
    loss_kind: str,
    palette_comparison: bool,
    baseline_run_id: str | None,
    baseline_files: dict[str, dict[str, str]],
) -> dict[str, object]:
    loss_kinds = _loss_kinds(loss_kind, palette_comparison)
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "data_hash": data_hash,
        "milestones": list(MILESTONES),
        "batch_size": 32,
        "frames_per_episode": 4,
        "initialization_seed": 904,
        "sampling_seed": 903,
        "loss_kind": loss_kind if not palette_comparison else None,
        "loss_kinds": list(loss_kinds),
        "palette_comparison": palette_comparison,
        "palette_run_ids": {
            condition: _condition_run_id(run_id, condition, palette_comparison)
            for condition in loss_kinds
        }
        if palette_comparison
        else {},
        "decoder_input_split": "train",
        "train_only_input": True,
        "world_model_updates": 0,
        "output_size": 128,
        "representations": list(MODES),
        "baseline_run_id": baseline_run_id,
        "baseline_files": baseline_files,
    }


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _palette_fields(
    payload: Mapping[str, object],
) -> tuple[list[object] | None, str | None]:
    """Read the exact palette provenance emitted by the probe core."""

    palette_rgb = payload.get("palette_rgb")
    if not isinstance(palette_rgb, list):
        palette_rgb = None
    palette_hash = payload.get("palette_sha256")
    if not isinstance(palette_hash, str) or not palette_hash:
        palette_hash = None
    return palette_rgb, palette_hash


def _validate_palette_provenance(
    payload: Mapping[str, object], label: str
) -> tuple[list[object], str]:
    palette_rgb, palette_hash = _palette_fields(payload)
    if palette_rgb is None or palette_hash is None:
        raise RuntimeError(f"{label} palette provenance is incomplete")
    if not 1 <= len(palette_rgb) <= 256:
        raise RuntimeError(f"{label} palette size is invalid")
    packed = bytearray()
    previous: tuple[int, int, int] | None = None
    for index, color in enumerate(palette_rgb):
        if not isinstance(color, list) or len(color) != 3:
            raise RuntimeError(f"{label} palette color {index} is invalid")
        if any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in color
        ):
            raise RuntimeError(f"{label} palette color {index} is invalid")
        normalized = (color[0], color[1], color[2])
        if previous is not None and normalized <= previous:
            raise RuntimeError(f"{label} palette is not strictly sorted")
        packed.extend(normalized)
        previous = normalized
    if hashlib.sha256(bytes(packed)).hexdigest() != palette_hash:
        raise RuntimeError(f"{label} palette hash does not match RGB entries")
    return palette_rgb, palette_hash


def _has_train_only_input(payload: Mapping[str, object]) -> bool:
    return payload.get("palette_source_split") == "train"


def _validate_run_artifacts(
    run: Path,
    *,
    expected_loss_kind: str,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Validate one complete two-head condition before releasing the T4."""

    for artifact in RUN_ARTIFACTS:
        if not (run / artifact).is_file():
            raise RuntimeError(f"missing {run.name}/{artifact}; session retained")
    checkpoint = run / "checkpoint.pt"
    if not checkpoint.is_file() or digest(checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError(f"{run.name} checkpoint artifact mismatch; session retained")
    manifest = _read_json(run / "manifest.json", f"{run.name}/manifest.json")
    config = _read_json(run / "config.json", f"{run.name}/config.json")
    report = _read_json(run / "report.json", f"{run.name}/report.json")
    status = _read_json(run / "status.json", f"{run.name}/status.json")
    for label, payload in (
        ("manifest", manifest),
        ("config", config),
        ("report", report),
    ):
        if payload.get("loss_kind", "mse") != expected_loss_kind:
            raise RuntimeError(f"{run.name} {label} loss provenance mismatch")
        if payload.get("world_model_sha256") != protocol["checkpoint_sha256"]:
            raise RuntimeError(f"{run.name} {label} checkpoint provenance mismatch")
        if payload.get("data_sha256", payload.get("data_hash")) != protocol[
            "data_hash"
        ]:
            raise RuntimeError(f"{run.name} {label} dataset provenance mismatch")
    palette_values: list[tuple[list[object] | None, str | None]] | None = None
    if expected_loss_kind in PALETTE_LOSS_KINDS:
        palette_values = [
            _validate_palette_provenance(payload, f"{run.name}/{label}")
            for label, payload in (
                ("manifest", manifest),
                ("config", config),
                ("report", report),
            )
        ]
        if any(value != palette_values[0] for value in palette_values[1:]):
            raise RuntimeError(f"{run.name} palette provenance differs")
        if any(
            not _has_train_only_input(payload)
            for payload in (manifest, config, report)
        ):
            raise RuntimeError(f"{run.name} palette source is not train-only")
    if report.get("milestones") != protocol["milestones"]:
        raise RuntimeError(f"{run.name} milestone provenance mismatch")
    if status.get("state") != "completed" or status.get("step") != protocol[
        "milestones"
    ][-1]:
        raise RuntimeError(f"{run.name} did not complete the declared schedule")
    if not run.joinpath("metrics.jsonl").read_text().strip():
        raise RuntimeError(f"{run.name} metrics are empty")
    for line in run.joinpath("metrics.jsonl").read_text().splitlines():
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{run.name} contains invalid metrics JSON") from error
    for step in protocol["milestones"]:
        decoder = run / f"decoder-{step}.pt"
        evaluation_path = run / f"evaluation-{step}.json"
        visuals = run / f"visualizations-{step}.json"
        if not decoder.is_file() or decoder.stat().st_size == 0:
            raise RuntimeError(f"missing {run.name}/decoder-{step}.pt")
        if not visuals.is_file():
            raise RuntimeError(f"missing {run.name}/visualizations-{step}.json")
        evaluation = _read_json(evaluation_path, f"{run.name}/evaluation-{step}.json")
        if (
            evaluation.get("step") != step
            or evaluation.get("representation") != run.name.rsplit("-", 1)[-1]
            or evaluation.get("loss_kind", "mse") != expected_loss_kind
            or evaluation.get("world_model_sha256") != protocol["checkpoint_sha256"]
            or evaluation.get("data_sha256") != protocol["data_hash"]
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
        if expected_loss_kind in PALETTE_LOSS_KINDS:
            assert palette_values is not None
            if _validate_palette_provenance(
                evaluation, f"{run.name}/evaluation-{step}.json"
            ) != palette_values[0] or not _has_train_only_input(evaluation):
                raise RuntimeError(
                    f"{run.name}/evaluation-{step}.json palette provenance mismatch"
                )
    return {"manifest": manifest, "config": config, "report": report}


def _validate_palette_comparison(
    history: Path,
    run_id: str,
    protocol: Mapping[str, object],
    *,
    source_hash: str | None = None,
) -> dict[str, object]:
    path = history / f"{run_id}-palette-comparison.json"
    payload = _read_json(path, path.name)
    if payload.get("loss_kinds") != list(PALETTE_LOSS_KINDS):
        raise RuntimeError(
            "palette comparison loss ordering mismatch; session retained"
        )
    if payload.get("checkpoint_sha256") != protocol["checkpoint_sha256"]:
        raise RuntimeError("palette comparison checkpoint mismatch; session retained")
    if payload.get("data_sha256", payload.get("data_hash")) != protocol[
        "data_hash"
    ]:
        raise RuntimeError("palette comparison dataset mismatch; session retained")
    if source_hash is not None and payload.get("source_sha256") != source_hash:
        raise RuntimeError("palette comparison source mismatch; session retained")
    if payload.get("palette_source_split") != "train":
        raise RuntimeError("palette comparison is not train-only; session retained")
    palette_rgb, palette_hash = _validate_palette_provenance(payload, path.name)
    conditions = payload.get("conditions", payload.get("runs"))
    if not isinstance(conditions, Mapping):
        raise RuntimeError(
            "palette comparison conditions are missing; session retained"
        )
    for loss_kind in PALETTE_LOSS_KINDS:
        condition = conditions.get(loss_kind)
        if not isinstance(condition, Mapping):
            raise RuntimeError(
                f"palette {loss_kind} evidence is missing; session retained"
            )
        if condition.get("run_id") != f"{run_id}-{PALETTE_SUFFIXES[loss_kind]}":
            raise RuntimeError(
                f"palette {loss_kind} run prefix mismatch; session retained"
            )
        condition_rgb, condition_hash = _validate_palette_provenance(
            condition, f"palette {loss_kind} comparison"
        )
        if not _has_train_only_input(condition):
            raise RuntimeError(
                "palette condition source is not train-only; session retained"
            )
        if (condition_rgb, condition_hash) != (palette_rgb, palette_hash):
            raise RuntimeError("CE/BCE palette identity/hash differs; session retained")
    return payload


def _baseline_bundle(
    baseline_root: Path, baseline_run_id: str
) -> dict[str, dict[str, object]]:
    bundle: dict[str, dict[str, object]] = {}
    for mode in MODES:
        run = baseline_root / f"{baseline_run_id}-{mode}"
        decoder = run / "decoder-8192.pt"
        evaluation = run / "evaluation-8192.json"
        if not decoder.is_file() or not evaluation.is_file():
            raise FileNotFoundError(
                f"baseline {mode} must contain decoder-8192.pt and "
                "evaluation-8192.json"
            )
        try:
            payload = json.loads(evaluation.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"baseline {mode} evaluation is invalid") from error
        if payload.get("step") != 8192 or payload.get("representation") != mode:
            raise ValueError(f"baseline {mode} evaluation is not the 8192 artifact")
        validation = payload.get("splits", {}).get("validation", {})
        if validation.get("frame_count") != 2048:
            raise ValueError(f"baseline {mode} evaluation has incomplete validation")
        bundle[mode] = {
            "decoder": decoder,
            "evaluation": evaluation,
            "decoder-8192.pt": digest(decoder),
            "evaluation-8192.json": digest(evaluation),
        }
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--loss-kind", choices=LOSS_KINDS, default="mse"
    )
    parser.add_argument(
        "--palette-comparison",
        action="store_true",
        help="run palette-ce then palette-bce in one frozen T4 session",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="history root containing the retained plain-MSE baseline runs",
    )
    parser.add_argument(
        "--baseline-run-id",
        help="run prefix for the retained baseline, including both mode suffixes",
    )
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    if (args.baseline_root is None) != (args.baseline_run_id is None):
        parser.error("--baseline-root and --baseline-run-id must be supplied together")
    if args.baseline_run_id is not None:
        _validate_run_id(args.baseline_run_id, "baseline run ID")
    try:
        loss_kinds = _loss_kinds(args.loss_kind, args.palette_comparison)
    except ValueError as error:
        parser.error(str(error))
    if args.palette_comparison and args.baseline_root is not None:
        parser.error("palette comparisons do not run baseline rescoring")
    if (
        not args.palette_comparison
        and args.baseline_root is not None
        and args.loss_kind != "balanced-bright"
    ):
        parser.error("baseline artifacts are only used by balanced-bright runs")
    baseline_bundle = None
    if args.baseline_root is not None:
        baseline_bundle = _baseline_bundle(
            args.baseline_root.expanduser().resolve(),
            args.baseline_run_id,
        )
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    baseline_files = (
        {
            mode: {
                "decoder-8192.pt": values["decoder-8192.pt"],
                "evaluation-8192.json": values["evaluation-8192.json"],
            }
            for mode, values in baseline_bundle.items()
        }
        if baseline_bundle is not None
        else {}
    )
    protocol = _protocol(
        checkpoint_sha256=digest(args.checkpoint),
        data_hash=digest(args.dataset / "manifest.json"),
        run_id=args.run_id,
        loss_kind=args.loss_kind,
        palette_comparison=args.palette_comparison,
        baseline_run_id=args.baseline_run_id,
        baseline_files=baseline_files,
    )
    (job / "large_probe_protocol.json").write_text(json.dumps(protocol, indent=2))
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
        output.add(args.checkpoint, arcname="checkpoint.pt")
        output.add(args.dataset, arcname="dataset")
        output.add(
            job / "large_probe_protocol.json", arcname="large_probe_protocol.json"
        )
        if baseline_bundle is not None:
            for mode, values in baseline_bundle.items():
                output.add(
                    values["decoder"],
                    arcname=f"baselines/{mode}/decoder-8192.pt",
                )
                output.add(
                    values["evaluation"],
                    arcname=f"baselines/{mode}/evaluation-8192.json",
                )
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
        .replace("colab_worker.py", "colab_large_probe_worker.py")
    )
    session = f"dodge-{args.run_id}"
    cli("new", "--session", session, "--gpu", "T4")
    (job / "session.json").write_text(json.dumps({"session": session}))
    upload(archive, session, job)
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    names = _run_names(
        args.run_id,
        loss_kinds=loss_kinds,
        palette_comparison=args.palette_comparison,
    )
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
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        while process.poll() is None:
            for name in names:
                for artifact in RUN_ARTIFACTS:
                    temp = history / name / (".mirror-" + artifact)
                    try:
                        result = subprocess.run(
                            [
                                "colab",
                                "--auth",
                                "adc",
                                "download",
                                f"/content/lewm-work/history/{name}/{artifact}",
                                str(temp),
                                "--session",
                                session,
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=30,
                        )
                        if result.returncode == 0 and temp.exists():
                            temp.replace(history / name / artifact)
                    except subprocess.TimeoutExpired:
                        pass
            if args.palette_comparison:
                comparison = history / (f".{args.run_id}-palette-comparison.json")
                try:
                    result = subprocess.run(
                        [
                            "colab",
                            "--auth",
                            "adc",
                            "download",
                            f"/content/lewm-work/history/{args.run_id}-palette-comparison.json",
                            str(comparison),
                            "--session",
                            session,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                    if result.returncode == 0 and comparison.exists():
                        comparison.replace(
                            history / f"{args.run_id}-palette-comparison.json"
                        )
                except subprocess.TimeoutExpired:
                    pass
            reevaluation = history / (f".{args.run_id}-baseline-reevaluation.json")
            if baseline_bundle is not None:
                try:
                    result = subprocess.run(
                        [
                            "colab",
                            "--auth",
                            "adc",
                            "download",
                            f"/content/lewm-work/history/{args.run_id}-baseline-reevaluation.json",
                            str(reevaluation),
                            "--session",
                            session,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                    if result.returncode == 0 and reevaluation.exists():
                        reevaluation.replace(
                            history / f"{args.run_id}-baseline-reevaluation.json"
                        )
                except subprocess.TimeoutExpired:
                    pass
            time.sleep(10)
    if (
        process.returncode
        or "LEWM_LARGE_PROBE_COMPLETE" not in (job / "remote.log").read_text()
    ):
        raise RuntimeError(f"Screen failed; session retained: {session}")
    cli(
        "download",
        "/content/lewm-large-probe-results.tar.gz",
        str(job / "results.tar.gz"),
        "--session",
        session,
        timeout=300,
    )
    (job / "results.sha256").write_text(digest(job / "results.tar.gz") + "\n")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    shutil.copytree(job / "results/history", history, dirs_exist_ok=True)
    if (job / "source.sha256").read_text().strip() != source_hash:
        raise RuntimeError("local source hash record changed; session retained")
    environment = _read_json(
        history / f"{args.run_id}-environment.json",
        f"{args.run_id}-environment.json",
    )
    if (
        environment["source_sha256"] != source_hash
        or environment["protocol"] != protocol
    ):
        raise RuntimeError("retrieved provenance mismatch; session retained")
    bank_provenance = history / f"{args.run_id}-bank-provenance"
    for split in ("train", "validation"):
        for filename in ("metadata.json", "index.json", "READY"):
            artifact = bank_provenance / split / filename
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(
                    f"missing bank provenance {split}/{filename}; session retained"
                )
    for loss_kind in loss_kinds:
        condition_id = _condition_run_id(
            args.run_id, loss_kind, args.palette_comparison
        )
        for mode in MODES:
            _validate_run_artifacts(
                history / f"{condition_id}-{mode}",
                expected_loss_kind=loss_kind,
                protocol=protocol,
            )
        condition_comparison = _read_json(
            history / f"{condition_id}-comparison.json",
            f"{condition_id}-comparison.json",
        )
        if (
            condition_comparison.get("loss_kind", "mse") != loss_kind
            or condition_comparison.get("checkpoint_sha256")
            != protocol["checkpoint_sha256"]
            or condition_comparison.get("data_sha256") != protocol["data_hash"]
            or condition_comparison.get("milestones") != protocol["milestones"]
        ):
            raise RuntimeError(
                f"{condition_id}-comparison.json provenance mismatch; session retained"
            )
    if args.palette_comparison:
        _validate_palette_comparison(
            history, args.run_id, protocol, source_hash=source_hash
        )
    if baseline_bundle is not None:
        reevaluation_path = history / f"{args.run_id}-baseline-reevaluation.json"
        if not reevaluation_path.is_file():
            raise RuntimeError(
                "baseline reevaluation evidence is missing; session retained"
            )
        reevaluation = json.loads(reevaluation_path.read_text())
        if (
            reevaluation.get("loss_kind") != protocol["loss_kind"]
            or reevaluation.get("baseline_loss_kind") != "mse"
            or reevaluation.get("baseline_run_id") != args.baseline_run_id
            or reevaluation.get("passed") is not True
            or set(reevaluation.get("representations", {})) != set(MODES)
        ):
            raise RuntimeError(
                "baseline reevaluation provenance mismatch; session retained"
            )
    cli("stop", "--session", session)
    print(f"Artifacts retrieved and T4 released: {job}", flush=True)


if __name__ == "__main__":
    main()
