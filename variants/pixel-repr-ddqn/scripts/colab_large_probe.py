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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODES = ("cls", "projected")
MILESTONES = (512, 2048, 8192)


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
        "--loss-kind", choices=("mse", "balanced-bright"), default="mse"
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
    if args.loss_kind == "mse" and args.baseline_root is not None:
        parser.error("baseline artifacts are only used by balanced-bright runs")
    baseline_bundle = None
    if args.baseline_root is not None:
        baseline_bundle = _baseline_bundle(
            args.baseline_root.expanduser().resolve(),
            args.baseline_run_id,
        )
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    protocol = {
        "checkpoint_sha256": digest(args.checkpoint),
        "data_hash": digest(args.dataset / "manifest.json"),
        "milestones": list(MILESTONES),
        "batch_size": 32,
        "frames_per_episode": 4,
        "initialization_seed": 904,
        "sampling_seed": 903,
        "loss_kind": args.loss_kind,
        "world_model_updates": 0,
        "output_size": 128,
        "representations": list(MODES),
        "baseline_run_id": args.baseline_run_id,
        "baseline_files": (
            {
                mode: {
                    "decoder-8192.pt": values["decoder-8192.pt"],
                    "evaluation-8192.json": values["evaluation-8192.json"],
                }
                for mode, values in baseline_bundle.items()
            }
            if baseline_bundle is not None
            else {}
        ),
    }
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
    names = [f"{args.run_id}-{mode}" for mode in MODES]
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
                for artifact in (
                    "manifest.json",
                    "status.json",
                    "metrics.jsonl",
                    "report.json",
                    "visualizations.json",
                ):
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
    environment = json.loads((history / f"{args.run_id}-environment.json").read_text())
    if (
        environment["source_sha256"] != source_hash
        or environment["protocol"] != protocol
    ):
        raise RuntimeError("retrieved provenance mismatch; session retained")
    for name in names:
        run = history / name
        report = json.loads((run / "report.json").read_text())
        status = json.loads((run / "status.json").read_text())
        if (
            report["milestones"] != protocol["milestones"]
            or report["world_model_sha256"] != protocol["checkpoint_sha256"]
            or report["data_sha256"] != protocol["data_hash"]
            or report.get("loss_kind", "mse") != protocol["loss_kind"]
            or status["state"] != "completed"
            or status["step"] != protocol["milestones"][-1]
        ):
            raise RuntimeError("completed run provenance mismatch; session retained")
        for step in protocol["milestones"]:
            evaluation = json.loads((run / f"evaluation-{step}.json").read_text())
            if (
                not (run / f"decoder-{step}.pt").is_file()
                or evaluation["splits"]["train"]["frame_count"] != 16384
                or evaluation["splits"]["validation"]["frame_count"] != 2048
            ):
                raise RuntimeError("incomplete milestone evidence; session retained")
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
