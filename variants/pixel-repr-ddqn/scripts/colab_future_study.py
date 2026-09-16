"""Run the frozen predicted next-frame decode study on one T4 and retrieve it."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from colab_large_probe import ROOT, _validate_run_id, cli, digest, upload

EXPERIMENT = "lewm-future-decode-v1"
MILESTONES = (512, 2048)
SCENE_COUNT = 16
READOUTS = {
    "predicted": "predicted-latent-broadcast",
    "actual": "actual-next-latent",
}


def _protocol(
    *,
    checkpoint_sha256: str,
    current_decoder_sha256: str,
    data_sha256: str,
    run_id: str,
    source_commit: str,
    readout: str,
    ab_decoder_sha256: str | None = None,
    ab_source_run: str | None = None,
) -> dict[str, object]:
    protocol: dict[str, object] = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "checkpoint_name": "world.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "current_decoder_name": "current-decoder.pt",
        "current_decoder_sha256": current_decoder_sha256,
        "data_sha256": data_sha256,
        "world_updates": 0,
        "world_model_updates": 0,
        "milestones": list(MILESTONES),
        "batch_size": 32,
        "decoder_init_seed": 904,
        "decoder_sampling_seed": 903,
        "loss_kind": "balanced-bright",
        "equal_class_weights": True,
        "readout": READOUTS[readout],
        "scene_count": SCENE_COUNT,
        "worker_timeout_seconds": 7200,
        "source_commit": source_commit,
    }
    if readout == "actual":
        protocol["ab_decoder_name"] = "ab-decoder.pt"
        protocol["ab_decoder_sha256"] = ab_decoder_sha256
        protocol["ab_source_run"] = ab_source_run
    return protocol


def _write_refresh_script(job: Path, session: str) -> Path:
    script = job / "refresh_session.py"
    script.write_text(
        "from colab_cli.auth import AuthProvider\n"
        "from colab_cli.common import state\n"
        "state.auth_provider = AuthProvider.ADC\n"
        f"session = state.store.get({session!r})\n"
        "if session is None:\n"
        "    raise RuntimeError('local session entry is missing')\n"
        "matches = [\n"
        "    a for a in state.client.list_assignments()\n"
        "    if a.endpoint == session.endpoint\n"
        "]\n"
        "if len(matches) != 1:\n"
        "    raise RuntimeError('expected Colab assignment is unavailable')\n"
        "proxy = matches[0].runtime_proxy_info\n"
        "session.token = proxy.token\n"
        "session.url = proxy.url\n"
        "state.store.add(session)\n"
        "lifetime = proxy.token_expires_in_seconds\n"
        "print(f'Runtime proxy refreshed; lifetime {lifetime}s')\n"
    )
    return script


def _refresh(session: str, job: Path) -> None:
    executable = (
        Path(shutil.which("colab")).read_text().splitlines()[0].removeprefix("#!")
    )
    subprocess.run(
        [executable, str(_write_refresh_script(job, session))],
        check=True,
    )


def _download(session: str, job: Path, remote_name: str, local_name: str) -> None:
    try:
        cli(
            "download",
            f"/content/{remote_name}",
            str(job / local_name),
            "--session",
            session,
            timeout=600,
        )
    except (RuntimeError, subprocess.TimeoutExpired):
        _refresh(session, job)
        cli(
            "download",
            f"/content/{remote_name}",
            str(job / local_name),
            "--session",
            session,
            timeout=600,
        )


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--current-decoder", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--readout", default="predicted", choices=tuple(READOUTS))
    parser.add_argument("--ab-decoder", type=Path, default=None)
    parser.add_argument("--ab-source-run", default=None)
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    checkpoint = args.checkpoint.expanduser().resolve()
    current_decoder = args.current_decoder.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if not current_decoder.is_file():
        parser.error(f"current decoder does not exist: {current_decoder}")
    if not (dataset / "manifest.json").is_file():
        parser.error(f"dataset manifest does not exist: {dataset / 'manifest.json'}")
    ab_decoder = None
    if args.readout == "actual":
        if args.ab_decoder is None or not args.ab_source_run:
            parser.error("--ab-decoder and --ab-source-run are required for actual")
        ab_decoder = args.ab_decoder.expanduser().resolve()
        if not ab_decoder.is_file():
            parser.error(f"reference decoder does not exist: {ab_decoder}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Freeze and commit source before fitting")
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    protocol = _protocol(
        checkpoint_sha256=digest(checkpoint),
        current_decoder_sha256=digest(current_decoder),
        data_sha256=digest(dataset / "manifest.json"),
        run_id=args.run_id,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        readout=args.readout,
        ab_decoder_sha256=digest(ab_decoder) if ab_decoder is not None else None,
        ab_source_run=args.ab_source_run,
    )
    (job / "future_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
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
        output.add(current_decoder, arcname="current-decoder.pt")
        if ab_decoder is not None:
            output.add(ab_decoder, arcname="ab-decoder.pt")
        output.add(dataset, arcname="dataset")
        output.add(job / "future_protocol.json", arcname="future_protocol.json")
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
        .replace("colab_worker.py", "colab_future_worker.py")
    )
    session = f"dodge-{args.run_id}"
    cli("new", "--session", session, "--gpu", "T4")
    (job / "session.json").write_text(json.dumps({"session": session}) + "\n")
    upload(archive, session, job)
    with (job / "remote.log").open("w") as log:
        result = subprocess.run(
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
    remote_log = (job / "remote.log").read_text()
    if result.returncode or "FUTURE_COMPLETE" not in remote_log:
        raise RuntimeError(f"Future study failed; session retained: {session}")
    _refresh(session, job)
    _download(session, job, "lewm-future-results.sha256", "results.sha256")
    _download(session, job, "lewm-future-results.tar.gz", "results.tar.gz")
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("retrieved archive checksum mismatch; session retained")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
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
    run = history / args.run_id
    report = _read_json(run / "report.json", "report.json")
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("world_model_sha256") != protocol["checkpoint_sha256"]
        or report.get("data_sha256") != protocol["data_sha256"]
        or report.get("current_decoder_sha256")
        != protocol["current_decoder_sha256"]
        or report.get("world_model_updates") != 0
        or report.get("milestones") != protocol["milestones"]
        or report.get("loss_kind") != "balanced-bright"
        or report.get("readout") != protocol["readout"]
    ):
        raise RuntimeError("future report provenance mismatch; session retained")
    if protocol["readout"] == "actual-next-latent" and (
        report.get("ab_source_run") != protocol["ab_source_run"]
        or report.get("ab_decoder_sha256") != protocol["ab_decoder_sha256"]
    ):
        raise RuntimeError("future reference provenance mismatch; session retained")
    gallery = history / f"{args.run_id}-comparison.html"
    if not gallery.is_file() or gallery.stat().st_size == 0:
        raise RuntimeError("future comparison gallery is missing; session retained")
    for name in (
        "manifest.json",
        "config.json",
        "status.json",
        "metrics.jsonl",
        "report.json",
        "visualizations.json",
        "decoder.pt",
        "current-decoder.pt",
    ):
        if not (run / name).is_file():
            raise RuntimeError(f"missing {args.run_id}/{name}; session retained")
    print(
        f"Future artifacts retrieved; verify checkpoints before releasing T4: {job}",
        flush=True,
    )


if __name__ == "__main__":
    main()
