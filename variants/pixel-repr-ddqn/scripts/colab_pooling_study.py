"""Run the frozen pooling comparison on a fresh T4 and retrieve it.

Restores the retained §Z inputs (dataset, world checkpoint, available bank
bytes) onto a new machine into an immutable input root.  Files Phase 0
confirmed missing are re-created on the T4 by hash-gated recovery only:
staged, byte-verified against the full recorded digests, then promoted.
Any mismatch stops the run before fitting.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from colab_large_probe import ROOT, _validate_run_id, cli, digest, upload

EXPERIMENT = "lewm-pooling-readout-v1"
MILESTONES = (512, 2048)
ARMS = ["mean", "max", "attention", "grid4"]
SMOKE_MILESTONES = [4]
SOURCE_NAMES = [
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
    "references/pooling",
]


def _recorded_hashes(provenance: Path) -> dict[str, str]:
    """Collect the full recorded digests for every expected input file."""

    recorded: dict[str, str] = {}
    for kind in ("standard", "spatial"):
        for split in ("train", "validation"):
            metadata = json.loads(
                (provenance / kind / split / "metadata.json").read_text()
            )
            if kind == "standard":
                for name, entry in metadata["files"].items():
                    recorded[f"spatial-banks/{kind}/{split}/{name}.npy"] = entry[
                        "sha256"
                    ]
            else:
                recorded[f"spatial-banks/{kind}/{split}/patches.npy"] = metadata[
                    "patches_sha256"
                ]
    return recorded


def build_protocol(
    *,
    provenance: Path,
    dataset: Path,
    run_id: str,
    spatial_run: str,
    source_commit: str,
    comparison: dict,
) -> tuple[dict, dict[str, Path]]:
    """Build the protocol and map bundled relpaths to local files.

    Files present locally are digested from bytes and must agree with the
    recorded digests.  Missing files enter ``recovery_targets`` with their
    full recorded digests and are excluded from the upload bundle.
    Returns (protocol, bundle) where bundle maps archive inputs/ relpaths to
    local paths.
    """

    recorded = _recorded_hashes(provenance)
    inputs: dict[str, str] = {
        "world.pt": comparison["world_model_sha256"],
        "dataset/manifest.json": comparison["data_sha256"],
    }
    bundle: dict[str, Path] = {
        "world.pt": provenance / "world.pt",
        "dataset/manifest.json": dataset / "manifest.json",
    }
    recovery_targets: dict[str, str] = {}
    for kind in ("standard", "spatial"):
        for split in ("train", "validation"):
            names = (
                ["metadata.json", "index.json", "READY"]
                if kind == "standard"
                else ["metadata.json", "READY"]
            )
            for name in names:
                relpath = f"spatial-banks/{kind}/{split}/{name}"
                local = provenance / kind / split / name
                inputs[relpath] = digest(local)
                bundle[relpath] = local
            for relpath, recorded_digest in sorted(recorded.items()):
                prefix = f"spatial-banks/{kind}/{split}/"
                if not relpath.startswith(prefix):
                    continue
                local = provenance / relpath.removeprefix("spatial-banks/")
                if local.is_file():
                    actual = digest(local)
                    if actual != recorded_digest:
                        raise ValueError(
                            f"local bytes disagree with recorded digest: {relpath}"
                        )
                    inputs[relpath] = actual
                    bundle[relpath] = local
                else:
                    inputs[relpath] = recorded_digest
                    recovery_targets[relpath] = recorded_digest
    if digest(provenance / "world.pt") != inputs["world.pt"]:
        raise ValueError("local world checkpoint disagrees with recorded digest")
    if digest(dataset / "manifest.json") != inputs["dataset/manifest.json"]:
        raise ValueError("local dataset manifest disagrees with recorded digest")
    protocol = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "source_spatial_run": spatial_run,
        "source_commit": source_commit,
        "milestones": list(MILESTONES),
        "world_model_updates": 0,
        "inputs": inputs,
        "recovery_targets": recovery_targets,
        "smoke_milestones": list(SMOKE_MILESTONES),
        "core_initial_state_sha256": comparison["initial_state_sha256"],
        "frame_index_sha256": comparison["frame_index_sha256"],
        "palette_sha256": comparison["palette_sha256"],
        "arms": list(ARMS),
        "worker_timeout_seconds": 7200,
    }
    return protocol, bundle


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


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spatial-run", default="lewm-spatial-study-20260915-v1")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("history/dodge/gymnasium/pixel-repr-ddqn/large-practice-20260914-v2"),
    )
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    _validate_run_id(args.spatial_run, "spatial run ID")
    dataset = args.dataset.expanduser()
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    dataset = dataset.resolve()
    if not (dataset / "manifest.json").is_file():
        parser.error(f"dataset manifest does not exist: {dataset / 'manifest.json'}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Freeze and commit source before fitting")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    provenance = history / f"{args.spatial_run}-bank-provenance"
    comparison = _read_json(
        history / f"{args.spatial_run}-spatial-comparison.json",
        "spatial comparison",
    )
    protocol, bundle = build_protocol(
        provenance=provenance,
        dataset=dataset,
        run_id=args.run_id,
        spatial_run=args.spatial_run,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        comparison=comparison,
    )
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    (job / "pooling_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in SOURCE_NAMES:
            output.add(
                ROOT / name,
                arcname=f"code/{name}",
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(
            job / "pooling_protocol.json", arcname="code/pooling_protocol.json"
        )
        output.add(dataset, arcname="inputs/dataset")
        for relpath, local in sorted(bundle.items()):
            if relpath == "dataset/manifest.json":
                continue
            output.add(local, arcname=f"inputs/{relpath}")
    if archive.stat().st_size > 1024**3:
        raise ValueError("source archive exceeds 1 GiB")
    source_hash = digest(archive)
    (job / "source.sha256").write_text(source_hash + "\n")
    remote = job / "remote.py"
    remote.write_text(
        "import os,sys,hashlib,tarfile,subprocess\n"
        "from pathlib import Path\n"
        "archive=Path('/content/lewm-source.tar.gz')\n"
        f"assert hashlib.sha256(archive.read_bytes()).hexdigest()=={source_hash!r}\n"
        "print('SOURCE_ARCHIVE_VERIFIED',flush=True)\n"
        "stage=Path('/content/lewm-pooling-stage');stage.mkdir(exist_ok=False)\n"
        "with tarfile.open(archive) as bundle: bundle.extractall(stage,filter='data')\n"
        "code=Path('/content/lewm-pooling-code');(stage/'code').rename(code)\n"
        "inputs=Path('/content/lewm-pooling-inputs');(stage/'inputs').rename(inputs)\n"
        "stage.rmdir()\n"
        "worker=str(code/'variants/pixel-repr-ddqn/scripts/colab_pooling_worker.py')\n"
        "base=dict(os.environ,PYTHONPATH=str(code/'src'),"
        f"LEWM_SOURCE_HASH={source_hash!r},LEWM_RUN_ID={args.run_id!r})\n"
        "smoke=subprocess.run([sys.executable,worker,'--mode','smoke'],"
        "env=base,check=False,timeout=2400)\n"
        "print('SMOKE_RETURNCODE',smoke.returncode,flush=True)\n"
        "if smoke.returncode: raise RuntimeError('pooling smoke verification failed')\n"
        "scored=subprocess.run([sys.executable,worker,'--mode','scored'],"
        "env=base,check=False,timeout=4600)\n"
        "print('SCORED_RETURNCODE',scored.returncode,flush=True)\n"
        "if scored.returncode: raise RuntimeError('pooling scored run failed')\n"
        "print('POOLING_DRIVER_COMPLETE',flush=True)\n"
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
    if result.returncode or "POOLING_DRIVER_COMPLETE" not in remote_log:
        raise RuntimeError(f"Pooling run failed; session retained: {session}")
    _refresh(session, job)
    _download(session, job, "lewm-pooling-results.sha256", "results.sha256")
    _download(session, job, "lewm-pooling-results.tar.gz", "results.tar.gz")
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
    for mode in ARMS:
        report = _read_json(
            history / f"{args.run_id}-{mode}/report.json",
            f"{mode} report",
        )
        if (
            report.get("experiment") != EXPERIMENT
            or report.get("step") != 2048
            or report.get("world_model_updates") != 0
        ):
            raise RuntimeError(f"{mode} report contract mismatch; session retained")
    comparison_out = _read_json(
        history / f"{args.run_id}-pooling-comparison.json",
        "pooling comparison",
    )
    if (
        comparison_out.get("arms") != ARMS
        or comparison_out.get("world_model_updates") != 0
    ):
        raise RuntimeError("pooling comparison contract mismatch; session retained")
    if not (history / f"{args.run_id}-pooling-initial.pt").is_file():
        raise RuntimeError("pooling initial state missing; session retained")
    gallery = history / f"{args.run_id}-comparison.html"
    if not gallery.is_file() or gallery.stat().st_size == 0:
        raise RuntimeError("pooling comparison gallery is missing; session retained")
    print(
        "Pooling artifacts retrieved; verify checkpoints before releasing T4: "
        f"{job}"
    )


if __name__ == "__main__":
    main()
