"""Run the frozen pooling comparison on a fresh T4 and retrieve it.

Restores the retained §Z inputs (dataset, world checkpoint, available bank
bytes) onto a new machine into an immutable input root.  Files Phase 0
confirmed missing are re-created on the T4 by hash-gated recovery only:
staged, byte-verified against the full recorded digests, then promoted.
Any mismatch stops the run before fitting.
"""

from __future__ import annotations

import argparse
import gzip
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
UPLOAD_WORKERS = 6
UPLOAD_PART_SIZE = 32 * 1024**2
SITE_PACKAGES = [
    "gymnasium>=1",
    "numpy>=2.4.6",
    "transformers==4.57.6",
    "einops==0.8.2",
    "Pillow",
    "pytest",
]
WHEEL_CACHE_ROOT = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-cache/wheels"
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
        "wheel": None,
        "inputs_archive_sha256": None,
        "smoke_milestones": list(SMOKE_MILESTONES),
        "core_initial_state_sha256": comparison["initial_state_sha256"],
        "frame_index_sha256": comparison["frame_index_sha256"],
        "palette_sha256": comparison["palette_sha256"],
        "arms": list(ARMS),
        "worker_timeout_seconds": 7200,
    }
    return protocol, bundle


def build_remote_driver(
    *,
    source_hash: str,
    run_id: str,
    wheel: dict | None,
    inputs_url: str | None,
    inputs_archive_sha256: str | None,
    site_packages: list[str],
) -> str:
    """Generate the remote driver: setup, restore inputs, smoke, then score.

    Environment setup uses uv (falling back to pip) for site packages; the
    native extension comes from a content-keyed cached abi3 wheel when the
    protocol carries one, otherwise it is built with the Rust toolchain and
    captured back into a wheel for the local cache.  Inputs arrive either
    bundled in the source archive or via a hash-verified URL fetch when the
    protocol records an inputs archive digest.  Smoke and scored phases run
    as separate fresh worker processes after setup completes.
    """

    wheel_name = wheel["filename"] if wheel else None
    fetch_inputs = bool(inputs_url and inputs_archive_sha256)
    driver = (
        "import os,sys,hashlib,tarfile,subprocess,pathlib\n"
        "from pathlib import Path\n"
    )
    if fetch_inputs:
        driver += f"os.environ['LEWM_INPUTS_URL']={inputs_url!r}\n"
    driver += (
        "archive=Path('/content/lewm-source.tar.gz')\n"
        f"assert hashlib.sha256(archive.read_bytes()).hexdigest()=={source_hash!r}\n"
        "print('SOURCE_ARCHIVE_VERIFIED',flush=True)\n"
        "stage=Path('/content/lewm-pooling-stage');stage.mkdir(exist_ok=False)\n"
        "with tarfile.open(archive) as bundle: bundle.extractall(stage,filter='data')\n"
        "code=Path('/content/lewm-pooling-code');(stage/'code').rename(code)\n"
    )
    if fetch_inputs:
        driver += (
            "import urllib.request\n"
            "inputs_archive=Path('/content/lewm-pooling-inputs.tar.gz')\n"
            "print('INPUTS_FETCH_START',flush=True)\n"
            "urllib.request.urlretrieve("
            "os.environ['LEWM_INPUTS_URL'],inputs_archive)\n"
            "inputs_digest=hashlib.sha256(inputs_archive.read_bytes()).hexdigest()\n"
            f"assert inputs_digest=={inputs_archive_sha256!r}\n"
            "print('INPUTS_ARCHIVE_VERIFIED',flush=True)\n"
            "with tarfile.open(inputs_archive) as bundle:\n"
            " bundle.extractall(stage,filter='data')\n"
            "inputs_archive.unlink()\n"
            "inputs=Path('/content/lewm-pooling-inputs');(stage/'inputs').rename(inputs)\n"
        )
    else:
        driver += (
            "inputs=Path('/content/lewm-pooling-inputs');(stage/'inputs').rename(inputs)\n"
        )
    driver += (
        "stage.rmdir()\n"
        "def run_command(command):\n"
        " process=subprocess.Popen(command,stdout=subprocess.PIPE,"
        "stderr=subprocess.STDOUT,text=True)\n"
        " [print(line,end='',flush=True) for line in process.stdout]\n"
        " if process.wait(): raise RuntimeError(f'Setup failed: {command[0]}')\n"
        "run_command([sys.executable,'-m','pip','install','-q','uv'])\n"
        f"packages={site_packages!r}\n"
        "uv_binary=pathlib.Path(sys.executable).parent/'uv'\n"
        "uv_ok=subprocess.run([str(uv_binary),'pip','install','--system','-q',*packages]"
        + (f"+[str(code/'wheel'/{wheel_name!r})]" if wheel_name else "")
        + ",capture_output=True).returncode==0\n"
        "print('UV_SETUP_OK' if uv_ok else 'UV_SETUP_FALLBACK',flush=True)\n"
        "if not uv_ok:\n"
        " run_command([sys.executable,'-m','pip','install','-q',*packages])\n"
    )
    if wheel_name:
        driver += (
            "print('WHEEL_CACHE_HIT',"
            f"{wheel_name!r},flush=True)\n"
        )
    else:
        driver += (
            "import shutil\n"
            "if not shutil.which('cargo'):\n"
            " import urllib.request\n"
            " urllib.request.urlretrieve("
            "'https://sh.rustup.rs','/content/rustup-init.sh')\n"
            " run_command(['sh','/content/rustup-init.sh','-y',"
            "'--profile','minimal'])\n"
            "os.environ['PATH']=str(pathlib.Path.home()/'.cargo/bin')+':'+os.environ['PATH']\n"
            "crate=str(code/'native/crates/dodge-python')\n"
            "if uv_ok:\n"
            " run_command([str(uv_binary),'pip','install','--system','-q',crate])\n"
            "else:\n"
            " run_command([sys.executable,'-m','pip','install','-q',crate])\n"
            "wheelhouse=Path('/content/lewm-pooling-wheelhouse');wheelhouse.mkdir(exist_ok=False)\n"
            "run_command([sys.executable,'-m','pip','wheel','--no-deps','-q','-w',str(wheelhouse),crate])\n"
            "wheels=list(wheelhouse.glob('dodge_native-*.whl'))\n"
            "assert len(wheels)==1, f'expected one captured wheel: {wheels}'\n"
            "shutil.copy2(wheels[0],Path('/content/lewm-pooling-wheel.whl'))\n"
            "print('WHEEL_CAPTURED',wheels[0].name,flush=True)\n"
        )
    driver += (
        "worker=str(code/'variants/pixel-repr-ddqn/scripts/colab_pooling_worker.py')\n"
        "base=dict(os.environ,PYTHONPATH=str(code/'src'),"
        f"LEWM_SOURCE_HASH={source_hash!r},LEWM_RUN_ID={run_id!r},"
        "OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')\n"
        "def run_phase(mode,logname,timeout):\n"
        " log=Path('/content')/logname\n"
        " handle=log.open('w')\n"
        " proc=subprocess.run([sys.executable,worker,'--mode',mode],"
        "env=base,check=False,timeout=timeout,stdout=handle,"
        "stderr=subprocess.STDOUT,text=True)\n"
        " handle.close()\n"
        " print(mode.upper()+'_RETURNCODE',proc.returncode,flush=True)\n"
        " if proc.returncode:\n"
        "  print(mode.upper()+'_LOG_TAIL',log.read_text()[-6000:],flush=True)\n"
        "  raise RuntimeError(f'pooling {mode} phase failed')\n"
        " return proc\n"
        "run_phase('smoke','smoke.log',2400)\n"
        "run_phase('scored','scored.log',4600)\n"
        "print('POOLING_DRIVER_COMPLETE',flush=True)\n"
    )
    return driver


def native_tree_key() -> str:
    """Content key for the native extension sources (abi3: python-agnostic)."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD:native/crates"], cwd=ROOT, text=True
    ).strip()


def cached_wheel(key: str) -> Path | None:
    """Return a cached abi3 wheel for this native tree, if one was stored."""

    candidates = sorted((WHEEL_CACHE_ROOT / key).glob("dodge_native-*.whl"))
    return candidates[0] if candidates else None


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
    parser.add_argument(
        "--inputs-url",
        default=None,
        help="Optional URL of a provisioned inputs.tar.gz (inputs/ layout); "
        "when set, --inputs-sha256 is required and inputs travel outside "
        "the source archive. Provisioning the cache needs bucket "
        "credentials this host does not hold yet.",
    )
    parser.add_argument("--inputs-sha256", default=None)
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    _validate_run_id(args.spatial_run, "spatial run ID")
    if bool(args.inputs_url) != bool(args.inputs_sha256):
        parser.error("--inputs-url and --inputs-sha256 are required together")
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
    wheel: dict | None = None
    if args.inputs_url is None:
        wheel_path = cached_wheel(native_tree_key())
        if wheel_path is not None:
            wheel = {
                "key": native_tree_key(),
                "filename": wheel_path.name,
                "sha256": digest(wheel_path),
            }
    protocol["wheel"] = wheel
    protocol["inputs_archive_sha256"] = args.inputs_sha256
    (job / "pooling_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    raw = job / "source.tar"
    with tarfile.open(raw, "w") as output:
        for name in SOURCE_NAMES:
            output.add(
                ROOT / name,
                arcname=f"code/{name}",
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(
            job / "pooling_protocol.json", arcname="code/pooling_protocol.json"
        )
        if wheel is not None:
            cached = cached_wheel(wheel["key"])
            assert cached is not None and cached.name == wheel["filename"]
            output.add(cached, arcname=f"code/wheel/{cached.name}")
        if args.inputs_url is None:
            output.add(dataset, arcname="inputs/dataset")
            for relpath, local in sorted(bundle.items()):
                if relpath == "dataset/manifest.json":
                    continue
                output.add(local, arcname=f"inputs/{relpath}")
    archive = job / "source.tar.gz"
    with raw.open("rb") as stream, gzip.open(
        archive, "wb", compresslevel=6
    ) as compressed:
        shutil.copyfileobj(stream, compressed, length=8 * 1024**2)
    raw.unlink()
    if archive.stat().st_size > 1024**3:
        raise ValueError("source archive exceeds 1 GiB")
    source_hash = digest(archive)
    (job / "source.sha256").write_text(source_hash + "\n")
    if args.inputs_url is None:
        inputs_archive = job / "inputs.tar.gz"
        with tarfile.open(inputs_archive, "w:gz") as output:
            output.add(dataset, arcname="inputs/dataset")
            for relpath, local in sorted(bundle.items()):
                if relpath == "dataset/manifest.json":
                    continue
                output.add(local, arcname=f"inputs/{relpath}")
        print(f"inputs side artifact for future provisioning: {inputs_archive}")
    remote = job / "remote.py"
    remote.write_text(
        build_remote_driver(
            source_hash=source_hash,
            run_id=args.run_id,
            wheel=wheel,
            inputs_url=args.inputs_url,
            inputs_archive_sha256=args.inputs_sha256,
            site_packages=list(SITE_PACKAGES),
        )
    )
    session = f"dodge-{args.run_id}"
    cli("new", "--session", session, "--gpu", "T4")
    (job / "session.json").write_text(json.dumps({"session": session}) + "\n")
    upload(
        archive,
        session,
        job,
        workers=UPLOAD_WORKERS,
        part_size=UPLOAD_PART_SIZE,
        remote_prefix="lewm-pooling",
    )
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
    if wheel is None:
        _download(session, job, "lewm-pooling-wheel.whl", "wheel.whl")
        wheel_dir = WHEEL_CACHE_ROOT / native_tree_key()
        wheel_dir.mkdir(parents=True, exist_ok=True)
        target = wheel_dir / "dodge_native-0.1.0-cp311-abi3-linux_x86_64.whl"
        shutil.copy2(job / "wheel.whl", target)
        print(f"native wheel cached for reuse: {target}")
    else:
        recovered = environment.get("recovery", {})
        wheel_record = recovered.get("wheel_source") or environment.get("wheel_source")
        print(f"native wheel reused from cache: {wheel['filename']} ({wheel_record})")
    print(
        "Pooling artifacts retrieved; verify checkpoints before releasing T4: "
        f"{job}"
    )


if __name__ == "__main__":
    main()
