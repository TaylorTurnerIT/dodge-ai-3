"""Run the AD5 Track-B survival probe and gated MPC eval on one T4.

Uploads the frozen study checkpoint and mortal probe set with the frozen
source, fits the protocol probe on CUDA, applies the prespecified AUPRC
gate, and runs the 32-scenario MPC comparison only on a pass.  A gate
failure retrieves the probe evidence with an mpc-skipped record instead.
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

EXPERIMENT = "lewm-survival-trackb-v1"
PROBE_STEPS = 512
PROBE_BATCH = 256
PROBE_SEED = 905
PROBE_MAX_WINDOWS = 20000
MIN_VAL_AUPRC = 0.10
MPC_POLICIES = {
    "mpc_h3": "mpc_h3",
    "mpc_h3_last": "mpc_h3_last",
    "random": "random",
    "neutral": "neutral",
}
SCENARIO_COHORTS = (0, 1)
MPC_HISTORY = 3
MPC_DECISIONS = 128
UPLOAD_WORKERS = 6
UPLOAD_PART_SIZE = 8 * 1024**2
SITE_PACKAGES = [
    "gymnasium>=1",
    "numpy>=2.4.6",
    "transformers==4.57.6",
    "einops==0.8.2",
    "Pillow",
    "pytest",
]
EXPECTED_TRANSFORMERS = "4.57.6"
WHEEL_CACHE_ROOT = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-cache/wheels"
SOURCE_NAMES = [
    "src",
    "native/Cargo.toml",
    "native/Cargo.lock",
    "native/crates",
    "third_party",
    "variants/pixel-repr-ddqn",
]
CHECKPOINT_RELPATH = "scale-resume-20000/checkpoint.pt"
PROBE_SET_NAME = "mortal-probe-20260917-v1"


def build_protocol(
    *,
    run_id: str,
    checkpoint: Path,
    probe_set: Path,
    source_commit: str,
) -> tuple[dict, dict[str, Path]]:
    """Pin the frozen inputs and prespecified gate; map bundle relpaths."""

    manifest = probe_set / "manifest.json"
    for path in (checkpoint, manifest):
        if not path.is_file():
            raise ValueError(f"frozen input is missing: {path}")
    protocol = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "source_commit": source_commit,
        "inputs": {
            "checkpoint.pt": digest(checkpoint),
            "probe-set/manifest.json": digest(manifest),
        },
        "probe": {
            "steps": PROBE_STEPS,
            "batch_size": PROBE_BATCH,
            "seed": PROBE_SEED,
            "max_windows": PROBE_MAX_WINDOWS,
        },
        "gate": {
            "min_val_auprc": MIN_VAL_AUPRC,
            "rationale": (
                "chance AUPRC equals the validation positive rate "
                "(~0.025); 0.10 demands roughly 4x chance before any "
                "MPC comparison, prespecified before the run"
            ),
        },
        "mpc": {
            "policies": dict(MPC_POLICIES),
            "scenario_cohorts": list(SCENARIO_COHORTS),
            "history_size": MPC_HISTORY,
            "max_decisions": MPC_DECISIONS,
        },
        "worker_timeout_seconds": 7200,
    }
    bundle = {
        "checkpoint.pt": checkpoint,
        "probe-set/manifest.json": manifest,
    }
    return protocol, bundle


def build_remote_driver(
    *,
    source_hash: str,
    run_id: str,
    wheel: dict | None,
    site_packages: list[str],
) -> str:
    """Generate the remote driver: setup, native build, smoke, then scored.

    Environment setup uses uv (falling back to pip) for site packages; the
    native extension comes from a content-keyed cached abi3 wheel when the
    protocol carries one, otherwise it is built with the Rust toolchain and
    captured back into a wheel for the local cache.  Smoke and scored phases
    run as separate fresh worker processes after setup completes.
    """

    wheel_name = wheel["filename"] if wheel else None
    driver = (
        "import os,sys,hashlib,tarfile,subprocess,pathlib\n"
        "from pathlib import Path\n"
        "archive=Path('/content/lewm-source.tar.gz')\n"
        f"assert hashlib.sha256(archive.read_bytes()).hexdigest()=={source_hash!r}\n"
        "print('SOURCE_ARCHIVE_VERIFIED',flush=True)\n"
        "work=Path('/content/lewm-survival-work');work.mkdir(exist_ok=True)\n"
        "stage=Path('/content/lewm-survival-stage');stage.mkdir(exist_ok=False)\n"
        "with tarfile.open(archive) as bundle: bundle.extractall(stage,filter='data')\n"
        "code=Path('/content/lewm-survival-code');(stage/'code').rename(code)\n"
        "inputs=Path('/content/lewm-survival-inputs');(stage/'inputs').rename(inputs)\n"
        "stage.rmdir()\n"
        "def run_command(command):\n"
        " process=subprocess.Popen(command,stdout=subprocess.PIPE,"
        "stderr=subprocess.STDOUT,text=True)\n"
        " [print(line,end='',flush=True) for line in process.stdout]\n"
        " if process.wait(): raise RuntimeError(f'Setup failed: {command[0]}')\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q','uv'],"
        "capture_output=True)\n"
        f"packages={site_packages!r}\n"
        "uv_ok=False\n"
        "try:\n"
        " uv_ok=subprocess.run([sys.executable,'-m','uv','pip','install',"
        "'--system','-q',*packages]"
        + (f"+[str(code/'wheel'/{wheel_name!r})]" if wheel_name else "")
        + ",capture_output=True).returncode==0\n"
        "except (FileNotFoundError,OSError):\n"
        " pass\n"
        "print('UV_SETUP_OK' if uv_ok else 'UV_SETUP_FALLBACK',flush=True)\n"
        "if not uv_ok:\n"
        " run_command([sys.executable,'-m','pip','install','-q',*packages])\n"
    )
    if wheel_name:
        driver += "print('WHEEL_CACHE_HIT'," f"{wheel_name!r},flush=True)\n"
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
            " run_command([sys.executable,'-m','uv','pip','install',"
            "'--system','-q',crate])\n"
            "else:\n"
            " run_command([sys.executable,'-m','pip','install','-q',crate])\n"
            "wheelhouse=Path('/content/lewm-survival-wheelhouse');wheelhouse.mkdir(exist_ok=False)\n"
            "run_command([sys.executable,'-m','pip','wheel','--no-deps','-q','-w',str(wheelhouse),crate])\n"
            "wheels=list(wheelhouse.glob('dodge_native-*.whl'))\n"
            "assert len(wheels)==1, f'expected one captured wheel: {wheels}'\n"
            "shutil.copy2(wheels[0],Path('/content/lewm-survival-wheel.whl'))\n"
            "print('WHEEL_CAPTURED',wheels[0].name,flush=True)\n"
        )
    driver += (
        "worker=str(code/'variants/pixel-repr-ddqn/scripts/colab_survival_worker.py')\n"
        "base=dict(os.environ,PYTHONPATH=str(code/'src'),"
        f"LEWM_SOURCE_HASH={source_hash!r},LEWM_RUN_ID={run_id!r},"
        "OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')\n"
        "def run_phase(mode,logname,timeout,marker):\n"
        " log=Path('/content')/logname\n"
        " handle=log.open('w')\n"
        " proc=subprocess.run([sys.executable,worker,'--mode',mode],"
        "env=base,check=False,timeout=timeout,stdout=handle,"
        "stderr=subprocess.STDOUT,text=True)\n"
        " handle.close()\n"
        " print(mode.upper()+'_RETURNCODE',proc.returncode,flush=True)\n"
        " body=log.read_text()\n"
        " if proc.returncode or marker not in body:\n"
        "  print(mode.upper()+'_LOG_TAIL',body[-6000:],flush=True)\n"
        "  raise RuntimeError(f'survival {mode} phase failed')\n"
        " return proc\n"
        "run_phase('smoke','smoke.log',2400,'SURVIVAL_SMOKE_COMPLETE')\n"
        "run_phase('scored','scored.log',4600,'SURVIVAL_DRIVER_COMPLETE')\n"
        "print('SURVIVAL_LAUNCHER_COMPLETE',flush=True)\n"
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


def _publish_results(extracted: Path, run_dir: Path) -> None:
    """Copy the verified result files (and trace tree) into the run dir."""

    for name in (
        "probe.pt",
        "probe.json",
        "mpc-report.json",
        "mpc-skipped.json",
        "environment.json",
    ):
        candidate = extracted / name
        if candidate.is_file():
            shutil.copy2(candidate, run_dir / name)
    trace_src = extracted / "trace"
    if trace_src.is_dir():
        shutil.copytree(trace_src, run_dir / "trace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Freeze and commit source before fitting")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    checkpoint = history / CHECKPOINT_RELPATH
    probe_set = history / PROBE_SET_NAME
    protocol, bundle = build_protocol(
        run_id=args.run_id,
        checkpoint=checkpoint,
        probe_set=probe_set,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    wheel_path = cached_wheel(native_tree_key())
    wheel: dict | None = None
    if wheel_path is not None:
        wheel = {"filename": wheel_path.name, "sha256": digest(wheel_path)}
    (job / "survival_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    raw = job / "source.tar"
    with tarfile.open(raw, "w") as output:
        for name in SOURCE_NAMES:
            output.add(
                ROOT / name,
                arcname=f"code/{name}",
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(
            job / "survival_protocol.json",
            arcname="code/survival_protocol.json",
        )
        if wheel is not None:
            output.add(wheel_path, arcname=f"code/wheel/{wheel_path.name}")
        output.add(checkpoint, arcname="inputs/checkpoint.pt")
        for relpath, local in sorted(bundle.items()):
            if relpath == "checkpoint.pt":
                continue
            output.add(local, arcname=f"inputs/{relpath}")
        probe_episodes = probe_set / "episodes"
        output.add(probe_episodes, arcname="inputs/probe-set/episodes")
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
    remote = job / "remote.py"
    remote.write_text(
        build_remote_driver(
            source_hash=source_hash,
            run_id=args.run_id,
            wheel=wheel,
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
        remote_prefix="lewm-survival",
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
    if result.returncode or "SURVIVAL_LAUNCHER_COMPLETE" not in remote_log:
        raise RuntimeError(f"Survival run failed; session retained: {session}")
    _refresh(session, job)
    _download(session, job, "lewm-survival-results.sha256", "results.sha256")
    _download(session, job, "lewm-survival-results.tar.gz", "results.tar.gz")
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("retrieved archive checksum mismatch; session retained")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    run_dir = history / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _publish_results(job / "results" / "results", run_dir)
    probe_report = _read_json(run_dir / "probe.json", "probe report")
    if probe_report.get("world_model_sha256") != protocol["inputs"]["checkpoint.pt"]:
        raise RuntimeError("retrieved probe ran on a different checkpoint")
    if not (run_dir / "environment.json").is_file():
        raise RuntimeError("retrieved probe evidence is incomplete")
    environment = _read_json(run_dir / "environment.json", "environment")
    if "gate_verdict" not in environment:
        raise RuntimeError("retrieved probe evidence has no gate verdict")
    if environment.get("transformers_version") != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            "retrieved environment used an unexpected transformers version; "
            "session retained"
        )
    if (
        environment.get("source_sha256") != source_hash
        or environment.get("protocol") != protocol
    ):
        raise RuntimeError("retrieved provenance mismatch; session retained")
    if wheel is None:
        _download(session, job, "lewm-survival-wheel.whl", "wheel.whl")
        wheel_dir = WHEEL_CACHE_ROOT / native_tree_key()
        wheel_dir.mkdir(parents=True, exist_ok=True)
        target = wheel_dir / "dodge_native-0.1.0-cp311-abi3-linux_x86_64.whl"
        shutil.copy2(job / "wheel.whl", target)
        print(f"native wheel cached for reuse: {target}")
    print(f"Survival artifacts retrieved; run dir: {run_dir}; job: {job}")


if __name__ == "__main__":
    main()
