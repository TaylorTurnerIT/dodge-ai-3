"""Run the frozen P9 DDQN screen on one T4.

Uploads the frozen 20k checkpoint with the frozen source, runs the DDQN
worker smoke then the scored screen on CUDA, and retrieves the
checksum-verified results.  Training budgets, epsilon schedule, and eval
cadence are pinned in the protocol (SPEC P9.protocol); nothing here tunes.
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

EXPERIMENT = "lewm-ddqn-screen-v1"
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
ARCHIVE_LIMIT = 2 * 1024**3
CHECKPOINT_RELPATH = "scale-resume-20000/checkpoint.pt"

TRAINING = {
    "seed": 7,
    "decisions": 200_000,
    "warmup_decisions": 5_000,
    "epsilon_start": 1.0,
    "epsilon_final": 0.05,
    "epsilon_decay_decisions": 120_000,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "sync_interval": 1000,
    "replay_capacity": 100_000,
    "scenario_cohorts": [0, 1],
    "eval_every_decisions": 20_000,
    "max_decisions": 128,
}


def build_protocol(
    *,
    run_id: str,
    checkpoint: Path,
    source_commit: str,
) -> tuple[dict, dict[str, Path]]:
    """Pin the frozen checkpoint and training hyperparameters."""

    if not checkpoint.is_file():
        raise ValueError(f"frozen input is missing: {checkpoint}")
    protocol = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "source_commit": source_commit,
        "inputs": {"checkpoint.pt": digest(checkpoint)},
        "training": dict(TRAINING),
    }
    bundle = {"checkpoint.pt": checkpoint}
    return protocol, bundle


def build_remote_driver(
    *,
    source_hash: str,
    run_id: str,
    wheel: dict | None,
    site_packages: list[str],
) -> str:
    """Generate the remote driver: setup once, smoke, then scored screen."""

    digest_line = (
        "assert hashlib.sha256(target.read_bytes()).hexdigest()"
        f" == '{source_hash}', 'archive mismatch'"
    )
    uv_line = (
        "    uv_ok = subprocess.run([sys.executable, '-m', 'uv', 'pip',"
        " 'install', '--system', '-q', *packages],"
        " capture_output=True).returncode == 0"
    )
    lines = [
        "import hashlib, json, os, shutil, subprocess, sys, tarfile",
        "from pathlib import Path",
        "archive = Path('/content/lewm-ddqn.tar.gz')",
        "assert archive.is_file(), 'missing archive'",
        "target = archive",
        digest_line,
        "print('SOURCE_ARCHIVE_VERIFIED', flush=True)",
        "code = Path('/content/lewm-ddqn-code')",
        "work = Path('/content/lewm-ddqn-work')",
        "inputs = Path('/content/lewm-ddqn-inputs')",
        "for p in (code, work, inputs):",
        "    shutil.rmtree(p, ignore_errors=True)",
        "work.mkdir(exist_ok=False)",
        "with tarfile.open(target) as bundle:",
        "    bundle.extractall('/content', filter='data')",
        "shutil.move('/content/code', code)",
        "shutil.move('/content/inputs', inputs)",
        "protocol = json.loads((code / 'ddqn_protocol.json').read_text())",
        "assert protocol['run_id'] == " + json.dumps(run_id),
        "def run_command(command):",
        "    proc = subprocess.Popen(command, stdout=subprocess.PIPE,",
        "                            stderr=subprocess.STDOUT, text=True)",
        "    for line in proc.stdout:",
        "        print(line, end='', flush=True)",
        "    if proc.wait():",
        "        raise RuntimeError(f'Setup failed: {command[0]}')",
        "subprocess.run([sys.executable, '-m', 'pip', 'install',",
        "                '-q', 'uv'], capture_output=True)",
        f"packages = {site_packages!r}",
        "uv_ok = False",
        "try:",
        uv_line,
        "except (FileNotFoundError, OSError):",
        "    pass",
        "print('UV_SETUP_OK' if uv_ok else 'UV_SETUP_FALLBACK', flush=True)",
        "if not uv_ok:",
        "    run_command([sys.executable, '-m', 'pip', 'install',",
        "                 '-q', *packages])",
        "import transformers",
        "assert transformers.__version__ =="
        f" '{EXPECTED_TRANSFORMERS}', transformers.__version__",
    ]
    if wheel is not None:
        lines += [
            f"wheel = code / 'wheel' / {wheel['filename']!r}",
            "assert hashlib.sha256(wheel.read_bytes()).hexdigest()"
            f" == {wheel['sha256']!r}, 'wheel mismatch'",
            "wheelhouse = Path('/content/wheelhouse')",
            "wheelhouse.mkdir(exist_ok=True)",
            "run_command([sys.executable, '-m', 'pip', 'install', '-q',"
            "             '--no-deps', '--target', str(wheelhouse),"
            "             str(wheel)])",
            "print('WHEEL_CAPTURED', flush=True)",
            "sys.path.insert(0, str(wheelhouse))",
            "native_path = str(wheelhouse)",
        ]
    else:
        lines += [
            "if not shutil.which('cargo'):",
            "    import urllib.request",
            "    urllib.request.urlretrieve('https://sh.rustup.rs',",
            "                               '/content/rustup-init.sh')",
            "    run_command(['sh', '/content/rustup-init.sh', '-y',",
            "                 '--profile', 'minimal'])",
            "    os.environ['PATH'] = str(Path.home() / '.cargo/bin')"
            " + ':' + os.environ['PATH']",
            "crate = str(code / 'native/crates/dodge-python')",
            "if uv_ok:",
            "    run_command([sys.executable, '-m', 'uv', 'pip',",
            "                 'install', '--system', '-q', crate])",
            "else:",
            "    run_command([sys.executable, '-m', 'pip',",
            "                 'install', '-q', crate])",
            "native_path = ''",
        ]
    lines += [
        "print('SETUP_COMPLETE', flush=True)",
        "env = dict(os.environ, PYTHONPATH=str(code / 'src')",
        "           + (os.pathsep + native_path if native_path else ''),",
        "           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')",
        "worker = str(code / 'variants/pixel-repr-ddqn/scripts'"
        "                '/colab_ddqn_worker.py')",
        "base = [sys.executable, worker]",
        "base += ['--run-id', " + json.dumps(run_id) + "]",
        "base += ['--source-hash', '" + source_hash + "']",
        "for mode, marker, timeout in (('smoke', 'DDQN_SMOKE_COMPLETE', 2400),"
        "                             ('scored', 'DDQN_DRIVER_COMPLETE', 7000)):",
        "    log = work / f'ddqn-{mode}.log'",
        "    with log.open('w') as handle:",
        "        proc = subprocess.run([*base, '--mode', mode],",
        "                              env=env, stdout=handle,",
        "                              stderr=subprocess.STDOUT,",
        "                              text=True, timeout=timeout)",
        "    print(f'DDQN {mode} rc={proc.returncode}', flush=True)",
        "    if proc.returncode or marker not in log.read_text():",
        "        print(log.read_text()[-3000:], flush=True)",
        "        raise RuntimeError(f'ddqn {mode} failed; session retained')",
        "print('DDQN_DRIVER_COMPLETE', flush=True)",
        "print('DDQN_LAUNCHER_COMPLETE', flush=True)",
    ]
    return "\n".join(lines) + "\n"


def _publish_results(extracted: Path, run_dir: Path) -> None:
    """Copy the verified results tree into the run dir."""

    for name in ("metrics.jsonl", "eval.json", "environment.json"):
        candidate = extracted / name
        if not candidate.is_file():
            raise RuntimeError(f"ddqn result missing: {name}")
        shutil.copy2(candidate, run_dir / name)
    checkpoints = extracted / "checkpoints"
    if not checkpoints.is_dir():
        raise RuntimeError("ddqn checkpoints missing")
    shutil.copytree(checkpoints, run_dir / "checkpoints")


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
    protocol, bundle = build_protocol(
        run_id=args.run_id,
        checkpoint=checkpoint,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    from colab_survival_study import cached_wheel, native_tree_key

    wheel_path = cached_wheel(native_tree_key())
    wheel: dict | None = None
    if wheel_path is not None:
        wheel = {"filename": wheel_path.name, "sha256": digest(wheel_path)}
    (job / "ddqn_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    raw = job / "source.tar"
    with tarfile.open(raw, "w") as output:
        for name in SOURCE_NAMES:
            output.add(
                ROOT / name,
                arcname=f"code/{name}",
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(
            job / "ddqn_protocol.json",
            arcname="code/ddqn_protocol.json",
        )
        if wheel is not None:
            output.add(wheel_path, arcname=f"code/wheel/{wheel_path.name}")
        for relpath, local in sorted(bundle.items()):
            output.add(local, arcname=f"inputs/{relpath}")
    archive = job / "source.tar.gz"
    with raw.open("rb") as stream, gzip.open(
        archive, "wb", compresslevel=6
    ) as compressed:
        shutil.copyfileobj(stream, compressed, length=8 * 1024**2)
    raw.unlink()
    if archive.stat().st_size > ARCHIVE_LIMIT:
        raise ValueError("source archive exceeds 2 GiB")
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
        remote_prefix="lewm-ddqn",
        assemble_name="lewm-ddqn.tar.gz",
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
    if result.returncode or "DDQN_LAUNCHER_COMPLETE" not in remote_log:
        raise RuntimeError(f"DDQN run failed; session retained: {session}")
    from colab_survival_study import _download, _refresh

    _refresh(session, job)
    _download(session, job, "lewm-ddqn-results.sha256", "results.sha256")
    _download(session, job, "lewm-ddqn-results.tar.gz", "results.tar.gz")
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("retrieved archive checksum mismatch; session retained")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    run_dir = history / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _publish_results(job / "results" / "results", run_dir)


if __name__ == "__main__":
    main()
