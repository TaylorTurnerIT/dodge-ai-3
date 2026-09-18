"""Run the AD4 Track-A dynamics audit for arbitrary checkpoints on one T4.

Uploads frozen checkpoints plus the frozen audit dataset with the frozen
source, runs run_dynamics_audit per checkpoint and split on CUDA, and
retrieves the per-file audit JSONs.  Procedure (splits, fractions,
window counts) matches lewm-audit-20260917-v1; only the inputs vary.
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

EXPERIMENT = "lewm-dynamics-audit-v1"
SPLITS = ("train", "validation")
FRACTIONS = (0.25, 0.5, 0.75)
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


def build_protocol(
    *,
    run_id: str,
    checkpoints: dict[str, Path],
    dataset: Path,
    source_commit: str,
) -> tuple[dict, dict[str, Path]]:
    """Pin the frozen checkpoints/dataset; map bundle relpaths."""

    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    manifest = dataset / "manifest.json"
    for path in list(checkpoints.values()) + [manifest]:
        if not path.is_file():
            raise ValueError(f"frozen input is missing: {path}")
    protocol = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "source_commit": source_commit,
        "checkpoints": {
            step: digest(path) for step, path in sorted(checkpoints.items())
        },
        "data_hash": digest(manifest),
        "splits": list(SPLITS),
        "fractions": list(FRACTIONS),
    }
    bundle = {
        f"checkpoints/{step}.pt": path
        for step, path in sorted(checkpoints.items())
    }
    bundle["dataset/manifest.json"] = manifest
    return protocol, bundle


def build_remote_driver(
    *,
    source_hash: str,
    run_id: str,
    wheel: dict | None,
    site_packages: list[str],
    checkpoints: list[str],
) -> str:
    """Generate the remote driver: setup once, audit each ckpt and split."""

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
        "archive = Path('/content/lewm-audit.tar.gz')",
        "assert archive.is_file(), 'missing archive'",
        "target = archive",
        digest_line,
        "print('SOURCE_ARCHIVE_VERIFIED', flush=True)",
        "code = Path('/content/lewm-audit-code')",
        "work = Path('/content/lewm-audit-work')",
        "for p in (code, work):",
        "    shutil.rmtree(p, ignore_errors=True)",
        "work.mkdir(exist_ok=False)",
        "with tarfile.open(target) as bundle:",
        "    bundle.extractall('/content', filter='data')",
        "shutil.move('/content/code', code)",
        "shutil.move('/content/inputs', work / 'inputs')",
        "protocol = json.loads((code / 'audit_protocol.json').read_text())",
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
        ]
    audit_call = (
        "            proc = subprocess.run([sys.executable, runner,"
        " '--dataset', str(work / 'inputs' / 'dataset'),"
        " '--checkpoint', str(member), '--output', str(out),"
        " '--device', 'cuda', '--split', split, '--fractions',"
        " *(str(v) for v in protocol['fractions'])],"
        " env=env, stdout=handle, stderr=subprocess.STDOUT,"
        " text=True, timeout=3600)"
    )
    lines += [
        "print('SETUP_COMPLETE', flush=True)",
        "env = dict(os.environ, PYTHONPATH=str(code / 'src'),",
        "           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')",
        "runner = str(code / 'variants/pixel-repr-ddqn/scripts'",
        "                '/run_dynamics_audit.py')",
        "results = work / 'results'; results.mkdir(exist_ok=False)",
        f"CKPTS = {checkpoints!r}",
        "for step in CKPTS:",
        "    member = work / 'inputs' / 'checkpoints' / f'{step}.pt'",
        "    got = hashlib.sha256(member.read_bytes()).hexdigest()",
        "    want = protocol['checkpoints'][step]",
        "    assert got == want, f'checkpoint mismatch: {step}'",
        "    for split in protocol['splits']:",
        "        out = results / f'audit-ckpt-{step}-{split}.json'",
        "        log = results / f'audit-ckpt-{step}-{split}.log'",
        "        with log.open('w') as handle:",
        audit_call,
        "        print(f'AUDIT ckpt-{step} {split}"
        " rc={proc.returncode}', flush=True)",
        "        if proc.returncode:",
        "            print(log.read_text()[-3000:], flush=True)",
        "            raise RuntimeError(f'audit failed: {step} {split};"
        " session retained')",
        "archive = Path('/content/lewm-audit-results.tar.gz')",
        "with tarfile.open(archive, 'w:gz') as output:",
        "    output.add(results, arcname='results')",
        "digest = hashlib.sha256(archive.read_bytes()).hexdigest()",
        "Path('/content/lewm-audit-results.sha256').write_text(digest + '\\n')",
        "print('AUDIT_DRIVER_COMPLETE', flush=True)",
        "print('AUDIT_LAUNCHER_COMPLETE', flush=True)",
    ]
    return "\n".join(lines) + "\n"


def _publish_results(extracted: Path, run_dir: Path, steps: list[str]) -> None:
    """Copy the verified per-ckpt audit JSONs into the run dir."""

    for step in steps:
        for split in SPLITS:
            name = f"audit-ckpt-{step}-{split}.json"
            candidate = extracted / name
            if not candidate.is_file():
                raise RuntimeError(f"audit result missing: {name}")
            shutil.copy2(candidate, run_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="STEP=PATH",
        help="repeatable STEP=PATH frozen checkpoint",
    )
    default_dataset = (
        ROOT / "history/dodge/gymnasium/pixel-repr-ddqn/large-practice-20260914-v2"
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    checkpoints: dict[str, Path] = {}
    for item in args.checkpoint:
        if "=" not in item:
            parser.error(f"--checkpoint must look like STEP=PATH, got {item!r}")
        step, _, raw = item.partition("=")
        if not step.isdigit() or step in checkpoints:
            parser.error(f"bad or repeated checkpoint step: {step!r}")
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        checkpoints[step] = path
    dataset = args.dataset
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Freeze and commit source before fitting")
    protocol, bundle = build_protocol(
        run_id=args.run_id,
        checkpoints=checkpoints,
        dataset=dataset,
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
    (job / "audit_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    raw = job / "source.tar"
    with tarfile.open(raw, "w") as output:
        for name in SOURCE_NAMES:
            output.add(
                ROOT / name,
                arcname=f"code/{name}",
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(
            job / "audit_protocol.json",
            arcname="code/audit_protocol.json",
        )
        if wheel is not None:
            output.add(wheel_path, arcname=f"code/wheel/{wheel_path.name}")
        for relpath, local in sorted(bundle.items()):
            output.add(local, arcname=f"inputs/{relpath}")
        output.add(dataset, arcname="inputs/dataset")
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
            checkpoints=sorted(checkpoints),
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
        remote_prefix="lewm-audit",
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
    if result.returncode or "AUDIT_LAUNCHER_COMPLETE" not in remote_log:
        raise RuntimeError(f"Audit run failed; session retained: {session}")
    from colab_survival_study import _download, _refresh

    _refresh(session, job)
    _download(session, job, "lewm-audit-results.sha256", "results.sha256")
    _download(session, job, "lewm-audit-results.tar.gz", "results.tar.gz")
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("retrieved archive checksum mismatch; session retained")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    run_dir = history / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _publish_results(job / "results" / "results", run_dir, sorted(checkpoints))


if __name__ == "__main__":
    main()
