"""Launch one bounded T4 chunk of §AD scaled palette training.

Each chunk resumes a verified base checkpoint for ``extra_steps`` more
updates, then retrieves and verifies the new checkpoint plus continuation
manifest.  The local orchestrator chains chunks until the stage target.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from colab_large_probe import ROOT, _validate_run_id, cli, digest, upload

SITE_PACKAGES = [
    "gymnasium>=1",
    "numpy>=2.4.6",
    "transformers==4.57.6",
    "einops==0.8.2",
    "Pillow",
    "pytest",
]

CODE_NAMES = [
    "src",
    "native/Cargo.toml",
    "native/Cargo.lock",
    "native/crates",
    "third_party",
    "variants/pixel-repr-ddqn",
    "tests/variant_pixel_repr_ddqn",
    "references/manifest.json",
    "references/le-wm/module.py",
]


def build_remote_driver(
    *,
    source_hash: str,
    run_id: str,
    site_packages: list[str],
) -> str:
    """Generate the remote driver: setup, verify, fit one chunk."""

    driver = (
        "import os,sys,hashlib,tarfile,subprocess,pathlib\n"
        "from pathlib import Path\n"
        "archive=Path('/content/lewm-source.tar.gz')\n"
        f"assert hashlib.sha256(archive.read_bytes()).hexdigest()=={source_hash!r}\n"
        "print('SOURCE_ARCHIVE_VERIFIED',flush=True)\n"
        "stage=Path('/content/lewm-scale-stage');stage.mkdir(exist_ok=False)\n"
        "with tarfile.open(archive) as bundle: bundle.extractall(stage,filter='data')\n"
        "work=Path('/content/lewm-scale-work');work.mkdir(exist_ok=False)\n"
        "code=work/'code';(stage/'code').rename(code)\n"
        "dataset=work/'dataset';(stage/'dataset').rename(dataset)\n"
        "base=work/'base';(stage/'base').rename(base)\n"
        "(stage/'scale_protocol.json').rename(work/'scale_protocol.json')\n"
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
        "'--system','-q',*packages],capture_output=True).returncode==0\n"
        "except (FileNotFoundError,OSError):\n"
        " pass\n"
        "print('UV_SETUP_OK' if uv_ok else 'UV_SETUP_FALLBACK',flush=True)\n"
        "if not uv_ok:\n"
        " run_command([sys.executable,'-m','pip','install','-q',*packages])\n"
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
        " run_command([sys.executable,'-m','uv','pip','install',\n"
        " '--system','-q',crate])\n"
        "else:\n"
        " run_command([sys.executable,'-m','pip','install','-q',crate])\n"
        "wheelhouse=Path('/content/lewm-scale-wheelhouse');wheelhouse.mkdir(exist_ok=False)\n"
        "run_command([sys.executable,'-m','pip','wheel','--no-deps','-q','-w',str(wheelhouse),crate])\n"
        "wheels=list(wheelhouse.glob('dodge_native-*.whl'))\n"
        "assert len(wheels)==1, f'expected one captured wheel: {wheels}'\n"
        "shutil.copy2(wheels[0],Path('/content/lewm-scale-wheel.whl'))\n"
        "print('WHEEL_CAPTURED',wheels[0].name,flush=True)\n"
        "worker=str(code/'variants/pixel-repr-ddqn/scripts/colab_scale_worker.py')\n"
        "base_env=dict(os.environ,PYTHONPATH=str(code/'src'),"
        f"LEWM_SOURCE_HASH={source_hash!r},LEWM_RUN_ID={run_id!r},"
        "OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')\n"
        "log=Path('/content/scale.log')\n"
        "handle=log.open('w')\n"
        "proc=subprocess.run([sys.executable,worker],"
        "env=base_env,check=False,timeout=7000,stdout=handle,"
        "stderr=subprocess.STDOUT,text=True)\n"
        "handle.close()\n"
        "print('SCALE_RETURNCODE',proc.returncode,flush=True)\n"
        "if proc.returncode:\n"
        " print('SCALE_LOG_TAIL',log.read_text()[-6000:],flush=True)\n"
        " raise RuntimeError('scale chunk failed')\n"
        "print('SCALE_DRIVER_COMPLETE',flush=True)\n"
    )
    return driver


def build_scale_protocol(
    *,
    run_id: str,
    dataset: Path,
    base_checkpoint: Path,
    base_step: int,
    extra_steps: int,
    checkpoint_every: int,
    source_commit: str,
) -> dict[str, object]:
    """Build the frozen chunk protocol plus its file digest map."""

    base_dir = base_checkpoint.parent
    base_files = {
        "base/checkpoint.pt": base_checkpoint,
        "base/sample-trace.json": base_dir / "sample-trace.json",
        "base/stochastic-trace.json": base_dir / "stochastic-trace.json",
    }
    inputs = {"dataset/manifest.json": digest(dataset / "manifest.json")}
    for relpath, local in base_files.items():
        inputs[relpath] = digest(local)
    return {
        "experiment": "lewm-scale-continuation-v1",
        "run_id": run_id,
        "input_arm": "palette",
        "base_step": base_step,
        "extra_steps": extra_steps,
        "checkpoint_every": checkpoint_every,
        "world_batch_size": 32,
        "world_init_seed": 42,
        "world_sampling_seed": 43,
        "world_stochastic_seed": 44,
        "data_sha256": digest(dataset / "manifest.json"),
        "base_checkpoint_sha256": digest(base_checkpoint),
        "inputs": inputs,
        "worker_timeout_seconds": 7200,
        "source_commit": source_commit,
    }


def _base_files(base_checkpoint: Path) -> dict[str, Path]:
    base_dir = base_checkpoint.parent
    return {
        "base/checkpoint.pt": base_checkpoint,
        "base/sample-trace.json": base_dir / "sample-trace.json",
        "base/stochastic-trace.json": base_dir / "stochastic-trace.json",
    }


def _mirror_checkpoint(
    session: str, job: Path, run_id: str, step: int
) -> bool:
    """Mirror one resumable (checkpoint, traces) triple; True when complete."""

    target_dir = job / "mirror-checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    remotes = (
        (f"history/{run_id}/checkpoint-{step}.pt", f"checkpoint-{step}.pt"),
        (f"history/{run_id}/sample-trace.json", f"sample-trace-{step}.json"),
        (
            f"history/{run_id}/stochastic-trace.json",
            f"stochastic-trace-{step}.json",
        ),
    )
    complete = True
    for remote, local in remotes:
        target = target_dir / local
        if target.is_file():
            continue
        temporary = target.with_name(".mirror-" + target.name)
        try:
            result = subprocess.run(
                ["colab", "--auth", "adc", "download",
                 f"/content/lewm-scale-work/{remote}", str(temporary),
                 "--session", session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=600,
            )
            if result.returncode == 0 and temporary.is_file():
                temporary.replace(target)
            else:
                complete = False
        except subprocess.TimeoutExpired:
            complete = False
    return complete


def _newest_mirrored_step(job: Path) -> int:
    """Newest step with a complete mirrored (checkpoint, traces) triple."""

    steps = []
    target_dir = job / "mirror-checkpoints"
    if target_dir.is_dir():
        for path in target_dir.glob("checkpoint-*.pt"):
            try:
                step = int(path.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            if (
                (target_dir / f"sample-trace-{step}.json").is_file()
                and (target_dir / f"stochastic-trace-{step}.json").is_file()
            ):
                steps.append(step)
    return max(steps) if steps else 0


def _mirror_metrics(session: str, history: Path, run_id: str) -> None:
    names = [run_id]
    for name in names:
        for artifact in ("status.json", "metrics.jsonl"):
            target = history / name / artifact
            temporary = target.with_name(".mirror-" + artifact)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = subprocess.run(
                    [
                        "colab",
                        "--auth",
                        "adc",
                        "download",
                        f"/content/lewm-scale-work/history/{name}/{artifact}",
                        str(temporary),
                        "--session",
                        session,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                )
                if result.returncode == 0 and temporary.is_file():
                    temporary.replace(target)
            except subprocess.TimeoutExpired:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-step", type=int, required=True)
    parser.add_argument("--extra-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    dataset = args.dataset.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    base_dir = base_checkpoint.parent
    for name in ("sample-trace.json", "stochastic-trace.json"):
        if not (base_dir / name).is_file():
            parser.error(f"base trace missing: {base_dir / name}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Freeze and commit source before fitting")
    history = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn"
    protocol = build_scale_protocol(
        run_id=args.run_id,
        dataset=dataset,
        base_checkpoint=base_checkpoint,
        base_step=args.base_step,
        extra_steps=args.extra_steps,
        checkpoint_every=args.checkpoint_every,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    base_files = _base_files(base_checkpoint)
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    (job / "scale_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in CODE_NAMES:
            output.add(
                ROOT / name,
                arcname=f"code/{name}",
                filter=lambda item: None if "__pycache__" in item.name else item,
            )
        output.add(dataset, arcname="dataset")
        for relpath, local in base_files.items():
            output.add(local, arcname=relpath)
        output.add(job / "scale_protocol.json", arcname="scale_protocol.json")
    if archive.stat().st_size > 1024**3:
        raise ValueError("source archive exceeds 1 GiB")
    source_hash = digest(archive)
    (job / "source.sha256").write_text(source_hash + "\n")
    remote = job / "remote.py"
    remote.write_text(
        build_remote_driver(
            source_hash=source_hash,
            run_id=args.run_id,
            site_packages=list(SITE_PACKAGES),
        )
    )
    session = f"dodge-{args.run_id}"
    cli("new", "--session", session, "--gpu", "T4")
    (job / "session.json").write_text(json.dumps({"session": session}) + "\n")
    upload(archive, session, job, workers=6, part_size=32 * 1024**2,
           remote_prefix="lewm-scale")
    mirror_root = history / f"{args.run_id}-mirror"
    with (job / "remote.log").open("w") as log:
        process = subprocess.Popen(
            ["colab", "--auth", "adc", "exec", "--session", session,
             "--file", str(remote), "--timeout", "7200"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        polls = 0
        while process.poll() is None:
            _mirror_metrics(session, mirror_root, args.run_id)
            # Mirror the newest intermediate checkpoint triple every ~10
            # minutes so a preempted chunk can resume instead of restarting.
            if polls % 10 == 0:
                status_path = (
                    mirror_root / args.run_id / "status.json"
                )
                try:
                    status_step = int(
                        json.loads(status_path.read_text())["step"]
                    )
                except (OSError, ValueError, KeyError):
                    status_step = args.base_step
                grid = (status_step // args.checkpoint_every) * args.checkpoint_every
                if grid > args.base_step:
                    _mirror_checkpoint(session, job, args.run_id, grid)
            polls += 1
            time.sleep(60)
        _mirror_metrics(session, mirror_root, args.run_id)
        newest = _newest_mirrored_step(job)
        print(f"newest mirrored checkpoint step: {newest}", flush=True)
        remote_log = (job / "remote.log").read_text()
        if process.returncode or "SCALE_DRIVER_COMPLETE" not in remote_log:
            raise RuntimeError(f"Scale chunk failed; session retained: {session}")
    for remote_name, local_name in (
        ("lewm-scale-results.sha256", "results.sha256"),
        ("lewm-scale-results.tar.gz", "results.tar.gz"),
        ("lewm-scale-wheel.whl", "wheel.whl"),
    ):
        cli("download", f"/content/{remote_name}", str(job / local_name),
            "--session", session, timeout=600)
    if digest(job / "results.tar.gz") != (job / "results.sha256").read_text().strip():
        raise RuntimeError("retrieved archive checksum mismatch; session retained")
    with tarfile.open(job / "results.tar.gz") as results:
        results.extractall(job / "results", filter="data")
    shutil.copytree(job / "results/history", history, dirs_exist_ok=True)
    manifest = json.loads(
        (history / f"{args.run_id}-continuation.json").read_text()
    )
    total = args.base_step + args.extra_steps
    if manifest.get("steps") != total:
        raise RuntimeError("continuation manifest step mismatch; session retained")
    if manifest.get("base_checkpoint_sha256") != protocol["base_checkpoint_sha256"]:
        raise RuntimeError("continuation base mismatch; session retained")
    checkpoint = history / args.run_id / "checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError("continuation checkpoint missing; session retained")
    print(f"SCALE_CHUNK_COMPLETE steps={total} checkpoint={checkpoint}")
    print(f"verify metrics before releasing T4: {job}")


if __name__ == "__main__":
    main()
