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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,70}", args.run_id):
        raise ValueError("invalid run ID")
    job = ROOT / "history/dodge/gymnasium/pixel-repr-ddqn-jobs" / args.run_id
    job.mkdir(parents=True, exist_ok=False)
    protocol = {
        "checkpoint_sha256": digest(args.checkpoint),
        "data_hash": digest(args.dataset / "manifest.json"),
        "milestones": [512, 2048, 8192],
        "batch_size": 32,
        "frames_per_episode": 4,
        "initialization_seed": 904,
        "sampling_seed": 903,
        "world_model_updates": 0,
        "output_size": 128,
        "representations": ["cls", "projected"],
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
    names = [f"{args.run_id}-{mode}" for mode in ("cls", "projected")]
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
    cli("stop", "--session", session)
    print(f"Artifacts retrieved and T4 released: {job}", flush=True)


if __name__ == "__main__":
    main()
