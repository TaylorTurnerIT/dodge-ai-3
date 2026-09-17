"""Reusable Colab job transport: session, upload, exec, retrieve.

Study launchers keep protocol logic; this module owns the mechanical
flow every remote job repeats:

1. create a GPU session,
2. upload one or more local files (split into parts, parallel upload,
   remote reassembly with SHA256 verification),
3. execute a driver script with a live log,
4. download result files with retries and digest checks.

CLI form runs the whole flow from a prepared job directory::

    python colab_job.py --run-id my-run --gpu T4 \\
        --payload job/source.tar.gz:/content/source.tar.gz \\
        --driver job/remote.py --timeout 7200 \\
        --result /content/results.tar.gz:job/results.tar.gz \\
        --result /content/results.sha256:job/results.sha256 \\
        --complete-marker JOB_COMPLETE
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from colab_large_probe import ROOT, _validate_run_id, cli, digest


def split_upload(
    local: Path,
    remote: str,
    session: str,
    job: Path,
    *,
    workers: int = 6,
    part_size: int = 32 * 1024**2,
    prefix: str | None = None,
) -> None:
    """Upload one file in parallel parts; reassemble remotely verified."""

    local = Path(local)
    data = local.read_bytes()
    tag = prefix or local.stem
    parts = [data[i : i + part_size] for i in range(0, len(data), part_size)]
    if not parts:
        raise ValueError(f"payload is empty: {local}")
    payloads = []
    for index, block in enumerate(parts):
        part_path = job / f"{tag}.upload-{index:03d}"
        part_path.write_bytes(block)
        payloads.append((str(part_path), f"/content/{tag}.part-{index:03d}"))

    def one(args: tuple[str, str]) -> None:
        local_part, remote_part = args
        cli("upload", local_part, remote_part, "--session", session)
        Path(local_part).unlink()

    if workers < 2:
        for payload in payloads:
            one(payload)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, payloads))
    expected = hashlib.sha256(data).hexdigest()
    assembly = job / f"{tag}-assemble.py"
    assembly.write_text(
        "from pathlib import Path\nimport hashlib\n"
        f"target=Path({remote!r})\n"
        "with target.open('wb') as output:\n"
        f" for i in range({len(parts)}):\n"
        f"  part=Path(f'/content/{tag}.part-{{i:03d}}')\n"
        "  output.write(part.read_bytes())\n  part.unlink()\n"
        f"assert hashlib.sha256(target.read_bytes()).hexdigest()=={expected!r}\n"
        "print('ASSEMBLED', target.stat().st_size, flush=True)\n"
    )
    cli("exec", "--session", session, "--file", str(assembly), "--timeout", 300)


def run_driver(
    session: str,
    driver: Path,
    log: Path,
    *,
    timeout: int,
    complete_marker: str,
) -> None:
    """Execute the driver file; raise unless it exits 0 with the marker."""

    with log.open("w") as handle:
        result = subprocess.run(
            ["colab", "--auth", "adc", "exec", "--session", session,
             "--file", str(driver), "--timeout", str(timeout)],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    text = log.read_text()
    if result.returncode or complete_marker not in text:
        raise RuntimeError(f"Remote driver failed; session retained: {session}")


def retrieve(
    session: str,
    job: Path,
    pairs: list[tuple[str, str]],
    *,
    attempts: int = 4,
    pause: int = 60,
) -> None:
    """Download (remote, local) pairs with retries."""

    for remote, local in pairs:
        last: Exception | None = None
        for _ in range(attempts):
            try:
                cli("download", remote, str(job / local),
                    "--session", session, timeout=600)
                break
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                last = error
                time.sleep(pause)
        else:
            raise RuntimeError(f"retrieval failed: {remote}") from last


def verify_digest(job: Path, archive: str, sha_file: str) -> None:
    """Check a retrieved archive against its retrieved SHA file."""

    expected = (job / sha_file).read_text().strip()
    if digest(job / archive) != expected:
        raise RuntimeError(f"checksum mismatch: {archive}")


def _pair(value: str) -> tuple[str, str]:
    left, _, right = value.partition(":")
    if not left or not right:
        raise ValueError(f"expected REMOTE:LOCAL, got {value!r}")
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--payload", action="append", default=[],
                        help="LOCAL:REMOTE, uploaded with split/verify")
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--result", action="append", default=[],
                        help="REMOTE:LOCAL, downloaded with retries")
    parser.add_argument("--complete-marker", required=True)
    parser.add_argument("--digest", nargs=2, default=None,
                        metavar=("ARCHIVE", "SHA_FILE"),
                        help="verify job-dir archive against sha file")
    args = parser.parse_args()
    _validate_run_id(args.run_id, "run ID")
    job = args.job_dir
    job.mkdir(parents=True, exist_ok=True)
    session = f"dodge-{args.run_id}"
    cli("new", "--session", session, "--gpu", args.gpu)
    (job / "session.json").write_text(
        __import__("json").dumps({"session": session}) + "\n"
    )
    for local, remote in (_pair(p) for p in args.payload):
        split_upload(Path(local), remote, session, job)
    run_driver(session, args.driver, job / "remote.log",
               timeout=args.timeout, complete_marker=args.complete_marker)
    retrieve(session, job, [_pair(p) for p in args.result])
    if args.digest is not None:
        verify_digest(job, args.digest[0], args.digest[1])
    print(f"JOB_COMPLETE results in {job}")


if __name__ == "__main__":
    main()
