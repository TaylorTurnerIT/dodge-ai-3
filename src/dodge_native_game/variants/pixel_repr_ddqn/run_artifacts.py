"""Small, atomic, variant-owned artifacts for the LeWM MVP."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_identity() -> dict[str, str]:
    """Hash actual source files too, so uncommitted MVP code has an identity."""
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".html"}:
            digest.update(str(path.relative_to(package)).encode())
            digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {"revision": revision, "source_sha256": digest.hexdigest()}


def create_run(root: Path, run_id: str, manifest: dict[str, Any]) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("run_id must be a simple alphanumeric identifier")
    root.mkdir(parents=True, exist_ok=True)
    path = root / run_id
    path.mkdir(exist_ok=False)
    atomic_json(
        path / "manifest.json",
        {
            **manifest,
            "run_id": run_id,
            "created_at": utc_now(),
            **source_identity(),
        },
    )
    return path


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    payload = json.dumps(metric, allow_nan=False)
    with path.open("a") as stream:
        stream.write(payload + "\n")
        stream.flush()


def write_status(
    path: Path, *, state: str, phase: str, step: int, total_steps: int, message: str
) -> None:
    atomic_json(
        path / "status.json",
        {
            "state": state,
            "phase": phase,
            "step": step,
            "total_steps": total_steps,
            "message": message,
            "updated_at": utc_now(),
        },
    )
