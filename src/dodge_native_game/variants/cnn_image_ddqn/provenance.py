"""Small, failure-tolerant provenance helpers for DDQN artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TypedDict


class SourceProvenance(TypedDict):
    git_commit: str | None
    git_dirty: bool | None


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the exact checkpoint bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_source_provenance(project_root: Path) -> SourceProvenance:
    """Read Git revision and tracked dirty state without making Git mandatory."""

    root = Path(project_root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    if revision.returncode != 0 or status.returncode != 0:
        return {"git_commit": None, "git_dirty": None}
    commit = revision.stdout.strip()
    if not commit:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.stdout.strip())}


def infer_parent_run_id(path: Path, checkpoint_run_id: object) -> str | None:
    """Prefer checkpoint metadata, then infer a conventional run directory."""

    if isinstance(checkpoint_run_id, str) and checkpoint_run_id:
        return checkpoint_run_id
    candidate = Path(path)
    if candidate.parent.name == "checkpoints" and candidate.parent.parent.name:
        return candidate.parent.parent.name
    return None


__all__ = [
    "SourceProvenance",
    "git_source_provenance",
    "infer_parent_run_id",
    "sha256_file",
]
