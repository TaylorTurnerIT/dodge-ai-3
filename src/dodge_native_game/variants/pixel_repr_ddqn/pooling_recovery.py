"""Hash-gated recovery of missing frozen bank bytes.

The pooling study consumes the exact §Z standard/spatial banks as read-only
inputs.  When Phase-0 verification confirms specific array files are missing
locally, this module re-creates ONLY those files with the frozen extraction
code, verifies every byte against the full recorded SHA-256 digests in a
staging directory, and promotes them into the immutable input root solely on
exact match.  Any mismatch stops the run before fitting: no tolerance, no
replacement hashes, no silent substitution.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from .run_artifacts import file_hash

__all__ = [
    "missing_files",
    "verify_files",
    "promote_verified",
    "RECOVERY_EXTRACTORS",
]

# Frozen §Z extraction parameters.  Recovery must reproduce the original
# bytes, so these mirror spatial_probe/large_probe exactly.
FRAMES_PER_EPISODE = 4
SAMPLING_SEED = 903
ENCODE_BATCH_SIZE = 32
EXTRACT_BATCH_SIZE = 32

# Which staged builder reproduces each missing array.  Keys are bank-relative
# paths of the form "<kind>/<split>/<array>.npy".
RECOVERY_EXTRACTORS = {
    "standard/train/pixels.npy": "standard",
    "standard/train/changed.npy": "standard",
    "standard/train/cls.npy": "standard",
    "standard/train/projected.npy": "standard",
    "standard/validation/pixels.npy": "standard",
    "standard/validation/changed.npy": "standard",
    "standard/validation/projected.npy": "standard",
    "spatial/train/patches.npy": "spatial",
}


def missing_files(root: Path, expectations: Mapping[str, str]) -> dict[str, str]:
    """Return the expected relpaths absent under ``root`` with their digests."""

    root = Path(root)
    return {
        relpath: digest
        for relpath, digest in expectations.items()
        if not (root / relpath).is_file()
    }


def verify_files(root: Path, expectations: Mapping[str, str]) -> None:
    """Require every expected file present with its exact recorded digest."""

    root = Path(root)
    problems = []
    for relpath in sorted(expectations):
        path = root / relpath
        if not path.is_file():
            problems.append(f"missing: {relpath}")
        elif file_hash(path) != expectations[relpath]:
            problems.append(f"digest mismatch: {relpath}")
    if problems:
        raise ValueError(
            "frozen input verification failed: " + "; ".join(problems)
        )


def promote_verified(
    staging: Path, dest: Path, expectations: Mapping[str, str]
) -> list[str]:
    """Verify staged files, then copy them into the immutable input root.

    Every staged file must match its full recorded digest.  Destination paths
    must not exist yet: recovered bytes are appended once, never overwritten.
    Returns the promoted relpaths in sorted order.
    """

    staging, dest = Path(staging), Path(dest)
    verify_files(staging, expectations)
    promoted = []
    for relpath in sorted(expectations):
        target = dest / relpath
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable input: {relpath}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging / relpath, target)
        if file_hash(target) != expectations[relpath]:
            raise ValueError(f"promoted file failed re-verification: {relpath}")
        promoted.append(relpath)
    return promoted
