"""Hash-gated recovery of missing frozen bank bytes.

The pooling study consumes the exact §Z standard/spatial banks as read-only
inputs.  When Phase-0 verification confirms specific array files are missing
locally, the worker re-creates ONLY those files with the frozen extraction
code and verifies every byte against the full recorded SHA-256 digests in
place.  Any mismatch stops the run before fitting: no tolerance, no
replacement hashes, no silent substitution.

The builders publish via ``os.replace``, which fails on FUSE mounts when
the destination exists, so recovery relocates a split's recorded
scaffolding (metadata/index/READY) into a staging mirror first; the
regenerated scaffolding must then be byte-identical to the relocated
originals before the recovered arrays verify.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .run_artifacts import file_hash

__all__ = [
    "missing_files",
    "verify_files",
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
