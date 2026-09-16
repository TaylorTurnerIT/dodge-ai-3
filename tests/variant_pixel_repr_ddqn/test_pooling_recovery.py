from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dodge_native_game.variants.pixel_repr_ddqn.pooling_recovery import (
    RECOVERY_EXTRACTORS,
    missing_files,
    promote_verified,
    verify_files,
)


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_missing_files_reports_only_absent_relpaths(tmp_path: Path) -> None:
    present_digest = _write(tmp_path / "a.bin", b"present")
    expectations = {"a.bin": present_digest, "b.bin": "0" * 64}
    assert missing_files(tmp_path, expectations) == {"b.bin": "0" * 64}


def test_verify_files_accepts_exact_digests(tmp_path: Path) -> None:
    digest = _write(tmp_path / "a.bin", b"exact")
    verify_files(tmp_path, {"a.bin": digest})


def test_verify_files_rejects_missing_and_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "a.bin", b"changed")
    with pytest.raises(ValueError, match="digest mismatch: a.bin"):
        verify_files(tmp_path, {"a.bin": "0" * 64})
    with pytest.raises(ValueError, match="missing: gone.bin"):
        verify_files(tmp_path, {"gone.bin": "1" * 64})


def test_promote_verified_copies_and_reverifies(tmp_path: Path) -> None:
    staging, dest = tmp_path / "staging", tmp_path / "inputs"
    digest = _write(staging / "standard/train/pixels.npy", b"pixels")
    promoted = promote_verified(staging, dest, {"standard/train/pixels.npy": digest})
    assert promoted == ["standard/train/pixels.npy"]
    assert (dest / "standard/train/pixels.npy").read_bytes() == b"pixels"


def test_promote_verified_refuses_mismatch_and_overwrite(tmp_path: Path) -> None:
    staging, dest = tmp_path / "staging", tmp_path / "inputs"
    _write(staging / "a.bin", b"bytes")
    with pytest.raises(ValueError, match="digest mismatch: a.bin"):
        promote_verified(staging, dest, {"a.bin": "0" * 64})
    assert not (dest / "a.bin").exists()
    digest = _write(staging / "b.bin", b"one")
    _write(dest / "b.bin", b"other")
    with pytest.raises(FileExistsError, match="immutable input"):
        promote_verified(staging, dest, {"b.bin": digest})


def test_recovery_extractor_map_covers_planned_targets() -> None:
    planned = {
        "spatial-banks/standard/train/pixels.npy",
        "spatial-banks/standard/train/changed.npy",
        "spatial-banks/standard/train/cls.npy",
        "spatial-banks/standard/train/projected.npy",
        "spatial-banks/standard/validation/pixels.npy",
        "spatial-banks/standard/validation/changed.npy",
        "spatial-banks/standard/validation/projected.npy",
        "spatial-banks/spatial/train/patches.npy",
    }
    bank_relative = {
        relpath.removeprefix("spatial-banks/") for relpath in planned
    }
    assert bank_relative <= set(RECOVERY_EXTRACTORS)
