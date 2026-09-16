from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dodge_native_game.variants.pixel_repr_ddqn.pooling_recovery import (
    RECOVERY_EXTRACTORS,
    missing_files,
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


def test_worker_spatial_recovery_matches_extract_patches_signature() -> None:
    import inspect

    from dodge_native_game.variants.pixel_repr_ddqn.spatial_bank import (
        extract_patches,
    )

    worker = Path("variants/pixel-repr-ddqn/scripts/colab_pooling_worker.py")
    source = worker.read_text()
    call = source.index("extract_patches(")
    snippet = source[call : call + 220]
    params = list(inspect.signature(extract_patches).parameters)
    assert params[:3] == ["model", "bank", "root"]
    for name in ("model", "getattr(bank, split)", "target"):
        assert name in snippet


def test_worker_recovery_scopes_to_absent_targets() -> None:
    worker = Path("variants/pixel-repr-ddqn/scripts/colab_pooling_worker.py")
    source = worker.read_text()
    block = source[source.index("def _recover_missing") :]
    assert "if not (inputs / key).exists()" in block.split("dataset =")[0]
