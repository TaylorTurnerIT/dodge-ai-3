from __future__ import annotations

import subprocess
from pathlib import Path

from dodge_native_game.variants.cnn_image_ddqn.provenance import (
    git_source_provenance,
    infer_parent_run_id,
    sha256_file,
)


def test_checkpoint_hash_uses_exact_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"ddqn")

    assert sha256_file(checkpoint) == (
        "e84ee784c93e1cbc120be65541f9c6106061d322d9a2ef196d839f7bdd2088dd"
    )


def test_parent_run_id_prefers_payload_then_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent-run" / "checkpoints" / "step-10.pt"

    assert infer_parent_run_id(checkpoint, "payload-run") == "payload-run"
    assert infer_parent_run_id(checkpoint, None) == "parent-run"
    assert infer_parent_run_id(tmp_path / "resume.pt", None) is None


def test_git_provenance_is_explicit_when_git_is_unavailable(
    monkeypatch,
) -> None:
    def unavailable(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", unavailable)

    assert git_source_provenance(Path(".")) == {
        "git_commit": None,
        "git_dirty": None,
    }
