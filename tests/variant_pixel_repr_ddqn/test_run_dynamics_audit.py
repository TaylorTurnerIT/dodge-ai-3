from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from dodge_native_game.variants.pixel_repr_ddqn.model import (
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
)
from dodge_native_game.variants.pixel_repr_ddqn.run_artifacts import file_hash


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset_module():
    return _load_module(
        "audit_dataset_fixture",
        Path(__file__).with_name("test_large_dataset.py"),
    )


def _runner_module():
    return _load_module(
        "dynamics_audit_runner",
        Path(__file__).resolve().parents[2]
        / "variants/pixel-repr-ddqn/scripts/run_dynamics_audit.py",
    )


class _StubWorld(torch.nn.Module):
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, time = pixels.shape[:2]
        base = pixels.float().mean(dim=(2, 3, 4))
        return base.unsqueeze(-1).expand(batch, time, 1).contiguous()

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return z + actions.to(z.dtype).unsqueeze(-1)


def test_audit_runner_writes_provenance_report(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    fixture = _dataset_module()
    fixture._make_dataset(dataset, train_count=1, validation_count=2)
    payload = {
        "input_arm": "palette",
        "step": 1024,
        "input_encoding": INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
        "data_hash": file_hash(dataset / "manifest.json"),
        "config": {"palette_rgb": [[0, 0, 0]], "history_size": 3},
    }
    checkpoint = tmp_path / "world.pt"
    torch.save(payload, checkpoint)
    world_hash = file_hash(checkpoint)

    monkeypatch.setattr(
        "dodge_native_game.variants.pixel_repr_ddqn.pretrain.load_model",
        lambda path: (_StubWorld(), {}),
    )

    runner = _runner_module()
    output = tmp_path / "audit.json"
    report = runner.main(
        [
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--split",
            "validation",
            "--fractions",
            "0.5",
        ]
    )

    assert report["experiment"] == "lewm-dynamics-audit-v1"
    assert report["world_model_sha256"] == world_hash
    assert report["world_model_updates"] == 0
    assert report["split"] == "validation"
    assert report["fractions"] == [0.5]
    assert report["window_count"] == 2
    assert report["scope"] == "one-step teacher-forced action audit"
    assert [row["episode_id"] for row in report["windows"]] == [
        "validation-000000",
        "validation-000001",
    ]
    assert json.loads(output.read_text())["window_count"] == 2
    assert file_hash(checkpoint) == world_hash


def test_audit_runner_rejects_bad_fractions(tmp_path: Path) -> None:
    import pytest

    runner = _runner_module()
    with pytest.raises(ValueError, match="fractions"):
        runner.main(
            [
                "--dataset",
                str(tmp_path),
                "--checkpoint",
                str(tmp_path),
                "--output",
                str(tmp_path / "out.json"),
                "--fractions",
                "1.5",
            ]
        )
