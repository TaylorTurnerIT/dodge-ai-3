from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn import large_probe
from dodge_native_game.variants.pixel_repr_ddqn.large_probe import (
    FrameBank,
    evaluate_decoder_stream,
    fit_matched_decoders,
    make_decoder_pair,
    stream_train_mean,
)


class _TinyDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, 3 * 4 * 4)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.projection(latent)).reshape(-1, 3, 4, 4)


class _NativeTinyDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.latent_dim = latent_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = torch.sigmoid(latent[:, :1] + self.bias)
        return value[:, :, None, None].expand(-1, 3, 128, 128)


class _IdentityDecoder(nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.reshape(-1, 3, 4, 4)


class _Rows:
    """Array-like rows that make the source passed to fitting observable."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.accesses: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: object) -> np.ndarray:
        self.accesses.append(np.asarray(key).copy())
        return self.values[key]  # type: ignore[index]


def _records(count: int, split: str, offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "split": split,
            "index": offset + index,
            "episode_id": f"{split}-episode-{index}",
            "frame_index": index % 4 + 1,
            "recipe_family": f"family-{index}",
        }
        for index in range(count)
    ]


def test_matched_fit_uses_identical_initialization_and_sampling() -> None:
    torch.manual_seed(71)
    cls_decoder, projected_decoder = make_decoder_pair(4, decoder_factory=_TinyDecoder)
    for left, right in zip(
        cls_decoder.state_dict().values(),
        projected_decoder.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    cls_values = _Rows(np.random.default_rng(1).normal(size=(8, 4)).astype(np.float32))
    projected_values = _Rows(
        np.random.default_rng(2).normal(size=(8, 4)).astype(np.float32)
    )
    targets = np.random.default_rng(3).integers(
        0, 256, size=(8, 3, 4, 4), dtype=np.uint8
    )
    milestones: list[int] = []
    result = fit_matched_decoders(
        cls_decoder,
        projected_decoder,
        cls_values,
        projected_values,
        targets,
        milestones=(1, 3),
        batch_size=4,
        sampler=torch.Generator().manual_seed(903),
        on_milestone=lambda snapshot, *_: milestones.append(snapshot.step),
    )

    assert milestones == [1, 3]
    assert set(result.snapshots) == {1, 3}
    assert len(result.metrics) == 3
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            cls_values.accesses, projected_values.accesses, strict=True
        )
    )
    expected = torch.Generator().manual_seed(903)
    expected_indices = [
        tuple(torch.randint(8, (4,), generator=expected).tolist()) for _ in range(3)
    ]
    assert result.sampled_indices == tuple(expected_indices)
    assert result.snapshots[3].optimizer["cls"]["state"]


def test_train_mean_streams_normalized_rows_and_evaluation_uses_bank_masks() -> None:
    pixels = np.zeros((16, 3, 4, 4), dtype=np.uint8)
    pixels[:8] = np.arange(8, dtype=np.uint8)[:, None, None, None]
    mean = stream_train_mean(pixels, range(0, 8), batch_size=3)
    np.testing.assert_allclose(mean, (3.5 / 255.0), rtol=0, atol=1e-7)

    features = pixels.astype(np.float32).reshape(16, -1) / 255.0
    changed = np.zeros((16, 4, 4), dtype=bool)
    changed[8, 1, 2] = True
    records = _records(8, "train") + _records(8, "validation", offset=8)
    identity = _IdentityDecoder()
    result = evaluate_decoder_stream(
        identity,
        features,
        pixels,
        changed,
        records,
        {"train": (0, 8), "validation": (8, 16)},
        np.zeros((3, 4, 4), dtype=np.float32),
        device="cpu",
        batch_size=5,
    )
    assert result["splits"]["train"]["mse"] == pytest.approx(0.0)
    assert result["splits"]["train"]["changed_region_mse"] is None
    assert result["splits"]["validation"]["changed_pixel_count"] == 1
    assert result["splits"]["validation"]["changed_region_mse"] == pytest.approx(0.0)

    wrong = evaluate_decoder_stream(
        identity,
        features,
        pixels,
        changed,
        records,
        {"train": (0, 8), "validation": (8, 16)},
        np.zeros((3, 4, 4), dtype=np.float32),
        device="cpu",
        batch_size=5,
        wrong_permutation=True,
    )
    first = next(item for item in wrong["examples"] if item["index"] == 0)
    np.testing.assert_allclose(first["reconstructed"], pixels[4] / 255.0)


def _fake_frame_bank(train_rows: int = 32, validation_rows: int = 16) -> FrameBank:
    rng = np.random.default_rng(22)
    pixels_train = rng.integers(0, 256, size=(train_rows, 3, 128, 128), dtype=np.uint8)
    pixels_validation = rng.integers(
        0, 256, size=(validation_rows, 3, 128, 128), dtype=np.uint8
    )
    changed_train = np.zeros((train_rows, 128, 128), dtype=bool)
    changed_validation = np.zeros((validation_rows, 128, 128), dtype=bool)
    changed_validation[:, 0, 0] = True
    cls_train = rng.normal(size=(train_rows, 4)).astype(np.float32)
    cls_validation = rng.normal(size=(validation_rows, 4)).astype(np.float32)
    projected_train = rng.normal(size=(train_rows, 4)).astype(np.float32)
    projected_validation = rng.normal(size=(validation_rows, 4)).astype(np.float32)
    train_cls = _Rows(cls_train)
    train_projected = _Rows(projected_train)
    train_pixels = _Rows(pixels_train)
    train = SimpleNamespace(
        cls=train_cls,
        projected=train_projected,
        pixels=train_pixels,
    )
    validation = SimpleNamespace(
        cls=cls_validation,
        projected=projected_validation,
        pixels=pixels_validation,
    )
    records = _records(train_rows, "train") + _records(
        validation_rows, "validation", offset=train_rows
    )
    return FrameBank(
        root=Path("/fake-bank"),
        train=train,  # type: ignore[arg-type]
        validation=validation,  # type: ignore[arg-type]
        pixels=large_probe._Concat((pixels_train, pixels_validation)),
        changed=large_probe._Concat((changed_train, changed_validation)),
        features={
            "cls": large_probe._Concat((cls_train, cls_validation)),
            "projected": large_probe._Concat((projected_train, projected_validation)),
        },
        records=tuple(records),
        split_ranges={
            "train": (0, train_rows),
            "validation": (train_rows, train_rows + validation_rows),
        },
        data_hash="d" * 64,
        frame_index_hash="f" * 64,
    )


def test_run_study_fits_train_rows_and_publishes_each_milestone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _fake_frame_bank()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-world-model")
    frozen = nn.Linear(1, 1)
    before = {key: value.detach().clone() for key, value in frozen.state_dict().items()}

    monkeypatch.setattr(
        large_probe.pretrain,
        "load_model",
        lambda path: (
            frozen,
            {
                "inference_only": True,
                "experiment": "practice-batch32-v1",
                "step": 512,
                "batch_size": 32,
                "profile": "reference",
                "calibration": {},
                "data_hash": "dataset",
            },
        ),
    )
    monkeypatch.setattr(large_probe, "_validate_protocol", lambda *_: "NVIDIA T4")
    monkeypatch.setattr(large_probe, "_ensure_bank", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(large_probe, "open_frame_bank", lambda *_args, **_kwargs: bank)
    monkeypatch.setattr(
        large_probe,
        "make_decoder_pair",
        lambda *_args, **_kwargs: (
            _NativeTinyDecoder(4),
            _NativeTinyDecoder(4),
        ),
    )
    save_calls: list[str] = []
    save_checkpoint = large_probe.pretrain.save_checkpoint

    def observe_checkpoint(path: Path, payload: dict[str, object]) -> None:
        if path.name == "decoder-2.pt":
            save_calls.append(path.parent.name)
            previous = json.loads((path.parent / "visualizations.json").read_text())
            assert previous[0]["step"] == 1
        save_checkpoint(path, payload)

    monkeypatch.setattr(large_probe.pretrain, "save_checkpoint", observe_checkpoint)

    runs = large_probe.run_study(
        checkpoint,
        tmp_path / "dataset",
        tmp_path / "history",
        "tiny-large-probe",
        milestones=(1, 2),
        device="cpu",
        bank_root=tmp_path / "bank",
    )

    assert {path.name for path in runs} == {
        "tiny-large-probe-cls",
        "tiny-large-probe-projected",
    }
    assert bank.train.cls.accesses
    assert bank.train.projected.accesses
    assert bank.train.pixels.accesses
    assert all(int(indices.max()) < 32 for indices in bank.train.cls.accesses)
    assert all(int(indices.max()) < 32 for indices in bank.train.projected.accesses)
    assert all(int(indices.max()) < 32 for indices in bank.train.pixels.accesses)
    assert sorted(save_calls) == [
        "tiny-large-probe-cls",
        "tiny-large-probe-projected",
    ]
    for run in runs:
        assert (run / "decoder-1.pt").is_file()
        assert (run / "decoder-2.pt").is_file()
        visualizations = json.loads((run / "visualizations.json").read_text())
        assert len(visualizations) == 16
        assert all(view["metadata"]["current_frame_only"] for view in visualizations)
        rows = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        assert {row["step"] for row in rows if row["phase"].endswith("evaluation")} == {
            1,
            2,
        }
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
