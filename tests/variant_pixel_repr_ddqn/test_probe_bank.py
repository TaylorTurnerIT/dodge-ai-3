from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn.large_dataset import (
    LARGE_DATASET_FORMAT,
    _read_episode,
)
from dodge_native_game.variants.pixel_repr_ddqn.probe_bank import (
    BANK_FORMAT,
    ProbeBank,
    build_bank,
)


def _make_dataset(
    root: Path, *, train_count: int, validation_count: int
) -> None:
    records: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    for split, count in (("train", train_count), ("validation", validation_count)):
        for index in range(count):
            episode = root / "episodes" / split / f"episode-{index:06d}.npz"
            episode.parent.mkdir(parents=True, exist_ok=True)
            frames = np.empty((129, 3, 128, 128), dtype=np.uint8)
            offset = 0 if split == "train" else 80
            for frame in range(129):
                frames[frame].fill((offset + index * 40 + frame) % 256)
            actions = np.asarray([(index + step) % 9 for step in range(128)])
            terminated = np.zeros(128, dtype=np.bool_)
            truncated = np.zeros(128, dtype=np.bool_)
            truncated[-1] = True
            with episode.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    pixels=frames,
                    actions=actions,
                    terminated=terminated,
                    truncated=truncated,
                )
            records[split].append(
                {
                    "episode_id": f"{split}-{index:06d}",
                    "split": split,
                    "seed": (1000 if split == "train" else 2000) + index,
                    "path": episode.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(episode.read_bytes()).hexdigest(),
                    "transition_count": 128,
                    "config_identity": f"recipe-{split}",
                }
            )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_format": LARGE_DATASET_FORMAT,
        "variant": "pixel-repr-ddqn",
        "status": "complete",
        "ready_marker": "READY",
        "observation": {
            "profile": "native-rgb-v1",
            "shape": [3, 128, 128],
            "dtype": "uint8",
        },
        "action_count": 9,
        "step_frames": 4,
        "history_size_default": 3,
        "decisions_per_episode": 128,
        "episodes": records,
        "splits": {
            split: {
                "episode_count": len(rows),
                "episode_ids": [row["episode_id"] for row in rows],
                "seeds": [row["seed"] for row in rows],
                "transition_count": sum(row["transition_count"] for row in rows),
            }
            for split, rows in records.items()
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "READY").write_text(f"{LARGE_DATASET_FORMAT}\n")


class _FakeModel(nn.Module):
    def __init__(self, dimension: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, dimension, bias=False)
        self.projector = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.BatchNorm1d(dimension),
        )

    def encode_cls(self, pixels: torch.Tensor) -> torch.Tensor:
        values = pixels.float().mean(dim=(-1, -2))
        return self.encoder(values)

    def _apply_projector(
        self, projector: nn.Module, values: torch.Tensor
    ) -> torch.Tensor:
        batch, time, dimension = values.shape
        return projector(values.reshape(batch * time, dimension)).reshape(
            batch, time, dimension
        )


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _expected_indices(count: int, frames: int = 4) -> list[int]:
    rng = np.random.default_rng(903)
    result: list[int] = []
    for _ in range(count):
        result.extend(
            sorted(
                int(value)
                for value in rng.choice(np.arange(1, 129), size=frames, replace=False)
            )
        )
    return result


def test_build_samples_deterministically_and_preserves_frozen_model(
    tmp_path: Path,
) -> None:
    _make_dataset(tmp_path / "dataset", train_count=2, validation_count=1)
    model = _FakeModel()
    model.train()
    model.projector.eval()
    before = _state(model)
    modes = [module.training for module in model.modules()]

    output = build_bank(
        model,
        tmp_path / "dataset",
        tmp_path / "bank-train",
        "train",
        device="cpu",
        encode_batch_size=3,
        checkpoint_sha256="a" * 64,
    )
    bank = ProbeBank(output)

    assert len(bank) == 8
    assert [row["frame"] for row in bank.index] == _expected_indices(2)
    assert bank.metadata["format"] == BANK_FORMAT
    assert bank.metadata["frames_per_episode"] == 4
    assert bank.metadata["max_frames_per_episode"] == 4
    assert bank.metadata["checkpoint_sha256"] == "a" * 64
    assert [module.training for module in model.modules()] == modes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    for mapped in (bank.pixels, bank.changed, bank.cls, bank.projected):
        assert isinstance(mapped, np.memmap)
        assert not mapped.flags.writeable
    assert bank.pixels.shape == (8, 3, 128, 128)
    assert bank.changed.shape == (8, 128, 128)
    assert bank.cls.shape == (8, 5)
    assert bank.projected.shape == (8, 5)


def test_bank_is_aligned_with_current_pixels_previous_mask_and_features(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root, train_count=1, validation_count=1)
    model = _FakeModel().eval()
    bank = ProbeBank(
        build_bank(
            model,
            dataset_root,
            tmp_path / "bank",
            "train",
            device="cpu",
        )
    )
    record = bank.index[0]
    source = dataset_root / "episodes" / "train" / "episode-000000.npz"
    from dodge_native_game.variants.pixel_repr_ddqn.large_dataset import (
        LargeEpisodeRecord,
    )

    episode = LargeEpisodeRecord(
        episode_id=record["episode_id"],
        split="train",
        seed=1000,
        path=source,
        sha256="",
        count=128,
        config_identity="recipe-train",
    )
    pixels, _ = _read_episode(episode, verify_hash=False)
    frame = int(record["frame"])
    np.testing.assert_array_equal(bank.pixels[0], pixels[frame])
    np.testing.assert_array_equal(
        bank.changed[0], np.any(pixels[frame] != pixels[frame - 1], axis=0)
    )
    expected_input = torch.from_numpy(pixels[frame : frame + 1]).unsqueeze(1)
    with torch.no_grad():
        expected_cls = model.encode_cls(expected_input)[:, 0]
        expected_projected = model._apply_projector(
            model.projector, model.encode_cls(expected_input)
        )[:, 0]
    np.testing.assert_allclose(bank.cls[0], expected_cls[0].numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(
        bank.projected[0], expected_projected[0].numpy(), rtol=0, atol=1e-5
    )


def test_fetch_returns_selected_representation_normalized_pixels_and_mask(
    tmp_path: Path,
) -> None:
    _make_dataset(tmp_path / "dataset", train_count=1, validation_count=1)
    bank = ProbeBank(
        build_bank(
            _FakeModel().eval(),
            tmp_path / "dataset",
            tmp_path / "bank",
            "validation",
            device="cpu",
            frames_per_episode=2,
        )
    )
    projected, pixels, changed = bank.fetch([1, 0], "projected", "cpu")
    assert projected.dtype == torch.float32
    assert pixels.dtype == torch.float32
    assert changed.dtype == torch.bool
    assert projected.shape == (2, 5)
    assert pixels.shape == (2, 3, 128, 128)
    assert changed.shape == (2, 128, 128)
    torch.testing.assert_close(projected, torch.from_numpy(bank.projected[[1, 0]]))
    expected_pixels = torch.from_numpy(bank.pixels[[1, 0]]).float() / 255
    torch.testing.assert_close(pixels, expected_pixels)
    torch.testing.assert_close(changed, torch.from_numpy(bank.changed[[1, 0]]))


def test_bank_sampling_is_repeatable_and_limits_frames_per_episode(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root, train_count=1, validation_count=1)
    model = _FakeModel().eval()
    first = ProbeBank(
        build_bank(model, dataset_root, tmp_path / "first", "train", device="cpu")
    )
    second = ProbeBank(
        build_bank(model, dataset_root, tmp_path / "second", "train", device="cpu")
    )
    assert first.index == second.index
    np.testing.assert_array_equal(first.pixels, second.pixels)
    np.testing.assert_array_equal(first.changed, second.changed)
    np.testing.assert_array_equal(first.cls, second.cls)
    np.testing.assert_array_equal(first.projected, second.projected)
    with pytest.raises(ValueError, match="frames_per_episode"):
        build_bank(
            model,
            dataset_root,
            tmp_path / "too-many",
            "train",
            device="cpu",
            frames_per_episode=5,
        )
