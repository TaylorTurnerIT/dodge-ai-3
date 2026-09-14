from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn import collect as collector
from dodge_native_game.variants.pixel_repr_ddqn.collect import collect_dataset
from dodge_native_game.variants.pixel_repr_ddqn.dataset import (
    DatasetValidationError,
    PixelSequenceDataset,
)
from dodge_native_game.variants.pixel_repr_ddqn.native_adapter import PixelNativeAdapter


class FakeEnvironment:
    """A deterministic RGB-only Gym fixture with an optional terminal step."""

    def __init__(self, terminal_after: dict[int, int] | None = None) -> None:
        self.terminal_after = terminal_after or {}
        self.seed = 0
        self.steps = 0
        self.closed = False

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        self.seed = seed
        self.steps = 0
        return self._frame(), {"secret_player_position": (7, 9)}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.steps += 1
        terminated = self.steps >= self.terminal_after.get(self.seed, 10_000)
        return (
            self._frame(action),
            float(action),
            terminated,
            False,
            {"secret_player_position": (self.steps, action)},
        )

    def close(self) -> None:
        self.closed = True

    def _frame(self, action: int = 0) -> np.ndarray:
        value = (self.seed + self.steps + action) % 256
        return np.full((3, 128, 128), value, dtype=np.uint8)


def _install_fake(
    monkeypatch: pytest.MonkeyPatch, terminal_after: dict[int, int] | None = None
) -> None:
    monkeypatch.setattr(
        collector,
        "PixelNativeAdapter",
        lambda: PixelNativeAdapter(FakeEnvironment(terminal_after)),
    )


def _collect(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, object]:
    _install_fake(monkeypatch, {0: 2})
    return collect_dataset(root, [0, 1], [10_000], max_steps_per_episode=4, seed=17)


def test_adapter_whitelists_pixels_actions_and_boundary_values() -> None:
    environment = FakeEnvironment({3: 1})
    adapter = PixelNativeAdapter(environment)
    try:
        reset = adapter.reset(seed=3)
        assert isinstance(reset, np.ndarray)
        assert reset.shape == (3, 128, 128)
        assert reset.dtype == np.uint8
        result = adapter.step(2)
        assert len(result) == 4
        pixels, reward, terminated, truncated = result
        assert pixels.shape == (3, 128, 128)
        assert reward == 2.0
        assert terminated is True
        assert truncated is False
        assert not isinstance(reset, tuple)
    finally:
        adapter.close()
    assert environment.closed


def test_collection_is_deterministic_and_has_disjoint_hashed_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _collect(monkeypatch, tmp_path / "left")
    right = _collect(monkeypatch, tmp_path / "right")
    assert left == right
    assert left["datasets"]["train"]  # type: ignore[index]
    assert left["seeds"] == {"train": [0, 1], "validation": [10_000]}
    for split in ("train", "validation"):
        for record in left["datasets"][split]:  # type: ignore[index]
            assert record["episode_id"].startswith(split)
            assert len(record["sha256"]) == 64
            assert (tmp_path / "left" / record["path"]).is_file()
    assert (tmp_path / "left" / "READY").read_text() == "pixel-repr-ddqn-episodes-v1\n"


def test_terminal_and_cap_boundaries_never_cross_episode_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _collect(monkeypatch, tmp_path / "dataset")
    root = tmp_path / "dataset"
    train = PixelSequenceDataset(root, split="train", history_size=2)
    assert len(train) == 4  # terminal episode: 1 window; capped episode: 3
    samples = [train[index] for index in range(len(train))]
    assert [sample["episode_id"] for sample in samples] == [
        "train-000000",
        "train-000001",
        "train-000001",
        "train-000001",
    ]
    assert [sample["start"] for sample in samples] == [0, 0, 1, 2]
    assert all(sample["pixels"].shape == (3, 3, 128, 128) for sample in samples)
    assert all(sample["actions"].shape == (2,) for sample in samples)
    assert all(sample["pixels"].dtype == torch.uint8 for sample in samples)
    assert all(sample["actions"].dtype == torch.int64 for sample in samples)
    assert manifest["datasets"]["train"][0]["terminal"] is True  # type: ignore[index]
    assert manifest["datasets"]["train"][0]["truncated"] is False  # type: ignore[index]
    assert manifest["datasets"]["train"][1]["terminal"] is False  # type: ignore[index]
    assert manifest["datasets"]["train"][1]["truncated"] is True  # type: ignore[index]


def test_hash_and_schema_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    _collect(monkeypatch, root)
    manifest = json.loads((root / "manifest.json").read_text())
    episode = root / manifest["datasets"]["train"][0]["path"]
    with np.load(episode, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["actions"] = arrays["actions"].copy()
    arrays["actions"][0] = 8 if arrays["actions"][0] != 8 else 7
    np.savez_compressed(episode, **arrays)
    with pytest.raises(DatasetValidationError, match="hash mismatch"):
        PixelSequenceDataset(root)

    # A manifest-only hash edit is independently rejected, too.
    _collect(monkeypatch, tmp_path / "manifest-tamper")
    other = tmp_path / "manifest-tamper"
    payload = json.loads((other / "manifest.json").read_text())
    payload["datasets"]["train"][0]["sha256"] = "0" * 64
    (other / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(DatasetValidationError, match="hash mismatch|dataset_id"):
        PixelSequenceDataset(other)


def test_seed_overlap_and_silent_overwrite_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch)
    with pytest.raises(ValueError, match="overlap"):
        collect_dataset(tmp_path / "overlap", [4], [4], 1)
    root = tmp_path / "dataset"
    collect_dataset(root, [4], [5], 1)
    with pytest.raises(FileExistsError):
        collect_dataset(root, [4], [5], 1)
