from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.large_dataset import (
    LARGE_DATASET_FORMAT,
    DatasetValidationError,
    LargePixelSequenceDataset,
    validate_dataset,
)


def _write_episode(
    root: Path, split: str, index: int
) -> tuple[dict[str, object], Path]:
    count = 128
    path = root / "episodes" / split / f"episode-{index:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.empty((count + 1, 3, 128, 128), dtype=np.uint8)
    split_offset = 0 if split == "train" else 80
    for step in range(count + 1):
        frames[step].fill((split_offset + index * 40 + step) % 256)
    actions = np.asarray([(index + step) % 9 for step in range(count)], dtype=np.int64)
    terminated = np.zeros(count, dtype=np.bool_)
    truncated = np.zeros(count, dtype=np.bool_)
    truncated[-1] = True
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            pixels=frames,
            actions=actions,
            terminated=terminated,
            truncated=truncated,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record: dict[str, object] = {
        "episode_id": f"{split}-{index:06d}",
        "split": split,
        "seed": (1000 if split == "train" else 2000) + index,
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "transition_count": count,
        "config_identity": f"recipe-{split}",
    }
    return record, path


def _make_dataset(
    root: Path, *, train_count: int = 2, validation_count: int = 1
) -> dict[str, object]:
    records: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    for split, count in (("train", train_count), ("validation", validation_count)):
        for index in range(count):
            record, _ = _write_episode(root, split, index)
            records[split].append(record)
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
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "READY").write_text(f"{LARGE_DATASET_FORMAT}\n")
    return manifest


def test_windows_use_prefix_sum_and_never_cross_episode_boundaries(
    tmp_path: Path,
) -> None:
    _make_dataset(tmp_path)
    dataset = LargePixelSequenceDataset(tmp_path, history_size=3)

    assert len(dataset) == 252
    assert not hasattr(dataset, "_windows")
    first = dataset[0]
    boundary = dataset[126]
    assert set(first) == {"pixels", "actions", "episode_id", "start"}
    assert first["episode_id"] == "train-000000"
    assert first["start"] == 0
    assert first["pixels"].shape == (4, 3, 128, 128)
    assert first["actions"].shape == (3,)
    assert first["pixels"].dtype == torch.uint8
    assert first["actions"].dtype == torch.int64
    assert boundary["episode_id"] == "train-000001"
    assert boundary["start"] == 0
    assert int(boundary["pixels"][0, 0, 0, 0]) == 40
    assert int(dataset[125]["pixels"][-1, 0, 0, 0]) == 128


def test_lru_cache_is_bounded_and_reuses_decoded_episodes(tmp_path: Path) -> None:
    _make_dataset(tmp_path, train_count=5)
    dataset = LargePixelSequenceDataset(tmp_path, cache_size=2)
    assert dataset.cache_info().currsize == 0

    for episode in range(4):
        dataset[episode * 126]
    assert dataset.cache_info().currsize == 2
    misses = dataset.cache_info().misses
    dataset[3 * 126]
    assert dataset.cache_info().hits == 1
    assert dataset.cache_info().misses == misses
    assert len(dataset.cached_episode_ids) <= 2


def test_hash_is_checked_on_cache_miss_after_loader_construction(
    tmp_path: Path,
) -> None:
    _make_dataset(tmp_path)
    dataset = LargePixelSequenceDataset(tmp_path)
    record = dataset._records[0]
    record.path.write_bytes(record.path.read_bytes() + b"tamper")
    with pytest.raises(DatasetValidationError, match="hash mismatch"):
        dataset[0]


def test_thorough_validation_streams_and_checks_array_metadata(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    validate_dataset(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    record = manifest["episodes"]["train"][0]
    path = tmp_path / record["path"]
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["actions"] = arrays["actions"].astype(np.int32)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")

    # Metadata is cheap and does not decompress the corpus; the explicit
    # thorough pass catches a malformed member after its new hash is recorded.
    LargePixelSequenceDataset(tmp_path)
    with pytest.raises(DatasetValidationError, match="actions shape/dtype"):
        validate_dataset(tmp_path, thorough=True)


def test_cross_split_duplicate_hash_and_path_escape_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _make_dataset(tmp_path)
    train_record = manifest["episodes"]["train"][0]
    validation_record = manifest["episodes"]["validation"][0]
    validation_record["sha256"] = train_record["sha256"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetValidationError, match="duplicate episode hash"):
        LargePixelSequenceDataset(tmp_path, split="validation")

    escape_root = tmp_path / "escape"
    manifest = _make_dataset(escape_root)
    manifest["episodes"]["train"][0]["path"] = "../outside.npz"
    (escape_root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetValidationError, match="escapes dataset root"):
        LargePixelSequenceDataset(escape_root)


def test_config_identity_cannot_cross_splits(tmp_path: Path) -> None:
    manifest = _make_dataset(tmp_path)
    manifest["episodes"]["validation"][0]["config_identity"] = "recipe-train"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetValidationError, match="config identity"):
        LargePixelSequenceDataset(tmp_path)


def test_coordinates_are_not_exposed_as_window_features(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    sample = LargePixelSequenceDataset(tmp_path)[0]
    assert set(sample) == {"pixels", "actions", "episode_id", "start"}
    assert "config_coordinates" not in sample
