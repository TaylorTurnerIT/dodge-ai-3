"""Validated episode-contained windows for the pixel representation MVP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .collect import (
    ACTION_COUNT,
    DATASET_ID_VERSION,
    DATASET_SCHEMA_VERSION,
    MAX_NATIVE_DECISIONS,
    OBSERVATION_PROFILE,
    OBSERVATION_SHAPE,
    READY_CONTENT,
    READY_MARKER,
    STEP_FRAMES,
)
from .native_adapter import NATIVE_SEED_MAX


class DatasetValidationError(ValueError):
    """Raised when a published corpus is incomplete, malformed, or tampered."""


@dataclass(frozen=True, slots=True)
class _Episode:
    episode_id: str
    split: str
    seed: int
    pixels: np.ndarray
    actions: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_id(records: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    for split in sorted(records):
        for record in records[split]:
            lines.append(
                f"{split}\0{record['episode_id']}\0{record['seed']}\0{record['sha256']}"
            )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DatasetValidationError(f"{name} must be an integer")
    return int(value)


def _records(raw: object, split: str) -> list[dict[str, Any]]:
    value = raw
    if isinstance(value, dict):
        value = value.get("episodes")
    if not isinstance(value, list):
        raise DatasetValidationError(f"manifest datasets[{split!r}] must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise DatasetValidationError(
                f"manifest {split} record {index} is not an object"
            )
        result.append(item)
    return result


def _inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise DatasetValidationError(f"episode path escapes dataset root: {relative!r}")
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetValidationError(
            f"episode path escapes dataset root: {relative!r}"
        ) from error
    return candidate


def _check_optional_metadata(
    arrays: Any,
    *,
    transition_count: int,
    record: dict[str, Any],
    path: Path,
) -> None:
    for name in ("rewards", "terminated", "truncated"):
        if name not in arrays:
            continue
        value = np.asarray(arrays[name])
        if value.shape != (transition_count,):
            raise DatasetValidationError(f"{path}: {name} shape does not match actions")
        if name == "rewards" and not np.issubdtype(value.dtype, np.floating):
            raise DatasetValidationError(f"{path}: rewards must be floating point")
        if name in {"terminated", "truncated"} and value.dtype != np.bool_:
            raise DatasetValidationError(f"{path}: {name} must have dtype bool")

    if "terminated" in arrays and "truncated" in arrays:
        terminal = np.asarray(arrays["terminated"], dtype=np.bool_)
        truncated = np.asarray(arrays["truncated"], dtype=np.bool_)
        if np.any(terminal & truncated):
            raise DatasetValidationError(
                f"{path}: transition cannot terminate and truncate"
            )
        if transition_count:
            if np.any(terminal[:-1]) or np.any(truncated[:-1]):
                raise DatasetValidationError(
                    f"{path}: episode boundary must be on its final transition"
                )
            if bool(record.get("terminal", bool(terminal[-1]))) != bool(terminal[-1]):
                raise DatasetValidationError(f"{path}: terminal metadata disagrees")
            if bool(record.get("truncated", bool(truncated[-1]))) != bool(
                truncated[-1]
            ):
                raise DatasetValidationError(f"{path}: truncation metadata disagrees")


def _validate_manifest(
    root: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise DatasetValidationError(
            f"dataset root is not a published directory: {root}"
        )
    marker = root / READY_MARKER
    if not marker.is_file() or marker.read_text() != READY_CONTENT:
        raise DatasetValidationError("dataset has no valid final READY marker")
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            "dataset manifest is missing or invalid JSON"
        ) from error
    if not isinstance(manifest, dict):
        raise DatasetValidationError("dataset manifest must be an object")
    if (
        _as_int(manifest.get("schema_version"), "schema_version")
        != DATASET_SCHEMA_VERSION
    ):
        raise DatasetValidationError("unsupported dataset schema version")
    if manifest.get("dataset_format") != DATASET_ID_VERSION:
        raise DatasetValidationError("unsupported dataset format")
    if manifest.get("variant") != "pixel-repr-ddqn":
        raise DatasetValidationError("dataset variant does not match pixel-repr-ddqn")
    if manifest.get("status") != "complete":
        raise DatasetValidationError("dataset is not marked complete")
    observation = manifest.get("observation")
    if not isinstance(observation, dict):
        raise DatasetValidationError("manifest observation declaration is missing")
    if observation.get("profile") != OBSERVATION_PROFILE:
        raise DatasetValidationError("dataset observation profile is not native-rgb-v1")
    if (
        observation.get("shape") != list(OBSERVATION_SHAPE)
        or observation.get("dtype") != "uint8"
    ):
        raise DatasetValidationError("dataset observation declaration is invalid")
    if _as_int(manifest.get("action_count"), "action_count") != ACTION_COUNT:
        raise DatasetValidationError("dataset action count does not match native Dodge")
    if _as_int(manifest.get("step_frames"), "step_frames") != STEP_FRAMES:
        raise DatasetValidationError("dataset cadence does not match native Dodge")
    collection = manifest.get("collection")
    if not isinstance(collection, dict):
        raise DatasetValidationError("manifest collection declaration is missing")
    if (
        _as_int(collection.get("max_native_decisions"), "max_native_decisions")
        != MAX_NATIVE_DECISIONS
    ):
        raise DatasetValidationError("dataset native decision cap is invalid")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise DatasetValidationError("manifest datasets declaration is missing")
    split_records = {
        split: _records(datasets.get(split), split) for split in ("train", "validation")
    }
    seeds = manifest.get("seeds")
    splits = manifest.get("splits")
    if not isinstance(seeds, dict) or not isinstance(splits, dict):
        raise DatasetValidationError("manifest split declarations are missing")
    try:
        expected_dataset_id = _dataset_id(split_records)
    except KeyError as error:
        raise DatasetValidationError(
            "manifest split record is missing dataset identity"
        ) from error
    if manifest.get("dataset_id") != expected_dataset_id:
        raise DatasetValidationError("dataset_id does not match split records")
    all_ids: set[str] = set()
    all_seeds: dict[int, str] = {}
    for split, records in split_records.items():
        manifest_seed_list = seeds.get(split)
        if not isinstance(manifest_seed_list, list):
            raise DatasetValidationError(f"manifest seeds[{split!r}] is missing")
        if [record.get("seed") for record in records] != manifest_seed_list:
            raise DatasetValidationError(
                f"manifest seeds[{split!r}] disagree with datasets"
            )
        split_info = splits.get(split)
        if not isinstance(split_info, dict):
            raise DatasetValidationError(f"manifest splits[{split!r}] is missing")
        if split_info.get("episode_ids") != [
            record.get("episode_id") for record in records
        ]:
            raise DatasetValidationError(f"manifest split IDs for {split!r} disagree")
        if split_info.get("seeds") != manifest_seed_list:
            raise DatasetValidationError(f"manifest split seeds for {split!r} disagree")
        for record in records:
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise DatasetValidationError(f"{split} record has invalid episode_id")
            if episode_id in all_ids:
                raise DatasetValidationError(f"duplicate episode_id: {episode_id}")
            all_ids.add(episode_id)
            if record.get("split") != split:
                raise DatasetValidationError(
                    f"episode {episode_id} has the wrong split"
                )
            game_seed = _as_int(record.get("seed"), f"{episode_id}.seed")
            if not 0 <= game_seed <= NATIVE_SEED_MAX:
                raise DatasetValidationError(
                    f"episode {episode_id} seed is out of native range"
                )
            if game_seed in all_seeds:
                previous = all_seeds[game_seed]
                raise DatasetValidationError(
                    f"native seed {game_seed} appears in {previous!r} and {split!r}"
                )
            all_seeds[game_seed] = split
            relative = record.get("path")
            expected_hash = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise DatasetValidationError(
                    f"episode {episode_id} path/hash is missing"
                )
            path = _inside(root, relative)
            if not path.is_file():
                raise DatasetValidationError(f"episode {episode_id} file is missing")
            actual_hash = _sha256(path)
            if (
                actual_hash != expected_hash
                or record.get("hash", expected_hash) != actual_hash
            ):
                raise DatasetValidationError(f"episode {episode_id} hash mismatch")
            try:
                with np.load(path, allow_pickle=False) as payload:
                    if "pixels" not in payload or "actions" not in payload:
                        raise DatasetValidationError(
                            f"episode {episode_id} lacks pixels/actions"
                        )
                    pixels = np.asarray(payload["pixels"])
                    actions = np.asarray(payload["actions"])
                    if (
                        pixels.dtype != np.uint8
                        or pixels.ndim != 4
                        or pixels.shape[1:] != OBSERVATION_SHAPE
                    ):
                        raise DatasetValidationError(
                            f"episode {episode_id} has invalid pixel shape/dtype"
                        )
                    if actions.dtype != np.int64 or actions.ndim != 1:
                        raise DatasetValidationError(
                            f"episode {episode_id} has invalid action shape/dtype"
                        )
                    transition_count = int(actions.shape[0])
                    if pixels.shape[0] != transition_count + 1:
                        raise DatasetValidationError(
                            f"episode {episode_id} frame count does not equal actions+1"
                        )
                    if transition_count < 1:
                        raise DatasetValidationError(
                            f"episode {episode_id} has no transitions"
                        )
                    if np.any(actions < 0) or np.any(actions >= ACTION_COUNT):
                        raise DatasetValidationError(
                            f"episode {episode_id} contains an invalid action"
                        )
                    if (
                        _as_int(
                            record.get("transition_count"),
                            f"{episode_id}.transition_count",
                        )
                        != transition_count
                    ):
                        raise DatasetValidationError(
                            f"episode {episode_id} transition count metadata disagrees"
                        )
                    if (
                        _as_int(record.get("frame_count"), f"{episode_id}.frame_count")
                        != pixels.shape[0]
                    ):
                        raise DatasetValidationError(
                            f"episode {episode_id} frame count metadata disagrees"
                        )
                    if record.get("pixel_shape") != list(pixels.shape) or record.get(
                        "action_shape"
                    ) != list(actions.shape):
                        raise DatasetValidationError(
                            f"episode {episode_id} shape metadata disagrees"
                        )
                    _check_optional_metadata(
                        payload,
                        transition_count=transition_count,
                        record=record,
                        path=path,
                    )
            except (OSError, ValueError, KeyError) as error:
                if isinstance(error, DatasetValidationError):
                    raise
                raise DatasetValidationError(
                    f"episode {episode_id} cannot be read"
                ) from error
    actual_decisions = sum(
        int(record["transition_count"])
        for records in split_records.values()
        for record in records
    )
    if (
        _as_int(collection.get("native_decisions"), "native_decisions")
        != actual_decisions
    ):
        raise DatasetValidationError("manifest native decision count disagrees")
    if actual_decisions > MAX_NATIVE_DECISIONS:
        raise DatasetValidationError("dataset exceeds the bounded native decision cap")
    return manifest, split_records


class PixelSequenceDataset(Dataset[dict[str, object]]):
    """Return contiguous ``history_size`` action windows within one episode."""

    def __init__(self, root: Path, split: str = "train", history_size: int = 3) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        history = _as_int(history_size, "history_size")
        if history < 1:
            raise ValueError("history_size must be positive")
        self.root = Path(root)
        self.manifest, split_records = _validate_manifest(self.root)
        self.split = split
        self.history_size = history
        self._episodes: list[_Episode] = []
        self._windows: list[tuple[int, int]] = []
        for record in split_records[split]:
            path = _inside(self.root, str(record["path"]))
            with np.load(path, allow_pickle=False) as payload:
                pixels = np.array(payload["pixels"], dtype=np.uint8, copy=True)
                actions = np.array(payload["actions"], dtype=np.int64, copy=True)
            episode_index = len(self._episodes)
            self._episodes.append(
                _Episode(
                    episode_id=str(record["episode_id"]),
                    split=split,
                    seed=int(record["seed"]),
                    pixels=pixels,
                    actions=actions,
                )
            )
            windows = max(0, int(actions.shape[0]) - history + 1)
            self._windows.extend((episode_index, start) for start in range(windows))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        item = int(index)
        if item < 0:
            item += len(self._windows)
        if item < 0 or item >= len(self._windows):
            raise IndexError("dataset index out of range")
        episode_index, start = self._windows[item]
        episode = self._episodes[episode_index]
        stop = start + self.history_size
        return {
            "pixels": torch.from_numpy(episode.pixels[start : stop + 1].copy()),
            "actions": torch.from_numpy(episode.actions[start:stop].copy()),
            "episode_id": episode.episode_id,
            "start": start,
        }


__all__ = ["DatasetValidationError", "PixelSequenceDataset"]
