"""Collect a deterministic, hashed native RGB episode corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .native_adapter import ACTION_COUNT, NATIVE_SEED_MAX, PixelNativeAdapter

DATASET_SCHEMA_VERSION = 1
DATASET_ID_VERSION = "pixel-repr-ddqn-episodes-v1"
OBSERVATION_PROFILE = "native-rgb-v1"
OBSERVATION_SHAPE = (3, 128, 128)
STEP_FRAMES = 4
MAX_NATIVE_DECISIONS = 256
READY_MARKER = "READY"
READY_CONTENT = f"{DATASET_ID_VERSION}\n"


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_seed(value: object, name: str) -> int:
    result = _validate_integer(value, name)
    if not 0 <= result <= NATIVE_SEED_MAX:
        raise ValueError(f"{name} must be between 0 and {NATIVE_SEED_MAX}")
    return result


def _validate_seeds(values: Sequence[int], name: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    result = [_validate_seed(value, f"{name} seed") for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate native seeds")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_episode(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a deterministic compressed NPZ and publish it atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    try:
        with Path(temporary).open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _episode(
    adapter: PixelNativeAdapter,
    *,
    game_seed: int,
    max_steps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int | bool]:
    first = adapter.reset(seed=game_seed)
    pixels = [np.array(first, dtype=np.uint8, copy=True)]
    actions: list[int] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    for _step in range(max_steps):
        action = int(rng.integers(0, ACTION_COUNT))
        observation, reward, is_terminal, is_truncated = adapter.step(action)
        actions.append(action)
        pixels.append(np.array(observation, dtype=np.uint8, copy=True))
        rewards.append(float(reward))
        terminated.append(bool(is_terminal))
        truncated.append(bool(is_truncated))
        if is_terminal or is_truncated:
            break

    # The native environment has no Python-side time limit.  The collector's
    # cap is therefore represented as a truncation on the last transition.
    if not terminated[-1] and not truncated[-1]:
        truncated[-1] = True

    return {
        "pixels": np.ascontiguousarray(np.stack(pixels), dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "seed": game_seed,
        "terminal": bool(terminated[-1]),
        "capped": bool(truncated[-1]),
    }


def _record(
    *,
    split: str,
    index: int,
    game_seed: int,
    path: Path,
    file_path: Path,
    arrays: dict[str, np.ndarray | int | bool],
) -> dict[str, Any]:
    pixels = arrays["pixels"]
    actions = arrays["actions"]
    assert isinstance(pixels, np.ndarray)
    assert isinstance(actions, np.ndarray)
    episode_id = f"{split}-{index:06d}"
    transition_count = int(actions.shape[0])
    episode_hash = _sha256(file_path)
    return {
        "episode_id": episode_id,
        "id": episode_id,
        "split": split,
        "seed": game_seed,
        "path": path.as_posix(),
        "sha256": episode_hash,
        "hash": episode_hash,
        "transition_count": transition_count,
        "frame_count": int(pixels.shape[0]),
        "pixel_shape": list(pixels.shape),
        "action_shape": list(actions.shape),
        "terminal": bool(arrays["terminal"]),
        "truncated": bool(arrays["capped"]),
    }


def _dataset_id(records: dict[str, list[dict[str, Any]]]) -> str:
    lines = []
    for split in sorted(records):
        for record in records[split]:
            lines.append(
                f"{split}\0{record['episode_id']}\0{record['seed']}\0{record['sha256']}"
            )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _manifest(
    *,
    records: dict[str, list[dict[str, Any]]],
    train_seeds: list[int],
    validation_seeds: list[int],
    max_steps: int,
    collection_seed: int,
) -> dict[str, Any]:
    split_summary = {
        split: {
            "episode_ids": [record["episode_id"] for record in split_records],
            "seeds": [record["seed"] for record in split_records],
            "transition_count": sum(
                int(record["transition_count"]) for record in split_records
            ),
        }
        for split, split_records in records.items()
    }
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_format": DATASET_ID_VERSION,
        "dataset_id": _dataset_id(records),
        "variant": "pixel-repr-ddqn",
        "status": "complete",
        "ready_marker": READY_MARKER,
        "observation": {
            "profile": OBSERVATION_PROFILE,
            "shape": list(OBSERVATION_SHAPE),
            "dtype": "uint8",
        },
        "action_count": ACTION_COUNT,
        "step_frames": STEP_FRAMES,
        "history_size_default": 3,
        "collection": {
            "policy": "uniform-random-v1",
            "seed": collection_seed,
            "max_steps_per_episode": max_steps,
            "max_native_decisions": MAX_NATIVE_DECISIONS,
            "native_decisions": sum(
                int(record["transition_count"])
                for split_records in records.values()
                for record in split_records
            ),
        },
        "seeds": {"train": train_seeds, "validation": validation_seeds},
        "splits": split_summary,
        "datasets": records,
    }


def collect_dataset(
    root: Path,
    train_seeds: list[int],
    validation_seeds: list[int],
    max_steps_per_episode: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Collect and atomically publish a train/validation RGB corpus.

    The collector owns only a local NumPy generator for its uniform random
    action policy.  It never seeds or draws from Python's global random state
    or a learner's generator.  ``root`` must not exist: a completed or
    incomplete prior corpus is never silently replaced.
    """

    output_root = Path(root)
    train = _validate_seeds(train_seeds, "train_seeds")
    validation = _validate_seeds(validation_seeds, "validation_seeds")
    overlap = set(train).intersection(validation)
    if overlap:
        raise ValueError(f"train and validation seeds overlap: {sorted(overlap)}")
    max_steps = _validate_integer(max_steps_per_episode, "max_steps_per_episode")
    if max_steps < 1:
        raise ValueError("max_steps_per_episode must be positive")
    requested_decisions = (len(train) + len(validation)) * max_steps
    if requested_decisions > MAX_NATIVE_DECISIONS:
        raise ValueError(
            "bounded MVP collection allows at most "
            f"{MAX_NATIVE_DECISIONS} native decisions; requested {requested_decisions}"
        )
    collection_seed = _validate_integer(seed, "seed")

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"dataset output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    # Reserve a sibling lock so a concurrent collector cannot claim the same
    # destination.  The final root is created only after all files and READY
    # have been written in the staging directory.
    lock = output_root.parent / f".{output_root.name}.collecting"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"dataset output is being collected: {output_root}"
        ) from error
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent)
    )
    rng = np.random.default_rng(collection_seed)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    try:
        assert staging is not None
        adapter = PixelNativeAdapter()
        try:
            for split, seeds in (("train", train), ("validation", validation)):
                split_dir = staging / "episodes" / split
                for index, game_seed in enumerate(seeds):
                    arrays = _episode(
                        adapter,
                        game_seed=game_seed,
                        max_steps=max_steps,
                        rng=rng,
                    )
                    relative = Path("episodes") / split / f"episode-{index:06d}.npz"
                    path = staging / relative
                    payload = {
                        key: value
                        for key, value in arrays.items()
                        if isinstance(value, np.ndarray)
                    }
                    _write_episode(path, payload)
                    records[split].append(
                        _record(
                            split=split,
                            index=index,
                            game_seed=game_seed,
                            path=relative,
                            file_path=path,
                            arrays=arrays,
                        )
                    )
                split_dir.mkdir(parents=True, exist_ok=True)
        finally:
            adapter.close()

        manifest = _manifest(
            records=records,
            train_seeds=train,
            validation_seeds=validation,
            max_steps=max_steps,
            collection_seed=collection_seed,
        )
        _atomic_bytes(
            staging / "manifest.json",
            (
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode(),
        )
        _atomic_bytes(staging / READY_MARKER, READY_CONTENT.encode())
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError(
                f"dataset output appeared during collection: {output_root}"
            )
        os.rename(staging, output_root)
        staging = None
        return manifest
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock.rmdir() if lock.exists() else None


def _seed_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seeds", type=_seed_list, default=[])
    parser.add_argument("--validation-seeds", type=_seed_list, default=[])
    parser.add_argument("--max-steps-per-episode", "--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="collector action RNG seed")
    args = parser.parse_args(argv)
    manifest = collect_dataset(
        args.output,
        args.train_seeds,
        args.validation_seeds,
        args.max_steps_per_episode,
        args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_COUNT",
    "DATASET_ID_VERSION",
    "DATASET_SCHEMA_VERSION",
    "MAX_NATIVE_DECISIONS",
    "OBSERVATION_PROFILE",
    "OBSERVATION_SHAPE",
    "READY_MARKER",
    "collect_dataset",
    "main",
]
