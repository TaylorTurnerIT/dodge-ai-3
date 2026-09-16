"""Bounded-memory reader for the large pixel/action practice corpus."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import DatasetValidationError
from .native_adapter import ACTION_COUNT, NATIVE_SEED_MAX

LARGE_DATASET_SCHEMA_VERSION: Final = 1
LARGE_DATASET_FORMAT: Final = "pixel-repr-ddqn-large-practice-v1"
LARGE_READY_MARKER: Final = "READY"
LARGE_READY_CONTENT: Final = f"{LARGE_DATASET_FORMAT}\n"
LARGE_VARIANT: Final = "pixel-repr-ddqn"
LARGE_OBSERVATION_PROFILE: Final = "native-rgb-v1"
LARGE_OBSERVATION_SHAPE: Final = (3, 128, 128)
LARGE_DEFAULT_HISTORY_SIZE: Final = 3
LARGE_DEFAULT_CACHE_SIZE: Final = 4
LARGE_EPISODE_ACTION_COUNT: Final = 128
LARGE_STEP_FRAMES: Final = 4
LARGE_SPLITS: Final = ("train", "validation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DatasetValidationError(f"{name} must be an integer")
    return int(value)


def _inside(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DatasetValidationError("episode path must be a non-empty string")
    path = Path(relative)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise DatasetValidationError(f"episode path escapes dataset root: {relative!r}")
    root_resolved = root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise DatasetValidationError(
            f"episode path escapes dataset root: {relative!r}"
        ) from error
    return candidate


def _records_for_split(
    manifest: Mapping[str, Any], split: str
) -> list[Mapping[str, Any]]:
    episodes = manifest.get("episodes")
    if not isinstance(episodes, Mapping):
        raise DatasetValidationError("manifest episodes declaration is missing")
    value: object = episodes.get(split)
    if not isinstance(value, list):
        raise DatasetValidationError(f"manifest episodes[{split!r}] must be a list")
    result: list[Mapping[str, Any]] = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise DatasetValidationError(
                f"manifest {split} record {index} is not an object"
            )
        result.append(record)
    return result


@dataclass(frozen=True, slots=True)
class LargeEpisodeRecord:
    """Metadata for one episode; its arrays are loaded on demand."""

    episode_id: str
    split: str
    seed: int
    path: Path
    sha256: str
    count: int
    config_identity: str


@dataclass(frozen=True, slots=True)
class LargeDatasetMetadata:
    manifest: dict[str, Any]
    root: Path
    records: dict[str, tuple[LargeEpisodeRecord, ...]]


class CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int


def _validate_header(root: Path, *, require_ready: bool) -> dict[str, Any]:
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise DatasetValidationError(
            f"dataset root is not a published directory: {root}"
        )
    if require_ready:
        try:
            if (root / LARGE_READY_MARKER).read_text() != LARGE_READY_CONTENT:
                raise DatasetValidationError("dataset has no valid final READY marker")
        except OSError as error:
            raise DatasetValidationError(
                "dataset has no valid final READY marker"
            ) from error
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            "dataset manifest is missing or invalid JSON"
        ) from error
    if not isinstance(manifest, dict):
        raise DatasetValidationError("dataset manifest must be an object")
    if _as_int(manifest.get("schema_version"), "schema_version") != (
        LARGE_DATASET_SCHEMA_VERSION
    ):
        raise DatasetValidationError("unsupported large dataset schema version")
    if manifest.get("dataset_format") != LARGE_DATASET_FORMAT:
        raise DatasetValidationError("unsupported large dataset format")
    if manifest.get("variant") != LARGE_VARIANT:
        raise DatasetValidationError("dataset variant does not match pixel-repr-ddqn")
    if manifest.get("status") != "complete":
        raise DatasetValidationError("dataset is not marked complete")
    if manifest.get("ready_marker") != LARGE_READY_MARKER:
        raise DatasetValidationError("manifest READY marker declaration is invalid")
    observation = manifest.get("observation")
    if not isinstance(observation, Mapping):
        raise DatasetValidationError("manifest observation declaration is missing")
    if (
        observation.get("profile") != LARGE_OBSERVATION_PROFILE
        or observation.get("shape") != list(LARGE_OBSERVATION_SHAPE)
        or observation.get("dtype") != "uint8"
    ):
        raise DatasetValidationError("manifest observation declaration is invalid")
    if _as_int(manifest.get("action_count"), "action_count") != ACTION_COUNT:
        raise DatasetValidationError("dataset action count does not match native Dodge")
    if _as_int(manifest.get("step_frames"), "step_frames") != LARGE_STEP_FRAMES:
        raise DatasetValidationError("dataset cadence does not match native Dodge")
    if _as_int(manifest.get("history_size_default"), "history_size_default") != (
        LARGE_DEFAULT_HISTORY_SIZE
    ):
        raise DatasetValidationError(
            "dataset history default does not match the contract"
        )
    if _as_int(
        manifest.get("decisions_per_episode"), "decisions_per_episode"
    ) != LARGE_EPISODE_ACTION_COUNT:
        raise DatasetValidationError("dataset episode action count is not 128")
    if not isinstance(manifest.get("episodes"), Mapping):
        raise DatasetValidationError("manifest episodes declaration is missing")
    return manifest


def _record(root: Path, split: str, raw: Mapping[str, Any]) -> LargeEpisodeRecord:
    required = (
        "path",
        "sha256",
        "episode_id",
        "seed",
        "transition_count",
        "config_identity",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise DatasetValidationError(
            f"{split} episode record is missing {', '.join(missing)}"
        )
    episode_id = raw["episode_id"]
    if not isinstance(episode_id, str) or not episode_id:
        raise DatasetValidationError(f"{split} record has invalid id")
    digest = raw["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DatasetValidationError(f"episode {episode_id} has an invalid hash")
    path = _inside(root, raw["path"])
    if not path.is_file():
        raise DatasetValidationError(f"episode {episode_id} file is missing")
    count = _as_int(raw["transition_count"], f"{episode_id}.transition_count")
    if count != LARGE_EPISODE_ACTION_COUNT:
        raise DatasetValidationError(f"episode {episode_id} count is not 128")
    seed = _as_int(raw["seed"], f"{episode_id}.seed")
    if not 0 <= seed <= NATIVE_SEED_MAX:
        raise DatasetValidationError(
            f"episode {episode_id} seed is out of native range"
        )
    config_identity = raw["config_identity"]
    if not isinstance(config_identity, str) or not config_identity:
        raise DatasetValidationError(
            f"episode {episode_id} has invalid config identity"
        )
    return LargeEpisodeRecord(
        episode_id=episode_id,
        split=split,
        seed=seed,
        path=path,
        sha256=digest,
        count=count,
        config_identity=config_identity,
    )


def read_dataset_metadata(
    root: Path, *, require_ready: bool = True
) -> LargeDatasetMetadata:
    """Validate manifest metadata without opening/decompressing episode arrays."""

    dataset_root = Path(root)
    manifest = _validate_header(dataset_root, require_ready=require_ready)
    records: dict[str, tuple[LargeEpisodeRecord, ...]] = {}
    ids: set[str] = set()
    paths: set[Path] = set()
    seeds: set[int] = set()
    hashes: dict[str, str] = {}
    recipes: dict[str, str] = {}
    for split in LARGE_SPLITS:
        parsed: list[LargeEpisodeRecord] = []
        raw_records = _records_for_split(manifest, split)
        if not raw_records:
            raise DatasetValidationError(f"manifest {split} split is empty")
        for raw in raw_records:
            if "split" in raw and raw["split"] != split:
                raise DatasetValidationError(
                    f"episode record is assigned to the wrong split: {split}"
                )
            episode = _record(dataset_root, split, raw)
            if episode.episode_id in ids:
                raise DatasetValidationError(
                    f"duplicate episode id: {episode.episode_id}"
                )
            if episode.path in paths:
                raise DatasetValidationError("episode path is listed more than once")
            if episode.seed in seeds:
                raise DatasetValidationError(
                    f"native seed {episode.seed} is duplicated"
                )
            previous_split = hashes.get(episode.sha256)
            if previous_split is not None and previous_split != split:
                raise DatasetValidationError(
                    f"duplicate episode hash across splits: {episode.sha256}"
                )
            previous_recipe = recipes.get(episode.config_identity)
            if previous_recipe is not None and previous_recipe != split:
                raise DatasetValidationError(
                    "config identity appears in both train and validation"
                )
            ids.add(episode.episode_id)
            paths.add(episode.path)
            seeds.add(episode.seed)
            hashes[episode.sha256] = split
            recipes[episode.config_identity] = split
            parsed.append(episode)
        records[split] = tuple(parsed)
    split_counts = manifest.get("splits")
    if not isinstance(split_counts, Mapping):
        raise DatasetValidationError("manifest splits declaration is missing")
    for split in LARGE_SPLITS:
        summary = split_counts.get(split)
        if not isinstance(summary, Mapping):
            raise DatasetValidationError(f"manifest splits[{split!r}] is missing")
        actual_count = len(records[split])
        declared_count = _as_int(
            summary.get("episode_count"), f"splits[{split!r}].episode_count"
        )
        if declared_count != actual_count:
            raise DatasetValidationError(
                f"manifest {split} count disagrees with records"
            )
        expected_ids = [record.episode_id for record in records[split]]
        expected_seeds = [record.seed for record in records[split]]
        if summary.get("episode_ids") != expected_ids:
            raise DatasetValidationError(
                f"manifest {split} episode IDs disagree with records"
            )
        if summary.get("seeds") != expected_seeds:
            raise DatasetValidationError(
                f"manifest {split} seeds disagree with records"
            )
        expected_transitions = sum(record.count for record in records[split])
        if _as_int(
            summary.get("transition_count"), f"splits[{split!r}].transition_count"
        ) != expected_transitions:
            raise DatasetValidationError(
                f"manifest {split} transition count disagrees with records"
            )
    declared_split_counts = manifest.get("split_counts")
    if declared_split_counts is not None:
        if not isinstance(declared_split_counts, Mapping):
            raise DatasetValidationError("manifest split_counts declaration is invalid")
        for split in LARGE_SPLITS:
            if _as_int(
                declared_split_counts.get(split), f"split_counts[{split!r}]"
            ) != len(records[split]):
                raise DatasetValidationError(
                    f"manifest split_counts[{split!r}] disagrees with records"
                )
    declared_seeds = manifest.get("seeds")
    if declared_seeds is not None:
        if not isinstance(declared_seeds, Mapping):
            raise DatasetValidationError("manifest seeds declaration is invalid")
        for split in LARGE_SPLITS:
            if declared_seeds.get(split) != [
                record.seed for record in records[split]
            ]:
                raise DatasetValidationError(
                    f"manifest seeds[{split!r}] disagree with records"
                )
    return LargeDatasetMetadata(manifest, dataset_root, records)


def _read_episode(
    record: LargeEpisodeRecord, *, verify_hash: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Read one archive and validate every field needed for a training window."""

    if verify_hash and _sha256(record.path) != record.sha256:
        raise DatasetValidationError(f"episode {record.episode_id} hash mismatch")
    try:
        with np.load(record.path, allow_pickle=False) as payload:
            expected = {"pixels", "actions", "terminated", "truncated"}
            if set(payload.files) != expected:
                raise DatasetValidationError(
                    f"episode {record.episode_id} archive fields must be "
                    "pixels/actions/terminated/truncated"
                )
            pixels = np.asarray(payload["pixels"])
            actions = np.asarray(payload["actions"])
            terminated = np.asarray(payload["terminated"])
            truncated = np.asarray(payload["truncated"])
            if pixels.dtype != np.uint8 or pixels.shape != (
                record.count + 1,
                *LARGE_OBSERVATION_SHAPE,
            ):
                raise DatasetValidationError(
                    f"episode {record.episode_id} pixels shape/dtype is invalid"
                )
            if actions.dtype != np.int64 or actions.shape != (record.count,):
                raise DatasetValidationError(
                    f"episode {record.episode_id} actions shape/dtype is invalid"
                )
            if np.any(actions < 0) or np.any(actions >= ACTION_COUNT):
                raise DatasetValidationError(
                    f"episode {record.episode_id} contains an invalid action"
                )
            for name, flags in (("terminated", terminated), ("truncated", truncated)):
                if flags.dtype != np.bool_ or flags.shape != (record.count,):
                    raise DatasetValidationError(
                        f"episode {record.episode_id} {name} shape/dtype is invalid"
                    )
            if np.any(terminated & truncated):
                raise DatasetValidationError(
                    f"episode {record.episode_id} terminates and truncates together"
                )
            if np.any(terminated[:-1]) or np.any(truncated[:-1]):
                raise DatasetValidationError(
                    f"episode {record.episode_id} has an internal boundary"
                )
            if not bool(terminated[-1] or truncated[-1]):
                raise DatasetValidationError(
                    f"episode {record.episode_id} has no final boundary"
                )
            return (
                np.array(pixels, dtype=np.uint8, copy=True),
                np.array(actions, dtype=np.int64, copy=True),
            )
    except DatasetValidationError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise DatasetValidationError(
            f"episode {record.episode_id} cannot be read"
        ) from error


def validate_dataset(
    root: Path,
    *,
    split: str | None = None,
    thorough: bool = True,
    require_ready: bool = True,
) -> dict[str, Any]:
    """Hash and validate every archive one at a time, then return the manifest."""

    if split is not None and split not in LARGE_SPLITS:
        raise ValueError("split must be 'train', 'validation', or None")
    metadata = read_dataset_metadata(Path(root), require_ready=require_ready)
    wanted = LARGE_SPLITS if split is None else (split,)
    if thorough:
        for wanted_split in wanted:
            for record in metadata.records[wanted_split]:
                pixels, actions = _read_episode(record, verify_hash=True)
                del pixels, actions
    return metadata.manifest


class LargePixelSequenceDataset(Dataset[dict[str, object]]):
    """Return fixed-history windows while caching at most four episodes."""

    def __init__(
        self,
        root: Path,
        split: str = "train",
        history_size: int = LARGE_DEFAULT_HISTORY_SIZE,
        cache_size: int = LARGE_DEFAULT_CACHE_SIZE,
        fast_dir: Path | None = None,
    ) -> None:
        if split not in LARGE_SPLITS:
            raise ValueError("split must be 'train' or 'validation'")
        if isinstance(history_size, bool) or not isinstance(
            history_size, (int, np.integer)
        ):
            raise TypeError("history_size must be an integer")
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if isinstance(cache_size, bool) or not isinstance(
            cache_size, (int, np.integer)
        ):
            raise TypeError("cache_size must be an integer")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.root = Path(root)
        self.split = split
        self.history_size = int(history_size)
        self.cache_size = int(cache_size)
        metadata = read_dataset_metadata(self.root)
        self.manifest = metadata.manifest
        self._records = metadata.records[split]
        self._prefix: list[int] = [0]
        for record in self._records:
            self._prefix.append(
                self._prefix[-1] + max(0, record.count - self.history_size + 1)
            )
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._fast_dir = Path(fast_dir) if fast_dir is not None else None
        self._fast_verified: set[int] = set()
        if self._fast_dir is not None:
            self._fast_manifest = _read_fast_manifest(
                self._fast_dir,
                split,
                metadata.manifest,
                _sha256(self.root / "manifest.json"),
            )

    @property
    def episode_count(self) -> int:
        return len(self._records)

    @property
    def cached_episode_ids(self) -> tuple[str, ...]:
        return tuple(self._records[index].episode_id for index in self._cache)

    def cache_info(self) -> CacheInfo:
        return CacheInfo(
            self._cache_hits,
            self._cache_misses,
            self.cache_size,
            len(self._cache),
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return self._prefix[-1]

    def _load_episode(self, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        if self._fast_dir is not None:
            value = self._load_fast_episode(episode_index)
        else:
            value = _read_episode(self._records[episode_index], verify_hash=True)
            value[0].setflags(write=False)
            value[1].setflags(write=False)
        self._cache[episode_index] = value
        self._cache.move_to_end(episode_index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _load_fast_episode(
        self, episode_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load one episode from the uncompressed cache (mmap, zero-copy).

        Converted bytes are hash-verified against the conversion manifest
        once per dataset instance; the conversion itself verified the
        frozen npz sources, so steady-state loads skip decompression and
        per-load hashing entirely.
        """

        assert self._fast_dir is not None
        record = self._records[episode_index]
        entry = self._fast_manifest[record.episode_id]
        pixels_path = self._fast_dir / f"{record.episode_id}.pixels.npy"
        actions_path = self._fast_dir / f"{record.episode_id}.actions.npy"
        if episode_index not in self._fast_verified:
            if _sha256(pixels_path) != entry["pixels_sha256"]:
                raise DatasetValidationError(
                    f"fast cache pixels mismatch: {record.episode_id}"
                )
            if _sha256(actions_path) != entry["actions_sha256"]:
                raise DatasetValidationError(
                    f"fast cache actions mismatch: {record.episode_id}"
                )
            self._fast_verified.add(episode_index)
        pixels = np.load(pixels_path, mmap_mode="r")
        actions = np.load(actions_path, mmap_mode="r")
        if pixels.dtype != np.uint8 or actions.dtype != np.int64:
            raise DatasetValidationError(
                f"fast cache dtype invalid: {record.episode_id}"
            )
        if pixels.shape[0] != record.count + 1 or actions.shape[0] != record.count:
            raise DatasetValidationError(
                f"fast cache shape invalid: {record.episode_id}"
            )
        return pixels, actions

    def __getitem__(self, index: int) -> dict[str, object]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        item = int(index)
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError("dataset index out of range")
        episode_index = bisect.bisect_right(self._prefix, item) - 1
        start = item - self._prefix[episode_index]
        pixels, actions = self._load_episode(episode_index)
        stop = start + self.history_size
        return {
            "pixels": torch.from_numpy(pixels[start : stop + 1].copy()),
            "actions": torch.from_numpy(actions[start:stop].copy()),
            "episode_id": self._records[episode_index].episode_id,
            "start": start,
        }


FAST_CACHE_MANIFEST: Final = "fast-manifest.json"


def _read_fast_manifest(
    fast_dir: Path, split: str, manifest: Mapping[str, Any], manifest_sha256: str
) -> dict[str, dict[str, str]]:
    """Load and cross-check an uncompressed conversion manifest."""

    path = fast_dir / FAST_CACHE_MANIFEST
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            f"fast cache manifest unreadable: {path}"
        ) from error
    if payload.get("split") != split:
        raise DatasetValidationError("fast cache split mismatch")
    if payload.get("dataset_manifest_sha256") != manifest_sha256:
        raise DatasetValidationError("fast cache dataset manifest mismatch")
    records = manifest.get("episodes", {}).get(split, [])
    expected_ids = {
        record["episode_id"] for record in records  # type: ignore[union-attr]
    }
    entries = payload.get("episodes", {})
    if set(entries) != expected_ids:
        raise DatasetValidationError("fast cache episode set mismatch")
    return entries


def materialize_uncompressed(
    root: Path, split: str, dest: Path
) -> dict[str, object]:
    """Convert one split to uncompressed mmap-able arrays (verified once).

    Each source npz is hash-verified against the frozen manifest during
    conversion; converted bytes are recorded so training loads verify each
    file once per dataset instance and then skip decompression entirely.
    Safe to re-run: completed episodes are skipped by digest comparison.
    """

    root, dest = Path(root), Path(dest)
    metadata = read_dataset_metadata(root)
    records = metadata.records[split]
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / FAST_CACHE_MANIFEST
    try:
        existing = json.loads(manifest_path.read_text()).get("episodes", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        existing = {}
    entries: dict[str, dict[str, str]] = {}
    converted = 0
    for record in records:
        pixels_path = dest / f"{record.episode_id}.pixels.npy"
        actions_path = dest / f"{record.episode_id}.actions.npy"
        known = existing.get(record.episode_id, {})
        if (
            pixels_path.is_file()
            and actions_path.is_file()
            and known.get("source_sha256") == record.sha256
            and _sha256(pixels_path) == known.get("pixels_sha256")
            and _sha256(actions_path) == known.get("actions_sha256")
        ):
            entries[record.episode_id] = known
            continue
        pixels, actions = _read_episode(record, verify_hash=True)
        with pixels_path.open("wb") as stream:
            np.save(stream, np.ascontiguousarray(pixels))
        with actions_path.open("wb") as stream:
            np.save(stream, np.ascontiguousarray(actions))
        entries[record.episode_id] = {
            "pixels_sha256": _sha256(pixels_path),
            "actions_sha256": _sha256(actions_path),
            "source_sha256": record.sha256,
        }
        converted += 1
    payload = {
        "split": split,
        "dataset_format": LARGE_DATASET_FORMAT,
        "dataset_manifest_sha256": _sha256(root / "manifest.json"),
        "episodes": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"split": split, "episodes": len(entries), "converted": converted}


def make_large_dataset(
    root: Path,
    *,
    split: str = "train",
    history_size: int = LARGE_DEFAULT_HISTORY_SIZE,
    cache_size: int = LARGE_DEFAULT_CACHE_SIZE,
) -> LargePixelSequenceDataset:
    """Factory used by future training code without changing old trainers."""

    return LargePixelSequenceDataset(
        root, split=split, history_size=history_size, cache_size=cache_size
    )


__all__ = [
    "CacheInfo",
    "DatasetValidationError",
    "FAST_CACHE_MANIFEST",
    "LARGE_DATASET_FORMAT",
    "LARGE_DATASET_SCHEMA_VERSION",
    "LARGE_DEFAULT_CACHE_SIZE",
    "LARGE_DEFAULT_HISTORY_SIZE",
    "LARGE_EPISODE_ACTION_COUNT",
    "LARGE_OBSERVATION_PROFILE",
    "LARGE_OBSERVATION_SHAPE",
    "LARGE_READY_CONTENT",
    "LARGE_READY_MARKER",
    "LARGE_STEP_FRAMES",
    "LargeDatasetMetadata",
    "LargeEpisodeRecord",
    "LargePixelSequenceDataset",
    "make_large_dataset",
    "materialize_uncompressed",
    "read_dataset_metadata",
    "validate_dataset",
]
