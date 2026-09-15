"""Plan and collect the large native scripted-practice corpus.

The planner owns authoring only. Episode NPZ files store native RGB pixels,
executed actions, and episode boundary flags. The loader exposes pixels and
actions as learner inputs; flags validate sequence boundaries. Coordinates,
scripts, recipe names, and other provenance live in JSON sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..cnn_image_ddqn import pixels as _pixels_module
from ..cnn_image_ddqn.pixels import native_rgb_from_result

# ``practice.py`` deliberately keeps the authoring API small.  Importing the
# concrete names separately keeps this module usable with Python 3.11 and
# makes the native contract obvious at its call sites.
from .practice import (  # noqa: E402
    ACTION_NAMES,
    EnemyScript,
    EnemySegment,
    PlayerCommand,
    PracticeConfig,
)

LARGE_SCHEMA_VERSION = 1
LARGE_DATASET_FORMAT = "pixel-repr-ddqn-large-practice-v1"
VARIANT = "pixel-repr-ddqn"
READY_MARKER = "READY"
READY_CONTENT = f"{LARGE_DATASET_FORMAT}\n"
OBSERVATION_PROFILE = "native-rgb-v1"
OBSERVATION_SHAPE = (3, 128, 128)
ACTION_COUNT = 9
STEP_FRAMES = 4
MAX_DECISIONS = 256
DEFAULT_DECISIONS_PER_EPISODE = 128
DEFAULT_TRAIN_EPISODES = 4096
DEFAULT_VALIDATION_EPISODES = 512
DEFAULT_TRAIN_SEED_START = 4000
DEFAULT_VALIDATION_SEED_START = 16000
DEFAULT_PLANNER_SEED = 20260914
MAX_WORKERS = 4
NATIVE_SEED_MAX = 32767
HISTORY_SIZE_DEFAULT = 3

# Sixteen fixed slots make both requested split sizes exactly balanced while
# leaving enough independent recipe variation inside each family.
RECIPE_FAMILIES = (
    "player.stationary",
    "player.cardinal",
    "player.diagonal",
    "player.zigzag",
    "player.stop-go",
    "enemy.static",
    "enemy.static-dense",
    "enemy.moving-horizontal",
    "enemy.moving-vertical",
    "enemy.moving-diagonal",
    "enemy.moving-speed",
    "enemy.moving-size",
    "enemy.moving-many",
    "mixed.static-moving",
    "mixed.moving-many",
    "pattern.permanent",
)
FAMILY_SLOTS = len(RECIPE_FAMILIES)
_MASK64 = (1 << 64) - 1


class LargePracticeError(ValueError):
    """Raised when a large-practice plan or artifact is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_replace(path: Path, payload: bytes) -> None:
    """Replace one control file only after its bytes are fully written."""

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


def _atomic_create(path: Path, payload: bytes) -> None:
    """Create an episode sidecar without replacing an existing artifact."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"large-practice artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"large-practice artifact appeared: {path}")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"large-practice artifact already exists: {path}")
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
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"large-practice artifact appeared: {path}")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _mix(*values: int) -> int:
    """Stable integer mixer; avoids Python's process-randomized hash()."""

    value = 0x9E3779B97F4A7C15
    for item in values:
        value ^= int(item) & _MASK64
        value = (value * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & _MASK64
        value ^= value >> 31
    return value & _MASK64


def _coordinate(salt: int, low: float, high: float) -> float:
    lo, hi = math.ceil(low), math.floor(high)
    if hi < lo:
        raise LargePracticeError(f"invalid coordinate bounds {low}, {high}")
    return float(lo + (_mix(salt) % (hi - lo + 1)))


def _point(salt: int, margin: float = 4.0) -> tuple[float, float]:
    return (
        _coordinate(salt * 2 + 1, margin, 128.0 - margin),
        _coordinate(salt * 2 + 2, margin, 128.0 - margin),
    )


def _bounded_target(
    start: tuple[float, float],
    size: int,
    salt: int,
    axis: int,
    span: int,
) -> tuple[float, float]:
    margin = max(2.0, size / 2.0)
    target = list(start)
    direction = -1.0 if _mix(salt, 7) & 1 else 1.0
    target[axis] = max(margin, min(128.0 - margin, target[axis] + direction * span))
    return (float(target[0]), float(target[1]))


def _enemy(
    salt: int,
    *,
    size: int = 4,
    axis: int = 0,
    span: int = 24,
    frames: int = 64,
    segments: int = 2,
) -> EnemyScript:
    size = int(size)
    start = _point(salt, margin=size / 2.0)
    target = _bounded_target(start, size, salt + 3, axis, span)
    if segments <= 2:
        path = (EnemySegment(target, frames), EnemySegment(start, frames))
    else:
        other_axis = 1 - axis
        target2 = _bounded_target(
            target, size, salt + 11, other_axis, max(8, span // 2)
        )
        path = (
            EnemySegment(target, frames),
            EnemySegment(target2, frames),
            EnemySegment(start, frames),
        )
    return EnemyScript(start=start, size=size, loop=True, segments=path)


def _static_enemies(salt: int, count: int) -> tuple[EnemyScript, ...]:
    result: list[EnemyScript] = []
    for index in range(count):
        size = 3 + int(_mix(salt, index, 31) % 4)
        result.append(
            EnemyScript(
                start=_point(_mix(salt, index, 41), margin=size / 2.0), size=size
            )
        )
    return tuple(result)


def _block_commands(
    actions: Sequence[int], decisions: int
) -> tuple[PlayerCommand, ...]:
    if not actions:
        raise LargePracticeError("player action recipe is empty")
    if decisions < 1:
        raise LargePracticeError("decisions must be positive")
    chunks = len(actions)
    base, remainder = divmod(decisions, chunks)
    commands = []
    for index, action in enumerate(actions):
        budget = base + (1 if index < remainder else 0)
        if budget:
            commands.append(PlayerCommand("action", budget, action=int(action)))
    return tuple(commands)


def _player_commands(
    family: str, variant: int, salt: int, decisions: int
) -> tuple[PlayerCommand, ...]:
    neutral = ACTION_NAMES.index("neutral")
    left = ACTION_NAMES.index("left")
    right = ACTION_NAMES.index("right")
    up = ACTION_NAMES.index("up")
    down = ACTION_NAMES.index("down")
    up_left = ACTION_NAMES.index("up_left")
    up_right = ACTION_NAMES.index("up_right")
    down_left = ACTION_NAMES.index("down_left")
    down_right = ACTION_NAMES.index("down_right")
    if family == "player.stationary":
        actions = (neutral,)
    elif family == "player.cardinal":
        cardinal = (left, right, up, down)
        offset = variant % len(cardinal)
        actions = tuple(
            cardinal[(offset + index) % len(cardinal)] for index in range(8)
        )
    elif family == "player.diagonal":
        diagonal = (up_left, up_right, down_left, down_right)
        offset = variant % len(diagonal)
        actions = tuple(
            diagonal[(offset + index) % len(diagonal)] for index in range(8)
        )
    elif family == "player.zigzag":
        horizontal = (left, right) if variant % 2 else (right, left)
        actions = tuple(horizontal[index % 2] for index in range(8))
    elif family == "player.stop-go":
        moving = (right, left, down, up)[variant % 4]
        reverse = {right: left, left: right, up: down, down: up}[moving]
        actions = (neutral, moving, neutral, reverse, neutral, moving, neutral, reverse)
    elif family in {"mixed.static-moving", "mixed.moving-many"}:
        moving = (right, left, up, down)[variant % 4]
        reverse = {right: left, left: right, up: down, down: up}[moving]
        actions = (neutral, moving, moving, neutral, reverse, reverse, neutral, neutral)
    else:
        # Enemy-focused recipes still expose a varied, ordinary player trace.
        choices = (left, right, up, down, up_left, up_right, down_left, down_right)
        offset = int(_mix(salt, variant) % len(choices))
        actions = tuple(choices[(offset + index) % len(choices)] for index in range(8))
    if len(actions) > 1:
        # Four-decision bursts keep scripted movement from running into a wall
        # for most of a 128-decision episode while staying under the native
        # 64-command authoring limit.
        actions = actions * 4
    return _block_commands(actions, decisions)


@dataclass(frozen=True, slots=True)
class LargePracticeEpisode:
    """One deterministic recipe/seed assignment in the large corpus."""

    split: str
    index: int
    seed: int
    recipe_id: str
    recipe_family: str
    recipe_variant: int
    config: PracticeConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "index": self.index,
            "seed": self.seed,
            "episode_id": f"{self.split}-{self.index:06d}",
            "recipe_id": self.recipe_id,
            "recipe_family": self.recipe_family,
            "recipe_variant": self.recipe_variant,
            "config": self.config.to_dict(),
            "resolved_config_sha256": self.config.resolved_sha256(),
            "config_identity": _config_identity(self.config),
        }

    def __getitem__(self, key: str) -> Any:
        """Permit small callers/loaders to treat a plan item as a mapping."""

        return self.to_dict()[key]


EpisodeSpec = LargePracticeEpisode


def _config_identity(config: PracticeConfig) -> str:
    """Hash authored semantics while ignoring the per-episode display name."""

    payload = config.to_dict()
    payload["name"] = "<recipe>"
    return _sha256_bytes(_canonical_json(payload))


def _validate_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return int(value)


def _validate_seed_start(value: object, field: str, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    end = int(value) + count - 1
    if int(value) < 0 or end > NATIVE_SEED_MAX:
        raise ValueError(f"{field} range must stay within 0..{NATIVE_SEED_MAX}")
    return int(value)


def _validate_split(split: str) -> None:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")


def _make_config(
    split: str,
    index: int,
    seed: int,
    decisions: int,
    planner_seed: int,
) -> LargePracticeEpisode:
    _validate_split(split)
    family = RECIPE_FAMILIES[index % FAMILY_SLOTS]
    variant = index // FAMILY_SLOTS
    split_offset = 0 if split == "train" else 100_003
    salt = _mix(planner_seed, split_offset, index, seed)
    # Player-only recipes remain clean motion controls.  The permanent family
    # cycles native IDs 1..39; enemy/mixed families alternate a clean episode
    # with a permanent-pattern episode, covering the same IDs over variants.
    if family.startswith("player."):
        permanent_pattern = 0
    elif family == "pattern.permanent":
        permanent_pattern = 1 + (variant + (7 if split == "validation" else 0)) % 39
    elif variant % 2:
        permanent_pattern = (
            1 + (variant // 2 + (7 if split == "validation" else 0)) % 39
        )
    else:
        permanent_pattern = 0
    player_start = _point(_mix(salt, 1), margin=8.0)
    enemies: tuple[EnemyScript, ...]
    if family == "enemy.static":
        enemies = _static_enemies(_mix(salt, 101), 1 + variant % 2)
    elif family == "enemy.static-dense":
        enemies = _static_enemies(_mix(salt, 103), 2 + variant % 4)
    elif family == "enemy.moving-horizontal":
        enemies = (_enemy(_mix(salt, 107), axis=0, span=16 + variant % 48, frames=48),)
    elif family == "enemy.moving-vertical":
        enemies = (_enemy(_mix(salt, 109), axis=1, span=16 + variant % 48, frames=64),)
    elif family == "enemy.moving-diagonal":
        enemies = (
            _enemy(
                _mix(salt, 113),
                axis=variant % 2,
                span=20 + variant % 40,
                frames=56,
                segments=3,
            ),
        )
    elif family == "enemy.moving-speed":
        frames = (16, 32, 64, 96)[variant % 4]
        enemies = (
            _enemy(
                _mix(salt, 127), axis=variant % 2, span=24 + variant % 40, frames=frames
            ),
        )
    elif family == "enemy.moving-size":
        size = 2 + variant % 15
        enemies = (
            _enemy(
                _mix(salt, 131),
                size=size,
                axis=variant % 2,
                span=16 + variant % 48,
                frames=64,
            ),
        )
    elif family == "enemy.moving-many":
        count = 2 + variant % 5
        enemies = tuple(
            _enemy(
                _mix(salt, 137, enemy_index),
                size=2 + (variant + enemy_index) % 8,
                axis=(variant + enemy_index) % 2,
                span=12 + (variant + enemy_index) % 40,
                frames=(24, 48, 72)[enemy_index % 3],
            )
            for enemy_index in range(count)
        )
    elif family == "mixed.static-moving":
        enemies = _static_enemies(_mix(salt, 149), 1 + variant % 2) + (
            _enemy(
                _mix(salt, 151), axis=variant % 2, span=20 + variant % 32, frames=64
            ),
        )
    elif family == "mixed.moving-many":
        enemies = _static_enemies(_mix(salt, 157), 1) + tuple(
            _enemy(
                _mix(salt, 163, enemy_index),
                axis=(variant + enemy_index) % 2,
                span=16 + (variant + enemy_index) % 48,
                frames=(32, 64, 96)[enemy_index % 3],
            )
            for enemy_index in range(1 + variant % 3)
        )
    elif family == "pattern.permanent":
        enemies = _static_enemies(_mix(salt, 167), 1 + variant % 3) + (
            _enemy(
                _mix(salt, 173), axis=variant % 2, span=20 + variant % 40, frames=64
            ),
        )
    else:
        enemies = ()
    config = PracticeConfig(
        name=f"large-{split}-{index:06d}",
        max_decisions=decisions,
        # A split-specific native difficulty is part of the recipe identity;
        # it prevents coincident coordinates/action cycles from collapsing
        # train and validation recipes while retaining invulnerability.
        difficulty=1 if split == "train" else 2,
        permanent_pattern=permanent_pattern,
        invulnerable=True,
        player_start=player_start,
        player_script=_player_commands(family, variant, salt, decisions),
        enemies=enemies,
    )
    return LargePracticeEpisode(
        split=split,
        index=index,
        seed=seed,
        recipe_id=f"{split}:{family}:{variant:04d}",
        recipe_family=family,
        recipe_variant=variant,
        config=config,
    )


def plan_large_practice(
    *,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    validation_episodes: int = DEFAULT_VALIDATION_EPISODES,
    decisions_per_episode: int = DEFAULT_DECISIONS_PER_EPISODE,
    train_seed_start: int = DEFAULT_TRAIN_SEED_START,
    validation_seed_start: int = DEFAULT_VALIDATION_SEED_START,
    planner_seed: int = DEFAULT_PLANNER_SEED,
) -> list[LargePracticeEpisode]:
    """Build a deterministic, balanced train/validation recipe plan."""

    train_count = _validate_count(train_episodes, "train_episodes")
    validation_count = _validate_count(validation_episodes, "validation_episodes")
    decisions = _validate_count(decisions_per_episode, "decisions_per_episode")
    if decisions != DEFAULT_DECISIONS_PER_EPISODE:
        raise ValueError(
            "large practice requires exactly "
            f"{DEFAULT_DECISIONS_PER_EPISODE} decisions per episode"
        )
    train_start = _validate_seed_start(
        train_seed_start, "train_seed_start", train_count
    )
    validation_start = _validate_seed_start(
        validation_seed_start, "validation_seed_start", validation_count
    )
    if set(range(train_start, train_start + train_count)).intersection(
        range(validation_start, validation_start + validation_count)
    ):
        raise ValueError("train and validation seed ranges overlap")
    if isinstance(planner_seed, bool) or not isinstance(planner_seed, int):
        raise TypeError("planner_seed must be an integer")
    result: list[LargePracticeEpisode] = []
    identities: set[str] = set()
    for split, count, start in (
        ("train", train_count, train_start),
        ("validation", validation_count, validation_start),
    ):
        for index in range(count):
            for attempt in range(100):
                episode = _make_config(
                    split,
                    index,
                    start + index,
                    decisions,
                    int(planner_seed) + attempt * 1_000_003,
                )
                identity = _config_identity(episode.config)
                if identity not in identities:
                    identities.add(identity)
                    result.append(episode)
                    break
            else:
                raise LargePracticeError("cannot generate a unique practice recipe")
    return result


build_large_practice_plan = plan_large_practice
make_large_practice_plan = plan_large_practice


def _native_binary_info(module: Any) -> tuple[str | None, str | None]:
    path = getattr(module, "__file__", None)
    if not path:
        return None, None
    candidate = Path(path)
    if candidate.suffix not in {".so", ".pyd", ".dll"}:
        nested = getattr(getattr(module, "dodge_native", None), "__file__", None)
        candidate = Path(nested) if nested else candidate
    if candidate.suffix not in {".so", ".pyd", ".dll"} or not candidate.is_file():
        return None, None
    return candidate.as_posix(), _sha256(candidate)


def _collection_provenance(native_factory: Callable[..., Any] | None) -> dict[str, Any]:
    if native_factory is None:
        import dodge_native

        module: Any = dodge_native
    else:
        module = native_factory
    native_path, native_hash = _native_binary_info(module)
    source_paths = {
        "large_practice": Path(__file__).resolve(),
        "practice": Path(__file__).with_name("practice.py").resolve(),
        "pixels": Path(_pixels_module.__file__).resolve(),
    }
    return {
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__)),
        "source_hashes": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "native_module_path": native_path,
        "native_module_sha256": native_hash,
    }


def _native_rgb(value: object) -> np.ndarray:
    raw = value.get("pixels") if isinstance(value, Mapping) else value
    if raw is None:
        raise LargePracticeError("native result is missing pixels")
    array = np.asarray(raw)
    if array.shape == OBSERVATION_SHAPE and array.dtype == np.uint8:
        return np.ascontiguousarray(array, dtype=np.uint8)
    try:
        frame = native_rgb_from_result({"pixels": array})
    except Exception as error:  # native adapter raises its own ValueError type
        raise LargePracticeError(
            "native pixels must be uint8[128,128] palette IDs"
        ) from error
    if frame.shape != OBSERVATION_SHAPE or frame.dtype != np.uint8:
        raise LargePracticeError("native RGB frame must be uint8[3,128,128]")
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _paths(root: Path, spec: LargePracticeEpisode) -> dict[str, Path]:
    episode_id = f"{spec.split}-{spec.index:06d}"
    return {
        "episode": root / "episodes" / spec.split / f"episode-{spec.index:06d}.npz",
        "config": root / "configs" / spec.split / f"episode-{spec.index:06d}.json",
        "provenance": root
        / "provenance"
        / spec.split
        / f"episode-{spec.index:06d}.json",
        "receipt": root / "receipts" / spec.split / f"episode-{spec.index:06d}.json",
        "episode_id": Path(episode_id),
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _make_provenance(
    spec: LargePracticeEpisode,
    module: Any,
    frozen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = _collection_provenance(
        None if module.__name__ == "dodge_native" else module
    )
    provenance = dict(frozen or observed)
    return {
        "schema_version": LARGE_SCHEMA_VERSION,
        "artifact_format": LARGE_DATASET_FORMAT,
        "variant": VARIANT,
        "episode_id": f"{spec.split}-{spec.index:06d}",
        "split": spec.split,
        "seed": spec.seed,
        "recipe_id": spec.recipe_id,
        "recipe_family": spec.recipe_family,
        "generator": {
            "module": provenance["generator_path"],
            "sha256": provenance["generator_sha256"],
        },
        "sources": provenance["source_hashes"],
        "native": {
            "module_path": provenance["native_module_path"],
            "module_sha256": provenance["native_module_sha256"],
        },
        "observation": {
            "profile": OBSERVATION_PROFILE,
            "shape": list(OBSERVATION_SHAPE),
            "dtype": "uint8",
        },
        "action_count": ACTION_COUNT,
        "step_frames": STEP_FRAMES,
        "invulnerable": True,
        "learner_boundary": {
            "arrays": ["pixels", "actions", "terminated", "truncated"],
            "excluded": [
                "coordinates",
                "player_script",
                "enemy_scripts",
                "recipe_id",
                "seed",
                "config",
            ],
        },
    }


def _collect_one(
    spec: LargePracticeEpisode,
    root: Path,
    native_factory: Callable[..., Any] | None = None,
    frozen_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _paths(root, spec)
    sidecar_paths = [
        paths[key] for key in ("episode", "config", "provenance", "receipt")
    ]
    if any(path.exists() or path.is_symlink() for path in sidecar_paths):
        raise FileExistsError(
            "large-practice episode artifacts already exist for "
            f"{spec.split}-{spec.index:06d}"
        )
    requested_factory = native_factory
    module: Any
    if requested_factory is None:
        import dodge_native

        module = dodge_native
        native_factory = dodge_native.NativePracticeEnv
    else:
        module = requested_factory
    observed_provenance = _collection_provenance(requested_factory)
    if frozen_provenance is not None and dict(observed_provenance) != dict(
        frozen_provenance
    ):
        raise LargePracticeError(
            "native/generator provenance changed during collection"
        )
    environment = native_factory(
        step_frames=spec.config.step_frames,
        difficulty=spec.config.difficulty,
        permanent_pattern=spec.config.permanent_pattern,
        invulnerable=spec.config.invulnerable,
        player_start=spec.config.player_start,
        commands=spec.config.native_commands(),
        enemies=spec.config.native_enemies(),
    )
    frames: list[np.ndarray] = []
    actions: list[int] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    finished = failed = False
    try:
        frames.append(_native_rgb(environment.reset(spec.seed)))
        for _ in range(spec.config.max_decisions):
            result = environment.step()
            if not isinstance(result, Mapping) or not {
                "pixels",
                "action",
                "terminated",
                "finished",
                "failed",
            } <= set(result):
                raise LargePracticeError(
                    "native practice step has an invalid result shape"
                )
            action = result["action"]
            if (
                isinstance(action, bool)
                or not isinstance(action, (int, np.integer))
                or not 0 <= int(action) < ACTION_COUNT
            ):
                raise LargePracticeError(
                    "native practice action must be an integer 0..8"
                )
            flags = {key: result[key] for key in ("terminated", "finished", "failed")}
            if any(not isinstance(value, (bool, np.bool_)) for value in flags.values()):
                raise LargePracticeError(
                    "native practice terminal flags must be boolean"
                )
            is_terminated, is_finished, is_failed = (
                bool(flags[key]) for key in ("terminated", "finished", "failed")
            )
            if (is_terminated and (is_finished or is_failed)) or (
                is_finished and is_failed
            ):
                raise LargePracticeError(
                    "native practice returned incompatible terminal flags"
                )
            actions.append(int(action))
            frames.append(_native_rgb(result["pixels"]))
            terminated.append(is_terminated)
            truncated.append(not is_terminated and (is_finished or is_failed))
            finished, failed = is_finished, is_failed
            if is_terminated or is_finished or is_failed:
                break
        if actions and not terminated[-1] and not truncated[-1]:
            truncated[-1] = True
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    if len(actions) != spec.config.max_decisions:
        raise LargePracticeError(
            f"native script {spec.split}-{spec.index:06d} produced "
            f"{len(actions)} decisions; expected {spec.config.max_decisions}"
        )
    if not actions:
        raise LargePracticeError("native practice produced no decisions")
    pixels = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)
    array_payload = {
        "pixels": pixels,
        "actions": np.asarray(actions, dtype=np.int64),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
    }
    _atomic_npz(paths["episode"], array_payload)
    config_payload = {
        "schema_version": LARGE_SCHEMA_VERSION,
        "artifact_format": LARGE_DATASET_FORMAT,
        "variant": VARIANT,
        "episode_id": f"{spec.split}-{spec.index:06d}",
        "split": spec.split,
        "index": spec.index,
        "seed": spec.seed,
        "recipe_id": spec.recipe_id,
        "recipe_family": spec.recipe_family,
        "recipe_variant": spec.recipe_variant,
        "config_identity": _config_identity(spec.config),
        "config": spec.config.to_dict(),
        "practice_config": spec.config.to_dict(),
        "resolved_config_sha256": spec.config.resolved_sha256(),
    }
    _atomic_create(paths["config"], _canonical_json(config_payload))
    provenance_payload = _make_provenance(spec, module, frozen_provenance)
    _atomic_create(paths["provenance"], _canonical_json(provenance_payload))
    receipt = {
        "schema_version": LARGE_SCHEMA_VERSION,
        "artifact_format": LARGE_DATASET_FORMAT,
        "variant": VARIANT,
        "episode_id": f"{spec.split}-{spec.index:06d}",
        "split": spec.split,
        "index": spec.index,
        "seed": spec.seed,
        "recipe_id": spec.recipe_id,
        "recipe_family": spec.recipe_family,
        "recipe_variant": spec.recipe_variant,
        "config_identity": _config_identity(spec.config),
        "path": _relative(paths["episode"], root),
        "sha256": _sha256(paths["episode"]),
        "bytes": paths["episode"].stat().st_size,
        "config_path": _relative(paths["config"], root),
        "config_sha256": _sha256(paths["config"]),
        "resolved_config_sha256": spec.config.resolved_sha256(),
        "provenance_path": _relative(paths["provenance"], root),
        "provenance_sha256": _sha256(paths["provenance"]),
        "transition_count": len(actions),
        "frame_count": len(frames),
        "pixel_shape": list(pixels.shape),
        "action_shape": [len(actions)],
        "terminal": bool(terminated[-1]),
        "truncated": bool(truncated[-1]),
        "finished": bool(finished),
        "failed": bool(failed),
    }
    _atomic_create(paths["receipt"], _canonical_json(receipt))
    receipt["receipt_path"] = _relative(paths["receipt"], root)
    receipt["receipt_sha256"] = _sha256(paths["receipt"])
    return receipt


def collect_large_practice_episode(
    spec: LargePracticeEpisode,
    output: Path | str,
    *,
    native_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Collect one episode, primarily for bounded native smoke tests."""

    return _collect_one(spec, Path(output), native_factory)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LargePracticeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise LargePracticeError(f"{label} must be an object: {path}")
    return value


def _validate_episode(
    root: Path,
    spec: LargePracticeEpisode,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _paths(root, spec)
    expected = [paths[key] for key in ("episode", "config", "provenance", "receipt")]
    if any(not path.is_file() or path.is_symlink() for path in expected):
        raise LargePracticeError(
            f"incomplete artifacts for {spec.split}-{spec.index:06d}"
        )
    receipt = _read_json(paths["receipt"], "episode receipt")
    episode_id = f"{spec.split}-{spec.index:06d}"
    if (
        receipt.get("episode_id") != episode_id
        or receipt.get("split") != spec.split
        or receipt.get("index") != spec.index
        or receipt.get("seed") != spec.seed
    ):
        raise LargePracticeError(f"receipt identity mismatch for {episode_id}")
    if (
        receipt.get("recipe_id") != spec.recipe_id
        or receipt.get("recipe_family") != spec.recipe_family
    ):
        raise LargePracticeError(f"receipt recipe mismatch for {episode_id}")
    if receipt.get("path") != _relative(paths["episode"], root):
        raise LargePracticeError(f"receipt episode path mismatch for {episode_id}")
    if receipt.get("sha256") != _sha256(paths["episode"]):
        raise LargePracticeError(f"episode hash mismatch for {episode_id}")
    if receipt.get("config_sha256") != _sha256(paths["config"]):
        raise LargePracticeError(f"config hash mismatch for {episode_id}")
    if receipt.get("provenance_sha256") != _sha256(paths["provenance"]):
        raise LargePracticeError(f"provenance hash mismatch for {episode_id}")
    config_payload = _read_json(paths["config"], "episode config")
    if (
        config_payload.get("config") != spec.config.to_dict()
        or config_payload.get("practice_config") != spec.config.to_dict()
    ):
        raise LargePracticeError(f"config mismatch for {episode_id}")
    if config_payload.get("resolved_config_sha256") != spec.config.resolved_sha256():
        raise LargePracticeError(f"resolved config hash mismatch for {episode_id}")
    if config_payload.get("config_identity") != _config_identity(spec.config):
        raise LargePracticeError(f"config identity mismatch for {episode_id}")
    provenance = _read_json(paths["provenance"], "episode provenance")
    if provenance.get("episode_id") != episode_id or provenance.get(
        "learner_boundary", {}
    ).get("arrays") != ["pixels", "actions", "terminated", "truncated"]:
        raise LargePracticeError(f"provenance mismatch for {episode_id}")
    if expected_provenance is not None:
        native = provenance.get("native", {})
        generator = provenance.get("generator", {})
        if (
            generator.get("sha256") != expected_provenance.get("generator_sha256")
            or native.get("module_path")
            != expected_provenance.get("native_module_path")
            or native.get("module_sha256")
            != expected_provenance.get("native_module_sha256")
        ):
            raise LargePracticeError(f"provenance source mismatch for {episode_id}")
        sources = provenance.get("sources")
        if sources != expected_provenance.get("source_hashes"):
            raise LargePracticeError(f"provenance dependency mismatch for {episode_id}")
    try:
        with np.load(paths["episode"], allow_pickle=False) as archive:
            if set(archive.files) != {"pixels", "actions", "terminated", "truncated"}:
                raise LargePracticeError(f"episode has unexpected arrays: {episode_id}")
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as error:
        raise LargePracticeError(f"cannot read episode {episode_id}") from error
    pixels, actions = arrays["pixels"], arrays["actions"]
    terminal, capped = arrays["terminated"], arrays["truncated"]
    if pixels.dtype != np.uint8 or pixels.shape != (
        spec.config.max_decisions + 1,
        *OBSERVATION_SHAPE,
    ):
        raise LargePracticeError(f"invalid pixel array for {episode_id}")
    if actions.dtype != np.int64 or actions.shape != (spec.config.max_decisions,):
        raise LargePracticeError(f"invalid action array for {episode_id}")
    if (
        terminal.dtype != np.bool_
        or capped.dtype != np.bool_
        or terminal.shape != actions.shape
        or capped.shape != actions.shape
    ):
        raise LargePracticeError(f"invalid terminal arrays for {episode_id}")
    if (
        np.any(actions < 0)
        or np.any(actions >= ACTION_COUNT)
        or np.any(terminal & capped)
        or np.any(terminal[:-1])
        or np.any(capped[:-1])
    ):
        raise LargePracticeError(f"invalid action/terminal trace for {episode_id}")
    if receipt.get("transition_count") != int(actions.shape[0]) or receipt.get(
        "frame_count"
    ) != int(pixels.shape[0]):
        raise LargePracticeError(f"episode count metadata mismatch for {episode_id}")
    if receipt.get("pixel_shape") != list(pixels.shape) or receipt.get(
        "action_shape"
    ) != list(actions.shape):
        raise LargePracticeError(f"episode shape metadata mismatch for {episode_id}")
    return {
        "episode_id": episode_id,
        "id": episode_id,
        "split": spec.split,
        "index": spec.index,
        "seed": spec.seed,
        "path": _relative(paths["episode"], root),
        "sha256": _sha256(paths["episode"]),
        "hash": _sha256(paths["episode"]),
        "bytes": paths["episode"].stat().st_size,
        "transition_count": int(actions.shape[0]),
        "count": int(actions.shape[0]),
        "frame_count": int(pixels.shape[0]),
        "pixel_shape": list(pixels.shape),
        "action_shape": list(actions.shape),
        "terminal": bool(terminal[-1]),
        "truncated": bool(capped[-1]),
        "recipe_id": spec.recipe_id,
        "recipe_family": spec.recipe_family,
        "recipe_variant": spec.recipe_variant,
        "config_path": _relative(paths["config"], root),
        "config_sha256": _sha256(paths["config"]),
        "resolved_config_sha256": spec.config.resolved_sha256(),
        "config_identity": _config_identity(spec.config),
        "provenance_path": _relative(paths["provenance"], root),
        "provenance_sha256": _sha256(paths["provenance"]),
        "receipt_path": _relative(paths["receipt"], root),
        "receipt_sha256": _sha256(paths["receipt"]),
    }


def _plan_metadata(
    plan: Sequence[LargePracticeEpisode],
    planner_seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    by_split: dict[str, list[LargePracticeEpisode]] = {"train": [], "validation": []}
    for spec in plan:
        by_split[spec.split].append(spec)
    return {
        "planner_version": 1,
        "planner_seed": planner_seed,
        "provenance": dict(provenance),
        "decisions_per_episode": plan[0].config.max_decisions,
        "recipe_families": list(RECIPE_FAMILIES),
        "seed_ranges": {
            split: [specs[0].seed, specs[-1].seed] for split, specs in by_split.items()
        },
        "episodes": {split: len(specs) for split, specs in by_split.items()},
        "recipe_ids": {
            split: [spec.recipe_id for spec in specs]
            for split, specs in by_split.items()
        },
    }


def _validate_or_write_plan(
    root: Path,
    plan: Sequence[LargePracticeEpisode],
    planner_seed: int,
    provenance: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": LARGE_SCHEMA_VERSION,
        "artifact_format": LARGE_DATASET_FORMAT,
        "variant": VARIANT,
        "plan": _plan_metadata(plan, planner_seed, provenance),
    }
    path = root / "plan.json"
    if path.exists():
        existing = _read_json(path, "large-practice plan")
        if existing != expected:
            raise LargePracticeError(
                "existing large-practice plan does not match requested protocol"
            )
    else:
        _atomic_create(path, _canonical_json(expected))


def _quarantine_partial(
    root: Path, spec: LargePracticeEpisode, paths: Sequence[Path]
) -> None:
    """Move an interrupted episode aside so a later run can collect it safely."""

    destination = root / "quarantine" / spec.split / f"episode-{spec.index:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists() or path.is_symlink():
            target = destination / path.name
            if target.exists() or target.is_symlink():
                target = destination / f"{path.name}.{_sha256(path)[:12]}"
            os.replace(path, target)


def _validate_with_sibling_loader(root: Path) -> None:
    """Validate the full corpus before publishing its completion marker."""
    from .large_dataset import validate_dataset

    validate_dataset(root, require_ready=False)


def collect_large_practice(
    output: Path | str,
    *,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    validation_episodes: int = DEFAULT_VALIDATION_EPISODES,
    decisions_per_episode: int = DEFAULT_DECISIONS_PER_EPISODE,
    train_seed_start: int = DEFAULT_TRAIN_SEED_START,
    validation_seed_start: int = DEFAULT_VALIDATION_SEED_START,
    planner_seed: int = DEFAULT_PLANNER_SEED,
    workers: int = 1,
    native_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Incrementally collect and atomically publish the large practice set.

    A partial root is intentionally retained after an exception.  A later
    invocation verifies completed receipts, collects only missing episodes,
    and never replaces an existing episode, config, provenance, or receipt.
    """

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAX_WORKERS
    ):
        raise ValueError(f"workers must be an integer in 1..{MAX_WORKERS}")
    if workers > 1 and native_factory is not None:
        raise ValueError("native_factory injection requires workers=1")
    provenance = _collection_provenance(native_factory)
    plan = plan_large_practice(
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        decisions_per_episode=decisions_per_episode,
        train_seed_start=train_seed_start,
        validation_seed_start=validation_seed_start,
        planner_seed=planner_seed,
    )
    root = Path(output)
    if root.is_symlink():
        raise FileExistsError(f"large-practice output is a symlink: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / f".{root.name}.large-practice.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"large-practice output is being collected: {root}"
        ) from error
    try:
        if root.exists() and not root.is_dir():
            raise FileExistsError(f"large-practice output is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        if (root / READY_MARKER).exists():
            raise FileExistsError(f"large-practice output is already complete: {root}")
        _validate_or_write_plan(root, plan, planner_seed, provenance)
        pending: list[LargePracticeEpisode] = []
        for spec in plan:
            paths = _paths(root, spec)
            artifacts = [
                paths[key] for key in ("episode", "config", "provenance", "receipt")
            ]
            existing = [
                path for path in artifacts if path.exists() or path.is_symlink()
            ]
            if not existing:
                pending.append(spec)
                continue
            if len(existing) != len(artifacts):
                _quarantine_partial(root, spec, existing)
                pending.append(spec)
                continue
            _validate_episode(root, spec, provenance)
        if workers == 1:
            for spec in pending:
                _collect_one(spec, root, native_factory, provenance)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool:
                futures = [
                    pool.submit(_collect_one, spec, root, None, provenance)
                    for spec in pending
                ]
                for future in as_completed(futures):
                    future.result()
        records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        for spec in plan:
            records[spec.split].append(_validate_episode(root, spec, provenance))
        seen_hashes: dict[str, str] = {}
        seen_config_identities: dict[str, str] = {}
        for split, rows in records.items():
            for record in rows:
                episode_id = str(record["episode_id"])
                digest = str(record["sha256"])
                previous = seen_hashes.get(digest)
                if previous is not None and previous != split:
                    raise LargePracticeError(
                        "identical episode bytes appear in train and validation: "
                        f"{episode_id} / {previous}"
                    )
                seen_hashes[digest] = split
                identity = str(record["config_identity"])
                previous = seen_config_identities.get(identity)
                if previous is not None and previous != split:
                    raise LargePracticeError(
                        f"config identity is shared across splits: {identity}"
                    )
                seen_config_identities[identity] = split
        recipe_counts = Counter(
            record["recipe_family"] for rows in records.values() for record in rows
        )
        split_counts = {
            split: {
                "episode_count": len(rows),
                "episode_ids": [record["episode_id"] for record in rows],
                "seeds": [record["seed"] for record in rows],
                "transition_count": sum(record["transition_count"] for record in rows),
            }
            for split, rows in records.items()
        }
        manifest = {
            "schema_version": LARGE_SCHEMA_VERSION,
            "dataset_format": LARGE_DATASET_FORMAT,
            "variant": VARIANT,
            "status": "complete",
            "ready_marker": READY_MARKER,
            "observation": {
                "profile": OBSERVATION_PROFILE,
                "shape": list(OBSERVATION_SHAPE),
                "dtype": "uint8",
            },
            "action_count": ACTION_COUNT,
            "step_frames": STEP_FRAMES,
            "history_size_default": HISTORY_SIZE_DEFAULT,
            "episode_action_count": DEFAULT_DECISIONS_PER_EPISODE,
            "decisions_per_episode": decisions_per_episode,
            "plan": _plan_metadata(plan, planner_seed, provenance),
            "provenance": dict(provenance),
            "collection": {
                "policy": "deterministic-scripted-native-v1",
                "workers": workers,
                "native_decisions": sum(
                    record["transition_count"]
                    for rows in records.values()
                    for record in rows
                ),
                "episode_count": len(plan),
                "no_gifs": True,
            },
            "recipe_counts": dict(sorted(recipe_counts.items())),
            "split_counts": {
                split: info["episode_count"] for split, info in split_counts.items()
            },
            "seeds": {split: info["seeds"] for split, info in split_counts.items()},
            "splits": split_counts,
            "episodes": records,
        }
        _atomic_replace(root / "manifest.json", _canonical_json(manifest))
        _validate_with_sibling_loader(root)
        # READY is the last publication step.  Readers reject a root without
        # this marker even if a process stopped after writing manifest.json.
        _atomic_replace(root / READY_MARKER, READY_CONTENT.encode("utf-8"))
        return manifest
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument(
        "--validation-episodes", type=int, default=DEFAULT_VALIDATION_EPISODES
    )
    parser.add_argument(
        "--decisions-per-episode", type=int, default=DEFAULT_DECISIONS_PER_EPISODE
    )
    parser.add_argument(
        "--train-seed-start", type=int, default=DEFAULT_TRAIN_SEED_START
    )
    parser.add_argument(
        "--validation-seed-start", type=int, default=DEFAULT_VALIDATION_SEED_START
    )
    parser.add_argument("--planner-seed", type=int, default=DEFAULT_PLANNER_SEED)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = collect_large_practice(
        args.output,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        decisions_per_episode=args.decisions_per_episode,
        train_seed_start=args.train_seed_start,
        validation_seed_start=args.validation_seed_start,
        planner_seed=args.planner_seed,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "dataset_format": manifest["dataset_format"],
                "episode_count": manifest["collection"]["episode_count"],
                "native_decisions": manifest["collection"]["native_decisions"],
                "splits": {
                    split: {
                        "episode_count": details["episode_count"],
                        "transition_count": details["transition_count"],
                    }
                    for split, details in manifest["splits"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ACTION_COUNT",
    "DEFAULT_DECISIONS_PER_EPISODE",
    "DEFAULT_PLANNER_SEED",
    "DEFAULT_TRAIN_EPISODES",
    "DEFAULT_TRAIN_SEED_START",
    "DEFAULT_VALIDATION_EPISODES",
    "DEFAULT_VALIDATION_SEED_START",
    "EpisodeSpec",
    "LARGE_DATASET_FORMAT",
    "LARGE_SCHEMA_VERSION",
    "LargePracticeEpisode",
    "LargePracticeError",
    "RECIPE_FAMILIES",
    "build_large_practice_plan",
    "collect_large_practice",
    "collect_large_practice_episode",
    "main",
    "make_large_practice_plan",
    "plan_large_practice",
]


if __name__ == "__main__":
    raise SystemExit(main())
