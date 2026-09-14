"""Validate and capture bounded, scripted native practice episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..cnn_image_ddqn.pixels import native_rgb_from_result

PRACTICE_VERSION = 1
STEP_FRAMES = 4
MAX_DECISIONS = 256
MAX_COMMANDS = 64
MAX_ENEMIES = 32
MAX_SEGMENTS = 64
HISTORY_SIZE = 3
OBSERVATION_SHAPE = (3, 128, 128)
ACTION_NAMES = (
    "neutral",
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TOP_KEYS = {
    "version",
    "name",
    "step_frames",
    "max_decisions",
    "difficulty",
    "permanent_pattern",
    "invulnerable",
    "player_start",
    "player_script",
    "enemies",
}
Point = tuple[float, float]


class PracticeConfigError(ValueError):
    """Raised for an invalid version-1 practice document."""


def _int(
    value: object,
    field: str,
    low: int | None = None,
    high: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PracticeConfigError(f"{field} must be an integer")
    result = int(value)
    if (low is not None and result < low) or (high is not None and result > high):
        bounds = (
            f"{low if low is not None else '-inf'}.."
            f"{high if high is not None else 'inf'}"
        )
        raise PracticeConfigError(f"{field} must be in {bounds}")
    return result


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PracticeConfigError(f"{field} must contain numbers")
    result = float(value)
    if not math.isfinite(result):
        raise PracticeConfigError(f"{field} must contain finite numbers")
    return result


def _point(value: object, field: str, low: float, high: float) -> Point:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise PracticeConfigError(f"{field} must be [x, y]")
    result = (
        _number(value[0], f"{field}[0]"),
        _number(value[1], f"{field}[1]"),
    )
    if not all(low <= coordinate <= high for coordinate in result):
        raise PracticeConfigError(
            f"{field} coordinates must be between {low:g} and {high:g}"
        )
    return result


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise PracticeConfigError(f"{field} must be a boolean")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes((payload + "\n").encode())


@dataclass(frozen=True, slots=True)
class PlayerCommand:
    kind: str
    decisions: int
    action: int | None = None
    target: Point | None = None

    def __post_init__(self) -> None:
        _int(self.decisions, "command decisions", 1, MAX_DECISIONS)
        if self.kind == "action":
            _int(self.action, "action index", 0, 8)
            if self.target is not None:
                raise PracticeConfigError("action command cannot have move_to")
        elif self.kind == "move_to":
            if self.action is not None or self.target is None:
                raise PracticeConfigError("move_to command requires a target")
            object.__setattr__(
                self,
                "target",
                _point(self.target, "move_to", 4.0, 124.0),
            )
        else:
            raise PracticeConfigError("command kind must be action or move_to")

    def native_tuple(self) -> tuple[int, float | int, float, int]:
        if self.kind == "action":
            assert self.action is not None
            return (0, self.action, 0.0, self.decisions)
        assert self.target is not None
        return (1, self.target[0], self.target[1], self.decisions)

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "action":
            assert self.action is not None
            return {"action": ACTION_NAMES[self.action], "decisions": self.decisions}
        assert self.target is not None
        return {"move_to": list(self.target), "max_decisions": self.decisions}


@dataclass(frozen=True, slots=True)
class EnemySegment:
    target: Point
    frames: int

    def __post_init__(self) -> None:
        _int(self.frames, "enemy segment frames", 1, 3600)
        object.__setattr__(
            self,
            "target",
            _point(self.target, "enemy segment target", -math.inf, math.inf),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"to": list(self.target), "frames": self.frames}


@dataclass(frozen=True, slots=True)
class EnemyScript:
    start: Point
    size: int = 4
    loop: bool = False
    segments: tuple[EnemySegment, ...] = ()

    def __post_init__(self) -> None:
        _int(self.size, "enemy size", 2, 16)
        if not isinstance(self.loop, bool):
            raise PracticeConfigError("enemy loop must be a boolean")
        object.__setattr__(
            self,
            "start",
            _point(self.start, "enemy start", -math.inf, math.inf),
        )
        object.__setattr__(self, "segments", tuple(self.segments))
        if len(self.segments) > MAX_SEGMENTS:
            raise PracticeConfigError(
                f"an enemy may have at most {MAX_SEGMENTS} segments"
            )
        low, high = self.size / 2.0, 128.0 - self.size / 2.0
        _point(self.start, "enemy start", low, high)
        for index, segment in enumerate(self.segments):
            if not isinstance(segment, EnemySegment):
                raise PracticeConfigError(f"enemy segment {index} is invalid")
            _point(segment.target, f"enemy segment {index} to", low, high)
        if self.loop and (not self.segments or self.segments[-1].target != self.start):
            raise PracticeConfigError(
                "looping enemy paths must end at their start position"
            )

    def native_tuple(
        self,
    ) -> tuple[float, float, float, bool, list[tuple[float, float, int]]]:
        return (
            self.start[0],
            self.start[1],
            float(self.size),
            self.loop,
            [(s.target[0], s.target[1], s.frames) for s in self.segments],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": list(self.start),
            "size": self.size,
            "loop": self.loop,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class PracticeConfig:
    version: int = PRACTICE_VERSION
    name: str = "practice"
    step_frames: int = STEP_FRAMES
    max_decisions: int = 1
    difficulty: int = 1
    permanent_pattern: int = 0
    invulnerable: bool = True
    player_start: Point = (64.0, 112.0)
    player_script: tuple[PlayerCommand, ...] = ()
    enemies: tuple[EnemyScript, ...] = ()

    def __post_init__(self) -> None:
        _int(self.version, "version", PRACTICE_VERSION, PRACTICE_VERSION)
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise PracticeConfigError("name must be 1..64 ASCII identifier characters")
        _int(self.step_frames, "step_frames", STEP_FRAMES, STEP_FRAMES)
        _int(self.max_decisions, "max_decisions", 1, MAX_DECISIONS)
        _int(self.difficulty, "difficulty", 1, 3)
        _int(self.permanent_pattern, "permanent_pattern", 0, 39)
        _bool(self.invulnerable, "invulnerable")
        object.__setattr__(
            self,
            "player_start",
            _point(self.player_start, "player_start", 4.0, 124.0),
        )
        object.__setattr__(self, "player_script", tuple(self.player_script))
        object.__setattr__(self, "enemies", tuple(self.enemies))
        if not self.player_script or len(self.player_script) > MAX_COMMANDS:
            raise PracticeConfigError(
                f"player_script must contain 1..{MAX_COMMANDS} commands"
            )
        if any(
            not isinstance(command, PlayerCommand) for command in self.player_script
        ):
            raise PracticeConfigError("player_script contains an invalid command")
        if (
            sum(command.decisions for command in self.player_script)
            > self.max_decisions
        ):
            raise PracticeConfigError("player command budgets exceed max_decisions")
        if len(self.enemies) > MAX_ENEMIES:
            raise PracticeConfigError(f"at most {MAX_ENEMIES} enemies are allowed")
        if any(not isinstance(enemy, EnemyScript) for enemy in self.enemies):
            raise PracticeConfigError("enemies contains an invalid actor")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PracticeConfig:
        if not isinstance(raw, Mapping):
            raise PracticeConfigError("practice TOML must contain a table")
        unknown = sorted(set(raw) - _TOP_KEYS)
        if unknown:
            raise PracticeConfigError(f"unknown practice key(s): {', '.join(unknown)}")
        missing = sorted((_TOP_KEYS - {"enemies"}) - set(raw))
        if missing:
            raise PracticeConfigError(f"missing practice key(s): {', '.join(missing)}")
        return cls(
            version=_int(raw["version"], "version"),
            name=raw["name"],
            step_frames=_int(raw["step_frames"], "step_frames"),
            max_decisions=_int(raw["max_decisions"], "max_decisions"),
            difficulty=_int(raw["difficulty"], "difficulty"),
            permanent_pattern=_int(raw["permanent_pattern"], "permanent_pattern"),
            invulnerable=_bool(raw["invulnerable"], "invulnerable"),
            player_start=_point(raw["player_start"], "player_start", 4.0, 124.0),
            player_script=_parse_commands(raw["player_script"]),
            enemies=_parse_enemies(raw.get("enemies", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "step_frames": self.step_frames,
            "max_decisions": self.max_decisions,
            "difficulty": self.difficulty,
            "permanent_pattern": self.permanent_pattern,
            "invulnerable": self.invulnerable,
            "player_start": list(self.player_start),
            "player_script": [command.to_dict() for command in self.player_script],
            "enemies": [enemy.to_dict() for enemy in self.enemies],
        }

    def resolved_sha256(self) -> str:
        return _resolved_hash(self.to_dict())

    def native_commands(self) -> list[tuple[int, float | int, float, int]]:
        return [command.native_tuple() for command in self.player_script]

    def native_enemies(
        self,
    ) -> list[tuple[float, float, float, bool, list[tuple[float, float, int]]]]:
        return [enemy.native_tuple() for enemy in self.enemies]


def _parse_commands(value: object) -> tuple[PlayerCommand, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PracticeConfigError("player_script must be an array of tables")
    commands = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise PracticeConfigError(f"player_script[{index}] must be a table")
        keys, field = set(item), f"player_script[{index}]"
        if "action" in item:
            if keys != {"action", "decisions"} or item["action"] not in ACTION_INDEX:
                raise PracticeConfigError(f"{field} must be action + decisions")
            commands.append(
                PlayerCommand(
                    "action",
                    _int(item["decisions"], f"{field}.decisions"),
                    ACTION_INDEX[item["action"]],
                )
            )
        elif keys == {"move_to", "max_decisions"}:
            commands.append(
                PlayerCommand(
                    "move_to",
                    _int(item["max_decisions"], f"{field}.max_decisions"),
                    target=_point(item["move_to"], f"{field}.move_to", 4.0, 124.0),
                )
            )
        else:
            raise PracticeConfigError(
                f"{field} must be action + decisions or move_to + max_decisions"
            )
    return tuple(commands)


def _parse_enemies(value: object) -> tuple[EnemyScript, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PracticeConfigError("enemies must be an array of tables")
    enemies = []
    for index, item in enumerate(value):
        field = f"enemies[{index}]"
        if not isinstance(item, Mapping) or set(item) - {
            "start",
            "size",
            "loop",
            "segments",
        }:
            raise PracticeConfigError(f"{field} has unknown keys or is not a table")
        if "start" not in item:
            raise PracticeConfigError(f"{field}.start is required")
        size = _int(item.get("size", 4), f"{field}.size", 2, 16)
        low, high = size / 2.0, 128.0 - size / 2.0
        raw_segments = item.get("segments", [])
        if isinstance(raw_segments, (str, bytes)) or not isinstance(
            raw_segments, Sequence
        ):
            raise PracticeConfigError(f"{field}.segments must be an array of tables")
        segments = []
        for segment_index, raw in enumerate(raw_segments):
            segment_field = f"{field}.segments[{segment_index}]"
            if not isinstance(raw, Mapping) or set(raw) != {"to", "frames"}:
                raise PracticeConfigError(f"{segment_field} must contain to and frames")
            segments.append(
                EnemySegment(
                    _point(raw["to"], f"{segment_field}.to", low, high),
                    _int(raw["frames"], f"{segment_field}.frames", 1, 3600),
                )
            )
        enemies.append(
            EnemyScript(
                _point(item["start"], f"{field}.start", low, high),
                size,
                _bool(item.get("loop", False), f"{field}.loop"),
                tuple(segments),
            )
        )
    return tuple(enemies)


def load_practice(path: Path | str) -> PracticeConfig:
    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise PracticeConfigError(f"invalid practice TOML: {source}") from error
    return PracticeConfig.from_mapping(raw)


def _native_frame(value: object) -> np.ndarray:
    if isinstance(value, Mapping):
        value = value.get("pixels")
    if value is None:
        raise RuntimeError("native practice result is missing pixels")
    frame = native_rgb_from_result({"pixels": value})
    if frame.shape != OBSERVATION_SHAPE or frame.dtype != np.uint8:
        raise RuntimeError("native practice RGB frame must be uint8[3,128,128]")
    return np.array(frame, dtype=np.uint8, copy=True, order="C")


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def _write_png(path: Path, frame: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.moveaxis(frame, 0, -1)).save(path, format="PNG")


def _write_gif(path: Path, frames: np.ndarray) -> None:
    from PIL import Image

    images = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in frames]
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=67,
        loop=0,
    )


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _resolve_config(
    config: PracticeConfig | Path | str,
) -> tuple[PracticeConfig, Path | None, bytes | None, str | None]:
    if isinstance(config, PracticeConfig):
        return config, None, None, None
    path = Path(config)
    data = path.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeError) as error:
        raise PracticeConfigError(f"invalid practice TOML: {path}") from error
    return PracticeConfig.from_mapping(raw), path, data, _sha256_bytes(data)


def _native_binary_info(module: Any) -> tuple[str | None, str | None]:
    path = getattr(module, "__file__", None)
    if not path:
        return None, None
    candidate = Path(path)
    if candidate.suffix not in {".so", ".pyd", ".dll"}:
        loaded = getattr(getattr(module, "dodge_native", None), "__file__", None)
        candidate = Path(loaded) if loaded else candidate
    if candidate.suffix not in {".so", ".pyd", ".dll"} or not candidate.is_file():
        return None, None
    return candidate.as_posix(), file_sha256(candidate)


def generate_practice(
    config: PracticeConfig | Path | str,
    output: Path | str,
    seed: int,
    *,
    native_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one script and write pixels, actions, visual clips, and provenance."""

    resolved, source_path, source_bytes, source_hash = _resolve_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 32767:
        raise ValueError("seed must be an integer between 0 and 32767")
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"practice output already exists: {root}")
    root.mkdir(parents=True)

    native_path = native_hash = None
    if native_factory is None:
        try:
            import dodge_native
        except ImportError as error:  # pragma: no cover - real native required
            raise RuntimeError(
                "dodge_native is required for practice generation"
            ) from error
        native_factory = dodge_native.NativePracticeEnv
        native_path, native_hash = _native_binary_info(dodge_native)
    if source_bytes is not None:
        (root / "source.toml").write_bytes(source_bytes)
    environment = native_factory(
        step_frames=resolved.step_frames,
        difficulty=resolved.difficulty,
        permanent_pattern=resolved.permanent_pattern,
        invulnerable=resolved.invulnerable,
        player_start=resolved.player_start,
        commands=resolved.native_commands(),
        enemies=resolved.native_enemies(),
    )

    frames: list[np.ndarray] = []
    actions: list[int] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    finished = failed = native_terminated = False
    try:
        frames.append(_native_frame(environment.reset(seed)))
        for _ in range(resolved.max_decisions):
            result = environment.step()
            if not isinstance(result, Mapping) or not {"pixels", "action"} <= set(
                result
            ):
                raise RuntimeError("native practice step must return pixels and action")
            action = result["action"]
            if (
                isinstance(action, bool)
                or not isinstance(action, (int, np.integer))
                or not 0 <= action < 9
            ):
                raise RuntimeError("native practice action must be an integer 0..8")
            flags = {
                key: result.get(key) for key in ("terminated", "finished", "failed")
            }
            if any(not isinstance(value, (bool, np.bool_)) for value in flags.values()):
                raise RuntimeError("native practice terminal flags must be boolean")
            is_terminated, is_finished, is_failed = (
                bool(flags[key]) for key in ("terminated", "finished", "failed")
            )
            if (
                is_terminated
                and (is_finished or is_failed)
                or is_finished
                and is_failed
            ):
                raise RuntimeError(
                    "native practice returned incompatible terminal flags"
                )
            actions.append(int(action))
            frames.append(_native_frame(result["pixels"]))
            terminated.append(is_terminated)
            truncated.append(not is_terminated and (is_finished or is_failed))
            native_terminated |= is_terminated
            finished, failed = is_finished, is_failed
            if is_terminated or is_finished or is_failed:
                break
        else:
            truncated[-1] = True
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    if not actions:
        raise RuntimeError("native practice produced no decisions")

    episode_pixels = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)
    episode_actions = np.asarray(actions, dtype=np.int64)
    episode_terminated = np.asarray(terminated, dtype=np.bool_)
    episode_truncated = np.asarray(truncated, dtype=np.bool_)
    success = bool(finished and not failed and not native_terminated)
    failure = None if success else "terminated" if native_terminated else "timeout"
    _write_npz(
        root / "episode.npz",
        pixels=episode_pixels,
        actions=episode_actions,
        terminated=episode_terminated,
        truncated=episode_truncated,
    )
    _write_npz(root / "initial.npz", pixels=episode_pixels[0])
    clip_start = max(0, len(frames) - HISTORY_SIZE - 1)
    _write_npz(
        root / "final_clip.npz",
        pixels=episode_pixels[clip_start:],
        actions=episode_actions[clip_start:],
        terminated=episode_terminated[clip_start:],
        truncated=episode_truncated[clip_start:],
    )
    _write_png(root / "initial.png", episode_pixels[0])
    _write_png(root / "final.png", episode_pixels[-1])
    _write_gif(root / "replay.gif", episode_pixels)
    if success:
        _write_gif(root / "goal.gif", episode_pixels[clip_start:])

    filenames = {
        "episode": "episode.npz",
        "initial": "initial.npz",
        "final_clip": "final_clip.npz",
        "initial_png": "initial.png",
        "final_png": "final.png",
        "replay_gif": "replay.gif",
    }
    if source_bytes is not None:
        filenames["source"] = "source.toml"
    if success:
        filenames["goal_gif"] = "goal.gif"
    files = {
        name: _file_record(root / filename, root)
        for name, filename in filenames.items()
    }
    manifest = {
        "schema_version": 1,
        "artifact_format": "pixel-repr-ddqn-practice-v1",
        "variant": "pixel-repr-ddqn",
        "status": "complete",
        "seed": seed,
        "config": resolved.to_dict(),
        "config_sha256": resolved.resolved_sha256(),
        "source_path": source_path.as_posix() if source_path else None,
        "source_sha256": source_hash,
        "generator_sha256": file_sha256(Path(__file__)),
        "native_module_path": native_path,
        "native_module_sha256": native_hash,
        "observation": {
            "profile": "native-rgb-v1",
            "shape": list(OBSERVATION_SHAPE),
            "dtype": "uint8",
        },
        "history_size": HISTORY_SIZE,
        "step_frames": resolved.step_frames,
        "decision_count": len(actions),
        "action_trace": actions,
        "terminated": native_terminated,
        "truncated": bool(episode_truncated[-1]),
        "finished": finished,
        "failed": failed,
        "success": success,
        "failure": failure,
        "goal": {
            "valid": success,
            "frame_index": len(frames) - 1,
            "frame": "final.png" if success else None,
            "clip": "goal.gif" if success else None,
        },
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = generate_practice(args.config, args.output, args.seed)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("success", "failure", "decision_count", "goal")
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ACTION_INDEX",
    "ACTION_NAMES",
    "EnemyScript",
    "EnemySegment",
    "MAX_COMMANDS",
    "MAX_DECISIONS",
    "MAX_ENEMIES",
    "MAX_SEGMENTS",
    "PRACTICE_VERSION",
    "PlayerCommand",
    "PracticeConfig",
    "PracticeConfigError",
    "file_sha256",
    "generate_practice",
    "load_practice",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
