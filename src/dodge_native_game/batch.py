"""Owned NumPy data exposures over the native Rust Dodge batch engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import dodge_native
import numpy as np

ACTION_COUNT = 9
BATCH_SCHEMA_VERSION = int(dodge_native.BATCH_SCHEMA_VERSION)
BOARD_SHAPE = tuple(int(value) for value in dodge_native.BOARD_SHAPE)
PIXEL_SHAPE = tuple(int(value) for value in dodge_native.PIXEL_SHAPE)
ML_OBSERVATION_SHAPE = tuple(
    int(value) for value in dodge_native.ML_OBSERVATION_SHAPE
)
COLLISION_IMAGE_SHAPE = tuple(
    int(value) for value in dodge_native.COLLISION_IMAGE_SHAPE
)
COLLISION_IMAGE_OBSERVATION_VERSION = int(
    dodge_native.COLLISION_IMAGE_OBSERVATION_VERSION
)
COLLISION_IMAGE_CONFIG = str(dodge_native.COLLISION_IMAGE_CONFIG)
HAZARD_CHANNELS = int(dodge_native.HAZARD_CHANNELS)
HAZARD_SCALARS = int(dodge_native.HAZARD_SCALARS)

Payload = Mapping[str, Any]


def _array(
    payload: Payload, name: str, *, dtype: np.dtype[Any] | None = None
) -> np.ndarray:
    value = payload.get(name)
    if value is None:
        raise RuntimeError(f"native payload is missing {name!r}")
    result = np.asarray(value, dtype=dtype)
    return np.array(result, copy=True)


def _optional_array(
    payload: Payload, name: str, *, dtype: np.dtype[Any] | None = None
) -> np.ndarray | None:
    value = payload.get(name)
    if value is None:
        return None
    result = np.asarray(value, dtype=dtype)
    return np.array(result, copy=True)


def _bytes_tuple(payload: Payload, name: str) -> tuple[bytes | None, ...]:
    values = payload.get(name)
    if not isinstance(values, Sequence):
        raise RuntimeError(f"native payload {name!r} is not a sequence")
    result: list[bytes | None] = []
    for value in values:
        if value is None:
            result.append(None)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            result.append(bytes(value))
        else:
            raise RuntimeError(f"native payload {name!r} contains non-bytes data")
    return tuple(result)


def _integer_array(values: object, name: str, *, maximum: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(result < 0) or np.any(result > maximum):
        raise ValueError(f"{name} values must be between 0 and {maximum}")
    return np.ascontiguousarray(result, dtype=np.uint32)


def _action_array(actions: object) -> np.ndarray:
    return _integer_array(actions, "actions", maximum=ACTION_COUNT - 1).astype(
        np.uint8, copy=False
    )


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Full native batch result with owned NumPy arrays."""

    lane_ids: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    state_hashes: np.ndarray
    pixel_hashes: np.ndarray
    modes: np.ndarray
    event_flags: np.ndarray
    shattered: np.ndarray
    score: np.ndarray
    pixels: np.ndarray | None
    board: np.ndarray | None
    collision_image: np.ndarray | None
    ml_observation: np.ndarray | None
    player_positions: np.ndarray | None
    snapshot_bytes: tuple[bytes | None, ...]
    powerups_collected: np.ndarray | None = None
    reward_terms: np.ndarray | None = None

    @classmethod
    def from_payload(cls, payload: Payload) -> BatchResult:
        return cls(
            lane_ids=_array(payload, "lane_ids"),
            frames=_array(payload, "frames"),
            frames_advanced=_array(payload, "frames_advanced"),
            rewards=_array(payload, "rewards"),
            done=_array(payload, "done"),
            seeds=_array(payload, "seeds"),
            state_hashes=_array(payload, "state_hashes"),
            pixel_hashes=_array(payload, "pixel_hashes"),
            modes=_array(payload, "modes"),
            event_flags=_array(payload, "event_flags"),
            shattered=_array(payload, "shattered"),
            score=_array(payload, "score"),
            pixels=_optional_array(payload, "pixels"),
            board=_optional_array(payload, "board"),
            collision_image=_optional_array(payload, "collision_image"),
            ml_observation=_optional_array(payload, "ml_observation"),
            player_positions=_optional_array(payload, "player_positions"),
            snapshot_bytes=_bytes_tuple(payload, "snapshot_bytes"),
            powerups_collected=_optional_array(payload, "powerups_collected"),
            reward_terms=_optional_array(payload, "reward_terms"),
        )

    @property
    def lane_count(self) -> int:
        return int(self.lane_ids.shape[0])


@dataclass(frozen=True, slots=True)
class PixelResult:
    """Minimal native pixel result."""

    lane_ids: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    modes: np.ndarray
    pixels: np.ndarray
    player_positions: np.ndarray

    @classmethod
    def from_payload(cls, payload: Payload) -> PixelResult:
        return cls(
            lane_ids=_array(payload, "lane_ids"),
            frames=_array(payload, "frames"),
            frames_advanced=_array(payload, "frames_advanced"),
            rewards=_array(payload, "rewards"),
            done=_array(payload, "done"),
            seeds=_array(payload, "seeds"),
            modes=_array(payload, "modes"),
            pixels=_array(payload, "pixels"),
            player_positions=_array(payload, "player_positions"),
        )

    @property
    def lane_count(self) -> int:
        return int(self.lane_ids.shape[0])


@dataclass(frozen=True, slots=True)
class MlResult:
    """Render-free native ML result."""

    lane_ids: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    modes: np.ndarray
    ml_observation: np.ndarray
    player_positions: np.ndarray

    @classmethod
    def from_payload(cls, payload: Payload) -> MlResult:
        return cls(
            lane_ids=_array(payload, "lane_ids"),
            frames=_array(payload, "frames"),
            frames_advanced=_array(payload, "frames_advanced"),
            rewards=_array(payload, "rewards"),
            done=_array(payload, "done"),
            seeds=_array(payload, "seeds"),
            modes=_array(payload, "modes"),
            ml_observation=_array(payload, "ml_observation"),
            player_positions=_array(payload, "player_positions"),
        )

    @property
    def lane_count(self) -> int:
        return int(self.lane_ids.shape[0])


@dataclass(frozen=True, slots=True)
class HazardResult:
    """Native frozen-center hazard observation and TTC reference."""

    lane_ids: np.ndarray
    frames: np.ndarray
    survival_frames: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    modes: np.ndarray
    grid_size: int
    prediction_horizon_frames: int
    spawn_halo_radius: int
    hazard_observation: np.ndarray
    ttc_reference: np.ndarray
    player_positions: np.ndarray

    @classmethod
    def from_payload(cls, payload: Payload) -> HazardResult:
        return cls(
            lane_ids=_array(payload, "lane_ids"),
            frames=_array(payload, "frames"),
            survival_frames=_array(payload, "survival_frames"),
            done=_array(payload, "done"),
            seeds=_array(payload, "seeds"),
            modes=_array(payload, "modes"),
            grid_size=int(payload["grid_size"]),
            prediction_horizon_frames=int(payload["prediction_horizon_frames"]),
            spawn_halo_radius=int(payload["spawn_halo_radius"]),
            hazard_observation=_array(payload, "hazard_observation"),
            ttc_reference=_array(payload, "ttc_reference"),
            player_positions=_array(payload, "player_positions"),
        )

    @property
    def lane_count(self) -> int:
        return int(self.lane_ids.shape[0])


class NativeBatchEnvironment:
    """Small standalone owner for native Rust game/data lanes."""

    def __init__(
        self,
        *,
        step_frames: int = 4,
        execution: str = "serial",
        full_state: bool = False,
        pixels: bool = False,
        board: bool = True,
        difficulty: int = 2,
        patterns_enabled: bool = True,
        powerups_enabled: bool = True,
        include_offscreen_board: bool = False,
        preserve_offscreen_coordinates: bool = False,
        ml: bool = False,
        ml_grid_spacing: int = 32,
        collision_image: bool = False,
    ) -> None:
        self._native = dodge_native.NativeBatchEnv(
            step_frames,
            execution,
            full_state,
            pixels,
            board,
            difficulty,
            patterns_enabled,
            powerups_enabled,
            include_offscreen_board,
            preserve_offscreen_coordinates,
            ml,
            ml_grid_spacing,
            collision_image,
        )
        self.step_frames = int(step_frames)
        self.execution = execution
        self._closed = False

    @property
    def lane_count(self) -> int:
        self._ensure_open()
        return int(self._native.lane_count)

    @property
    def observation_flags(self) -> dict[str, Any]:
        self._ensure_open()
        return dict(self._native.observation_flags)

    def reset_batch(self, seeds: object, *, startup: bool = False) -> BatchResult:
        self._ensure_open()
        values = _integer_array(seeds, "seeds", maximum=32_767)
        method = (
            self._native.reset_batch_with_startup
            if startup
            else self._native.reset_batch
        )
        return BatchResult.from_payload(method(values))

    def reset_lanes(
        self, lanes: object, seeds: object, *, startup: bool = False
    ) -> BatchResult:
        self._ensure_open()
        lane_values = _integer_array(lanes, "lanes", maximum=2**31 - 1)
        seed_values = _integer_array(seeds, "seeds", maximum=32_767)
        if lane_values.shape != seed_values.shape:
            raise ValueError("lanes and seeds must have the same shape")
        method = (
            self._native.reset_lanes_with_startup
            if startup
            else self._native.reset_lanes
        )
        return BatchResult.from_payload(method(lane_values, seed_values))

    def step_batch(self, actions: object) -> BatchResult:
        self._ensure_open()
        return BatchResult.from_payload(self._native.step_batch(_action_array(actions)))

    def step_batch_active(self, actions: object, active: object) -> BatchResult:
        self._ensure_open()
        action_values = _action_array(actions)
        active_values = np.asarray(active)
        if (
            active_values.dtype != np.bool_
            or active_values.shape != action_values.shape
        ):
            raise ValueError("active must be a boolean array matching actions")
        return BatchResult.from_payload(
            self._native.step_batch_active(
                action_values, np.ascontiguousarray(active_values)
            )
        )

    def reset_ml(self, seeds: object, *, startup: bool = False) -> MlResult:
        self._ensure_open()
        values = _integer_array(seeds, "seeds", maximum=32_767)
        method = (
            self._native.reset_ml_batch_with_startup
            if startup
            else self._native.reset_ml_batch
        )
        return MlResult.from_payload(method(values))

    def step_ml(self, actions: object) -> MlResult:
        self._ensure_open()
        return MlResult.from_payload(self._native.step_ml_batch(_action_array(actions)))

    def step_pixels(self, actions: object) -> PixelResult:
        self._ensure_open()
        return PixelResult.from_payload(
            self._native.step_pixels(_action_array(actions))
        )

    def step_pixels_active(self, actions: object, active: object) -> PixelResult:
        self._ensure_open()
        action_values = _action_array(actions)
        active_values = np.asarray(active)
        if (
            active_values.dtype != np.bool_
            or active_values.shape != action_values.shape
        ):
            raise ValueError("active must be a boolean array matching actions")
        return PixelResult.from_payload(
            self._native.step_pixels_active(
                action_values, np.ascontiguousarray(active_values)
            )
        )

    def hazard_observations(
        self,
        grid_size: int,
        *,
        prediction_horizon_frames: int = 32,
        spawn_halo_radius: int = 1,
    ) -> HazardResult:
        self._ensure_open()
        payload = self._native.hazard_observations(
            grid_size,
            prediction_horizon_frames,
            spawn_halo_radius,
        )
        return HazardResult.from_payload(payload)

    def score_actions(
        self, snapshots: Sequence[bytes], lookahead_steps: int
    ) -> np.ndarray:
        self._ensure_open()
        if not snapshots or any(not isinstance(value, bytes) for value in snapshots):
            raise ValueError("snapshots must be a non-empty sequence of bytes")
        payload = self._native.score_actions(list(snapshots), lookahead_steps)
        return np.array(payload["scores"], copy=True)

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("native batch environment is closed")

    def __enter__(self) -> NativeBatchEnvironment:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
