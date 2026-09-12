"""Batched native image lanes for throughput-oriented DDQN collection."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import ACTION_COUNT, DEFAULT_STACK_SIZE, DEFAULT_STEP_FRAMES
from .image import NativeBatchResultError
from .pixels import (
    COLLISION_PROFILE,
    GRAY_PROFILE,
    OBSERVATION_PROFILES,
    PICO8_LUMA_PALETTE,
    PICO8_PALETTE,
    RGB_PROFILE,
    observation_shape,
)

_PALETTE = np.asarray(PICO8_PALETTE, dtype=np.uint8)
_LUMA_PALETTE = np.asarray(PICO8_LUMA_PALETTE, dtype=np.uint8)


class NativeVectorBoundary(Protocol):
    def reset_batch(self, seeds: object, *, startup: bool = False) -> object: ...

    def reset_lanes(
        self, lanes: object, seeds: object, *, startup: bool = False
    ) -> object: ...

    def step_batch(self, actions: object) -> object: ...

    def step_batch_active(self, actions: object, active: object) -> object: ...

    def close(self) -> None: ...


class CNNImageDDQNVectorEnv:
    """Own several native lanes and one temporal frame ring per lane."""

    def __init__(
        self,
        lanes: int,
        *,
        stack_size: int = DEFAULT_STACK_SIZE,
        step_frames: int = DEFAULT_STEP_FRAMES,
        difficulty: int = 2,
        patterns: bool = True,
        powerups: bool = True,
        observation_profile: str = COLLISION_PROFILE,
        execution: str = "serial",
        native_environment: NativeVectorBoundary | None = None,
    ) -> None:
        if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1:
            raise ValueError("lanes must be a positive integer")
        if observation_profile not in OBSERVATION_PROFILES:
            raise ValueError(f"unknown observation profile: {observation_profile!r}")
        if execution not in ("serial", "parallel"):
            raise ValueError("execution must be 'serial' or 'parallel'")
        self.lanes = lanes
        self.stack_size = stack_size
        self.step_frames = step_frames
        self.observation_profile = observation_profile
        self.observation_shape = observation_shape(observation_profile, stack_size)
        channels = self.observation_shape[0] // stack_size
        dtype = np.float32 if observation_profile == COLLISION_PROFILE else np.uint8
        self._frames = np.empty(
            (lanes, stack_size, channels, *self.observation_shape[1:]), dtype=dtype
        )
        self._cursor = 0
        self._initialized = np.zeros(lanes, dtype=np.bool_)
        self._closed = False
        self._native = native_environment or self._make_native(
            step_frames=step_frames,
            execution=execution,
            difficulty=difficulty,
            patterns=patterns,
            powerups=powerups,
            observation_profile=observation_profile,
        )

    @staticmethod
    def _make_native(**kwargs: object) -> NativeVectorBoundary:
        from ...batch import NativeBatchEnvironment

        profile = str(kwargs.pop("observation_profile"))
        patterns = bool(kwargs.pop("patterns"))
        powerups = bool(kwargs.pop("powerups"))
        return NativeBatchEnvironment(
            **kwargs,
            full_state=False,
            pixels=profile in (RGB_PROFILE, GRAY_PROFILE),
            board=False,
            patterns_enabled=patterns,
            powerups_enabled=powerups,
            collision_image=profile == COLLISION_PROFILE,
        )

    def reset(self, seeds: object) -> tuple[np.ndarray, object]:
        values = self._seeds(seeds, expected=self.lanes)
        result = self._native.reset_batch(values)
        images = self._images(result)
        if images.shape[0] != self.lanes:
            raise NativeBatchResultError("native reset lane count changed")
        self._frames[...] = images[:, None, ...]
        self._cursor = 0
        self._initialized[:] = True
        return self.observations(), result

    def reset_lanes(
        self, lanes: object, seeds: object
    ) -> tuple[np.ndarray, object]:
        lane_values = np.asarray(lanes)
        if lane_values.ndim != 1 or not np.issubdtype(lane_values.dtype, np.integer):
            raise ValueError("lanes must be a one-dimensional integer array")
        lane_values = np.ascontiguousarray(lane_values, dtype=np.uint32)
        if np.any(lane_values >= self.lanes):
            raise ValueError("lane index out of bounds")
        seed_values = self._seeds(seeds, expected=lane_values.size)
        result = self._native.reset_lanes(lane_values, seed_values)
        result_lanes = np.asarray(result.lane_ids, dtype=np.int64)  # type: ignore[attr-defined]
        if not np.array_equal(result_lanes, lane_values.astype(np.int64)):
            raise NativeBatchResultError("native reset lanes changed order")
        images = self._images(result)
        self._frames[result_lanes] = images[:, None, ...]
        self._initialized[result_lanes] = True
        return self.observations(result_lanes), result

    def step(
        self, actions: object, active: object | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
        if not self._initialized.all():
            raise RuntimeError("reset must initialize every lane before step")
        action_values = np.asarray(actions)
        if (
            action_values.shape != (self.lanes,)
            or not np.issubdtype(action_values.dtype, np.integer)
            or np.any(action_values < 0)
            or np.any(action_values >= ACTION_COUNT)
        ):
            raise ValueError(f"actions must be integers with shape ({self.lanes},)")
        action_values = np.ascontiguousarray(action_values, dtype=np.uint8)
        if active is None:
            result = self._native.step_batch(action_values)
        else:
            active_values = np.asarray(active)
            if active_values.shape != (self.lanes,) or active_values.dtype != np.bool_:
                raise ValueError(f"active must be bool with shape ({self.lanes},)")
            result = self._native.step_batch_active(
                action_values, np.ascontiguousarray(active_values)
            )
        lane_ids = np.asarray(result.lane_ids, dtype=np.int64)  # type: ignore[attr-defined]
        images = self._images(result)
        self._frames[lane_ids, self._cursor] = images
        self._cursor = (self._cursor + 1) % self.stack_size
        return (
            self.observations(lane_ids),
            np.asarray(result.rewards, dtype=np.float32),  # type: ignore[attr-defined]
            np.asarray(result.done, dtype=np.bool_),  # type: ignore[attr-defined]
            result,
        )

    def observations(self, lanes: object | None = None) -> np.ndarray:
        lane_values = (
            np.arange(self.lanes, dtype=np.int64)
            if lanes is None
            else np.asarray(lanes, dtype=np.int64)
        )
        order = (np.arange(self.stack_size) + self._cursor) % self.stack_size
        frames = self._frames[lane_values][:, order]
        return np.ascontiguousarray(frames).reshape(
            lane_values.size, *self.observation_shape
        )

    def palette_indices(self, result: object) -> np.ndarray | None:
        if self.observation_profile == COLLISION_PROFILE:
            return None
        return np.array(result.pixels, dtype=np.uint8, copy=True)  # type: ignore[attr-defined]

    def close(self) -> None:
        if not self._closed:
            self._native.close()
            self._closed = True

    def _images(self, result: object) -> np.ndarray:
        if self.observation_profile == COLLISION_PROFILE:
            value = np.asarray(result.collision_image)  # type: ignore[attr-defined]
            if value.ndim != 3 or value.shape[1:] != (84, 84):
                raise NativeBatchResultError(
                    "native collision images must have shape (N,84,84)"
                )
            images = value.astype(np.float32, copy=True)
            if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
                raise NativeBatchResultError("native collision images are invalid")
            if images.max(initial=0.0) > 1.0:
                images /= np.float32(255.0)
            return images[:, None, ...]
        pixels = np.asarray(result.pixels)  # type: ignore[attr-defined]
        if pixels.ndim != 3 or pixels.shape[1:] != (128, 128):
            raise NativeBatchResultError("native pixels must have shape (N,128,128)")
        if pixels.dtype != np.uint8 or np.any(pixels > 15):
            raise NativeBatchResultError("native pixels must be PICO-8 palette IDs")
        if self.observation_profile == RGB_PROFILE:
            return np.ascontiguousarray(_PALETTE[pixels].transpose(0, 3, 1, 2))
        return np.ascontiguousarray(_LUMA_PALETTE[pixels][:, None, ...])

    @staticmethod
    def _seeds(seeds: object, *, expected: int) -> np.ndarray:
        values = np.asarray(seeds)
        if (
            values.shape != (expected,)
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
            or np.any(values > 32_767)
        ):
            raise ValueError(f"seeds must be integers with shape ({expected},)")
        return np.ascontiguousarray(values, dtype=np.uint32)


__all__ = ["CNNImageDDQNVectorEnv", "NativeVectorBoundary"]
