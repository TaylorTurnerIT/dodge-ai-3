"""Low-memory replay storage for native RGB and grayscale observations.

Native display observations contain only the sixteen colors of the final
PICO-8 palette. This replay ring keeps those palette IDs packed two per byte
and reconstructs the configured RGB or grayscale stacks only when sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .model import validate_observation_shape
from .pixels import (
    GRAY_PROFILE,
    PICO8_LUMA_PALETTE,
    PICO8_PALETTE,
    RGB_PROFILE,
)
from .replay import ReplayBatch

RGB_FRAME_SHAPE: Final = (3, 128, 128)
GRAY_FRAME_SHAPE: Final = (1, 128, 128)
RGB_FRAME_PIXELS: Final = 128 * 128
GRAY_FRAME_PIXELS: Final = RGB_FRAME_PIXELS
PACKED_FRAME_BYTES: Final = RGB_FRAME_PIXELS // 2
# Preserve the existing RGB name while exposing the profile-independent
# storage width for callers that need to reason about grayscale replay too.
PACKED_RGB_FRAME_BYTES: Final = PACKED_FRAME_BYTES
PACKED_GRAY_FRAME_BYTES: Final = PACKED_FRAME_BYTES

_PALETTE = np.asarray(PICO8_PALETTE, dtype=np.uint8)
_LUMA_PALETTE = np.asarray(PICO8_LUMA_PALETTE, dtype=np.uint8)
_PALETTE_CODES = (
    # packed RGB24 values, sorted for vectorized inverse lookup
    (_PALETTE[:, 0].astype(np.uint32) << 16)
    | (_PALETTE[:, 1].astype(np.uint32) << 8)
    | _PALETTE[:, 2].astype(np.uint32)
)
_PALETTE_ORDER = np.argsort(_PALETTE_CODES)
_SORTED_PALETTE_CODES = _PALETTE_CODES[_PALETTE_ORDER]
# Stable ordering selects the first equivalent palette entry when luma values
# collide. Any equivalent ID decodes to the same grayscale value.
_LUMA_ORDER = np.argsort(_LUMA_PALETTE, kind="stable")
_SORTED_LUMA_VALUES = _LUMA_PALETTE[_LUMA_ORDER]


def _validate_pixel_profile(profile: object) -> str:
    if profile not in (RGB_PROFILE, GRAY_PROFILE):
        raise ValueError(
            "native pixel replay supports profiles "
            f"{RGB_PROFILE!r} and {GRAY_PROFILE!r}"
        )
    return str(profile)


def _profile_channels(profile: str) -> int:
    return 3 if profile == RGB_PROFILE else 1


def _profile_observation_shape(profile: str, stack_size: int) -> tuple[int, int, int]:
    return (_profile_channels(profile) * stack_size, 128, 128)


@dataclass(frozen=True, slots=True)
class PackedPixelReplayBatch:
    """Owned packed-palette minibatch with a logical profile shape."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    observation_profile: str = RGB_PROFILE

    def __post_init__(self) -> None:
        self._validate_and_store(copy=True)

    @classmethod
    def _from_owned(
        cls,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        observation_profile: str,
    ) -> PackedPixelReplayBatch:
        """Build from sampler-owned arrays without making a second copy."""

        value = object.__new__(cls)
        object.__setattr__(value, "observations", observations)
        object.__setattr__(value, "actions", actions)
        object.__setattr__(value, "rewards", rewards)
        object.__setattr__(value, "next_observations", next_observations)
        object.__setattr__(value, "dones", dones)
        object.__setattr__(value, "observation_profile", observation_profile)
        value._validate_and_store(copy=False)
        return value

    def _validate_and_store(self, *, copy: bool) -> None:
        observations = np.asarray(self.observations)
        next_observations = np.asarray(self.next_observations)
        actions = np.asarray(self.actions)
        rewards = np.asarray(self.rewards)
        dones = np.asarray(self.dones)
        profile = _validate_pixel_profile(self.observation_profile)
        if observations.ndim != 3:
            raise ValueError("packed observations must have shape (N, stack, 8192)")
        if observations.dtype != np.uint8 or next_observations.dtype != np.uint8:
            raise ValueError("packed observations must have dtype uint8")
        if observations.shape[2] != PACKED_FRAME_BYTES:
            raise ValueError("packed observations must have shape (N, stack, 8192)")
        if observations.shape[1] < 1:
            raise ValueError("packed observations must contain at least one frame")
        if next_observations.shape != observations.shape:
            raise ValueError("packed next_observations must match observations shape")
        batch_size = observations.shape[0]
        if actions.shape != (batch_size,):
            raise ValueError("actions must have shape (N,)")
        if rewards.shape != (batch_size,):
            raise ValueError("rewards must have shape (N,)")
        if dones.shape != (batch_size,):
            raise ValueError("dones must have shape (N,)")
        if not copy and not all(
            value.flags.owndata
            for value in (observations, next_observations, actions, rewards, dones)
        ):
            raise ValueError("owned replay batch arrays must own their storage")
        object.__setattr__(
            self,
            "observations",
            np.array(observations, dtype=np.uint8, copy=True) if copy else observations,
        )
        object.__setattr__(
            self,
            "next_observations",
            np.array(next_observations, dtype=np.uint8, copy=True)
            if copy
            else next_observations,
        )
        object.__setattr__(
            self, "actions", np.array(actions, copy=True) if copy else actions
        )
        object.__setattr__(
            self, "rewards", np.array(rewards, copy=True) if copy else rewards
        )
        object.__setattr__(self, "dones", np.array(dones, copy=True) if copy else dones)
        object.__setattr__(self, "observation_profile", profile)

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def stack_size(self) -> int:
        return int(self.observations.shape[1])

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return _profile_observation_shape(self.observation_profile, self.stack_size)

    @property
    def logical_shape(self) -> tuple[int, int, int]:
        """Alias for the unpacked model-facing observation shape."""

        return self.observation_shape

    @property
    def packed_observations(self) -> np.ndarray:
        return self.observations

    @property
    def packed_next_observations(self) -> np.ndarray:
        return self.next_observations


def estimated_storage_bytes(
    capacity: int,
    stack_size: int = 4,
    observation_profile: str = RGB_PROFILE,
) -> int:
    """Estimate native-palette replay allocation without allocating it."""

    _validate_capacity(capacity)
    _validate_stack_size(stack_size)
    _validate_pixel_profile(observation_profile)
    frame_capacity = 2 * int(capacity) + int(stack_size) + 1
    frame_bytes = PACKED_FRAME_BYTES + 2 * np.dtype(np.int64).itemsize
    transition_bytes = (
        2 * np.dtype(np.int64).itemsize
        + np.dtype(np.int64).itemsize
        + np.dtype(np.float32).itemsize
        + np.dtype(np.bool_).itemsize
    )
    return int(frame_capacity * frame_bytes + int(capacity) * transition_bytes)


class NativePixelReplayBuffer:
    """Fixed-capacity packed replay ring for native RGB or grayscale frames.

    ``reset`` starts an episode from a repeated stack. Each subsequent ``add``
    must provide the exact previous observation and its one-frame temporal
    successor. RGB frames use the display palette; grayscale frames use the
    fixed integer-luma palette derived from those same sixteen colors.
    """

    def __init__(
        self,
        capacity: int,
        *,
        stack_size: int = 4,
        seed: int | None = None,
        num_actions: int | None = None,
        observation_profile: str = RGB_PROFILE,
    ) -> None:
        _validate_capacity(capacity)
        _validate_stack_size(stack_size)
        profile = _validate_pixel_profile(observation_profile)
        if num_actions is not None and (
            isinstance(num_actions, bool) or num_actions < 1
        ):
            raise ValueError("num_actions must be positive when provided")

        self.capacity = int(capacity)
        self.stack_size = int(stack_size)
        self.observation_profile = profile
        self.channels = _profile_channels(profile)
        self.frame_shape = (self.channels, 128, 128)
        self.observation_shape = _profile_observation_shape(profile, self.stack_size)
        validate_observation_shape(self.observation_shape)
        self.num_actions = None if num_actions is None else int(num_actions)
        self.frame_capacity = 2 * self.capacity + self.stack_size + 1
        self._rng = np.random.default_rng(seed)

        self._packed_frames = np.empty(
            (self.frame_capacity, PACKED_FRAME_BYTES), dtype=np.uint8
        )
        self._frame_ids = np.empty(self.frame_capacity, dtype=np.int64)
        self._frame_parents = np.empty(self.frame_capacity, dtype=np.int64)
        self._observation_frame_ids = np.empty(self.capacity, dtype=np.int64)
        self._next_observation_frame_ids = np.empty(
            self.capacity, dtype=np.int64
        )
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=np.float32)
        self._dones = np.empty(self.capacity, dtype=np.bool_)
        self._size = 0
        self._next_index = 0
        self._next_frame_id = 1
        self._last_frame_id: int | None = None
        self._last_observation: np.ndarray | None = None
        self._pending_reset_observation: np.ndarray | None = None
        self._pending_reset_packed: np.ndarray | None = None
        self._awaiting_reset = True

    def __len__(self) -> int:
        return self._size

    @property
    def allocated_bytes(self) -> int:
        """Return the bytes allocated by this packed replay ring."""

        return estimated_storage_bytes(
            self.capacity, self.stack_size, self.observation_profile
        )

    def reset(self, observation: object) -> np.ndarray:
        """Stage a repeated episode-start stack without storing frames."""

        value = _validate_native_stack(
            observation, self.stack_size, self.observation_profile
        )
        if not _is_repeated_stack(value, self.stack_size, self.channels):
            raise ValueError("reset observation must repeat one native frame")
        self._pending_reset_observation = value.copy()
        self._pending_reset_packed = _frame_to_packed(
            value[-self.channels :], self.observation_profile
        )
        self._awaiting_reset = True
        return value.copy()

    def reset_palette_ids(self, frame: object) -> None:
        """Stage one native palette frame without expanding it to RGB or luma."""

        palette_ids = _validate_palette_ids(frame)
        self._pending_reset_observation = None
        self._pending_reset_packed = _pack_palette_ids(palette_ids)
        self._awaiting_reset = True

    def add_palette_ids(
        self,
        next_frame: object,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """Store a sequential transition directly from native palette IDs."""

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action_value = int(action)
        if self.num_actions is not None and not 0 <= action_value < self.num_actions:
            raise ValueError(f"action must be between 0 and {self.num_actions - 1}")
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("reward must be finite")
        next_packed = _pack_palette_ids(_validate_palette_ids(next_frame))

        if self._pending_reset_packed is not None:
            observation_frame_id = self._append_frame(
                self._pending_reset_packed, parent_id=None
            )
            self._pending_reset_packed = None
        else:
            if self._last_frame_id is None or self._awaiting_reset:
                raise RuntimeError("reset_palette_ids must be called before add")
            observation_frame_id = self._last_frame_id
            self._checked_frame_slot(observation_frame_id)
        next_frame_id = self._append_frame(next_packed, parent_id=observation_frame_id)

        transition_index = self._next_index
        self._observation_frame_ids[transition_index] = observation_frame_id
        self._next_observation_frame_ids[transition_index] = next_frame_id
        self._actions[transition_index] = action_value
        self._rewards[transition_index] = reward_value
        self._dones[transition_index] = bool(done)
        self._next_index = (transition_index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)
        self._last_frame_id = int(next_frame_id)
        self._last_observation = None
        self._awaiting_reset = bool(done)

    def add(
        self,
        observation: object,
        action: int,
        reward: float,
        next_observation: object,
        done: bool,
    ) -> None:
        """Store one sequential native transition without duplicating stacks."""

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action_value = int(action)
        if self.num_actions is not None and not 0 <= action_value < self.num_actions:
            raise ValueError(f"action must be between 0 and {self.num_actions - 1}")
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("reward must be finite")
        value = _validate_native_stack(
            observation, self.stack_size, self.observation_profile
        )
        next_value = _validate_native_stack(
            next_observation, self.stack_size, self.observation_profile
        )
        if self._pending_reset_observation is None and (
            self._last_frame_id is None or self._awaiting_reset
        ):
            if self._last_frame_id is not None and not _is_repeated_stack(
                value, self.stack_size, self.channels
            ):
                raise RuntimeError(
                    "a terminal transition requires a repeated reset stack"
                )
            self.reset(value)
        if self._pending_reset_observation is not None:
            if not np.array_equal(value, self._pending_reset_observation):
                raise ValueError("observation does not match the pending reset")
            initial_packed = self._pending_reset_packed
            if initial_packed is None:
                raise RuntimeError("pending reset frame is unavailable")
            if not np.array_equal(
                next_value[: -self.channels], value[self.channels :]
            ):
                raise ValueError(
                    "next_observation is not a one-frame stack successor"
                )
            # Decode all pixels before mutating the frame ring so malformed
            # transitions cannot consume frame slots.
            next_packed = _frame_to_packed(
                next_value[-self.channels :], self.observation_profile
            )
            observation_frame_id = self._append_frame(initial_packed, parent_id=None)
            next_frame_id = self._append_frame(
                next_packed, parent_id=observation_frame_id
            )
            self._pending_reset_observation = None
            self._pending_reset_packed = None
        else:
            if self._last_frame_id is None or self._last_observation is None:
                raise RuntimeError("reset must be called before add")
            if not np.array_equal(value, self._last_observation):
                raise ValueError("observation does not continue the replay sequence")
            if not np.array_equal(
                next_value[: -self.channels], value[self.channels :]
            ):
                raise ValueError(
                    "next_observation is not a one-frame stack successor"
                )

            # The inverse palette lookup is limited to the newest frame.
            # Earlier frames are already validated by exact stack equality.
            next_packed = _frame_to_packed(
                next_value[-self.channels :], self.observation_profile
            )
            observation_frame_id = self._last_frame_id
            self._checked_frame_slot(observation_frame_id)
            next_frame_id = self._append_frame(
                next_packed, parent_id=observation_frame_id
            )

        transition_index = self._next_index
        self._observation_frame_ids[transition_index] = observation_frame_id
        self._next_observation_frame_ids[transition_index] = next_frame_id
        self._actions[transition_index] = action_value
        self._rewards[transition_index] = reward_value
        self._dones[transition_index] = bool(done)
        self._next_index = (transition_index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

        self._last_frame_id = int(next_frame_id)
        self._last_observation = next_value.copy()
        self._awaiting_reset = bool(done)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample owned uint8 stacks reconstructed for this profile."""

        indices = self._sample_indices(batch_size)
        observation_ids = self._observation_frame_ids[indices]
        next_observation_ids = self._next_observation_frame_ids[indices]
        return ReplayBatch(
            observations=self._reconstruct_stacks(observation_ids),
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._reconstruct_stacks(next_observation_ids),
            dones=self._dones[indices],
        )

    def sample_packed(self, batch_size: int) -> PackedPixelReplayBatch:
        """Sample packed stacks without CPU palette expansion."""

        indices = self._sample_indices(batch_size)
        observation_ids = self._observation_frame_ids[indices]
        next_observation_ids = self._next_observation_frame_ids[indices]
        observations = self._reconstruct_packed_stacks(observation_ids)
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_observations = self._reconstruct_packed_stacks(next_observation_ids)
        dones = self._dones[indices]
        return PackedPixelReplayBatch._from_owned(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            observation_profile=self.observation_profile,
        )

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError("cannot sample more transitions than are stored")
        return self._rng.choice(self._size, size=int(batch_size), replace=False)

    def _append_frame(self, packed: np.ndarray, parent_id: int | None) -> int:
        frame_id = self._next_frame_id
        slot = (frame_id - 1) % self.frame_capacity
        self._packed_frames[slot] = packed
        self._frame_ids[slot] = frame_id
        self._frame_parents[slot] = frame_id if parent_id is None else parent_id
        self._next_frame_id += 1
        return frame_id

    def _checked_frame_slot(self, frame_id: int) -> int:
        slot = (int(frame_id) - 1) % self.frame_capacity
        if self._frame_ids[slot] != frame_id:
            raise RuntimeError(f"replay frame {frame_id} was evicted")
        return slot

    def _reconstruct_stacks(self, last_frame_ids: np.ndarray) -> np.ndarray:
        frame_ids = self._stack_frame_ids(last_frame_ids)
        batch_size = frame_ids.shape[0]
        slots = (frame_ids - 1) % self.frame_capacity
        packed = self._packed_frames[slots]
        palette_ids = _unpack_palette_ids(packed)
        if self.observation_profile == RGB_PROFILE:
            rgb = _PALETTE[palette_ids]
            rgb = rgb.reshape(batch_size, self.stack_size, 128, 128, 3)
            rgb = rgb.transpose(0, 1, 4, 2, 3)
            return np.ascontiguousarray(rgb).reshape(
                batch_size, 3 * self.stack_size, 128, 128
            )
        gray = _LUMA_PALETTE[palette_ids]
        return np.ascontiguousarray(
            gray.reshape(batch_size, self.stack_size, 128, 128)
        )

    def _reconstruct_packed_stacks(self, last_frame_ids: np.ndarray) -> np.ndarray:
        """Gather packed stack frames; intentionally performs no palette decode."""

        frame_ids = self._stack_frame_ids(last_frame_ids)
        slots = (frame_ids - 1) % self.frame_capacity
        # Advanced indexing already returns storage independent of the ring.
        return self._packed_frames[slots]

    def _stack_frame_ids(self, last_frame_ids: np.ndarray) -> np.ndarray:
        last_frame_ids = np.asarray(last_frame_ids, dtype=np.int64)
        if last_frame_ids.ndim != 1:
            raise ValueError("frame IDs must have shape (N,)")
        batch_size = last_frame_ids.shape[0]
        frame_ids = np.empty((batch_size, self.stack_size), dtype=np.int64)
        frame_ids[:, -1] = last_frame_ids
        for position in range(self.stack_size - 1, -1, -1):
            slots = (frame_ids[:, position] - 1) % self.frame_capacity
            stored_ids = self._frame_ids[slots]
            if not np.array_equal(stored_ids, frame_ids[:, position]):
                raise RuntimeError(
                    "sampled replay transition references an evicted frame"
                )
            if position:
                frame_ids[:, position - 1] = self._frame_parents[slots]

        slots = (frame_ids - 1) % self.frame_capacity
        if not np.array_equal(self._frame_ids[slots], frame_ids):
            raise RuntimeError("sampled replay transition references an evicted frame")
        return frame_ids


def _validate_capacity(capacity: int) -> None:
    if isinstance(capacity, bool) or capacity < 1:
        raise ValueError("capacity must be positive")


def _validate_stack_size(stack_size: int) -> None:
    if isinstance(stack_size, bool) or not isinstance(stack_size, int):
        raise TypeError("stack_size must be an integer")
    if stack_size < 1:
        raise ValueError("stack_size must be positive")


def _validate_rgb_stack(observation: object, stack_size: int) -> np.ndarray:
    value = np.asarray(observation)
    expected_shape = (3 * stack_size, 128, 128)
    if value.shape != expected_shape:
        raise ValueError(
            f"RGB observation must have shape {expected_shape}, got {value.shape}"
        )
    if value.dtype == np.uint8:
        return value
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError("RGB observation must have dtype uint8 or floating point")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("floating RGB observation must contain values in [0, 1]")
    return value


def _validate_gray_stack(observation: object, stack_size: int) -> np.ndarray:
    value = np.asarray(observation)
    expected_shape = (stack_size, 128, 128)
    if value.shape != expected_shape:
        raise ValueError(
            f"grayscale observation must have shape {expected_shape}, got {value.shape}"
        )
    if value.dtype == np.uint8:
        return value
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(
            "grayscale observation must have dtype uint8 or floating point"
        )
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError(
            "floating grayscale observation must contain values in [0, 1]"
        )
    return value


def _validate_native_stack(
    observation: object, stack_size: int, profile: str
) -> np.ndarray:
    if profile == RGB_PROFILE:
        return _validate_rgb_stack(observation, stack_size)
    return _validate_gray_stack(observation, stack_size)


def _is_repeated_stack(
    value: np.ndarray, stack_size: int, channels: int
) -> bool:
    first = value[:channels]
    return all(
        np.array_equal(value[offset : offset + channels], first)
        for offset in range(channels, channels * stack_size, channels)
    )


def _unpack_palette_ids(packed: np.ndarray) -> np.ndarray:
    palette_ids = np.empty(
        (*packed.shape[:-1], RGB_FRAME_PIXELS), dtype=np.uint8
    )
    palette_ids[..., 0::2] = packed >> 4
    palette_ids[..., 1::2] = packed & 0x0F
    return palette_ids


def _frame_to_packed(frame: np.ndarray, profile: str) -> np.ndarray:
    if profile == RGB_PROFILE:
        return _rgb_frame_to_packed(frame)
    return _gray_frame_to_packed(frame)


def _rgb_frame_to_packed(frame: np.ndarray) -> np.ndarray:
    is_uint8 = frame.dtype == np.uint8
    scaled = (
        frame.astype(np.uint16, copy=False)
        if is_uint8
        else np.rint(frame * np.float32(255.0)).astype(np.uint16)
    )
    codes = (
        (scaled[0].astype(np.uint32) << 16)
        | (scaled[1].astype(np.uint32) << 8)
        | scaled[2].astype(np.uint32)
    )
    positions = np.searchsorted(_SORTED_PALETTE_CODES, codes)
    safe_positions = np.minimum(positions, len(_SORTED_PALETTE_CODES) - 1)
    valid = (positions < len(_SORTED_PALETTE_CODES)) & (
        _SORTED_PALETTE_CODES[safe_positions] == codes
    )
    if not bool(np.all(valid)):
        raise ValueError("RGB frame contains colors outside the native palette")
    palette_ids = _PALETTE_ORDER[safe_positions].astype(np.uint8, copy=False)
    reconstructed = _PALETTE[palette_ids].transpose(2, 0, 1)
    if not is_uint8:
        reconstructed = reconstructed.astype(np.float32, copy=False) / np.float32(
            255.0
        )
    if not np.array_equal(reconstructed, frame):
        raise ValueError("RGB frame is not an exact native-palette image")
    return _pack_palette_ids(palette_ids)


def _gray_frame_to_packed(frame: np.ndarray) -> np.ndarray:
    is_uint8 = frame.dtype == np.uint8
    values = frame[0]
    scaled = (
        values.astype(np.uint16, copy=False)
        if is_uint8
        else np.rint(values * np.float32(255.0)).astype(np.uint16)
    )
    positions = np.searchsorted(_SORTED_LUMA_VALUES, scaled)
    safe_positions = np.minimum(positions, len(_SORTED_LUMA_VALUES) - 1)
    valid = (positions < len(_SORTED_LUMA_VALUES)) & (
        _SORTED_LUMA_VALUES[safe_positions] == scaled
    )
    if not bool(np.all(valid)):
        raise ValueError("grayscale frame contains values outside native luma palette")
    palette_ids = _LUMA_ORDER[safe_positions].astype(np.uint8, copy=False)
    reconstructed = _LUMA_PALETTE[palette_ids][None, ...]
    if not is_uint8:
        reconstructed = reconstructed.astype(np.float32, copy=False) / np.float32(
            255.0
        )
    if not np.array_equal(reconstructed, frame):
        raise ValueError("grayscale frame is not an exact native-luma image")
    return _pack_palette_ids(palette_ids)


def _pack_palette_ids(palette_ids: np.ndarray) -> np.ndarray:
    flat_ids = palette_ids.reshape(-1)
    return np.bitwise_or(flat_ids[0::2] << 4, flat_ids[1::2]).astype(
        np.uint8, copy=True
    )


def _validate_palette_ids(frame: object) -> np.ndarray:
    palette_ids = np.asarray(frame)
    if palette_ids.shape != (128, 128) or palette_ids.dtype != np.uint8:
        raise ValueError("native palette frame must have shape (128, 128) and uint8")
    if np.any(palette_ids > 15):
        raise ValueError("native palette frame values must be between 0 and 15")
    return palette_ids


__all__ = [
    "GRAY_FRAME_SHAPE",
    "NativePixelReplayBuffer",
    "PACKED_FRAME_BYTES",
    "PACKED_GRAY_FRAME_BYTES",
    "PACKED_RGB_FRAME_BYTES",
    "PackedPixelReplayBatch",
    "RGB_FRAME_SHAPE",
    "estimated_storage_bytes",
]
