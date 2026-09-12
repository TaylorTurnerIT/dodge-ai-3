"""Generate bounded native collision-image replays from DDQN checkpoints."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import torch

from ...batch import NativeBatchEnvironment
from .agent import DoubleDQNAgent
from .env import CNNImageDDQNEnv
from .pixels import PICO8_PALETTE
from .run import _choose_device, _configure_torch_backend, _load_checkpoint

FRAME_HEIGHT: Final = 84
FRAME_WIDTH: Final = 84
GAME_HEIGHT: Final = 128
GAME_WIDTH: Final = 128
MAX_REPLAY_STEPS: Final = 600
DEFAULT_REPLAY_STEPS: Final = 360
ACTION_NAMES: Final = (
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


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """One browser-ready frame produced by the native environment."""

    index: int
    native_frame: int
    action: int | None
    reward: float
    done: bool
    png: bytes
    collision_png: bytes

    @property
    def action_name(self) -> str | None:
        if self.action is None:
            return None
        return ACTION_NAMES[self.action]


@dataclass(frozen=True, slots=True)
class NativeReplay:
    """Immutable, bounded replay generated from one saved policy."""

    checkpoint: Path
    seed: int
    step_frames: int
    created_at: str
    frames: tuple[ReplayFrame, ...]
    run_id: str | None = None
    label: str | None = None
    source: str | None = None
    eval_reward: float | None = None

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def total_reward(self) -> float:
        return float(sum(frame.reward for frame in self.frames))

    @property
    def survival_frames(self) -> int:
        if not self.frames:
            return 0
        return int(self.frames[-1].native_frame)

    @property
    def terminated(self) -> bool:
        return bool(self.frames and self.frames[-1].done)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def grayscale_png(image: object) -> bytes:
    """Encode one 84x84 uint8 grayscale image without a Pillow dependency."""

    value = np.asarray(image)
    if value.shape == (1, FRAME_HEIGHT, FRAME_WIDTH):
        value = value[0]
    if value.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"image must have shape (84, 84), got {value.shape}")
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError("image must contain integer grayscale values")
    value = np.asarray(value, dtype=np.uint8)
    rows = b"".join(b"\x00" + row.tobytes() for row in value)
    header = struct.pack(">IIBBBBB", FRAME_WIDTH, FRAME_HEIGHT, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _observation_frame(observation: object) -> np.ndarray:
    value = np.asarray(observation)
    if value.ndim != 3 or value.shape[1:] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            "native replay observation must have shape "
            f"(channels, 84, 84), got {value.shape}"
        )
    latest = np.asarray(value[-1], dtype=np.float32)
    if not np.isfinite(latest).all() or np.any(latest < 0.0) or np.any(latest > 1.0):
        raise ValueError("native replay image values must be finite and in [0, 1]")
    return np.rint(latest * np.float32(255.0)).astype(np.uint8)


def rgb_png(indexed: object) -> bytes:
    """Encode one 128x128 indexed native framebuffer as truecolor PNG."""

    value = np.asarray(indexed)
    if value.shape == (1, GAME_HEIGHT, GAME_WIDTH):
        value = value[0]
    if value.shape != (GAME_HEIGHT, GAME_WIDTH):
        raise ValueError(f"image must have shape (128, 128), got {value.shape}")
    if not np.issubdtype(value.dtype, np.integer) or int(value.max()) > 15:
        raise ValueError("image must contain PICO-8 palette indices")
    palette = np.asarray(PICO8_PALETTE, dtype=np.uint8)
    rgb = palette[value.astype(np.uint8)]
    rows = b"".join(b"\x00" + row.tobytes() for row in rgb)
    header = struct.pack(">IIBBBBB", GAME_WIDTH, GAME_HEIGHT, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _validate_replay_request(seed: int, steps: int) -> tuple[int, int]:
    if isinstance(seed, bool) or not 0 <= int(seed) <= 32_767:
        raise ValueError("replay seed must be between 0 and 32767")
    if isinstance(steps, bool) or not 1 <= int(steps) <= MAX_REPLAY_STEPS:
        raise ValueError(f"replay steps must be between 1 and {MAX_REPLAY_STEPS}")
    return int(seed), int(steps)


def _infer_dueling(payload: dict) -> bool:
    """Detect the checkpoint's Q-head from its state-dict keys.

    Dueling checkpoints store `value_stream`/`advantage_stream`; plain-head
    checkpoints store `q_head`. The replay must build the matching network
    before loading, otherwise plain-head runs fail to replay at all.
    """

    online = payload.get("online_network", {})
    keys = set(online.keys() if hasattr(online, "keys") else [])
    return not any(str(key).startswith("q_head") for key in keys)


def generate_native_replay(
    checkpoint: Path | str,
    *,
    seed: int = 42,
    steps: int = DEFAULT_REPLAY_STEPS,
    device: str = "auto",
    step_frames: int = 4,
    difficulty: int = 2,
    patterns: bool = True,
    powerups: bool = True,
    forced_actions: dict[int, int] | None = None,
    epsilon: float = 0.0,
    dueling: bool | None = None,
) -> NativeReplay:
    """Run a saved DDQN through a new native Rust lane.

    The current trainer does not share its mutable lane with the dashboard.
    This function reconstructs an episode from the checkpoint, seed, and
    native game configuration. Every displayed image is emitted by Rust and
    is encoded only for browser delivery.

    ``forced_actions`` maps 1-based decision indices to actions. When
    provided, those steps use the forced action instead of the policy one.
    This powers the lock-in probe: force the rarely used move once, then
    run greedy and compare total reward against the pure greedy replay.

    ``epsilon`` adds exploration noise (seeded by the replay seed). Zero
    replays the pure greedy policy; small positive values show
    deployed-with-exploration behavior, which can survive much longer
    when the argmax has collapsed onto one action.

    ``dueling`` selects the Q-head architecture. ``None`` (default)
    detects it from the checkpoint's state-dict keys so both dueling and
    plain-head runs replay without the caller knowing which is which.
    """

    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("replay epsilon must be between 0 and 1")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError(f"checkpoint is not a regular file: {checkpoint_path}")
    seed, steps = _validate_replay_request(seed, steps)
    if isinstance(step_frames, bool) or not 1 <= int(step_frames) <= 8:
        raise ValueError("step_frames must be between 1 and 8")
    if difficulty not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2, or 3")
    if not isinstance(patterns, bool) or not isinstance(powerups, bool):
        raise TypeError("patterns and powerups must be booleans")
    if dueling is not None and not isinstance(dueling, bool):
        raise TypeError("dueling must be a boolean or None")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    shape = tuple(int(value) for value in payload.get("observation_shape", ()))
    if len(shape) != 3:
        raise ValueError("checkpoint observation_shape is invalid")
    chosen_device = _choose_device(device)
    _configure_torch_backend(chosen_device)
    dueling_effective = _infer_dueling(payload) if dueling is None else bool(dueling)

    def _network_factory(actions: int) -> object:
        from .model import AtariCnnQNetwork

        return AtariCnnQNetwork(
            actions, dueling=dueling_effective, input_channels=shape[0]
        )

    agent = DoubleDQNAgent(
        num_actions=int(payload["num_actions"]),
        device=chosen_device,
        observation_shape=shape,
        network_factory=_network_factory,  # type: ignore[arg-type]
    )
    _load_checkpoint(agent, checkpoint_path)
    env = CNNImageDDQNEnv(
        stack_size=shape[0],
        step_frames=int(step_frames),
        difficulty=int(difficulty),
        patterns=patterns,
        powerups=powerups,
    )
    # A pixel lane runs the identical game in lockstep so the browser shows
    # the default game view; the decision lane still drives the policy.
    pixels = NativeBatchEnvironment(
        step_frames=int(step_frames),
        pixels=True,
        board=False,
        difficulty=int(difficulty),
        patterns_enabled=patterns,
        powerups_enabled=powerups,
    )
    frames: list[ReplayFrame] = []
    rng = np.random.default_rng(seed)
    try:
        observation, info = env.reset(seed=seed)
        pixel_result = pixels.reset_batch(np.asarray([seed], dtype=np.uint32))
        if pixel_result.pixels is None:
            raise ValueError("native pixel lane did not expose pixels")
        frames.append(
            ReplayFrame(
                index=0,
                native_frame=int(info["native_frame"]),
                action=None,
                reward=0.0,
                done=False,
                png=rgb_png(pixel_result.pixels),
                collision_png=grayscale_png(_observation_frame(observation)),
            )
        )
        for index in range(1, steps + 1):
            if forced_actions is not None and index in forced_actions:
                action = int(forced_actions[index])
            else:
                action = int(agent.select_action(observation, float(epsilon), rng=rng))
            observation, reward, terminated, truncated, info = env.step(action)
            pixel_result = pixels.step_batch(np.asarray([action], dtype=np.uint8))
            if pixel_result.pixels is None:
                raise ValueError("native pixel lane did not expose pixels")
            done = bool(terminated or truncated)
            frames.append(
                ReplayFrame(
                    index=index,
                    native_frame=int(info["native_frame"]),
                    action=action,
                    reward=float(reward),
                    done=done,
                    png=rgb_png(pixel_result.pixels),
                    collision_png=grayscale_png(_observation_frame(observation)),
                )
            )
            if done:
                break
    finally:
        env.close()
    return NativeReplay(
        checkpoint=checkpoint_path,
        seed=seed,
        step_frames=int(step_frames),
        created_at=datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        frames=tuple(frames),
    )


__all__ = [
    "ACTION_NAMES",
    "DEFAULT_REPLAY_STEPS",
    "GAME_HEIGHT",
    "GAME_WIDTH",
    "MAX_REPLAY_STEPS",
    "PICO8_PALETTE",
    "NativeReplay",
    "ReplayFrame",
    "generate_native_replay",
    "grayscale_png",
    "rgb_png",
]
