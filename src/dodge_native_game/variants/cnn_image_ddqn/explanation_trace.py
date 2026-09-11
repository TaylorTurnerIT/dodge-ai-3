"""Generate immutable, checkpoint-backed traces for policy explanations.

The trace is deliberately separate from :mod:`native_replay`: each decision
keeps the exact uint8 model input seen before the action, the model's raw
outputs, and the native result produced after that action.  A second native
lane supplies the rendered game PNG for the same pre-action state.  Neither
lane is shared with a live trainer and this module never performs an optimizer
update or injects an action.
"""

from __future__ import annotations

import base64
import contextlib
import json
import operator
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ...batch import NativeBatchEnvironment
from .env import CNNImageDDQNEnv
from .model import AtariCnnQNetwork, to_float_observations
from .native_replay import _infer_dueling, rgb_png
from .provenance import infer_parent_run_id, sha256_file
from .run import _choose_device, _configure_torch_backend, _load_checkpoint

FRAME_HEIGHT: Final = 84
FRAME_WIDTH: Final = 84
ACTION_COUNT: Final = 9
DEFAULT_TRACE_STEPS: Final = 512
MAX_TRACE_STEPS: Final = 4096
TRACE_SCHEMA_VERSION: Final = 1

# These are the stable presence bits emitted by dodge-python's native boundary.
# They are flags for the completed native step, not event counts.
_EVENT_BITS: Final = (
    ("enemy_spawn", 1 << 0),
    ("collision", 1 << 1),
    ("death", 1 << 2),
    ("pattern_active", 1 << 3),
    ("terminal", 1 << 4),
)


@dataclass(frozen=True, slots=True)
class ExplanationTrace:
    """A bounded explanation replay and its model-side inference objects.

    ``metadata`` is JSON-ready and contains the per-decision rows under
    ``metadata["frames"]``.  ``observations`` and ``game_pngs`` stay separate
    so the HTTP layer can serve PNG bytes without embedding binary data in the
    JSON response.  Observation arrays are owned uint8 ``(C,84,84)`` inputs;
    they are made read-only before being returned.
    """

    metadata: dict[str, Any]
    model: AtariCnnQNetwork
    observations: tuple[np.ndarray, ...]
    game_pngs: tuple[bytes, ...]
    final_game_png: bytes | None = None

    @property
    def frames(self) -> tuple[dict[str, Any], ...]:
        """Return the JSON-ready decision rows without exposing list mutation."""

        return tuple(self.metadata.get("frames", ()))

    @property
    def frame_count(self) -> int:
        """Number of pre-action decisions in the trace."""

        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload; game PNG bytes remain out-of-band."""

        return dict(self.metadata)

    as_dict = to_dict


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read an adjacent run artifact without making it a generation blocker."""

    try:
        if not path.is_file() or path.is_symlink():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _checkpoint_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"checkpoint is not a regular file: {path}")
    return path


def _integer(value: object, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 0 or (maximum is not None and result > maximum):
        bound = f" between 0 and {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a non-negative integer{bound}")
    return int(result)


def _validate_request(
    *,
    seed: object,
    steps: object,
    step_frames: object,
    difficulty: object,
    patterns: object,
    powerups: object,
) -> tuple[int, int, int, int, bool, bool]:
    seed_value = _integer(seed, "seed", maximum=32_767)
    steps_value = _integer(steps, "steps")
    if not 1 <= steps_value <= MAX_TRACE_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_TRACE_STEPS}")
    step_value = _integer(step_frames, "step_frames")
    # BatchConfig accepts only the native decision intervals 3..5.
    if not 3 <= step_value <= 5:
        raise ValueError("step_frames must be between 3 and 5")
    difficulty_value = _integer(difficulty, "difficulty")
    if difficulty_value not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2, or 3")
    if not isinstance(patterns, bool) or not isinstance(powerups, bool):
        raise TypeError("patterns and powerups must be booleans")
    return (
        seed_value,
        steps_value,
        step_value,
        difficulty_value,
        patterns,
        powerups,
    )


def _observation_to_uint8(
    observation: object, expected_shape: tuple[int, int, int]
) -> np.ndarray:
    """Copy one model input to the exact uint8 replay representation."""

    value = np.asarray(observation)
    if value.shape != expected_shape:
        raise ValueError(
            f"observation must have shape {expected_shape}, got {value.shape}"
        )
    if value.dtype == np.uint8:
        return np.ascontiguousarray(value, dtype=np.uint8).copy()
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError("observation must contain numeric image values")
    numeric = np.asarray(value, dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("observation values must be finite")
    if np.any(numeric < 0.0) or np.any(numeric > 1.0):
        raise ValueError("float observations must be in [0, 1]")
    return np.rint(numeric * np.float32(255.0)).astype(np.uint8, copy=True)


def _lane_field(result: object, name: str) -> object | None:
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def _lane_scalar(result: object, name: str) -> object | None:
    value = _lane_field(result, name)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"native result field {name!r} must contain one lane")
    return array.reshape(-1)[0].item()


def _native_events(
    info: Mapping[str, Any],
) -> tuple[dict[str, bool] | None, int | None]:
    """Read native event presence without turning missing data into no-events."""

    if "native_event_flags" not in info:
        return None, None
    raw = info["native_event_flags"]
    if raw is None:
        return None, None
    try:
        flags = _integer(raw, "native_event_flags")
    except (TypeError, ValueError):
        return None, None
    return {name: bool(flags & bit) for name, bit in _EVENT_BITS}, flags


def _native_scalar(info: Mapping[str, Any], name: str) -> float | int | None:
    """Keep optional native diagnostics absent when the info field is absent."""

    if name not in info or info[name] is None:
        return None
    try:
        value = float(info[name])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _native_position(result: object) -> list[float] | None:
    value = _lane_field(result, "player_positions")
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 2:
        return None
    position = np.asarray(array.reshape(2), dtype=np.float64)
    if not np.isfinite(position).all():
        return None
    return [float(position[0]), float(position[1])]


def _predict(
    model: AtariCnnQNetwork,
    input_uint8: np.ndarray,
) -> tuple[np.ndarray, float | None, np.ndarray | None, int]:
    """Return raw Q, decomposed head values, and greedy action for one input."""

    tensor = torch.as_tensor(input_uint8, device=next(model.parameters()).device)
    tensor = tensor.unsqueeze(0)
    with torch.inference_mode():
        q_output = model(tensor)
        q = q_output[0].detach().cpu().numpy().astype(np.float64, copy=True)
        if model.dueling:
            features = model.shared(
                model.features(
                    to_float_observations(
                        tensor, expected_shape=model.observation_shape
                    )
                )
            )
            value = model.value_stream(features)[0, 0]
            advantage = model.advantage_stream(features)[0]
            centered = advantage - advantage.mean()
            value_value = float(value.detach().cpu().item())
            centered_value = centered.detach().cpu().numpy().astype(
                np.float64, copy=True
            )
        else:
            value_value = None
            centered_value = None
    action = int(np.argmax(q))
    return q, value_value, centered_value, action


def _torch_nnpack_state() -> bool | None:
    getter = getattr(getattr(torch, "_C", None), "_get_nnpack_enabled", None)
    if not callable(getter):
        return None
    try:
        return bool(getter())
    except (RuntimeError, TypeError):
        return None


def _restore_torch_nnpack(state: bool | None) -> None:
    if state is None:
        return
    setter = getattr(getattr(torch, "_C", None), "_set_nnpack_enabled", None)
    if callable(setter):
        with contextlib.suppress(RuntimeError, TypeError):
            setter(state)


def _json_copy(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    # JSON round-tripping removes Path/NumPy values from test doubles while
    # retaining the artifact shape used by the dashboard.
    try:
        copied = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return None
    return dict(copied) if isinstance(copied, dict) else None


def _metadata_for_checkpoint(
    checkpoint: Path,
    payload: Mapping[str, Any],
    *,
    seed: int,
    step_frames: int,
    difficulty: int,
    patterns: bool,
    powerups: bool,
    steps: int,
    started_at: str,
) -> dict[str, Any]:
    run_dir = (
        checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else None
    )
    config = _json_copy(_read_json(run_dir / "config.json")) if run_dir else None
    manifest = _json_copy(_read_json(run_dir / "manifest.json")) if run_dir else None
    run_id = infer_parent_run_id(checkpoint, payload.get("run_id"))
    shape = tuple(int(item) for item in payload["observation_shape"])
    dueling = _infer_dueling(dict(payload))
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "variant_id": payload.get("variant_id"),
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
        "checkpoint_step": payload.get("global_environment_step", payload.get("step")),
        "seed": seed,
        "native_config": {
            "step_frames": step_frames,
            "difficulty": difficulty,
            "patterns": patterns,
            "powerups": powerups,
        },
        "observation_shape": list(shape),
        "input_dtype": "uint8",
        "input_encoding": "base64",
        "model": {
            "num_actions": int(payload["num_actions"]),
            "dueling": dueling,
            "architecture": (
                "atari-cnn-dueling-ddqn" if dueling else "atari-cnn-plain-ddqn"
            ),
        },
        "config": config,
        "manifest": manifest,
        "started_at": started_at,
        # Filled after generation; keeping these keys present makes incomplete
        # trace payloads explicit if a caller snapshots metadata while running.
        "frames": [],
        "observations": {
            "shape": [0, *shape],
            "dtype": "uint8",
            "encoding": "base64",
            "input_base64": "",
        },
        "termination": {},
        "cap": {
            "max_steps": MAX_TRACE_STEPS,
            "requested_steps": steps,
        },
        "generation": {},
    }


def _set_twin_parity(
    info: Mapping[str, Any], pixel_result: object, *, where: str
) -> None:
    """Assert frame/done parity when both native lanes expose the fields."""

    env_frame = info.get("native_frame")
    pixel_frame = _lane_scalar(pixel_result, "frames")
    if (
        env_frame is not None
        and pixel_frame is not None
        and int(env_frame) != int(pixel_frame)
    ):
        raise RuntimeError(
            f"native replay lanes diverged at {where}: "
            f"frame {env_frame} != {pixel_frame}"
        )
    env_done = info.get("native_done")
    pixel_done = _lane_scalar(pixel_result, "done")
    if (
        env_done is not None
        and pixel_done is not None
        and bool(env_done) != bool(pixel_done)
    ):
        raise RuntimeError(
            f"native replay lanes diverged at {where}: "
            f"done {env_done} != {pixel_done}"
        )


def _generate(
    checkpoint: Path,
    payload: Mapping[str, Any],
    *,
    seed: int,
    steps: int,
    device: str,
    step_frames: int,
    difficulty: int,
    patterns: bool,
    powerups: bool,
    started_at: str,
    started_clock: float,
) -> ExplanationTrace:
    shape = tuple(int(item) for item in payload["observation_shape"])
    num_actions = int(payload["num_actions"])
    dueling = _infer_dueling(dict(payload))
    model = AtariCnnQNetwork(
        num_actions,
        dueling=dueling,
        input_channels=shape[0],
    ).to(device)
    model.eval()
    # Use the same strict checkpoint validation as the training/replay paths,
    # but keep the returned object as the online model only.
    from .agent import DoubleDQNAgent

    agent = DoubleDQNAgent(
        num_actions=num_actions,
        device=device,
        observation_shape=shape,
        online_network=model,
        target_network=AtariCnnQNetwork(
            num_actions,
            dueling=dueling,
            input_channels=shape[0],
        ),
    )
    _load_checkpoint(agent, checkpoint)
    model = agent.online_network
    model.eval()

    env = CNNImageDDQNEnv(
        stack_size=shape[0],
        step_frames=step_frames,
        difficulty=difficulty,
        patterns=patterns,
        powerups=powerups,
    )
    pixels = NativeBatchEnvironment(
        step_frames=step_frames,
        full_state=False,
        pixels=True,
        board=False,
        difficulty=difficulty,
        patterns_enabled=patterns,
        powerups_enabled=powerups,
        ml=True,
    )
    observations: list[np.ndarray] = []
    game_pngs: list[bytes] = []
    rows: list[dict[str, Any]] = []
    try:
        observation, reset_info = env.reset(seed=seed)
        pixel_result = pixels.reset_batch(np.asarray([seed], dtype=np.uint32))
        _set_twin_parity(reset_info, pixel_result, where="reset")
        reset_position = _native_position(pixel_result)
        pre_info: Mapping[str, Any] = reset_info
        for index in range(steps):
            input_uint8 = _observation_to_uint8(observation, shape)
            q, value, centered_advantage, action = _predict(model, input_uint8)
            pixel_values = _lane_field(pixel_result, "pixels")
            if pixel_values is None:
                raise ValueError("native pixel lane did not expose pixels")
            observations.append(input_uint8)
            game_pngs.append(rgb_png(pixel_values))

            (
                next_observation,
                reward,
                terminated,
                truncated,
                post_info,
            ) = env.step(action)
            next_pixel_result = pixels.step_batch(
                np.asarray([action], dtype=np.uint8)
            )
            _set_twin_parity(post_info, next_pixel_result, where=f"step {index}")
            events, event_flags = _native_events(post_info)
            native_score = _native_scalar(post_info, "native_score")
            native_shattered = _native_scalar(post_info, "native_shattered")
            row: dict[str, Any] = {
                "index": index,
                "native_frame": int(pre_info["native_frame"]),
                "action": action,
                "actionargmax": action,
                "q": q.tolist(),
                "value": value,
                "centered_advantage": (
                    centered_advantage.tolist()
                    if centered_advantage is not None
                    else None
                ),
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "events": events,
                "native_events": events,
                "native_event_flags": event_flags,
                "native_score": native_score,
                "native_shattered": native_shattered,
                "post_native_frame": post_info.get("native_frame"),
                "post_native_done": post_info.get("native_done"),
                # The packed blob is in metadata["observations"]; this offset
                # keeps each frame JSON-small while retaining a direct input key.
                "input": {"index": index},
                "game_frame": index,
            }
            rows.append(row)
            observation = next_observation
            pre_info = post_info
            pixel_result = next_pixel_result
            if terminated or truncated:
                break
    finally:
        with contextlib.suppress(Exception):
            env.close()
        with contextlib.suppress(Exception):
            pixels.close()

    elapsed = max(time.perf_counter() - started_clock, 0.0)
    finished_at = datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    if observations:
        packed = np.ascontiguousarray(np.stack(observations, axis=0), dtype=np.uint8)
        packed_b64 = base64.b64encode(packed.tobytes()).decode("ascii")
        input_shape = list(packed.shape)
    else:
        packed_b64 = ""
        input_shape = [0, *shape]
    for observation in observations:
        observation.setflags(write=False)

    native_terminated = bool(rows and rows[-1]["terminated"])
    native_truncated = bool(rows and rows[-1]["truncated"])
    reached_cap = bool(
        rows
        and len(rows) >= steps
        and not native_terminated
        and not native_truncated
    )
    if native_terminated:
        termination_reason = "native_terminal"
    elif native_truncated:
        termination_reason = "environment_truncated"
    elif reached_cap:
        termination_reason = "trace_cap"
    else:
        termination_reason = "no_decisions"
    metadata = _metadata_for_checkpoint(
        checkpoint,
        payload,
        seed=seed,
        step_frames=step_frames,
        difficulty=difficulty,
        patterns=patterns,
        powerups=powerups,
        steps=steps,
        started_at=started_at,
    )
    metadata["frames"] = rows
    metadata["observations"] = {
        "shape": input_shape,
        "dtype": "uint8",
        "encoding": "base64",
        "input_base64": packed_b64,
    }
    metadata["reset_position"] = reset_position
    metadata["termination"] = {
        "reason": termination_reason,
        "terminated": native_terminated,
        "truncated": native_truncated,
        "native_frame": rows[-1].get("post_native_frame") if rows else None,
    }
    metadata["cap"] = {
        "max_steps": MAX_TRACE_STEPS,
        "requested_steps": steps,
        "decisions": len(rows),
        "reached": reached_cap,
    }
    metadata["generation"] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }
    return ExplanationTrace(
        metadata=metadata,
        model=model,
        observations=tuple(observations),
        game_pngs=tuple(game_pngs),
        final_game_png=rgb_png(_lane_field(pixel_result, "pixels")),
    )


def generate_explanation_trace(
    checkpoint: Path | str,
    *,
    seed: int,
    steps: int = DEFAULT_TRACE_STEPS,
    device: str = "cpu",
    step_frames: int = 4,
    difficulty: int = 2,
    patterns: bool = True,
    powerups: bool = True,
) -> ExplanationTrace:
    """Generate a greedy offline explanation trace from one checkpoint.

    The seed and game settings are explicit, and every action is the online
    checkpoint's argmax.  The generated trace is bounded to 4096 decisions;
    reaching that bound is reported as a trace cap rather than mislabeled as a
    native terminal or Gymnasium truncation.
    """

    checkpoint_path = _checkpoint_path(checkpoint)
    (
        seed_value,
        steps_value,
        step_value,
        difficulty_value,
        patterns_value,
        powerups_value,
    ) = (
        _validate_request(
            seed=seed,
            steps=steps,
            step_frames=step_frames,
            difficulty=difficulty,
            patterns=patterns,
            powerups=powerups,
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    if payload.get("variant_id") != "cnn-image-ddqn":
        raise ValueError("checkpoint variant does not match cnn-image-ddqn")
    shape_value = payload.get("observation_shape")
    if not isinstance(shape_value, (tuple, list)) or len(shape_value) != 3:
        raise ValueError("checkpoint observation_shape is invalid")
    shape = tuple(int(item) for item in shape_value)
    if shape[0] < 1 or shape[1:] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError("checkpoint observation_shape must be (C, 84, 84)")
    try:
        num_actions = _integer(payload["num_actions"], "num_actions")
    except KeyError as error:
        raise ValueError("checkpoint num_actions is missing") from error
    if num_actions < 1:
        raise ValueError("checkpoint num_actions must be positive")
    if num_actions != ACTION_COUNT:
        raise ValueError(
            f"checkpoint num_actions must match native action count {ACTION_COUNT}"
        )
    if "online_network" not in payload:
        raise ValueError("checkpoint online_network is missing")

    chosen_device = _choose_device(device)
    previous_nnpack = _torch_nnpack_state()
    _configure_torch_backend(chosen_device)
    started_clock = time.perf_counter()
    started_at = datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    try:
        return _generate(
            checkpoint_path,
            payload,
            seed=seed_value,
            steps=steps_value,
            device=chosen_device,
            step_frames=step_value,
            difficulty=difficulty_value,
            patterns=patterns_value,
            powerups=powerups_value,
            started_at=started_at,
            started_clock=started_clock,
        )
    finally:
        _restore_torch_nnpack(previous_nnpack)


__all__ = [
    "DEFAULT_TRACE_STEPS",
    "ExplanationTrace",
    "MAX_TRACE_STEPS",
    "TRACE_SCHEMA_VERSION",
    "generate_explanation_trace",
]
