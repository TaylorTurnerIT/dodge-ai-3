"""Headless 1-step greedy MPC evaluation for AD5 (Track B).

Read-only world model + frozen survival probe steer fresh scenarios:
encode the frame history, predict all 9 action consequences, take the
lowest-cost action.  Reports survival frames per episode on train-seed vs
novel-seed scenario sets plus random/neutral baselines.  Nothing here
updates the world model or fits a policy.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

EXPERIMENT: Final[str] = "lewm-mpc-eval-v1"
ACTION_COUNT: Final[int] = 9
NEUTRAL_ACTION: Final[int] = 0
MAX_DECISIONS: Final[int] = 128
ABORTED_UNKNOWN_COLOR: Final[str] = "aborted-unknown-color"
SHAKE_BLACK_RGB: Final[tuple[int, int, int]] = (0, 0, 0)
PLAYFIELD_BACKGROUND_RGB: Final[tuple[int, int, int]] = (41, 173, 255)


class _UnknownColorAbort(Exception):
    """A steering-policy frame carried RGB outside the model palette.

    Raised only for the palette contract refusal so evaluation can record
    the abort explicitly; every other exception still propagates.
    """


def mask_black_pixels(stacked: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace exact-black pixels with playfield background (AD6.rule).

    Screen-shake strips and game-over text shadows render (0, 0, 0), a
    color absent from every training corpus (AD5.diagnosis).  A shake
    strip exposes out-of-view playfield whose training-time content is
    background, so exact-black maps to background blue.  Any other
    off-palette color passes through untouched so the encoder's strict
    refusal still fires (fail-closed).  Returns the masked copy and the
    masked pixel count.
    """

    black_rgb = np.asarray(SHAKE_BLACK_RGB, dtype=np.uint8).reshape(1, 3, 1, 1)
    black = (stacked == black_rgb).all(axis=1, keepdims=True)
    count = int(black.sum())
    if not count:
        return stacked.copy(), 0
    background = np.asarray(PLAYFIELD_BACKGROUND_RGB, dtype=np.uint8).reshape(
        1, 3, 1, 1
    )
    return np.where(black, background, stacked).astype(np.uint8), count


def _history_batch(
    frames: deque[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, int]:
    stacked = np.stack(list(frames), axis=0).astype(np.uint8)
    masked, count = mask_black_pixels(stacked)
    return torch.from_numpy(masked).unsqueeze(0).to(device), count


@torch.no_grad()
def _greedy_action(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    history: torch.Tensor,
    past_actions: list[int],
    device: torch.device,
) -> tuple[int, list[float]]:
    z = model.encode(history)
    past = torch.tensor(past_actions, dtype=torch.int64, device=device)
    actions = past.unsqueeze(0).expand(ACTION_COUNT, -1).clone()
    actions[:, -1] = torch.arange(ACTION_COUNT, device=device)
    # Predictor context is H latents for H actions (training calls
    # predict(encoded[:, :-1], actions)); the H+1-frame rollout buffer
    # carries one extra frame, so drop the oldest before predicting.
    z_tiled = z.expand(ACTION_COUNT, -1, -1)[:, -actions.shape[1] :, :]
    predicted = model.predict(z_tiled, actions)
    costs = probe(predicted[:, -1, :]).reshape(-1).tolist()
    best = int(min(range(ACTION_COUNT), key=lambda a: costs[a]))
    return best, [float(value) for value in costs]


def run_episode(
    make_adapter: Callable[[], Any],
    seed: int,
    policy: Callable[[torch.Tensor, list[int]], int],
    *,
    model: torch.nn.Module,
    device: torch.device,
    history_size: int = 3,
    max_decisions: int = MAX_DECISIONS,
) -> dict[str, Any]:
    """Roll one episode under ``policy``; policy sees frames + past actions."""

    adapter = make_adapter()
    try:
        frame = adapter.reset(seed=seed)
        frames: deque[np.ndarray] = deque(
            [np.asarray(frame, dtype=np.uint8)] * (history_size + 1),
            maxlen=history_size + 1,
        )
        past_actions: list[int] = [NEUTRAL_ACTION] * history_size
        survived = 0
        outcome = "truncated"
        masked_pixels = 0
        for _ in range(max_decisions):
            history, newly_masked = _history_batch(frames, device)
            masked_pixels += newly_masked
            try:
                action = policy(history, list(past_actions))
            except _UnknownColorAbort:
                outcome = ABORTED_UNKNOWN_COLOR
                break
            frame, _reward, terminated, truncated = adapter.step(action)
            frames.append(np.asarray(frame, dtype=np.uint8))
            past_actions.append(int(action))
            past_actions = past_actions[-history_size:]
            if terminated:
                outcome = "terminated"
                break
            survived += 1
            if truncated:
                outcome = "truncated"
                break
        else:
            survived = max_decisions
        return {
            "seed": seed,
            "survived": survived,
            "outcome": outcome,
            "masked_pixels": masked_pixels,
        }
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def plan_mpc_eval() -> list[tuple[str, Any, int]]:
    """Build train-type vs novel-type headless scenario sets (fresh seeds).

    Set A mirrors the ordinary native distribution (difficulty 1-2, mixed
    options); set B pushes novel hard corners (difficulty 2-3, no powerups,
    permanent patterns).  Both use seeds disjoint from every training,
    validation, and probe range.  The distribution shift from scripted
    training recipes to organic play is intentional: it tests
    generalization, and baselines run the identical scenarios.
    """

    from .scenario import ScenarioConfig

    scenarios: list[tuple[str, Any, int]] = []
    index = 0
    for difficulty in (1, 2):
        for enemy_mode in ("all", "normal"):
            for patterns in (True, False):
                for repeat in range(2):
                    scenarios.append(
                        (
                            f"ordinary-d{difficulty}-{enemy_mode}"
                            f"-{'pat' if patterns else 'flat'}-{repeat}",
                            ScenarioConfig(
                                name=f"mpc-ordinary-{index:02d}",
                                difficulty=difficulty,
                                enemy_mode=enemy_mode,  # type: ignore[arg-type]
                                patterns_enabled=patterns,
                                powerups_enabled=True,
                                permanent_pattern=0,
                            ),
                            24000 + index,
                        )
                    )
                    index += 1
    permanents = (3, 11, 17, 23, 29, 35, 7, 31)
    for difficulty in (2, 3):
        for repeat, permanent in enumerate(permanents):
            scenarios.append(
                (
                    f"novel-d{difficulty}-p{permanent}-{repeat % 2}",
                    ScenarioConfig(
                        name=f"mpc-novel-{index:02d}",
                        difficulty=difficulty,
                        enemy_mode="all",
                        patterns_enabled=True,
                        powerups_enabled=False,
                        permanent_pattern=permanent,
                    ),
                    25000 + index,
                )
            )
            index += 1
    return scenarios


def evaluate(
    checkpoint: Path,
    probe_path: Path,
    scenarios: list[tuple[str, Any, int]],
    *,
    policies: dict[str, str],
    device: str = "cpu",
    history_size: int = 3,
    max_decisions: int = MAX_DECISIONS,
) -> dict[str, Any]:
    """Run every policy over every scenario; report survival frames."""

    from .native_adapter import PixelNativeAdapter
    from .pretrain import load_model
    from .run_artifacts import file_hash
    from .survival_probe import SurvivalProbe

    device_obj = torch.device(device)
    torch.manual_seed(906)
    np.random.seed(906)
    world_hash = file_hash(checkpoint)
    probe_payload = torch.load(probe_path, map_location="cpu", weights_only=True)
    if probe_payload.get("world_model_sha256") != world_hash:
        raise ValueError("probe was fitted for a different world checkpoint")
    model, _ = load_model(checkpoint)
    model = model.to(device_obj).eval()
    model.requires_grad_(False)
    probe = SurvivalProbe()
    probe.load_state_dict(probe_payload["model"], strict=True)
    probe = probe.to(device_obj).eval()
    probe.requires_grad_(False)

    def mpc_policy(history: torch.Tensor, past: list[int]) -> int:
        try:
            action, _ = _greedy_action(model, probe, history, past, device_obj)
        except ValueError as error:
            if "outside configured palette" not in str(error):
                raise
            raise _UnknownColorAbort from error
        return action

    def random_policy(history: torch.Tensor, past: list[int]) -> int:
        del history, past
        return int(torch.randint(ACTION_COUNT, (1,)).item())

    def neutral_policy(history: torch.Tensor, past: list[int]) -> int:
        del history, past
        return NEUTRAL_ACTION

    available = {"mpc": mpc_policy, "random": random_policy, "neutral": neutral_policy}
    for name in policies.values():
        if name not in available:
            raise ValueError(f"unknown policy: {name}")
    began = time.monotonic()
    episodes: list[dict[str, Any]] = []
    for label, scenario, seed in scenarios:
        adapter_factory = lambda scenario=scenario: PixelNativeAdapter(  # noqa: E731
            scenario=scenario
        )
        row: dict[str, Any] = {"scenario": label, "seed": seed}
        for key, policy_name in policies.items():
            result = run_episode(
                adapter_factory,
                seed,
                available[policy_name],
                model=model,
                device=device_obj,
                history_size=history_size,
                max_decisions=max_decisions,
            )
            row[key] = result["survived"]
            row[key + "_outcome"] = result["outcome"]
            row[key + "_masked_pixels"] = result["masked_pixels"]
        episodes.append(row)
    summary: dict[str, Any] = {}
    for key in policies:
        survived = [row[key] for row in episodes]
        aborted = sum(
            1 for row in episodes if row[key + "_outcome"] == ABORTED_UNKNOWN_COLOR
        )
        completed = [
            row[key]
            for row in episodes
            if row[key + "_outcome"] != ABORTED_UNKNOWN_COLOR
        ]
        summary[key] = {
            "mean": float(sum(survived) / len(survived)),
            "median": float(sorted(survived)[len(survived) // 2]),
            "min": min(survived),
            "max": max(survived),
            "completed": len(completed),
            "aborted_unknown_color": aborted,
            "masked_pixels_total": sum(
                row[key + "_masked_pixels"] for row in episodes
            ),
            "mean_completed": float(sum(completed) / len(completed))
            if completed
            else None,
        }
    report = {
        "experiment": EXPERIMENT,
        "world_model_sha256": world_hash,
        "probe_sha256": file_hash(probe_path),
        "device": device_obj.type,
        "policies": policies,
        "episodes": episodes,
        "summary": summary,
        "elapsed_seconds": time.monotonic() - began,
    }
    return report


__all__ = [
    "ABORTED_UNKNOWN_COLOR",
    "ACTION_COUNT",
    "EXPERIMENT",
    "MAX_DECISIONS",
    "NEUTRAL_ACTION",
    "PLAYFIELD_BACKGROUND_RGB",
    "SHAKE_BLACK_RGB",
    "evaluate",
    "mask_black_pixels",
    "run_episode",
]
