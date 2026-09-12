"""Offline policy checks; successful execution is not proof of useful learning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def assess_learning(evaluation: Mapping) -> dict:
    reasons = []
    splits = {}
    degenerate = False
    insufficient = False
    for name in ("inner", "holdout"):
        data = evaluation.get(name, {})
        try:
            counts = np.asarray(data["action_counts"], dtype=np.float64)
            if (
                counts.shape != (9,)
                or not np.isfinite(counts).all()
                or np.any(counts < 0)
                or np.any(counts != np.floor(counts))
                or counts.sum() <= 0
            ):
                raise ValueError("invalid action counts")
            share = float(counts.max() / counts.sum())
            splits[name] = {
                "dominant_action": int(counts.argmax()),
                "dominant_action_share": share,
            }
            if share == 1.0:
                degenerate = True
                reasons.append(f"{name}:single-action-policy")
            elif share >= 0.95:
                reasons.append(f"{name}:concentrated-action-policy")
            zeros = float(data["dead_units_mean"])
            if not math.isfinite(zeros) or not 0 <= zeros <= 1:
                raise ValueError("invalid activation sparsity")
            splits[name]["activation_zero_fraction"] = zeros
            if zeros >= 0.95:
                reasons.append(f"{name}:high-activation-sparsity")
        except (KeyError, ValueError, TypeError, OverflowError):
            insufficient = True
            reasons.append(f"{name}:insufficient-learning-diagnostics")
    status = (
        "degenerate"
        if degenerate
        else "insufficient"
        if insufficient
        else "concerning"
        if reasons
        else "unproven"
    )
    return {
        "protocol": "offline-learning-quality-v1",
        "status": status,
        "reasons": reasons,
        "splits": splits,
        "limits": (
            "Action concentration is diagnostic, not proof of cause; zero activations "
            "are not permanently dead units; unflagged policies are not "
            "proven improvements"
        ),
    }


def select_inner_checkpoint(candidates: Sequence[Mapping]) -> int:
    """Choose by frozen, uncensored inner survival only; ties prefer earlier steps."""
    if not candidates:
        raise ValueError("no eligible checkpoints")
    protocol = None
    scored = []
    seen = set()
    for candidate in candidates:
        step = candidate["step"]
        inner = candidate["inner"]
        seeds = tuple(inner["seeds"])
        cap = inner["max_steps_per_episode"]
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step < 1
            or step in seen
            or not seeds
            or len(set(seeds)) != len(seeds)
            or inner["episodes"] != len(seeds)
            or cap < 1
        ):
            raise ValueError("invalid checkpoint or inner protocol")
        seen.add(step)
        current = (seeds, cap)
        if protocol is not None and current != protocol:
            raise ValueError("inner evaluation protocols differ")
        protocol = current
        mean = float(inner["mean_survival_frames"])
        if not math.isfinite(mean) or mean < 0 or inner["censored_share"] != 0:
            raise ValueError("inner result missing, nonfinite or censored")
        scored.append((mean, -step))
    return -max(scored)[1]
