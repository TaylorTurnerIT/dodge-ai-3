"""Read-only one-step dynamics controls for a frozen LeWM."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["evaluate_dynamics"]


def _mean(total: float, count: int) -> float | None:
    return total / count if count else None


def evaluate_dynamics(
    model: torch.nn.Module,
    dataset: Iterable[Mapping[str, Any]],
    device: torch.device | str,
) -> dict[str, object]:
    """Evaluate all windows with actual encoded history and one-step controls.

    The predictor is teacher-forced with observed latent history.  ``dataset``
    may yield individual windows or already-collated batches.
    """

    modules = tuple(model.modules())
    modes = tuple(module.training for module in modules)
    model.eval()
    totals = [0.0, 0.0, 0.0, 0.0]
    count = 0
    try:
        target_device = torch.device(device)
        rng_devices = []
        if target_device.type == "cuda":
            rng_devices.append(
                target_device.index
                if target_device.index is not None
                else torch.cuda.current_device()
            )
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            for sample in dataset:
                pixels = torch.as_tensor(sample["pixels"])
                actions = torch.as_tensor(sample["actions"])
                if pixels.ndim == 4:
                    pixels = pixels.unsqueeze(0)
                    actions = actions.unsqueeze(0)
                pixels, actions = pixels.to(device), actions.to(device)
                encoded = model.encode(pixels)
                predicted = model.predict(encoded[:, :-1], actions)[:, -1]
                wrong_actions = (actions + 1) % 9
                wrong = model.predict(encoded[:, :-1], wrong_actions)[:, -1]
                target, current = encoded[:, -1], encoded[:, -2]
                values = (
                    F.mse_loss(predicted, target),
                    F.mse_loss(current, target),
                    F.mse_loss(wrong, target),
                    F.mse_loss(predicted, wrong),
                )
                if not all(bool(torch.isfinite(value)) for value in values):
                    raise RuntimeError("nonfinite dynamics metric")
                batch = int(target.shape[0])
                count += batch
                for index, value in enumerate(values):
                    totals[index] += float(value) * batch
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.training = mode

    prediction_mse = _mean(totals[0], count)
    persistence_mse = _mean(totals[1], count)
    return {
        "window_count": count,
        "prediction_mse": prediction_mse,
        "persistence_mse": persistence_mse,
        "wrong_action_mse": _mean(totals[2], count),
        "prediction_vs_persistence_ratio": (
            prediction_mse / persistence_mse
            if prediction_mse is not None and persistence_mse not in (None, 0.0)
            else None
        ),
        "action_sensitivity_mse": _mean(totals[3], count),
        "scope": "one-step teacher-forced",
    }
