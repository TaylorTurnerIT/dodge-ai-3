"""Read-only one-step dynamics controls for a frozen LeWM."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import torch
import torch.nn.functional as F

__all__ = [
    "AUDIT_FRACTIONS",
    "RANK_TOLERANCE",
    "audit_action_conditioning",
    "evaluate_dynamics",
    "iter_plan_windows",
    "migrate_audit_rows",
    "plan_audit_windows",
    "summarize_audit_rows",
]

AUDIT_FRACTIONS: Final[tuple[float, float, float]] = (0.25, 0.5, 0.75)
RANK_TOLERANCE: Final[float] = 1e-9


def _mean(total: float, count: int) -> float | None:
    return total / count if count else None


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    ordered = sorted(values)
    if len(ordered) == 1:
        return {
            "p10": float(ordered[0]),
            "p50": float(ordered[0]),
            "p90": float(ordered[0]),
        }
    picks = statistics.quantiles(ordered, n=10, method="inclusive")
    return {
        "p10": float(picks[0]),
        "p50": float(statistics.median(ordered)),
        "p90": float(picks[-1]),
    }


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


def plan_audit_windows(
    records: Sequence[Any],
    *,
    history_size: int,
    fractions: Sequence[float] = AUDIT_FRACTIONS,
) -> tuple[tuple[int, int], ...]:
    """Pick deterministic varied-offset windows per episode (SPEC AC.sampling).

    Each fraction selects one window start spread across the episode so the
    audit does not align every window with the same scripted phase.
    """

    if isinstance(history_size, bool) or not isinstance(history_size, int):
        raise TypeError("history_size must be an integer")
    if history_size < 1:
        raise ValueError("history_size must be positive")
    fractions = tuple(fractions)
    if not fractions:
        raise ValueError("fractions must not be empty")
    for fraction in fractions:
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError("fractions must lie in [0, 1]")
    plan: list[tuple[int, int]] = []
    for episode_index, record in enumerate(records):
        span = int(record.count) - history_size
        if span < 0:
            continue
        for fraction in fractions:
            start = min(int(float(fraction) * span), span)
            plan.append((episode_index, start))
    if not plan:
        raise ValueError("no episode can supply a window of the requested history")
    return tuple(plan)


def iter_plan_windows(
    dataset: Any,
    records: Sequence[Any],
    plan: Sequence[tuple[int, int]],
    *,
    history_size: int,
) -> Any:
    """Yield window mappings with episode provenance for a planned audit."""

    if isinstance(history_size, bool) or not isinstance(history_size, int):
        raise TypeError("history_size must be an integer")
    if history_size < 1:
        raise ValueError("history_size must be positive")
    records = list(records)
    plan = list(plan)
    if not plan:
        raise ValueError("plan must not be empty")
    prefix = [0]
    for record in records:
        prefix.append(prefix[-1] + max(0, int(record.count) - history_size + 1))
    for episode_index, start in plan:
        sample = dataset[prefix[episode_index] + start]
        yield {
            "pixels": sample["pixels"],
            "actions": sample["actions"],
            "episode_id": records[episode_index].episode_id,
            "start": start,
        }


def _ranking(
    prediction_mse: float,
    alternative_mses: Sequence[float],
    *,
    tolerance: float = RANK_TOLERANCE,
) -> dict[str, Any]:
    """Distinguish uniquely-best, tied-best, and worse recorded actions.

    An action-invariant predictor ties every alternative, so the legacy
    "no alternative strictly better" statistic cannot be read as an
    above-chance top-1 result.  Ties are declared within ``tolerance``.
    """

    better = sum(value < prediction_mse - tolerance for value in alternative_mses)
    tied = sum(
        abs(value - prediction_mse) <= tolerance for value in alternative_mses
    )
    return {
        "better_alternative_count": better,
        "tied_alternative_count": tied,
        "recorded_uniquely_best": better == 0 and tied == 0,
        "recorded_tied_best": better == 0 and tied > 0,
    }


def _ranking_from_min(
    prediction_mse: float, min_mse: float, *, tolerance: float = RANK_TOLERANCE
) -> dict[str, bool]:
    """Recover ranking categories for rows lacking per-action errors."""

    if min_mse > prediction_mse + tolerance:
        return {"recorded_uniquely_best": True, "recorded_tied_best": False}
    if min_mse < prediction_mse - tolerance:
        return {"recorded_uniquely_best": False, "recorded_tied_best": False}
    return {"recorded_uniquely_best": False, "recorded_tied_best": True}


def _audit_window(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    *,
    action_dim: int,
) -> dict[str, Any]:
    """Score one teacher-forced window with final-action interventions."""

    encoded = model.encode(pixels)
    history, target, current = encoded[:, :-1], encoded[:, -1], encoded[:, -2]
    recorded_action = int(actions[0, -1].item())
    if not 0 <= recorded_action < action_dim:
        raise ValueError("recorded final action is outside the action range")
    recorded = model.predict(history, actions)[:, -1]
    prediction_mse = float(F.mse_loss(recorded, target))
    persistence_mse = float(F.mse_loss(current, target))
    alternative_mses: list[float] = []
    spreads: list[float] = []
    for alternative in range(action_dim):
        if alternative == recorded_action:
            continue
        varied = actions.clone()
        varied[0, -1] = alternative
        varied_prediction = model.predict(history, varied)[:, -1]
        alternative_mses.append(float(F.mse_loss(varied_prediction, target)))
        spreads.append(float(F.mse_loss(varied_prediction, recorded)))
    frames = pixels[0].to(dtype=torch.float32)
    pixel_change = float(F.mse_loss(frames[-1], frames[-2]))
    row: dict[str, Any] = {
        "prediction_mse": prediction_mse,
        "persistence_mse": persistence_mse,
        "wrong_action_mean_mse": float(sum(alternative_mses) / len(alternative_mses)),
        "wrong_action_min_mse": float(min(alternative_mses)),
        "recorded_action": recorded_action,
        "alternative_mses": alternative_mses,
        "move_from_current_mse": float(F.mse_loss(recorded, current)),
        "action_spread_mse": float(sum(spreads) / len(spreads)),
        "pixel_change": pixel_change,
        "action_changed": bool(
            actions.shape[1] >= 2
            and bool(actions[0, -1].item() != actions[0, -2].item())
        ),
    }
    row.update(_ranking(prediction_mse, alternative_mses))
    scalars = [
        value
        for key, value in row.items()
        if key != "alternative_mses" and not isinstance(value, bool)
    ]
    if not all(
        bool(torch.isfinite(torch.as_tensor(value)))
        for value in scalars + alternative_mses
    ):
        raise RuntimeError("nonfinite action audit metric")
    return row


def migrate_audit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map pre-correction audit rows to the corrected schema.

    Saved rows lack per-action errors, so count fields stay None while the
    unique/tied-best categories recover exactly from the saved minimum.
    """

    migrated: list[dict[str, Any]] = []
    for row in rows:
        prediction_mse = float(row["prediction_mse"])
        new = dict(row)
        new.pop("recorded_action_best", None)
        new["recorded_action"] = row.get("recorded_action")
        new["alternative_mses"] = None
        new["better_alternative_count"] = None
        new["tied_alternative_count"] = None
        new.update(
            _ranking_from_min(
                prediction_mse, float(row["wrong_action_min_mse"])
            )
        )
        migrated.append(new)
    return migrated


def _audit_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one window group with win/loss/tie and ranking rates."""

    rows = list(rows)
    if not rows:
        return {
            "window_count": 0,
            "prediction_mse": None,
            "persistence_mse": None,
            "prediction_win_rate": None,
            "prediction_loss_rate": None,
            "prediction_tie_rate": None,
            "recorded_uniquely_best_rate": None,
            "recorded_tied_best_rate": None,
        }
    count = len(rows)
    wins = sum(1 for row in rows if row["prediction_mse"] < row["persistence_mse"])
    losses = sum(1 for row in rows if row["prediction_mse"] > row["persistence_mse"])
    return {
        "window_count": count,
        "prediction_mse": sum(float(row["prediction_mse"]) for row in rows) / count,
        "persistence_mse": sum(float(row["persistence_mse"]) for row in rows) / count,
        "prediction_win_rate": wins / count,
        "prediction_loss_rate": losses / count,
        "prediction_tie_rate": (count - wins - losses) / count,
        "recorded_uniquely_best_rate": (
            sum(1 for row in rows if row["recorded_uniquely_best"]) / count
        ),
        "recorded_tied_best_rate": (
            sum(1 for row in rows if row["recorded_tied_best"]) / count
        ),
    }


def _win_loss_tie(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, ...]:
    """Return overall win/loss/tie rates of prediction versus persistence."""

    if not rows:
        return None, None, None
    count = len(rows)
    wins = sum(1 for row in rows if row["prediction_mse"] < row["persistence_mse"])
    losses = sum(1 for row in rows if row["prediction_mse"] > row["persistence_mse"])
    return wins / count, losses / count, (count - wins - losses) / count


def summarize_audit_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-window audit rows without touching the model.

    Pixel-change groups are relative: ``higher``/``lower`` split at the
    median change, while ``unchanged`` holds exactly identical frames where
    persistence is unbeatable by construction.
    """

    rows = list(rows)
    changes = sorted(float(row["pixel_change"]) for row in rows)
    threshold = float(statistics.median(changes)) if changes else 0.0
    higher = [row for row in rows if float(row["pixel_change"]) > threshold]
    lower = [row for row in rows if 0.0 < float(row["pixel_change"]) <= threshold]
    unchanged = [row for row in rows if float(row["pixel_change"]) == 0.0]
    prediction_mean = _mean(sum(float(r["prediction_mse"]) for r in rows), len(rows))
    persistence_mean = _mean(sum(float(r["persistence_mse"]) for r in rows), len(rows))
    win_rate, loss_rate, tie_rate = _win_loss_tie(rows)
    return {
        "window_count": len(rows),
        "pixel_change_threshold": threshold,
        "prediction_mse": prediction_mean,
        "persistence_mse": persistence_mean,
        "prediction_win_rate": win_rate,
        "prediction_loss_rate": loss_rate,
        "prediction_tie_rate": tie_rate,
        "recorded_uniquely_best_rate": (
            sum(1 for r in rows if r["recorded_uniquely_best"]) / len(rows)
            if rows
            else None
        ),
        "recorded_tied_best_rate": (
            sum(1 for r in rows if r["recorded_tied_best"]) / len(rows)
            if rows
            else None
        ),
        "wrong_action_mean_mse": _mean(
            sum(float(r["wrong_action_mean_mse"]) for r in rows), len(rows)
        ),
        "move_from_current_mse": _mean(
            sum(float(r["move_from_current_mse"]) for r in rows), len(rows)
        ),
        "action_spread_mse": _mean(
            sum(float(r["action_spread_mse"]) for r in rows), len(rows)
        ),
        "prediction_vs_persistence_ratio": (
            prediction_mean / persistence_mean
            if prediction_mean is not None and persistence_mean not in (None, 0.0)
            else None
        ),
        "prediction_quantiles": _quantiles([float(r["prediction_mse"]) for r in rows]),
        "persistence_quantiles": _quantiles(
            [float(r["persistence_mse"]) for r in rows]
        ),
        "higher_pixel_change": _audit_split(higher),
        "lower_pixel_change": _audit_split(lower),
        "unchanged": _audit_split(unchanged),
        "scope": "one-step teacher-forced action audit",
    }


def audit_action_conditioning(
    model: torch.nn.Module,
    windows: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    *,
    action_dim: int = 9,
) -> dict[str, Any]:
    """Audit frozen one-step forecasting with final-action interventions.

    Earlier observations and actions are retained; only the final action is
    varied across all alternatives.  ``windows`` yields mappings with
    ``pixels`` shaped ``(H+1, ...)`` and ``actions`` shaped ``(H,)``, plus
    optional ``episode_id`` and ``start`` provenance.
    """

    if isinstance(action_dim, bool) or not isinstance(action_dim, int):
        raise TypeError("action_dim must be an integer")
    if action_dim < 2:
        raise ValueError("action_dim must be at least two for interventions")
    modules = tuple(model.modules())
    modes = tuple(module.training for module in modules)
    model.eval()
    rows: list[dict[str, Any]] = []
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
            for sample in windows:
                pixels = torch.as_tensor(sample["pixels"])
                actions = torch.as_tensor(sample["actions"])
                if pixels.ndim == 4:
                    pixels = pixels.unsqueeze(0)
                    actions = actions.unsqueeze(0)
                row = _audit_window(
                    model,
                    pixels.to(target_device),
                    actions.to(target_device),
                    action_dim=action_dim,
                )
                row["episode_id"] = sample.get("episode_id")
                row["start"] = sample.get("start")
                rows.append(row)
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.training = mode

    summary = summarize_audit_rows(rows)
    summary["windows"] = rows
    return summary
