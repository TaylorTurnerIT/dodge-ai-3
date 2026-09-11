"""Run-anchored replay comparison: best / median / worst game per run.

Reads a completed run's own artifacts (``config.json``, ``report.json`` /
``evaluation.json``) so a replay shows that run's checkpoint with that run's
game settings on that run's frozen eval seeds (inner + holdout only; the
forced-action counterfactual stays out of the ranking). No new game semantics: seeds,
rewards, and survival come from the stored evaluation; Rust re-simulates the
episode for pixels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .diagnostics import best_greedy_episode, greedy_eval_rows
from .native_replay import MAX_REPLAY_STEPS, NativeReplay, generate_native_replay
from .run_artifacts import VARIANT_ID

EPISODE_KEYS: Final = ("best", "median", "worst")
_EVAL_SOURCES: Final = ("inner", "holdout", "counterfactual")


@dataclass(frozen=True, slots=True)
class ComparisonEpisode:
    """One frozen eval episode selected for side-by-side replay."""

    key: str
    source: str
    seed: int
    eval_reward: float
    survival_frames: int


@dataclass(frozen=True, slots=True)
class RunReplayConfig:
    """Checkpoint + game settings resolved from one run directory."""

    run_id: str
    checkpoint: Path
    step_frames: int
    difficulty: int
    patterns: bool
    powerups: bool
    dueling: bool | None
    eval_max_steps: int
    evaluation: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(value) if isinstance(value, dict) else None


def _variant_run_dir(history_root: Path | str, run_id: str) -> Path:
    root = Path(history_root).expanduser()
    candidates = [
        root / VARIANT_ID / run_id,
        root / run_id,
    ]
    if root.name == VARIANT_ID:
        candidates.insert(0, root / run_id)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return root / VARIANT_ID / run_id


def list_eval_episodes(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pool inner/holdout/counterfactual episodes into comparable rows."""

    rows: list[dict[str, Any]] = []
    for source in _EVAL_SOURCES:
        section = evaluation.get(source)
        if not isinstance(section, Mapping):
            continue
        seeds = section.get("seeds")
        rewards = section.get("rewards")
        survival = section.get("survival_frames", [])
        if not isinstance(seeds, Sequence) or not isinstance(rewards, Sequence):
            continue
        for index, (seed, reward) in enumerate(zip(seeds, rewards, strict=False)):
            try:
                rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "eval_reward": float(reward),
                        "survival_frames": int(survival[index])
                        if index < len(survival)
                        else 0,
                    }
                )
            except (TypeError, ValueError):
                continue
    # Back-compat: runs predating the split store one flat seed/reward list.
    if not rows:
        seeds = evaluation.get("seeds")
        rewards = evaluation.get("rewards")
        if isinstance(seeds, Sequence) and isinstance(rewards, Sequence):
            for seed, reward in zip(seeds, rewards, strict=False):
                try:
                    rows.append(
                        {
                            "source": "eval",
                            "seed": int(seed),
                            "eval_reward": float(reward),
                            "survival_frames": 0,
                        }
                    )
                except (TypeError, ValueError):
                    continue
    return rows


def select_comparison_episodes(
    evaluation: Mapping[str, Any],
) -> dict[str, ComparisonEpisode]:
    """Pick best / median / worst episodes deterministically.

    Only inner + holdout seeds count: both run the pure greedy policy, so
    the three replays compare like with like. The counterfactual probe
    forces the least-used action on move one (off-policy) and stays out
    of the ranking. Ranking follows the shared ``diagnostics`` order —
    (reward, survival, lowest seed) — so the "best" here is the same
    episode the run report names as its best final episode. Median is the
    middle row after sorting (lower-middle for even counts).
    """

    rows = greedy_eval_rows(evaluation) or list_eval_episodes(evaluation)
    if not rows:
        raise ValueError("evaluation has no per-episode seeds/rewards to replay")
    ordered = sorted(
        rows,
        key=lambda row: (row["eval_reward"], row["survival_frames"], -row["seed"]),
    )
    median_row = ordered[len(ordered) // 2] if len(ordered) > 2 else ordered[0]
    picks = {
        "best": ordered[-1],
        "median": median_row,
        "worst": ordered[0],
    }
    return {
        key: ComparisonEpisode(
            key=key,
            source=str(row["source"]),
            seed=int(row["seed"]),
            eval_reward=float(row["eval_reward"]),
            survival_frames=int(row["survival_frames"]),
        )
        for key, row in picks.items()
    }


def resolve_run_replay_config(
    history_root: Path | str,
    run_id: str,
) -> RunReplayConfig:
    """Load one run's checkpoint + game settings + evaluation table."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"unknown run: {run_id!r}")
    run_dir = _variant_run_dir(history_root, run_id)
    config = _read_json(run_dir / "config.json") or {}
    report = _read_json(run_dir / "report.json") or {}
    evaluation = _read_json(run_dir / "evaluation.json")
    if evaluation is None:
        report_eval = report.get("evaluation")
        evaluation = dict(report_eval) if isinstance(report_eval, Mapping) else None
    if evaluation is None:
        raise FileNotFoundError(f"no evaluation.json for run {run_id}")
    if not list_eval_episodes(evaluation):
        raise ValueError(f"evaluation for run {run_id} has no replayable episodes")

    checkpoint_rel = report.get("checkpoint")
    checkpoint: Path | None = None
    if isinstance(checkpoint_rel, str) and checkpoint_rel:
        candidate = run_dir / checkpoint_rel
        if candidate.is_file() and not candidate.is_symlink():
            checkpoint = candidate
    if checkpoint is None:
        checkpoints = sorted(
            (
                path
                for path in (run_dir / "checkpoints").rglob("*.pt")
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        if checkpoints:
            checkpoint = checkpoints[-1]
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint found for run {run_id}")

    game = config.get("game", {}) if isinstance(config.get("game"), Mapping) else {}
    controls = (
        config.get("dashboard_game_controls", {})
        if isinstance(config.get("dashboard_game_controls"), Mapping)
        else {}
    )
    evaluation_cfg = (
        config.get("evaluation", {})
        if isinstance(config.get("evaluation"), Mapping)
        else {}
    )
    model_cfg = (
        config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    )
    dueling_cfg = model_cfg.get("dueling")
    return RunReplayConfig(
        run_id=run_id,
        checkpoint=checkpoint,
        step_frames=int(game.get("step_frames", 4)),
        difficulty=int(controls.get("difficulty", 2)),
        patterns=bool(controls.get("patterns", True)),
        powerups=bool(controls.get("powerups", True)),
        dueling=None if dueling_cfg is None else bool(dueling_cfg),
        eval_max_steps=int(evaluation_cfg.get("max_steps_per_episode", 64)),
        evaluation=evaluation,
    )


MAX_CURVE_POINTS: Final = 240
MAX_CURVE_BYTES: Final = 64_000_000


def run_training_curves(
    run_dir: Path | str, *, max_points: int = MAX_CURVE_POINTS
) -> dict[str, Any]:
    """Downsample one run's training curve for plotted context.

    Returns evenly strided ``points`` of ``{step, reward, survival_frames}``
    plus the exact ``train_max`` (``{reward, step}``) scanned over every
    row. The training max is labeled as what it is — an exploration-era
    episode under an older policy — so plots can show it next to the
    final-eval best without confusing the two.
    """

    path = Path(run_dir).expanduser() / "metrics.jsonl"
    points: list[dict[str, float]] = []
    train_max = {"reward": 0.0, "step": 0}
    try:
        if path.is_symlink() or path.stat().st_size > MAX_CURVE_BYTES:
            return {"points": points, "train_max": train_max, "rows": 0}
        with path.open("r", encoding="utf-8") as stream:
            lines = stream.readlines()
    except OSError:
        return {"points": points, "train_max": train_max, "rows": 0}
    rows: list[tuple[float, float, float]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        try:
            step = float(record.get("step", 0) or 0)
            reward = float(record.get("reward", 0.0) or 0.0)
            survival = float(record.get("survival_frames", 0) or 0)
        except (TypeError, ValueError):
            continue
        rows.append((step, reward, survival))
        if reward > float(train_max["reward"]):
            train_max = {"reward": reward, "step": step}
    stride = max(1, -(-len(rows) // max(1, int(max_points))))
    points = [
        {"step": step, "reward": reward, "survival_frames": survival}
        for step, reward, survival in rows[::stride]
    ]
    return {"points": points, "train_max": train_max, "rows": len(rows)}


def run_context(history_root: Path | str, run_id: str) -> dict[str, Any]:
    """Bundle one run's plotted context: training curve +     final eval table.

    Used by the ``/api/run/<run_id>/curves`` route and the run-comparison
    CLI output. Raises ``ValueError`` for unsafe ids and ``FileNotFoundError``
    when the run has no replayable evaluation.
    """

    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"unknown run: {run_id!r}")
    run_dir = _variant_run_dir(history_root, run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no such run: {run_id}")
    curves = run_training_curves(run_dir)
    config = _read_json(run_dir / "config.json") or {}
    report = _read_json(run_dir / "report.json") or {}
    evaluation = _read_json(run_dir / "evaluation.json")
    if evaluation is None:
        report_eval = report.get("evaluation")
        evaluation = dict(report_eval) if isinstance(report_eval, Mapping) else None
    if evaluation is None or not list_eval_episodes(evaluation):
        raise FileNotFoundError(f"no evaluation for run {run_id}")
    picks = select_comparison_episodes(evaluation)
    eval_table = {}
    for source in ("inner", "holdout"):
        section = evaluation.get(source)
        if isinstance(section, Mapping):
            eval_table[source] = {
                "seeds": list(section.get("seeds", [])),
                "rewards": list(section.get("rewards", [])),
                "survival_frames": list(section.get("survival_frames", [])),
                "mean_reward": section.get("mean_reward"),
            }
    run_cfg = config.get("run", {})
    if not isinstance(run_cfg, Mapping):
        run_cfg = {}
    return {
        "run_id": run_id,
        "steps": run_cfg.get("steps"),
        "seed": run_cfg.get("seed"),
        "quality_gate": report.get("quality_gate"),
        "train_curve": curves["points"],
        "train_rows": curves["rows"],
        "train_max": curves["train_max"],
        "eval": eval_table,
        "best_eval_episode": best_greedy_episode(evaluation),
        "picks": {
            key: {
                "seed": episode.seed,
                "source": episode.source,
                "eval_reward": episode.eval_reward,
                "survival_frames": episode.survival_frames,
            }
            for key, episode in picks.items()
        },
    }


def generate_run_comparison(
    history_root: Path | str,
    run_id: str,
    *,
    episodes: Sequence[str] = EPISODE_KEYS,
    steps: int | None = None,
    device: str = "auto",
    epsilon: float = 0.0,
) -> list[tuple[ComparisonEpisode, NativeReplay]]:
    """Re-simulate the run's own best/median/worst eval seeds to full episodes.

    ``steps`` bounds each replay; ``None`` replays the full episode until
    death (capped at ``MAX_REPLAY_STEPS`` decisions). Game settings and the
    checkpoint come from the run, so the pixels match what that policy saw.
    """

    wanted = [key for key in episodes if key in EPISODE_KEYS]
    if not wanted:
        raise ValueError(f"episodes must be a subset of {EPISODE_KEYS}")
    layout = resolve_run_replay_config(history_root, run_id)
    picks = select_comparison_episodes(layout.evaluation)
    bound = MAX_REPLAY_STEPS if steps is None else int(steps)
    if not 1 <= bound <= MAX_REPLAY_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_REPLAY_STEPS}")
    results: list[tuple[ComparisonEpisode, NativeReplay]] = []
    for key in wanted:
        episode = picks[key]
        replay = generate_native_replay(
            layout.checkpoint,
            seed=episode.seed,
            steps=bound,
            device=device,
            step_frames=layout.step_frames,
            difficulty=layout.difficulty,
            patterns=layout.patterns,
            powerups=layout.powerups,
            epsilon=float(epsilon),
            dueling=layout.dueling,
        )
        results.append(
            (
                episode,
                NativeReplay(
                    checkpoint=replay.checkpoint,
                    seed=replay.seed,
                    step_frames=replay.step_frames,
                    created_at=replay.created_at,
                    frames=replay.frames,
                    run_id=layout.run_id,
                    label=key,
                    source=episode.source,
                    eval_reward=episode.eval_reward,
                ),
            )
        )
    return results


__all__ = [
    "EPISODE_KEYS",
    "MAX_CURVE_BYTES",
    "MAX_CURVE_POINTS",
    "ComparisonEpisode",
    "RunReplayConfig",
    "generate_run_comparison",
    "list_eval_episodes",
    "resolve_run_replay_config",
    "run_context",
    "run_training_curves",
    "select_comparison_episodes",
]
