"""Bounded artifact discovery for the read-only explanation dashboard."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

METRIC_KEYS = (
    "step",
    "loss",
    "pre_clip_grad_norm",
    "td_error_std",
    "q_mean",
    "q_std",
    "epsilon",
    "dead_units",
    "action_balance",
    "throughput",
)


def metrics_for_run(root: Path, run_id: str) -> dict:
    """Read metrics without loading a model or requiring a supported replay ABI."""
    if not any(row["id"] == run_id for row in catalog(root)):
        raise ValueError("Unknown or excluded run")
    path = root / run_id
    if not path.is_dir():
        path = root.parent / "live-cnn-image-ddqn" / run_id
    rows = []
    truncated = False
    try:
        with (path / "metrics.jsonl").open("rb") as handle:
            size = handle.seek(0, 2)
            truncated = size > 8 * 1024 * 1024
            handle.seek(max(0, size - 8 * 1024 * 1024))
            if truncated:
                handle.readline()
            for line in handle:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    clean = {
                        key: value
                        for key in METRIC_KEYS
                        if isinstance(value := row.get(key), (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                    }
                    if "step" in clean:
                        episodes = row.get("completed_episodes", [])
                        lengths = [
                            e.get("survival_frames")
                            for e in episodes
                            if isinstance(e, dict)
                        ]
                        lengths = [
                            v
                            for v in lengths
                            if isinstance(v, (int, float)) and math.isfinite(v)
                        ]
                        if lengths:
                            clean["training_mean_frames"] = sum(lengths) / len(lengths)
                        rows.append(clean)
                except (ValueError, UnicodeError):
                    continue
    except OSError:
        pass
    # Bound browser payload; retain first/last point and disclose downsampling.
    count = len(rows)
    if count > 1000:
        rows = [rows[round(i * (count - 1) / 999)] for i in range(1000)]
    report = read_object(path / "report.json")
    evaluation = report.get("evaluation", {})
    holdout = evaluation.get("holdout", {})
    survival = holdout.get("survival_frames", [])
    survival = [v for v in survival if isinstance(v, (int, float)) and math.isfinite(v)]
    return {
        "run_id": run_id,
        "rows": rows,
        "source_rows": count,
        "tail_only": truncated,
        "holdout_survival": survival,
        "inner_mean_frames": evaluation.get("inner", {}).get("mean_survival_frames"),
        "gate": report.get("quality_gate"),
        "gate_reasons": report.get("gate_reasons", []),
    }


def read_object(path: Path) -> dict:
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def catalog(root: Path) -> list[dict]:
    """Canonical artifacts supersede mirrors; scores never use reward/maxima."""
    paths = {}
    for base in (root.parent / "live-cnn-image-ddqn", root):
        if base.is_dir():
            for path in base.iterdir():
                if path.is_dir() and not path.is_symlink():
                    paths[path.name] = path
    rows = []
    for name, path in paths.items():
        if re.search(r"smoke|preflight|(^|[-_])(test|gate|warmup)([-_]|$)", name, re.I):
            continue
        manifest = read_object(path / "manifest.json")
        config = read_object(path / "config.json")
        status = read_object(path / "status.json")
        report = read_object(path / "report.json")
        if not manifest or not config:
            continue
        budget = config.get("run", {}).get("steps", 0)
        if not isinstance(budget, (int, float)) or budget < 5000:
            continue
        holdout = report.get("evaluation", {}).get("holdout", {})
        mean = holdout.get("mean_survival_frames")
        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(mean)
            or mean < 0
        ):
            mean = None
        profile = manifest.get("engine", {}).get(
            "observation_profile", "collision-image-v1"
        )
        reason = None
        if not report:
            reason = (
                "Replay available after final evaluation and checkpoint collection."
            )
        elif profile != "collision-image-v1":
            reason = (
                "Metrics available. This explanation renderer does not yet "
                "support native pixel checkpoints."
            )
        elif config.get("reward_shaping", {}).get("enabled"):
            reason = "Metrics available. Legacy shaped-reward replay is unsupported."
        rows.append(
            {
                "id": name,
                "created_at": manifest.get("created_at", ""),
                "state": status.get("state", "completed" if report else "unknown"),
                "step": status.get(
                    "step", report.get("final_metrics", {}).get("step", 0)
                ),
                "budget": budget,
                "mean_frames": mean,
                "episodes": holdout.get("episodes"),
                "censored_share": holdout.get("censored_share"),
                "eval_cap": holdout.get("max_steps_per_episode"),
                "profile": profile,
                "reward_profile": config.get("native_reward_contract", {}).get(
                    "profile", "survival-v1"
                ),
                "replay_unavailable": reason,
                "best": False,
                "latest": False,
                "mirror": read_object(path / "live_mirror.json"),
            }
        )
    rows.sort(key=lambda row: (row["created_at"], row["id"]), reverse=True)
    if rows:
        rows[0]["latest"] = True
    eligible = [
        row
        for row in rows
        if row["mean_frames"] is not None and row["state"] == "completed"
    ]
    if eligible:
        max(
            eligible, key=lambda row: (row["mean_frames"], row["created_at"], row["id"])
        )["best"] = True
    return rows
