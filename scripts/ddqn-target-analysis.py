"""Plot measured target progression and clipping in the recovered A100 run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path("history/dodge/gymnasium/cnn-image-ddqn/exp500k-lr1e4-ts10k-s42-v2")
    rows = [
        json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
    ]
    periods = []
    for sync in range(13):
        selected = [
            r
            for r in rows
            if r["target_sync_count"] == sync and r["optimizer_step"] > 0
        ]
        periods.append(
            {
                "target_syncs": sync,
                "logged_samples": len(selected),
                "median_target": float(np.median([r["target_mean"] for r in selected])),
                "constant_reward_iterate": 4 * (1 - 0.99 ** (sync + 1)) / 0.01,
            }
        )
    windows = []
    for lower, upper in [
        (20000, 100000),
        (100000, 200000),
        (200000, 400000),
        (400000, 500000),
    ]:
        selected = [r for r in rows if lower < r["step"] <= upper]
        grads = np.array([r["pre_clip_grad_norm"] for r in selected])
        windows.append(
            {
                "from_step": lower,
                "to_step": upper,
                "logged_samples": len(selected),
                "clipped_share": float(np.mean(grads > 10)),
                "gradient_median": float(np.median(grads)),
            }
        )
    output = Path("analysis")
    (output / "DDQN_TARGET_ANALYSIS.json").write_text(
        json.dumps(
            {
                "run": root.name,
                "target_windows": periods,
                "gradient_windows": windows,
                "gradient_denominator": (
                    "logged update diagnostics, not every optimizer update"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(
        [p["target_syncs"] for p in periods],
        [p["median_target"] for p in periods],
        "o-",
        label="Measured median target",
    )
    axes[0].plot(
        [p["target_syncs"] for p in periods],
        [p["constant_reward_iterate"] for p in periods],
        "--",
        label="Constant reward Bellman iterate",
    )
    axes[0].set(
        xlabel="Completed target syncs",
        ylabel="Target value",
        title="Only 12 target refreshes in 500k steps",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(
        ["20–100k", "100–200k", "200–400k", "400–500k"],
        [100 * w["clipped_share"] for w in windows],
        color="#346997",
    )
    axes[1].set(
        xlabel="Environment-step window",
        ylabel="Clipped samples (%)",
        title="Clipping grows during training",
        ylim=(0, 100),
    )
    fig.savefig(output / "DDQN_TARGET_ANALYSIS.png", dpi=180)


if __name__ == "__main__":
    main()
