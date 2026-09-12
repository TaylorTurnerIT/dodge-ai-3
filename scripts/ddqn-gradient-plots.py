"""Render existing sampled training norms and bounded offline gradient probes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("history/dodge/gymnasium/cnn-image-ddqn")
    groups = {
        "RGB target 1k": "rgb700-ts1000-500k-s42-v1",
        "RGB target 10k": "rgb700-ts10000-500k-s42-v1",
        "Gray": "t4-gray-200k-v1",
        "Collision 3-step": "t4-nstep3-200k-v1",
    }
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    summaries = {}
    for label, run in groups.items():
        records = [
            json.loads(line)
            for line in (root / run / "metrics.jsonl").read_text().splitlines()
        ]
        records = [
            r
            for r in records
            if r.get("diagnostic_sampled") and r.get("pre_clip_grad_norm") is not None
        ]
        # A metrics row can repeat a sampled optimizer update; count each update once.
        unique = {r["diagnostic_optimizer_step"]: r for r in records}
        records = sorted(unique.values(), key=lambda r: r["step"])
        x = np.array([r["step"] for r in records]) / 1000
        norms = np.array([r["pre_clip_grad_norm"] for r in records])
        scales = np.minimum(1, 10 / (norms + 1e-6))
        spread = np.array(
            [
                np.std(norms[max(0, i - 19) : i + 1], ddof=1) if i else np.nan
                for i in range(len(norms))
            ]
        )
        axes[0].plot(x, norms, label=label, linewidth=1)
        axes[1].plot(x, scales, linewidth=1)
        axes[2].plot(x, spread, linewidth=1)
        summaries[run] = {
            "unique_sampled_updates": len(norms),
            "median_pre_clip_norm": float(np.median(norms)),
            "max_pre_clip_norm": float(norms.max()),
            "sampled_share_above_10": float((norms > 10).mean()),
            "median_clip_multiplier": float(np.median(scales)),
        }
    axes[0].set(ylabel="Pre-clipping L2 norm", yscale="log")
    axes[0].legend()
    axes[1].set(ylabel="Global clip multiplier", yscale="log")
    axes[2].set(
        ylabel="Rolling norm SD (20 samples)",
        yscale="log",
        xlabel="Training decisions (thousands)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Sampled training gradients — norm variation is NOT gradient-vector variance"
    )
    fig.savefig("analysis/DDQN_GRADIENT_TIMELINE.png", dpi=140)
    plt.close(fig)
    audit = json.loads(Path("analysis/DDQN_PIXEL_COLLAPSE_AUDIT.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    keys = [
        "features.0.weight",
        "features.2.weight",
        "features.4.weight",
        "shared.1.weight",
        "value_stream.weight",
        "advantage_stream.weight",
    ]
    for ax, run in zip(axes, list(groups.values())[:3], strict=True):
        rows = sorted(
            [r for r in audit["rows"] if r["run"] == run], key=lambda r: r["step"]
        )
        for key in keys:
            ax.plot(
                [r["step"] / 1000 for r in rows],
                [r["gradients"][key]["batch_gradient_std_rms"] for r in rows],
                marker="o",
                label=key.replace(".weight", ""),
            )
        ax.set(
            title=next(k for k, v in groups.items() if v == run),
            yscale="log",
            xlabel="Checkpoint (thousands of decisions)",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("RMS per-parameter gradient SD across 8 batches")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "Offline fixed-corpus gradient variability — not historical replay variance"
    )
    fig.savefig("analysis/DDQN_GRADIENT_VARIANCE.png", dpi=140)
    Path("analysis/DDQN_GRADIENT_SUMMARY.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
