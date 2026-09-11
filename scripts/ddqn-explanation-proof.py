"""Measure held-out replay parity and explanation identities on real checkpoints."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from dodge_native_game.variants.cnn_image_ddqn.explanation_server import service_for_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.nnpack.set_flags(False)
    started = time.perf_counter()
    service = service_for_run(args.run_dir)
    generation_seconds = time.perf_counter() - started
    rows = []
    for label, trace in service.traces.items():
        for index in sorted(
            {0, len(trace.observations) // 2, len(trace.observations) - 1}
        ):
            row = trace.metadata["frames"][index]
            for frame in (None, 0):
                result = service.explain(
                    label, index, frame=frame, alternative=None, channel=0
                )
                gap = row["q"][result["chosen"]] - row["q"][result["alternative"]]
                reconstruction = (
                    sum(result["contributions"]) + result["bias_difference"]
                )
                value_error = (
                    None
                    if row["value"] is None
                    else abs(row["value"] - sum(row["q"]) / len(row["q"]))
                )
                rows.append(
                    {
                        "episode": label,
                        "index": index,
                        "frame": frame,
                        "chosen": result["chosen"],
                        "alternative": result["alternative"],
                        "gap": gap,
                        "q": row["q"],
                        "value": row["value"],
                        "observed_reward": row["reward"],
                        "terminated_after_action": row["terminated"],
                        "decision_sensitivity_min": min(
                            map(min, result["decision_map"])
                        ),
                        "decision_sensitivity_max": max(
                            map(max, result["decision_map"])
                        ),
                        "conv0_ablation_gap_change": result["ablation"]["gap_delta"],
                        "contribution_residual": abs(gap - reconstruction),
                        "value_mean_q_residual": value_error,
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
                assert abs(gap - reconstruction) < 1e-4
                assert value_error is None or value_error < 1e-4
        assert trace.metadata["evaluation_reward_matches"], label
    result = {
        "run_dir": str(args.run_dir),
        "device": "cpu",
        "torch_threads": 1,
        "generation_seconds": generation_seconds,
        "generation_decisions": sum(
            len(t.observations) for t in service.traces.values()
        ),
        "episodes": [
            {
                k: v
                for k, v in trace.metadata.items()
                if k not in ("frames", "observations")
            }
            for trace in service.traces.values()
        ],
        "explanation_probes": rows,
        "scope": "replay parity and explanation identities; not learning improvement",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                *Path("src/dodge_native_game/variants/cnn_image_ddqn").glob("expla*"),
            ]
            if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"PASS {len(rows)} explanation probes; {args.output}")


if __name__ == "__main__":
    main()
