#!/usr/bin/env python3
"""Matched bounded DDQN throughput benchmark; never a learning-quality score."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from dodge_native_game.variants.cnn_image_ddqn.agent import LEARNER_BACKENDS
from dodge_native_game.variants.cnn_image_ddqn.pixels import OBSERVATION_PROFILES
from dodge_native_game.variants.cnn_image_ddqn.run import train_run


def _last_metrics(root: Path) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return rows[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--observation-profile", choices=OBSERVATION_PROFILES, default="native-rgb-v1"
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--vector-executions",
        nargs="+",
        choices=("serial", "parallel"),
        default=["serial", "parallel"],
    )
    parser.add_argument(
        "--learner-backends", nargs="+", choices=LEARNER_BACKENDS, default=None
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 32 or args.warmup_steps < 1 or not 1 <= args.repeats <= 10:
        parser.error("steps>=32, warmup>=1, repeats 1..10")
    if any(lane < 1 or lane > 64 for lane in args.lanes):
        parser.error("lanes must be in 1..64")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    backends = args.learner_backends or (
        ["baseline"]
        if args.device == "cpu"
        else ["baseline", "cuda-amp", "cuda-optimized"]
    )
    if args.device == "cpu" and any(mode != "baseline" for mode in backends):
        parser.error("CPU benchmarks support only the baseline learner")

    cases = [
        (lanes, execution, backend)
        for backend in backends
        for lanes in args.lanes
        for execution in (("serial",) if lanes == 1 else args.vector_executions)
    ]
    results: dict[str, list[dict[str, float | int | str]]] = {}
    with tempfile.TemporaryDirectory(prefix="ddqn-training-bench-") as directory:
        history = Path(directory)
        for repeat in range(args.repeats):
            ordered = cases if repeat % 2 == 0 else list(reversed(cases))
            for lanes, execution, backend in ordered:
                key = f"lanes{lanes}-{execution}-{backend}"
                run_id = f"bench-r{repeat}-{key}"
                started = time.perf_counter()
                root = train_run(
                    history_root=history,
                    run_id=run_id,
                    steps=args.steps,
                    seed=42,
                    stack_size=1,
                    observation_profile=args.observation_profile,
                    batch_size=32,
                    replay_capacity=max(args.steps, 64),
                    warmup_steps=args.warmup_steps,
                    update_every=4,
                    target_sync_interval=100,
                    log_interval=args.steps,
                    eval_episodes=1,
                    eval_steps=1,
                    eval_batch_size=1,
                    collector_lanes=lanes,
                    collector_execution=execution,
                    learner_backend=backend,
                    epsilon_decay_steps=args.steps,
                    device=args.device,
                )
                all_in_seconds = time.perf_counter() - started
                metrics = _last_metrics(root)
                throughput = float(metrics["throughput"])
                optimizer_steps = int(metrics["segment_optimizer_step"])
                results.setdefault(key, []).append(
                    {
                        "repeat": repeat,
                        "transitions": args.steps,
                        "optimizer_updates": optimizer_steps,
                        "training_seconds": args.steps / throughput,
                        "all_in_seconds": all_in_seconds,
                        "transitions_per_second": throughput,
                        "updates_per_second": optimizer_steps
                        / max(args.steps / throughput, 1e-9),
                    }
                )

    summary = {}
    for key, rows in results.items():
        rates = [float(row["transitions_per_second"]) for row in rows]
        all_in = [float(row["all_in_seconds"]) for row in rows]
        summary[key] = {
            "mean_transitions_per_second": float(np.mean(rates)),
            "p05_transitions_per_second": float(np.quantile(rates, 0.05)),
            "mean_all_in_seconds": float(np.mean(all_in)),
        }
    payload = {
        "benchmark": "ddqn-bounded-training-throughput-v1",
        "device": args.device,
        "hardware": (
            torch.cuda.get_device_name(0)
            if args.device == "cuda"
            else (platform.processor() or platform.machine())
        ),
        "torch": torch.__version__,
        "observation_profile": args.observation_profile,
        "transitions_per_trial": args.steps,
        "warmup_transitions": args.warmup_steps,
        "update_every_transitions": 4,
        "repeats": args.repeats,
        "summary": summary,
        "trials": results,
        "limits": (
            "Training throughput uses total native transitions as denominator. "
            "All-in time also includes setup, checkpoint, one-step inner/holdout "
            "evaluation, and counterfactual evaluation. This bounded benchmark "
            "does not measure policy quality or generalization."
        ),
    }
    output = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
