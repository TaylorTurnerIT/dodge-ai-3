"""Bounded RGB target-cadence experiment, gated by paired GPU telemetry trials."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np


def parameters(sync: int) -> dict:
    return dict(
        seed=42,
        training_seed_count=700,
        evaluation_seed=512,
        stack_size=4,
        observation_profile="native-rgb-v1",
        replay_capacity=100000,
        batch_size=32,
        learning_rate=1e-4,
        warmup_steps=5000,
        update_every=4,
        target_sync_interval=sync,
        epsilon_decay_steps=500000,
        device="cuda",
        shaping=False,
        dueling=True,
    )


def worker(
    history: Path, run_id: str, sync: int, log_interval: int, steps: int, campaign: bool
) -> None:
    import torch

    from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
    from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
    from dodge_native_game.variants.cnn_image_ddqn.run import (
        _evaluate,
        _load_checkpoint,
        train_run,
    )

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    gpu = torch.cuda.get_device_name(0)
    if not any(name in gpu for name in ("A100", "H100")):
        raise RuntimeError(f"A100/H100 required, got {gpu}")
    started = time.perf_counter()
    root = train_run(
        history_root=history,
        run_id=run_id,
        steps=steps,
        log_interval=log_interval,
        eval_episodes=128 if campaign else 1,
        eval_steps=4096 if campaign else 4,
        checkpoint_steps=(10000, 200000) if campaign else (),
        **parameters(sync),
    )
    torch.cuda.synchronize()
    all_in = time.perf_counter() - started
    rows = [
        json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
    ]
    final = rows[-1]
    timing = dict(
        all_in_seconds=all_in,
        training_loop_seconds=steps / final["throughput"],
        optimizer_steps=final["optimizer_step"],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        gpu=gpu,
        torch=torch.__version__,
        source_commit=os.getenv("DDQN_SOURCE_COMMIT"),
        log_interval=log_interval,
        parameters=parameters(sync),
        steps=steps,
    )
    (root / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    if campaign:
        agent = DoubleDQNAgent(9, device="cuda", observation_shape=(12, 128, 128))
        env = CNNImageDDQNEnv(stack_size=4, observation_profile="native-rgb-v1")
        snapshots = {}
        try:
            for horizon in (10000, 200000):
                _load_checkpoint(agent, root / "checkpoints" / f"step-{horizon}.pt")
                snapshots[str(horizon)] = {
                    split: _evaluate(
                        agent,
                        env,
                        seed=512,
                        episodes=128,
                        max_steps=4096,
                        seed_offset=offset,
                    )
                    for split, offset in (("inner", 10000), ("holdout", 20000))
                }
        finally:
            env.close()
        (root / "checkpoint_evaluations.json").write_text(
            json.dumps(snapshots, indent=2) + "\n"
        )
        temporary = history / f".{run_id}.tar.gz"
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(root, arcname=run_id)
        temporary.rename(history / f"{run_id}.tar.gz")
    print(f"COMPLETE {run_id}", flush=True)


def gate_summary(pairs: list[dict]) -> dict:
    ratios = [
        p["routine"]["training_loop_seconds"] / p["minimal"]["training_loop_seconds"]
        - 1
        for p in pairs
    ]
    return dict(
        pairs=pairs,
        added_training_wall_time_fractions=ratios,
        mean_added_fraction=float(np.mean(ratios)),
        passed=len(pairs) >= 3 and all(np.isfinite(ratios)) and max(ratios) <= 0.05,
        criterion="all three alternating paired training-loop overheads <=5%",
        limits="Minimal logging retains counters, first/final logs and evaluation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("/content/rgb-history"))
    parser.add_argument("--worker")
    parser.add_argument("--sync", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--campaign", action="store_true")
    args = parser.parse_args()
    args.history.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(
            args.history,
            args.worker,
            args.sync,
            args.log_interval,
            args.steps,
            args.campaign,
        )
        return

    def launch(name: str, sync: int, log: int, campaign: bool = False) -> Path:
        command = [
            sys.executable,
            __file__,
            "--history",
            str(args.history),
            "--worker",
            name,
            "--sync",
            str(sync),
            "--log-interval",
            str(log),
            "--steps",
            "500000" if campaign else "10000",
        ]
        if campaign:
            command.append("--campaign")
        subprocess.run(command, check=True)
        return args.history / "cnn-image-ddqn" / name

    # Warm the GPU and filesystem; excluded from paired measurements.
    launch("rgb-gate-warmup-v1", 1000, 1000)
    pairs = []
    for repeat in range(3):
        pair = {}
        order = ("minimal", "routine") if repeat % 2 == 0 else ("routine", "minimal")
        for mode in order:
            root = launch(
                f"rgb-gate-{repeat}-{mode}-v1",
                1000,
                1000 if mode == "routine" else 1000000,
            )
            pair[mode] = json.loads((root / "timing.json").read_text())
        pairs.append(pair)
    gate = gate_summary(pairs)
    (args.history / "telemetry_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate), flush=True)
    if not gate["passed"]:
        raise RuntimeError("Telemetry gate failed; long runs NOT launched")
    for sync in (1000, 10000):
        launch(f"rgb700-ts{sync}-500k-s42-v1", sync, 1000, campaign=True)
    print("CAMPAIGN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
