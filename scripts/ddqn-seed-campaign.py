"""Run the controlled 700/5000 game-seed comparison on an assigned Colab VM.

Run under the shipped source PYTHONPATH. Each learner is a fresh process;
intermediate checkpoints are evaluated only after training to preserve RNG flow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def worker(pool: int, history: Path) -> None:
    import torch

    from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
    from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
    from dodge_native_game.variants.cnn_image_ddqn.run import (
        _evaluate,
        _load_checkpoint,
        train_run,
    )

    run_id = f"seedpool-{pool}-500k-s42-v1"
    root = train_run(
        history_root=history,
        run_id=run_id,
        steps=500000,
        seed=42,
        training_seed_count=pool,
        evaluation_seed=42,
        stack_size=4,
        batch_size=32,
        learning_rate=1e-4,
        warmup_steps=20000,
        update_every=4,
        target_sync_interval=10000,
        epsilon_decay_steps=500000,
        eval_episodes=128,
        eval_steps=512,
        log_interval=1000,
        checkpoint_steps=(10000, 200000),
        device="cuda",
    )
    snapshots = {}
    agent = DoubleDQNAgent(num_actions=9, device="cuda", seed=42)
    env = CNNImageDDQNEnv(stack_size=4, step_frames=4)
    try:
        for steps in (10000, 200000):
            _load_checkpoint(agent, root / "checkpoints" / f"step-{steps}.pt")
            snapshots[str(steps)] = {
                split: _evaluate(
                    agent,
                    env,
                    seed=42,
                    episodes=128,
                    max_steps=512,
                    seed_offset=offset,
                )
                for split, offset in (("inner", 10000), ("holdout", 20000))
            }
    finally:
        env.close()
    (root / "checkpoint_evaluations.json").write_text(
        json.dumps(snapshots, indent=2) + "\n"
    )
    (root / "execution_provenance.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "source_commit": os.environ.get("DDQN_SOURCE_COMMIT"),
                "source_archive_sha256": os.environ.get("DDQN_SOURCE_SHA256"),
                "horizons": [10000, 200000, 500000],
                "horizon_protocol": "uninterrupted-500k-trajectory",
                "training_seed_count": pool,
            },
            indent=2,
        )
        + "\n"
    )
    with tarfile.open(history / f"{run_id}.tar.gz", "w:gz") as archive:
        archive.add(root, arcname=run_id)
    print(f"COMPLETE {run_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, choices=(700, 5000))
    parser.add_argument("--history", type=Path, default=Path("/content/seed-history"))
    args = parser.parse_args()
    args.history.mkdir(parents=True, exist_ok=True)
    if args.pool is not None:
        worker(args.pool, args.history)
        return
    for pool in (700, 5000):
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--pool",
                str(pool),
                "--history",
                str(args.history),
            ],
            check=True,
        )
    print("CAMPAIGN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
