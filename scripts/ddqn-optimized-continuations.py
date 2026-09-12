"""Run and archive provenance-locked optimized DDQN experiments on a T4."""

from __future__ import annotations

import argparse
import json
import os
import resource
import tarfile
from pathlib import Path

TREATMENTS = {
    "fresh-best": {
        "run_id": "t4-best-opt8-200k-s43-v1",
        "steps": 200_000,
        "resume": None,
        "reward_profile": "survival-v1",
    },
    "best-resume": {
        "run_id": "t4-best-resume500k-opt8-v1",
        "steps": 300_000,
        "resume": "/content/checkpoints/best-step-200000.pt",
        "reward_profile": "survival-v1",
    },
    "boundary10-resume": {
        "run_id": "t4-boundary10-resume500k-opt8-v1",
        "steps": 300_000,
        "resume": "/content/checkpoints/boundary10-step-200000.pt",
        "reward_profile": "boundary10-v1",
    },
}


def run(treatment: str, history: Path, *, smoke: bool) -> Path:
    import torch

    from dodge_native_game.variants.cnn_image_ddqn.run import train_run

    spec = TREATMENTS[treatment]
    smoke_revision = os.environ.get("DDQN_SOURCE_COMMIT", "unknown")[:7]
    run_id = (
        f"{spec['run_id']}-smoke-{smoke_revision}" if smoke else str(spec["run_id"])
    )
    resume = spec["resume"]
    root = train_run(
        history_root=history,
        run_id=run_id,
        steps=64 if smoke else int(spec["steps"]),
        seed=43,
        stack_size=4,
        observation_profile="collision-image-v1",
        batch_size=4 if smoke else 32,
        replay_capacity=128 if smoke else 100_000,
        warmup_steps=8 if smoke else 20_000,
        update_every=4,
        target_sync_interval=2 if smoke else 10_000,
        log_interval=32 if smoke else 1_000,
        eval_episodes=1 if smoke else 128,
        eval_steps=4 if smoke else 4_096,
        eval_batch_size=1 if smoke else 16,
        collector_lanes=8,
        collector_execution="parallel",
        learner_backend="cuda-optimized",
        device="cuda",
        resume_from=resume,
        dueling=True,
        learning_rate=1e-4,
        epsilon_decay_steps=500_000,
        training_seed_count=700,
        evaluation_seed=512,
        checkpoint_steps=() if smoke else ((10_000,) if resume is None else (100_000,)),
        n_step=3,
        shaping=False,
        reward_profile=str(spec["reward_profile"]),
    )
    if smoke:
        print("SMOKE_PASS", treatment, root.name, flush=True)
        return root
    record = {
        "treatment": treatment,
        "source_commit": os.environ.get("DDQN_SOURCE_COMMIT"),
        "source_archive_sha256": os.environ.get("DDQN_SOURCE_SHA256"),
        "native_wheel_sha256": os.environ.get("DDQN_WHEEL_SHA256"),
        "reward_profiles_sha256": os.environ.get("DDQN_REWARD_PROFILES_SHA256"),
        "launcher_sha256": os.environ.get("DDQN_LAUNCHER_SHA256"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "collector": {"lanes": 8, "execution": "parallel"},
        "eval_batch_size": 16,
        "learner_backend": "cuda-optimized",
        "resume_contract": "optimizer-state-with-fresh-replay-rng-env"
        if resume
        else None,
    }
    (root / "execution_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    temporary = history / f".{root.name}.tar.gz"
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    temporary.rename(history / f"{root.name}.tar.gz")
    print("COMPLETE", treatment, root.name, flush=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("/content/opt-history"))
    parser.add_argument("--queue", nargs="+", choices=tuple(TREATMENTS), required=True)
    args = parser.parse_args()
    args.history.mkdir(parents=True, exist_ok=True)
    for treatment in args.queue:
        run(treatment, args.history, smoke=True)
        run(treatment, args.history, smoke=False)


if __name__ == "__main__":
    main()
