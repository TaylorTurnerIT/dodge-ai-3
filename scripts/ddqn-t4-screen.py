"""Independent 200k T4 screens from the completed 700-game-seed baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tarfile
from pathlib import Path


def treatment(name: str) -> dict:
    if name in ("boundary5", "boundary10"):
        result = treatment("nstep3s43boundary")
        result["reward_profile"] = f"{name}-v1"
        return result
    sync_choices = {
        f"boundarysync{value}": value for value in (100, 1000, 2500, 5000, 10000)
    }
    if name in sync_choices:
        return dict(
            **treatment("nstep3s43boundary"),
            target_sync_interval=sync_choices[name],
        )
    if name == "nstep3s43boundary":
        return dict(
            seed=43,
            observation_profile="collision-image-v1",
            n_step=3,
            reward_profile="boundary-v1",
        )
    reward_names = {
        "rgbcontrol": "survival-v1",
        "rgbdeath": "death-v1",
        "rgbevents": "events-v1",
        "rgbboundary": "boundary-v1",
    }
    if name in reward_names:
        return dict(
            seed=42,
            observation_profile="native-rgb-v1",
            n_step=1,
            reward_profile=reward_names[name],
        )
    choices = {
        "rep42": (42, "collision-image-v1", 1),
        "rep43": (43, "collision-image-v1", 1),
        "rep44": (44, "collision-image-v1", 1),
        "nstep3": (42, "collision-image-v1", 3),
        "nstep3s43": (43, "collision-image-v1", 3),
        "nstep3s44": (44, "collision-image-v1", 3),
        "gray": (42, "native-gray-v1", 1),
    }
    seed, profile, horizon = choices[name]
    return dict(seed=seed, observation_profile=profile, n_step=horizon)


def worker(name: str, history: Path, smoke: bool) -> None:
    import torch

    from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
    from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
    from dodge_native_game.variants.cnn_image_ddqn.pixels import observation_shape
    from dodge_native_game.variants.cnn_image_ddqn.run import (
        _evaluate,
        _load_checkpoint,
        train_run,
    )

    torch.set_num_threads(1)
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"T4 required, got {gpu}")
    config = treatment(name)
    sync_interval = config.pop("target_sync_interval", 10000)
    horizon = config.pop("n_step")
    # Do not require new return support for unchanged baseline replicas.
    if horizon != 1:
        config["n_step"] = horizon
    run_id = f"t4-{name}-{'smoke' if smoke else '200k'}-v1"
    root = train_run(
        history_root=history,
        run_id=run_id,
        steps=64 if smoke else 200000,
        training_seed_count=700,
        evaluation_seed=512,
        stack_size=4,
        replay_capacity=128 if smoke else 100000,
        batch_size=4 if smoke else 32,
        learning_rate=1e-4,
        warmup_steps=8 if smoke else 20000,
        update_every=4,
        target_sync_interval=2 if smoke else sync_interval,
        epsilon_decay_steps=500000,
        device="cuda",
        shaping=False,
        dueling=True,
        log_interval=32 if smoke else 1000,
        eval_episodes=1 if smoke else 128,
        eval_steps=4 if smoke else 4096,
        checkpoint_steps=() if smoke else (10000,),
        **config,
    )
    if smoke:
        print(f"SMOKE_PASS {run_id}", flush=True)
        return
    profile = config["observation_profile"]
    agent = DoubleDQNAgent(
        9, device="cuda", observation_shape=observation_shape(profile, 4)
    )
    env = CNNImageDDQNEnv(stack_size=4, observation_profile=profile)
    snapshots = {}
    try:
        _load_checkpoint(
            agent,
            root / "checkpoints/step-10000.pt",
            observation_profile=profile,
            stack_size=4,
        )
        snapshots["10000"] = {
            split: _evaluate(
                agent, env, seed=512, episodes=128, max_steps=4096, seed_offset=offset
            )
            for split, offset in (("inner", 10000), ("holdout", 20000))
        }
    finally:
        env.close()
    (root / "checkpoint_evaluations.json").write_text(
        json.dumps(snapshots, indent=2) + "\n"
    )
    reference = history / "baseline_reference.json"
    reward_gate = history / "reward_telemetry_gate.json"
    if reward_gate.exists():
        (root / "reward_telemetry_gate.json").write_bytes(reward_gate.read_bytes())
    if reference.exists():
        (root / "baseline_reference.json").write_bytes(reference.read_bytes())
    smoke_root = history / "cnn-image-ddqn" / f"t4-{name}-smoke-v1"
    if smoke_root.exists():
        (root / "smoke_evidence.json").write_text(
            json.dumps(
                {
                    key: json.loads((smoke_root / f"{key}.json").read_text())
                    for key in ("status", "report", "config")
                },
                indent=2,
            )
            + "\n"
        )
    (root / "execution_provenance.json").write_text(
        json.dumps(
            {
                "source_commit": os.environ.get("DDQN_SOURCE_COMMIT"),
                "source_archive_sha256": os.environ.get("DDQN_SOURCE_SHA256"),
                "gpu": gpu,
                "torch": torch.__version__,
                "treatment": treatment(name),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "horizons": [10000, 200000],
                "training_seed_count": 700,
                "checkpoint_10000_trained": False,
                "protocol": "fresh-200k-baseline-500k-epsilon-20k-warmup",
            },
            indent=2,
        )
        + "\n"
    )
    temporary = history / f".{run_id}.tar.gz"
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(root, arcname=run_id)
    temporary.rename(history / f"{run_id}.tar.gz")
    print(f"COMPLETE {run_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("/content/t4-history"))
    parser.add_argument(
        "--treatments",
        nargs="+",
        required=True,
        choices=(
            "rep42",
            "rep43",
            "rep44",
            "nstep3",
            "nstep3s43",
            "nstep3s43boundary",
            "boundary5",
            "boundary10",
            "boundarysync100",
            "boundarysync1000",
            "boundarysync2500",
            "boundarysync5000",
            "boundarysync10000",
            "nstep3s44",
            "gray",
            "rgbcontrol",
            "rgbdeath",
            "rgbevents",
            "rgbboundary",
        ),
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline-checkpoint", type=Path)
    args = parser.parse_args()
    args.history.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(args.treatments) != 1:
            parser.error("worker needs exactly one treatment")
        worker(args.treatments[0], args.history, args.smoke)
        return
    if args.baseline_checkpoint is not None:
        evaluate_baseline(args.baseline_checkpoint, args.history)
    for name in args.treatments:
        command = [
            sys.executable,
            __file__,
            "--history",
            str(args.history),
            "--treatments",
            name,
            "--worker",
        ]
        subprocess.run(command + ["--smoke"], check=True)
        subprocess.run(command, check=True)
    print("CAMPAIGN_COMPLETE", flush=True)


def evaluate_baseline(checkpoint: Path, history: Path) -> None:
    """Evaluate the frozen 200k reference outside all training processes."""
    import torch

    from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
    from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
    from dodge_native_game.variants.cnn_image_ddqn.run import (
        _evaluate,
        _load_checkpoint,
    )

    torch.set_num_threads(1)
    agent = DoubleDQNAgent(9, device="cuda", seed=42)
    payload = _load_checkpoint(
        agent, checkpoint, observation_profile="collision-image-v1", stack_size=4
    )
    if payload.get("step") != 200000 or payload.get("seed") != 42:
        raise ValueError("reference must be the learner-42 200k checkpoint")
    env = CNNImageDDQNEnv(stack_size=4, observation_profile="collision-image-v1")
    try:
        results = {
            split: _evaluate(
                agent, env, seed=512, episodes=128, max_steps=4096, seed_offset=offset
            )
            for split, offset in (("inner", 10000), ("holdout", 20000))
        }
    finally:
        env.close()
    results["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    results["checkpoint_run"] = "seedpool-700-500k-s42-v1"
    (history / "baseline_reference.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print("BASELINE_REFERENCE_EVALUATED", flush=True)


if __name__ == "__main__":
    main()
