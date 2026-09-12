"""Bounded pixel replay and optimizer microbenchmarks; never a training score."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent
from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.pixel_replay import (
    NativePixelReplayBuffer,
)
from dodge_native_game.variants.cnn_image_ddqn.pixels import RGB_PROFILE
from dodge_native_game.variants.cnn_image_ddqn.replay import ReplayBuffer
from dodge_native_game.variants.cnn_image_ddqn.run import _configure_torch_backend


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values) * 1000),
        "p95_ms": float(np.quantile(values, 0.95) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--decisions", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (
        32 <= args.capacity <= 2048
        and args.decisions >= args.capacity
        and 1 <= args.updates <= 1000
        and 1 <= args.repeats <= 10
    ):
        parser.error("capacity 32..2048, decisions>=capacity, bounded updates/repeats")
    _configure_torch_backend(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    shape = (12, 128, 128)
    compact = NativePixelReplayBuffer(capacity=args.capacity, stack_size=4, seed=42)
    dense = ReplayBuffer(capacity=args.capacity, observation_shape=shape, seed=42)
    env = CNNImageDDQNEnv(observation_profile=RGB_PROFILE)
    times: dict[str, list[float]] = {
        "compact_add": [],
        "dense_add": [],
        "compact_sample32": [],
        "dense_sample32": [],
        "packed_sample32": [],
    }
    seed = 42
    try:
        observation, _ = env.reset(seed=seed)
        previous = observation
        for step in range(args.decisions):
            action = step % 9
            observation, reward, done, _, _ = env.step(action)
            following = observation
            for name, replay in (("compact", compact), ("dense", dense)):
                started = time.perf_counter()
                replay.add(previous, action, reward, following, done)
                times[name + "_add"].append(time.perf_counter() - started)
            previous = following
            if done:
                seed += 1
                observation, _ = env.reset(seed=seed)
                previous = observation
        for _ in range(50):
            for name, replay in (("compact", compact), ("dense", dense)):
                started = time.perf_counter()
                replay.sample(32)
                times[name + "_sample32"].append(time.perf_counter() - started)
            started = time.perf_counter()
            compact.sample_packed(32)
            times["packed_sample32"].append(time.perf_counter() - started)
        # Isolated benchmark only: use identical transition indices for both
        # transfer formats, after timing their independent sampler paths.
        compact._rng = np.random.default_rng(4242)
        batch = compact.sample(32)
        compact._rng = np.random.default_rng(4242)
        packed_batch = compact.sample_packed(32)
        np.testing.assert_array_equal(batch.actions, packed_batch.actions)
    finally:
        env.close()

    def synchronize() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    # Alternate trial order, reset initialization/optimizer for each trial.
    # GPU synchronization occurs at timed-region boundaries, not each update.
    optimizer_times: dict[str, list[float]] = {
        "every_update": [],
        "sampled": [],
        "packed_sampled": [],
    }
    reference_weights = None
    for repeat in range(args.repeats):
        order = (
            ("every_update", "sampled", "packed_sampled")
            if repeat % 2 == 0
            else ("packed_sampled", "sampled", "every_update")
        )
        for mode in order:
            torch.manual_seed(42)
            agent = DoubleDQNAgent(9, observation_shape=shape, device=args.device)
            decoded = agent._packed_observation_tensor(packed_batch.observations)
            expected = (
                torch.as_tensor(batch.observations, device=args.device).float() / 255
            )
            torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
            del decoded, expected
            selected_batch = packed_batch if mode == "packed_sampled" else batch
            for _ in range(3):
                agent.update(selected_batch, diagnostics=False)
            synchronize()
            started = time.perf_counter()
            for index in range(args.updates):
                agent.update(
                    selected_batch, diagnostics=mode == "every_update" or index == 0
                )
            synchronize()
            optimizer_times[mode].append((time.perf_counter() - started) / args.updates)
            weights = {
                key: value.detach().cpu().clone()
                for key, value in agent.online_network.state_dict().items()
            }
            if reference_weights is None:
                reference_weights = weights
            else:
                for key in weights:
                    torch.testing.assert_close(
                        weights[key], reference_weights[key], rtol=0, atol=0
                    )
            del agent

    result = {
        "benchmark": "pixel-replay-and-optimizer-microbench-v2",
        "device": args.device,
        "hardware": torch.cuda.get_device_name(0)
        if args.device == "cuda"
        else (platform.processor() or platform.machine()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "capacity": args.capacity,
        "decisions": args.decisions,
        "batch_size": 32,
        "observation_shape": list(shape),
        "compact_allocated_bytes": compact.allocated_bytes,
        "packed_inputs_equal_dense": True,
        "weight_updates_equal_across_modes": True,
        "replay_cpu": {name: summary(values) for name, values in times.items()},
        "optimizer": {
            name: summary(values) for name, values in optimizer_times.items()
        },
        "optimizer_updates_per_trial": args.updates,
        "repeats": args.repeats,
        "limits": (
            "Fixed native-action corpus; replay CPU only; optimizer includes "
            "batch transfer but excludes environment, replay sampling, file "
            "logging and evaluation. Not end-to-end training throughput."
        ),
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
