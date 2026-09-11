"""Measure fixed-action and uniform-random controls on the DDQN eval seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"episodes_per_split": 128, "max_steps": 512, "policies": {}}
    env = CNNImageDDQNEnv(stack_size=4, step_frames=4)
    try:
        for policy in (*range(9), "uniform-random"):
            sections = {}
            for split, offset in (("inner", 10000), ("holdout", 20000)):
                rows = []
                for seed in range(offset + 42, offset + 42 + 128):
                    env.reset(seed=seed)
                    rng = np.random.default_rng(seed)
                    reward_sum = 0.0
                    frames = 0
                    terminated = False
                    for _ in range(512):
                        action = (
                            int(rng.integers(9)) if isinstance(policy, str) else policy
                        )
                        _, reward, terminated, truncated, info = env.step(action)
                        reward_sum += reward
                        frames += info["native_frames_advanced"]
                        if terminated or truncated:
                            break
                    rows.append(
                        {
                            "seed": seed,
                            "reward": reward_sum,
                            "survival_frames": frames,
                            "censored": not terminated,
                        }
                    )
                sections[split] = rows
            result["policies"][str(policy)] = sections
            print(
                policy,
                {
                    s: float(np.mean([r["reward"] for r in rows]))
                    for s, rows in sections.items()
                },
                flush=True,
            )
    finally:
        env.close()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
