"""Offline frozen-policy terminal diagnostics; never updates a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .agent import DoubleDQNAgent
from .env import CNNImageDDQNEnv
from .reward_profiles import weighted_components
from .run import _load_checkpoint


def summarize_episode(rows: list[dict], gamma: float, horizon: int) -> list[dict]:
    """Exclude cap-crossing targets; terminal rows never bootstrap past death."""
    result = []
    ended = bool(rows and rows[-1]["terminated"])
    future_return = 0.0
    returns = [None] * len(rows)
    if ended:
        for i in range(len(rows) - 1, -1, -1):
            future_return = rows[i]["reward"] + gamma * future_return
            returns[i] = future_return
    for i, row in enumerate(rows):
        end = min(i + horizon, len(rows))
        terminal = any(r["terminated"] for r in rows[i:end])
        if end - i < horizon and not terminal:
            continue
        target = sum(gamma**j * r["reward"] for j, r in enumerate(rows[i:end]))
        if not terminal:
            target += gamma ** (end - i) * rows[end - 1]["next_value"]
        error = row["q"] - target
        result.append(
            {
                "group": "terminal_window"
                if terminal
                else ("near_death" if ended and len(rows) - i <= 12 else "ordinary"),
                "td_error": error,
                "abs_td_error": abs(error),
                "smooth_l1_saturated": float(abs(error) >= 1),
                "return_error": row["q"] - returns[i] if ended else None,
                "boundary_cost": row["boundary_cost"],
                "repeated": float(row["repeat"]),
                "q_gap": row["q_gap"],
                "action": row["action"],
                "greedy_action": row["greedy_action"],
            }
        )
    return result


def probe(run_dir: Path, episodes: int = 16, cap: int = 512) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    report = json.loads((run_dir / "report.json").read_text())
    checkpoint = (run_dir / report["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run_dir.resolve()):
        raise ValueError("checkpoint must be inside run")
    profile = config["observation"]["profile"]
    stack = config["observation"]["stack_size"]
    model_config = config["model"]
    agent = DoubleDQNAgent(
        9, device="cpu", observation_shape=tuple(config["observation"]["shape"])
    )
    _load_checkpoint(agent, checkpoint, observation_profile=profile, stack_size=stack)
    agent.online_network.eval()
    reward_profile = config.get("native_reward_contract", {}).get(
        "profile", "survival-v1"
    )
    env = CNNImageDDQNEnv(
        stack_size=stack,
        observation_profile=profile,
        step_frames=config["game"]["step_frames"],
    )
    result = {
        "run_id": run_dir.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "cap_decisions": cap,
        "episodes_per_mode": episodes,
        "modes": {},
        "limits": (
            "Frozen-checkpoint inner-seed probes, not training replay samples. "
            "Near death means <=12 decisions remaining. "
            "Return errors exclude capped episodes. No causal claim."
        ),
    }
    try:
        for epsilon in (0.0, 0.64):
            episode_rows, sample_rows = [], []
            greedy_counts = np.zeros(9, dtype=int)
            actual_counts = np.zeros(9, dtype=int)
            for seed in range(
                10000 + config["run"]["evaluation_seed"],
                10000 + config["run"]["evaluation_seed"] + episodes,
            ):
                obs, _ = env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                rows, frames = [], 0
                previous, streak, longest = None, 0, 0
                for _ in range(cap):
                    with torch.inference_mode():
                        qs = agent.online_network(torch.as_tensor(obs).unsqueeze(0))[0]
                    greedy = int(qs.argmax())
                    action = int(rng.integers(9)) if rng.random() < epsilon else greedy
                    greedy_counts[greedy] += 1
                    actual_counts[action] += 1
                    following, reward, terminated, truncated, info = env.step(action)
                    with torch.inference_mode():
                        tensor = torch.as_tensor(following).unsqueeze(0)
                        next_action = int(agent.online_network(tensor)[0].argmax())
                        next_value = float(agent.target_network(tensor)[0, next_action])
                    components = weighted_components(
                        info["native_reward_terms"], reward_profile
                    )
                    streak = streak + 1 if previous == action else 1
                    longest = max(longest, streak)
                    sorted_q = torch.sort(qs, descending=True).values
                    rows.append(
                        {
                            "q": float(qs[action]),
                            "reward": float(components.sum()),
                            "terminated": bool(terminated),
                            "next_value": next_value,
                            "repeat": previous == action,
                            "q_gap": float(sorted_q[0] - sorted_q[1]),
                            "action": action,
                            "greedy_action": greedy,
                            "boundary_cost": -float(components[4:].sum()),
                        }
                    )
                    previous = action
                    frames += int(info["native_frames_advanced"])
                    obs = following
                    if terminated or truncated:
                        break
                sample_rows.extend(
                    summarize_episode(
                        rows, model_config["gamma"], model_config.get("n_step", 1)
                    )
                )
                episode_rows.append(
                    {
                        "seed": seed,
                        "survival_frames": frames,
                        "terminated": bool(terminated),
                        "max_action_streak_decisions": longest,
                        "tail_actions": [row["action"] for row in rows[-24:]],
                        "tail_greedy_actions": [
                            row["greedy_action"] for row in rows[-24:]
                        ],
                        "tail_q_gaps": [row["q_gap"] for row in rows[-24:]],
                    }
                )
            groups = {}
            for group in ("ordinary", "near_death", "terminal_window"):
                items = [r for r in sample_rows if r["group"] == group]
                groups[group] = {"samples": len(items)}
                for key in (
                    "td_error",
                    "abs_td_error",
                    "smooth_l1_saturated",
                    "return_error",
                    "boundary_cost",
                    "repeated",
                    "q_gap",
                ):
                    values = [r[key] for r in items if r[key] is not None]
                    groups[group][key + "_mean"] = (
                        float(np.mean(values)) if values else None
                    )
            result["modes"][str(epsilon)] = {
                "episodes": episode_rows,
                "groups": groups,
                "mean_frames": float(
                    np.mean([r["survival_frames"] for r in episode_rows])
                ),
                "censored_episodes": sum(not r["terminated"] for r in episode_rows),
                "greedy_action_counts": greedy_counts.tolist(),
                "actual_action_counts": actual_counts.tolist(),
            }
    finally:
        env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 128:
        parser.error("episodes must be between 1 and 128")
    torch.set_num_threads(1)
    torch.backends.nnpack.set_flags(False)
    result = probe(args.run_dir, args.episodes)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
