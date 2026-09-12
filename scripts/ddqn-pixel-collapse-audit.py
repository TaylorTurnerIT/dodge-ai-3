"""Bounded offline layer audit on a shared native-pixel diagnostic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.model import AtariCnnQNetwork
from dodge_native_game.variants.cnn_image_ddqn.pixels import native_gray_from_rgb
from dodge_native_game.variants.cnn_image_ddqn.run import _configure_torch_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _configure_torch_backend("cpu")
    env = CNNImageDDQNEnv(stack_size=4, observation_profile="native-rgb-v1")
    observations = []
    successors, actions, rewards, terminals = [], [], [], []
    rng = np.random.default_rng(314159)
    try:
        for seed in (901, 902, 903, 904):
            obs, _ = env.reset(seed=seed)
            for _ in range(32):
                observations.append(obs.copy())
                action = int(rng.integers(9))
                obs, reward, terminated, truncated, _ = env.step(action)
                successors.append(obs.copy())
                actions.append(action)
                rewards.append(reward)
                terminals.append(terminated or truncated)
                if terminated or truncated:
                    obs, _ = env.reset(seed=seed)
    finally:
        env.close()
    rgb = np.stack(observations)
    next_rgb = np.stack(successors)
    gray = np.stack(
        [
            np.concatenate(
                [native_gray_from_rgb(frame) for frame in obs.reshape(4, 3, 128, 128)]
            )
            for obs in rgb
        ]
    )
    next_gray = np.stack(
        [
            np.concatenate(
                [native_gray_from_rgb(f) for f in obs.reshape(4, 3, 128, 128)]
            )
            for obs in next_rgb
        ]
    )
    root = Path("history/dodge/gymnasium/cnn-image-ddqn")
    rows = []
    for name in (
        "rgb700-ts1000-500k-s42-v1",
        "rgb700-ts10000-500k-s42-v1",
        "t4-gray-200k-v1",
    ):
        corpus = gray if name.startswith("t4-gray") else rgb
        next_corpus = next_gray if name.startswith("t4-gray") else next_rgb
        for path in sorted((root / name / "checkpoints").glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = AtariCnnQNetwork(9, input_channels=corpus.shape[1], input_size=128)
            model.load_state_dict(payload["online_network"])
            model.eval()
            activations = {key: [] for key in ("conv1", "conv2", "conv3", "shared")}
            q_values = []
            with torch.no_grad():
                for start in range(0, len(corpus), 8):
                    x = (
                        torch.as_tensor(corpus[start : start + 8], dtype=torch.float32)
                        / 255
                    )
                    for index, layer in enumerate(model.features):
                        x = layer(x)
                        if index in (1, 3, 5):
                            activations[f"conv{(index + 1) // 2}"].append(
                                (x == 0).float().mean(dim=(2, 3)).numpy()
                            )
                    h = model.shared(x)
                    activations["shared"].append((h == 0).float().numpy())
                    advantages = model.advantage_stream(h)
                    q_values.append(
                        (
                            model.value_stream(h)
                            + advantages
                            - advantages.mean(dim=1, keepdim=True)
                        ).numpy()
                    )
            stats = {}
            for key, batches in activations.items():
                zeros = np.concatenate(batches)
                stats[key] = {
                    "zero_fraction": float(zeros.mean()),
                    "silent_channels_on_corpus": int(np.all(zeros == 1, axis=0).sum()),
                    "channels": zeros.shape[1],
                }
            q = np.concatenate(q_values)
            counts = np.bincount(q.argmax(axis=1), minlength=9)
            target = AtariCnnQNetwork(9, input_channels=corpus.shape[1], input_size=128)
            target.load_state_dict(payload["target_network"])
            target.eval()
            # Fixed disjoint batches: variance of batch-mean gradients, not per-example
            # variance or the unknown historical replay distribution. No optimizer step.
            gradient_batches = {}
            losses, norms = [], []
            order = np.random.default_rng(271828).permutation(len(corpus))
            for indices in order.reshape(-1, 16):
                model.zero_grad(set_to_none=True)
                x = torch.as_tensor(corpus[indices])
                nx = torch.as_tensor(next_corpus[indices])
                with torch.no_grad():
                    chosen = model(nx).argmax(1, keepdim=True)
                    bootstrap = target(nx).gather(1, chosen).squeeze(1)
                    y = torch.tensor(np.asarray(rewards)[indices], dtype=torch.float32)
                    y += (
                        0.99
                        * (~torch.tensor(np.asarray(terminals)[indices]))
                        * bootstrap
                    )
                prediction = (
                    model(x)
                    .gather(1, torch.tensor(np.asarray(actions)[indices])[:, None])
                    .squeeze(1)
                )
                loss = torch.nn.functional.smooth_l1_loss(prediction, y)
                loss.backward()
                losses.append(float(loss.detach()))
                total = 0.0
                for key, parameter in model.named_parameters():
                    gradient = parameter.grad.detach().flatten().clone()
                    gradient_batches.setdefault(key, []).append(gradient)
                    total += float(gradient.square().sum())
                norms.append(total**0.5)
            gradient_stats = {}
            for key, batches in gradient_batches.items():
                gradients = torch.stack(batches).double()
                gradient_stats[key] = {
                    "mean_gradient_rms": float(
                        gradients.mean(0).square().mean().sqrt()
                    ),
                    "batch_gradient_std_rms": float(
                        gradients.var(0, unbiased=True).mean().sqrt()
                    ),
                    "mean_batch_l2_norm": float(gradients.norm(dim=1).mean()),
                    "always_zero_gradient_fraction": float(
                        (gradients == 0).all(0).double().mean()
                    ),
                }
            rows.append(
                dict(
                    run=name,
                    step=payload["step"],
                    layers=stats,
                    action_counts=counts.tolist(),
                    mean_per_action_q_std_across_scenes=float(q.std(axis=0).mean()),
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    gradients=gradient_stats,
                    diagnostic_batch_losses=losses,
                    diagnostic_pre_clip_norms=norms,
                )
            )
    result = dict(
        protocol="offline-shared-native-corpus-v1",
        seeds=[901, 902, 903, 904],
        action_rng_seed=314159,
        decisions_per_seed=32,
        gradient_protocol={
            "batch_size": 16,
            "batches": 8,
            "permutation_seed": 271828,
            "gamma": 0.99,
            "loss": "SmoothL1 beta=1 mean",
            "returns": 1,
            "variance": "unbiased across eight fixed disjoint batch-mean gradients",
            "distribution": "random-action diagnostic transitions, not training replay",
            "optimizer_steps": 0,
        },
        rgb_corpus_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        limits=(
            "Diagnostic corpus only; zeros do not prove permanent dead units "
            "or training causality"
        ),
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["run"], row["step"], row["layers"], row["action_counts"], flush=True)


if __name__ == "__main__":
    main()
