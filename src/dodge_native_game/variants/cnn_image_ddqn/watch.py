"""Watch a saved native collision-image DDQN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame
import torch

from .agent import DoubleDQNAgent
from .env import CNNImageDDQNEnv
from .run import _choose_device, _configure_torch_backend, _load_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scale", type=int, default=8)
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    shape = tuple(int(value) for value in payload["observation_shape"])
    device = _choose_device(args.device)
    _configure_torch_backend(device)
    agent = DoubleDQNAgent(
        num_actions=int(payload["num_actions"]),
        device=device,
        observation_shape=shape,
    )
    _load_checkpoint(agent, args.checkpoint)
    env = CNNImageDDQNEnv(stack_size=shape[0])

    pygame.init()
    scale = max(2, int(args.scale))
    image_size = 84 * scale
    surface = pygame.display.set_mode((image_size, image_size + 70))
    pygame.display.set_caption(f"Dodge collision view - {args.checkpoint.stem}")
    font = pygame.font.SysFont("segoeui", 16)
    clock = pygame.time.Clock()
    observation, _ = env.reset(seed=args.seed)
    running = True
    action = 0
    step = 0
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
            if not running:
                break

            action = int(agent.select_action(observation, 0.0))
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated:
                observation, _ = env.reset(seed=min(32_767, args.seed + step))

            image = np.clip(observation[-1] * 255.0, 0, 255).astype(np.uint8)
            rgb = np.repeat(image[..., None], 3, axis=2)
            rendered = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            rendered = pygame.transform.scale(rendered, (image_size, image_size))
            surface.fill((18, 18, 18))
            surface.blit(rendered, (0, 0))
            label = font.render(
                f"step {step:,}  action {action}  reward {reward:.1f}  "
                f"native frame {info['native_frame']:,}",
                True,
                (235, 235, 230),
            )
            surface.blit(label, (12, image_size + 14))
            pygame.display.flip()
            clock.tick(30)
    finally:
        env.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
