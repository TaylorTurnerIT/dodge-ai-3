"""The upstream LeWM SIGReg regularizer.

The implementation is intentionally kept numerically aligned with
``lucas-maes/le-wm`` commit ``8edfeb336732b5f3ce7b8b210d0ba370a09e2cac``.
In particular, its input is ``(time, batch, dimension)`` and the statistic is
averaged across batch at each time position before projection/time averaging.
"""

from __future__ import annotations

import torch

__all__ = ["SIGReg"]


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU reference)."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Return the Epps-Pulley statistic for ``proj`` shaped ``(T,B,D)``."""

        if proj.ndim != 3:
            raise ValueError(f"proj must have shape (T, B, D), got {tuple(proj.shape)}")
        # This draw and normalization intentionally match the upstream source.
        directions = torch.randn(
            proj.size(-1),
            self.num_proj,
            device=proj.device,
        )
        directions = directions.div_(directions.norm(p=2, dim=0))
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        err = (
            x_t.cos().mean(-3) - self.phi
        ).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
