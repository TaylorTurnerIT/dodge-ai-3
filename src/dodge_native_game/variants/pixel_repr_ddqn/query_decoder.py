"""A frozen-latent query decoder for pixel representation diagnostics.

The decoder is fitted after the representation model has been frozen.  It
consumes one projected latent per frame and has no access to pixels, actions,
or any game state.  Learned queries provide one output token per image patch;
the resulting patch predictions are explicitly rearranged into an RGB image.
"""

from __future__ import annotations

import torch
from torch import nn

from .upstream import FeedForward

__all__ = ["QueryPixelDecoder"]


_OUTPUT_CHANNELS = 3
_MLP_HIDDEN_DIM = 512


def _positive_int(name: str, value: int) -> None:
    """Validate one of the decoder's integer architecture dimensions."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class _QueryDecoderBlock(nn.Module):
    """Pre-normalized cross-attention followed by a residual source MLP."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        # FeedForward is the vendored upstream LeWM implementation.  Its own
        # LayerNorm is the pre-normalization for the MLP branch.
        self.mlp = FeedForward(hidden_dim, _MLP_HIDDEN_DIM, dropout=0.0)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.mlp(queries)


class QueryPixelDecoder(nn.Module):
    """Decode one latent vector into an RGB image using learned patch queries.

    The default dimensions are the locked diagnostic profile: a projected
    192-dimensional LeWM latent is decoded into a native 128x128 frame with
    16x16 patches.  Other dimensions are accepted for tiny CPU tests only;
    they do not define a second experiment profile.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 128,
        depth: int = 3,
        heads: int = 4,
        output_size: int = 128,
        patch_size: int = 16,
        output_channels: int = _OUTPUT_CHANNELS,
        raw_logits: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("hidden_dim", hidden_dim),
            ("depth", depth),
            ("heads", heads),
            ("output_size", output_size),
            ("patch_size", patch_size),
            ("output_channels", output_channels),
        ):
            _positive_int(name, value)
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if output_size % patch_size:
            raise ValueError("output_size must be divisible by patch_size")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.heads = heads
        self.output_size = output_size
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.raw_logits = raw_logits
        self.patches_per_side = output_size // patch_size
        self.num_patches = self.patches_per_side**2

        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.query_tokens = nn.Parameter(
            torch.empty(1, self.num_patches, hidden_dim)
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [_QueryDecoderBlock(hidden_dim, heads) for _ in range(depth)]
        )
        self.patch_head = nn.Linear(
            hidden_dim,
            output_channels * patch_size * patch_size,
        )

    @property
    def patch_queries(self) -> nn.Parameter:
        """Return the learnable one-token-per-patch query parameter."""

        return self.query_tokens

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Rearrange channels-last patch predictions into ``(B,C,H,W)``."""

        batch, num_patches, values = patches.shape
        expected_values = self.output_channels * self.patch_size**2
        if num_patches != self.num_patches or values != expected_values:
            raise ValueError(
                "patch predictions must have shape "
                f"(B, {self.num_patches}, {expected_values})"
            )
        patches = patches.reshape(
            batch,
            self.patches_per_side,
            self.patches_per_side,
            self.patch_size,
            self.patch_size,
            self.output_channels,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch,
            self.output_channels,
            self.output_size,
            self.output_size,
        )

    def forward_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """Return raw patch logits for latent input shaped ``(B, latent_dim)``."""

        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                "latent must have shape "
                f"(batch, {self.latent_dim}), got {tuple(latent.shape)}"
            )
        memory = self.latent_projection(latent).unsqueeze(1)
        queries = self.query_tokens.expand(latent.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        return self._unpatchify(self.patch_head(queries))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return sigmoid RGB by default, or raw logits when configured."""

        logits = self.forward_logits(latent)
        return logits if self.raw_logits else torch.sigmoid(logits)
