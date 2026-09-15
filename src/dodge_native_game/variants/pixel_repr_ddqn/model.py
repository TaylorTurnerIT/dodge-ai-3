"""Pixel/action LeWM world model for the independent Dodge variant.

The reference profile follows the LeWM v3 configuration: a randomly
initialized ViT-Tiny encoder, an action-conditioned causal predictor, two
BatchNorm projectors, and the prediction plus SIGReg objective.  The tiny
profile is an engineering smoke configuration and is not a scientific
comparison profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from .sigreg import SIGReg
from .upstream import MLP, ARPredictor

__all__ = [
    "IMAGE_MEAN",
    "IMAGE_STD",
    "INPUT_ENCODING_LEGACY",
    "INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC",
    "INPUT_ENCODING_RGB_NEAREST_SYMMETRIC",
    "INPUT_ENCODINGS",
    "LeWMConfig",
    "LeWorldModel",
]

IMAGE_MEAN: Final[tuple[float, float, float]] = (
    0.485,
    0.456,
    0.406,
)
IMAGE_STD: Final[tuple[float, float, float]] = (
    0.229,
    0.224,
    0.225,
)
INPUT_ENCODING_LEGACY: Final[str] = "legacy"
INPUT_ENCODING_RGB_NEAREST_SYMMETRIC: Final[str] = "rgb-nearest-symmetric"
INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC: Final[str] = (
    "palette-onehot-nearest-symmetric"
)
INPUT_ENCODINGS: Final[tuple[str, ...]] = (
    INPUT_ENCODING_LEGACY,
    INPUT_ENCODING_RGB_NEAREST_SYMMETRIC,
    INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC,
)
_PALETTE_INPUT_ENCODING = INPUT_ENCODING_PALETTE_ONEHOT_NEAREST_SYMMETRIC
_PALETTE_CHANNELS = 3


@dataclass(frozen=True)
class LeWMConfig:
    """Configuration for the reference or engineering-smoke LeWM profile."""

    profile: str = "reference"
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    encoder_mlp_dim: int = 768
    encoder_dropout: float = 0.0
    encoder_attention_dropout: float = 0.0
    history_size: int = 3
    action_dim: int = 9
    action_smoothing_dim: int = 9
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_embedding_dropout: float = 0.0
    projector_hidden_dim: int = 2048
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    input_encoding: str = INPUT_ENCODING_LEGACY
    palette_rgb: tuple[tuple[int, int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.input_encoding not in INPUT_ENCODINGS:
            raise ValueError(
                f"input_encoding must be one of {INPUT_ENCODINGS}, "
                f"got {self.input_encoding!r}"
            )
        if self.palette_rgb is None:
            canonical_palette = None
        else:
            if not isinstance(self.palette_rgb, (tuple, list)):
                raise TypeError("palette_rgb must be a sequence of RGB triples")
            canonical_colors: list[tuple[int, int, int]] = []
            for color in self.palette_rgb:
                if not isinstance(color, (tuple, list)) or len(color) != 3:
                    raise ValueError("palette_rgb must contain RGB triples")
                if any(
                    isinstance(channel, bool) or not isinstance(channel, Integral)
                    for channel in color
                ):
                    raise TypeError("palette_rgb channels must be integers")
                values = tuple(int(channel) for channel in color)
                if any(channel < 0 or channel > 255 for channel in values):
                    raise ValueError("palette_rgb channels must be in [0, 255]")
                canonical_colors.append(values)
            canonical_palette = tuple(canonical_colors)
            if tuple(sorted(canonical_palette)) != canonical_palette:
                raise ValueError("palette_rgb must be sorted lexicographically")
            if len(set(canonical_palette)) != len(canonical_palette):
                raise ValueError("palette_rgb must contain unique colors")
        if self.input_encoding == _PALETTE_INPUT_ENCODING:
            if canonical_palette is None:
                raise ValueError(
                    "palette_rgb is required for palette-onehot-nearest-symmetric"
                )
            if len(canonical_palette) != _PALETTE_CHANNELS:
                raise ValueError("palette-onehot-nearest-symmetric requires K=3")
        elif canonical_palette is not None:
            raise ValueError("palette_rgb is only valid for palette input encoding")
        object.__setattr__(self, "palette_rgb", canonical_palette)
        positive_fields = (
            "image_size",
            "patch_size",
            "embed_dim",
            "encoder_depth",
            "encoder_heads",
            "encoder_mlp_dim",
            "history_size",
            "action_dim",
            "action_smoothing_dim",
            "predictor_depth",
            "predictor_heads",
            "predictor_dim_head",
            "predictor_mlp_dim",
            "projector_hidden_dim",
            "sigreg_knots",
            "sigreg_num_proj",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.encoder_heads != 0:
            raise ValueError("embed_dim must be divisible by encoder_heads")
        if self.action_smoothing_dim < 1:
            raise ValueError("action_smoothing_dim must be positive")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be at least 2")
        for name in (
            "encoder_dropout",
            "encoder_attention_dropout",
            "predictor_dropout",
            "predictor_embedding_dropout",
        ):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.sigreg_weight < 0.0:
            raise ValueError("sigreg_weight must be non-negative")

    @classmethod
    def reference(cls, **overrides: object) -> LeWMConfig:
        """Return the paper/reference-shaped ViT-Tiny12 configuration."""

        values: dict[str, object] = {
            "profile": "reference",
            "image_size": 224,
            "patch_size": 14,
            "embed_dim": 192,
            "encoder_depth": 12,
            "encoder_heads": 3,
            "encoder_mlp_dim": 768,
            "history_size": 3,
            "action_dim": 9,
            "action_smoothing_dim": 9,
            "predictor_depth": 6,
            "predictor_heads": 16,
            "predictor_dim_head": 64,
            "predictor_mlp_dim": 2048,
            "predictor_dropout": 0.1,
            "projector_hidden_dim": 2048,
            "sigreg_weight": 0.09,
            "sigreg_knots": 17,
            "sigreg_num_proj": 1024,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def tiny(cls, **overrides: object) -> LeWMConfig:
        """Return a CPU-friendly engineering smoke profile."""

        values: dict[str, object] = {
            "profile": "tiny",
            "image_size": 64,
            "patch_size": 8,
            "embed_dim": 64,
            "encoder_depth": 2,
            "encoder_heads": 4,
            "encoder_mlp_dim": 256,
            "history_size": 3,
            "action_dim": 9,
            "action_smoothing_dim": 9,
            "predictor_depth": 2,
            "predictor_heads": 4,
            "predictor_dim_head": 16,
            "predictor_mlp_dim": 256,
            "predictor_dropout": 0.1,
            "projector_hidden_dim": 128,
            "sigreg_weight": 0.09,
            "sigreg_knots": 9,
            "sigreg_num_proj": 16,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_profile(cls, profile: str, **overrides: object) -> LeWMConfig:
        """Construct a named profile without accepting silent aliases."""

        if profile == "reference":
            return cls.reference(**overrides)
        if profile == "tiny":
            return cls.tiny(**overrides)
        raise ValueError(f"unknown LeWM profile: {profile!r}")

    @property
    def num_patches(self) -> int:
        """Number of spatial tokens produced by the frame encoder."""

        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side


class _ActionEmbedder(nn.Module):
    """Apply the upstream action MLP to Dodge's nine-way one-hot actions."""

    def __init__(self, config: LeWMConfig) -> None:
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Conv1d(
                config.action_dim,
                config.action_smoothing_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.Identity(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.action_smoothing_dim, 4 * config.embed_dim),
            nn.SiLU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.float().transpose(1, 2)
        actions = self.embedder[0](actions).transpose(1, 2)
        return self.mlp(actions)


class LeWorldModel(nn.Module):
    """Joint pixel encoder and causal action-conditioned latent predictor."""

    def __init__(self, config: LeWMConfig | None = None) -> None:
        super().__init__()
        self.config = config or LeWMConfig.reference()
        self.encoder = self._build_encoder(self.config)
        self.action_encoder = _ActionEmbedder(self.config)
        self.predictor = ARPredictor(
            num_frames=self.config.history_size,
            depth=self.config.predictor_depth,
            heads=self.config.predictor_heads,
            mlp_dim=self.config.predictor_mlp_dim,
            input_dim=self.config.embed_dim,
            hidden_dim=self.config.embed_dim,
            output_dim=self.config.embed_dim,
            dim_head=self.config.predictor_dim_head,
            dropout=self.config.predictor_dropout,
            emb_dropout=self.config.predictor_embedding_dropout,
        )
        self.projector = MLP(
            input_dim=self.config.embed_dim,
            hidden_dim=self.config.projector_hidden_dim,
            output_dim=self.config.embed_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.pred_projector = MLP(
            input_dim=self.config.embed_dim,
            hidden_dim=self.config.projector_hidden_dim,
            output_dim=self.config.embed_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.sigreg = SIGReg(
            knots=self.config.sigreg_knots,
            num_proj=self.config.sigreg_num_proj,
        )
        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

    @staticmethod
    def _build_encoder(config: LeWMConfig) -> nn.Module:
        try:
            from transformers import ViTConfig, ViTModel
        except Exception as error:
            raise ImportError(
                "LeWorldModel requires transformers for the HF ViT encoder"
            ) from error
        hf_config = ViTConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=3,
            hidden_size=config.embed_dim,
            num_hidden_layers=config.encoder_depth,
            num_attention_heads=config.encoder_heads,
            intermediate_size=config.encoder_mlp_dim,
            hidden_dropout_prob=config.encoder_dropout,
            attention_probs_dropout_prob=config.encoder_attention_dropout,
            layer_norm_eps=1e-12,
            qkv_bias=True,
            use_mask_token=False,
            attn_implementation="eager",
        )
        return ViTModel(hf_config, add_pooling_layer=False)

    def _prepare_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixels must be a torch.Tensor")
        if pixels.ndim != 5:
            raise ValueError(
                "pixels must have shape (B, T, 3, H, W), "
                f"got {tuple(pixels.shape)}"
            )
        if pixels.shape[2] != 3:
            raise ValueError(f"pixels must have three channels, got {pixels.shape[2]}")
        if pixels.shape[1] < 1 or pixels.shape[3] < 1 or pixels.shape[4] < 1:
            raise ValueError("pixels must have positive batch, time, and spatial sizes")
        if self.config.input_encoding != INPUT_ENCODING_LEGACY:
            return self._prepare_new_pixels(pixels)
        if pixels.dtype == torch.uint8:
            values = pixels.float() / 255.0
            needs_normalization = True
        elif torch.is_floating_point(pixels):
            values = pixels.float()
            low = values.detach().amin().item()
            high = values.detach().amax().item()
            if low >= 0.0 and high <= 1.0:
                needs_normalization = True
            else:
                raise ValueError(
                    "floating pixels must be normalized RGB values in [0, 1]"
                )
        else:
            raise TypeError("pixels must be uint8 or floating point")
        frames = values.reshape(-1, 3, values.shape[-2], values.shape[-1])
        if needs_normalization:
            frames = (frames - self.image_mean) / self.image_std
        if frames.shape[-2:] != (self.config.image_size, self.config.image_size):
            frames = F.interpolate(
                frames,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return frames.reshape(
            pixels.shape[0],
            pixels.shape[1],
            3,
            self.config.image_size,
            self.config.image_size,
        )

    def _unit_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Convert uint8 or normalized RGB input to float values in ``[0, 1]``."""

        if pixels.dtype == torch.uint8:
            return pixels.float().div(255.0)
        if torch.is_floating_point(pixels):
            values = pixels.float()
            low = values.detach().amin().item()
            high = values.detach().amax().item()
            if low >= 0.0 and high <= 1.0:
                return values
            raise ValueError(
                "floating pixels must be normalized RGB values in [0, 1]"
            )
        raise TypeError("pixels must be uint8 or floating point")

    def _native_palette_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return exact native RGB bytes for palette lookup before resizing."""

        if pixels.dtype == torch.uint8:
            return pixels
        values = self._unit_pixels(pixels)
        scaled = values * 255.0
        rounded = scaled.round()
        if not bool(torch.allclose(scaled, rounded, rtol=0.0, atol=1e-5)):
            raise ValueError(
                "palette input requires exact native RGB values before resizing"
            )
        return rounded.to(dtype=torch.uint8)

    def _palette_class_indices(self, pixels: torch.Tensor) -> torch.Tensor:
        """Map native RGB pixels to the configured three palette classes."""

        palette = torch.tensor(
            self.config.palette_rgb,
            device=pixels.device,
            dtype=torch.uint8,
        )
        native = self._native_palette_pixels(pixels)
        batch, time, _, height, width = native.shape
        native = native.reshape(-1, 3, height, width)
        matches = (
            native.unsqueeze(1)
            == palette.view(1, _PALETTE_CHANNELS, 3, 1, 1)
        ).all(dim=2)
        covered = matches.any(dim=1)
        if not bool(torch.all(covered)):
            raise ValueError("pixels contain an RGB color outside configured palette")
        return matches.to(dtype=torch.int64).argmax(dim=1).reshape(
            batch, time, height, width
        )

    def _prepare_new_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Apply nearest resize and symmetric normalization for new input arms."""

        if self.config.input_encoding == INPUT_ENCODING_RGB_NEAREST_SYMMETRIC:
            frames = self._unit_pixels(pixels).reshape(
                -1, 3, pixels.shape[-2], pixels.shape[-1]
            )
        else:
            classes = self._palette_class_indices(pixels).reshape(
                -1, pixels.shape[-2], pixels.shape[-1]
            )
            frames = F.one_hot(classes, num_classes=_PALETTE_CHANNELS).permute(
                0, 3, 1, 2
            ).to(dtype=torch.float32)
        if frames.shape[-2:] != (self.config.image_size, self.config.image_size):
            frames = F.interpolate(
                frames,
                size=(self.config.image_size, self.config.image_size),
                mode="nearest",
            )
        frames = frames.mul(2.0).sub(1.0)
        return frames.reshape(
            pixels.shape[0],
            pixels.shape[1],
            3,
            self.config.image_size,
            self.config.image_size,
        )

    def _encode_hidden_states(
        self,
        pixels: torch.Tensor,
        *,
        output_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        prepared = self._prepare_pixels(pixels)
        batch, time = prepared.shape[:2]
        frames = prepared.reshape(-1, 3, self.config.image_size, self.config.image_size)
        outputs = self.encoder(
            pixel_values=frames,
            output_attentions=output_attention,
            return_dict=True,
        )
        tokens = outputs.last_hidden_state
        attention = None
        if output_attention:
            attentions = outputs.attentions
            if attentions:
                attention = attentions[-1].mean(dim=1)[:, 0, 1:]
        expected_tokens = self.config.num_patches + 1
        if tokens.shape[1:] != (expected_tokens, self.config.embed_dim):
            raise ValueError(
                "encoder hidden states must have shape "
                f"(batch*time, {expected_tokens}, {self.config.embed_dim}), "
                f"got {tuple(tokens.shape)}"
            )
        if attention is not None and attention.shape[1] != self.config.num_patches:
            raise ValueError("encoder attention does not match patch-token count")
        return tokens, attention, batch, time

    def _encode_tokens(
        self,
        pixels: torch.Tensor,
        *,
        output_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens, attention, batch, time = self._encode_hidden_states(
            pixels,
            output_attention=output_attention,
        )
        cls = tokens[:, 0].reshape(batch, time, self.config.embed_dim)
        if attention is not None:
            attention = attention.reshape(batch, time, self.config.num_patches)
        return cls, attention

    def encode_readout_tokens(
        self, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CLS and final encoder patch tokens in raster order.

        The patch sequence is the ViT embedding order: row-major spatial
        patches after the leading CLS token.  Reference inputs therefore
        return ``(B, T, 192)`` and ``(B, T, 256, 192)``.
        """

        tokens, _, batch, time = self._encode_hidden_states(pixels)
        cls = tokens[:, 0].reshape(batch, time, self.config.embed_dim)
        patches = tokens[:, 1:].reshape(
            batch,
            time,
            self.config.num_patches,
            self.config.embed_dim,
        )
        return cls, patches

    def _apply_projector(self, projector: MLP, values: torch.Tensor) -> torch.Tensor:
        batch, time, dim = values.shape
        flat = values.reshape(batch * time, dim)
        output = projector(flat)
        return output.reshape(batch, time, self.config.embed_dim)

    def encode_cls(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return encoder CLS tokens before projection, shaped ``(B,T,D)``.

        This is the representation used by the paper's visualization decoder.
        The world-model predictor continues to consume projected latents.
        """

        cls, _ = self._encode_tokens(pixels)
        return cls

    def encode_representation(
        self, pixels: torch.Tensor, *, representation: str = "projected"
    ) -> torch.Tensor:
        """Select the input for a current-frame reconstruction probe."""

        if representation == "cls":
            return self.encode_cls(pixels)
        if representation == "projected":
            return self.encode(pixels)
        raise ValueError("representation must be 'cls' or 'projected'")

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode RGB frames into projected CLS latents shaped ``(B,T,D)``."""

        cls, _ = self._encode_tokens(pixels)
        return self._apply_projector(self.projector, cls)

    def encode_with_attention(
        self,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected latents and last-layer CLS-to-patch attention.

        Attention is averaged over encoder heads and has shape
        ``(B, T, grid, grid)``.  It is a visualization diagnostic, not a
        certified player-attention map.
        """

        cls, attention = self._encode_tokens(pixels, output_attention=True)
        if attention is None:
            raise RuntimeError("encoder did not return attention weights")
        grid = self.config.image_size // self.config.patch_size
        attention = attention.reshape(
            attention.shape[0],
            attention.shape[1],
            grid,
            grid,
        )
        return self._apply_projector(self.projector, cls), attention

    def _action_features(
        self,
        actions: torch.Tensor,
        *,
        batch: int,
        time: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.shape[:2] != (batch, time):
            raise ValueError(
                f"actions must start with shape {(batch, time)}, "
                f"got {tuple(actions.shape)}"
            )
        if actions.ndim == 2:
            if actions.dtype == torch.bool or actions.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError("index actions must use an integer dtype")
            if actions.numel():
                minimum = int(actions.detach().amin().item())
                maximum = int(actions.detach().amax().item())
                if minimum < 0 or maximum >= self.config.action_dim:
                    raise ValueError(
                        f"action indices must be in [0, {self.config.action_dim})"
                    )
            return F.one_hot(
                actions.long(),
                num_classes=self.config.action_dim,
            ).to(dtype=dtype)
        if actions.ndim != 3 or actions.shape[-1] != self.config.action_dim:
            raise ValueError(
                "actions must have shape (B,T) indices or "
                f"(B,T,{self.config.action_dim}) one-hot values"
            )
        if (
            not torch.is_floating_point(actions)
            and actions.dtype != torch.bool
            and actions.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        ):
            raise TypeError("one-hot actions must be numeric")
        return actions.to(dtype=dtype)

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next projected latents from latent/action history."""

        if not isinstance(z, torch.Tensor) or z.ndim != 3:
            raise ValueError("z must have shape (B,T,D)")
        if z.shape[-1] != self.config.embed_dim or z.shape[1] < 1:
            raise ValueError(
                f"z must have shape (B,T,{self.config.embed_dim}) with T >= 1"
            )
        if z.shape[1] > self.config.history_size:
            raise ValueError(
                f"predictor history T={z.shape[1]} exceeds "
                f"history_size={self.config.history_size}"
            )
        if not torch.is_floating_point(z):
            raise TypeError("z must be floating point")
        action_features = self._action_features(
            actions,
            batch=z.shape[0],
            time=z.shape[1],
            dtype=z.dtype,
        ).to(device=z.device)
        action_embeddings = self.action_encoder(action_features)
        predictions = self.predictor(z, action_embeddings)
        return self._apply_projector(self.pred_projector, predictions)

    def compute_loss(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute attached-target prediction plus SIGReg losses.

        ``pixels`` contains ``H+1`` frames and ``actions`` contains the first
        ``H`` executed actions.  The target ``z[:,1:]`` remains attached to
        the graph by design; no EMA or stop-gradient target is used.
        """

        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 5:
            raise ValueError("pixels must have shape (B,H+1,3,height,width)")
        if pixels.shape[1] < 2:
            raise ValueError("compute_loss requires at least two pixel frames")
        if not isinstance(actions, torch.Tensor) or actions.shape[:2] != (
            pixels.shape[0],
            pixels.shape[1] - 1,
        ):
            raise ValueError(
                "actions must have shape (B,H) for pixels shaped (B,H+1,3,height,width)"
            )
        embeddings = self.encode(pixels)
        predictions = self.predict(embeddings[:, :-1], actions)
        targets = embeddings[:, 1:]
        pred_loss = F.mse_loss(predictions, targets)
        sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
        loss = pred_loss + self.config.sigreg_weight * sigreg_loss
        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
        }

    def forward(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.compute_loss(pixels, actions)

    def rollout(
        self,
        context_pixels: torch.Tensor,
        actions: torch.Tensor,
        *,
        return_context: bool = True,
    ) -> torch.Tensor:
        """Roll latent predictions using context pixels and an action plan.

        ``actions`` must contain one action for every context frame followed by
        any future actions.  Only ``context_pixels`` are encoded.  By default
        the returned sequence includes the context latents, mirroring the
        upstream JEPA rollout; pass ``return_context=False`` for future-only
        predictions.
        """

        if not isinstance(context_pixels, torch.Tensor) or context_pixels.ndim != 5:
            raise ValueError("context_pixels must have shape (B,T,3,height,width)")
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        context = self.encode(context_pixels)
        batch, context_time, _ = context.shape
        if (
            actions.ndim < 2
            or actions.shape[0] != batch
            or actions.shape[1] < context_time
        ):
            raise ValueError(
                "rollout actions must provide at least one action per context frame"
            )
        action_features = self._action_features(
            actions,
            batch=batch,
            time=actions.shape[1],
            dtype=context.dtype,
        ).to(device=context.device)
        generated = context
        for _ in range(actions.shape[1] - context_time + 1):
            length = generated.shape[1]
            history = min(length, self.config.history_size)
            predicted = self.predict(
                generated[:, -history:],
                action_features[:, length - history : length],
            )[:, -1:]
            generated = torch.cat((generated, predicted), dim=1)
        if return_context:
            return generated
        return generated[:, context_time:]
