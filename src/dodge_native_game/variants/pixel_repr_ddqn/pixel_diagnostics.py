"""Read-only pixel diagnostics for a frozen LeWM and decoder."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["evaluate_pixels"]

_OUTPUT_SIZE = (32, 32)
_CHANNELS = 3
_CHANGED_THRESHOLD = 1.0 / 255.0


def _as_pixel_batch(value: object) -> torch.Tensor:
    pixels = torch.as_tensor(value)
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 5 or pixels.shape[2] != _CHANNELS:
        raise ValueError(
            "pixels must have shape (B,T,3,H,W) or (T,3,H,W), "
            f"got {tuple(pixels.shape)}"
        )
    if pixels.shape[1] < 2 or pixels.shape[3] < 1 or pixels.shape[4] < 1:
        raise ValueError("pixels must contain at least two non-empty frames")
    return pixels


def _as_action_batch(value: object, *, batch: int, time: int) -> torch.Tensor:
    actions = torch.as_tensor(value)
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    if actions.ndim != 2 or actions.shape != (batch, time):
        raise ValueError(
            f"actions must have shape {(batch, time)}, got {tuple(actions.shape)}"
        )
    return actions


def _rgb01(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.dtype == torch.uint8:
        return pixels.float() / 255.0
    if not torch.is_floating_point(pixels):
        raise TypeError("pixels must be uint8 or floating point")
    values = pixels.float()
    if values.numel():
        if not bool(torch.isfinite(values).all()):
            raise ValueError("floating pixels must be finite")
        low = float(values.detach().amin())
        high = float(values.detach().amax())
        if low < 0.0 or high > 1.0:
            raise ValueError("floating pixels must be normalized RGB values in [0, 1]")
    return values


def _area_targets(pixels: torch.Tensor) -> torch.Tensor:
    values = _rgb01(pixels)
    return F.interpolate(values, size=_OUTPUT_SIZE, mode="area")


def _decode(decoder: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    decoded = decoder(latent)
    if decoded.ndim == 2:
        expected = _CHANNELS * _OUTPUT_SIZE[0] * _OUTPUT_SIZE[1]
        if decoded.shape[1] != expected:
            raise ValueError(
                f"decoder output must contain {expected} values per frame, "
                f"got {decoded.shape[1]}"
            )
        decoded = decoded.reshape(-1, _CHANNELS, *_OUTPUT_SIZE)
    elif decoded.ndim == 4 and decoded.shape[1] == _CHANNELS:
        if decoded.shape[-2:] != _OUTPUT_SIZE:
            decoded = F.interpolate(decoded, size=_OUTPUT_SIZE, mode="area")
    else:
        raise ValueError(
            "decoder output must have shape (B,3,32,32) or (B,3072), "
            f"got {tuple(decoded.shape)}"
        )
    decoded = decoded.float()
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("decoder produced nonfinite pixels")
    return decoded


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} is nonfinite")


def _episode_ids(raw: object, batch: int) -> list[str]:
    if raw is None:
        return ["unknown"] * batch
    if isinstance(raw, str):
        return [raw] * batch
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 0:
            return [str(raw.item())] * batch
        values = raw.detach().cpu().tolist()
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        try:
            values = list(raw)  # type: ignore[arg-type]
        except TypeError:
            return [str(raw)] * batch
    if len(values) == 1 and batch != 1:
        values *= batch
    if len(values) != batch:
        raise ValueError(
            f"episode_id must contain one value per batch item, got {len(values)}"
        )
    return [str(value) for value in values]


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    ]


@dataclass
class _Accumulator:
    """Sums needed to form pixel and changed-pixel means."""

    window_count: int = 0
    total_pixel_count: int = 0
    changed_pixel_count: int = 0
    current_reconstruction_sum: float = 0.0
    current_mean_image_sum: float = 0.0
    next_prediction_sum: float = 0.0
    next_persistence_sum: float = 0.0
    next_mean_image_sum: float = 0.0
    changed_pixel_prediction_sum: float = 0.0
    changed_pixel_persistence_sum: float = 0.0
    changed_pixel_mean_image_sum: float = 0.0

    def add(
        self,
        *,
        current_target: torch.Tensor,
        next_target: torch.Tensor,
        current_prediction: torch.Tensor,
        next_prediction: torch.Tensor,
        mean_image: torch.Tensor,
        changed: torch.Tensor,
    ) -> None:
        batch = int(current_target.shape[0])
        pixel_count = int(current_target.shape[-2] * current_target.shape[-1])
        self.window_count += batch
        self.total_pixel_count += batch * pixel_count

        current_errors = (current_prediction - current_target).square()
        current_mean_errors = (mean_image - current_target).square()
        next_errors = (next_prediction - next_target).square()
        persistence_errors = (current_target - next_target).square()
        next_mean_errors = (mean_image - next_target).square()
        for name, errors in (
            ("current reconstruction", current_errors),
            ("current mean image", current_mean_errors),
            ("next prediction", next_errors),
            ("next persistence", persistence_errors),
            ("next mean image", next_mean_errors),
        ):
            _require_finite(errors, f"{name} pixel error")
        for name, errors in (
            ("current reconstruction", current_errors),
            ("current mean image", current_mean_errors),
            ("next prediction", next_errors),
            ("next persistence", persistence_errors),
            ("next mean image", next_mean_errors),
        ):
            total = float(errors.sum())
            if not math.isfinite(total):
                raise RuntimeError(f"{name} pixel error sum is nonfinite")
            if name == "current reconstruction":
                self.current_reconstruction_sum += total
            elif name == "current mean image":
                self.current_mean_image_sum += total
            elif name == "next prediction":
                self.next_prediction_sum += total
            elif name == "next persistence":
                self.next_persistence_sum += total
            else:
                self.next_mean_image_sum += total

        changed_channels = changed.unsqueeze(1)
        self.changed_pixel_count += int(changed.sum())
        changed_sums = (
            (next_errors * changed_channels).sum(),
            (persistence_errors * changed_channels).sum(),
            (next_mean_errors * changed_channels).sum(),
        )
        for name, value in zip(
            (
                "changed prediction",
                "changed persistence",
                "changed mean image",
            ),
            changed_sums,
            strict=True,
        ):
            total = float(value)
            if not math.isfinite(total):
                raise RuntimeError(f"{name} pixel error sum is nonfinite")
            if name == "changed prediction":
                self.changed_pixel_prediction_sum += total
            elif name == "changed persistence":
                self.changed_pixel_persistence_sum += total
            else:
                self.changed_pixel_mean_image_sum += total

    def as_dict(self) -> dict[str, object]:
        channel_pixel_count = self.total_pixel_count * _CHANNELS
        changed_channel_pixel_count = self.changed_pixel_count * _CHANNELS

        def mean(total: float, count: int) -> float | None:
            value = total / count if count else None
            if value is not None and not math.isfinite(value):
                raise RuntimeError("pixel diagnostic metric is nonfinite")
            return value

        next_prediction = mean(self.next_prediction_sum, channel_pixel_count)
        next_persistence = mean(self.next_persistence_sum, channel_pixel_count)
        changed_prediction = mean(
            self.changed_pixel_prediction_sum, changed_channel_pixel_count
        )
        changed_persistence = mean(
            self.changed_pixel_persistence_sum, changed_channel_pixel_count
        )
        return {
            "window_count": self.window_count,
            "total_pixel_count": self.total_pixel_count,
            "changed_pixel_count": self.changed_pixel_count,
            "changed_pixel_fraction": (
                self.changed_pixel_count / self.total_pixel_count
                if self.total_pixel_count
                else 0.0
            ),
            "current_reconstruction_mse": mean(
                self.current_reconstruction_sum, channel_pixel_count
            ),
            "current_mean_image_mse": mean(
                self.current_mean_image_sum, channel_pixel_count
            ),
            "next_prediction_mse": next_prediction,
            "next_persistence_mse": next_persistence,
            "next_mean_image_mse": mean(self.next_mean_image_sum, channel_pixel_count),
            "changed_pixel_prediction_mse": changed_prediction,
            "changed_pixel_persistence_mse": changed_persistence,
            "changed_pixel_mean_image_mse": mean(
                self.changed_pixel_mean_image_sum, changed_channel_pixel_count
            ),
            "prediction_vs_persistence_ratio": (
                next_prediction / next_persistence
                if next_prediction is not None
                and next_persistence is not None
                and next_persistence > 0.0
                else None
            ),
            "changed_pixel_prediction_vs_persistence_ratio": (
                changed_prediction / changed_persistence
                if changed_prediction is not None
                and changed_persistence is not None
                and changed_persistence > 0.0
                else None
            ),
        }


def _module_modes(
    *modules: torch.nn.Module,
) -> tuple[list[torch.nn.Module], list[bool]]:
    unique: list[torch.nn.Module] = []
    seen: set[int] = set()
    for root in modules:
        for module in root.modules():
            if id(module) not in seen:
                seen.add(id(module))
                unique.append(module)
    return unique, [module.training for module in unique]


def evaluate_pixels(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    train_dataset: Iterable[Mapping[str, Any]],
    validation_dataset: Iterable[Mapping[str, Any]],
    device: torch.device | str,
) -> dict[str, object]:
    """Evaluate frozen latent decoding and pixel controls over all windows.

    The train split is used only to form a fixed current-frame mean image.  The
    model and decoder receive only pixels/actions; changed-pixel masks are
    computed from validation targets after inference and never enter a model
    input or loss.  No module parameters, buffers, modes, or random state are
    intentionally changed.
    """

    target_device = torch.device(device)
    modules, modes = _module_modes(model, decoder)
    model.eval()
    decoder.eval()
    global_stats = _Accumulator()
    episode_stats: defaultdict[str, _Accumulator] = defaultdict(_Accumulator)
    try:
        with torch.random.fork_rng(
            devices=_rng_devices(target_device)
        ), torch.no_grad():
            mean_sum: torch.Tensor | None = None
            train_count = 0
            for sample in train_dataset:
                pixels = _as_pixel_batch(sample["pixels"])
                current = _area_targets(pixels[:, -2])
                _require_finite(current, "training pixel target")
                batch_count = int(current.shape[0])
                batch_sum = current.sum(dim=0)
                mean_sum = batch_sum if mean_sum is None else mean_sum + batch_sum
                train_count += batch_count
            if mean_sum is None or train_count == 0:
                raise ValueError("train_dataset must contain at least one window")
            mean_image = (mean_sum / train_count).to(target_device)
            _require_finite(mean_image, "training mean image")
            for sample in validation_dataset:
                pixels = _as_pixel_batch(sample["pixels"]).to(target_device)
                _require_finite(_rgb01(pixels), "validation pixel target")
                batch = int(pixels.shape[0])
                actions = _as_action_batch(
                    sample["actions"], batch=batch, time=int(pixels.shape[1] - 1)
                ).to(target_device)
                encoded = model.encode(pixels)
                if encoded.ndim != 3 or encoded.shape[:2] != pixels.shape[:2]:
                    raise ValueError(
                        "model.encode must return shape (B,T,D), "
                        f"got {tuple(encoded.shape)}"
                    )
                _require_finite(encoded, "encoded latent")
                predicted = model.predict(encoded[:, :-1], actions)[:, -1]
                _require_finite(predicted, "predicted latent")
                current_prediction = _decode(decoder, encoded[:, -2])
                next_prediction = _decode(decoder, predicted)
                current_target, next_target = (
                    _area_targets(pixels[:, -2]),
                    _area_targets(pixels[:, -1]),
                )
                changed = (
                    (next_target - current_target).abs().mean(dim=1)
                    > _CHANGED_THRESHOLD
                )
                global_stats.add(
                    current_target=current_target,
                    next_target=next_target,
                    current_prediction=current_prediction,
                    next_prediction=next_prediction,
                    mean_image=mean_image.expand(batch, -1, -1, -1),
                    changed=changed,
                )
                ids = _episode_ids(sample.get("episode_id"), batch)
                for index, episode_id in enumerate(ids):
                    episode_stats[episode_id].add(
                        current_target=current_target[index : index + 1],
                        next_target=next_target[index : index + 1],
                        current_prediction=current_prediction[index : index + 1],
                        next_prediction=next_prediction[index : index + 1],
                        mean_image=mean_image.unsqueeze(0),
                        changed=changed[index : index + 1],
                    )
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.training = mode

    return {
        **global_stats.as_dict(),
        "train_window_count": train_count,
        "output_size": _OUTPUT_SIZE[0],
        "changed_threshold": _CHANGED_THRESHOLD,
        "evaluation_only": True,
        "per_episode": {
            episode_id: stats.as_dict()
            for episode_id, stats in episode_stats.items()
        },
    }
