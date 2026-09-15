from __future__ import annotations

import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn import decoder_study
from dodge_native_game.variants.pixel_repr_ddqn.decoder_study import (
    cache_train_windows,
    fit_cached_decoder,
)


class _TinyDataset:
    def __init__(self, count: int = 5) -> None:
        self.count = count
        self.split = "train"
        self.calls: list[int] = []

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.calls.append(index)
        value = (index * 31) % 256
        return {
            "pixels": torch.full((4, 3, 128, 128), value, dtype=torch.uint8),
            "actions": torch.zeros(3, dtype=torch.int64),
        }


class _FrozenToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.encode_calls: list[int] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.encode_calls.append(int(pixels.shape[0]))
        value = pixels.float().mean(dim=(2, 3, 4), keepdim=False) / 255.0
        return value.unsqueeze(-1).repeat(1, 1, 4) * self.scale


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3 * 128 * 128)
        self.first_latents: list[torch.Tensor] = []

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.first_latents.append(latents[0].detach().cpu().clone())
        return torch.sigmoid(self.linear(latents))


def test_cache_encodes_each_train_window_once_and_keeps_native_targets() -> None:
    dataset = _TinyDataset()
    model = _FrozenToyModel()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    latents, targets = cache_train_windows(
        model, dataset, device="cpu", batch_size=2
    )

    assert model.encode_calls == [2, 2, 1]
    assert len(dataset.calls) == len(dataset)
    assert latents.shape == (5, 4, 4)
    assert targets.shape == (5, 4, 3, 128, 128)
    assert not latents.requires_grad and not targets.requires_grad
    assert torch.equal(targets[1, 0], torch.full((3, 128, 128), 31 / 255.0))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_cached_fit_matches_sampler_and_copies_retained_snapshots() -> None:
    windows = 4
    latents = torch.arange(windows * 4 * 4, dtype=torch.float32).reshape(
        windows, 4, 4
    )
    latents.requires_grad_()
    targets = torch.zeros(windows, 4, 3, 128, 128)
    decoder = _TinyDecoder()

    fit = fit_cached_decoder(
        decoder,
        latents,
        targets,
        steps=4,
        batch_size=2,
        seed=904,
        sampling_seed=903,
        snapshot_steps=(2, 4),
    )

    generator = torch.Generator().manual_seed(903)
    torch.randint(windows, (2,), generator=generator)
    expected_index = int(torch.randint(windows, (2,), generator=generator)[0])
    assert int(decoder.first_latents[0][0]) == expected_index * 16
    assert latents.grad is None
    assert [metric["step"] for metric in fit.metrics] == [1, 2, 3, 4]
    assert {"decoder_loss", "loss", "updates_per_second", "phase"} <= set(
        fit.metrics[-1]
    )

    step_two = fit.snapshots[2]["linear.weight"].clone()
    fit.snapshots[4]["linear.weight"].zero_()
    torch.testing.assert_close(
        fit.snapshots[2]["linear.weight"], step_two, rtol=0, atol=0
    )


def test_cached_fit_resume_matches_uninterrupted_optimizer_and_sampler() -> None:
    windows = 8
    latents = torch.arange(windows * 4 * 4, dtype=torch.float32).reshape(
        windows, 4, 4
    )
    targets = torch.zeros(windows, 4, 3, 128, 128)

    torch.manual_seed(123)
    initial_decoder = _TinyDecoder()
    initial_state = {
        key: value.detach().clone()
        for key, value in initial_decoder.state_dict().items()
    }
    uninterrupted = _TinyDecoder()
    uninterrupted.load_state_dict(initial_state, strict=True)
    full_fit = fit_cached_decoder(
        uninterrupted,
        latents,
        targets,
        steps=8,
        batch_size=2,
        seed=904,
        sampling_seed=903,
        snapshot_steps=(8,),
    )

    first_decoder = _TinyDecoder()
    first_decoder.load_state_dict(initial_state, strict=True)
    first_fit = fit_cached_decoder(
        first_decoder,
        latents,
        targets,
        steps=4,
        batch_size=2,
        seed=904,
        sampling_seed=903,
        snapshot_steps=(4,),
    )
    resumed_decoder = _TinyDecoder()
    resumed_decoder.load_state_dict(first_fit.snapshots[4], strict=True)
    resumed_fit = fit_cached_decoder(
        resumed_decoder,
        latents,
        targets,
        steps=8,
        batch_size=2,
        seed=904,
        sampling_seed=903,
        snapshot_steps=(8,),
        resume_state={
            "steps": 4,
            "optimizer": first_fit.optimizer_state,
            "generator_state": first_fit.generator_state,
            "torch_rng": first_fit.torch_rng_state,
            "cuda_rng": first_fit.cuda_rng_state,
        },
    )

    for key, value in full_fit.snapshots[8].items():
        torch.testing.assert_close(
            value, resumed_fit.snapshots[8][key], rtol=0, atol=0
        )
    assert [metric["step"] for metric in resumed_fit.metrics] == [5, 6, 7, 8]
    assert torch.equal(
        full_fit.generator_state, resumed_fit.generator_state
    )


def test_8192_decoder_resume_requires_the_2048_step_checkpoint() -> None:
    with pytest.raises(ValueError, match="contain 2048 updates"):
        decoder_study._validate_resume_step({"steps": 512}, total_steps=8192)

    decoder_study._validate_resume_step({"steps": 2048}, total_steps=8192)
