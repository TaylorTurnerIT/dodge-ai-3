from __future__ import annotations

import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn.query_decoder import QueryPixelDecoder


def _tiny_decoder(**overrides: int) -> QueryPixelDecoder:
    values = {
        "latent_dim": 8,
        "hidden_dim": 16,
        "depth": 2,
        "heads": 4,
        "output_size": 8,
        "patch_size": 4,
    }
    values.update(overrides)
    return QueryPixelDecoder(**values)


def _unpatchify(
    patches: torch.Tensor,
    *,
    output_size: int,
    patch_size: int,
) -> torch.Tensor:
    grid = output_size // patch_size
    batch = patches.shape[0]
    patches = patches.reshape(batch, grid, grid, patch_size, patch_size, 3)
    return patches.permute(0, 5, 1, 3, 2, 4).reshape(
        batch, 3, output_size, output_size
    )


def test_default_profile_has_native_shape_and_locked_decoder_structure() -> None:
    decoder = QueryPixelDecoder()

    assert decoder.latent_dim == 192
    assert decoder.hidden_dim == 128
    assert decoder.depth == 3
    assert decoder.heads == 4
    assert decoder.output_size == 128
    assert decoder.patch_size == 16
    assert decoder.num_patches == 64
    assert decoder.query_tokens.shape == (1, 64, 128)
    assert all(
        block.cross_attention.batch_first and block.cross_attention.dropout == 0.0
        for block in decoder.blocks
    )
    assert all(
        module.p == 0.0
        for module in decoder.modules()
        if isinstance(module, nn.Dropout)
    )


def test_decoder_returns_sigmoid_rgb_and_preserves_patch_layout() -> None:
    decoder = _tiny_decoder(
        latent_dim=4,
        hidden_dim=8,
        depth=1,
        heads=2,
        output_size=4,
        patch_size=2,
    )
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        patch_code = torch.arange(
            decoder.patch_size**2 * 3,
            dtype=torch.float32,
        )
        decoder.patch_head.bias.copy_(patch_code)

    latent = torch.zeros(1, decoder.latent_dim)
    actual = decoder(latent)
    repeated_patches = torch.sigmoid(patch_code).reshape(1, 1, -1).expand(
        1, decoder.num_patches, -1
    )
    expected = _unpatchify(
        repeated_patches,
        output_size=decoder.output_size,
        patch_size=decoder.patch_size,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 3, 4, 4)
    assert bool((actual >= 0).all() and (actual <= 1).all())

    coded_patches = torch.arange(
        decoder.num_patches * decoder.patch_size**2 * 3,
        dtype=torch.float32,
    ).reshape(1, decoder.num_patches, -1)
    torch.testing.assert_close(
        decoder._unpatchify(coded_patches),
        _unpatchify(
            coded_patches,
            output_size=decoder.output_size,
            patch_size=decoder.patch_size,
        ),
    )


def test_latent_queries_and_mlp_receive_gradients() -> None:
    decoder = _tiny_decoder()
    latent = torch.randn(3, decoder.latent_dim, requires_grad=True)

    decoder(latent).square().mean().backward()

    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert decoder.query_tokens.grad is not None
    assert decoder.query_tokens.grad.abs().sum() > 0
    mlp_parameters = [
        parameter
        for name, parameter in decoder.named_parameters()
        if ".mlp." in name
    ]
    assert mlp_parameters
    assert all(parameter.grad is not None for parameter in mlp_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in mlp_parameters)


def test_forward_is_deterministic_without_dropout() -> None:
    decoder = _tiny_decoder().train()
    latent = torch.randn(2, decoder.latent_dim)

    first = decoder(latent)
    second = decoder(latent)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dim": 7, "heads": 2}, "hidden_dim must be divisible"),
        ({"output_size": 7, "patch_size": 4}, "output_size must be divisible"),
        ({"depth": 0}, "depth must be a positive integer"),
        ({"heads": True}, "heads must be a positive integer"),
    ],
)
def test_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _tiny_decoder(**kwargs)  # type: ignore[arg-type]


def test_rejects_invalid_latent_shapes() -> None:
    decoder = _tiny_decoder()

    with pytest.raises(ValueError, match="latent must have shape"):
        decoder(torch.zeros(2, decoder.latent_dim, 1))
    with pytest.raises(ValueError, match="latent must have shape"):
        decoder(torch.zeros(2, decoder.latent_dim + 1))
