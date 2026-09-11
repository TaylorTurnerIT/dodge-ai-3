from __future__ import annotations

import pytest
import torch
from torch import nn

from dodge_native_game.variants.cnn_image_ddqn.model import (
    IMAGE_SHAPE,
    INITIALIZATION_ID,
    AtariCnnQNetwork,
)


def test_initialization_has_versioned_identity() -> None:
    assert INITIALIZATION_ID == "kaiming-relu-xavier-head-v1"


def test_atari_cnn_has_dueling_head_and_no_probability_layers() -> None:
    model = AtariCnnQNetwork(num_actions=5)

    observations = torch.zeros((2, *IMAGE_SHAPE))
    outputs = model(observations)

    assert outputs.shape == (2, 5)
    assert model.features[0].in_channels == 4
    assert model.features[0].out_channels == 32
    assert model.features[0].kernel_size == (8, 8)
    assert model.features[0].stride == (4, 4)
    assert model.features[2].out_channels == 64
    assert model.features[2].kernel_size == (4, 4)
    assert model.features[2].stride == (2, 2)
    assert model.features[4].kernel_size == (3, 3)
    assert model.features[4].stride == (1, 1)
    assert model.shared[1].in_features == 3136
    assert model.shared[1].out_features == 512
    assert model.value_stream.out_features == 1
    assert model.advantage_stream.out_features == 5
    assert not any(isinstance(layer, nn.Softmax) for layer in model.modules())
    assert not any(
        isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.MaxPool2d))
        for layer in model.modules()
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 1_684_128 + 513 + 513 * 5

    outputs.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    with torch.no_grad():
        shared = model.shared(model.features(observations))
        value = model.value_stream(shared)
        advantage = model.advantage_stream(shared)
        expected = value + advantage - advantage.mean(dim=1, keepdim=True)
    torch.testing.assert_close(outputs, expected)


def test_plain_head_is_an_explicit_comparison_option() -> None:
    model = AtariCnnQNetwork(num_actions=3, dueling=False)

    outputs = model(torch.zeros((1, *IMAGE_SHAPE)))

    assert outputs.shape == (1, 3)
    assert model.dueling is False
    assert model.q_head.in_features == 512
    assert model.q_head.out_features == 3


def test_uint8_model_input_is_normalized_at_forward_boundary() -> None:
    model = AtariCnnQNetwork(num_actions=2)
    raw = torch.full((1, *IMAGE_SHAPE), 128, dtype=torch.uint8)

    with torch.no_grad():
        raw_outputs = model(raw)
        normalized_outputs = model(raw.float() / 255.0)

    torch.testing.assert_close(raw_outputs, normalized_outputs)


@pytest.mark.parametrize("stack_size", [1, 2, 4, 8])
def test_model_input_channels_are_the_temporal_ablation_knob(stack_size: int) -> None:
    model = AtariCnnQNetwork(num_actions=9, input_channels=stack_size)

    outputs = model(torch.zeros((2, stack_size, 84, 84)))

    assert outputs.shape == (2, 9)
    assert model.features[0].in_channels == stack_size


def test_he_init_keeps_activation_variance_stable() -> None:
    torch.manual_seed(0)
    layer_vars: dict[str, list[float]] = {"c1": [], "c2": [], "c3": [], "fc": []}
    q_vars: list[float] = []
    for _ in range(10):
        model = AtariCnnQNetwork(num_actions=9)
        inputs = torch.rand(128, *IMAGE_SHAPE)
        with torch.no_grad():
            hidden = torch.relu(model.features[0](inputs))
            layer_vars["c1"].append(hidden.var().item())
            hidden = torch.relu(model.features[2](hidden))
            layer_vars["c2"].append(hidden.var().item())
            hidden = torch.relu(model.features[4](hidden))
            layer_vars["c3"].append(hidden.var().item())
            flat = torch.relu(model.shared[1](torch.flatten(hidden, 1)))
            layer_vars["fc"].append(flat.var().item())
            q_values = model.value_stream(flat) + model.advantage_stream(flat)
            q_vars.append(q_values.var().item())

    means = {
        name: torch.tensor(values).mean().item() for name, values in layer_vars.items()
    }
    for name, mean in means.items():
        assert 0.05 < mean < 1.0, f"{name} variance {mean} is not stable"
    assert max(means.values()) / min(means.values()) < 5.0
    q_mean = torch.tensor(q_vars).mean().item()
    assert 0.1 < q_mean < 3.0


def test_init_gradients_do_not_vanish() -> None:
    torch.manual_seed(1)
    grad_vars: list[float] = []
    for _ in range(5):
        model = AtariCnnQNetwork(num_actions=9)
        loss = model(torch.rand(32, *IMAGE_SHAPE)).pow(2).mean()
        model.zero_grad()
        loss.backward()
        grad_vars.append(model.features[0].weight.grad.var().item())
    assert torch.tensor(grad_vars).mean().item() > 1e-05
