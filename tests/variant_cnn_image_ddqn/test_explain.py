from __future__ import annotations

import pytest
import torch

from dodge_native_game.variants.cnn_image_ddqn.explain import (
    CONV_FEATURE_SHAPE,
    channel_ablation,
    decompose,
    explain_observation,
    pairwise_feature_contributions,
    perturbation_maps,
)
from dodge_native_game.variants.cnn_image_ddqn.model import (
    IMAGE_SHAPE,
    AtariCnnQNetwork,
)


@pytest.fixture
def model() -> AtariCnnQNetwork:
    torch.manual_seed(11)
    return AtariCnnQNetwork(num_actions=5, input_channels=4)


@pytest.fixture
def observations() -> torch.Tensor:
    torch.manual_seed(12)
    return torch.rand((2, *IMAGE_SHAPE))


def test_decompose_matches_forward_and_exposes_declared_intermediates(
    model: AtariCnnQNetwork, observations: torch.Tensor
) -> None:
    with torch.no_grad():
        expected = model(observations)
    result = decompose(model, observations, max_batch_size=1)

    torch.testing.assert_close(result.q_values, expected)
    assert result.values is not None
    assert result.advantages is not None
    assert result.centered_advantages is not None
    assert result.conv_features.shape == (2, *CONV_FEATURE_SHAPE)
    assert result.shared_features.shape == (2, 512)
    torch.testing.assert_close(
        result.q_values,
        result.values + result.centered_advantages,
    )
    torch.testing.assert_close(
        result.centered_advantages,
        result.advantages - result.advantages.mean(dim=1, keepdim=True),
    )
    assert not result.q_values.requires_grad
    assert not result.conv_features.requires_grad


def test_pairwise_contributions_reconstruct_fixed_gap_including_bias(
    model: AtariCnnQNetwork, observations: torch.Tensor
) -> None:
    with torch.no_grad():
        model.advantage_stream.bias.copy_(
            torch.tensor([-0.4, 0.1, 0.8, -0.2, 0.3])
        )
    result = decompose(model, observations)
    pair = pairwise_feature_contributions(result, 1, 3)

    expected_gap = result.q_values[:, 1] - result.q_values[:, 3]
    assert pair.feature_contributions.shape == (2, 512)
    assert pair.total_contributions.shape == (2, 513)
    torch.testing.assert_close(pair.q_gap, expected_gap)
    torch.testing.assert_close(pair.total_contributions.sum(dim=1), expected_gap)
    torch.testing.assert_close(
        pair.total_contributions[:, -1],
        torch.full((2,), 0.3),
    )


def test_perturbation_map_is_signed_and_keeps_action_pair_fixed(
    model: AtariCnnQNetwork, observations: torch.Tensor
) -> None:
    result = perturbation_maps(
        model,
        observations[:1],
        chosen_action=1,
        alternative_action=3,
        frame=0,
        stride=42,
        radius=3,
        max_batch_size=1,
    )

    assert result.grid_shape == (2, 2)
    assert result.channels == (0,)
    assert result.value_map is not None
    torch.testing.assert_close(
        result.value_map,
        result.original_value[:, None, None] - result.perturbed_value,
    )
    torch.testing.assert_close(
        result.decision_map,
        result.original_gap[:, None, None] - result.perturbed_gap,
    )
    torch.testing.assert_close(result.decision_delta_map, result.decision_map)
    assert result.chosen == 1
    assert result.alternative == 3


def test_frame_selection_and_all_stack_selection_are_explicit(
    model: AtariCnnQNetwork, observations: torch.Tensor
) -> None:
    one = perturbation_maps(
        model,
        observations[:1],
        1,
        3,
        frame=2,
        stride=84,
        radius=3,
        max_batch_size=1,
    )
    all_stack = perturbation_maps(
        model,
        observations[:1],
        1,
        3,
        frame=None,
        stride=84,
        radius=3,
        max_batch_size=1,
    )
    assert one.channels == (2,)
    assert one.frame == 2
    assert all_stack.channels == (0, 1, 2, 3)
    assert all_stack.frame is None
    assert one.grid_positions == ((42, 42),)


def test_conv_channel_ablation_runs_shared_and_heads_and_handles_channel_63(
    model: AtariCnnQNetwork, observations: torch.Tensor
) -> None:
    result = channel_ablation(
        model,
        observations[:1],
        chosen_action=1,
        alternative_action=3,
        channel=63,
        max_batch_size=1,
    )
    baseline = decompose(model, observations[:1])
    with torch.no_grad():
        conv = baseline.conv_features.clone()
        conv[:, 63] = 0
        shared = model.shared(conv)
        value = model.value_stream(shared)
        advantage = model.advantage_stream(shared)
        expected_q = value + advantage - advantage.mean(dim=1, keepdim=True)

    assert result.channel_indices == (63,)
    torch.testing.assert_close(result.ablated_q[:, 0], expected_q)
    torch.testing.assert_close(
        result.gap_delta[:, 0],
        baseline.q_values[:, 1]
        - baseline.q_values[:, 3]
        - (expected_q[:, 1] - expected_q[:, 3]),
    )


def test_plain_head_never_invents_value_or_advantage_terms(
    observations: torch.Tensor,
) -> None:
    torch.manual_seed(13)
    model = AtariCnnQNetwork(num_actions=5, dueling=False, input_channels=4)
    result = decompose(model, observations[:1])
    ablation = channel_ablation(model, observations[:1], channel=0)
    maps = perturbation_maps(model, observations[:1], 1, 3, stride=84, radius=3)

    assert result.values is None
    assert result.advantages is None
    assert result.centered_advantages is None
    assert ablation.original_value is None
    assert ablation.ablated_value is None
    assert ablation.value_delta is None
    assert maps.value_map is None
    assert maps.original_value is None


def test_explanation_does_not_mutate_model_state_or_gradients(
    model: AtariCnnQNetwork, observations: torch.Tensor
) -> None:
    model.train()
    model.features[0].weight.grad = torch.ones_like(model.features[0].weight)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    before_training = {id(module): module.training for module in model.modules()}
    before_grad = model.features[0].weight.grad.detach().clone()

    explain_observation(
        model,
        observations[0],
        chosen=1,
        alternative=3,
        frame=0,
        channel=63,
        stride=84,
        radius=3,
        max_batch_size=1,
    )

    assert all(
        torch.equal(value, before[key]) for key, value in model.state_dict().items()
    )
    assert all(
        module.training == before_training[id(module)] for module in model.modules()
    )
    assert model.features[0].weight.grad is not None
    torch.testing.assert_close(model.features[0].weight.grad, before_grad)


def test_explanation_adapter_has_json_ready_contract(model: AtariCnnQNetwork) -> None:
    result = explain_observation(
        model,
        torch.zeros(IMAGE_SHAPE, dtype=torch.uint8),
        chosen=1,
        alternative=3,
        frame=None,
        channel=63,
        stride=84,
        radius=3,
        max_batch_size=1,
    )

    assert result["chosen"] == 1
    assert result["alternative"] == 3
    assert len(result["conv_maps"]) == 64
    assert len(result["conv_maps"][0]) == 7
    assert len(result["contributions"]) == 512
    assert result["grid_positions"] == [[42, 42]]
    assert len(result["ablation"]["q"]) == 5


def test_pair_validation_rejects_same_or_out_of_range_actions(
    model: AtariCnnQNetwork, observations: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="must differ"):
        perturbation_maps(model, observations[:1], 2, 2, stride=84)
    with pytest.raises(ValueError, match="between"):
        perturbation_maps(model, observations[:1], 5, 1, stride=84)
