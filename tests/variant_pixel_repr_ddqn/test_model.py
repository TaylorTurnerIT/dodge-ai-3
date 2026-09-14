from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn.model import LeWMConfig, LeWorldModel
from dodge_native_game.variants.pixel_repr_ddqn.sigreg import SIGReg
from dodge_native_game.variants.pixel_repr_ddqn.upstream import ARPredictor

REFERENCE_MODULE = Path(__file__).parents[2] / "references" / "le-wm" / "module.py"


def _load_reference_module():
    pytest.importorskip("einops")
    spec = importlib.util.spec_from_file_location(
        "lewm_reference_module",
        REFERENCE_MODULE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream reference from {REFERENCE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_model() -> LeWorldModel:
    torch.manual_seed(7)
    return LeWorldModel(LeWMConfig.tiny()).eval()


def test_reference_and_tiny_profiles_pin_expected_dimensions() -> None:
    reference = LeWMConfig.reference()
    tiny = LeWMConfig.tiny()

    assert (reference.image_size, reference.patch_size) == (224, 14)
    assert (reference.embed_dim, reference.encoder_depth) == (192, 12)
    assert (reference.predictor_depth, reference.predictor_heads) == (6, 16)
    assert reference.predictor_dim_head == 64
    assert tiny.profile == "tiny"
    assert tiny.image_size == 64
    assert tiny.embed_dim < reference.embed_dim
    assert tiny.num_patches == 64


def test_sigreg_matches_upstream_random_draw_and_axes() -> None:
    reference = _load_reference_module()
    projection = torch.randn(5, 4, 7)
    expected = reference.SIGReg(knots=7, num_proj=11)
    actual = SIGReg(knots=7, num_proj=11)

    torch.manual_seed(1234)
    expected_value = expected(projection)
    torch.manual_seed(1234)
    actual_value = actual(projection)

    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_predictor_matches_upstream_after_same_initialization() -> None:
    reference = _load_reference_module()
    kwargs = {
        "num_frames": 3,
        "depth": 2,
        "heads": 2,
        "mlp_dim": 32,
        "input_dim": 8,
        "hidden_dim": 8,
        "output_dim": 8,
        "dim_head": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
    }
    torch.manual_seed(4321)
    expected = reference.ARPredictor(**kwargs).eval()
    torch.manual_seed(4321)
    actual = ARPredictor(**kwargs).eval()
    inputs = torch.randn(2, 3, 8)
    conditions = torch.randn(2, 3, 8)

    torch.testing.assert_close(actual(inputs, conditions), expected(inputs, conditions))


def test_encode_and_attention_shapes_and_uint8_float_equivalence(
    tiny_model: LeWorldModel,
) -> None:
    pixels = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    with torch.no_grad():
        uint8_latents = tiny_model.encode(pixels)
        float_latents, attention = tiny_model.encode_with_attention(
            pixels.float() / 255.0
        )

    assert uint8_latents.shape == (2, 3, 64)
    assert attention.shape == (2, 3, 8, 8)
    torch.testing.assert_close(uint8_latents, float_latents, rtol=1e-5, atol=1e-5)


def test_predictor_is_causal_in_eval_mode(tiny_model: LeWorldModel) -> None:
    z = torch.randn(2, 3, 64)
    actions = torch.tensor([[0, 1, 2], [3, 4, 5]])
    changed_z = z.clone()
    changed_z[:, 2:] += 100.0
    changed_actions = actions.clone()
    changed_actions[:, 2] = 8

    with torch.no_grad():
        original = tiny_model.predict(z, actions)
        changed = tiny_model.predict(changed_z, changed_actions)

    torch.testing.assert_close(original[:, :2], changed[:, :2], rtol=1e-5, atol=1e-5)


def test_compute_loss_keeps_target_encoder_attached() -> None:
    class CapturingModel(LeWorldModel):
        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            embeddings = super().encode(pixels)
            embeddings.retain_grad()
            self.captured_embeddings = embeddings
            return embeddings

    torch.manual_seed(11)
    model = CapturingModel(LeWMConfig.tiny()).train()
    pixels = torch.rand(2, 4, 3, 32, 32)
    actions = torch.tensor([[0, 1, 2], [3, 4, 5]])

    losses = model.compute_loss(pixels, actions)
    assert set(losses) == {"loss", "pred_loss", "sigreg_loss"}
    assert torch.allclose(
        losses["loss"],
        losses["pred_loss"] + model.config.sigreg_weight * losses["sigreg_loss"],
    )
    model.zero_grad(set_to_none=True)
    losses["pred_loss"].backward()

    assert model.captured_embeddings.grad is not None
    assert model.captured_embeddings.grad[:, -1].abs().sum() > 0
    projection = model.encoder.embeddings.patch_embeddings.projection
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert projection.weight.grad.abs().sum() > 0


def test_batchnorm_running_state_is_fixed_in_eval(tiny_model: LeWorldModel) -> None:
    before = {
        name: value.detach().clone()
        for name, value in tiny_model.state_dict().items()
        if "running_" in name
    }
    pixels = torch.rand(2, 3, 3, 32, 32)
    with torch.no_grad():
        tiny_model.encode(pixels)
        tiny_model.predict(torch.randn(2, 3, 64), torch.zeros(2, 3, dtype=torch.long))
    after = {
        name: value.detach().clone()
        for name, value in tiny_model.state_dict().items()
        if "running_" in name
    }
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name])


def test_rollout_encodes_context_only_and_returns_predicted_suffix() -> None:
    class RecordingModel(LeWorldModel):
        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            self.seen_pixels = pixels
            return super().encode(pixels)

    model = RecordingModel(LeWMConfig.tiny()).eval()
    context_pixels = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    actions = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    with torch.no_grad():
        all_latents = model.rollout(context_pixels, actions)
        future_latents = model.rollout(
            context_pixels,
            actions,
            return_context=False,
        )

    assert model.seen_pixels.shape == context_pixels.shape
    assert all_latents.shape == (2, 6, 64)
    assert future_latents.shape == (2, 3, 64)
    torch.testing.assert_close(all_latents[:, 3:], future_latents)


def test_model_rejects_invalid_shapes_and_future_history(
    tiny_model: LeWorldModel,
) -> None:
    with pytest.raises(ValueError, match="three channels"):
        tiny_model.encode(torch.zeros(1, 2, 1, 32, 32, dtype=torch.uint8))
    with pytest.raises(ValueError, match="history_size"):
        tiny_model.predict(
            torch.zeros(1, 4, 64),
            torch.zeros(1, 4, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="actions must have shape"):
        tiny_model.compute_loss(
            torch.zeros(1, 4, 3, 32, 32, dtype=torch.uint8),
            torch.zeros(1, 2, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="divisible"):
        LeWMConfig.tiny(image_size=63)


def test_v31_rollout_applies_every_action_including_final():
    class TraceModel(LeWorldModel):
        def encode(self, pixels):
            return torch.zeros(pixels.shape[0], pixels.shape[1], 64)

        def predict(self, z, actions):
            return actions.argmax(-1).float().unsqueeze(-1).expand(-1, -1, 64)

    model = TraceModel(LeWMConfig.tiny()).eval()
    context = torch.zeros(1, 3, 3, 16, 16, dtype=torch.uint8)
    actions = torch.tensor([[0, 1, 2, 3, 4]])
    result = model.rollout(context, actions, return_context=False)
    torch.testing.assert_close(result[0, :, 0], torch.tensor([2.0, 3.0, 4.0]))
