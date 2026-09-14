from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dodge_native_game.variants.pixel_repr_ddqn.normalization_audit import audit, mode


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(encoder_dropout=0, encoder_attention_dropout=0)

        def head():
            return nn.Sequential(
                nn.Linear(2, 4), nn.BatchNorm1d(4), nn.GELU(), nn.Linear(4, 2)
            )

        self.projector = head()
        self.pred_projector = head()
        self.predictor = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))

    def _encode_tokens(self, pixels):
        return pixels, None

    def _apply_projector(self, projector, x):
        return projector(x.flatten(0, 1)).reshape(x.shape)

    def predict(self, z, actions):
        value = self.predictor(z + actions.unsqueeze(-1) / 9)
        return self._apply_projector(self.pred_projector, value)


def inputs():
    generator = torch.Generator().manual_seed(4)
    return torch.randn(8, 4, 2, generator=generator), torch.arange(24).reshape(8, 3) % 9


def test_mode_restores_buffers_modes_and_rng_on_error():
    model = Fixture().train()
    model.predictor.eval()
    flags = [m.training for m in model.modules()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    with (
        pytest.raises(RuntimeError, match="test"),
        mode(model, encoder_batch=True, dropout=True),
    ):
        x, a = inputs()
        z = model._apply_projector(model.projector, x)
        model.predict(z[:, :-1], a)
        raise RuntimeError("test")
    assert flags == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


def test_factorial_causality_and_train_only_calibration():
    torch.set_num_threads(2)
    model = Fixture().eval()
    x, a = inputs()
    first = audit(model, x, a, x[:3], a[:3], torch.arange(8))
    second = audit(model, x, a, x[:3] + 100, a[:3], torch.arange(8))
    assert first["original_state_unchanged"] and first["optimizer_updates"] == 0
    assert len(first["modes"]) == 8 and len(first["calibration"]) == 4
    assert first["future_substitution"]["False"]["context_max_abs"] == 0
    assert first["future_substitution"]["True"]["context_max_abs"] > 0
    for key in first["calibration"]:
        assert first["calibration"][key]["train"] == second["calibration"][key]["train"]
    assert all(p.grad is None for p in model.parameters())
