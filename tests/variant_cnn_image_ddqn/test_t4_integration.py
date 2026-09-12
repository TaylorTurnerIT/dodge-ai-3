from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dodge_native_game.variants.cnn_image_ddqn import run

from .test_rgb_run import _RgbEnv, _TinyQNetwork


class _ScreenEnv(_RgbEnv):
    def __init__(self, profile):
        super().__init__([])
        self.observation_profile = profile
        size = 84 if profile == "collision-image-v1" else 128
        self.observation = np.zeros((4, size, size), dtype=np.uint8)


@pytest.mark.parametrize(
    "profile,n_step",
    [
        ("native-gray-v1", 1),
        ("collision-image-v1", 3),
    ],
)
def test_t4_profiles_train_and_preserve_checkpoint_contract(
    tmp_path, monkeypatch, profile, n_step
):
    monkeypatch.setattr(run, "AtariCnnQNetwork", _TinyQNetwork)
    root = run.train_run(
        history_root=tmp_path,
        run_id="screen",
        steps=8,
        observation_profile=profile,
        n_step=n_step,
        stack_size=4,
        replay_capacity=16,
        batch_size=2,
        warmup_steps=4,
        update_every=4,
        log_interval=4,
        eval_episodes=1,
        eval_steps=1,
        device="cpu",
        env_factory=lambda **kwargs: _ScreenEnv(kwargs["observation_profile"]),
    )
    payload = torch.load(root / "checkpoints/step-8.pt", weights_only=False)
    config = json.loads((root / "config.json").read_text())
    assert payload["observation_profile"] == profile
    assert payload["n_step"] == n_step
    assert payload["optimizer_steps"] == 2
    assert config["model"]["n_step"] == n_step
    metrics = [
        json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[-1]["replay_size"] == 8
    if profile == "native-gray-v1":
        assert payload["observation_shape"] == [4, 128, 128]
        assert config["replay"]["encoding"] == "native-palette4-frame-ring-v1"


def test_nstep_rejects_unsupported_combinations_before_artifacts(tmp_path):
    with pytest.raises(ValueError, match="requires collision"):
        run.train_run(
            history_root=tmp_path,
            run_id="bad",
            n_step=3,
            observation_profile="native-gray-v1",
        )
    assert not list(tmp_path.iterdir())
