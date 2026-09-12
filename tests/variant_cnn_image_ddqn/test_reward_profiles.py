import json

import numpy as np
import pytest
import torch

from dodge_native_game.variants.cnn_image_ddqn.env import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.reward_profiles import (
    contract,
    weighted_components,
)
from dodge_native_game.variants.cnn_image_ddqn.run import train_run


def test_v72_component_weights_and_missing_native_data():
    terms = [3, -1, 2, 5, -4, -2]
    np.testing.assert_allclose(
        weighted_components(terms, "combined-v1"), [3, -25, 20, 5, -0.04, -0.2]
    )
    for invalid in [None, [1], [float("nan")] * 6]:
        with pytest.raises(ValueError, match="native reward terms"):
            weighted_components(invalid, "death-v1")
    with pytest.raises(ValueError):
        contract("unknown")


def test_v72_native_terminal_terms_and_unmodified_pixels():
    with CNNImageDDQNEnv(stack_size=4, observation_profile="native-rgb-v1") as env:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (12, 128, 128)
        total_deaths = 0
        for _ in range(1000):
            obs, reward, done, truncated, info = env.step(0)
            terms = info["native_reward_terms"]
            assert terms.shape == (6,) and np.isfinite(terms).all()
            assert terms[0] == reward
            assert terms[1] in (0, -1)
            total_deaths -= terms[1]
            if done or truncated:
                assert done and total_deaths == 1
                break
        else:
            pytest.fail("fixture did not reach terminal")


@pytest.mark.parametrize("profile", ["death-v1", "events-v1", "boundary-v1"])
def test_v72_native_reward_smoke_manifest_checkpoint_and_resume(tmp_path, profile):
    torch.set_num_threads(1)
    root = train_run(
        history_root=tmp_path,
        run_id=profile,
        steps=64,
        observation_profile="native-rgb-v1",
        stack_size=4,
        device="cpu",
        replay_capacity=128,
        warmup_steps=8,
        batch_size=4,
        update_every=4,
        target_sync_interval=2,
        eval_episodes=1,
        eval_steps=4,
        reward_profile=profile,
        checkpoint_steps=(32,),
    )
    report = json.loads((root / "report.json").read_text())
    expected = contract(profile)
    assert report["native_reward_contract"] == expected
    assert len(report["reward_component_totals"]) == 6
    checkpoint = root / "checkpoints/step-64.pt"
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert payload["native_reward_contract"] == expected
    with pytest.raises(ValueError, match="native reward contract"):
        train_run(
            history_root=tmp_path,
            run_id="bad-resume",
            steps=8,
            observation_profile="native-rgb-v1",
            device="cpu",
            resume_from=checkpoint,
            reward_profile="survival-v1",
        )
