from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
import torch

from dodge_native_game.variants.cnn_image_ddqn.explanation_server import (
    ExplanationHTTPServer,
    ExplanationService,
)


def _service():
    trace = SimpleNamespace(
        metadata={
            "seed": 20042,
            "frames": [{"index": 0, "action": 0, "q": [1.0, 0.0], "reward": 4.0}],
        },
        observations=(np.zeros((4, 84, 84), dtype=np.uint8),),
        game_pngs=(b"native-png",),
        model=None,
    )
    return ExplanationService({"best": trace})


def test_explanation_http_trace_image_and_invalid_index():
    server = ExplanationHTTPServer(("127.0.0.1", 0), _service())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/trace?episode=best") as response:
            trace = json.loads(response.read())
        assert trace["frames"][0]["action"] == 0
        with urlopen(base + trace["frames"][0]["game_url"]) as response:
            assert response.read() == b"native-png"
        with urlopen(base + "/api/episodes") as response:
            assert json.loads(response.read())[0]["seed"] == 20042
        for path in (
            "/api/game/best/-1.png",
            "/api/trace?episode=missing",
            "/api/explain/-1",
            "/api/explain/0?alternative=0",
            "/api/explain/0?frame=9",
            "/api/explain/0?channel=64",
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_explanation_worker_rejects_concurrent_inference():
    service = _service()
    with service.lock, pytest.raises(BlockingIOError):
        service.explain("best", 0, frame=None, alternative=1, channel=0)


def test_inactive_channel_has_no_strong_examples():
    from dodge_native_game.variants.cnn_image_ddqn.model import AtariCnnQNetwork

    service = _service()
    model = AtariCnnQNetwork(9).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    service.traces["best"].model = model
    assert service.examples("best", "channel", 63) == []
    assert service.examples("best", "feature", 511) == []


def test_explanation_cache_is_bounded_and_does_not_change_trace(monkeypatch):
    import dodge_native_game.variants.cnn_image_ddqn.explain as explain

    monkeypatch.setattr(explain, "explain_observation", lambda *args, **kwargs: {})
    service = _service()
    before = json.dumps(service.metadata("best"), sort_keys=True)
    for channel in range(20):
        service.explain("best", 0, frame=None, alternative=1, channel=channel)
    assert len(service.cache) == 16
    assert json.dumps(service.metadata("best"), sort_keys=True) == before


def test_run_selection_is_holdout_only_and_mismatch_remains_visible(
    tmp_path, monkeypatch
):
    import dodge_native_game.variants.cnn_image_ddqn.explanation_server as module

    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    (tmp_path / "report.json").write_text(json.dumps({"checkpoint": "policy.pt"}))
    (tmp_path / "config.json").write_text("{}")
    config = SimpleNamespace(
        checkpoint=checkpoint,
        eval_max_steps=512,
        step_frames=4,
        difficulty=2,
        patterns=True,
        powerups=True,
        evaluation={
            "inner": {"seeds": [10001], "rewards": [9999]},
            "holdout": {"seeds": [20001, 20002, 20003], "rewards": [10, 20, 30]},
        },
    )
    monkeypatch.setattr(module, "resolve_run_replay_config", lambda *args: config)
    seen = []

    def generate(_checkpoint, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            metadata={"seed": kwargs["seed"], "frames": [{"reward": 4.0}]}
        )

    monkeypatch.setattr(module, "generate_explanation_trace", generate)
    service = module.service_for_run(tmp_path)
    assert {row["seed"] for row in seen} == {20001, 20002, 20003}
    assert all(row["steps"] == 512 for row in seen)
    assert service.traces["best"].metadata["eval_reward"] == 30
    assert service.traces["best"].metadata["evaluation_reward_matches"] is False
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="name the checkpoint"):
        module.service_for_run(tmp_path)


@pytest.mark.parametrize("dueling", [True, False])
def test_v53_v55_wire_payload_is_signed_and_ablates_conv63(dueling):
    from dodge_native_game.variants.cnn_image_ddqn.explain import explain_observation
    from dodge_native_game.variants.cnn_image_ddqn.model import AtariCnnQNetwork

    model = AtariCnnQNetwork(9, dueling=dueling, input_channels=1).eval()
    observation = np.full((1, 84, 84), 128, dtype=np.uint8)
    with torch.inference_mode():
        tensor = torch.tensor(observation).unsqueeze(0)
        baseline = model(tensor)[0]
        conv = model.features(tensor.float() / 255)
        conv[:, 63] = 0
        h = model.shared(conv)
        if dueling:
            advantage = model.advantage_stream(h)
            expected = (
                model.value_stream(h) + advantage - advantage.mean(1, keepdim=True)
            )
        else:
            expected = model.q_head(h)
    # Kernel=1 is an identity edit: every signed sensitivity must be zero.
    result = explain_observation(
        model,
        observation,
        chosen=0,
        alternative=1,
        channel=63,
        kernel_size=1,
        stride=84,
    )
    np.testing.assert_allclose(result["ablation"]["q"], expected[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(result["q"], baseline.numpy(), atol=1e-6)
    np.testing.assert_allclose(result["decision_map"], 0, atol=1e-6)
    if dueling:
        np.testing.assert_allclose(result["value_map"], 0, atol=1e-6)
    else:
        assert result["value"] is None
        assert result["value_map"] is None
