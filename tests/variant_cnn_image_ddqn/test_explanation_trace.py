from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dodge_native_game.variants.cnn_image_ddqn.explanation_trace import (
    MAX_TRACE_STEPS,
    ExplanationTrace,
    generate_explanation_trace,
)
from dodge_native_game.variants.cnn_image_ddqn.model import AtariCnnQNetwork


class FakeExplanationEnv:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        self.seed = seed
        self.step_count = 0
        return self._observation(), self._info(frame=0, flags=0)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert 0 <= action < 9
        self.step_count += 1
        terminal = self.step_count >= 2
        return (
            self._observation(),
            float(self.step_count),
            terminal,
            False,
            self._info(
                frame=self.step_count * 4,
                flags=1 if not terminal else (4 | 16),
            ),
        )

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> np.ndarray:
        value = np.float32((self.seed + self.step_count) % 256) / np.float32(255)
        return np.full((1, 84, 84), value, dtype=np.float32)

    def _info(self, *, frame: int, flags: int) -> dict[str, object]:
        return {
            "native_frame": frame,
            "native_done": bool(flags & 16),
            "native_event_flags": flags,
            "native_score": float(frame),
            "native_shattered": frame // 4,
        }


class FakePixelLane:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.step_count = 0
        self.closed = False

    def reset_batch(self, seeds: object) -> SimpleNamespace:
        self.seed = int(np.asarray(seeds).reshape(-1)[0])
        self.step_count = 0
        return self._result()

    def step_batch(self, actions: object) -> SimpleNamespace:
        assert 0 <= int(np.asarray(actions).reshape(-1)[0]) < 9
        self.step_count += 1
        return self._result()

    def close(self) -> None:
        self.closed = True

    def _result(self) -> SimpleNamespace:
        return SimpleNamespace(
            frames=np.asarray([self.step_count * 4], dtype=np.uint32),
            done=np.asarray([self.step_count >= 2], dtype=np.bool_),
            pixels=np.full(
                (1, 128, 128), self.step_count, dtype=np.uint8
            ),
            player_positions=np.asarray(
                [[10.0 + self.step_count, 20.0]], dtype=np.float32
            ),
        )


def _write_checkpoint(
    tmp_path: Path,
    *,
    dueling: bool = True,
    stack_size: int = 1,
) -> tuple[Path, AtariCnnQNetwork]:
    run_dir = tmp_path / "demo-run"
    checkpoint = run_dir / "checkpoints" / "step-7.pt"
    checkpoint.parent.mkdir(parents=True)
    model = AtariCnnQNetwork(
        9,
        dueling=dueling,
        input_channels=stack_size,
    )
    torch.save(
        {
            "checkpoint_schema_version": 2,
            "variant_id": "cnn-image-ddqn",
            "step": 7,
            "global_environment_step": 7,
            "run_id": "demo-run",
            "observation_shape": [stack_size, 84, 84],
            "num_actions": 9,
            "online_network": model.state_dict(),
            "target_network": model.state_dict(),
        },
        checkpoint,
    )
    (run_dir / "config.json").write_text(
        json.dumps({"game": {"step_frames": 4}, "model": {"dueling": dueling}})
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "demo-run", "engine": {"schema": 1}})
    )
    return checkpoint, model


def _patch_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    import dodge_native_game.variants.cnn_image_ddqn.explanation_trace as module

    monkeypatch.setattr(module, "CNNImageDDQNEnv", FakeExplanationEnv)
    monkeypatch.setattr(module, "NativeBatchEnvironment", FakePixelLane)


def test_trace_rows_align_pre_action_inputs_and_post_action_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)
    _patch_lanes(monkeypatch)

    trace = generate_explanation_trace(checkpoint, seed=42, steps=4, device="cpu")

    assert isinstance(trace, ExplanationTrace)
    assert trace.frame_count == 2
    rows = trace.metadata["frames"]
    assert [row["index"] for row in rows] == [0, 1]
    assert [row["native_frame"] for row in rows] == [0, 4]
    assert [row["reward"] for row in rows] == [1.0, 2.0]
    assert [row["done"] for row in rows] == [False, True]
    assert [row["terminated"] for row in rows] == [False, True]
    assert [row["truncated"] for row in rows] == [False, False]
    assert [row["native_event_flags"] for row in rows] == [1, 20]
    assert rows[0]["native_events"] == {
        "enemy_spawn": True,
        "collision": False,
        "death": False,
        "pattern_active": False,
        "terminal": False,
    }
    assert rows[1]["native_events"]["death"] is True
    assert rows[1]["native_events"]["terminal"] is True
    assert [row["native_score"] for row in rows] == [4.0, 8.0]
    assert [row["native_shattered"] for row in rows] == [1.0, 2.0]
    assert trace.metadata["reset_position"] == [10.0, 20.0]
    assert trace.metadata["termination"]["reason"] == "native_terminal"
    assert trace.metadata["cap"]["reached"] is False
    assert trace.metadata["config"]["game"]["step_frames"] == 4
    assert trace.metadata["manifest"]["run_id"] == "demo-run"
    assert all(png.startswith(b"\x89PNG\r\n\x1a\n") for png in trace.game_pngs)
    assert trace.final_game_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert trace.final_game_png != trace.game_pngs[-1]

    assert all(observation.dtype == np.uint8 for observation in trace.observations)
    assert all(not observation.flags.writeable for observation in trace.observations)
    packed = base64.b64decode(trace.metadata["observations"]["input_base64"])
    decoded = np.frombuffer(packed, dtype=np.uint8).reshape(2, 1, 84, 84)
    np.testing.assert_array_equal(decoded, np.stack(trace.observations))


def test_trace_q_values_are_model_forward_values_and_head_is_decomposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)
    _patch_lanes(monkeypatch)

    trace = generate_explanation_trace(checkpoint, seed=42, steps=1, device="cpu")
    row = trace.metadata["frames"][0]
    with torch.inference_mode():
        output = trace.model(
            torch.tensor(trace.observations[0][None, ...], dtype=torch.uint8)
        )[0].cpu().numpy()
    np.testing.assert_array_equal(np.asarray(row["q"]), output)
    assert row["actionargmax"] == int(np.argmax(output))
    assert row["action"] == row["actionargmax"]
    assert row["value"] == pytest.approx(float(np.mean(output)), abs=1e-6)
    np.testing.assert_allclose(
        np.asarray(row["q"]),
        float(row["value"]) + np.asarray(row["centered_advantage"]),
        rtol=0,
        atol=1e-6,
    )
    assert float(np.sum(row["centered_advantage"])) == pytest.approx(0.0, abs=1e-6)


def test_plain_head_does_not_invent_value_or_centered_advantage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path, dueling=False)
    _patch_lanes(monkeypatch)

    trace = generate_explanation_trace(checkpoint, seed=42, steps=1, device="cpu")

    assert trace.model.dueling is False
    assert trace.metadata["frames"][0]["value"] is None
    assert trace.metadata["frames"][0]["centered_advantage"] is None


def test_missing_native_events_stay_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _write_checkpoint(tmp_path)

    class NoEventsEnv(FakeExplanationEnv):
        def _info(self, *, frame: int, flags: int) -> dict[str, object]:
            del flags
            return {
                "native_frame": frame,
                "native_done": False,
            }

    import dodge_native_game.variants.cnn_image_ddqn.explanation_trace as module

    monkeypatch.setattr(module, "CNNImageDDQNEnv", NoEventsEnv)
    monkeypatch.setattr(module, "NativeBatchEnvironment", FakePixelLane)
    trace = generate_explanation_trace(checkpoint, seed=42, steps=1, device="cpu")

    row = trace.metadata["frames"][0]
    assert row["events"] is None
    assert row["native_events"] is None
    assert row["native_event_flags"] is None
    assert row["native_score"] is None
    assert row["native_shattered"] is None


def test_trace_limit_is_explicit() -> None:
    assert MAX_TRACE_STEPS == 4096


def test_live_native_small_trace_smoke(tmp_path: Path) -> None:
    pytest.importorskip("dodge_native")
    checkpoint, _ = _write_checkpoint(tmp_path)

    trace = generate_explanation_trace(checkpoint, seed=42, steps=1, device="cpu")

    assert len(trace.observations) == len(trace.game_pngs) == 1
    assert trace.observations[0].shape == (1, 84, 84)
    assert trace.metadata["frames"][0]["native_frame"] >= 0
