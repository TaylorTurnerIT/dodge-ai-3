from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge_native_game.variants.pixel_repr_ddqn import mpc_eval
from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import SurvivalProbe


def _write_episode(root: Path, split: str, index: int, die_at: int | None) -> dict:
    count = 128
    path = root / "episodes" / split / f"episode-{index:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((count + 1, 3, 128, 128), dtype=np.uint8)
    frames[:, 0] = (index * 40 + (80 if split == "validation" else 0)) % 256
    actions = np.zeros(count, dtype=np.int64)
    terminated = np.zeros(count, dtype=np.bool_)
    if die_at is not None:
        terminated[die_at] = True
    truncated = np.zeros(count, dtype=np.bool_)
    truncated[-1] = True
    with path.open("wb") as stream:
        np.savez_compressed(
            stream, pixels=frames, actions=actions,
            terminated=terminated, truncated=truncated,
        )
    return {
        "episode_id": f"{split}-{index:06d}",
        "recipe_family": "test",
        "seed": 22000 + index,
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decisions": count,
        "terminated": die_at is not None,
    }


def _make_dataset(root: Path) -> None:
    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import (
        PROBE_FORMAT,
    )

    episodes = {"train": [], "validation": []}
    for split, count in (("train", 2), ("validation", 1)):
        for index in range(count):
            episodes[split].append(
                _write_episode(root, split, index, die_at=127 if index == 0 else None)
            )
    manifest = {
        "format": PROBE_FORMAT,
        "splits": {split: len(rows) for split, rows in episodes.items()},
        "deaths": {
            split: sum(1 for row in rows if row["terminated"])
            for split, rows in episodes.items()
        },
        "episodes": episodes,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (root / "READY").write_text(PROBE_FORMAT + "\n")


def test_probe_fits_separable_latents() -> None:
    torch.manual_seed(0)
    probe = SurvivalProbe()
    alive = torch.randn(64, 192) - 2.0
    dead = torch.randn(64, 192) + 2.0
    features = torch.cat([alive, dead])
    labels = torch.cat([torch.zeros(64), torch.ones(64)])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        before = float(criterion(probe(features), labels))
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(probe(features), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = float(criterion(probe(features), labels))
    assert after < before / 4


def test_windows_label_death_within_horizon(tmp_path: Path) -> None:
    from dodge_native_game.variants.pixel_repr_ddqn import survival_probe as sp

    root = tmp_path / "dataset"
    _make_dataset(root)
    pixels, labels, provenance = sp._windows_with_labels(
        root, "train", max_windows=10000, seed=0
    )
    assert len(labels) == 2 * 125
    assert set(np.unique(labels)) == {0.0, 1.0}
    # Episode 0 dies at decision 64: windows ending within 16 steps flag it.
    first = [label for tag, label in zip(provenance, labels, strict=True)
             if tag.startswith("train-000000")]
    assert sum(first) == 16
    assert all(label == 0 for tag, label in zip(provenance, labels, strict=True)
               if tag.startswith("train-000001"))


class _StubModel(torch.nn.Module):
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, time = pixels.shape[:2]
        return torch.zeros(batch, time, 192)

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = z.clone()
        out[:, :, 0] = actions.float()
        return out


class _StubProbe(torch.nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, 0] - 4.0


def _background_frame() -> np.ndarray:
    frame = np.zeros((3, 128, 128), dtype=np.uint8)
    frame[:] = np.asarray(
        mpc_eval.PLAYFIELD_BACKGROUND_RGB, dtype=np.uint8
    ).reshape(3, 1, 1)
    return frame


class _ScriptedAdapter:
    def __init__(self, die_at: int | None = None) -> None:
        self._die_at = die_at
        self._step = 0
        self.actions: list[int] = []

    def reset(self, seed: int):
        del seed
        self._step = 0
        return _background_frame()

    def step(self, action: int):
        self.actions.append(int(action))
        if self._die_at is not None and self._step == self._die_at:
            return _background_frame(), 0.0, True, False
        self._step += 1
        return _background_frame(), 0.0, False, False

    def close(self) -> None:
        pass


def test_average_precision_null_when_degenerate() -> None:
    import json

    from dodge_native_game.variants.pixel_repr_ddqn.survival_probe import (
        _average_precision,
    )

    # No positives (or all positives): JSON null, never NaN.
    assert _average_precision(torch.zeros(8), torch.zeros(8)) is None
    assert _average_precision(torch.zeros(8), torch.ones(8)) is None
    perfect = _average_precision(
        torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0.0, 0.0, 1.0, 1.0])
    )
    assert perfect is not None and perfect > 0.99
    report = {"val_auprc": None}
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_run_episode_records_palette_abort_explicitly() -> None:
    def aborting_policy(history: torch.Tensor, past: list[int]) -> int:
        del history, past
        try:
            raise ValueError("pixels contain an RGB color outside configured palette")
        except ValueError as error:
            if "outside configured palette" not in str(error):
                raise
            raise mpc_eval._UnknownColorAbort from error

    result = mpc_eval.run_episode(
        lambda: _ScriptedAdapter(),
        0,
        aborting_policy,
        model=_StubModel(),
        device=torch.device("cpu"),
        max_decisions=32,
    )
    assert result["outcome"] == mpc_eval.ABORTED_UNKNOWN_COLOR
    assert result["survived"] == 0


def test_run_episode_propagates_unrelated_errors() -> None:
    def broken_policy(history: torch.Tensor, past: list[int]) -> int:
        del history, past
        raise ValueError("some other bug")

    with pytest.raises(ValueError, match="some other bug"):
        mpc_eval.run_episode(
            lambda: _ScriptedAdapter(),
            0,
            broken_policy,
            model=_StubModel(),
            device=torch.device("cpu"),
            max_decisions=32,
        )


def test_greedy_action_trims_history_to_action_trace() -> None:
    class _StrictHistoryModel(torch.nn.Module):
        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            batch, time = pixels.shape[:2]
            return torch.zeros(batch, time, 192)

        def predict(
            self, z: torch.Tensor, actions: torch.Tensor
        ) -> torch.Tensor:
            assert z.shape[1] == actions.shape[1] <= 3, (
                z.shape,
                actions.shape,
            )
            return z.clone()

    # Rollout buffer holds history_size+1 frames; predictor takes H.
    action, costs = mpc_eval._greedy_action(
        _StrictHistoryModel(),
        _StubProbe(),
        torch.zeros(1, 4, 3, 128, 128),
        [0, 0, 0],
        torch.device("cpu"),
    )
    assert len(costs) == 9
    assert action == 0


def test_greedy_action_picks_lowest_predicted_cost() -> None:
    action, costs = mpc_eval._greedy_action(
        _StubModel(), _StubProbe(),
        torch.zeros(1, 3, 3, 128, 128), [0, 0, 0],
        torch.device("cpu"),
    )
    assert costs[0] == pytest.approx(-4.0)
    assert costs[5] == pytest.approx(1.0)
    assert action == 0


def test_run_episode_counts_survived_and_outcome() -> None:
    model = _StubModel()
    device = torch.device("cpu")
    died = mpc_eval.run_episode(
        lambda: _ScriptedAdapter(die_at=10), 0,
        lambda history, past: (0, None),
        model=model, device=device, max_decisions=128,
    )
    assert died == {
        "seed": 0, "survived": 10, "outcome": "terminated", "masked_pixels": 0,
    }
    lived = mpc_eval.run_episode(
        lambda: _ScriptedAdapter(), 0,
        lambda history, past: (0, None),
        model=model, device=device, max_decisions=32,
    )
    assert lived == {
        "seed": 0, "survived": 32, "outcome": "truncated", "masked_pixels": 0,
    }


def test_mask_black_pixels_replaces_exact_black_only() -> None:
    stacked = np.full((2, 3, 4, 4), 41, dtype=np.uint8)
    stacked[:, 1] = 173
    stacked[:, 2] = 255
    stacked[0, :, 0, 0] = (0, 0, 0)
    stacked[1, :, 1, 1] = (0, 0, 0)
    stacked[0, :, 2, 2] = (0, 0, 1)  # near-black passes through
    stacked[1, :, 3, 3] = (255, 0, 0)  # foreign color passes through
    masked, count = mpc_eval.mask_black_pixels(stacked)
    assert count == 2
    assert masked.dtype == np.uint8
    assert tuple(masked[0, :, 0, 0]) == mpc_eval.PLAYFIELD_BACKGROUND_RGB
    assert tuple(masked[1, :, 1, 1]) == mpc_eval.PLAYFIELD_BACKGROUND_RGB
    assert tuple(masked[0, :, 2, 2]) == (0, 0, 1)
    assert tuple(masked[1, :, 3, 3]) == (255, 0, 0)
    # Input untouched; clean input returns a copy with zero count.
    assert tuple(stacked[0, :, 0, 0]) == (0, 0, 0)
    clean, clean_count = mpc_eval.mask_black_pixels(
        np.full((1, 3, 2, 2), 41, dtype=np.uint8)
    )
    assert clean_count == 0
    assert clean.shape == (1, 3, 2, 2)


def test_run_episode_masks_shake_frames_and_counts() -> None:
    class _ShakeAdapter(_ScriptedAdapter):
        def step(self, action: int):
            frame, reward, terminated, truncated = super().step(action)
            frame = frame.copy()
            frame[:, :, 0] = 0  # 1px shake strip, 128 black px
            return frame, reward, terminated, truncated

    seen_black: list[bool] = []

    def watching_policy(
        history: torch.Tensor, past: list[int]
    ) -> tuple[int, list[float] | None]:
        del past
        pixels = history[0].cpu().numpy()
        black = (pixels == 0).all(axis=1).any()
        seen_black.append(bool(black))
        return 0, None

    result = mpc_eval.run_episode(
        lambda: _ShakeAdapter(), 0, watching_policy,
        model=_StubModel(), device=torch.device("cpu"), max_decisions=8,
    )
    assert not any(seen_black)
    # Reset frame is clean; each of the 8 steps appends one 128px strip,
    # and the 4-frame history window accumulates them: steps see
    # 0,128,256,384, then 4 resident strips (512) for steps 4-7.
    assert result["masked_pixels"] == 128 + 256 + 384 + 4 * 512
    assert result["outcome"] == "truncated"


def test_plan_action_horizon1_matches_greedy() -> None:
    history = torch.zeros(1, 4, 3, 8, 8)
    expected = mpc_eval._greedy_action(
        _StubModel(), _StubProbe(), history, [0, 0, 0], torch.device("cpu")
    )
    actual = mpc_eval._plan_action(
        _StubModel(), _StubProbe(), history, [0, 0, 0], 1, torch.device("cpu")
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]


def test_plan_action_horizon2_minimizes_over_suffix() -> None:
    # _StubModel copies the last action index into latent dim 0 and
    # _StubProbe reads it back: sequence (a, b) costs ((a + b) / 2 - 4),
    # so V(a) = min_b = a / 2 - 4 with best action 0.
    action, values = mpc_eval._plan_action(
        _StubModel(),
        _StubProbe(),
        torch.zeros(1, 4, 3, 8, 8),
        [0, 0, 0],
        2,
        torch.device("cpu"),
    )
    assert action == 0
    assert values == pytest.approx([a / 2 - 4 for a in range(9)])


def test_plan_action_cost_modes_reduce_depth_scores() -> None:
    # Stub sequence (a, b) yields depth costs [a - 4, b - 4]: max picks
    # the worse moment (V(a) = a - 4), last sees only the suffix
    # (V flat at -4, tie broken to action 0).
    history = torch.zeros(1, 4, 3, 8, 8)
    device = torch.device("cpu")
    action_max, values_max = mpc_eval._plan_action(
        _StubModel(), _StubProbe(), history, [0, 0, 0], 2, device, "max"
    )
    assert action_max == 0
    assert values_max == pytest.approx([a - 4 for a in range(9)])
    action_last, values_last = mpc_eval._plan_action(
        _StubModel(), _StubProbe(), history, [0, 0, 0], 2, device, "last"
    )
    assert action_last == 0
    assert values_last == pytest.approx([-4.0] * 9)
    with pytest.raises(ValueError, match="cost mode"):
        mpc_eval._plan_action(
            _StubModel(), _StubProbe(), history, [0, 0, 0], 2, device,
            "median",
        )


def test_steering_registry_exposes_planners_and_baselines() -> None:
    policies = mpc_eval._steering_policies(
        _StubModel(), _StubProbe(), torch.device("cpu")
    )
    assert sorted(policies) == [
        "mpc", "mpc_h2", "mpc_h3", "mpc_h3_last", "mpc_h3_max", "mpc_h4",
        "neutral", "random",
    ]
    history = torch.zeros(1, 4, 3, 8, 8)
    for name, policy in policies.items():
        action, costs = policy(history, [0, 0, 0])
        assert action in range(9)
        if name in ("random", "neutral"):
            assert costs is None
        else:
            assert costs is not None and len(costs) == 9


def test_plan_action_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        mpc_eval._plan_action(
            _StubModel(),
            _StubProbe(),
            torch.zeros(1, 4, 3, 8, 8),
            [0, 0, 0],
            0,
            torch.device("cpu"),
        )


def test_run_episode_writes_trace_steps_and_frames(tmp_path: Path) -> None:
    from PIL import Image

    trace = tmp_path / "trace" / "mpc" / "scene-0"
    costs = [float(index) for index in range(9)]
    result = mpc_eval.run_episode(
        lambda: _ScriptedAdapter(), 0,
        lambda history, past: (3, costs),
        model=_StubModel(), device=torch.device("cpu"), max_decisions=3,
        trace_dir=trace,
    )
    assert result["outcome"] == "truncated"
    rows = (trace / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    first = json.loads(rows[0])
    assert first == {
        "step": 0, "action": 3, "costs": costs, "masked_pixels": 0,
    }
    for step in range(3):
        frame = np.asarray(Image.open(trace / f"frame_{step:03d}.png"))
        assert frame.shape == (128, 128, 3)
        assert (frame == np.asarray(mpc_eval.PLAYFIELD_BACKGROUND_RGB)).all()
    # Baselines record null costs, never zero-filled values.
    baseline_trace = tmp_path / "trace" / "neutral" / "scene-0"
    mpc_eval.run_episode(
        lambda: _ScriptedAdapter(), 0,
        lambda history, past: (0, None),
        model=_StubModel(), device=torch.device("cpu"), max_decisions=2,
        trace_dir=baseline_trace,
    )
    baseline_rows = (baseline_trace / "steps.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(baseline_rows) == 2
    assert json.loads(baseline_rows[0])["costs"] is None


def test_evaluate_rejects_probe_world_mismatch(tmp_path: Path) -> None:
    world = tmp_path / "world.pt"
    torch.save({"config": {}}, world)
    probe_path = tmp_path / "probe.pt"
    torch.save({"model": SurvivalProbe().state_dict(),
                "world_model_sha256": "b" * 64}, probe_path)
    with pytest.raises(ValueError, match="different world checkpoint"):
        mpc_eval.evaluate(
            world, probe_path, [], policies={"mpc": "mpc"}, device="cpu",
        )


def test_plan_mpc_eval_splits_and_seeds() -> None:
    scenarios = mpc_eval.plan_mpc_eval()
    labels = [label for label, _, _ in scenarios]
    assert len(scenarios) == 32
    assert sum(1 for label in labels if label.startswith("ordinary-")) == 16
    assert sum(1 for label in labels if label.startswith("novel-")) == 16
    seeds = [seed for _, _, seed in scenarios]
    assert len(set(seeds)) == 32
    assert min(seeds) >= 24000
    # Outside every training/validation/probe seed range.
    assert max(seeds) <= 32767
    cfgs = [cfg for _, cfg, _ in scenarios]
    assert all(cfg.invulnerable is False for cfg in cfgs)


def test_plan_mpc_eval_cohort_zero_unchanged() -> None:
    assert mpc_eval.plan_mpc_eval() == mpc_eval.plan_mpc_eval(cohorts=(0,))
    scenarios = mpc_eval.plan_mpc_eval()
    assert scenarios[0][0] == "ordinary-d1-all-pat-0"
    assert scenarios[0][2] == 24000
    assert scenarios[16][0] == "novel-d2-p3-0"
    assert scenarios[16][2] == 25016


def test_plan_mpc_eval_confirmation_cohort_is_fresh_but_matched() -> None:
    base = mpc_eval.plan_mpc_eval()
    both = mpc_eval.plan_mpc_eval(cohorts=(0, 1))
    assert len(both) == 64
    assert both[:32] == base
    fresh = both[32:]
    labels = [label for label, _, _ in fresh]
    assert all(label.endswith("-c1") for label in labels)
    assert [label[:-3] for label in labels] == [
        label for label, _, _ in base
    ]
    seeds = [seed for _, _, seed in fresh]
    base_seeds = {seed for _, _, seed in base}
    assert len(set(seeds)) == 32
    assert not (set(seeds) & base_seeds)
    assert seeds[:16] == list(range(26000, 26016))
    assert seeds[16:] == list(range(27016, 27032))
    assert max(seeds) <= 32767
    for (_, cfg, _), (_, fresh_cfg, _) in zip(base, fresh, strict=True):
        assert fresh_cfg.invulnerable is False
        assert (
            fresh_cfg.difficulty,
            fresh_cfg.enemy_mode,
            fresh_cfg.patterns_enabled,
            fresh_cfg.powerups_enabled,
            fresh_cfg.permanent_pattern,
        ) == (
            cfg.difficulty,
            cfg.enemy_mode,
            cfg.patterns_enabled,
            cfg.powerups_enabled,
            cfg.permanent_pattern,
        )
    with pytest.raises(ValueError, match="unknown scenario cohort"):
        mpc_eval.plan_mpc_eval(cohorts=(2,))
