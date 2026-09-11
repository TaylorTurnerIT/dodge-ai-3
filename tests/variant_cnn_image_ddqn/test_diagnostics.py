from __future__ import annotations

import json

import numpy as np

from dodge_native_game.variants.cnn_image_ddqn import CNNImageDDQNEnv
from dodge_native_game.variants.cnn_image_ddqn.diagnostics import (
    action_balance_ratio,
    action_histogram,
    best_greedy_episode,
    decide_gate,
    effective_decay_steps,
    eval_seed_list,
    greedy_eval_rows,
    q_spread_stats,
    reward_mix_stats,
    td_error_stats,
)
from dodge_native_game.variants.cnn_image_ddqn.run import (
    _counterfactual_eval,
    train_run,
)


def test_reward_mix_flags_sparse_windows() -> None:
    stats = reward_mix_stats([0.0] * 19 + [1.0])
    assert stats["zero_share"] == 0.95
    assert stats["mean_nonzero"] == 1.0
    empty = reward_mix_stats([])
    assert empty["zero_share"] == 1.0


def test_td_and_q_spread_stats() -> None:
    td = td_error_stats([1.0, 2.0], [2.0, 4.0])
    assert td["mean"] == 1.5
    assert td["std"] > 0.0
    spread = q_spread_stats([[0.0, 10.0], [0.0, 10.0]])
    assert spread["gap"] == 10.0
    assert spread["mean"] == 5.0


def test_action_histogram_balance() -> None:
    hist = action_histogram([0, 0, 0, 1], 2)
    assert hist["counts"] == [3, 1]
    assert hist["least_used_action"] == 1
    assert action_balance_ratio([3, 1]) == 1 / 3
    assert action_balance_ratio([]) == 0.0


def test_effective_decay_caps_at_run_length() -> None:
    assert effective_decay_steps(1_000_000, 256) == 256
    assert effective_decay_steps(100, 10_000) == 100
    assert effective_decay_steps(0, 0) == 1


def test_eval_seeds_are_deterministic_and_clamped() -> None:
    seeds = eval_seed_list(42, 3, offset=10_000)
    assert seeds == [10_042, 10_043, 10_044]
    clamped = eval_seed_list(32_767, 2, offset=10_000)
    assert clamped == [32_767, 32_767]


def test_decide_gate_warns_on_smoke_and_lock_in() -> None:
    gate, reasons = decide_gate(
        steps=256,
        warmup_steps=32,
        target_sync_interval=100,
        decay_configured=1_000_000,
        decay_effective=256,
        balance_ratio=1.0,
        zero_share=0.5,
        train_mean=1.0,
        holdout_mean=1.0,
        updates=5,
    )
    assert gate == "warn"
    assert "bounded-smoke" in reasons
    assert "epsilon-decay-scaled" in reasons

    gate, reasons = decide_gate(
        steps=6000,
        warmup_steps=32,
        target_sync_interval=100,
        decay_configured=6000,
        decay_effective=6000,
        balance_ratio=0.01,
        zero_share=0.99,
        train_mean=10.0,
        holdout_mean=0.0,
        updates=100,
    )
    assert gate == "warn"
    assert "lock-in-risk" in reasons
    assert "sparse-reward" in reasons
    assert "generalization-gap" in reasons

    gate, _ = decide_gate(
        steps=6000,
        warmup_steps=32,
        target_sync_interval=100,
        decay_configured=1000,
        decay_effective=1000,
        balance_ratio=0.5,
        zero_share=0.5,
        train_mean=2.0,
        holdout_mean=2.0,
        updates=100,
    )
    assert gate == "pass"


class _Batch:
    def __init__(self, *, done: bool, flags: int, reward: float = 1.0) -> None:
        image = np.full((1, 84, 84), 7, dtype=np.uint8)
        self.collision_image = image
        self.rewards = np.asarray([reward], dtype=np.float32)
        self.done = np.asarray([done], dtype=np.bool_)
        self.frames = np.asarray([4], dtype=np.uint32)
        self.frames_advanced = np.asarray([4], dtype=np.uint32)
        self.seeds = np.asarray([3], dtype=np.uint32)
        self.event_flags = np.asarray([flags], dtype=np.uint32)
        self.modes = np.asarray([2], dtype=np.uint8)


class _Native:
    def __init__(self) -> None:
        self._steps = 0

    def reset_batch(self, seeds: object, *, startup: bool = False) -> _Batch:
        del seeds, startup
        self._steps = 0
        return _Batch(done=False, flags=1)

    def step_batch(self, actions: object) -> _Batch:
        del actions
        self._steps += 1
        if self._steps == 2:
            return _Batch(done=True, flags=4)
        return _Batch(done=False, flags=1)

    def close(self) -> None:
        return None


def _factory(**kwargs) -> CNNImageDDQNEnv:
    return CNNImageDDQNEnv(
        stack_size=kwargs.get("stack_size", 1),
        step_frames=kwargs.get("step_frames", 4),
        native_environment=_Native(),
    )


def test_train_run_logs_diagnostics_and_frozen_eval(tmp_path) -> None:
    root = train_run(
        history_root=tmp_path,
        run_id="diag-smoke",
        steps=6,
        stack_size=1,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        log_interval=2,
        eval_episodes=2,
        eval_steps=2,
        device="cpu",
        env_factory=_factory,  # type: ignore[arg-type]
    )
    rows = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows
    last = rows[-1]
    for key in (
        "td_error_mean",
        "td_error_std",
        "reward_zero_share",
        "action_balance",
        "least_used_action",
        "action_counts",
        "dead_units",
        "epsilon_decay_effective",
        "q_mean",
    ):
        assert key in last, f"missing metric {key}"
    assert last["epsilon_decay_effective"] == 6

    report = json.loads((root / "report.json").read_text())
    assert report["quality_gate"] == "warn"
    assert "gate_reasons" in report
    assert "bounded-smoke" in report["gate_reasons"]
    evaluation = report["evaluation"]
    assert evaluation["protocol"] == "frozen-train-holdout-v1"
    assert "inner" in evaluation and "holdout" in evaluation
    assert "counterfactual" in evaluation
    assert evaluation["inner"]["seeds"] != evaluation["holdout"]["seeds"]
    assert "diagnostics" in report

    config = json.loads((root / "config.json").read_text())
    assert config["exploration"]["epsilon_decay_steps_effective"] == 6
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["eval"]["protocol"] == "frozen-train-holdout-v1"


def test_greedy_rows_exclude_the_counterfactual_probe() -> None:
    evaluation = {
        "inner": {"seeds": [1], "rewards": [5.0], "survival_frames": [6]},
        "holdout": {"seeds": [2], "rewards": [7.0], "survival_frames": [8]},
        "counterfactual": {"seeds": [3], "rewards": [999.0]},
    }
    rows = greedy_eval_rows(evaluation)

    assert {row["source"] for row in rows} == {"inner", "holdout"}
    assert best_greedy_episode(evaluation) == {
        "source": "holdout",
        "seed": 2,
        "eval_reward": 7.0,
        "survival_frames": 8,
    }


def test_best_greedy_episode_breaks_ties_by_survival_then_seed() -> None:
    evaluation = {
        "inner": {
            "seeds": [12, 10, 11],
            "rewards": [4.0, 4.0, 4.0],
            "survival_frames": [9, 9, 5],
        }
    }
    best = best_greedy_episode(evaluation)

    assert best is not None
    assert best["seed"] == 10
    assert best_greedy_episode({}) is None
    assert best_greedy_episode({"inner": {"seeds": [], "rewards": []}}) is None


def test_report_names_the_best_final_episode_not_the_training_max(tmp_path) -> None:
    root = train_run(
        history_root=tmp_path,
        run_id="best-smoke",
        steps=6,
        stack_size=1,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        log_interval=2,
        eval_episodes=2,
        eval_steps=2,
        device="cpu",
        env_factory=_factory,  # type: ignore[arg-type]
    )
    report = json.loads((root / "report.json").read_text())
    evaluation = report["evaluation"]
    best = evaluation["best_eval_episode"]

    assert best is not None
    assert set(best) == {"source", "seed", "eval_reward", "survival_frames"}
    assert best["source"] in ("inner", "holdout")
    pooled = greedy_eval_rows(evaluation)
    assert best["eval_reward"] == max(row["eval_reward"] for row in pooled)
    assert report["final_metrics"]["best_eval_reward"] == best["eval_reward"]
    assert report["final_metrics"]["best_eval_episode"] == best


def test_counterfactual_forces_rare_action() -> None:
    from dodge_native_game.variants.cnn_image_ddqn.agent import DoubleDQNAgent

    agent = DoubleDQNAgent(num_actions=9, device="cpu", observation_shape=(1, 84, 84))
    env = _factory(stack_size=1)
    try:
        result = _counterfactual_eval(
            agent, env, seed=3, episodes=1, max_steps=2, forced_action=8
        )
        assert result["forced_action"] == 8
        assert len(result["rewards"]) == 1
    finally:
        env.close()
