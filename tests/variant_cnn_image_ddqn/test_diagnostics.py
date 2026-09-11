from __future__ import annotations

import json

import numpy as np
import pytest
import torch

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
from dodge_native_game.variants.cnn_image_ddqn.provenance import sha256_file
from dodge_native_game.variants.cnn_image_ddqn.run import (
    _checkpoint_global_step,
    _counterfactual_eval,
    build_parser,
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
    assert action_histogram([0, 0, 0], 2)["balance_ratio"] == 0.0
    assert action_balance_ratio([3, 1]) == 1 / 3
    assert action_balance_ratio([3, 1, 0]) == 0.0
    assert action_balance_ratio([]) == 0.0


def test_effective_decay_preserves_configured_schedule() -> None:
    assert effective_decay_steps(1_000_000, 256) == 1_000_000
    assert effective_decay_steps(100, 10_000) == 100
    assert effective_decay_steps(0, 0) == 1


def test_cli_accepts_explicit_epsilon_decay() -> None:
    args = build_parser().parse_args(
        ["--run-id", "smoke", "--epsilon-decay-steps", "32"]
    )

    assert args.epsilon_decay_steps == 32


def test_checkpoint_global_step_rejects_invalid_or_conflicting_counters() -> None:
    assert _checkpoint_global_step({"step": 7}) == 7
    assert _checkpoint_global_step({"step": 7, "global_environment_step": 7}) == 7
    with pytest.raises(ValueError, match="conflicts"):
        _checkpoint_global_step({"step": 7, "global_environment_step": 8})
    with pytest.raises(ValueError, match="non-negative integer"):
        _checkpoint_global_step({"step": -1})


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
        target_sync_count=0,
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
        target_sync_count=1,
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
        target_sync_count=1,
    )
    assert gate == "pass"


def test_decide_gate_uses_actual_target_sync_count() -> None:
    gate, reasons = decide_gate(
        steps=6000,
        warmup_steps=32,
        target_sync_interval=10_000,
        decay_configured=1000,
        decay_effective=1000,
        balance_ratio=0.5,
        zero_share=0.5,
        train_mean=2.0,
        holdout_mean=2.0,
        updates=100,
        target_sync_count=2,
    )
    assert gate == "pass"
    assert "target-never-synced" not in reasons

    gate, reasons = decide_gate(
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
        target_sync_count=0,
    )
    assert gate == "warn"
    assert "target-never-synced" in reasons


def test_decide_gate_rejects_censored_evaluation() -> None:
    gate, reasons = decide_gate(
        steps=6000,
        warmup_steps=32,
        target_sync_interval=100,
        decay_configured=6000,
        decay_effective=6000,
        balance_ratio=0.5,
        zero_share=0.5,
        train_mean=2.0,
        holdout_mean=2.0,
        updates=100,
        target_sync_count=1,
        evaluation_censored_share=0.25,
    )
    assert gate == "warn"
    assert "evaluation-censored" in reasons


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
    assert last["epsilon_decay_effective"] == 1_000_000
    completed = [
        episode
        for row in rows
        for episode in row.get("completed_episodes", [])
    ]
    assert [episode["episode"] for episode in completed] == [1, 2, 3]
    assert all(episode["episode_return"] == 2.0 for episode in completed)
    assert all(row["reward_scope"] == "completed-episode" for row in rows[1:])

    report = json.loads((root / "report.json").read_text())
    assert report["quality_gate"] == "warn"
    assert "gate_reasons" in report
    assert "bounded-smoke" in report["gate_reasons"]
    evaluation = report["evaluation"]
    assert evaluation["protocol"] == "frozen-train-holdout-v1"
    assert "inner" in evaluation and "holdout" in evaluation
    assert "counterfactual" in evaluation
    assert evaluation["inner"]["censored_share"] == 0.0
    assert evaluation["holdout"]["censored_share"] == 0.0
    assert evaluation["counterfactual"]["paired"]["protocol"] == (
        "paired-reward-delta-v1"
    )
    assert evaluation["inner"]["seeds"] != evaluation["holdout"]["seeds"]
    assert "diagnostics" in report

    config = json.loads((root / "config.json").read_text())
    assert config["exploration"]["epsilon_decay_steps_effective"] == 1_000_000
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["eval"]["protocol"] == "frozen-train-holdout-v1"
    assert manifest["initialization_id"] == "kaiming-relu-xavier-head-v1"


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


def test_resume_records_global_progress_and_phase_reset(tmp_path) -> None:
    parent_root = train_run(
        history_root=tmp_path,
        run_id="resume-parent",
        steps=4,
        stack_size=1,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        target_sync_interval=1,
        log_interval=1,
        eval_episodes=1,
        eval_steps=2,
        epsilon_decay_steps=10,
        device="cpu",
        env_factory=_factory,  # type: ignore[arg-type]
    )
    parent_checkpoint = parent_root / "checkpoints" / "step-4.pt"

    child_root = train_run(
        history_root=tmp_path,
        run_id="resume-child",
        steps=2,
        stack_size=1,
        batch_size=2,
        warmup_steps=2,
        update_every=2,
        target_sync_interval=1,
        log_interval=1,
        eval_episodes=1,
        eval_steps=2,
        epsilon_decay_steps=10,
        resume_from=parent_checkpoint,
        device="cpu",
        env_factory=_factory,  # type: ignore[arg-type]
    )

    manifest = json.loads((child_root / "manifest.json").read_text())
    assert manifest["resume_mode"] == "optimizer-state-with-fresh-replay-rng-env"
    assert manifest["parent_checkpoint"]["run_id"] == "resume-parent"
    assert manifest["parent_checkpoint"]["sha256"] == sha256_file(parent_checkpoint)
    assert set(manifest["resume_state"]["reset"]) == {
        "replay_buffer",
        "replay_rng",
        "action_rng",
        "numpy_global_rng",
        "torch_rng",
        "environment",
        "episode_accumulator",
        "diagnostic_windows",
        "segment_counters",
    }
    config = json.loads((child_root / "config.json").read_text())
    assert config["run"]["global_step_start"] == 4
    assert config["run"]["global_step_end"] == 6
    rows = [
        json.loads(line)
        for line in (child_root / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["step"] == 1
    assert rows[0]["global_environment_step"] == 5
    assert rows[0]["epsilon"] == 0.55
    assert rows[-1]["optimizer_step"] == 3
    assert rows[-1]["global_optimizer_step"] == 3
    assert rows[-1]["segment_optimizer_step"] == 1
    assert rows[-1]["target_sync_count"] == 3
    assert rows[-1]["segment_target_sync_count"] == 1
    report = json.loads((child_root / "report.json").read_text())
    assert report["diagnostics"]["optimizer_updates"] == 1
    assert report["diagnostics"]["global_optimizer_updates"] == 3
    child_checkpoint = child_root / "checkpoints" / "step-6.pt"
    payload = torch.load(child_checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_schema_version"] == 2
    assert payload["global_environment_step"] == 6
    assert payload["run_id"] == "resume-child"
    assert payload["initialization_id"] == "kaiming-relu-xavier-head-v1"
    assert payload["target_sync_count"] == 3


def test_resume_setup_failure_marks_artifact_failed(tmp_path) -> None:
    malformed = tmp_path / "malformed.pt"
    torch.save(
        {
            "variant_id": "cnn-image-ddqn",
            "step": 3,
            "observation_shape": [1, 84, 84],
            "num_actions": 9,
        },
        malformed,
    )

    with pytest.raises(KeyError, match="online_network"):
        train_run(
            history_root=tmp_path,
            run_id="broken-resume",
            steps=1,
            stack_size=1,
            resume_from=malformed,
            device="cpu",
            env_factory=_factory,  # type: ignore[arg-type]
        )

    status = json.loads(
        (tmp_path / "cnn-image-ddqn" / "broken-resume" / "status.json").read_text()
    )
    assert status["state"] == "failed"
    assert status["gate"] == "fail"


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
        assert result["paired"]["rows"][0]["seed"] == result["seeds"][0]
        assert result["reward_deltas"] == [0.0]
    finally:
        env.close()
