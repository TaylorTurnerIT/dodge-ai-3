from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pytest

from dodge_native_game.variants.cnn_image_ddqn.native_replay import (
    NativeReplay,
    ReplayFrame,
    _infer_dueling,
)
from dodge_native_game.variants.cnn_image_ddqn.replay_server import (
    ReplayServer,
    ReplayStore,
)
from dodge_native_game.variants.cnn_image_ddqn.run_replay import (
    list_eval_episodes,
    resolve_run_replay_config,
    run_context,
    run_training_curves,
    select_comparison_episodes,
)


def _evaluation() -> dict:
    return {
        "inner": {
            "seeds": [10042, 10043, 10044],
            "rewards": [10.0, 90.0, 50.0],
            "survival_frames": [20, 200, 100],
        },
        "holdout": {
            "seeds": [20042, 20043],
            "rewards": [30.0, 70.0],
            "survival_frames": [60, 140],
        },
        "counterfactual": {
            "seeds": [30042],
            "rewards": [999.0],
        },
    }


def test_select_picks_best_median_worst_from_greedy_policy_only() -> None:
    picks = select_comparison_episodes(_evaluation())

    assert picks["best"].seed == 10043
    assert picks["best"].eval_reward == 90.0
    assert picks["worst"].seed == 10042
    # Sorted rewards: 10, 30, 50, 70, 90 -> lower-middle of 5 is 50.
    assert picks["median"].eval_reward == 50.0
    assert picks["median"].seed == 10044
    # The off-policy forced-action probe never wins a comparison slot.
    assert all(pick.source != "counterfactual" for pick in picks.values())


def test_select_breaks_reward_ties_by_survival_then_seed() -> None:
    evaluation = {
        "inner": {
            "seeds": [10043, 10042, 10044],
            "rewards": [50.0, 50.0, 50.0],
            "survival_frames": [100, 200, 200],
        }
    }
    picks = select_comparison_episodes(evaluation)

    assert picks["best"].seed == 10042
    assert picks["worst"].seed == 10043


def test_select_supports_legacy_flat_evaluation() -> None:
    evaluation = {"seeds": [7, 8], "rewards": [1.0, 2.0]}
    picks = select_comparison_episodes(evaluation)

    assert picks["best"].seed == 8
    assert picks["worst"].seed == 7


def test_select_rejects_evaluation_without_episodes() -> None:
    with pytest.raises(ValueError):
        select_comparison_episodes({"inner": {"seeds": [], "rewards": []}})


def test_list_eval_episodes_keeps_counterfactual_visible() -> None:
    rows = list_eval_episodes(_evaluation())

    assert {row["source"] for row in rows} == {"inner", "holdout", "counterfactual"}


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "cnn-image-ddqn" / "demo-run"
    (run_dir / "checkpoints").mkdir(parents=True)
    checkpoint = run_dir / "checkpoints" / "step-64.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "game": {"step_frames": 4},
                "dashboard_game_controls": {
                    "difficulty": 2,
                    "patterns": True,
                    "powerups": True,
                },
                "evaluation": {"max_steps_per_episode": 64},
                "run": {"steps": 64, "seed": 42},
            }
        )
    )
    evaluation: dict = {
        "inner": {
            "seeds": [10042, 10043],
            "rewards": [5.0, 9.0],
            "survival_frames": [10, 18],
        },
        "holdout": {"seeds": [20042], "rewards": [6.0], "survival_frames": [12]},
    }
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation))
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "checkpoint": "checkpoints/step-64.pt",
                "quality_gate": "warn",
                "evaluation": evaluation,
            }
        )
    )
    (run_dir / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "reward": float(step), "survival_frames": step})
            + "\n"
            for step in range(1, 501)
        )
    )
    return run_dir


def test_resolve_run_replay_config_reads_run_artifacts(tmp_path: Path) -> None:
    _write_run(tmp_path)

    layout = resolve_run_replay_config(tmp_path, "demo-run")

    assert layout.run_id == "demo-run"
    assert layout.checkpoint.name == "step-64.pt"
    assert layout.step_frames == 4
    assert layout.difficulty == 2
    assert layout.patterns is True
    assert layout.eval_max_steps == 64


def test_infer_dueling_distinguishes_plain_head_checkpoints() -> None:
    assert _infer_dueling({"online_network": {"q_head.weight": 1}}) is False
    assert _infer_dueling({"online_network": {"value_stream.weight": 1}}) is True


def _labeled_replay(label: str, seed: int) -> NativeReplay:
    frame = ReplayFrame(
        index=0,
        native_frame=0,
        action=None,
        reward=0.0,
        done=False,
        png=b"\x89PNG\r\n\x1a\n",
        collision_png=b"\x89PNG\r\n\x1a\n",
    )
    return NativeReplay(
        checkpoint=Path("step-64.pt"),
        seed=seed,
        step_frames=4,
        created_at="2026-09-11T00:00:00Z",
        frames=(frame,),
        run_id="demo-run",
        label=label,
        source="inner",
        eval_reward=float(seed),
    )


def test_compare_page_serves_labeled_replays_side_by_side() -> None:
    store = ReplayStore()
    server = ReplayServer(store=store, host="127.0.0.1", port=0)
    server.start()
    try:
        first = store.add(_labeled_replay("best", 1))
        second = store.add(_labeled_replay("worst", 2))
        with urlopen(server.url_for_compare([first, second])) as response:
            page = response.read()
        with urlopen(f"{server.base_url}/api/replay/{first}") as response:
            metadata = json.loads(response.read())
    finally:
        server.close()

    assert b"Replay Comparison" in page
    assert b"Run context" in page
    assert metadata["run_id"] == "demo-run"
    assert metadata["label"] == "best"
    assert metadata["total_reward"] == 0.0


def test_training_curves_downsample_and_keep_the_exact_max(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    curves = run_training_curves(run_dir)

    assert curves["rows"] == 500
    assert len(curves["points"]) <= 240
    assert curves["points"][0]["step"] == 1.0
    assert curves["points"][-1]["step"] >= 490.0
    assert curves["train_max"] == {"reward": 500.0, "step": 500.0}


def test_run_context_names_the_final_best_not_the_training_max(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)

    context = run_context(tmp_path, "demo-run")

    assert context["run_id"] == "demo-run"
    assert context["quality_gate"] == "warn"
    assert context["best_eval_episode"] == {
        "source": "inner",
        "seed": 10043,
        "eval_reward": 9.0,
        "survival_frames": 18,
    }
    assert context["train_max"] == {"reward": 500.0, "step": 500.0}
    assert set(context["picks"]) == {"best", "median", "worst"}
    assert set(context["eval"]) == {"inner", "holdout"}


def test_curves_route_serves_context_and_rejects_unknown_runs(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)
    store = ReplayStore()
    server = ReplayServer(
        store=store, host="127.0.0.1", port=0, history_root=tmp_path
    )
    server.start()
    try:
        with urlopen(f"{server.base_url}/api/run/demo-run/curves") as response:
            context = json.loads(response.read())
        error = None
        try:
            urlopen(f"{server.base_url}/api/run/no-such-run/curves")
        except Exception as exc:  # HTTPError 404
            error = exc
    finally:
        server.close()

    assert context["best_eval_episode"]["seed"] == 10043
    assert len(context["train_curve"]) <= 240
    assert error is not None
