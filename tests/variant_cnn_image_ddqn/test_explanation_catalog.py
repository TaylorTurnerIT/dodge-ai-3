import json

from dodge_native_game.variants.cnn_image_ddqn.explanation_catalog import (
    catalog,
    metrics_for_run,
)
from dodge_native_game.variants.cnn_image_ddqn.explanation_server import (
    ExplanationHTTPServer,
)


def run(root, name, mean, date="2026-09-12T01:00:00Z", state="completed"):
    path = root / name
    path.mkdir(parents=True)
    objects = {
        "manifest": {
            "created_at": date,
            "engine": {"observation_profile": "native-rgb-v1"},
        },
        "config": {"run": {"steps": 200000}},
        "status": {"state": state, "step": 200000},
        "report": {
            "evaluation": {
                "holdout": {"mean_survival_frames": mean, "mean_reward": 9999}
            }
        },
    }
    for key, value in objects.items():
        (path / f"{key}.json").write_text(json.dumps(value))
    return path


def test_best_mean_latest_and_smoke_exclusion(tmp_path):
    run(tmp_path, "best", 400)
    run(tmp_path, "new", 100, "2026-09-12T02:00:00Z")
    run(tmp_path, "smoke-new", 9999, "2026-09-12T03:00:00Z")
    run(tmp_path, "preflight", 9999)
    run(tmp_path, "missing", None)
    run(tmp_path, "nonfinite", float("nan"))
    rows = catalog(tmp_path)
    assert len(rows) == 4
    assert next(r["id"] for r in rows if r["best"]) == "best"
    assert next(r["id"] for r in rows if r["latest"]) == "new"
    assert next(r for r in rows if r["id"] == "nonfinite")["mean_frames"] is None
    assert all(r["replay_unavailable"] for r in rows)


def test_canonical_supersedes_mirror_and_active_is_unranked(tmp_path):
    root = tmp_path / "cnn-image-ddqn"
    run(tmp_path / "live-cnn-image-ddqn", "candidate", 9999)
    run(root, "candidate", 200)
    run(root, "active", 1000, state="running")
    rows = catalog(root)
    assert next(r for r in rows if r["best"])["mean_frames"] == 200


def test_run_selection_rejects_unknown_and_unsupported(tmp_path):
    import pytest

    path = run(tmp_path, "rgb", 100)
    server = ExplanationHTTPServer(("127.0.0.1", 0), None, path)
    server.selected_id = None
    try:
        with pytest.raises(ValueError, match="Unknown"):
            server.for_run("../outside")
        with pytest.raises(ValueError, match="native pixel"):
            server.for_run("rgb")
    finally:
        server.server_close()


def test_native_pixel_metrics_do_not_require_replay(tmp_path):
    path = run(tmp_path, "rgb", 200)
    (path / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 1000,
                "loss": 2,
                "pre_clip_grad_norm": 3,
                "completed_episodes": [
                    {"survival_frames": 100},
                    {"survival_frames": 200},
                ],
            }
        )
        + '\n{"step":2000,"loss":NaN}\n{"step":'
    )
    result = metrics_for_run(tmp_path, "rgb")
    assert result["run_id"] == "rgb"
    assert result["rows"][0]["training_mean_frames"] == 150
    assert result["rows"][0]["pre_clip_grad_norm"] == 3
    assert "loss" not in result["rows"][1]


def test_metrics_http_bypasses_incompatible_replay(tmp_path):
    import threading
    from types import SimpleNamespace
    from urllib.request import urlopen

    path = run(tmp_path, "rgb", 200)
    server = ExplanationHTTPServer(
        ("127.0.0.1", 0), SimpleNamespace(traces={"best": None}), path
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/api/metrics?run=rgb"
        ) as response:
            assert json.load(response)["run_id"] == "rgb"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
