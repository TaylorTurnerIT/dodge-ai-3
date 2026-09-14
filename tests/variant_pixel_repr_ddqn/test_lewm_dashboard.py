from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from dodge_native_game.variants.pixel_repr_ddqn.dashboard import (
    MAX_METRIC_ROWS,
    MAX_VISUALIZATIONS,
    DashboardData,
    create_server,
    safe_run_id,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@contextmanager
def _running_server(root: Path) -> Iterator[str]:
    server = create_server(root, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get_json(base_url: str, path: str) -> tuple[int, dict[str, object]]:
    with urlopen(base_url + path, timeout=2) as response:
        return response.status, json.loads(response.read())


def test_safe_run_id_rejects_path_components() -> None:
    assert safe_run_id("valid-run_01") == "valid-run_01"
    assert safe_run_id("../outside") is None
    assert safe_run_id("nested/run") is None
    assert safe_run_id("C:\\outside") is None
    assert safe_run_id("") is None


def test_run_index_skips_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "history"
    root.mkdir()
    (root / "valid-run").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_json(outside / "manifest.json", {"profile": "outside"})
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this filesystem")

    data = DashboardData(root)
    assert [run["run_id"] for run in data.list_runs()] == ["valid-run"]
    assert data.load_run("escape") is None


def test_http_assets_and_missing_run_data_are_safe(tmp_path: Path) -> None:
    root = tmp_path / "history"
    (root / "empty-run").mkdir(parents=True)
    with _running_server(root) as base_url:
        status, page = _get_json(base_url, "/api/run?run_id=empty-run")
        assert status == 200
        assert page["manifest"] == {}
        assert page["status"] == {}
        assert page["metrics"] == []
        assert page["visualization"] is None
        assert page["visualizations"] == []
        assert page["report"] is None

        with urlopen(base_url + "/", timeout=2) as response:
            html = response.read().decode("utf-8")
        assert response.status == 200
        assert "LeWM" in html
        assert "overflow: hidden" in html
        with urlopen(base_url + "/dashboard.html", timeout=2) as response:
            assert response.status == 200

        status, runs = _get_json(base_url, "/api/runs")
        assert status == 200
        assert runs["runs"][0]["run_id"] == "empty-run"


def test_metrics_and_visualizations_are_bounded_during_partial_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "history"
    run = root / "training-run"
    run.mkdir(parents=True)
    _write_json(
        run / "manifest.json",
        {
            "run_id": "training-run",
            "profile": "engineering-smoke",
            "architecture": "lewm-v3",
            "data_hash": "abc123",
        },
    )
    _write_json(
        run / "status.json",
        {"state": "running", "phase": "pretrain", "step": 344, "total_steps": 400},
    )
    with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(MAX_METRIC_ROWS + 44):
            handle.write(json.dumps({"step": step, "loss": 1.0 / (step + 1)}) + "\n")
        handle.write('{"step": 999')
        handle.write("\nthis is not json\n")
    _write_json(
        run / "visualizations.json",
        [{"step": step, "frames": []} for step in range(MAX_VISUALIZATIONS + 4)],
    )
    _write_json(run / "visualization.json", {"step": 999, "frames": []})
    _write_json(run / "report.json", {"quality": "diagnostic"})

    with _running_server(root) as base_url:
        status, payload = _get_json(base_url, "/api/run?run_id=training-run")
        assert status == 200
        metrics = payload["metrics"]
        assert len(metrics) == MAX_METRIC_ROWS
        assert metrics[0]["step"] == 44
        assert metrics[-1]["step"] == 343
        visualizations = payload["visualizations"]
        assert len(visualizations) <= MAX_VISUALIZATIONS
        assert payload["visualization"]["step"] == 999
        assert payload["report"]["quality"] == "diagnostic"


def test_http_rejects_unsafe_run_path(tmp_path: Path) -> None:
    root = tmp_path / "history"
    root.mkdir()
    with _running_server(root) as base_url:
        with pytest.raises(HTTPError) as error:
            urlopen(base_url + "/api/run?run_id=..%2Foutside", timeout=2)
        assert error.value.code == 404
