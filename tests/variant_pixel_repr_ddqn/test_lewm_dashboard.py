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


def test_input_diagnostic_adapter_displays_retained_loss_and_evaluation():
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the dashboard JavaScript fixture")
    from dodge_native_game.variants.pixel_repr_ddqn import dashboard

    html = Path(dashboard.__file__).with_suffix(".html").read_text()
    helper = html.split("      function inputDiagnosticDetail(detail) {", 1)[1]
    helper = (
        "function inputDiagnosticDetail(detail) {"
        + helper.split("      function setConnection", 1)[0]
    )
    script = (
        helper
        + """
const assert=require('node:assert/strict');
for(const [arm,loss] of [['rgb',0.12],['palette',0.08]]) {
 const original={manifest:{experiment:'lewm-input-encoding-v1',diagnostic_only:true,
 input_arm:arm,data_sha256:'data'},metrics:[{step:8192,cls_loss:0.12,projected_loss:0.08}],
 report:{splits:{validation:{mse:0.003,changed_region_mse:0.14,training_mean_mse:0.013}}}};
 const output=inputDiagnosticDetail(original);
 assert.equal(output.metrics[0].loss,loss);
 assert.equal(output.metrics[0].validation_mse,0.003);
 assert.equal(output.manifest.current_frame_only,true);
 assert.equal(output.manifest.data_hash,'data');
 assert.equal(original.metrics[0].loss,undefined);
}
const world={manifest:{experiment:'lewm-input-encoding-v1'},metrics:[{loss:0.2}]};
assert.equal(inputDiagnosticDetail(world),world);
"""
    )
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


def test_future_run_snapshots_and_diff_strip_flow_through_dashboard(
    tmp_path: Path,
) -> None:
    root = tmp_path / "history"
    run = root / "future-run"
    run.mkdir(parents=True)
    _write_json(
        run / "manifest.json",
        {"experiment": "lewm-future-decode-v1", "data_sha256": "d" * 64},
    )
    _write_json(
        run / "status.json",
        {"state": "completed", "phase": "frozen diagnostic", "step": 2048},
    )
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 2048, "loss": 0.05, "predicted_loss": 0.05}) + "\n"
    )
    view = {
        "step": 2048,
        "scene": "validation-000000",
        "diagnostic_only": True,
        "frames": [
            {"label": "Observed current frame", "image": "data:image/png;base64,AAA"},
            {"label": "Observed next frame", "image": "data:image/png;base64,BBB"},
            {
                "label": "Decoded current frame (frozen patch readout)",
                "image": "data:image/png;base64,CCC",
            },
            {
                "label": "Decoded predicted next frame",
                "image": "data:image/png;base64,DDD",
            },
            {
                "label": "Pixel difference map (predicted vs observed next)",
                "image": "data:image/png;base64,EEE",
            },
        ],
        "metadata": {
            "label": "validation-000000 · next-frame prediction",
            "step": 2048,
        },
    }
    _write_json(run / "visualizations.json", [view])
    with _running_server(root) as base_url:
        status, payload = _get_json(base_url, "/api/run?run_id=future-run")
        assert status == 200
        snapshot = payload["visualizations"][0]
        assert [frame["label"] for frame in snapshot["frames"]][3] == (
            "Decoded predicted next frame"
        )
        assert payload["visualization"]["metadata"]["label"].startswith(
            "validation-000000"
        )
        with urlopen(base_url + "/", timeout=2) as response:
            html = response.read().decode()
        assert 'id="diff-strip"' in html
        assert "function renderDiff(snapshot)" in html
