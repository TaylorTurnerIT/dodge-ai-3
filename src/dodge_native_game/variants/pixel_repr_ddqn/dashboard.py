"""Small read-only browser dashboard for the LeWM representation runs.

The trainer owns the files under the history root.  This module only reads
those artifacts and serves a bounded JSON view of them; it never opens a
trainer process, writes a run, or mutates an artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

DEFAULT_HISTORY_ROOT: Final = Path("history/dodge/gymnasium/pixel-repr-ddqn")
DEFAULT_HOST: Final = "0.0.0.0"
DEFAULT_PORT: Final = 8790
MAX_METRIC_ROWS: Final = 300
MAX_RUNS: Final = 24
MAX_VISUALIZATIONS: Final = 16
MAX_RUN_ID_LENGTH: Final = 128
MAX_TRACE_STEPS: Final = 512

# A run id is used as one path component.  Rejection is preferable to
# normalising a path supplied by a browser because normalisation can hide a
# traversal attempt in logs and callers.
RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def safe_run_id(value: str | None) -> str | None:
    """Return *value* when it is a safe single run-directory name."""

    if not isinstance(value, str) or not value or len(value) > MAX_RUN_ID_LENGTH:
        return None
    if value in {".", ".."} or not RUN_ID_RE.fullmatch(value):
        return None
    return value


def _inside(path: Path, parent: Path) -> bool:
    """Whether a resolved path is contained by a resolved directory."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _read_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object for partial artifacts."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_optional_mapping(path: Path) -> dict[str, Any] | None:
    """Read an optional JSON object, preserving the missing state for clients."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_visualizations(path: Path) -> list[dict[str, Any]]:
    """Read a bounded validation-snapshot list while a writer may be active."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, ValueError):
        return []
    if isinstance(value, dict):
        value = value.get("visualizations", [])
    if not isinstance(value, list):
        return []
    snapshots = [item for item in value if isinstance(item, dict)]
    return snapshots[-MAX_VISUALIZATIONS:]


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    """Read the last bounded set of valid JSONL records.

    A trainer can be writing the final line while the dashboard polls.  A
    malformed or half-written line is therefore ignored independently of the
    other records in the file.
    """

    rows: deque[dict[str, Any]] = deque(maxlen=MAX_METRIC_ROWS)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, UnicodeError):
        return []
    return list(rows)


def _json_safe(value: Any) -> Any:
    """Keep an artifact with non-finite numeric probes JSON-serializable."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


class DashboardData:
    """Safe, read-only artifact access for the HTTP handlers."""

    def __init__(self, history_root: str | Path) -> None:
        self.history_root = Path(history_root).expanduser().resolve(strict=False)
        html_path = Path(__file__).with_name("dashboard.html")
        try:
            self.html = html_path.read_bytes()
        except OSError:
            # A source checkout always carries the page.  Keeping a tiny
            # fallback makes a broken package-data install report a useful page
            # instead of crashing the HTTP server during construction.
            self.html = (
                b"<!doctype html><title>LeWM dashboard</title>"
                b"<p>dashboard.html is unavailable.</p>"
            )

    def _run_dir(self, run_id: str | None) -> tuple[str, Path] | None:
        clean_id = safe_run_id(run_id)
        if clean_id is None:
            return None
        candidate = self.history_root / clean_id
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        if not _inside(resolved, self.history_root):
            return None
        try:
            if not resolved.is_dir():
                return None
        except (OSError, RuntimeError):
            return None
        return clean_id, resolved

    @staticmethod
    def _artifact(run_dir: Path, name: str) -> Path | None:
        """Resolve a known artifact while refusing file-level symlink escapes."""

        candidate = run_dir / name
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        if not _inside(resolved, run_dir):
            return None
        return resolved

    def load_run(self, run_id: str | None) -> dict[str, Any] | None:
        """Load one run, or ``None`` when the requested path is not safe."""

        found = self._run_dir(run_id)
        if found is None:
            return None
        clean_id, run_dir = found
        manifest_path = self._artifact(run_dir, "manifest.json")
        status_path = self._artifact(run_dir, "status.json")
        metrics_path = self._artifact(run_dir, "metrics.jsonl")
        visualization_path = self._artifact(run_dir, "visualization.json")
        visualizations_path = self._artifact(run_dir, "visualizations.json")
        report_path = self._artifact(run_dir, "report.json")
        visualization = (
            _read_optional_mapping(visualization_path)
            if visualization_path
            else None
        )
        visualizations = (
            _read_visualizations(visualizations_path)
            if visualizations_path
            else []
        )
        if visualization is not None and visualization not in visualizations:
            visualizations.append(visualization)
            visualizations = visualizations[-MAX_VISUALIZATIONS:]
        elif visualization is None and visualizations:
            visualization = visualizations[-1]
        return {
            "run_id": clean_id,
            "manifest": _read_mapping(manifest_path) if manifest_path else {},
            "status": _read_mapping(status_path) if status_path else {},
            "metrics": _read_metrics(metrics_path) if metrics_path else [],
            "visualization": visualization,
            "visualizations": visualizations,
            "report": (
                _read_optional_mapping(report_path) if report_path else None
            ),
        }

    def _trace_entry(
        self, run_id: str | None, *segments: str
    ) -> Path | None:
        """Resolve a trace artifact below a run while refusing escapes."""

        found = self._run_dir(run_id)
        if found is None:
            return None
        _, run_dir = found
        cleaned: list[str] = []
        for segment in segments:
            clean = safe_run_id(segment)
            if clean is None:
                return None
            cleaned.append(clean)
        candidate = run_dir.joinpath("trace", *cleaned)
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        if not _inside(resolved, run_dir):
            return None
        return resolved

    def load_trace_index(self, run_id: str | None) -> dict[str, Any] | None:
        """List traced policies/scenarios plus per-episode outcomes."""

        found = self._run_dir(run_id)
        if found is None:
            return None
        _, run_dir = found
        trace_root = self._artifact(run_dir, "trace")
        policies: dict[str, list[str]] = {}
        if trace_root is not None:
            try:
                entries = sorted(trace_root.iterdir())
            except OSError:
                entries = []
            for policy_dir in entries:
                if not policy_dir.is_dir():
                    continue
                if safe_run_id(policy_dir.name) is None:
                    continue
                try:
                    scenarios = sorted(
                        entry.name
                        for entry in policy_dir.iterdir()
                        if entry.is_dir() and safe_run_id(entry.name)
                    )
                except OSError:
                    scenarios = []
                if scenarios:
                    policies[policy_dir.name] = scenarios
        episodes: dict[str, dict[str, Any]] = {}
        report_path = self._artifact(run_dir, "mpc-report.json")
        report = _read_optional_mapping(report_path) if report_path else None
        rows = report.get("episodes", []) if report else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = row.get("scenario")
                if not isinstance(label, str):
                    continue
                for key, value in row.items():
                    if not key.endswith("_outcome"):
                        continue
                    policy = key[: -len("_outcome")]
                    survived = row.get(policy)
                    episodes.setdefault(policy, {})[label] = {
                        "seed": row.get("seed"),
                        "survived": survived
                        if isinstance(survived, int)
                        else None,
                        "outcome": value if isinstance(value, str) else None,
                    }
        return {"policies": policies, "episodes": episodes}

    def load_trace(
        self, run_id: str | None, policy: str | None, scenario: str | None
    ) -> dict[str, Any] | None:
        """Load one episode trace: per-decision values plus frame count."""

        if policy is None or scenario is None:
            return None
        steps_path = self._trace_entry(run_id, policy, scenario, "steps.jsonl")
        if steps_path is None:
            return None
        steps: list[dict[str, Any]] = []
        try:
            with steps_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(value, dict):
                        continue
                    costs = value.get("costs")
                    if costs is not None and (
                        not isinstance(costs, list) or len(costs) != 9
                    ):
                        continue
                    steps.append(
                        {
                            "step": value.get("step"),
                            "action": value.get("action"),
                            "costs": costs,
                            "masked_pixels": value.get("masked_pixels", 0),
                        }
                    )
                    if len(steps) >= MAX_TRACE_STEPS:
                        break
        except (OSError, UnicodeError):
            return None
        if not steps:
            return None
        return {"steps": steps, "frames": len(steps)}

    def load_trace_frame(
        self,
        run_id: str | None,
        policy: str | None,
        scenario: str | None,
        step: int | None,
    ) -> bytes | None:
        """Read one raw trace frame PNG; ``None`` when unsafe or missing."""

        if policy is None or scenario is None:
            return None
        if not isinstance(step, int) or step < 0 or step >= MAX_TRACE_STEPS:
            return None
        frame_path = self._trace_entry(
            run_id, policy, scenario, f"frame_{step:03d}.png"
        )
        if frame_path is None:
            return None
        try:
            body = frame_path.read_bytes()
        except OSError:
            return None
        if len(body) > 4 * 1024**2 or not body.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        return body

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a small latest-first run index from safe child directories."""

        try:
            entries = list(self.history_root.iterdir())
        except (OSError, ValueError):
            return []

        def mtime(entry: Path) -> float:
            try:
                return entry.stat().st_mtime
            except OSError:
                return 0.0

        runs: list[dict[str, Any]] = []
        for entry in sorted(entries, key=mtime, reverse=True):
            clean_id = safe_run_id(entry.name)
            if clean_id is None:
                continue
            loaded = self.load_run(clean_id)
            if loaded is None:
                continue
            manifest = loaded["manifest"]
            status = loaded["status"]
            runs.append(
                {
                    "run_id": clean_id,
                    "profile": manifest.get("profile"),
                    "architecture": manifest.get("architecture"),
                    "data_hash": manifest.get("data_hash"),
                    "state": status.get("state", "unknown"),
                    "phase": status.get("phase"),
                    "step": status.get("step"),
                    "total_steps": status.get("total_steps"),
                    "message": status.get("message"),
                    "updated_at": status.get("updated_at"),
                }
            )
            if len(runs) >= MAX_RUNS:
                break
        return runs


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP surface for the page and the two read-only JSON endpoints."""

    server: DashboardServer

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _not_found(self, message: str = "not found") -> None:
        self._send_json({"error": message}, HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        self._dispatch()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        self._send_json({"error": "read-only dashboard"}, HTTPStatus.METHOD_NOT_ALLOWED)

    def _dispatch(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        data = self.server.dashboard
        if path in {"/", "/dashboard.html"}:
            self._send_bytes(data.html, "text/html; charset=utf-8")
            return
        if path == "/api/runs":
            self._send_json({"runs": data.list_runs()})
            return
        if path == "/api/run":
            query = parse_qs(parsed.query, keep_blank_values=True)
            values = query.get("run_id", [])
            if not values or not values[0]:
                self._send_json(
                    {"error": "run_id is required"}, HTTPStatus.BAD_REQUEST
                )
                return
            loaded = data.load_run(values[0])
            if loaded is None:
                self._not_found("run not found")
                return
            self._send_json(loaded)
            return
        if path == "/api/traces":
            query = parse_qs(parsed.query, keep_blank_values=True)
            values = query.get("run_id", [])
            if not values or not values[0]:
                self._send_json(
                    {"error": "run_id is required"}, HTTPStatus.BAD_REQUEST
                )
                return
            index = data.load_trace_index(values[0])
            if index is None:
                self._not_found("run not found")
                return
            self._send_json(index)
            return
        if path == "/api/trace":
            query = parse_qs(parsed.query, keep_blank_values=True)
            run_id = (query.get("run_id", [""]) or [""])[0]
            policy = (query.get("policy", [""]) or [""])[0]
            scenario = (query.get("scenario", [""]) or [""])[0]
            if not run_id or not policy or not scenario:
                self._send_json(
                    {"error": "run_id, policy, and scenario are required"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            trace = data.load_trace(run_id, policy, scenario)
            if trace is None:
                self._not_found("trace not found")
                return
            self._send_json(trace)
            return
        if path == "/api/trace-frame":
            query = parse_qs(parsed.query, keep_blank_values=True)
            run_id = (query.get("run_id", [""]) or [""])[0]
            policy = (query.get("policy", [""]) or [""])[0]
            scenario = (query.get("scenario", [""]) or [""])[0]
            raw_step = (query.get("step", [""]) or [""])[0]
            try:
                step = int(raw_step)
            except (TypeError, ValueError):
                step = -1
            if not run_id or not policy or not scenario or step < 0:
                self._send_json(
                    {"error": "run_id, policy, scenario, step are required"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            body = data.load_trace_frame(run_id, policy, scenario, step)
            if body is None:
                self._not_found("frame not found")
                return
            self._send_bytes(body, "image/png")
            return
        self._not_found()

    def log_message(self, format: str, *args: object) -> None:
        # Keep the CLI quiet during two-second polling.  Errors still reach the
        # client as HTTP status codes and can be inspected with a browser.
        return None


class DashboardServer(ThreadingHTTPServer):
    """Threaded server carrying the immutable history-root reader."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        history_root: str | Path,
    ) -> None:
        self.dashboard = DashboardData(history_root)
        super().__init__(server_address, DashboardRequestHandler)


def create_server(
    history_root: str | Path = DEFAULT_HISTORY_ROOT,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> DashboardServer:
    """Build a server for tests or an embedding process without starting it."""

    return DashboardServer((host, port), history_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the LeWM artifact dashboard")
    parser.add_argument(
        "--history-root",
        type=Path,
        default=DEFAULT_HISTORY_ROOT,
        help="run artifact directory (default: %(default)s)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    server = create_server(args.history_root, host=args.host, port=args.port)
    print(f"LeWM dashboard: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
