"""Read-only, checkpoint-backed explanation replay; no training controls."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import numpy as np
import torch

from .explanation_catalog import catalog, metrics_for_run
from .explanation_trace import ExplanationTrace, generate_explanation_trace
from .replay_server import tailscale_ipv4
from .run_replay import resolve_run_replay_config, select_comparison_episodes


class ExplanationService:
    """Serve immutable traces; serialize and bound expensive interventions."""

    def __init__(self, traces: dict[str, ExplanationTrace]) -> None:
        if not traces or len(traces) > 3:
            raise ValueError("one to three explanation traces required")
        self.traces = traces
        self.lock = threading.Lock()
        self.cache: OrderedDict[tuple, dict] = OrderedDict()

    def trace(self, episode: str) -> ExplanationTrace:
        return self.traces[episode]

    def episodes(self) -> list[dict]:
        return [
            {
                "id": key,
                "label": key,
                "seed": trace.metadata["seed"],
                "eval_reward": trace.metadata.get("eval_reward"),
            }
            for key, trace in self.traces.items()
        ]

    def metadata(self, episode: str) -> dict:
        trace = self.trace(episode)
        return {
            **{k: v for k, v in trace.metadata.items() if k != "observations"},
            "step_frames": trace.metadata.get("native_config", {}).get("step_frames"),
            "total_reward": sum(row["reward"] for row in trace.metadata["frames"]),
            "final_game_url": f"/api/final-game/{episode}"
            if getattr(trace, "final_game_png", None)
            else None,
            "frames": [
                {
                    **row,
                    "game_url": f"/api/game/{episode}/{i}.png",
                    "input": None,
                    "input_shape": list(trace.observations[i].shape),
                    "input_base64": base64.b64encode(
                        trace.observations[i].tobytes()
                    ).decode("ascii"),
                }
                for i, row in enumerate(trace.metadata["frames"])
            ],
        }

    def explain(
        self,
        episode: str,
        index: int,
        *,
        frame: int | None,
        alternative: int | None,
        channel: int,
        radius: int = 6,
    ) -> dict:
        # Imports kept here so CLI --help does not initialize explanation work.
        from .explain import explain_observation

        trace = self.trace(episode)
        if not 0 <= index < len(trace.observations):
            raise ValueError("decision index out of range")
        if not 0 <= channel < 64 or radius not in (3, 6, 12):
            raise ValueError("channel or blur radius out of range")
        observation = trace.observations[index]
        if frame is not None and not 0 <= frame < observation.shape[0]:
            raise ValueError("stack frame out of range")
        chosen = trace.metadata["frames"][index]["action"]
        q = trace.metadata["frames"][index]["q"]
        if alternative is None:
            alternative = max(
                (a for a in range(len(q)) if a != chosen), key=lambda a: q[a]
            )
        if not 0 <= alternative < len(q) or alternative == chosen:
            raise ValueError("alternative must differ from chosen action")
        key = (episode, index, frame, alternative, channel, radius)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Explanation worker busy; retry after it finishes")
        try:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            started = time.perf_counter()
            result = explain_observation(
                trace.model,
                observation,
                chosen=chosen,
                alternative=alternative,
                frame=frame,
                channel=channel,
                radius=radius,
            )
            result["elapsed_seconds"] = time.perf_counter() - started
            self.cache[key] = result
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
            return result
        finally:
            self.lock.release()

    def examples(self, episode: str, kind: str, unit: int) -> list[dict]:
        from .model import to_float_observations

        trace = self.trace(episode)
        limit = 64 if kind == "channel" else 512
        if kind not in ("channel", "feature") or not 0 <= unit < limit:
            raise ValueError("invalid example feature")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Explanation worker busy; retry after it finishes")
        try:
            key = (episode, kind, unit)
            if key in self.cache:
                return self.cache[key]["examples"]
            device = next(trace.model.parameters()).device
            rows = []
            with torch.inference_mode():
                for start in range(0, len(trace.observations), 32):
                    batch = torch.as_tensor(
                        np.stack(trace.observations[start : start + 32]), device=device
                    )
                    conv = trace.model.features(
                        to_float_observations(
                            batch, expected_shape=trace.model.observation_shape
                        )
                    )
                    values = (
                        conv[:, unit].flatten(1)
                        if kind == "channel"
                        else (trace.model.shared(conv)[:, unit : unit + 1])
                    )
                    for offset, value in enumerate(values.cpu().numpy()):
                        at = int(value.argmax())
                        rows.append(
                            {
                                "index": start + offset,
                                "activation": float(value[at]),
                                "region": [at // 7, at % 7]
                                if kind == "channel"
                                else None,
                                "input_region": [8 * (at // 7), 8 * (at % 7), 36, 36]
                                if kind == "channel"
                                else None,
                            }
                        )
            rows = sorted(
                (row for row in rows if row["activation"] > 0),
                key=lambda row: (-row["activation"], row["index"]),
            )[:8]
            self.cache[key] = {"examples": rows}
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
            return rows
        finally:
            self.lock.release()


class ExplanationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: ExplanationService,
        run_dir: Path | None = None,
    ):
        self.service = service
        self.run_dir = run_dir.resolve() if run_dir else None
        self.run_lock = threading.Lock()
        self.selected_id = self.run_dir.name if self.run_dir else None
        self.selected_service = service
        super().__init__(address, ExplanationHandler)

    def for_run(self, run_id: str | None) -> ExplanationService:
        if run_id is None or self.run_dir is None:
            return self.service
        with self.run_lock:
            if run_id == self.selected_id:
                return self.selected_service
        rows = catalog(self.run_dir.parent)
        row = next((row for row in rows if row["id"] == run_id), None)
        if row is None:
            raise ValueError("Unknown or excluded run")
        if row["replay_unavailable"]:
            raise ValueError(row["replay_unavailable"])
        with self.run_lock:
            if run_id != self.selected_id:
                self.selected_service = service_for_run(self.run_dir.parent / run_id)
                self.selected_id = run_id
            return self.selected_service


class ExplanationHandler(BaseHTTPRequestHandler):
    server: ExplanationHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        request = urlsplit(self.path)
        query = parse_qs(request.query)
        episode = query.get("episode", [next(iter(self.server.service.traces))])[0]
        parts = request.path.strip("/").split("/")
        try:
            if request.path == "/":
                self.send_payload(
                    Path(__file__).with_name("explain.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
                return
            if request.path == "/api/runs":
                rows = (
                    catalog(self.server.run_dir.parent) if self.server.run_dir else []
                )
                self.send_payload(
                    json.dumps(rows, allow_nan=False).encode(), "application/json"
                )
                return
            run_id = query.get("run", [None])[0]
            if request.path == "/api/metrics":
                if not self.server.run_dir or not run_id:
                    raise ValueError("Run required")
                result = metrics_for_run(self.server.run_dir.parent, run_id)
                self.send_payload(
                    json.dumps(result, allow_nan=False).encode(), "application/json"
                )
                return
            service = self.server.for_run(run_id)
            if request.path == "/api/episodes":
                result = service.episodes()
            elif request.path == "/api/trace":
                result = service.metadata(episode)
                if run_id:
                    suffix = "?" + urlencode({"run": run_id})
                    for frame in result["frames"]:
                        frame["game_url"] += suffix
                    if result["final_game_url"]:
                        result["final_game_url"] += suffix
            elif parts[:2] == ["api", "final-game"] and len(parts) == 3:
                body = service.trace(parts[2]).final_game_png
                if body is None:
                    raise KeyError("final image unavailable")
                self.send_payload(body, "image/png")
                return
            elif parts[:2] == ["api", "game"] and len(parts) == 4:
                index = int(parts[3].removesuffix(".png"))
                trace = service.trace(parts[2])
                if index < 0:
                    raise ValueError("negative index")
                self.send_payload(trace.game_pngs[index], "image/png")
                return
            elif parts[:2] == ["api", "explain"] and len(parts) == 3:
                frame = query.get("frame", ["all"])[0]
                alt = query.get("alternative", ["auto"])[0]
                result = service.explain(
                    episode,
                    int(parts[2]),
                    frame=None if frame == "all" else int(frame),
                    alternative=None if alt == "auto" else int(alt),
                    channel=int(query.get("channel", ["0"])[0]),
                    radius=int(query.get("radius", ["6"])[0]),
                )
            elif request.path == "/api/examples":
                result = service.examples(
                    episode,
                    query.get("kind", ["channel"])[0],
                    int(query.get("unit", ["0"])[0]),
                )
            else:
                self.send_error(404)
                return
            self.send_payload(
                json.dumps(result, allow_nan=False).encode(), "application/json"
            )
        except BlockingIOError as error:
            self.send_error(429, str(error))
        except (KeyError, IndexError, ValueError, OSError) as error:
            self.send_error(400, str(error))

    def send_payload(self, body: bytes, content_type: str) -> None:
        compress = (
            "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024
        )
        if compress:
            body = gzip.compress(body, compresslevel=1)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if compress:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        pass


def service_for_run(run_dir: Path, *, device: str = "cpu") -> ExplanationService:
    """Resolve recorded final policy and choose only held-out representatives."""
    run_dir = run_dir.resolve()
    config = resolve_run_replay_config(run_dir.parent, run_dir.name)
    report = json.loads((run_dir / "report.json").read_text())
    # No newest-checkpoint fallback: the final evaluation must name its policy.
    recorded = report.get("checkpoint")
    if not recorded or (run_dir / recorded).resolve() != config.checkpoint.resolve():
        raise ValueError("final evaluation must name the checkpoint explicitly")
    if not config.checkpoint.resolve().is_relative_to(run_dir):
        raise ValueError("checkpoint must remain inside the selected run")
    full_config = json.loads((run_dir / "config.json").read_text())
    if full_config.get("reward_shaping", {}).get("enabled", False):
        raise ValueError(
            "shaped-reward replay not yet supported; do not mislabel reward"
        )
    holdout = config.evaluation.get("holdout")
    if not holdout:
        raise ValueError("held-out per-seed evaluation required")
    picks = select_comparison_episodes({"holdout": holdout})
    traces = {}
    for label, pick in picks.items():
        trace = generate_explanation_trace(
            config.checkpoint,
            seed=pick.seed,
            steps=config.eval_max_steps,
            device=device,
            step_frames=config.step_frames,
            difficulty=config.difficulty,
            patterns=config.patterns,
            powerups=config.powerups,
        )
        trace.metadata.update(
            {
                "label": label,
                "source": "holdout",
                "run_id": run_dir.name,
                "eval_reward": pick.eval_reward,
                "gamma": full_config.get("model", {}).get("gamma"),
            }
        )
        actual = sum(row["reward"] for row in trace.metadata["frames"])
        trace.metadata["evaluation_reward_matches"] = (
            abs(actual - pick.eval_reward) < 1e-6
        )
        # Mismatch remains visible evidence; never substitute expected score.
        print(
            json.dumps(
                {
                    "episode": label,
                    "seed": pick.seed,
                    "reward": actual,
                    "evaluation_reward": pick.eval_reward,
                    "matches": trace.metadata["evaluation_reward_matches"],
                }
            ),
            flush=True,
        )
        traces[label] = trace
    return ExplanationService(traces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    # Dedicated viewer process: avoid oversubscribing tiny single-state CNNs.
    torch.set_num_threads(1)
    if args.device == "cpu":
        torch.backends.nnpack.set_flags(False)
    service = service_for_run(args.run_dir, device=args.device)
    first_model = next(iter(service.traces.values())).model
    if next(first_model.parameters()).device.type == "cpu":
        torch.backends.nnpack.set_flags(False)
    host = args.host or tailscale_ipv4() or "127.0.0.1"
    server = ExplanationHTTPServer((host, args.port), service, args.run_dir)
    print(f"Explanation replay: http://{host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
