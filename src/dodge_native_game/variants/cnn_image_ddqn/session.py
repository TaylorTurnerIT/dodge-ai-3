"""Native DDQN session adapter for the LaunchSpark-style dashboard."""

from __future__ import annotations

import re
import threading
import time
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import best_greedy_episode
from .native_replay import DEFAULT_REPLAY_STEPS, generate_native_replay
from .replay_server import (
    DEFAULT_REPLAY_PORT,
    ReplayServer,
    ReplayStore,
    tailscale_ipv4,
)
from .rewards import CONFIG_PATH as DEFAULT_REWARD_CONFIG
from .run import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_STEPS,
    DEFAULT_HISTORY_ROOT,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_STACK_SIZE,
    DEFAULT_STEPS,
    DEFAULT_TARGET_SYNC_INTERVAL,
    DEFAULT_UPDATE_EVERY,
    DEFAULT_WARMUP_STEPS,
    TrainingControl,
    TrainingEvent,
    _native_game_config,
    train_run,
)
from .run_artifacts import VARIANT_ID, load_run
from .telemetry import EVENT_KEYS, DashboardSnapshot, ModelEntry


@dataclass(frozen=True, slots=True)
class _MetricCursor:
    count: int = 0


def _variant_root(history_root: Path) -> Path:
    root = history_root.expanduser()
    return root if root.name == VARIANT_ID else root / VARIANT_ID


class DodgeDDQNSession:
    """Dashboard-facing owner of one native Rust + PyTorch DDQN run.

    LaunchSpark's dashboard only depends on a small session surface. This
    class keeps that surface while replacing the upstream PPO/PICO-8 backend
    with the existing native collision-image trainer and artifact contract.
    """

    model_suffix = ".pt"

    def __init__(
        self,
        *,
        history_root: Path | str = DEFAULT_HISTORY_ROOT,
        run_id: str = "cnn-image-ddqn-dashboard-001",
        seed: int = 42,
        total_steps: int = DEFAULT_STEPS,
        stack_size: int = DEFAULT_STACK_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        update_every: int = DEFAULT_UPDATE_EVERY,
        target_sync_interval: int = DEFAULT_TARGET_SYNC_INTERVAL,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        eval_episodes: int = DEFAULT_EVAL_EPISODES,
        eval_steps: int = DEFAULT_EVAL_STEPS,
        device: str = "auto",
        model_dir: Path | str | None = None,
        difficulty: int = 1,
        patterns: bool = False,
        powerups: bool = True,
        normal_enemies: bool = True,
        kamikaze_enemies: bool = True,
        spawn_rate_percent: int = 100,
        lives: int = 0,
        architecture: str = "cnn-image-ddqn",
        replay_host: str | None = None,
        replay_port: int = DEFAULT_REPLAY_PORT,
        replay_steps: int = DEFAULT_REPLAY_STEPS,
    ) -> None:
        if architecture != VARIANT_ID:
            raise ValueError(f"unsupported architecture: {architecture}")
        if not 0 <= int(seed) <= 32_767:
            raise ValueError("seed must be between 0 and 32767")
        self.history_root = Path(history_root).expanduser()
        self.variant_root = _variant_root(self.history_root)
        self.run_id = str(run_id)
        self.run_root = self.variant_root / self.run_id
        self.seed = int(seed)
        self.total_steps = int(total_steps)
        self.stack_size = int(stack_size)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.update_every = int(update_every)
        self.target_sync_interval = int(target_sync_interval)
        self.log_interval = int(log_interval)
        self.eval_episodes = int(eval_episodes)
        self.eval_steps = int(eval_steps)
        self.device = device
        self.architecture = architecture
        self.replay_host = replay_host
        self.replay_port = int(replay_port)
        self.replay_steps = int(replay_steps)
        self.active_model_name = self.run_id
        self.model_dir = (
            Path(model_dir).expanduser()
            if model_dir
            else (self.variant_root / "models")
        )
        self.reward_config_path = self.variant_root / DEFAULT_REWARD_CONFIG.name
        self.env_kwargs: dict[str, Any] = {
            "difficulty": int(difficulty),
            "patterns": bool(patterns),
            "powerups": bool(powerups),
            "normal_enemies": bool(normal_enemies),
            "kamikaze_enemies": bool(kamikaze_enemies),
            "spawn_rate_percent": int(spawn_rate_percent),
            "action_repeat": 4,
            "lives": int(lives),
        }
        self.lock = threading.RLock()
        self.data = DashboardSnapshot(
            status="starting",
            model_name=self.active_model_name,
            seed=self.seed,
        )
        self.control = TrainingControl()
        self.worker: threading.Thread | None = None
        self.last_error: Exception | None = None
        self._started_at = time.monotonic()
        self._metric_cursor = _MetricCursor()
        self.replay_store = ReplayStore()
        self.replay_server: ReplayServer | None = None
        self._replay_lock = threading.Lock()
        self._replay_pending = False
        self._replay_thread: threading.Thread | None = None
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reward_config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.reward_config_path.exists():
            from .rewards import RewardConfig, save

            save(RewardConfig(), self.reward_config_path)
        self.start()

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.worker = threading.Thread(
            target=self._run,
            name="dodge-native-ddqn",
            daemon=True,
        )
        self.worker.start()

    def _run(self) -> None:
        try:
            train_run(
                history_root=self.history_root,
                run_id=self.run_id,
                steps=self.total_steps,
                seed=self.seed,
                stack_size=self.stack_size,
                batch_size=self.batch_size,
                warmup_steps=self.warmup_steps,
                update_every=self.update_every,
                target_sync_interval=self.target_sync_interval,
                log_interval=self.log_interval,
                eval_episodes=self.eval_episodes,
                eval_steps=self.eval_steps,
                device=self.device,
                control=self.control,
                game_config=self.env_kwargs,
            )
        except Exception as error:  # the artifact writer has published failure
            self.last_error = error
            self.control.emit(
                TrainingEvent("training", False, f"Training failed: {error}")
            )

    def _sync_artifacts(self) -> None:
        if not self.run_root.is_dir():
            return
        record = load_run(self.run_root)
        if not record.manifest and not record.status:
            return
        metrics = record.metrics
        with self.lock:
            if len(metrics) < self._metric_cursor.count:
                self.data = DashboardSnapshot(
                    status="starting",
                    model_name=self.active_model_name,
                    seed=self.seed,
                )
                self._metric_cursor = _MetricCursor()
            for metric in metrics[self._metric_cursor.count :]:
                self._append_metric(metric)
            self._metric_cursor = _MetricCursor(len(metrics))

            latest = record.latest_metrics
            state = record.state
            self.data.status = {
                "queued": "starting",
                "running": "training",
                "paused": "paused",
                "stale": "error",
                "completed": "complete",
                "stopped": "complete",
                "failed": "error",
            }.get(state, "starting")
            if self.control.paused and state == "running":
                self.data.status = "paused"
            self.data.model_name = self.active_model_name
            self.data.seed = int(record.manifest.get("seed", self.seed))
            self.data.update = int(
                latest.get("optimizer_step", record.status.get("step", 0)) or 0
            )
            self.data.steps_per_second = float(
                latest.get("throughput", self.data.steps_per_second) or 0.0
            )
            self.data.elapsed_seconds = max(
                0.0,
                time.monotonic() - self._started_at,
            )
            total_step = int(record.status.get("step", latest.get("step", 0)) or 0)
            if state == "completed":
                self.data.progress_phase = "complete"
                self.data.rollout_progress = 1.0
                # The reported best is the best final-eval episode, not the
                # exploration-era training max. Recompute from the stored
                # table so runs predating best_eval_episode still qualify.
                try:
                    best = (
                        best_greedy_episode(record.evaluation)
                        if record.evaluation
                        else None
                    )
                    self.data.best_eval_reward = (
                        float(best["eval_reward"]) if best else None
                    )
                except (TypeError, ValueError):
                    self.data.best_eval_reward = None
            elif state == "failed":
                self.data.progress_phase = "rollout"
                self.data.rollout_progress = min(
                    1.0, total_step / max(1, self.total_steps)
                )
            else:
                self.data.progress_phase = "rollout"
                self.data.rollout_progress = min(
                    1.0, total_step / max(1, self.total_steps)
                )

    def _append_metric(self, metric: Mapping[str, Any]) -> None:
        reward = float(metric.get("reward", 0.0) or 0.0)
        frames = float(metric.get("survival_frames", 0.0) or 0.0)
        length = float(metric.get("episode_length", frames / 60.0) or 0.0)
        kills = float(metric.get("enemies_killed", 0.0) or 0.0)
        entropy = float(metric.get("policy_entropy", metric.get("entropy", 0.0)) or 0.0)
        self.data.reward.append(reward)
        self.data.episode_length.append(length)
        self.data.enemies_killed.append(kills)
        self.data.entropy.append(entropy)
        self.data.life_lengths[0].append(length)
        for band in self.data.life_lengths[1:]:
            band.append(0.0)
        self.data.best_score = max(
            self.data.best_score,
            float(metric.get("best_score", reward) or 0.0),
        )
        try:
            self.data.dead_units = float(metric.get("dead_units", 0.0) or 0.0)
            self.data.action_balance = float(metric.get("action_balance", 1.0) or 0.0)
            self.data.reward_zero_share = float(
                metric.get("reward_zero_share", 0.0) or 0.0
            )
            self.data.td_error_mean = float(metric.get("td_error_mean", 0.0) or 0.0)
            self.data.train_holdout_gap = float(
                metric.get("train_holdout_gap", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            pass
        for key in EVENT_KEYS:
            event = self.data.events[key]
            if key == "survival_frames":
                value = int(frames)
            elif key == "deaths":
                value = int(metric.get("deaths", 0) or 0)
            else:
                value = int(metric.get(key, 0) or 0)
            event.rollout = value
            event.total += value
            event.history.append(value)

    def tick(self, _dt: float) -> None:
        self._sync_artifacts()

    def drain_events(self) -> list[TrainingEvent]:
        events = self.control.drain_events()
        for event in events:
            if event.kind == "save" and event.ok and event.path:
                with self.lock:
                    self.active_model_name = event.path.stem
            if event.kind == "load" and event.ok and event.path:
                with self.lock:
                    self.active_model_name = event.path.stem
        return events

    def toggle_pause(self) -> bool:
        paused = self.control.toggle_pause()
        with self.lock:
            if paused:
                self.data.status = "paused"
        return paused

    def request_save(self, path: Path, kind: str = "save") -> None:
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.control.emit(TrainingEvent(kind, False, str(error), path))
            return
        self.control.request_save(path, kind=kind)

    def load_model(self, entry: ModelEntry) -> Path:
        if entry.kind != "DDQN":
            raise ValueError("Select a native DDQN checkpoint")
        if not entry.path.is_file():
            raise ValueError("That checkpoint no longer exists")
        with self.lock:
            self.active_model_name = entry.name
            self.data.model_name = entry.name
            self.data.status = "loading"
        self.control.request_load(entry.path)
        return entry.path

    def available_models(self) -> list[ModelEntry]:
        roots = [self.model_dir]
        if self.model_dir != self.variant_root:
            roots.append(self.variant_root)
        paths: dict[Path, None] = {}
        for root in roots:
            if not root.is_dir():
                continue
            try:
                for path in root.rglob("*.pt"):
                    if path.is_file() and not path.is_symlink():
                        paths[path] = None
            except OSError:
                continue
        entries = [ModelEntry(path.stem, path, "DDQN") for path in paths]
        return sorted(
            entries,
            key=lambda entry: entry.path.stat().st_mtime,
            reverse=True,
        )

    @staticmethod
    def _safe_name(name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip(".-")
        if not safe_name:
            raise ValueError("Enter a model name")
        return safe_name

    def game_config(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.env_kwargs)

    def configure_game(
        self,
        *,
        difficulty: int,
        patterns: bool,
        powerups: bool,
        normal_enemies: bool = True,
        kamikaze_enemies: bool = True,
        spawn_rate_percent: int = 100,
        action_repeat: int = 4,
        lives: int = 0,
    ) -> bool | str:
        difficulty = int(difficulty)
        action_repeat = int(action_repeat)
        spawn_rate_percent = int(spawn_rate_percent)
        lives = int(lives)
        if difficulty not in (1, 2, 3):
            raise ValueError("Difficulty must be between 1 and 3")
        if not 1 <= action_repeat <= 8:
            raise ValueError("Decision interval must be between 1 and 8 frames")
        if not 50 <= spawn_rate_percent <= 150:
            raise ValueError("Spawn rate must be between 50 and 150 percent")
        if not 0 <= lives <= 5:
            raise ValueError("Training lives must be between 0 and 5")
        changes = {
            "difficulty": difficulty,
            "patterns": bool(patterns),
            "powerups": bool(powerups),
            "normal_enemies": bool(normal_enemies),
            "kamikaze_enemies": bool(kamikaze_enemies),
            "spawn_rate_percent": spawn_rate_percent,
            "action_repeat": action_repeat,
            "lives": lives,
        }
        with self.lock:
            config = {**self.env_kwargs, **changes}
            if config == self.env_kwargs:
                return False
            self.env_kwargs = config
        self.control.request_config(config)
        return "pending"

    def _ensure_replay_server(self) -> ReplayServer:
        if self.replay_server is not None:
            return self.replay_server
        bind_host = self.replay_host or tailscale_ipv4() or "127.0.0.1"
        public_host = bind_host
        if bind_host in {"0.0.0.0", "::"}:
            public_host = tailscale_ipv4() or "127.0.0.1"
        server = ReplayServer(
            store=self.replay_store,
            host=bind_host,
            port=self.replay_port,
            public_host=public_host,
            history_root=self.history_root,
        )
        server.start()
        self.replay_server = server
        return server

    def _generate_replay(self, checkpoint: Path) -> None:
        try:
            native_config = _native_game_config(
                self.game_config(), default_step_frames=4
            )
            server = self._ensure_replay_server()
            replay = generate_native_replay(
                checkpoint,
                seed=self.seed,
                steps=self.replay_steps,
                device=self.device,
                step_frames=int(native_config["native_step_frames"]),
                difficulty=int(native_config["difficulty"]),
                patterns=bool(native_config["patterns"]),
                powerups=bool(native_config["powerups"]),
            )
            token = server.store.add(replay)
            url = server.url_for(token)
            opened = webbrowser.open_new(url)
            suffix = " (browser opened)" if opened else ""
            self.control.emit(
                TrainingEvent(
                    "watch",
                    True,
                    f"Native replay ready: {url}{suffix}",
                    checkpoint,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            self.control.emit(TrainingEvent("watch", False, str(error), checkpoint))
        finally:
            with self._replay_lock:
                self._replay_pending = False

    def _generate_run_comparison(self, run_id: str) -> None:
        try:
            from .native_replay import MAX_REPLAY_STEPS as _MAX
            from .run_replay import generate_run_comparison

            server = self._ensure_replay_server()
            pairs = generate_run_comparison(
                self.history_root,
                run_id,
                episodes=("best", "median", "worst"),
                steps=_MAX,
                device=self.device,
            )
            tokens = [server.store.add(replay) for _, replay in pairs]
            url = server.url_for_compare(tokens)
            opened = webbrowser.open_new(url)
            suffix = " (browser opened)" if opened else ""
            detail = ", ".join(
                f"{episode.key} seed {episode.seed} ({episode.source}, "
                f"eval {episode.eval_reward:.1f})"
                for episode, _ in pairs
            )
            self.control.emit(
                TrainingEvent(
                    "watch",
                    True,
                    f"Run {run_id} comparison ready: {url}{suffix} [{detail}]",
                    None,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            self.control.emit(TrainingEvent("watch", False, str(error), None))
        finally:
            with self._replay_lock:
                self._replay_pending = False

    def watch_run(self, run_id: str) -> bool:
        """Generate best/median/worst replays for one run and open the compare page."""

        with self._replay_lock:
            if self._replay_pending:
                return True
            self._replay_pending = True
        self.control.emit(
            TrainingEvent(
                "watch", True, f"Preparing replay comparison for {run_id}…", None
            )
        )
        self._replay_thread = threading.Thread(
            target=self._generate_run_comparison,
            args=(run_id,),
            name="dodge-native-replay-compare",
            daemon=True,
        )
        self._replay_thread.start()
        return True

    def watch_agent(self) -> bool:
        entries = self.available_models()
        if not entries:
            return False
        checkpoint = entries[0].path
        with self._replay_lock:
            if self._replay_pending:
                return True
            self._replay_pending = True
        self.control.emit(
            TrainingEvent(
                "watch",
                True,
                f"Preparing native replay for {checkpoint.stem}…",
                checkpoint,
            )
        )
        self._replay_thread = threading.Thread(
            target=self._generate_replay,
            args=(checkpoint,),
            name="dodge-native-replay",
            daemon=True,
        )
        self._replay_thread.start()
        return True

    def close(self) -> None:
        self.control.stop()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=10)
        if self._replay_thread and self._replay_thread.is_alive():
            self._replay_thread.join(timeout=10)
        if self.replay_server is not None:
            self.replay_server.close()
            self.replay_server = None


__all__ = ["DodgeDDQNSession"]
