# ruff: noqa: E402

import os
from collections import deque
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from dodge_native_game.variants.cnn_image_ddqn import rewards
from dodge_native_game.variants.cnn_image_ddqn.dashboard import (
    BLUE,
    GREEN,
    MIN_WINDOW_SIZE,
    MUTED,
    RED,
    WINDOW_SIZE,
    DashboardRenderer,
    TrainingDashboard,
    format_duration,
)
from dodge_native_game.variants.cnn_image_ddqn.telemetry import (
    DashboardSnapshot,
    ModelEntry,
    sample_snapshot,
)


def test_format_duration() -> None:
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661.9) == "01:01:01"


def test_sample_snapshot_is_deterministic() -> None:
    left = sample_snapshot(seed=9)
    right = sample_snapshot(seed=9)
    assert left.reward.as_list() == right.reward.as_list()
    assert left.events["deaths"].total == right.events["deaths"].total


def test_sample_snapshot_shows_improvement() -> None:
    data = sample_snapshot()
    assert data.reward.delta > 0
    assert data.episode_length.delta > 0
    assert data.entropy.delta < 0


@pytest.mark.parametrize("size", [WINDOW_SIZE, MIN_WINDOW_SIZE, (1100, 680)])
def test_dashboard_renders_at_supported_sizes(size: tuple[int, int]) -> None:
    pygame.font.init()
    renderer = DashboardRenderer()
    surface = pygame.Surface(size)
    renderer.render(surface, sample_snapshot())
    assert renderer.buttons
    assert surface.get_at((0, 0))[:3] != (0, 0, 0)


def test_dashboard_renders_before_any_data_arrives() -> None:
    pygame.font.init()
    DashboardRenderer().render(pygame.Surface(WINDOW_SIZE), DashboardSnapshot())


def test_stats_show_final_best_beside_training_max() -> None:
    pygame.font.init()
    renderer = DashboardRenderer()
    data = sample_snapshot()
    data.best_eval_reward = 9.5
    renderer.render(pygame.Surface(WINDOW_SIZE), data)
    assert renderer.buttons
    renderer.render(pygame.Surface(WINDOW_SIZE), DashboardSnapshot())


def test_training_progress_and_delta_colors() -> None:
    data = DashboardSnapshot(
        progress_phase="learning", rollout_progress=1.0, learning_progress=0.375
    )
    assert DashboardRenderer._training_progress(data) == ("Learning", 0.375, BLUE)
    style = DashboardRenderer._delta_style
    assert style(5.0, rising=True)[1] == GREEN
    assert style(-5.0, rising=True)[1] == RED
    assert style(-0.6, rising=False)[1] == GREEN
    assert style(0.0, rising=True)[1] == MUTED


class _StubSession:
    def __init__(self, reward_config_path: Path) -> None:
        self.data = sample_snapshot()
        import threading

        self.lock = threading.RLock()
        self.reward_config_path = reward_config_path
        self.model_dir = Path(reward_config_path).parent
        self.model_suffix = ".pt"
        self.env_kwargs = {
            "difficulty": 1,
            "patterns": False,
            "powerups": True,
            "normal_enemies": True,
            "kamikaze_enemies": True,
            "spawn_rate_percent": 100,
            "action_repeat": 4,
            "lives": 0,
            "max_episode_frames": None,
        }
        self.configured: list[object] = []
        self.pending_events: list[object] = []
        self.watch_result = True

    def tick(self, _dt: float) -> None:
        return None

    def close(self) -> None:
        return None

    def available_models(self) -> list[ModelEntry]:
        return []

    def drain_events(self) -> list[object]:
        drained, self.pending_events = self.pending_events, []
        return drained

    def request_save(self, path: Path, kind: str = "save") -> None:
        self.configured.append(("request_save", Path(path).name, kind))

    @staticmethod
    def _safe_name(name: str) -> str:
        from dodge_native_game.variants.cnn_image_ddqn.session import (
            DodgeDDQNSession,
        )

        return DodgeDDQNSession._safe_name(name)

    def watch_agent(self) -> bool:
        self.configured.append(("watch_agent",))
        return self.watch_result

    def game_config(self) -> dict[str, object]:
        return dict(self.env_kwargs)

    def configure_game(self, **config: object) -> bool:
        self.env_kwargs = {**self.env_kwargs, **config}
        self.configured.append(dict(config))
        return True


def _dashboard(tmp_path: Path) -> TrainingDashboard:
    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_rewards()
    dashboard.draw()
    return dashboard


def test_controls_match_launchspark_layout() -> None:
    pygame.font.init()
    renderer = DashboardRenderer()
    renderer.render(pygame.Surface(WINDOW_SIZE), sample_snapshot())
    names = [button.name for button in renderer.buttons]
    assert names == ["Pause", "Rewards", "Game Config", "Watch Agent", "Models"]


def test_game_config_applies_all_dashboard_settings(tmp_path: Path) -> None:
    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_game_config()
    dashboard.draw()
    dashboard._click(dashboard.game_regions["toggles"]["powerups"].center)
    dashboard._click(dashboard.game_regions["toggles"]["patterns"].center)
    for _spec, track in dashboard.game_regions["tracks"]:
        dashboard._click((track.right - 1, track.centery))
    dashboard._click(dashboard.game_regions["apply"].center)
    assert session.configured[0]["difficulty"] == 3
    assert session.configured[0]["patterns"] is True
    assert session.configured[0]["powerups"] is False
    assert session.configured[0]["spawn_rate_percent"] == 150
    assert not dashboard.game_menu_open
    pygame.quit()


def test_reward_slider_writes_the_upstream_config(tmp_path: Path) -> None:
    dashboard = _dashboard(tmp_path)
    spec, track = next(
        (spec, track)
        for spec, track in dashboard.reward_regions["tracks"]
        if spec.name == "death_penalty"
    )
    dashboard._click((track.right - 1, track.centery))
    assert rewards.load(tmp_path / "rewards.json").death_penalty == spec.maximum
    dashboard._set_reward_from_position(spec, track, (track.x, track.centery))
    assert rewards.load(tmp_path / "rewards.json").death_penalty == spec.minimum
    pygame.quit()


def test_model_menu_saves_pt_and_sanitizes_names(tmp_path: Path) -> None:
    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard.model_name = "sub/dir/name"
    dashboard._save_model()
    assert session.configured == [("request_save", "sub-dir-name.pt", "save")]
    assert dashboard.save_pending
    pygame.quit()


def test_model_discovery_does_not_select_a_default(tmp_path: Path) -> None:
    class DiscoveringSession(_StubSession):
        def __init__(self, reward_config_path: Path) -> None:
            super().__init__(reward_config_path)
            self.entries: list[ModelEntry] = []

        def available_models(self) -> list[ModelEntry]:
            return list(self.entries)

    session = DiscoveringSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_models()
    path = tmp_path / "saved.pt"
    session.entries.append(ModelEntry("saved", path, "DDQN"))
    dashboard.model_discovery_at = 0.0
    dashboard.draw()
    assert dashboard.selected_model is None
    assert dashboard.model_entries[0].name == "saved"
    pygame.quit()


def test_rendering_happens_outside_the_telemetry_lock(tmp_path: Path) -> None:
    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    seen: list[bool] = []
    original = DashboardRenderer.render

    def spy(renderer: DashboardRenderer, surface, data, toast=None):
        seen.append(session.lock._is_owned())
        return original(renderer, surface, data, toast)

    dashboard.renderer.render = spy.__get__(dashboard.renderer)
    dashboard.draw()
    assert seen == [False]
    pygame.quit()


def test_empty_sparkline_is_safe() -> None:
    pygame.font.init()
    DashboardRenderer._sparkline(
        pygame.Surface((100, 50)), pygame.Rect(0, 0, 100, 50), deque(), (1, 2, 3)
    )


def test_session_events_become_toasts(tmp_path: Path) -> None:
    from dodge_native_game.variants.cnn_image_ddqn.run import TrainingEvent

    session = _StubSession(tmp_path / "rewards.json")
    session.pending_events = [TrainingEvent("save", True, "Saved bot.pt")]
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._drain_session_events()
    assert dashboard.toast == "Saved bot.pt"
    pygame.quit()


def test_screenshot_cli_uses_the_real_renderer(tmp_path: Path) -> None:
    from dodge_native_game.variants.cnn_image_ddqn.dashboard import main

    output = tmp_path / "dashboard.png"
    assert main(["--screenshot", str(output)]) == 0
    assert output.is_file()
    assert pygame.image.load(output).get_size() == WINDOW_SIZE
