#!/usr/bin/env python3
"""LaunchSpark-style live dashboard for native collision-image DDQN runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pygame

from . import rewards
from .rewards import SLIDERS as REWARD_SLIDERS
from .telemetry import PROGRESS_ROWS, sample_snapshot

# The upstream dashboard exposed several PPO architectures. This project
# intentionally keeps the variant selector narrow: the native Rust engine is
# paired with this CNN/DDQN experiment only.
ARCHITECTURES = ("cnn-image-ddqn",)

try:
    import psutil
except ImportError:  # The GUI remains usable before requirements are refreshed.
    psutil = None


# Charts move at the pace of finished episodes, not of a game loop, so a
# 60Hz redraw spent a whole core to show the same numbers three times over.
DASHBOARD_FPS = 20
SYSTEM_POLL_SECONDS = 1.0
MODEL_DISCOVERY_SECONDS = 1.0

WINDOW_SIZE = (1328, 760)
MIN_WINDOW_SIZE = (1000, 650)
BG = (27, 26, 25)
PANEL = (14, 15, 15)
HEADER = (24, 23, 22)
BORDER = (57, 55, 51)
GRID = (45, 44, 41)
TEXT = (224, 223, 219)
MUTED = (139, 137, 130)
CYAN = (21, 194, 235)
BLUE = (65, 145, 220)
GREEN = (113, 190, 89)
YELLOW = (244, 189, 47)
ORANGE = (224, 116, 65)
RED = (255, 76, 76)
WHITE = (244, 244, 241)


EVENT_SPECS = (
    ("survival_frames", "Survival frames", CYAN),
    ("enemies_destroyed", "Enemies destroyed", YELLOW),
    ("explosion_pickups", "Explosion pickups", (224, 116, 65)),
    ("freeze_pickups", "Freeze pickups", (94, 181, 225)),
    ("shrink_pickups", "Shrink pickups", (177, 130, 224)),
    ("patterns_survived", "Patterns survived", GREEN),
    ("deaths", "Deaths", RED),
    ("lives_spent", "Lives spent", (177, 130, 224)),
)

GAME_DEFAULTS = {
    "difficulty": 1,
    "patterns": False,
    "powerups": True,
    "normal_enemies": True,
    "kamikaze_enemies": True,
    "spawn_rate_percent": 100,
    "action_repeat": 4,
    "lives": 0,
}


@dataclass(frozen=True)
class GameSliderSpec:
    name: str
    label: str
    minimum: int
    maximum: int
    step: int = 1


GAME_SLIDERS = (
    GameSliderSpec("difficulty", "Difficulty", 1, 3),
    GameSliderSpec("spawn_rate_percent", "Spawn rate", 50, 150, 5),
    GameSliderSpec("action_repeat", "Decision interval", 1, 8),
    # Training only: the first N lethal collisions end the episode without
    # ending the cartridge run; active patterns restart from their beginning.
    GameSliderSpec("lives", "Training lives", 0, 5),
)


@lru_cache(maxsize=1)
def accelerator_name():
    """What the policy update will actually run on.

    Reported rather than assumed: a CPU-only torch build cannot use a GPU even
    when the machine has one, and that is invisible from the outside.
    """
    try:
        import torch
    except ImportError:
        return "none"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace("NVIDIA ", "")
    if getattr(torch.version, "cuda", None) is None:
        return "CPU (torch has no CUDA)"
    return "CPU"


@dataclass
class Button:
    name: str
    rect: pygame.Rect
    fill: tuple[int, int, int]
    text: tuple[int, int, int]


class DashboardRenderer:
    # One type scale for the whole cockpit: 12 caption, 13 label, 14 body,
    # 20 hero values. Numerals always mono so columns align. Rule: panels
    # 8px radius, controls 6px, meters pill.
    MONO_STACK = "consolas,dejavusansmono,freemono,monospace"

    def __init__(self):
        self.font_caption = pygame.font.SysFont("segoeui", 12)
        self.font_small = pygame.font.SysFont("segoeui", 13)
        self.font_body = pygame.font.SysFont("segoeui", 14)
        self.font_body_bold = pygame.font.SysFont("segoeui", 14, bold=True)
        self.font_title = pygame.font.SysFont("segoeui", 14, bold=True)
        self.font_value = pygame.font.SysFont(self.MONO_STACK, 20, bold=True)
        self.font_mono = pygame.font.SysFont(self.MONO_STACK, 14)
        self.font_mono_bold = pygame.font.SysFont(self.MONO_STACK, 14, bold=True)
        self.font_mono_small = pygame.font.SysFont(self.MONO_STACK, 12)
        self.buttons: list[Button] = []
        self._background = None
        self._system_reading = None
        self._system_read_at = 0.0

    @staticmethod
    def _card(surface, rect, title=None):
        pygame.draw.rect(surface, PANEL, rect, border_radius=8)
        pygame.draw.rect(surface, BORDER, rect, width=2, border_radius=8)
        if title:
            header = pygame.Rect(rect.x + 2, rect.y + 2, rect.w - 4, 36)
            pygame.draw.rect(surface, HEADER, header, border_radius=6)
            pygame.draw.line(
                surface,
                BORDER,
                (rect.x + 2, rect.y + 38),
                (rect.right - 2, rect.y + 38),
            )

    @staticmethod
    def _text(surface, font, text, color, pos, anchor="topleft"):
        image = font.render(str(text), True, color)
        rect = image.get_rect()
        setattr(rect, anchor, pos)
        surface.blit(image, rect)
        return rect

    @staticmethod
    def _fit_text(font, text, max_width):
        if font.size(text)[0] <= max_width:
            return text
        suffix = "..."
        while text and font.size(text + suffix)[0] > max_width:
            text = text[:-1]
        return text + suffix

    def _draw_background(self, surface):
        """Blit the dotted backdrop, rebuilding it only when the window resizes.

        Drawn directly this was 2,560 one-pixel circles a frame, none of which
        ever change.
        """
        size = surface.get_size()
        if self._background is None or self._background.get_size() != size:
            background = pygame.Surface(size)
            background.fill(BG)
            for y in range(10, size[1], 20):
                for x in range(10, size[0], 20):
                    pygame.draw.circle(background, (49, 47, 44), (x, y), 1)
            self._background = background
        surface.blit(self._background, (0, 0))

    def _draw_chart(
        self,
        surface,
        rect,
        title,
        series,
        color,
        suffix="",
        rising=True,
        elapsed=0.0,
        bands=None,
    ):
        """Draw one run-spanning metric: faint raw values under a bright trend.

        The y-axis is anchored to the whole run rather than to the visible
        window, so a climbing curve climbs instead of being flattened by the
        axis rescaling underneath it.
        """
        self._card(surface, rect)
        self._text(surface, self.font_title, title, color, (rect.x + 14, rect.y + 12))
        self._text(
            surface,
            self.font_value,
            f"{series.latest:.1f}{suffix}",
            color,
            (rect.right - 14, rect.y + 10),
            "topright",
        )

        plot = pygame.Rect(rect.x + 14, rect.y + 52, rect.w - 28, rect.h - 74)
        for i in range(4):
            y = plot.y + i * plot.h // 3
            pygame.draw.line(surface, GRID, (plot.x, y), (plot.right, y), 1)
        for i in range(5):
            x = plot.x + i * plot.w // 4
            pygame.draw.line(surface, GRID, (x, plot.y), (x, plot.bottom), 1)

        # Bands are only worth stacking once more than one of them is in use.
        # With training lives off every run is a single band, and anchoring the
        # axis to zero for it would flatten the curve the chart exists to show.
        stack = [band for band in (bands or ()) if any(band.values)]
        stack = (
            stack
            if len(stack) > 1
            and all(len(band.values) == len(series.values) for band in stack)
            else None
        )

        if len(series) >= 2:
            low, high = series.bounds()
            if stack:
                # A stack has to stand on zero, or its lowest band is clipped.
                low = 0.0
            span = max(high - low, 1e-9)

            def to_points(source):
                return [
                    (
                        round(plot.x + i * plot.w / (len(source) - 1)),
                        round(plot.bottom - (value - low) / span * plot.h),
                    )
                    for i, value in enumerate(source)
                ]

            raw = to_points(series.values)
            trend = to_points(series.trend)

            # The translucent fill needs its own surface, but only plot-sized:
            # allocating and blitting a whole window per chart was most of the
            # render cost, and eleven of them were done every frame.
            overlay = pygame.Surface((plot.w, plot.h + 1), pygame.SRCALPHA)
            if stack:

                def local_points(values):
                    return [(x - plot.x, y - plot.y) for x, y in to_points(values)]

                # Bottom band is the run's first life, so the stack reads
                # upward in the order the lives were spent.
                floor = [0.0] * len(series.values)
                for index, band in enumerate(stack):
                    ceiling = [f + v for f, v in zip(floor, band.values)]
                    top = local_points(ceiling)
                    pygame.draw.polygon(
                        overlay,
                        (*color, max(30, 116 - index * 17)),
                        top + local_points(floor)[::-1],
                    )
                    pygame.draw.lines(overlay, (*color, 150), False, top, 1)
                    floor = ceiling
            else:
                local = [(x - plot.x, y - plot.y) for x, y in raw]
                polygon = local + [(local[-1][0], plot.h), (local[0][0], plot.h)]
                pygame.draw.polygon(overlay, (*color, 34), polygon)
                pygame.draw.lines(overlay, (*color, 105), False, local, 1)
            surface.blit(overlay, plot.topleft)
            pygame.draw.lines(surface, color, False, trend, 3)

            self._text(
                surface, self.font_mono_small, f"{high:.1f}", MUTED, (plot.x + 5, plot.y + 4)
            )
            self._text(
                surface,
                self.font_mono_small,
                f"{low:.1f}",
                MUTED,
                (plot.x + 5, plot.bottom - 18),
            )

        # Footer: where the run started, and how far it has come.
        self._text(
            surface,
            self.font_caption,
            self._span_label(elapsed),
            MUTED,
            (plot.x, rect.bottom - 18),
        )
        if len(series) >= 2:
            arrow, tint = self._delta_style(series.delta, rising)
            self._text(
                surface,
                self.font_mono_small,
                f"{arrow} {series.delta:+.1f}{suffix} since start",
                tint,
                (rect.right - 14, rect.bottom - 18),
                "topright",
            )

    @staticmethod
    def _delta_style(delta, rising):
        """Arrow and colour for a change, given which direction counts as better."""
        if abs(delta) < 1e-6:
            return "=", MUTED
        improved = delta > 0 if rising else delta < 0
        return ("▲" if delta > 0 else "▼"), (GREEN if improved else RED)

    @staticmethod
    def _span_label(elapsed):
        if elapsed <= 0:
            return "run start → now"
        return f"run start → now ({format_duration(elapsed)})"

    def _draw_progress(self, surface, rect, data):
        """Answer 'am I improving?' in words rather than shapes.

        Each row reads baseline -> current with the change, measured from the
        smoothed trend so a single lucky episode cannot claim a win.
        """
        self._card(surface, rect, "Progress")
        self._text(
            surface, self.font_title, "Progress", TEXT, (rect.x + 14, rect.y + 12)
        )
        self._text(
            surface,
            self.font_caption,
            "since run start",
            MUTED,
            (rect.right - 14, rect.y + 16),
            "topright",
        )

        rows = [
            (label, getattr(data, key), unit, rising)
            for key, label, unit, rising in PROGRESS_ROWS
            if hasattr(data, key)
        ]
        if not rows:
            return
        top = rect.y + 44
        step = (rect.bottom - 14 - top) // max(1, len(rows))
        # Two-line rows (delta top-right, range below) need 40px; short
        # panels fall back to one shared line instead of bleeding into
        # the panel below.
        two_line = step >= 40
        step = step if two_line else max(26, step)
        for index, (label, series, unit, rising) in enumerate(rows):
            y = top + index * step
            self._text(surface, self.font_small, label, MUTED, (rect.x + 14, y))
            if len(series) < 2:
                if two_line:
                    self._text(
                        surface,
                        self.font_mono,
                        "collecting…",
                        MUTED,
                        (rect.x + 14, y + 16),
                    )
                else:
                    self._text(
                        surface,
                        self.font_mono,
                        "collecting…",
                        MUTED,
                        (rect.right - 14, y + 14),
                        "topright",
                    )
                continue
            arrow, tint = self._delta_style(series.delta, rising)
            if two_line:
                self._text(
                    surface,
                    self.font_mono_bold,
                    f"{arrow} {series.delta:+.1f}{unit}",
                    tint,
                    (rect.right - 14, y),
                    "topright",
                )
                self._text(
                    surface,
                    self.font_mono,
                    f"{series.baseline:.1f} → {series.smoothed:.1f}{unit}",
                    TEXT,
                    (rect.x + 14, y + 16),
                )
            else:
                self._text(
                    surface,
                    self.font_mono,
                    f"{series.baseline:.1f} → {series.smoothed:.1f}{unit}",
                    TEXT,
                    (rect.x + 14, y + 14),
                )
                self._text(
                    surface,
                    self.font_mono_bold,
                    f"{arrow} {series.delta:+.1f}{unit}",
                    tint,
                    (rect.right - 14, y + 14),
                    "topright",
                )

    def _draw_controls(self, surface, rect, data):
        self._card(surface, rect, "Learning Control")
        self._text(
            surface,
            self.font_title,
            "Learning Control",
            TEXT,
            (rect.centerx, rect.y + 11),
            "midtop",
        )
        gap = 8
        button_h = 36
        x = rect.x + 12
        y = rect.y + 52
        width = rect.w - 24
        paused = data.status == "paused"
        # One chrome system: neutral buttons, light labels. Only the primary
        # Pause/Resume action takes the cyan accent; metric colors stay on
        # the charts where they encode data.
        configs = (
            ("Resume" if paused else "Pause", (22, 88, 106), TEXT, True),
            ("Rewards", (36, 35, 33), TEXT, False),
            ("Game Config", (36, 35, 33), TEXT, False),
            ("Watch Agent", (36, 35, 33), TEXT, False),
            ("Models", (36, 35, 33), TEXT, False),
        )
        self.buttons = []
        for name, fill, text_color, primary in configs:
            button = Button(name, pygame.Rect(x, y, width, button_h), fill, text_color)
            self.buttons.append(button)
            pygame.draw.rect(surface, fill, button.rect, border_radius=6)
            if not primary:
                pygame.draw.rect(surface, BORDER, button.rect, width=1, border_radius=6)
            self._text(
                surface,
                self.font_body_bold,
                name,
                text_color,
                button.rect.center,
                "center",
            )
            y += button_h + gap

    def draw_reward_menu(self, surface, config, config_path):
        """Sliders over the reward weights, and where they are stored.

        Returns the clickable regions: one track rect per weight, plus the
        close and reset-to-defaults controls.
        """
        veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 165))
        surface.blit(veil, (0, 0))

        width = min(560, surface.get_width() - 50)
        height = min(150 + 58 * len(REWARD_SLIDERS), surface.get_height() - 50)
        modal = pygame.Rect(0, 0, width, height)
        modal.center = surface.get_rect().center
        self._card(surface, modal)
        self._text(
            surface,
            self.font_title,
            "Reward Weights",
            TEXT,
            (modal.x + 20, modal.y + 17),
        )
        close = pygame.Rect(modal.right - 44, modal.y + 9, 34, 34)
        self._text(surface, self.font_title, "X", MUTED, close.center, "center")

        self._text(
            surface,
            self.font_small,
            f"Applied from the next episode  ·  {config_path}",
            MUTED,
            (modal.x + 20, modal.y + 46),
        )

        tracks = []
        y = modal.y + 78
        for spec in REWARD_SLIDERS:
            value = getattr(config, spec.name)
            self._text(surface, self.font_body, spec.label, TEXT, (modal.x + 20, y))
            self._text(
                surface,
                self.font_body_bold,
                f"{value:g}",
                ORANGE,
                (modal.right - 20, y),
                "topright",
            )
            self._text(
                surface, self.font_small, spec.hint, MUTED, (modal.x + 20, y + 20)
            )

            track = pygame.Rect(modal.x + 20, y + 42, modal.w - 40, 6)
            pygame.draw.rect(surface, (52, 50, 47), track, border_radius=3)
            span = spec.maximum - spec.minimum
            fraction = (value - spec.minimum) / span if span else 0.0
            fraction = min(1.0, max(0.0, fraction))
            filled = pygame.Rect(track.x, track.y, int(track.w * fraction), track.h)
            pygame.draw.rect(surface, ORANGE, filled, border_radius=3)
            knob = (track.x + int(track.w * fraction), track.centery)
            pygame.draw.circle(surface, WHITE, knob, 8)
            pygame.draw.circle(surface, ORANGE, knob, 8, width=2)
            # Widened so the knob stays grabbable without pixel-perfect aim.
            tracks.append((spec, track.inflate(0, 26)))
            y += 58

        defaults = pygame.Rect(modal.x + 20, modal.bottom - 46, 150, 34)
        pygame.draw.rect(surface, (48, 48, 46), defaults, border_radius=6)
        self._text(
            surface,
            self.font_body_bold,
            "Restore defaults",
            WHITE,
            defaults.center,
            "center",
        )
        return {"close": close, "defaults": defaults, "tracks": tracks, "modal": modal}

    @staticmethod
    def _game_value(spec, value):
        if spec.name == "difficulty":
            return {1: "Easy", 2: "Normal", 3: "Hard"}[int(value)]
        if spec.name == "lives":
            value = int(value)
            if value == 0:
                return "Off"
            return f"{value} life" if value == 1 else f"{value} lives"
        if spec.name == "spawn_rate_percent":
            return f"{int(value)}%"
        return f"{int(value)} frames"

    def draw_game_menu(self, surface, config):
        veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 165))
        surface.blit(veil, (0, 0))

        width = min(560, surface.get_width() - 50)
        height = min(600, surface.get_height() - 30)
        modal = pygame.Rect(0, 0, width, height)
        modal.center = surface.get_rect().center
        self._card(surface, modal)
        self._text(
            surface,
            self.font_title,
            "Game Configuration",
            TEXT,
            (modal.x + 20, modal.y + 17),
        )
        close = pygame.Rect(modal.right - 44, modal.y + 9, 34, 34)
        self._text(surface, self.font_title, "X", MUTED, close.center, "center")
        self._text(
            surface,
            self.font_small,
            "Applied together while the current policy continues",
            MUTED,
            (modal.x + 20, modal.y + 47),
        )

        toggles = {}
        y = modal.y + 78
        for name, label in (
            ("powerups", "Powerups"),
            ("patterns", "Patterns"),
            ("normal_enemies", "Normal enemies"),
            ("kamikaze_enemies", "Kamikaze enemies"),
        ):
            row = pygame.Rect(modal.x + 20, y, modal.w - 40, 42)
            self._text(
                surface, self.font_body, label, TEXT, (row.x, row.centery), "midleft"
            )
            switch = pygame.Rect(row.right - 50, row.centery - 12, 50, 24)
            enabled = bool(config[name])
            pygame.draw.rect(
                surface, GREEN if enabled else (66, 64, 60), switch, border_radius=12
            )
            knob_x = switch.right - 12 if enabled else switch.x + 12
            pygame.draw.circle(surface, WHITE, (knob_x, switch.centery), 9)
            toggles[name] = row
            y += 44

        tracks = []
        y += 10
        for spec in GAME_SLIDERS:
            value = int(config[spec.name])
            self._text(surface, self.font_body, spec.label, TEXT, (modal.x + 20, y))
            self._text(
                surface,
                self.font_body_bold,
                self._game_value(spec, value),
                GREEN,
                (modal.right - 20, y),
                "topright",
            )
            track = pygame.Rect(modal.x + 20, y + 38, modal.w - 40, 6)
            pygame.draw.rect(surface, (52, 50, 47), track, border_radius=3)
            span = spec.maximum - spec.minimum
            fraction = (value - spec.minimum) / span if span else 0.0
            filled = pygame.Rect(track.x, track.y, int(track.w * fraction), track.h)
            pygame.draw.rect(surface, GREEN, filled, border_radius=3)
            knob = (track.x + int(track.w * fraction), track.centery)
            pygame.draw.circle(surface, WHITE, knob, 8)
            pygame.draw.circle(surface, GREEN, knob, 8, width=2)
            tracks.append((spec, track.inflate(0, 26)))
            y += 67

        defaults = pygame.Rect(modal.x + 20, modal.bottom - 50, 150, 34)
        cancel = pygame.Rect(modal.right - 226, modal.bottom - 50, 96, 34)
        apply = pygame.Rect(modal.right - 116, modal.bottom - 50, 96, 34)
        pygame.draw.rect(surface, (48, 48, 46), defaults, border_radius=6)
        pygame.draw.rect(surface, (48, 48, 46), cancel, border_radius=6)
        pygame.draw.rect(surface, GREEN, apply, border_radius=6)
        self._text(
            surface,
            self.font_body_bold,
            "Restore defaults",
            WHITE,
            defaults.center,
            "center",
        )
        self._text(
            surface, self.font_body_bold, "Cancel", WHITE, cancel.center, "center"
        )
        self._text(
            surface, self.font_body_bold, "Apply", (24, 42, 24), apply.center, "center"
        )
        return {
            "close": close,
            "defaults": defaults,
            "cancel": cancel,
            "apply": apply,
            "toggles": toggles,
            "tracks": tracks,
            "modal": modal,
        }

    def draw_model_menu(
        self,
        surface,
        name,
        entries,
        selected_index,
        offset,
        name_focused=False,
        name_cursor=None,
        name_selected=False,
    ):
        veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 165))
        surface.blit(veil, (0, 0))

        width = min(500, surface.get_width() - 50)
        height = min(440, surface.get_height() - 50)
        modal = pygame.Rect(0, 0, width, height)
        modal.center = surface.get_rect().center
        self._card(surface, modal)
        self._text(
            surface,
            self.font_title,
            "Model Checkpoints",
            TEXT,
            (modal.x + 20, modal.y + 17),
        )
        close = pygame.Rect(modal.right - 44, modal.y + 9, 34, 34)
        self._text(surface, self.font_title, "X", MUTED, close.center, "center")

        self._text(
            surface,
            self.font_small,
            "SAVE CURRENT SESSION",
            MUTED,
            (modal.x + 20, modal.y + 59),
        )
        name_rect = pygame.Rect(modal.x + 20, modal.y + 82, modal.w - 150, 38)
        save_rect = pygame.Rect(name_rect.right + 10, name_rect.y, 100, 38)
        pygame.draw.rect(surface, (26, 27, 27), name_rect, border_radius=5)
        pygame.draw.rect(
            surface,
            CYAN if name_focused else (83, 81, 75),
            name_rect,
            width=2 if name_focused else 1,
            border_radius=5,
        )
        inner = pygame.Surface((name_rect.w - 20, name_rect.h - 8), pygame.SRCALPHA)
        if name:
            cursor = (
                len(name)
                if name_cursor is None
                else max(0, min(name_cursor, len(name)))
            )
            text_image = self.font_body.render(name, True, TEXT)
            cursor_x = self.font_body.size(name[:cursor])[0]
            scroll = max(0, cursor_x - inner.get_width() + 2)
            if name_selected:
                selection = pygame.Rect(
                    -scroll, 3, text_image.get_width(), self.font_body.get_height() + 2
                )
                pygame.draw.rect(inner, (31, 92, 108), selection)
            inner.blit(text_image, (-scroll, 4))
            if (
                name_focused
                and not name_selected
                and pygame.time.get_ticks() % 1000 < 600
            ):
                pygame.draw.line(
                    inner,
                    CYAN,
                    (cursor_x - scroll, 3),
                    (cursor_x - scroll, 3 + self.font_body.get_height()),
                    1,
                )
        else:
            self._text(inner, self.font_body, "Model name", MUTED, (0, 4))
            if name_focused and pygame.time.get_ticks() % 1000 < 600:
                pygame.draw.line(
                    inner, CYAN, (0, 3), (0, 3 + self.font_body.get_height()), 1
                )
        surface.blit(inner, (name_rect.x + 10, name_rect.y + 4))
        pygame.draw.rect(surface, CYAN, save_rect, border_radius=5)
        self._text(
            surface,
            self.font_body_bold,
            "Save",
            (22, 39, 43),
            save_rect.center,
            "center",
        )

        self._text(
            surface,
            self.font_small,
            "AVAILABLE MODELS",
            MUTED,
            (modal.x + 20, modal.y + 145),
        )
        if entries:
            last = min(offset + 5, len(entries))
            self._text(
                surface,
                self.font_small,
                f"{offset + 1}-{last} of {len(entries)}",
                MUTED,
                (modal.right - 20, modal.y + 145),
                "topright",
            )
        list_rect = pygame.Rect(
            modal.x + 20, modal.y + 168, modal.w - 40, modal.h - 240
        )
        pygame.draw.rect(surface, (11, 12, 12), list_rect, border_radius=5)
        pygame.draw.rect(surface, GRID, list_rect, width=1, border_radius=5)

        row_rects = []
        visible_entries = entries[offset : offset + 5]
        if not visible_entries:
            self._text(
                surface,
                self.font_body,
                "No saved models yet",
                MUTED,
                list_rect.center,
                "center",
            )
        for index, entry in enumerate(visible_entries):
            entry_index = offset + index
            row = pygame.Rect(
                list_rect.x + 5, list_rect.y + 5 + index * 36, list_rect.w - 10, 32
            )
            row_rects.append((entry_index, row))
            if entry_index == selected_index:
                pygame.draw.rect(surface, (31, 66, 76), row, border_radius=4)
                pygame.draw.rect(surface, CYAN, row, width=1, border_radius=4)
            self._text(
                surface, self.font_body_bold, entry.name, TEXT, (row.x + 10, row.y + 6)
            )
            self._text(
                surface,
                self.font_small,
                entry.kind,
                MUTED,
                (row.right - 10, row.y + 7),
                "topright",
            )

        load_rect = pygame.Rect(modal.right - 130, modal.bottom - 52, 110, 36)
        cancel_rect = pygame.Rect(load_rect.x - 100, load_rect.y, 88, 36)
        can_load = selected_index is not None and 0 <= selected_index < len(entries)
        pygame.draw.rect(
            surface, GREEN if can_load else GRID, load_rect, border_radius=5
        )
        self._text(
            surface,
            self.font_body_bold,
            "Load",
            (26, 40, 25) if can_load else MUTED,
            load_rect.center,
            "center",
        )
        pygame.draw.rect(surface, (45, 45, 43), cancel_rect, border_radius=5)
        self._text(
            surface, self.font_body, "Cancel", TEXT, cancel_rect.center, "center"
        )
        return {
            "modal": modal,
            "close": close,
            "name": name_rect,
            "save": save_rect,
            "rows": row_rects,
            "load": load_rect,
            "cancel": cancel_rect,
        }

    @staticmethod
    def _dead_unit_color(fraction):
        """Amber once a tenth of the policy trunk is gone, red past a third.

        Worth its own colour because nothing else on this dashboard shows it:
        a run whose policy has died keeps producing plausible entropy, reward
        and step counts while learning nothing at all.
        """
        if fraction >= 0.34:
            return RED
        if fraction >= 0.10:
            return YELLOW
        return TEXT

    def _draw_stats(self, surface, rect, data):
        self._card(surface, rect, "Learning Statistics")
        self._text(
            surface,
            self.font_title,
            "Learning Statistics",
            TEXT,
            (rect.centerx, rect.y + 11),
            "midtop",
        )
        model_name = data.model_name or "None"
        if len(model_name) > 18:
            model_name = model_name[:15] + "..."
        best_eval = getattr(data, "best_eval_reward", None)
        rows = (
            ("Status", data.status.title()),
            ("Model", model_name),
            ("Update", f"{data.update:,}"),
            ("Seed", str(data.seed)),
            ("Steps / second", f"{data.steps_per_second:,.0f}"),
            ("Best episode (train)", f"{data.best_score:.1f}"),
            (
                "Best episode (final)",
                f"{best_eval:.1f}" if best_eval is not None else "-",
            ),
            ("Learning time", format_duration(data.elapsed_seconds)),
            (
                "Dead units",
                f"{data.dead_units:.0%}",
                self._dead_unit_color(data.dead_units),
            ),
        )
        y = rect.y + 52
        row_step = 19 if len(rows) <= 8 else 18
        for label, value, *tint in rows:
            color = tint[0] if tint else TEXT
            self._text(surface, self.font_small, label, MUTED, (rect.x + 13, y))
            self._text(
                surface, self.font_mono, value, color, (rect.right - 13, y), "topright"
            )
            y += row_step
        bar = pygame.Rect(rect.x + 13, rect.bottom - 19, rect.w - 26, 5)
        pygame.draw.rect(surface, GRID, bar, border_radius=2)
        label, progress, color = self._training_progress(data)
        fill = bar.copy()
        fill.w = round(bar.w * progress)
        pygame.draw.rect(surface, color, fill, border_radius=2)
        self._text(surface, self.font_small, label, color, (bar.x, bar.y - 17))
        self._text(
            surface,
            self.font_small,
            f"{progress:.0%}",
            TEXT,
            (bar.right, bar.y - 17),
            "topright",
        )

    @staticmethod
    def _training_progress(data):
        phase = getattr(data, "progress_phase", "rollout")
        if phase == "learning":
            return "Learning", data.learning_progress, BLUE
        if phase == "complete":
            return "Complete", 1.0, GREEN
        return "Rollout", data.rollout_progress, GREEN

    def _system_reading_cached(self):
        """CPU and memory, read at most once a second.

        psutil.virtual_memory() alone measured 5.8ms on this machine, which is
        an absurd share of a frame for three numbers that move slowly and are
        displayed to the nearest percent.
        """
        now = time.monotonic()
        if (
            self._system_reading is not None
            and now - self._system_read_at < SYSTEM_POLL_SECONDS
        ):
            return self._system_reading
        if psutil:
            reading = (
                psutil.cpu_percent(),
                psutil.virtual_memory().percent,
                psutil.Process().memory_info().rss / (1024 * 1024),
            )
        else:
            reading = (0, 0, 0)
        self._system_reading = reading
        self._system_read_at = now
        return reading

    def _draw_system(self, surface, rect):
        self._card(surface, rect, "System")
        self._text(
            surface,
            self.font_title,
            "System",
            TEXT,
            (rect.centerx, rect.y + 11),
            "midtop",
        )
        cpu, memory, process_mb = self._system_reading_cached()
        rows = (
            ("CPU", cpu, f"{cpu:.0f}%", CYAN),
            ("Memory", memory, f"{process_mb:.0f} MB", GREEN),
            ("Accelerator", 0, accelerator_name(), YELLOW),
        )
        y = rect.y + 50
        for label, percent, value, color in rows:
            self._text(surface, self.font_small, label, MUTED, (rect.x + 13, y))
            self._text(
                surface, self.font_mono, value, TEXT, (rect.right - 13, y), "topright"
            )
            bar = pygame.Rect(rect.x + 13, y + 19, rect.w - 26, 4)
            pygame.draw.rect(surface, GRID, bar, border_radius=2)
            if label == "Accelerator":
                width = bar.w * 0.18
            else:
                width = bar.w * min(percent, 100) / 100
            pygame.draw.rect(
                surface, color, (bar.x, bar.y, width, bar.h), border_radius=2
            )
            y += 38

    def _draw_events(self, surface, rect, data):
        self._card(surface, rect, "Achievements")
        title = (
            f"{data.model_name}'s Achievements" if data.model_name else "Achievements"
        )
        title = self._fit_text(self.font_title, title, rect.w - 30)
        self._text(
            surface, self.font_title, title, TEXT, (rect.centerx, rect.y + 11), "midtop"
        )
        header_y = rect.y + 49
        columns = (rect.x + 17, rect.x + rect.w * 0.49, rect.x + rect.w * 0.67)
        self._text(surface, self.font_caption, "Event", MUTED, (columns[0], header_y))
        self._text(
            surface,
            self.font_caption,
            "Rollout",
            MUTED,
            (columns[1], header_y),
            "topright",
        )
        self._text(
            surface, self.font_caption, "Total", MUTED, (columns[2], header_y), "topright"
        )
        self._text(
            surface,
            self.font_caption,
            "Recent rollouts",
            MUTED,
            (rect.right - 18, header_y),
            "topright",
        )
        pygame.draw.line(
            surface,
            GRID,
            (rect.x + 14, header_y + 24),
            (rect.right - 14, header_y + 24),
        )

        available = rect.bottom - (header_y + 31)
        row_h = max(31, available // len(EVENT_SPECS))
        y = header_y + 31
        spark_x = round(rect.x + rect.w * 0.73)
        spark_w = rect.right - 18 - spark_x
        for key, label, color in EVENT_SPECS:
            event = data.events[key]
            self._text(surface, self.font_body, label, color, (columns[0], y + 5))
            self._text(
                surface,
                self.font_mono,
                f"{event.rollout:,}",
                color,
                (columns[1], y + 5),
                "topright",
            )
            self._text(
                surface,
                self.font_mono,
                f"{event.total:,}",
                color,
                (columns[2], y + 5),
                "topright",
            )
            self._sparkline(
                surface,
                pygame.Rect(spark_x, y + 4, spark_w, row_h - 10),
                event.history,
                color,
            )
            y += row_h

    @staticmethod
    def _sparkline(surface, rect, values, color):
        if len(values) < 2:
            pygame.draw.line(
                surface, GRID, (rect.x, rect.centery), (rect.right, rect.centery)
            )
            return
        values = list(values)
        high = max(max(values), 1)
        points = [
            (
                round(rect.x + i * rect.w / (len(values) - 1)),
                round(rect.bottom - value / high * rect.h),
            )
            for i, value in enumerate(values)
        ]
        overlay = pygame.Surface((rect.w, rect.h + 1), pygame.SRCALPHA)
        local = [(x - rect.x, y - rect.y) for x, y in points]
        polygon = local + [(local[-1][0], rect.h), (local[0][0], rect.h)]
        pygame.draw.polygon(overlay, (*color, 35), polygon)
        surface.blit(overlay, rect.topleft)
        pygame.draw.lines(surface, color, False, points, 2)

    def _draw_toast(self, surface, text):
        image = self.font_body_bold.render(text, True, TEXT)
        rect = image.get_rect()
        box = pygame.Rect(0, 0, rect.w + 30, 40)
        box.midbottom = (surface.get_width() // 2, surface.get_height() - 14)
        pygame.draw.rect(surface, (41, 40, 37), box, border_radius=6)
        pygame.draw.rect(surface, BORDER, box, width=1, border_radius=6)
        rect.center = box.center
        surface.blit(image, rect)

    def render(self, surface, data, toast=None):
        self._draw_background(surface)
        width, height = surface.get_size()
        # One spacing unit (12px) and a sidebar wide enough for label/value
        # rows to sit on one line. The old 190-215px rail starved the stats
        # panel while the charts sprawled; both zones now share one rhythm.
        margin, gap = 12, 12
        sidebar_w = max(200, min(248, int(width * 0.185)))
        main_x = margin + sidebar_w + gap
        main_w = width - main_x - margin
        top_h = max(190, int(height * 0.30))
        lower_y = margin + top_h + gap
        lower_h = height - lower_y - margin
        left_w = int(main_w * 0.51)
        right_x = main_x + left_w + gap
        right_w = main_w - left_w - gap
        # Reward is the headline metric, so it gets the wider share of the top
        # row; the Progress panel takes the space the old Average Reward chart
        # occupied, since a trend line now makes that chart redundant.
        reward_w = int(main_w * 0.62)

        controls_h = 276
        stats_h = 262
        controls = pygame.Rect(margin, margin, sidebar_w, controls_h)
        stats = pygame.Rect(margin, controls.bottom + gap, sidebar_w, stats_h)
        system = pygame.Rect(
            margin, stats.bottom + gap, sidebar_w, height - stats.bottom - gap - margin
        )
        reward = pygame.Rect(main_x, margin, reward_w, top_h)
        progress = pygame.Rect(
            main_x + reward_w + gap, margin, main_w - reward_w - gap, top_h
        )
        events = pygame.Rect(main_x, lower_y, left_w, lower_h)
        # Right column: entropy, episode length, enemies killed.
        row_h = (lower_h - 2 * gap) // 3
        entropy = pygame.Rect(right_x, lower_y, right_w, row_h)
        length = pygame.Rect(right_x, entropy.bottom + gap, right_w, row_h)
        enemies = pygame.Rect(
            right_x, length.bottom + gap, right_w, lower_h - 2 * (row_h + gap)
        )

        self._draw_controls(surface, controls, data)
        self._draw_stats(surface, stats, data)
        self._draw_system(surface, system)
        elapsed = data.elapsed_seconds
        self._draw_chart(
            surface, reward, "Episode Reward", data.reward, CYAN, elapsed=elapsed
        )
        self._draw_progress(surface, progress, data)
        self._draw_events(surface, events, data)
        self._draw_chart(
            surface,
            entropy,
            "Policy Entropy",
            data.entropy,
            RED,
            rising=False,
            elapsed=elapsed,
        )
        self._draw_chart(
            surface,
            length,
            "Survival Time",
            data.episode_length,
            YELLOW,
            " s",
            elapsed=elapsed,
            bands=data.life_lengths,
        )
        self._draw_chart(
            surface,
            enemies,
            "Enemies Killed",
            data.enemies_killed,
            ORANGE,
            elapsed=elapsed,
        )
        if toast:
            self._draw_toast(surface, toast)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingDashboard:
    def __init__(self, session, size=WINDOW_SIZE, hidden=False):
        pygame.init()
        flags = pygame.RESIZABLE
        if hidden:
            flags |= pygame.HIDDEN
        self.surface = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("Dodge DDQN Training")
        self.renderer = DashboardRenderer()
        self.session = session
        self.toast = None
        self.toast_until = 0.0
        self.model_menu_open = False
        # A checkpoint name is an explicit user choice.  In particular, do not
        # make the current architecture look like a selected/default model.
        self.model_name = ""
        self.model_entries = []
        self.selected_model = None
        self.model_offset = 0
        self.model_discovery_at = 0.0
        self.model_regions = {}
        self.save_pending = False
        self.model_name_focused = False
        self.model_name_cursor = len(self.model_name)
        self.model_name_selected = False
        self.reward_menu_open = False
        self.reward_regions = {}
        self.reward_config_path = getattr(
            self.session, "reward_config_path", rewards.CONFIG_PATH
        )
        self.reward_config = rewards.load(self.reward_config_path)
        self.game_menu_open = False
        self.game_regions = {}
        self.game_config = dict(GAME_DEFAULTS)
        self.dragging_slider = None
        if getattr(self.session.data, "status", None) == "idle":
            self._open_models()

    def _open_rewards(self):
        # Re-read on open so a hand-edited file is what the sliders show.
        self.reward_config = rewards.load(self.reward_config_path)
        self.reward_menu_open = True

    def _open_game_config(self):
        getter = getattr(self.session, "game_config", None)
        current = getter() if getter else getattr(self.session, "env_kwargs", {})
        # Only the settings this panel owns. The session's env kwargs carry
        # others -- the frame cap, for one -- and _apply_game_config splats
        # this dict straight into configure_game, which does not take them.
        self.game_config = {
            **GAME_DEFAULTS,
            **{key: value for key, value in current.items() if key in GAME_DEFAULTS},
        }
        self.game_menu_open = True

    def _set_game_from_position(self, spec, track, position):
        span = spec.maximum - spec.minimum
        fraction = (position[0] - track.x) / track.w if track.w else 0.0
        fraction = min(1.0, max(0.0, fraction))
        raw = spec.minimum + fraction * span
        value = spec.minimum + round((raw - spec.minimum) / spec.step) * spec.step
        self.game_config[spec.name] = min(spec.maximum, max(spec.minimum, value))

    def _apply_game_config(self):
        configure = getattr(self.session, "configure_game", None)
        if configure is None:
            self._show_toast("This training session cannot change game settings")
            return
        try:
            changed = configure(**{key: self.game_config[key] for key in GAME_DEFAULTS})
        except (OSError, ValueError) as error:
            self._show_toast(str(error))
            return
        self.game_menu_open = False
        if changed == "pending":
            self._show_toast("Applying after this update…")
        else:
            self._show_toast(
                "Game configuration applied"
                if changed
                else "Game configuration unchanged"
            )

    def _set_reward_from_position(self, spec, track, position):
        span = spec.maximum - spec.minimum
        fraction = (position[0] - track.x) / track.w if track.w else 0.0
        fraction = min(1.0, max(0.0, fraction))
        value = spec.minimum + fraction * span
        value = round(value / spec.step) * spec.step
        value = min(spec.maximum, max(spec.minimum, round(value, 6)))
        if getattr(self.reward_config, spec.name) == value:
            return
        setattr(self.reward_config, spec.name, value)
        rewards.save(self.reward_config, self.reward_config_path)

    def _show_toast(self, message):
        self.toast = message
        self.toast_until = time.monotonic() + 2.5

    def _open_models(self):
        self.selected_model = None
        self.model_offset = 0
        self._discover_models(force=True)
        self.model_menu_open = True
        self._focus_model_name(select_all=True)

    def _discover_models(self, force=False):
        """Refresh checkpoints without inventing a selection.

        Checkpoints can be produced by another trainer or copied into the
        models directory while this process is running. Keep an explicit
        selection by path when possible, but never promote the first/newest
        discovered file to a default selection.
        """
        now = time.monotonic()
        if not force and now < self.model_discovery_at:
            return
        self.model_discovery_at = now + MODEL_DISCOVERY_SECONDS

        selected_path = None
        if self.selected_model is not None and 0 <= self.selected_model < len(
            self.model_entries
        ):
            selected_path = self.model_entries[self.selected_model].path

        self.model_entries = self.session.available_models()
        self.selected_model = next(
            (
                index
                for index, entry in enumerate(self.model_entries)
                if entry.path == selected_path
            ),
            None,
        )
        max_offset = max(0, len(self.model_entries) - 5)
        self.model_offset = min(self.model_offset, max_offset)
        if self.selected_model is not None:
            self.model_offset = min(
                max(self.model_offset, self.selected_model - 4),
                self.selected_model,
            )

    def _focus_model_name(self, select_all=False):
        self.model_name_focused = True
        self.model_name_cursor = len(self.model_name)
        self.model_name_selected = bool(select_all and self.model_name)
        pygame.key.start_text_input()

    def _blur_model_name(self):
        self.model_name_focused = False
        self.model_name_selected = False
        pygame.key.stop_text_input()

    def _close_models(self):
        self.model_menu_open = False
        self._blur_model_name()

    def _insert_model_text(self, text):
        if not self.model_name_focused or not text:
            return
        text = "".join(character for character in text if character.isprintable())
        if not text:
            return
        if self.model_name_selected:
            self.model_name = ""
            self.model_name_cursor = 0
            self.model_name_selected = False
        available = 48 - len(self.model_name)
        text = text[:available]
        cursor = self.model_name_cursor
        self.model_name = self.model_name[:cursor] + text + self.model_name[cursor:]
        self.model_name_cursor += len(text)

    def _save_model(self):
        if self.save_pending:
            return
        if not self.model_name:
            self._show_toast("Enter a model name")
            return
        # _insert_model_text only filters non-printable characters, so a "/"
        # or "\" typed into the field reaches here unchanged. Route through
        # the session's own sanitizer so Save and the session agree on what
        # a name means, instead of building a path a stray separator could
        # turn into a subdirectory.
        try:
            safe_name = self.session._safe_name(self.model_name)
        except ValueError as error:
            self._show_toast(str(error))
            return
        self.save_pending = True
        suffix = getattr(self.session, "model_suffix", ".pt")
        self.session.request_save(
            self.session.model_dir / f"{safe_name}{suffix}", kind="save"
        )
        self._show_toast("Saving…")

    def _drain_session_events(self):
        """Turn asynchronous trainer results into toasts.

        Nothing here may raise: this runs inside the event loop, and an
        exception escaping it closes the window and takes training with it.
        `_discover_models` reaches the filesystem (`available_models` stats
        every checkpoint it globs), and a file can vanish between the two --
        another trainer cleaning up, a user deleting it -- so that part is
        wrapped separately: the toast and the cleared pending flag must still
        land even if the refresh itself blows up.
        """
        drain = getattr(self.session, "drain_events", None)
        if drain is None:
            return
        try:
            events = drain()
        except Exception:
            return
        for event in events:
            if event.kind == "save":
                self.save_pending = False
                if event.ok:
                    try:
                        self._discover_models(force=True)
                        self.selected_model = next(
                            (
                                i
                                for i, entry in enumerate(self.model_entries)
                                if entry.path == event.path
                            ),
                            None,
                        )
                        if self.selected_model is not None:
                            self.model_offset = max(0, self.selected_model - 4)
                    except Exception:
                        pass
            self._show_toast(event.message)

    def _load_model(self):
        if self.selected_model is None or not 0 <= self.selected_model < len(
            self.model_entries
        ):
            return
        entry = self.model_entries[self.selected_model]
        try:
            self.session.load_model(entry)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            self._show_toast(str(error))
            return
        self._close_models()
        self._show_toast(f"Loaded {entry.name}")

    def _click(self, position):
        if self.game_menu_open:
            regions = self.game_regions
            empty = pygame.Rect(0, 0, 0, 0)
            if regions.get("close", empty).collidepoint(position) or regions.get(
                "cancel", empty
            ).collidepoint(position):
                self.game_menu_open = False
                return
            if regions.get("defaults", empty).collidepoint(position):
                self.game_config = dict(GAME_DEFAULTS)
                return
            if regions.get("apply", empty).collidepoint(position):
                self._apply_game_config()
                return
            for name, row in regions.get("toggles", {}).items():
                if row.collidepoint(position):
                    self.game_config[name] = not self.game_config[name]
                    return
            for spec, track in regions.get("tracks", []):
                if track.collidepoint(position):
                    self.dragging_slider = ("game", spec, track)
                    self._set_game_from_position(spec, track, position)
                    return
            if not regions.get("modal", empty).collidepoint(position):
                self.game_menu_open = False
            return

        if self.reward_menu_open:
            regions = self.reward_regions
            if regions.get("close", pygame.Rect(0, 0, 0, 0)).collidepoint(position):
                self.reward_menu_open = False
                return
            if regions.get("defaults", pygame.Rect(0, 0, 0, 0)).collidepoint(position):
                self.reward_config = rewards.RewardConfig()
                rewards.save(self.reward_config, self.reward_config_path)
                self._show_toast("Reward weights restored to defaults")
                return
            for spec, track in regions.get("tracks", []):
                if track.collidepoint(position):
                    self.dragging_slider = ("reward", spec, track)
                    self._set_reward_from_position(spec, track, position)
                    return
            if not regions.get("modal", pygame.Rect(0, 0, 0, 0)).collidepoint(position):
                self.reward_menu_open = False
            return

        if self.model_menu_open:
            regions = self.model_regions
            if regions.get("close", pygame.Rect(0, 0, 0, 0)).collidepoint(
                position
            ) or regions.get("cancel", pygame.Rect(0, 0, 0, 0)).collidepoint(position):
                self._close_models()
            elif regions.get("name", pygame.Rect(0, 0, 0, 0)).collidepoint(position):
                self._focus_model_name()
            elif regions.get("save", pygame.Rect(0, 0, 0, 0)).collidepoint(position):
                self._save_model()
            elif regions.get("load", pygame.Rect(0, 0, 0, 0)).collidepoint(position):
                self._load_model()
            else:
                for index, row in regions.get("rows", []):
                    if row.collidepoint(position):
                        self._blur_model_name()
                        self.selected_model = index
                        break
            return

        for button in self.renderer.buttons:
            if not button.rect.collidepoint(position):
                continue
            if button.name in ("Pause", "Resume"):
                self.session.toggle_pause()
            elif button.name == "Rewards":
                self._open_rewards()
            elif button.name == "Game Config":
                self._open_game_config()
            elif button.name == "Watch Agent":
                # A launch's success toast now arrives as a SessionEvent; only
                # the "still waiting" case has nothing else to report it.
                if not self.session.watch_agent():
                    self._show_toast("Waiting for current update…")
            elif button.name == "Models":
                self._open_models()
            return

    def _model_key(self, event):
        if event.key == pygame.K_ESCAPE:
            self._close_models()
        elif event.key == pygame.K_RETURN:
            self._save_model()
        elif (
            self.model_name_focused
            and event.key == pygame.K_a
            and event.mod & pygame.KMOD_CTRL
        ):
            self.model_name_selected = bool(self.model_name)
            self.model_name_cursor = len(self.model_name)
        elif self.model_name_focused and event.key == pygame.K_BACKSPACE:
            if self.model_name_selected:
                self.model_name = ""
                self.model_name_cursor = 0
                self.model_name_selected = False
            elif self.model_name_cursor > 0:
                cursor = self.model_name_cursor
                self.model_name = (
                    self.model_name[: cursor - 1] + self.model_name[cursor:]
                )
                self.model_name_cursor -= 1
        elif self.model_name_focused and event.key == pygame.K_DELETE:
            if self.model_name_selected:
                self.model_name = ""
                self.model_name_cursor = 0
                self.model_name_selected = False
            elif self.model_name_cursor < len(self.model_name):
                cursor = self.model_name_cursor
                self.model_name = (
                    self.model_name[:cursor] + self.model_name[cursor + 1 :]
                )
        elif self.model_name_focused and event.key in (
            pygame.K_LEFT,
            pygame.K_RIGHT,
            pygame.K_HOME,
            pygame.K_END,
        ):
            self.model_name_selected = False
            if event.key == pygame.K_LEFT:
                self.model_name_cursor = max(0, self.model_name_cursor - 1)
            elif event.key == pygame.K_RIGHT:
                self.model_name_cursor = min(
                    len(self.model_name), self.model_name_cursor + 1
                )
            elif event.key == pygame.K_HOME:
                self.model_name_cursor = 0
            else:
                self.model_name_cursor = len(self.model_name)
        elif event.key == pygame.K_UP and self.model_entries:
            self._blur_model_name()
            if self.selected_model is None:
                self.selected_model = len(self.model_entries) - 1
            else:
                self.selected_model = max(0, self.selected_model - 1)
            self.model_offset = min(self.model_offset, self.selected_model)
        elif event.key == pygame.K_DOWN and self.model_entries:
            self._blur_model_name()
            if self.selected_model is None:
                self.selected_model = 0
            else:
                self.selected_model = min(
                    len(self.model_entries) - 1, self.selected_model + 1
                )
            self.model_offset = max(self.model_offset, self.selected_model - 4)

    def draw(self):
        toast = self.toast if time.monotonic() < self.toast_until else None
        lock = getattr(self.session, "lock", nullcontext())
        modal_open = (
            self.model_menu_open or self.reward_menu_open or self.game_menu_open
        )
        # Copy under the lock, draw outside it. The PPO callback takes this
        # same lock on every step, so rendering while holding it left the
        # trainer -- and all eight environment processes behind it -- waiting
        # on charts.
        with lock:
            data = self.session.data.copy()
        self.renderer.render(self.surface, data, None if modal_open else toast)
        if self.reward_menu_open:
            self.reward_regions = self.renderer.draw_reward_menu(
                self.surface, self.reward_config, self.reward_config_path
            )
            if toast:
                self.renderer._draw_toast(self.surface, toast)
        if self.game_menu_open:
            self.game_regions = self.renderer.draw_game_menu(
                self.surface, self.game_config
            )
            if toast:
                self.renderer._draw_toast(self.surface, toast)
        if self.model_menu_open:
            self._discover_models()
            self.model_regions = self.renderer.draw_model_menu(
                self.surface,
                self.model_name,
                self.model_entries,
                self.selected_model,
                self.model_offset,
                self.model_name_focused,
                self.model_name_cursor,
                self.model_name_selected,
            )
            if toast:
                self.renderer._draw_toast(self.surface, toast)
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        # Input is polled at 60Hz so clicks and drags stay responsive; only the
        # redraw is throttled, since nothing on screen changes faster than
        # DASHBOARD_FPS can show.
        frame_interval = 1.0 / DASHBOARD_FPS
        next_draw = 0.0
        try:
            while running:
                dt = min(clock.tick(60) / 1000.0, 0.1)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if self.model_menu_open:
                            self._model_key(event)
                        elif self.reward_menu_open and event.key == pygame.K_ESCAPE:
                            self.reward_menu_open = False
                        elif self.game_menu_open and event.key == pygame.K_ESCAPE:
                            self.game_menu_open = False
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                    elif event.type == pygame.TEXTINPUT and self.model_menu_open:
                        self._insert_model_text(event.text)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._click(event.pos)
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self.dragging_slider = None
                    elif event.type == pygame.MOUSEMOTION and self.dragging_slider:
                        kind, spec, track = self.dragging_slider
                        if kind == "reward":
                            self._set_reward_from_position(spec, track, event.pos)
                        else:
                            self._set_game_from_position(spec, track, event.pos)
                    elif event.type == pygame.MOUSEWHEEL and self.model_menu_open:
                        max_offset = max(0, len(self.model_entries) - 5)
                        self.model_offset = max(
                            0, min(max_offset, self.model_offset - event.y)
                        )
                        if self.selected_model is not None:
                            self.selected_model = max(
                                self.model_offset,
                                min(self.selected_model, self.model_offset + 4),
                            )
                    elif event.type == pygame.VIDEORESIZE:
                        size = (
                            max(event.w, MIN_WINDOW_SIZE[0]),
                            max(event.h, MIN_WINDOW_SIZE[1]),
                        )
                        self.surface = pygame.display.set_mode(size, pygame.RESIZABLE)
                self.session.tick(dt)
                self._drain_session_events()
                now = time.monotonic()
                if now >= next_draw:
                    # Advance from the deadline, not from now, so the cadence
                    # does not drift; clamped so a stall cannot leave a backlog.
                    next_draw = max(now, next_draw + frame_interval)
                    self.draw()
        finally:
            pygame.key.stop_text_input()
            self.session.close()
            pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-root", type=Path, default=Path("history/dodge/gymnasium")
    )
    parser.add_argument("--run-id", default="cnn-image-ddqn-dashboard-001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        type=int,
        default=250_000,
        help="DDQN decisions in this dashboard session",
    )
    parser.add_argument("--stack-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--target-sync-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for the policy update: auto, cpu or cuda",
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="cnn-image-ddqn",
        help="native model variant shown by the dashboard",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="optional directory for named DDQN checkpoints",
    )
    parser.add_argument(
        "--difficulty", choices=("easy", "normal", "hard"), default="easy"
    )
    parser.add_argument("--patterns", action="store_true")
    parser.add_argument(
        "--powerups", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--normal-enemies", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--kamikaze-enemies", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--spawn-rate-percent",
        type=int,
        default=100,
        help="enemy and pattern cadence, from 50 to 150 percent",
    )
    parser.add_argument(
        "--lives",
        type=int,
        default=0,
        help="training only: lethal collisions that end the "
        "episode without ending the cartridge run; "
        "patterns restart from their beginning "
        "(default 0)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="render one dashboard frame to this PNG and exit",
    )
    parser.add_argument(
        "--replay-host",
        default=None,
        help="browser replay bind address; defaults to this host's Tailscale IPv4",
    )
    parser.add_argument(
        "--replay-port",
        type=int,
        default=8890,
        help="browser replay HTTP port",
    )
    parser.add_argument(
        "--replay-steps",
        type=int,
        default=360,
        help="maximum native decisions generated for Watch Agent",
    )
    args = parser.parse_args(argv)

    if args.screenshot:
        # Draws sample telemetry, so a frame can be produced without training.
        pygame.init()
        surface = pygame.Surface(WINDOW_SIZE)
        DashboardRenderer().render(surface, sample_snapshot(seed=args.seed))
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, args.screenshot)
        pygame.quit()
        print(f"Saved dashboard screenshot to {args.screenshot}")
        return 0

    from .session import DodgeDDQNSession

    session = DodgeDDQNSession(
        history_root=args.history_root,
        run_id=args.run_id,
        seed=args.seed,
        total_steps=args.steps,
        stack_size=args.stack_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        update_every=args.update_every,
        target_sync_interval=args.target_sync_interval,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        device=args.device,
        model_dir=args.models_dir,
        difficulty={"easy": 1, "normal": 2, "hard": 3}[args.difficulty],
        patterns=args.patterns,
        powerups=args.powerups,
        normal_enemies=args.normal_enemies,
        kamikaze_enemies=args.kamikaze_enemies,
        spawn_rate_percent=args.spawn_rate_percent,
        lives=args.lives,
        architecture=args.architecture,
        replay_host=args.replay_host,
        replay_port=args.replay_port,
        replay_steps=args.replay_steps,
    )
    TrainingDashboard(session).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
