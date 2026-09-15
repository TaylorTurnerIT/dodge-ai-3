"""Write the current/next frame predicted-decode comparison gallery."""

from __future__ import annotations

import json
import re
from pathlib import Path

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,70}\Z")


def _validate_run_id(value: str) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ValueError("invalid run ID")
    return value


def write_gallery(history: Path, run_id: str) -> Path:
    """Write a relative-path five-panel gallery for one future-decode run."""

    run_id = _validate_run_id(run_id)
    history = Path(history)
    template = Path(__file__).with_suffix(".html").read_text()
    rendered = template.replace("__RUN__", json.dumps(run_id))
    target = history / f"{run_id}-comparison.html"
    target.write_text(rendered)
    return target


__all__ = ["write_gallery"]
