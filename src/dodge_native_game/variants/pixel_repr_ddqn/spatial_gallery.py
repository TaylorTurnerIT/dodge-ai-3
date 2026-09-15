"""Write a square, single-screen matched spatial readout gallery."""

from pathlib import Path


def write_gallery(history: Path, run_id: str) -> Path:
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in run_id
    ):
        raise ValueError("invalid run ID")
    template = Path(__file__).with_suffix(".html").read_text()
    target = history / f"{run_id}-comparison.html"
    target.write_text(template.replace("__RUN__", run_id))
    return target
