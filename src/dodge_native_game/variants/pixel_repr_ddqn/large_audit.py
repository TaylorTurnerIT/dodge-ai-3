"""Small visual and action audit of a published large practice corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .large_dataset import read_dataset_metadata


def audit_corpus(root: Path, output: Path) -> dict[str, object]:
    """Summarize executed actions and render first/middle/last sample frames.

    This reads one episode at a time. Pixel change is a visual coverage probe,
    not an entity detector or a measure of representation quality.
    """
    metadata = read_dataset_metadata(root)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"dataset": str(root.resolve()), "splits": {}}
    split_summary = summary["splits"]
    assert isinstance(split_summary, dict)
    for split, records in metadata.records.items():
        actions = np.zeros(9, dtype=np.int64)
        change_sum, transitions, frames, byte_count = 0.0, 0, 0, 0
        # Evenly spaced episodes cover the complete split, without cherry-picking.
        selected = set(
            np.linspace(0, len(records) - 1, min(12, len(records)), dtype=int)
        )
        gallery = Image.new("RGB", (3 * 256, len(selected) * 282), "#171717")
        draw = ImageDraw.Draw(gallery)
        row = 0
        for index, record in enumerate(records):
            with np.load(record.path, allow_pickle=False) as episode:
                pixels = episode["pixels"]
                executed = episode["actions"]
                actions += np.bincount(executed, minlength=9)
                changed = np.any(pixels[1:] != pixels[:-1], axis=1)
                change_sum += float(changed.mean(axis=(1, 2)).sum())
                transitions += len(executed)
                frames += len(pixels)
                byte_count += record.path.stat().st_size
                if index in selected:
                    for column, frame_index in enumerate(
                        (0, len(pixels) // 2, len(pixels) - 1)
                    ):
                        frame = Image.fromarray(pixels[frame_index].transpose(1, 2, 0))
                        gallery.paste(
                            frame.resize((256, 256), Image.Resampling.NEAREST),
                            (column * 256, row * 282 + 26),
                        )
                        draw.text(
                            (column * 256 + 4, row * 282 + 6),
                            f"{record.episode_id} frame {frame_index}",
                            fill="white",
                        )
                    row += 1
        gallery.save(output / f"{split}-gallery.png")
        split_summary[split] = {
            "episodes": len(records),
            "frames": frames,
            "transitions": transitions,
            "compressed_episode_bytes": byte_count,
            "action_counts": actions.tolist(),
            "mean_changed_pixel_fraction": change_sum / max(1, transitions),
            "gallery": f"{split}-gallery.png",
        }
    (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_corpus(args.root, args.output), indent=2))
