"""Render matched RGB-input and palette-input CLS diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(history: Path, run_id: str) -> Path:
    data = {}
    for arm in ("rgb", "palette"):
        data[arm] = {}
        run = history / f"{run_id}-{arm}-cls"
        for step in (512, 2048, 8192):
            e = json.loads((run / f"evaluation-{step}.json").read_text())
            entries = {}
            for split in ("train", "validation"):
                for i, item in enumerate(
                    x for x in e["examples"] if x["split"] == split
                ):
                    entry = {
                        "episode": item["episode_id"],
                        "frame": item["frame_index"],
                    }
                    for name, suffix in [
                        ("observed", "observed"),
                        ("normal", "reconstructed"),
                        ("wrong", "wrong-latent"),
                    ]:
                        p = (
                            run
                            / "images"
                            / f"step-{step}"
                            / f"{split}-{i:02d}-{suffix}.png"
                        )
                        if not p.is_file():
                            raise FileNotFoundError(p)
                        entry[name] = p.relative_to(history).as_posix()
                    entries[split + "|" + item["recipe_family"]] = entry
            data[arm][str(step)] = {
                "entries": entries,
                "normal": e["splits"],
                "wrong": e["wrong_latent_control"],
            }
    for step in data["rgb"]:
        a, b = data["rgb"][step]["entries"], data["palette"][step]["entries"]
        if a.keys() != b.keys() or any(
            (a[k]["episode"], a[k]["frame"]) != (b[k]["episode"], b[k]["frame"])
            for k in a
        ):
            raise ValueError("Input gallery examples do not match")
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    output = history / f"{run_id}-comparison.html"
    output.write_text(
        Path(__file__).with_suffix(".html").read_text().replace("__DATA__", encoded)
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(build(args.history_root, args.run_id))
