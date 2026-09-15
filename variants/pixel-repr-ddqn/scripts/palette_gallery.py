"""Publish matched, native-pixel palette diagnostics from completed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(history: Path, run_id: str, baseline: str, output: Path) -> None:
    data: dict = {}
    for arm, prefix in (
        ("mse", baseline),
        ("ce", run_id + "-ce"),
        ("bce", run_id + "-bce"),
    ):
        data[arm] = {}
        for step in (512, 2048, 8192):
            data[arm][str(step)] = {}
            for mode in ("cls", "projected"):
                run = history / f"{prefix}-{mode}"
                evaluation = json.loads((run / f"evaluation-{step}.json").read_text())
                entries = {}
                for split in ("train", "validation"):
                    examples = [
                        e for e in evaluation["examples"] if e["split"] == split
                    ]
                    for ordinal, example in enumerate(examples):
                        images = run / "images" / f"step-{step}"
                        entry = {
                            "episode": example["episode_id"],
                            "frame": example["frame_index"],
                        }
                        for name, suffix in (
                            ("observed", "observed"),
                            ("normal", "reconstructed"),
                            ("wrong", "wrong-latent"),
                        ):
                            path = images / f"{split}-{ordinal:02d}-{suffix}.png"
                            if not path.is_file():
                                raise FileNotFoundError(path)
                            # Output lives in history so all references remain local.
                            entry[name] = path.relative_to(history).as_posix()
                        key = split + "|" + example["recipe_family"]
                        if key in entries:
                            raise ValueError(f"duplicate example family: {key}")
                        entries[key] = entry
                data[arm][str(step)][mode] = {
                    "entries": entries,
                    "normal": evaluation["splits"],
                    "wrong": evaluation["wrong_latent_control"],
                }
    for step in data["mse"]:
        for mode in ("cls", "projected"):
            base = data["mse"][step][mode]["entries"]
            for arm in ("ce", "bce"):
                entries = data[arm][step][mode]["entries"]
                if set(base) != set(entries) or any(
                    (base[k]["episode"], base[k]["frame"])
                    != (entries[k]["episode"], entries[k]["frame"])
                    for k in base
                ):
                    raise ValueError("comparison examples are not matched")
    if output.parent.resolve() != history.resolve():
        raise ValueError("output must be directly inside history root")
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_suffix(".html").read_text()
    output.write_text(template.replace("__DATA__", encoded))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id", default="lewm-large-probe-20260915-v2")
    args = parser.parse_args()
    output = args.history_root / f"{args.run_id}-comparison.html"
    build(args.history_root, args.run_id, args.baseline_run_id, output)
    print(output)


if __name__ == "__main__":
    main()
