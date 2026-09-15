# Large practice corpus

Scope: SPEC §Q. Expand coverage before further reconstruction fitting.

1. **Implementation:** deterministic recipes, resumable native collector, bounded-memory sequence loader. Preserve historical tiny corpus. Expose raw encoder CLS for a later matched reconstruction comparison.
2. **Pilot:** validate a small real capture; inspect actions, boundaries, compression and throughput. Fix defects before source freeze.
3. **Collection:** 4,096 training and 512 validation episodes, 128 decisions each, four native frames per decision. Native 128×128 RGB. Up to four CPU workers. Training seeds 4000–8095; validation seeds 16000–16511. No model updates.
4. **Audit:** stream through every episode; verify arrays, hashes, counts and split separation. Produce representative images and a coverage summary. Publish the complete corpus only after validation.
5. **Later fitting:** separately scoped Colab T4 run. Compare raw CLS and projected states on identical examples and budgets. Do not feed raw CLS into the existing projected-state predictor.

Practice varies player starts and action sequences, stationary and moving enemies, enemy sizes and paths, and permanent patterns. Invulnerable scenes build visual and motion coverage; they do not teach death or certify survival behavior. Native practice support limits which enemy types are available.

The learner receives pixels and actions, plus sequence indexing metadata. Recipe coordinates remain capture provenance. Whole episodes stay in one split. Validation covers unseen recipes within the same practice families; it is not a separate out-of-distribution benchmark.

Expected full budget: 589,824 transitions and 594,432 stored frames. Report actual totals after collection. Memory use must stay bounded as episode count grows. Keep collection and future model training as separate phases.

The paper's Appendix D specifies the last encoder CLS token as the visualization decoder input. `LeWorldModel.encode_cls(pixels)` now exposes that tensor; `encode_representation(pixels, representation="cls")` and `representation="projected"` select probe inputs. Existing `encode()` and world-model prediction behavior remain unchanged. Source: `references/paper/lewm-v3.md`, “Decoder (Visualization Only).”

Load published sequences with:

```python
from pathlib import Path
from dodge_native_game.variants.pixel_repr_ddqn.large_dataset import LargePixelSequenceDataset

dataset = LargePixelSequenceDataset(Path("corpus"), split="train", cache_size=4)
window = dataset[0]
# pixels: uint8[4, 3, 128, 128]; actions: int64[3]
# episode_id and start identify the window; they are not model features.
```

The historical training launchers still target the tiny corpus. A future training phase must explicitly use the large loader and freeze its sampling protocol; do not substitute the new directory into an old bounded launcher.

The first full collection was withheld: one validation recording matched a training recording byte for byte. Difficulty differed between splits but did not change that empty-arena recording. The corrected planner uses difficulty 1 throughout. Version 2 imports the unchanged training captures only after checking their configurations, receipts, original source hash, capture-function AST, dependencies and native binary; it records that lineage and collects validation again. Version 1 remains unpublished. The pixel-hash rejection remains enabled.

Published dataset: `history/dodge/gymnasium/pixel-repr-ddqn/large-practice-20260914-v2`. Native 128×128 RGB, 4,096 training and 512 validation episodes, 589,824 transitions and 594,432 frames. Each of 16 families has 256 training and 32 validation recipes. Both splits cover all 64 cells of an 8×8 player-start grid, enemy counts 0–6 and sizes 2–16; training includes all 39 permanent patterns. Whole-episode hashes and recipe identities have zero cross-split overlap.

Implementation checks: 365 tests, Ruff clean, bounded legacy smoke, native collection/import pilots. The published dataset passed full streaming validation. Sampling 64 windows per split confirmed four cached episodes (25,366,528 array bytes), with 516,096 training and 64,512 validation windows. Version 2 import/collection/publication took 534.08 seconds; original training captures came from the earlier collection.

Reproduce from scratch with the frozen corrected source:

```bash
python -m dodge_native_game.variants.pixel_repr_ddqn.large_practice \
  --output history/dodge/gymnasium/pixel-repr-ddqn/large-practice-reproduction \
  --workers 4
```

Source freeze: `4e38fcc`; reused training source: `1469782`. Dataset manifest SHA256: `683a524eee34030517713d3e29d0f06959c2cf83186313197983b9a9ffef6afa`. Adjacent source archives, dataset tar, timing record and audit directory retain provenance. No model fitting ran during this phase.

[Download dataset](http://100.100.169.122:8791/large-practice-20260914-v2.tar) · [Training gallery](http://100.100.169.122:8791/large-practice-20260914-v2-audit/train-gallery.png) · [Validation gallery](http://100.100.169.122:8791/large-practice-20260914-v2-audit/validation-gallery.png). Artifact server binds only the Tailscale address on port 8791; existing model dashboard remains on 8790.
