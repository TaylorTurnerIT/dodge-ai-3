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
