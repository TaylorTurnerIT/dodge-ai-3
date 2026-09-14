# Local LeWM references

- Upstream clone: [le-wm](le-wm/README.md), pinned commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Paper: [Markdown](paper/lewm-v3.md), [original HTML](paper/lewm-v3.html), version3, 2026-06-03.
- [Provenance manifest](manifest.json) records hashes. Upstream code MIT; paper CC BY4.0, authors credited in Markdown.

The clone is ignored by parent Git to avoid an accidental gitlink. Recreate with `git clone https://github.com/lucas-maes/le-wm.git references/le-wm`, then `git -C references/le-wm checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. Runtime code must not depend on this checkout; reused code belongs in variant with MIT notice. Paper Markdown is automatically converted, not an edited scientific source; use HTML for exact figures/math.

Reference paths: `module.py` (SIGReg/AdaLN/predictor), `jepa.py` (CLS/rollout), `train.py` (attached-target two-term loss), `config/train/model/lewm.yaml` (architecture), `config/train/lewm.yaml` (optimizer/defaults), `utils.py` (preprocessing).
