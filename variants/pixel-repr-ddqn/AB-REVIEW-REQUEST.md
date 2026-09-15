# Review request: predicted next-frame decode (§AB) — is pooling the fix?

Date: 2026-09-15. Branch: `experiments/pixel-repr-ddqn` (public repo
`TaylorTurnerIT/dodge-ai-3`). HEAD at review time: `7c7d0ea`.
Reviewer: check out this branch, read the files below, and advise on direction.
Do not start new training runs; this request is for analysis and a plan.

## The question

Our frozen world model recovers the current frame near-perfectly from patch
tokens, but decoding its one-step-ahead prediction yields noise that is no
better than decoding the wrong frame. Two candidate explanations compete:

- **H1 (readout artifact):** the predicted latent ẑ carries entity detail,
  but our broadcast readout + weighted-MSE training cannot extract or
  localize it (the loss hedges the rare class everywhere).
- **H2 (predictor bottleneck):** ẑ carries no forecast-specific detail at
  all — the predictor learned near-persistence — so no readout can fix it.

Our proposal is (a) launch the already-implemented §AA pooling study to test
whether any 192-dim readout of patch tokens retains entities, and (b) run a
cheap predictor action-sensitivity check on the frozen model. We want your
judgment on whether that plan is right, and what the fix is under each
hypothesis.

## Codebase pointers

- Contract: `variants/pixel-repr-ddqn/SPEC.md` (§Z, §AA, §AB; bug rows B64–B67).
- Study code: `src/dodge_native_game/variants/pixel_repr_ddqn/future_decode.py`
  (window selection, teacher-forced ẑ features, readout fit, 5-panel gallery),
  `future_gallery.py` + `future_gallery.html`.
- Shared fitter (extended with an optional `loss_kind`, §Z/§AA default
  unchanged): `spatial_fit.py` (`fit_spatial_decoders`).
- Colab T4 worker/launcher pair: `variants/pixel-repr-ddqn/scripts/colab_future_study.py`,
  `colab_future_worker.py`. Launch requires a clean tree; results download
  with checksum verification.
- Pooling study (implemented, frozen, NOT yet run): `pooling_readout.py`,
  `pooling_probe.py`, `scripts/colab_pooling_study.py`,
  `scripts/colab_pooling_worker.py`, kit card `kit/POOLING.md`.
- Tests: `tests/variant_pixel_repr_ddqn/test_future_decode.py`,
  `test_future_gallery.py`, `test_colab_future.py`, `test_lewm_dashboard.py`.
- Dashboard (serves the run; restarted 2026-09-15 on :8790):
  `src/.../dashboard.py`, `dashboard.html` (optional diff strip, hidden for
  runs without a diff frame).

## Evidence

World model: frozen §Y palette LeWM (`aee9698c…`), 192-dim CLS space,
6-layer causal predictor, history 3, 9 actions. No world updates anywhere.

§Z (current-frame decode, frozen encoder, matched heads, T4):
CLS broadcast changed-cream recall 0.51%; full patch tokens 99.96%;
changed-region MSE patch 0.00051 vs CLS 0.14157. Detail exists in patch
tokens, not in CLS.

§AB (this work; `lewm-future-decode-20260915-v1`, 2048 updates at ~58/s on
T4, 4096 train windows, balanced-bright weighted MSE 0.5 cream / 0.5 other,
world hash verified unchanged, launcher contract audits passed). 16
validation scenes, teacher-forced one step:

| condition | class err | changed-region err | cream recall | MSE |
|---|---|---|---|---|
| predicted ẑ readout | 0.3142 | 0.5639 | 0.5965 | 0.04943 |
| persistence latent, same head | 0.3191 | 0.5362 | 0.7152 | 0.04978 |
| wrong-latent control | 0.3211 | 0.5751 | 0.4905 | — |
| pixel copy baseline | 0.0072 | 1.0000 | 0.0000 | — |

Visually: current-frame decode is 1:1; predicted decode is cream speckle;
the diff map is mostly white. Per-color recall order is
[background, blue, cream]; background recall is 0.0 in every row — the head
never predicts background, consistent with H1 hedging, while the flat
predicted ≈ persistence ≈ wrong-latent ordering supports H2.

## Our read and proposed plan

1. Launch §AA pooling on Colab (mean / max / learned-attention / 4×4-grid
   readouts over frozen patch tokens, matched protocol). If no 192-dim
   pooling retains entities, the predictor's output bottleneck is
   architecturally insufficient and the fix is predicting spatial tokens,
   not a better CLS readout. If pooling succeeds, the fix is on the
   world-model side (predict into the successful pooled space).
2. Cheap frozen-model check first: measure ẑ action-sensitivity on moving
   validation windows (true-action vs wrong-action ẑ distance, vs zₜ
   distance). If ẑ ignores actions, H2 is confirmed without any fitting.

## Questions for you

1. Does the §AB evidence distinguish H1 from H2, or is a control missing?
   Which one would you add?
2. Is §AA pooling decisive for the "192 values suffice" question, or would
   you change its arms/budget first?
3. If H2 holds, what is the smallest world-model-side fix you would try
   (predictor objective, target space, action conditioning)?
4. Anything in the §AB protocol (mid-episode windows, weighted MSE, 2048
   updates, 16 reused-validation scenes) that undermines the conclusion?

## Viewing / reproducing

- Dashboard: serve the repo and open the variant dashboard; run
  `lewm-future-decode-20260915-v1` is listed with frame comparison,
  snapshot selector, and diff strip. File-based fallback:
  `history/dodge/gymnasium/pixel-repr-ddqn/lewm-future-decode-20260915-v1-standalone.html`
  (inlined images, works from `file://`; gitignored build artifact).
- Machine-readable: `history/dodge/gymnasium/pixel-repr-ddqn/lewm-future-decode-20260915-v1/report.json`
  (metrics under `predicted`, `persistence_control`,
  `wrong_latent_control`, `pixel_persistence_baseline`).
- Verify: `scripts/uv-run pytest` (467 passing at HEAD), `scripts/uv-run ruff check .`.
- Constraints: implementation vs training phases stay separate per
  `variants/pixel-repr-ddqn/AGENTS.md`; no world-model updates, no new
  collection, frozen code/data/protocol for any T4 run; retrieved archives
  keep their checksums (the local `visualizations.json` regen is a
  presentation transform of retrieved PNGs, round-trip exact, audit files
  untouched — see SPEC B67).
