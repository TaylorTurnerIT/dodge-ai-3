# Batch-size screen and pixel controls

L1 implementation → L2 fixed-data training → L3 calibration and diagnostics. Continue user-authorized iteration; no collection or controller work.

Same frozen diverse corpus, seed42, reference model, float32, AdamW5e-5/decay1e-3/clip1, SIGReg0.09. New batch32 run512 updates retains128-step checkpoint. Compare128 with prior batch8/512 at4096 sampled windows; compare512 at equal optimizer updates, with4× exposure. No claim that batch statistics alone cause any difference. Pinned reference uses batch128;32 is a bounded intermediate experiment on T4.

Both retained checkpoints get original/calibrated copies and separate256-update batch8 decoders. Encoder calibration uses training windows only. Existing batch8 calibrated model/decoder supplied read-only for identical pixel diagnostics; no refit. Preserve all checkpoint/decoder/data hashes. All validation windows and per-episode metrics reported, not just best snapshots.

Pixel controls:32×32 area targets; train-current-frame mean image; actual-current persistence. Global errors plus error restricted to pixels whose mean RGB absolute change exceeds1/255. Mask derived only for evaluation, never passed to learners or used in a training loss. Counts and null for zero changing pixels. Includes trails/any changing graphics; it is not an object mask. Static score/background detail can be reconstructed by the mean baseline; test whether conditioning adds moving-scene information.

One fresh T4,3000s worker;512 model updates,1024 decoder updates total. Full Python/Ruff, remote variant tests and32-step legacy smoke. Stop on source defect/OOM/nonfinite; no fallback changing protocol during run. Freeze code and baseline/corpus hashes before training. After retrieval inspect both checkpoint steps, calibration/decoder links, pixel/dynamics controls and no-scroll square dashboard.

Frozen baseline: checkpointSHA `875050d70f298a33d17a7a8d56ed6c600d1457fc685b7d7151eda9968d796b5f`; decoderSHA `f030a80f2d59ad11a73262460af21d96afedfb3f557cfb522d0668516c20565a`. Imported corpus manifestSHA `7769398938b2be30934779fc5851e767653df7b31b2e7fad7a2a0adcd80e6a1f`. No new game transitions.

L1 accepted:323 local tests passed in66.33s; Ruff, Python script compilation and diff whitespace checks passed. Luna max implementation reviewed by parent. Tests prove sampler prefix equality, retained checkpoint immutability, train-only mean/mask denominators, frozen modes/RNG and nonfinite rejection. Run identity `lewm-batch32-20260915-v1`; source and artifacts recorded by launcher.

## Results: L2/L3 complete

Batch32 training completed512 updates in565.810s on Tesla T4. All four diagnostic conditions and the unchanged old calibrated decoder were evaluated. Both calibrated views pass browser checks at1280×720 and1366×768: four square images fully visible, both tabs, no scrolling/page errors. Images remain blurry; no reliable player/enemy delineation claimed.

Lower prediction/persistence ratio is better;1 is the persistence baseline. Each latent comparison uses its own encoder target space. Wrong-action penalty compares deliberately changed actions with correct actions on the same observed history.

| Batch / updates / condition | Train latent ratio | Validation latent ratio | Validation wrong-action penalty | Changed-pixel prediction MSE |
|---|---:|---:|---:|---:|
| 8 / 512 prior calibrated | 1.254339 | 0.996191 | 4.732% | 0.046285 |
| 32 / 128 original | 430.947012 | 160.704877 | 0.001% | 0.088921 |
| 32 / 128 calibrated | 0.796148 | 0.953074 | 0.255% | 0.043240 |
| 32 / 512 original | 617.689118 | 198.754947 | -0.001% | 0.070292 |
| 32 / 512 calibrated | 0.208404 | 0.994907 | 9.615% | 0.042836 |

The128-update checkpoint shares exactly the same4096 sampled-window sequence and sampler state as the previous batch8/512 run; both unit and retrieved-checkpoint checks passed. It improves the changed-region decoder metric by6.58% and beats latent persistence by4.69%, but its wrong-action penalty is only0.26%. The512-update checkpoint has4× sample exposure and a stronger train fit: prediction error0.2084×persistence; wrong actions3.493×correct-action error on training data. Its held-out wrong-action penalty is9.62%, but prediction only beats latent persistence by0.51%. Extra training did not produce a clear improvement in held-out prediction relative to persistence. Do not infer that normalization alone caused any gain or select either checkpoint as a validated controller model.

Pixel results use exactly56validation windows and1307 changing pixels among57344 pixel positions (2.279%). Changing masks include player trails and any changing graphics; they carry no object labels and enter no learner input/loss. Fixed train-mean baseline uses168training current frames.

| Pixel metric | Prior calibrated8/512 | New calibrated32/128 | New calibrated32/512 | Fixed train-mean baseline |
|---|---:|---:|---:|---:|
| Current reconstruction MSE | 0.019983 | 0.014680 | 0.020282 | 0.001531 |
| Next prediction MSE | 0.018874 | 0.013971 | 0.015449 | 0.001541 |
| Changing-region next MSE | 0.046285 | 0.043240 | 0.042836 | 0.049535 |

Actual-current pixel persistence next MSE0.000827913; changing-region MSE0.0363243. The final calibrated prediction improves changing-region error by7.45% versus the old decoder, in all four validation episodes (one only slightly), and by13.52% versus the mean-image control. It remains17.93% worse than pixel persistence in those regions. Current reconstruction worsens1.49% overall; global prediction error remains much worse than the fixed mean. Recognizable static score placement is therefore insufficient evidence of retained moving entities. These decoder results combine representation, decoder capacity and fitting budget; they do not isolate which component limits image quality.

Provenance:

- Source commit`abab212`; archiveSHA `ea4c00c4fdbaffa988ad6764301c1fca2eaa676f9b61b95957a849139edae1c7`.
- Torch2.11.0+cu128, CUDA12.8; peak allocated6,590,788,608bytes. Final train-mode loss0.188991, prediction loss0.023568, preclip gradient norm0.835027. Feature rank across different training batch sizes is not a matched comparison.
-323 local tests;104 remote variant tests; Ruff, script compilation and diff checks passed. Legacy32-step CUDA smoke completed with expected bounded-smoke warnings.
- Only encoder running_mean/running_var differ between each original/calibrated pair. All learned tensors preserved; all checkpoint/decoder/snapshot hashes match. Both calibrated checkpoints inference-only; old model/decoder files unchanged. Four decoders×256updates verified.
- Run roots: `history/dodge/gymnasium/pixel-repr-ddqn/lewm-batch32-20260915-v1`, `-calibrated`, `-step128`, `-step128-calibrated`. `parent-verification.json` under the job root records complete metrics and checkpoint hashes; review PNGs retained beside it.766,671,495-byte archive retrieved; T4 session terminated.

L2/L3 engineering acceptance complete. Scientific P4/P6 and controller work remain deferred. Next bounded diagnostic should separate decoder limitations from representation retention; broader practice coverage and further model training should be a separate subsequent experiment. No automatic longer run.

Final visual pass covers snapshots1,3,5,7, one from each validation episode. Fixed HUD placement visible; player/enemy outlines remain indistinct. Additional review PNGs stored in job root.
