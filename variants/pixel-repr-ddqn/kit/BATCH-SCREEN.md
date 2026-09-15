# Batch-size screen and pixel controls

L1 implementation → L2 fixed-data training → L3 calibration and diagnostics. Continue user-authorized iteration; no collection or controller work.

Same frozen diverse corpus, seed42, reference model, float32, AdamW5e-5/decay1e-3/clip1, SIGReg0.09. New batch32 run512 updates retains128-step checkpoint. Compare128 with prior batch8/512 at4096 sampled windows; compare512 at equal optimizer updates, with4× exposure. No claim that batch statistics alone cause any difference. Pinned reference uses batch128;32 is a bounded intermediate experiment on T4.

Both retained checkpoints get original/calibrated copies and separate256-update batch8 decoders. Encoder calibration uses training windows only. Existing batch8 calibrated model/decoder supplied read-only for identical pixel diagnostics; no refit. Preserve all checkpoint/decoder/data hashes. All validation windows and per-episode metrics reported, not just best snapshots.

Pixel controls:32×32 area targets; train-current-frame mean image; actual-current persistence. Global errors plus error restricted to pixels whose mean RGB absolute change exceeds1/255. Mask derived only for evaluation, never passed to learners or used in a training loss. Counts and null for zero changing pixels. Includes trails/any changing graphics; it is not an object mask. Static score/background detail can be reconstructed by the mean baseline; test whether conditioning adds moving-scene information.

One fresh T4,3000s worker;512 model updates,1024 decoder updates total. Full Python/Ruff, remote variant tests and32-step legacy smoke. Stop on source defect/OOM/nonfinite; no fallback changing protocol during run. Freeze code and baseline/corpus hashes before training. After retrieval inspect both checkpoint steps, calibration/decoder links, pixel/dynamics controls and no-scroll square dashboard.

Frozen baseline: checkpointSHA `875050d70f298a33d17a7a8d56ed6c600d1457fc685b7d7151eda9968d796b5f`; decoderSHA `f030a80f2d59ad11a73262460af21d96afedfb3f557cfb522d0668516c20565a`. Imported corpus manifestSHA `7769398938b2be30934779fc5851e767653df7b31b2e7fad7a2a0adcd80e6a1f`. No new game transitions.

L1 accepted:323 local tests passed in66.33s; Ruff, Python script compilation and diff whitespace checks passed. Luna max implementation reviewed by parent. Tests prove sampler prefix equality, retained checkpoint immutability, train-only mean/mask denominators, frozen modes/RNG and nonfinite rejection. Run identity `lewm-batch32-20260915-v1`; source and artifacts recorded by launcher.
