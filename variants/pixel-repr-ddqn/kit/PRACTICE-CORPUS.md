# Practice corpus import

SPEC §D separates importer implementation (D1), bounded artifact verification (D2), and future training (D3). This delivery imports existing recordings only. It does not run the game or fit a model.

The importer takes explicit train and validation recording directories and publishes a new dataset accepted by `PixelSequenceDataset`. It preserves episode NPZ bytes and archives each source manifest. Script coordinates, goal images, success flags and configuration remain provenance; learner windows still contain only pixels/actions plus existing indexing metadata.

D1 verification scope: new artifact importer only; native game, model, trainer, dashboard and legacy modules unchanged. Run focused importer tests and full Python/Ruff regression; D2 supplies the bounded data smoke. No additional GPU/model run belongs to this data-only delivery.

Input checks cover source hashes, resolved configuration, cadence, action trace, episode length and terminal flags. Seeds must be unique, and identical episode files cannot appear across splits. Invulnerable and lethal dynamics cannot share this corpus. The existing MVP cap remains 256 total transitions; both splits need at least one three-action window. Short death or timeout episodes can remain in the archive but do not create longer windows.

Different seeds alone do not ensure distinct scripted scenes. D2 uses a different validation trajectory as well as a different seed. This small check proves import and split handling, not task generalization or sufficient scene coverage for representation training.

After implementation passes, run:

```bash
scripts/uv-run --extra native --extra training --extra lewm python \
  -m dodge_native_game.variants.pixel_repr_ddqn.practice_dataset \
  --output history/dodge/gymnasium/pixel-repr-ddqn-datasets/practice-example \
  --train PATH_TO_TRAIN_CAPTURE \
  --validation PATH_TO_VALIDATION_CAPTURE
```

D2 protocol: reuse G2 example as train; generate one validation capture with seed43, player start40,60, four right decisions followed by four neutral decisions, same configured enemies and invulnerability. Maximum8 new native decisions. Verify both splits load, NPZ bytes match sources, and windows reproduce recorded actions and RGB. No optimizer or controller work. Record evidence below after execution.

D1 evidence (2026-09-14): 15 focused importer tests; full Python suite291 passed in64.04s; Ruff and whitespace checks passed. Luna max task produced no code before interruption; parent implemented and reviewed importer/tests. Native/model/dashboard/legacy code unchanged.

D2 evidence: `history/dodge/gymnasium/pixel-repr-ddqn-practice/d2-validation-20260914/verification.json`. Eight new native decisions; imported21 total transitions;11 train and6 validation windows. Exact episode bytes and every window's RGB/actions matched sources. No model updates. Dataset ID `2450223a4b8a4352c4ed2cb2f2c5237d832e5c419dbd37f73d697672cb5f3d63`; manifest SHA256 `e8476999274c7ae43d20ae03ecb4c8e37dcdb32202585ff585ae735f7b2abcee`; importer SHA256 `995d71821268100bc0bfb7ca8269430d11d2693621f520400c98957b3ed1435a`. D1/D2 engineering gates complete under user continuation scope; D3 training remains deferred pending a frozen protocol. No representation-quality claim.

## Practice overfit protocol v1

User continuation authorizes D3a implementation → D3 training → D4 diagnostics. This is a small-capacity/optimization diagnostic, not scientific P4/P6 acceptance. Freeze implementation before allocation. No collection or source changes during model/decoder fitting.

- Input: D2 corpus,21 transitions,11 train/6 validation windows; manifest SHA256 `e8476999274c7ae43d20ae03ecb4c8e37dcdb32202585ff585ae735f7b2abcee`.
- Model: fresh reference architecture,512 updates,batch8,seed42,float32, existing AdamW5e-5/decay1e-3/clip1/SIGReg0.09; Colab T4 only. No resume, controller, corpus growth or automatic extension.
- Decoder: after model training ends, freeze LeWM;256 updates,batch8, existing32×32 architecture/train-only fitting, fixed validation snapshots.
- Evaluation: every train and validation window; one-step latent MSE, same-space persistence MSE, wrong-action MSE, action sensitivity, prediction/persistence ratio. Zero denominator reported null. No open-loop or causal-control claim.
- Bound: one T4 session, remote worker timeout3000s, device16GiB ceiling, archive cap2GiB; nonfinite/OOM/error stops and preserves run. Tiny repeated data and actual batch8 limit SIGReg evidence; no accumulation=batch128 claim.
- Interpretation: prediction below persistence with wrong actions worse is encouraging within this tiny corpus; it is insufficient for learned object separation or generalization. Failure on train directs attention to optimization/representation/predictor before larger data. Both train and validation reported; checkpoint chosen at fixed512, not by validation score.
- Implementation checks: envelope rejects budget/device/profile/resume bypass; source numerical tests retained; dynamics alignment/persistence/wrong-action/zero denominator/no-mutation tests; full Python/Ruff checks. T4 worker repeats targeted tests and includes32-step legacy smoke after diagnostic fitting.

Launch after D3a closes: `python3 variants/pixel-repr-ddqn/scripts/colab_mvp.py --run-id lewm-practice-overfit-20260914-v1 --practice-dataset history/dodge/gymnasium/pixel-repr-ddqn-practice/d2-validation-20260914/corpus`.

D3a evidence: parent reviewed Luna evaluator, protocol guards and launcher;303 Python tests passed in63.94s; Ruff and whitespace checks pass. Source frozen before D3; reference numerical tests retained.
