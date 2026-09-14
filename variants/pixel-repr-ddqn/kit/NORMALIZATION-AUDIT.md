# Frozen normalization audit

N1 implementation → N2 evaluation. User requested deeper diagnosis after negative D3/D4 result. Use build/spec workflow; no retraining or controller work.

Checkpoint `9aa1c17b3b897272baf78617a399a2fc50ae28f8981563cf1a515a06306abf9a`; corpus manifest `e8476999274c7ae43d20ae03ecb4c8e37dcdb32202585ff585ae735f7b2abcee`. Reconstruct final batch indices from seed43 and512 draws, assert saved sampler RNG matches. Compare identical final-batch8 windows, plus all6 validation windows. Cache raw CLS once because encoder dropout is zero; predictor retains its existing source implementation.

Factorial conditions: encoder BN uses stored/batch moments × predictor BN uses stored/batch moments × predictor dropout off/on. Dropout uses seeds2026/2027/2028; same mask for original/wrong actions. Report each of three prediction positions, all-position mean, last-position persistence ratio and action sensitivity. Raw errors across different encoder modes are different target spaces; they are not a common-scale quality comparison.

Additional tests: substitute final-frame CLS with first-frame CLS while preserving observed context; measure earlier-latent dependence with encoder BN stored/batch modes. Record pre-BN mean/variance shifts against running statistics. Disposable copies replace encoder-only, predictor-only or both sets of moments from all11 TRAIN windows, weighted as windows. Use population variance (explicitly different from EMA unbiased running estimates); no parameter updates or validation moment fitting. Predictor-only intervention keeps encoder/target space fixed. No corrected checkpoint exported.

Budget: one Colab T4, <=1800s remote execution, <=1GiB upload archive; zero optimizer/native/decoder updates. N1 tests use small synthetic tensors locally. Source frozen before N2; errors stop affected run and preserve evidence under new identity. Diagnostic mode changes do not establish a deployable fix. Dashboard retains original negative-run images.

PyTorch documents training-batch normalization and evaluation running statistics, including different training/running variance estimators: [BatchNorm1d, version2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm1d.html). Pinned upstream `jepa.py` and `module.py` use batch×time flattened projectors; local source crosswalk remains applicable. These are hypotheses to test, not evidence of cause by themselves.

Source audit (Luna max, parent reviewed): local `_apply_projector` flattens batch/time like pinned `jepa.py:34-55`; local attached-target loss matches pinned `train.py:27-43`; no BN/dropout/loss semantic divergence found. Batch8 gives32 encoder and24 predictor normalization rows per update, versus512/384 at reference batch128. Adjacent frames and overlapping windows are correlated. Local action one-hot encoding and small-data/float32 protocol remain explicit adaptations.

N1 verification: full305 Python tests passed in64.79s; focused2 tests repeated after device-scoped RNG adjustment; Ruff and generated remote-script syntax checks pass. No native/model/trainer modifications.

Transport note: direct191MiB archive upload failed twice before remote execution. Host launcher now sends8MiB chunks and verifies reassembled SHA256; diagnostic source archive unchanged. Transport-only change does not alter N2 model/data/protocol identity.

## N2 findings

Completed on Tesla T4 with zero optimizer/native/decoder updates. Original checkpoint SHA256, all parameters/buffers and sampler identity verified unchanged. Two remote tests passed; session released. Raw evidence: `history/dodge/gymnasium/pixel-repr-ddqn-audits/normalization-audit-20260914-v1/audit.json`; source archive SHA256 `6f09b997a9356bfb58456268cc85955a6b20b18927489d213a51ee4e690eed26`. Matched final-batch indices `[10,1,5,7,2,10,9,6]`. Peak allocated237,274,624 bytes for cached-feature audit, not training capacity.

The encoder's stored normalization statistics explain most of this checkpoint's train/evaluation discrepancy. This is intervention evidence on one checkpoint, not proof of why the statistics became mismatched. Small correlated batches and changing feature distributions remain candidate causes. Upstream uses the same normalization mechanism; no local BN/dropout/loss semantic discrepancy was found.

| Disposable copy intervention | Train prediction/persistence | Validation prediction/persistence |
|---|---|---|
| Original stored statistics | 16.1043 | 10.8561 |
| Predictor statistics only | 25.3896 | 13.8197 |
| Encoder statistics only | 0.01628 | 1.14132 |
| Both projectors' statistics | 0.02187 | 1.13922 |

Below1 beats persistence. Calibration uses training windows only, population moments, dropout off, no learned-weight change. Changing encoder normalization changes the target representation: raw MSE across those rows is not a common-space comparison. Each ratio uses persistence in that row's own representation. Predictor-only calibration preserves the original target space and worsens prediction. Encoder-only calibration restores strong training fit, but validation still loses to persistence by14.1%; no generalization claim.

On the matched training batch, all-position MSE is2.8131 with stored statistics/dropout off, versus0.003829 with both projectors using batch statistics/dropout on. Turning dropout on while keeping stored statistics yields2.8136; dropout is not the dominant cause. Scoring only the last position gives2.6807 in ordinary evaluation, so position reduction does not explain the discrepancy either.

Encoder pre-BN training activations have mean absolute shift1.753 stored standard deviations; median current/stored variance ratio0.564. Predictor inputs under the original eval encoder also shift (0.769 stored standard deviations). Encoder-only train-population replacement makes training last-step error0.002726 versus0.032655 under wrong actions, roughly12× worse with wrong actions. On validation, wrong-action error0.374858 is marginally lower than correct-action0.375447: useful action conditioning has not generalized.

Future-substitution check: replacing only final-frame CLS changes earlier latents by MSE0.002729/max-absolute0.176657 under batch normalization; stored-stat evaluation changes them by exactly0. This confirms train-mode coupling across time, as expected from flattened batch×time normalization. It does not establish that the predictor exploited this coupling. Fixed train-only calibration does not consume validation/future frames at inference. Leaving the model in training mode is not an acceptable deployment fix.

Training data is one13-transition script:7 neutral,3 down-right,3 right actions. Validation is a different8-transition script:4 right,4 neutral. The result establishes a recoverable training fit and a failed small validation test; it cannot establish player/enemy separation, broad scene understanding or survival.

Recommended next implementation: explicit train-only normalization calibration provenance for checkpoint evaluation/export, with original/calibrated artifacts kept separate; fixed-stat causal inference regression; validation persistence and wrong-action gates on every checkpoint. Then broaden practice trajectories under a separately frozen collection/training protocol. No LayerNorm replacement, longer run or controller is justified by this audit alone. Current dashboard retains original checkpoint/decoder views; no recalibrated copy was exported or passed through a decoder fitted to the old latent distribution.
