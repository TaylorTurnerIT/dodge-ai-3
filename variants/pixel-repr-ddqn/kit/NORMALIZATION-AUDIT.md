# Frozen normalization audit

N1 implementation → N2 evaluation. User requested deeper diagnosis after negative D3/D4 result. Use build/spec workflow; no retraining or controller work.

Checkpoint `9aa1c17b3b897272baf78617a399a2fc50ae28f8981563cf1a515a06306abf9a`; corpus manifest `e8476999274c7ae43d20ae03ecb4c8e37dcdb32202585ff585ae735f7b2abcee`. Reconstruct final batch indices from seed43 and512 draws, assert saved sampler RNG matches. Compare identical final-batch8 windows, plus all6 validation windows. Cache raw CLS once because encoder dropout is zero; predictor retains its existing source implementation.

Factorial conditions: encoder BN uses stored/batch moments × predictor BN uses stored/batch moments × predictor dropout off/on. Dropout uses seeds2026/2027/2028; same mask for original/wrong actions. Report each of three prediction positions, all-position mean, last-position persistence ratio and action sensitivity. Raw errors across different encoder modes are different target spaces; they are not a common-scale quality comparison.

Additional tests: substitute final-frame CLS with first-frame CLS while preserving observed context; measure earlier-latent dependence with encoder BN stored/batch modes. Record pre-BN mean/variance shifts against running statistics. Disposable copies replace encoder-only, predictor-only or both sets of moments from all11 TRAIN windows, weighted as windows. Use population variance (explicitly different from EMA unbiased running estimates); no parameter updates or validation moment fitting. Predictor-only intervention keeps encoder/target space fixed. No corrected checkpoint exported.

Budget: one Colab T4, <=1800s remote execution, <=1GiB upload archive; zero optimizer/native/decoder updates. N1 tests use small synthetic tensors locally. Source frozen before N2; errors stop affected run and preserve evidence under new identity. Diagnostic mode changes do not establish a deployable fix. Dashboard retains original negative-run images.

PyTorch documents training-batch normalization and evaluation running statistics, including different training/running variance estimators: [BatchNorm1d, version2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm1d.html). Pinned upstream `jepa.py` and `module.py` use batch×time flattened projectors; local source crosswalk remains applicable. These are hypotheses to test, not evidence of cause by themselves.

Source audit (Luna max, parent reviewed): local `_apply_projector` flattens batch/time like pinned `jepa.py:34-55`; local attached-target loss matches pinned `train.py:27-43`; no BN/dropout/loss semantic divergence found. Batch8 gives32 encoder and24 predictor normalization rows per update, versus512/384 at reference batch128. Adjacent frames and overlapping windows are correlated. Local action one-hot encoding and small-data/float32 protocol remain explicit adaptations.

N1 verification: full305 Python tests passed in64.79s; focused2 tests repeated after device-scoped RNG adjustment; Ruff and generated remote-script syntax checks pass. No native/model/trainer modifications.
