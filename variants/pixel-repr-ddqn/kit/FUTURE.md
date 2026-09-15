# Predicted next-frame decode view

Follow SPEC §AB. The frozen §Y palette world moves one step into the future:
the causal predictor consumes observed context and executed actions, and a
fresh broadcast readout — architecturally identical to the §Z CLS head — is
fit on train-split predicted latents with balanced-bright weighted MSE (0.5
cream, 0.5 other, per-frame-then-batch).

1. AB1: implement window selection, teacher-forced one-step ẑ features,
   latent-persistence/pixel-persistence/wrong-latent controls, and the
   five-panel gallery (current observed/decoded, next observed/decoded,
   high-contrast diff map bottom right). Review, test, freeze, commit.
2. AB2: fit the predicted readout 512/2048 on one T4 via
   `scripts/colab_future_study.py` (one deterministic mid-episode window per
   train episode; world and current-frame decoder frozen by hash).
3. AB3: retrieve the verified archive, inspect the gallery images against
   the predicted/persistence/pixel baselines, and record limitations.

A trivial pixel copy scores well on raw error because frames barely change,
so the persistence and wrong-latent controls decide whether the decode shows
forecasting. Teacher-forced one-step only; no open-loop rollout claim.
