# Dodge LeWM MVP

Pixels and executed actions train a LeWM encoder/predictor. A separate frozen-model decoder makes its latents visible. There is no controller yet.

Open the running dashboard at [t3-dev:8790](http://100.100.169.122:8790/). Overview compares observed current/next frames with decoded current/predicted frames. Diagnostics shows loss components, gradient norm, feature spread/rank, attention and latent predictions. Snapshot selection uses held-out windows.

From this worktree, launch a new bounded Colab T4 run:

```bash
python3 variants/pixel-repr-ddqn/scripts/colab_mvp.py
```

The launcher uses the existing `colab --auth adc` connection, creates its own T4 session, builds the native extension remotely, runs checks, collects a small corpus, trains32 world-model updates, then fits a frozen-model decoder for32 updates. It mirrors dashboard artifacts during execution and downloads the full results before releasing the successful session. Failed sessions remain available for diagnosis. Job records and logs live under `history/dodge/gymnasium/pixel-repr-ddqn-jobs/`; dashboard runs live under `history/dodge/gymnasium/pixel-repr-ddqn/`.

Start the dashboard separately when needed:

```bash
scripts/uv-run --extra lewm python -m dodge_native_game.variants.pixel_repr_ddqn.dashboard --host 0.0.0.0 --port 8790
```

Reference-shaped model: random ViT-Tiny encoder, causal action-conditioned predictor, attached-target prediction loss plus SIGReg. The smoke uses float32 and batch4 on a T4; these differ from the upstream training recipe. It checks operation and visibility. It does not establish useful representations, object separation, danger prediction or survival skill. Attention is not a segmentation mask, and the short decoder fit limits reconstruction quality.

- [SPEC and phase gates](SPEC.md)
- [Phase kit](kit/README.md)
- [Source/paper crosswalk](kit/SOURCE-CROSSWALK.md)
- [Validation evidence](kit/MVP-EVIDENCE.md)
- [Local source and paper](../../references/README.md)

Testing worlds are ordinary TOML files in [scenarios](scenarios/): `empty.toml`, `permanent-patterns.toml`, `normal-easy.toml`, and `standard.toml`. The first three enable safe exploration; `standard.toml` retains ordinary game behavior. Change `invulnerable` to `false` when you want collisions to end the episode. Invulnerability disables player collisions, including powerup pickups; safe presets disable powerups too. These worlds are diagnostic environments, not evidence of survival skill in the standard game.

A permanent pattern freezes the selected catalog pattern's initial geometry, fully visible, without its preview or movement schedule. The supplied preset uses pattern 1. The `difficulty` field accepts `easy`, `medium`, `hard`, or the corresponding integers 1, 2, 3. Difficulty still follows the game's normal progression from the selected initial setting.

Choose a scenario for the next bounded T4 run:

```bash
python3 variants/pixel-repr-ddqn/scripts/colab_mvp.py \
  --scenario variants/pixel-repr-ddqn/scenarios/empty.toml
```

The launcher freezes the configuration in its source archive. Collection records the resolved configuration and its hash; the model still receives only pixels and actions. Each scenario needs a separate corpus/run so results stay attributable. See [scenario gates and checks](kit/SCENARIOS.md).
