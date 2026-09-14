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

Scripted practice adds player action sequences and coordinate waypoints, plus static or moving enemy squares. Its implementation and capture checks follow [a separate phase contract](kit/SCRIPTED-PRACTICE.md). Practice coordinates configure the native driver; they are not model observations. Practice artifacts remain separate from training corpora until an ingestion/training phase is specified.

Generate the supplied practice example (native collection only):

```bash
scripts/uv-run --extra native --extra training --extra lewm python \
  -m dodge_native_game.variants.pixel_repr_ddqn.practice \
  --config variants/pixel-repr-ddqn/practice/moving-enemy.toml \
  --output history/dodge/gymnasium/pixel-repr-ddqn-practice/moving-enemy-demo \
  --seed 42
```

The [example script](practice/moving-enemy.toml) moves the player to a waypoint, holds neutral, and drives one square along a closed path while another stays still. Player commands accept `move_to = [x, y]` with a decision budget, or a named action such as `right` with a duration in decisions. Each decision lasts four native frames; enemy path durations use native frames (the cartridge updates at 60 Hz). Positions are centers in the 128×128 playfield. The waypoint helper uses a 2-pixel arrival tolerance and a low-speed check; it may time out on a route it cannot settle at and does not plan around obstacles.

Inspect `initial.png`, `final.png`, and `replay.gif`. Successful scripts also produce `goal.gif`; the NPZ files retain exact pixels and executed actions. End with movement for a moving goal, or add neutral holds for a settled player. Scripted enemies currently use normal square geometry and authored paths in place of pursuit AI. A failed or lethal episode retains its replay but has no valid goal. PNG/GIF files are visual previews; exact timing and transitions belong to the recorded trajectory and manifest.
