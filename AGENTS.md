# Dodge Gymnasium — Spotting Failed Runs

Read artifacts first. Runs live under `history/dodge/gymnasium/cnn-image-ddqn/<run_id>/` with `manifest.json`, `config.json`, `metrics.jsonl`, `evaluation.json`, `report.json`. Implementation: `file://src/dodge_native_game/variants/cnn_image_ddqn/run.py`, diagnostics: `file://src/dodge_native_game/variants/cnn_image_ddqn/diagnostics.py`, gate rules: `file://src/dodge_native_game/variants/cnn_image_ddqn/diagnostics.py:decide_gate`, variant contract: `file://variants/cnn-image-ddqn/SPEC.md`.

## Read order

Check `report.json` `quality_gate` and `gate_reasons` before charts. `warn` is normal for runs under 5000 steps. `pass` needs 5000+ steps with no warnings. Then read the last rows of `metrics.jsonl`, then `evaluation.inner` vs `evaluation.holdout` vs `evaluation.counterfactual`.

## Signals and thresholds

`reward_zero_share >= 0.95` means feedback is too thin; Q estimates pile at zero. Confirm with `reward_mean_nonzero` and `reward_std` in the same row.

`action_balance < 0.05` after warmup means lock-in risk; one action owns the run. Confirm with `action_counts` and `least_used_action`. If `evaluation.counterfactual.mean_reward` beats `evaluation.inner.mean_reward`, the avoided move was better and the policy is anchored.

`dead_units` near 1.0 means the trunk emits zeros; training does nothing even when loss looks flat. Confirm with `q_std` near 0 and `td_error_std` near 0.

`train_holdout_gap > 0.5` means the policy memorized train seeds. Inner uses offset 10000, holdout uses offset 20000, counterfactual uses offset 30000; seeds are frozen in `manifest.json` under `eval`.

`warmup-no-updates` means `warmup_steps > steps` so no optimizer step ran. `target-never-synced` means `target_sync_interval > steps`. `epsilon-decay-scaled` means configured decay exceeded run length and the run used `epsilon_decay_steps_effective = min(configured, steps)`; check `epsilon` reached near `epsilon_final` by the last row.

Flat `loss` with flat `reward` plus shrinking `td_error_std` and `q_gap` means the net learned only the visited path. Check `replay_size` against `action_balance`; full replay with narrow coverage is random moves without learning.

## Quick checks

Show gate and eval gap:

```bash
python3 -c "import json,pathlib; r=pathlib.Path('history/dodge/gymnasium/cnn-image-ddqn/<run_id>'); d=json.loads((r/'report.json').read_text()); print(d['quality_gate'], d['gate_reasons'], d['evaluation']['train_holdout_gap'])"
```

Show last diagnostics row:

```bash
python3 -c "import json; rows=[json.loads(l) for l in open('history/dodge/gymnasium/cnn-image-ddqn/<run_id>/metrics.jsonl')]; print({k: rows[-1].get(k) for k in ['step','reward','loss','td_error_mean','td_error_std','reward_zero_share','action_balance','least_used_action','dead_units','q_mean','q_std','epsilon','optimizer_step']})"
```

## Constraints

Rust owns game state, reward, termination, and observations. Python adds no game semantics; reward tuning edits `file://src/dodge_native_game/variants/cnn_image_ddqn/rewards.py` weights only. Keep changes inside the variant namespace and the artifact contract in `file://src/dodge_native_game/variants/cnn_image_ddqn/run_artifacts.py`.

Verify with `scripts/uv-run pytest`, `scripts/uv-run ruff check .`, and one bounded run: `scripts/uv-run dodge-cnn-image-run --history-root history/dodge/gymnasium --run-id smoke-check --steps 32 --stack-size 1 --device cpu`. On this host prefix Python commands with `LD_LIBRARY_PATH=/nix/store/0vqb1mcas5j8dv6bhbrshinlgsg6bvgi-gcc-15.3.0-lib/lib:$LD_LIBRARY_PATH` when NumPy reports missing `libstdc++.so.6`.
