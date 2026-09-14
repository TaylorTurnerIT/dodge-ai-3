# MVP acceptance evidence

M0–M5 engineering gates passed. Scientific representation validation and controller phases remain unopened.

- Full local suite: **245 passed** (55.84s); Ruff clean. Colab worker: **26 variant checks passed** (18.68s). Dashboard-specific checks after final display fixes:5 passed; real-artifact browser assertions passed. Authored files pass whitespace checks; exact upstream HTML retains its original whitespace.
- Source parity: exact SIGReg output against pinned upstream; predictor numerical parity; attached future-target gradients; causal evaluation; final-action rollout regression. Reference files and paper hashes verified.
- Data: 174 train transitions from seeds11/12/13; 64 validation transitions from seed10011. Native128×128 RGB, one executed action per four native frames. 168 train windows and62 validation windows at history3. Episode boundaries, split separation and all file hashes verified.
- Run: `lewm-t4-mvp-20260914-v3`; **Tesla T4**, PyTorch2.11.0+cu128, CUDA12.8; reference-shaped ViT/predictor, float32, actual batch4. 32 world-model updates, then32 frozen-decoder updates. World-model phase11.13s including checkpoint work; observed peak PyTorch allocation1.02GiB. This small-batch measurement does not predict paper-batch memory or long-run throughput.
- Checkpoint SHA256: `3c9420b0f678b087388a23287ccdffde67b667d019ee992dde5c9d6ef51fe9d8`. Data manifest, source archive, checkpoint, decoder linkage and8 held-out snapshots verified. Python model source matches the deployed archive exactly.
- Dashboard: live over Tailscale at http://100.100.169.122:8790/; Overview/Diagnostics checked at1280×720 and1366×768 with real artifacts. Four image panes render; card bounds fit; no document scroll or JavaScript errors. Tiny metric values retain scientific notation; populated plots hide empty-state text; gradient norm shown before clipping. Final HTML display fixes occurred after completed training and changed no Python model code or tensors.
- Legacy32-step CUDA smoke completed; artifacts retained in this job's separate `results/legacy` tree. Its short-run quality warnings do not imply learning success.
- Remote results downloaded and T4 released. Job source archive, logs, corpus, checkpoints and legacy results remain under `history/dodge/gymnasium/pixel-repr-ddqn-jobs/lewm-t4-mvp-20260914-v3/`.

Observed limitation: final feature std0.0002686; decoded current/future images do not yet recover useful objects. Prediction loss reduction is not evidence of meaningful representations. The corpus and32-update decoder are engineering fixtures; batch4, float32 and constant LR differ from upstream training. No gameplay, danger-field, segmentation or survival claim.

Attempts preserved: v1 stopped during setup after parent found missing final rollout prediction. v2 collected data but no optimizer step ran: notebook retained stale NumPy after installation. v3 executes checks/training in a fresh process and requires a completion sentinel plus downloaded artifacts. See SPEC §B.
