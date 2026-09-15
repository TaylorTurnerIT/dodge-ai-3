# Native-resolution query decoder screen

O1 implementation → O2 diagnostic fitting → O3 evaluation. User superseded two-decoder2,048 plan: one query decoder256updates; continue same decoder to512 total only if promising.

Frozen LeWM: `lewm-batch32-20260915-v1-calibrated`, SHA `7714c2afcea1f755513a624a7d0f885361271d4c144869f25c8d3a0189432b5d`. Corpus168train/56validation windows, manifestSHA `7769398938b2be30934779fc5851e767653df7b31b2e7fad7a2a0adcd80e6a1f`. No LeWM updates, recalibration, new pixels or game steps.

Cache projected192latents and native128RGB training targets in eval/no_grad. Query decoder:192→128 projection;64 learnable queries,3 cross-attention/residual FeedForward blocks,4 heads,MLP512,no dropout;16×16×3 patch head/sigmoid. Plain native128pixel MSE; AdamW1e-3,decay0.01,batch8windows×4frames,init904,sampler903/discard2once. Save weights, optimizer and sampler state for exact continuation.

[Paper Appendix D](https://arxiv.org/html/2603.19312v3#A4) describes a diagnostic decoder from encoder CLS; pinned repository8edfeb336732b5f3ce7b8b210d0ba370a09e2cac has no decoder implementation. Reuse released FeedForward; decoder structure paper-inspired. Width/depth/heads,128resolution and projected-state input are local choices. Reconstruction gradients never reach LeWM.

Native128 metrics primary: current reconstruction, next prediction, fixed training-mean and current-frame persistence controls, changed regions, per-episode and full-training-set errors.32 metrics historical comparison only. Eight fixed validation views with four128PNG panels.

Before any fitting: full Python/Ruff, frozen source/protocol. One T4 per stage,3000s worker maximum; each stage256 new decoder updates. Retrieve results and release T4. Source/data/world tensors must remain identical.

Continuation gate: changed-region current reconstruction beats training-mean baseline and fixed views show scene-dependent detail beyond fixed HUD. Examine next-prediction versus persistence separately; reconstructing current scenes does not prove useful dynamics. If promising, resume same decoder to512 total; no fresh initialization, optimizer reset or sampler restart. Otherwise stop256. This exploratory validation gate does not establish generalization or open controller training.

## Result: stopped at 256

Run `lewm-decoder128-20260915-v1-query256`; source `81251c5`. One Tesla T4,256 decoder updates in3.252s;192,290,816bytes peak allocated. Frozen world model unchanged; no new game steps. Artifacts retrieved and T4 released. All337 local and118 remote variant tests pass; Ruff clean; bounded legacy smoke expected warnings. Resume equivalence verified with a small local test; no512 fit executed.

| Native128 held-out metric | Query decoder | Control |
|---|---:|---:|
| Current reconstruction MSE |0.00291185|0.00268604 fixed train mean|
| Current reconstruction on changing regions |0.13859245|0.13795829 fixed train mean|
| Next prediction MSE |0.00291038|0.00233309 current-frame persistence|

Current reconstruction is8.41% worse overall and0.46% worse on changing regions than the fixed training mean. Training-set errors also lose to mean:10.34% overall and2.28% on changing regions. Eight fixed views show mostly blue background and HUD; player/enemy details remain indistinct. The continuation gate failed, so the run stops at256.

Native changing-region next-prediction error does beat persistence, but the fixed mean does too. Removing moving objects can reduce this error; that result alone does not show useful motion prediction. The decoder and representation remain possible bottlenecks.

All32 panel PNGs verified128×128; optimizer step256, sampler progression and checkpoint/data/source hashes verified. Dashboard passes1280×720 and1366×768 with square images and no scrolling. Evidence: run `parent-verification.json` and `review-contact-sheet.png`; job `lewm-decoder128-20260915-v1-256/decision.json`, archived results and browser screenshots. Source archive SHA `a14e6d9a54a8890a1317472e0899fc90f9ca4c89bd1d4eb7ea3ad04a2bbd696a`.

## User-directed continuation to 512

User reports emerging detail and requests another iteration. Resume the saved query256 decoder for256 additional updates, reaching512 total. This supersedes the earlier stop decision for this extension; the original negative measurements remain unchanged. Reuse model, data, native128 target, optimizer state, sampler state and architecture. No code changes or new collection. Compare the same eight validation views and all56 validation windows before and after, with fixed mean/persistence controls. Retrieve artifacts, verify source/resume/world identity and release the T4 at512.

## Continuation result

Completed `lewm-decoder128-20260915-v1-query512` from the saved256 checkpoint, preserving AdamW and sampler state. Exactly256 additional updates;3.557s fitting on Tesla T4. All118 remote tests passed; implementation unchanged from the337-test local validation. Artifacts retrieved and T4 released.

| Native128 validation metric |256 updates|512 updates|Change|
|---|---:|---:|---:|
| Current reconstruction MSE|0.00291185|0.00276618|−5.00%|
| Current reconstruction on changing regions|0.13859245|0.13800573|−0.42%|
| Next prediction MSE|0.00291038|0.00277741|−4.57%|
| Next prediction on changing regions|0.14212736|0.14137371|−0.53%|

Current reconstruction improves, but still loses to the fixed training mean by2.98% overall and0.034% on changing regions. Changed-current error improves in3/4 validation episodes; training changed-current error increases0.14%. Score reconstruction is visibly clearer; player/enemy shapes remain indistinct. Object tracking is unverified.

Eight observed-frame/256/512 comparisons are saved in job `lewm-decoder128-20260915-v1-512/decoder-256-vs-512.png`; metrics and resume checks in `comparison.json`. Source archive SHA `c9ea5037366c559e37c76d74fcf228e0bf6fbc9ff20fcebdc3e66a4d7be98618`. World checkpoint, input frames, latent values and fixed baselines match; optimizer step512 and exact sampler progression verified. Both dashboard viewports pass with native128 square images. This bounded iteration ends at512.

## Current-frame extension to 2,048

User requests2,048 total decoder updates, prioritizing current-frame reconstruction. Resume query512 for1,536 additional updates on one T4. Keep architecture, native128 plain pixel-MSE, AdamW state, sampler, LeWM weights and practice corpus fixed. Implementation phase only expands the accepted resume envelope; full tests/Ruff precede source freeze and fitting. Invalid resume steps must fail.

Review the same eight current-frame views at512/2048 and train/validation reconstruction errors against the fixed training-mean baseline. Changed-region measurements remain pixel-only evaluation diagnostics. Predicted frames remain available through the existing exporter, with no prediction-specific tuning. Stop at2048, retrieve evidence and release the T4.

## 2,048 result and requested extension to 8,192

Completed1,536 additional updates from512 in18.347s on T4. Current-frame reconstruction MSE fell from0.00264568 to0.00126317 on training windows (52.26% improvement), while held-out MSE rose from0.00276618 to0.00340290 (23.02% worse). Held-out changing-region error rose0.42%. Images contain more visible shapes, often at incorrect positions. The decoder fits training images more closely; held-out generalization remains weak. Source/resume/optimizer/sampler checks pass; T4 released.

User requests8,192 total updates. Add6,144 updates from saved2048 with the same optimizer, sampler, native128 MSE, LeWM and data. Validate the expanded resume envelope before source freeze. Compare current-frame train/validation errors and identical2048/8192 views; retain512 as reference. Stop8192 and release T4. More distinct shapes alone do not establish accurate reconstruction.

## 8,192 result

Completed6,144 additional updates from the saved2048 decoder in70.982s on Tesla T4. Optimizer step8192 and exact sampler continuation verified. LeWM weights and corpus remain unchanged. All338 local and119 remote tests passed; source `4524c5f`, archive SHA `3c7dd7ba33505d6a5668f79b9d6cdda255c21fccce7dd69a6d9125a9e64b1115`. Results retrieved; T4 released.

| Current-frame reconstruction |2,048|8,192|Change|
|---|---:|---:|---:|
| Training MSE|0.00126317|0.000763095|−39.59%|
| Training changing-region MSE|0.0796794|0.0442144|−44.51%|
| Held-out MSE|0.00340290|0.00366236|+7.62%|
| Held-out changing-region MSE|0.138582|0.139783|+0.87%|

Some player-like tails and squares are sharper, but often appear at incorrect positions. Training reconstruction continues improving while held-out error worsens. This supports an overfitting concern for the current-frame probe; it does not establish whether the decoder or the learned representation limits generalization. No prediction-specific tuning was performed.

Comparison image: job `lewm-decoder128-20260915-v1-8192/decoder-2048-vs-8192.png`. Paired metrics, fixed-view checks and provenance: `comparison.json`. Dashboard retains all checkpoints;32 exported panel images verified native128×128, with no scrolling at1280×720 or1366×768. This iteration ends at8192.
