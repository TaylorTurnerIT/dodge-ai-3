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
