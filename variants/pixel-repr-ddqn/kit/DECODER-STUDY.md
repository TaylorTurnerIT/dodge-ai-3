# Native-resolution query decoder screen

O1 implementation → O2 diagnostic fitting → O3 evaluation. User superseded two-decoder2,048 plan: one query decoder256updates; continue same decoder to512 total only if promising.

Frozen LeWM: `lewm-batch32-20260915-v1-calibrated`, SHA `7714c2afcea1f755513a624a7d0f885361271d4c144869f25c8d3a0189432b5d`. Corpus168train/56validation windows, manifestSHA `7769398938b2be30934779fc5851e767653df7b31b2e7fad7a2a0adcd80e6a1f`. No LeWM updates, recalibration, new pixels or game steps.

Cache projected192latents and native128RGB training targets in eval/no_grad. Query decoder:192→128 projection;64 learnable queries,3 cross-attention/residual FeedForward blocks,4 heads,MLP512,no dropout;16×16×3 patch head/sigmoid. Plain native128pixel MSE; AdamW1e-3,decay0.01,batch8windows×4frames,init904,sampler903/discard2once. Save weights, optimizer and sampler state for exact continuation.

[Paper Appendix D](https://arxiv.org/html/2603.19312v3#A4) describes a diagnostic decoder from encoder CLS; pinned repository8edfeb336732b5f3ce7b8b210d0ba370a09e2cac has no decoder implementation. Reuse released FeedForward; decoder structure paper-inspired. Width/depth/heads,128resolution and projected-state input are local choices. Reconstruction gradients never reach LeWM.

Native128 metrics primary: current reconstruction, next prediction, fixed training-mean and current-frame persistence controls, changed regions, per-episode and full-training-set errors.32 metrics historical comparison only. Eight fixed validation views with four128PNG panels.

Before any fitting: full Python/Ruff, frozen source/protocol. One T4 per stage,3000s worker maximum; each stage256 new decoder updates. Retrieve results and release T4. Source/data/world tensors must remain identical.

Continuation gate: changed-region current reconstruction beats training-mean baseline and fixed views show scene-dependent detail beyond fixed HUD. Examine next-prediction versus persistence separately; reconstructing current scenes does not prove useful dynamics. If promising, resume same decoder to512 total; no fresh initialization, optimizer reset or sampler restart. Otherwise stop256. This exploratory validation gate does not establish generalization or open controller training.
