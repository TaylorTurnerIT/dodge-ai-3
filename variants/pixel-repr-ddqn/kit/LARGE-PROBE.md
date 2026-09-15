# Matched reconstruction screen

SPEC §U implements Q4 on the published large corpus. The paper's Appendix D decodes the last encoder CLS token; historical Dodge probes decoded its projected state. Compare these inputs with the same query decoder and examples.

1. Test the disk-backed feature bank, matched fitting loop and launcher. Freeze source.
2. On one Colab T4, extract four deterministically selected frames from every episode using the frozen calibrated LeWM checkpoint. Keep uint8 targets, temporal-change masks and features on disk. Training: 16,384 frames; validation: 2,048 frames. These are a bounded sample across every episode, not every stored frame.
3. Fit two fresh heads with identical initialization (904), minibatches (sampling seed 903), batch size 32, AdamW learning rate 0.001 and weight decay 0.01. Plain native128 current-frame MSE. Retain 512, 2,048 and 8,192 updates for both heads.
4. Evaluate both completed heads at each milestone against a training-only mean-image baseline, including changing pixels. Use incorrect latent pairing to test whether outputs depend on scene input. Export fixed training/validation views, verify frozen weights and hashes, retrieve results and release the T4.

The world model stays frozen throughout this screen. This isolates what its raw and projected representations retain; it does not test whether retraining LeWM on the larger corpus improves dynamics. No future-frame loss or controller fitting belongs to this phase.

The plain-MSE screen completed as `lewm-large-probe-20260915-v2` on source `2870984`. Both heads reached 8,192 updates. Projected features had 6.5% lower overall validation MSE than CLS at the final checkpoint; both scored about 4.3% worse than the mean image on changing pixels. All six checkpoints and their optimizer/sampling states were verified. The T4 was released.

SPEC §W is the authorized follow-up: equal loss mass for bright target pixels and remaining pixels. A pixel is bright when all RGB channels are at least 0.8. Normalize each group separately within each frame, then average frames; keep background errors to penalize false bright regions. Use the same examples, initialization, sampling and budgets. Re-score the retained plain-MSE final decoders on the same bank for direct class-metric comparisons. The mask is derived only from pixels and includes bright HUD pixels as well as objects.

The bright-balanced follow-up completed at 8,192 updates per head. White-pixel error fell, but background error rose sharply and held-out images contained haze and false patches. It was not promoted; ordinary MSE remains the baseline. The final baseline rescoring, six checkpoints and matching frame/sampling provenance passed verification. The T4 was released.

## Palette experiment — SPEC §X

1. Derive a sorted exact RGB palette from selected training frames only, in bounded memory. Reject unseen validation colors rather than adding them. Check loss targets, gradients, palette rendering and legacy behavior before freezing source.
2. On one T4, extract the same frozen features and selected frames as the preceding screens.
3. Fit categorical cross-entropy and one-hot binary cross-entropy arms. Each has fresh CLS and projected heads with matching initialization, minibatches and 512/2,048/8,192 update budgets. BCE averages independent class terms; CE models mutually exclusive colors. Both consume raw logits.
4. Display the highest-scoring palette color at each pixel. Retain RGB errors and report per-color confusion, including changing pixels. Compare exact held-out views and wrong-latent controls. Verify artifacts, retrieve results and release the T4.

Only the diagnostic decoder trains. Sharp palette images can still put objects in the wrong place; image crispness alone does not qualify as improvement.

The palette head is a local diagnostic experiment. It keeps the Appendix D query-decoder structure and vendored upstream feed-forward blocks, but replaces sigmoid RGB regression with color logits. The paper does not establish this palette loss; results must be judged as a Dodge adaptation.
