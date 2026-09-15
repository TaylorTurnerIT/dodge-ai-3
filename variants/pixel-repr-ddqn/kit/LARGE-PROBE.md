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

### Palette loss result and input experiment

The completed `lewm-palette-probe-20260915-v1` comparison did not resolve moving detail. CE reduced global error relative to BCE, but changing cream recall stayed below 0.5%. Snapping the retained MSE outputs to the palette produced essentially the same changed-pixel errors. See the artifact report and both galleries under the run prefix.

§Y tests the input representation with two fresh LeWM models. RGB and one-hot palette inputs each have three channels. Both new arms use nearest-neighbor resize and symmetric normalization, `2*x − 1`. Palette membership comes from training pixels; unknown colors fail. The old RGB preprocessing remains the default for existing checkpoints.

World fitting uses the original prediction plus SIGReg objective, 1,024 updates per arm, batch 32, matched initial weights, windows, and stochastic draws. Then the worlds freeze. Two identical CE diagnostic decoders read raw CLS, each for 512/2,048/8,192 updates. No reconstruction gradients enter LeWM. This follows the paper's Appendix D separation between world training and visualization; palette inputs, categorical outputs, and native 128×128 diagnostics are local adaptations.

Run only after full checks and a source freeze:

```bash
python3 variants/pixel-repr-ddqn/scripts/colab_input_study.py \
  --dataset history/dodge/gymnasium/pixel-repr-ddqn/large-practice-20260914-v2 \
  --run-id lewm-input-study-20260915-v1
```

The launcher retrieves the archive and leaves the session available for parent checkpoint verification. Release it after verifying all retained world and decoder checkpoints. Compare the two new arms directly: comparison with the older world checkpoint also changes training corpus, update budget, and preprocessing.


The input screen completed as `lewm-input-study-20260915-v1`, trained from source `6905cab`. Palette inputs reduced held-out global MSE by 51.7% against the matched RGB arm, but changing cream recall fell from 0.996% to 0.036%. The static layout improved; small moving objects remain unresolved. Four world and six decoder checkpoints passed parent audits, and the T4 was released. A read-only probability inspection found weak changing-cream scores before argmax too.

Open the [overnight report](http://100.100.169.122:8791/lewm-overnight-report-20260915.html) and [input comparison](http://100.100.169.122:8791/lewm-input-study-20260915-v1-comparison.html). Source changes after this run only complete the results documentation and adapt the shared dashboard to the paired artifact schema; they do not alter the retained training results.
