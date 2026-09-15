# Matched reconstruction screen

SPEC §U implements Q4 on the published large corpus. The paper's Appendix D decodes the last encoder CLS token; historical Dodge probes decoded its projected state. Compare these inputs with the same query decoder and examples.

1. Test the disk-backed feature bank, matched fitting loop and launcher. Freeze source.
2. On one Colab T4, extract four deterministically selected frames from every episode using the frozen calibrated LeWM checkpoint. Keep uint8 targets, temporal-change masks and features on disk. Training: 16,384 frames; validation: 2,048 frames. These are a bounded sample across every episode, not every stored frame.
3. Fit two fresh heads with identical initialization (904), minibatches (sampling seed 903), batch size 32, AdamW learning rate 0.001 and weight decay 0.01. Plain native128 current-frame MSE. Retain 512, 2,048 and 8,192 updates for both heads.
4. Evaluate both completed heads at each milestone against a training-only mean-image baseline, including changing pixels. Use incorrect latent pairing to test whether outputs depend on scene input. Export fixed training/validation views, verify frozen weights and hashes, retrieve results and release the T4.

The world model stays frozen throughout this screen. This isolates what its raw and projected representations retain; it does not test whether retraining LeWM on the larger corpus improves dynamics. No future-frame loss or controller fitting belongs to this phase.
