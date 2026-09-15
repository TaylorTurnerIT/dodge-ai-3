# Pooling diagnostic

Follow SPEC §AA. Finish §Z artifact audit, then freeze implementation before fitting.

1. AA1: implement four readouts with the existing local decoder; review source, parameter accounting, shared initialization and sampler. Validate the real artifact producer and launcher together.
2. AA2: reuse the original T4 banks read-only. Match checkpoint, corpus, frame indexes, palette and bank hashes. Fresh source and output directories; no extraction.
3. AA3: fit all four heads for 512 and 2048 updates. Keep the encoder frozen. Record the attention query separately from shared decoder weights.
4. AA4: evaluate the same frames and wrong-frame controls, audit checkpoints, publish square comparisons, then release the T4.

A compact global vector may lose recoverable local detail. This experiment tests that possibility under a fixed readout and budget; it does not establish future prediction or control.
