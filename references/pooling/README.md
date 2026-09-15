# Pooling source notes

The user's [Beyond CLS article](https://medium.com/@imabhi1216/beyond-cls-advanced-pooling-strategies-for-vision-transformers-8df1785ec81c) motivated the comparison. Primary source cross-check: the pinned DINO `eval_linear.py` computes the mean of patch tokens with CLS excluded, then concatenates it with CLS for classification. `provenance.json` records the commit and SHA-256; the upstream Apache license is retained.

Our diagnostic uses the patch mean alone, channelwise max, one learned attention query, and a 4×4 spatial grid. These are explicit adaptations for reconstruction, not a claim to reproduce DINO's classifier. The attention query uses scaled dot-product softmax weighting as in [Attention Is All You Need](https://arxiv.org/html/1706.03762v7). It starts at zero, giving uniform weights. The decoder's pixel loss trains the query; the encoder remains frozen.

LeWM Appendix D remains the reference for the original CLS decoder (`../paper/lewm-v3.md`). A local patch decoder and pooled patch readouts diagnose retained detail; they do not replace LeWM's prediction objective in this experiment.
