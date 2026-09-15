# Spatial readout experiment

SPEC §Z tests whether small-object detail is recoverable from the frozen encoder's local tokens. The source is the completed palette-input world model from §Y; native RGB targets and selected frames stay fixed.

1. Implement and validate extraction, matching, image assembly, and artifact handling. Freeze code.
2. Extract final-layer CLS and 256 patch tokens from the same images. Store tokens in disk-backed arrays; verify the model did not change.
3. Fit three identical local MLP decoders for 512, 2,048, and 8,192 updates: broadcast CLS, local patch tokens, and direct pixel patches. The last is a positive control for the training harness.
4. Evaluate all frames, changed regions, individual colors, wrong-image inputs, and mean-image controls. Retrieve and audit checkpoints before releasing the T4.

Each token supplies 192 values plus fixed image coordinates. A shared 194→256→256→192 MLP produces one 8×8 patch of three palette logits. The 16×16 patch grid reconstructs the native 128×128 image. No semantic labels enter any learner.

The paper's Appendix D decodes one global CLS vector with cross-attention. This experiment deliberately changes the readout to expose local features. Compare its three matched arms internally; do not attribute differences from earlier studies solely to token choice. Recovering current-frame pixels does not validate future prediction or control.
