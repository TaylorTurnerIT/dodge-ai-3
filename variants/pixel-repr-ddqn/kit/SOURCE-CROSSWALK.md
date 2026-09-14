# LeWM implementation crosswalk

Reference: [paper v3](../../../references/paper/lewm-v3.md), [source manifest](../../../references/manifest.json). Upstream commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.

| Component | Official source | Dodge implementation / adaptation |
| --- | --- | --- |
| SIGReg characteristic-function loss | `module.py:12–37` | `sigreg.py`; keep T,B,D axes, random projections and batch estimator |
| Action-conditioned causal predictor | `module.py:40–281` | `upstream.py`; retain AdaLN-zero, causal attention and predictor projection |
| CLS encoder and projections | `jepa.py:29–59`; `config/train/model/lewm.yaml` | `model.py`; randomly initialized Hugging Face ViT, no pretrained weights |
| Attached-target prediction loss + SIGReg | `train.py:17–41` | `model.py`; target retains gradient; no EMA |
| Architecture | Paper architecture appendix; model YAML | Reference ViT-Tiny: 224 pixels, patch14, width192, 12 layers; predictor6 layers,16 heads, head width64 |
| Input preprocessing | `utils.py`; dataset configs | Native 128×128 RGB resized to224 and normalized; nine discrete actions encoded one-hot; action bottleneck9 instead of upstream Embedder default10 |
| Optimizer | `config/train/lewm.yaml` | AdamW lr5e-5, decay1e-3, clip1; bounded smoke uses constant LR |
| SIGReg coefficient | Paper method and training YAML | YAML default0.09; paper describes0.1. Record actual configuration |
| Pixel visualization | Paper diagnostic decoder discussion | Separate lightweight32×32 decoder fitted on training pixels with frozen LeWM; not a reconstruction loss for LeWM |
| Control | Paper goal-image planning | Deferred: Dodge survival needs its own objective and evaluation; no controller in MVP |

The first Colab T4 run checks implementation and observability. It uses the reference architecture, a small explicitly recorded batch and float32. Upstream uses batch128 and bfloat16; the smoke is not a reproduction of its training protocol. SIGReg depends on the simultaneous batch: gradient accumulation does not make a small batch equivalent to128.

Attention shows where a CLS token attends. Latent coordinate plots show embedding values. Neither supplies player/enemy labels or proves learned object segmentation. Decoder output also reflects decoder fitting quality. All views identify their checkpoint and held-out window.

Validation must cover source numerical parity, attached target gradients, causal prediction with evaluation statistics, episode-contained windows, disjoint game seeds, frozen model parameters/buffers during decoder fitting, and dashboard path confinement. Record actual results in the phase evidence after checks run.
