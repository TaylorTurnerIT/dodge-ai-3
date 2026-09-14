# Calibrated diverse-practice screen

User authorized applying the normalization finding and further ideas. K1 implementation → K2 collection → K3 training → K4 calibration/diagnostics. This screen preserves the reference model and objective; it does not open controllers or establish scientific P4/P6 acceptance.

Calibration: encoder BatchNorm only, exact training-window-weighted population moments, float64 accumulation, batched reference encoding on T4. Original optimizer checkpoint preserved; export a new inference-only checkpoint with unchanged learned tensors and predictor statistics. Record source/data/derived hashes. Resume from inference-only copies rejected. New latent distribution gets its own train-only decoder fit; never reinterpret the original decoder as calibrated output.

Collection:12train seeds500-511 and4validation seeds1500-1503;16 decisions each,256total. Eight two-decision action blocks/capture; all nine actions across training; varied player starts and independent static/moving enemies. Safe invulnerable difficulty1, no patterns; no privileged learner inputs. Fixed suite seed20260914; config/source/artifact hashes retained. Freeze all captures and rendered review before training.

Training: one fresh `practice-diverse-v1` run,512 updates,batch8,seed42, reference architecture,float32,AdamW5e-5/decay1e-3/clip1,SIGReg0.09. Colab T4 required. No optimizer/architecture changes or automatic extension. Native/model code unchanged.

Evaluation: original and encoder-calibrated copies of the same trained weights. Evaluate all train/validation windows against same-representation persistence and wrong-action controls. Decoder256 updates,batch8 per copy, training pixels only; fixed validation views. Target spaces differ between copies, so raw cross-copy MSE is not a common-space quality measure. Both results reported; no validation-based selection. New corpus differs from earlier overfit corpus, so comparisons with that run remain descriptive.

Resource bound: one T4,3000s remote worker timeout,2GiB source cap;512 model updates and512 decoder updates total across the two diagnostic fits. Stop on nonfinite/OOM/source defect; preserve artifacts and reopen code under new identity. Full local Python/Ruff regression; T4 variant regression and32-step legacy smoke. Dashboard keeps original and calibrated runs separate.

K1 accepted:310 local tests passed in68.46s; Ruff and diff whitespace checks passed. Calibration tests verify original hash and non-stat tensor preservation, train-only use, inference-only resume rejection and fixed-stat future independence. Suite tests cover deterministic bounds, actions, TOML roundtrip and explicit split import.

K2 accepted: collected from frozen implementation703e84d. Corpus `history/dodge/gymnasium/pixel-repr-ddqn-practice/diverse-20260914-v1/corpus`; manifest SHA256 `7769398938b2be30934779fc5851e767653df7b31b2e7fad7a2a0adcd80e6a1f`.192train/64validation decisions,168/56windows. Training action counts0..8:22,22,22,18,18,24,24,22,20; validation:8,6,8,8,8,8,6,6,6. All16 emitted configs byte-match the frozen serializer. Inspected initial train500 and final validation1502 images: native player/enemy shapes and player trail present. Late agent serializer refactor discarded; frozen capture source retained.

K3 frozen run:`lewm-diverse-calibrated-20260914-v1`; original and `-calibrated` inference copy. Use the corpus hash above and exact512/8/42 protocol.
