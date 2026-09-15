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

## Result: K3/K4 complete

Both runs are available on the [dashboard](http://100.100.169.122:8790/). Select the run ending in `-calibrated` for the inference copy. Its training metrics are blank because it received no optimizer updates; its 256-step status belongs to decoder fitting. The bottom panels remain diagnostic reconstructions, not segmentation masks.

| Checkpoint | Split | Prediction MSE | Persistence MSE | Prediction / persistence | Wrong-action MSE |
|---|---|---:|---:|---:|---:|
| Original | Train | 5.194165 | 0.027968 | 185.7208 | 5.193057 |
| Original | Validation | 4.902855 | 0.174228 | 28.1404 | 4.901618 |
| Encoder-calibrated | Train | 0.013526 | 0.010784 | 1.25434 | 0.015480 |
| Encoder-calibrated | Validation | 0.126048 | 0.126530 | 0.996191 | 0.132013 |

Calibration removes much of the discrepancy seen in frozen inference. It does not establish useful predictive dynamics: validation barely beats persistence by 0.38%, training still loses to persistence, and only four validation episodes were tested. Wrong actions increase calibrated error by 14.45% on train and 4.73% on validation, a modest action-conditioning signal under this control. It is not a causal counterfactual game rollout or proof of controllable object identity. The changed target space prevents treating the raw MSE reduction across checkpoints as a common-space accuracy gain.

Visual review: all four square panels are fully visible at1280×720 and1366×768, both tabs, no page errors or scrolling. Original predicted decoding is noisy; calibration makes it resemble the decoded current image, but both calibrated panels lack clear player/enemy shapes. This can reflect the representation, the small decoder, its fitting budget, or their combination; a low background-dominated pixel loss cannot distinguish those causes. Each checkpoint has eight matched validation snapshots and its own256-update decoder. Last decoder training MSE: original0.00130685, calibrated0.00312057.

Provenance and checks:

- Frozen source commit077f623; archive SHA256 `2c1f61223610f489598cb2cbdecdb0a6f30d0f2908689cd7a8fe8aaab7208475`.
- Original checkpoint SHA256 `f47a9623ac317622d02948228b645127ad3e9f470580d43e737c0f4685acb0c6`; calibrated `875050d70f298a33d17a7a8d56ed6c600d1457fc685b7d7151eda9968d796b5f`.
- Parent validation confirms only `projector.net.1.running_mean` and `projector.net.1.running_var` differ. Calibration uses168training windows/672frame occurrences. Derived checkpoint contains no optimizer/sampling state; both decoders and all snapshots reference the correct checkpoint hashes.
- Tesla T4, Torch2.11.0+cu128, CUDA12.8; peak allocated1,877,840,384bytes. Model512updates in194.787s. Final train-mode loss0.124761,prediction loss0.018388,preclip grad norm0.701207. Training loss and fixed-stat evaluation are different measurements.
-310 local tests,91 remote variant tests, Ruff/diff checks; legacy32-step CUDA run completed with expected bounded-smoke/target-never-synced warnings. T4 session terminated after retrieval.
- Artifacts under `history/dodge/gymnasium/pixel-repr-ddqn/lewm-diverse-calibrated-20260914-v1` and its `-calibrated` sibling. Job archive, protocol and review screenshots under `pixel-repr-ddqn-jobs/lewm-diverse-calibrated-20260914-v1`.

Engineering K1-K4 accepted. No long run, controller promotion or scientific P4/P6 acceptance. Next design question: a larger, more independent training batch closer to the reference, tested separately from data-volume and architecture changes. This screen does not answer that question.
