# P3 — Implement LeWM core and diagnostics

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T10–T12. Mode: implementation; predecessor: P2. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Implement encoder, projector, action-conditioned causal predictor and SIGReg.
2. Test attached-target gradients, statistic axes, BatchNorm/evaluation causality and checkpoint equivalence.
3. Calibrate diagnostics and resources; freeze world-model code and P4 protocol.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
