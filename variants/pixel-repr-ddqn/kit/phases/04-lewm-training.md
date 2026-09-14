# P4 — Train world model independently of controllers

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T13–T15. Mode: training; predecessor: P3. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Run only frozen LeWM screen and declared controls.
2. Measure noncollapse, prediction and sample diversity on representation-validation.
3. Freeze accepted world model; publish limitations without survival claims.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
