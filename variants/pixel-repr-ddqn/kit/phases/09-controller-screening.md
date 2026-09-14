# P9 — Screen one accepted controller branch

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T28–T30. Mode: deferred training/evaluation; predecessor: P8. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Run one accepted controller experiment without source mutation.
2. Evaluate inner behavior, survival and latency versus declared controls.
3. Freeze confirmation recipes and unused learner seeds without opening holdout.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
