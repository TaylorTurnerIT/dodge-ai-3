# P2 — Collect and freeze dynamics corpus

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T7–T9. Mode: collection; predecessor: P1. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Collect accepted pixel/action corpus with no optimizer.
2. Audit rendered detail/halo/motion coverage and sequence hashes.
3. Freeze dataset/splits and publish acceptance evidence.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
