# P6 — Validate retained detail and open-loop dynamics

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T19–T21. Mode: diagnostic fitting/evaluation; predecessor: P5. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Fit diagnostic probes on TRAIN only with world model frozen.
2. Evaluate unseen detail, tails, hollow enemies, boundaries, halos and temporal futures.
3. Publish core acceptance or failure; stop before controller work.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
