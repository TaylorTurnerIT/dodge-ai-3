# P1 — Implement pixel/action sequence boundary

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T4–T6. Mode: implementation; predecessor: P0. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Implement raw-pixel/action whitelist and episode-contained sequence loader.
2. Test exact cadence, truncation/reset, split isolation and artifact namespaces.
3. Freeze collector revision/protocol after bounded verification.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
