# P8 — Implement one accepted controller branch

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T25–T27. Mode: deferred implementation; predecessor: P7. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Implement only one controller branch selected by accepted P7 protocol.
2. Verify action legality, frozen core and branch-specific objective/data semantics.
3. Run bounded smoke and hand off fixed controller training/evaluation protocol.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
