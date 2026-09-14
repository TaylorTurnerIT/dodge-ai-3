# P10 — Confirm matched controller choices

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T31–T33. Mode: deferred training/evaluation; predecessor: P9. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Run matched frozen controller recipes on at least5 new learner seeds.
2. Freeze checkpoint hashes and evaluate final holdout once.
3. Publish complete paired results and limitations; no automatic extension.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
