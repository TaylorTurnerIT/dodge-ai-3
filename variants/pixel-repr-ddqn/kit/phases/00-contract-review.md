# P0 — Review LeWM adaptation and pin reference

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T1–T3. Mode: design; predecessor: —. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Pin upstream revision, audit objective/config and decide pixel resize/action encoding.
2. Lock corpus splits, resource/accounting rules and diagnostic questions; reserve unknown controller choice.
3. Publish LeWM architecture review; hand off data implementation only.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
