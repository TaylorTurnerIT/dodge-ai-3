# P7 — Scope planning, policy and hybrid comparisons

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T22–T24. Mode: deferred design; predecessor: P6. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Define survival-based planning cost from ordinary outcomes; learned-policy alternatives are DDQN and PPO.
2. Specify separate planning, DDQN, PPO and hybrid experiments with fair data/compute accounting.
3. Amend individual controller coding/training phase contracts before opening P8.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
