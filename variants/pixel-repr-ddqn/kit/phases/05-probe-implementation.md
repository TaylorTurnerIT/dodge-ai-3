# P5 — Implement frozen-model probes and rollout evaluation

Authority: [SPEC](../../SPEC.md) §P/§E/§V; tasks T16–T18. Mode: implementation; predecessor: P4. Canonical phase state in SPEC.

Read [variant instructions](../../AGENTS.md) and [kit workflow](../README.md).

1. Implement detached decoder/readouts and open-loop rollout evaluator.
2. Test probe gradient isolation and persistence/action/temporal controls.
3. Freeze probe fitting budget and visual review protocol for P6.

Exit criteria and caps: canonical SPEC row and §E. Publish actual commands, return codes, hashes, raw artifact paths, limitations and visual review in [acceptance template](../templates/ACCEPTANCE.md). Freeze [protocol](../templates/PROTOCOL.md) before any collection/learning. No fake acceptance.

Legacy boundaries unchanged. Coding phases run narrow tests plus required regression checks; learning phases verify frozen code identity. Defect stops affected execution and reopens owning implementation phase. Successor work never repairs failed predecessor evidence.
