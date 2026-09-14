Practice corpus bridge: [contract](PRACTICE-CORPUS.md), SPEC §D; D1 implementation → D2 bounded import verification → separately scoped D3 training. [Reading the diagnostic panels](DIAGNOSTICS.md).

Scripted practice: [scripted practice and goal generation](SCRIPTED-PRACTICE.md). Both player and enemies selected; G1/G2 complete; goal-seeking controller design remains deferred.

Current change: [scenario delivery](SCENARIOS.md), SPEC §S. Square-image dashboard fix first, then native configuration support, then bounded collection verification. Model training remains separate.

# Current execution: LeWM MVP

User authorized MVP implementation with Luna max agents and parent validation. Follow canonical SPEC §M sequence M0 references → M1 code → M2 native collection → M3 bounded world-model training → M4 frozen diagnostic fitting → M5 dashboard review. Scientific P0-P10 remains separate; MVP completion does not certify model quality or open controllers. [MVP card](phases/mvp.md).

# LeWM phase kit

[SPEC](../SPEC.md) owns architecture, phases, tasks, gates and budgets. P0-P6 validate LeWM independently of controller selection. Existing filesystem slug retained for continuity; DDQN is optional.

| Phase | Mode | Card | Depends on |
|---|---|---|---|
| P0 | design | [Review LeWM adaptation and pin reference](phases/00-contract-review.md) | — |
| P1 | implementation | [Implement pixel/action sequence boundary](phases/01-data-implementation.md) | P0 |
| P2 | collection | [Collect and freeze dynamics corpus](phases/02-data-collection.md) | P1 |
| P3 | implementation | [Implement LeWM core and diagnostics](phases/03-lewm-implementation.md) | P2 |
| P4 | training | [Train world model independently of controllers](phases/04-lewm-training.md) | P3 |
| P5 | implementation | [Implement frozen-model probes and rollout evaluation](phases/05-probe-implementation.md) | P4 |
| P6 | diagnostic fitting/evaluation | [Validate retained detail and open-loop dynamics](phases/06-world-model-validation.md) | P5 |
| P7 | deferred design | [Scope planning, policy and hybrid comparisons](phases/07-controller-design.md) | P6 |
| P8 | deferred implementation | [Implement one accepted controller branch](phases/08-controller-implementation.md) | P7 |
| P9 | deferred training/evaluation | [Screen one accepted controller branch](phases/09-controller-screening.md) | P8 |
| P10 | deferred training/evaluation | [Confirm matched controller choices](phases/10-controller-confirmation.md) | P9 |

P7-P10 deferred placeholders: each controller gets separately specified coding and training subphases before execution. Controller options: planning; DDQN; PPO; planning with learned terminal value. No need to choose before LeWM validation.

Before execution: verify requested scope, cwd/branch, SPEC revision and predecessor acceptance. Coding phases follow build→check; defects consider backprop→spec. Architecture changes use review/research→spec. Collection/training phases consume frozen code and [protocol](templates/PROTOCOL.md). Complete [acceptance](templates/ACCEPTANCE.md) only with real evidence and update canonical SPEC. Automated pass, scientific positive and owner acceptance remain distinct. MVP M0-M5 may advance through their accepted engineering gates; research phases require their declared acceptance.

Source defects stop affected runs and reopen coding phase; never silently patch ongoing experiments. MVP execution is authorized by the user. Run `python3 variants/pixel-repr-ddqn/scripts/colab_mvp.py` from the worktree to create a fresh T4 session, execute the bounded phases and retrieve artifacts. The dashboard entrypoint is `python -m dodge_native_game.variants.pixel_repr_ddqn.dashboard`. See [source crosswalk](SOURCE-CROSSWALK.md) and [validation evidence](MVP-EVIDENCE.md).
