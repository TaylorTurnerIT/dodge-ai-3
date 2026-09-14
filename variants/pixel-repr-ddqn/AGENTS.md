# Pixel representation DDQN instructions

Read [SPEC](SPEC.md), [kit](kit/README.md), and selected phase card before work. SPEC owns scope, gates, tasks and state; cards own sequence. Current phase status is in SPEC §P.

- Preserve pixel/action-only representation boundary; ordinary native rewards only for RL.
- New implementation belongs in `src/dodge_native_game/variants/pixel_repr_ddqn/`; tests in `tests/variant_pixel_repr_ddqn/`; artifacts in `history/dodge/gymnasium/pixel-repr-ddqn/`.
- Legacy/native changes require separately scoped review; explicit audited imports allowed.
- Implementation, collection and training are distinct phase types. Follow SPEC §E caps; training uses frozen code/data/protocol. A defect reopens the owning implementation phase.
- No code, training, data collection or servers are authorized merely because this scaffold exists.
- Report evidence-ready separately from accepted and scientific positive. Do not auto-open successors or fill acceptance templates with assumed success.
- Run spec/build/check/backprop flows according to phase kit; never use root board/PPO task statuses for this variant.
