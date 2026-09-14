# Scenario delivery

User requested native configuration files and an aspect-ratio fix. Follow SPEC §S; implement and verify configuration support before collection. No controller or world-model training in this change.

Review findings:
- `game.rs:update_fyou/add_corner_enemies` can spawn independently of the normal timer: disable both sources for empty worlds.
- `game.rs:restart_gameplay` recreates patterns: reapply permanent geometry after reset and restart.
- `snapshot.rs:canonical_bytes/from_canonical_bytes` plus `NativeGame::restore` must preserve scenario rules. Keep default wire8 byte layout; nondefault snapshots use explicit wire9 extension.
- Player collision code emits death events separately from `die`: invulnerability must suppress the collision path, not only the death flag.
- Scenario metadata belongs in corpus/run provenance, never model inputs.

Gate: GO with V34–V36. Oracle: native scenario tests, existing native parity suite, Python schema/artifact tests, bounded configured collection. Dashboard first: V33 browser image-content bounds, no-scroll layouts and screenshot.

Engineering evidence (2026-09-14):
- S0 complete: Chromium checked four real square artifacts at 1280×720 and 1366×768; contained image bounds inside each panel; no scroll or JS errors. Screenshot inspected. Panel positions unchanged.
- Native workspace: 91 tests pass, including five new scenario tests. Follow-up collision test also verifies permanent obstacles remain lethal when invulnerability is disabled.
- Default snapshots retain wire8; custom snapshots round-trip wire9 and reproduce the same next frames. Existing native regression tests pass.
- S1 complete: parent reviewed Luna implementation; full Python suite 255 passed; Ruff and authored-file whitespace checks pass. Strict schema and tampered-provenance tests pass.
- S2 complete: four native collections, each seeds 11/10011 and 32 decisions per episode (256 decisions total), 30 training windows per preset. Safe presets reached their caps without termination; resolved config/hash verified and scenario fields absent from learner samples. Temporary check corpora at `/tmp/lewm-scenarios-n_cmc9m0`; no model or controller trained.
- Colab launcher accepts `--scenario`; copies exact TOML into source archive and supplies it to the frozen worker. This configuration path was checked locally; no new T4 training run launched for this change.
