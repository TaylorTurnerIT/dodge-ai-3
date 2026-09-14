# MVP execution card

Authority: [SPEC](../../SPEC.md) §M, T34-T39, V28-V30. Scope explicitly authorized by user; use Luna max agents and parent validation. Source references in root references/ plus source crosswalk.

1. Pin upstream + paper; document MIT/CC attribution and defaults.
2. Build isolated modules; parent reviews source parity, frozen probe gradients, data boundaries and dashboard security. Pass tests/lint before data runs.
3. Freeze code identity. Collect <=256 native decisions across separate train/validation seeds.
4. Train reference architecture on Colab T4 <=32 updates; record actual batch, GPU and float32 deviation; no quality claim. Tiny profile only local unit checks.
5. Freeze world model. Fit decoder <=32 updates on train pixels; render held-out attention, features, decoded current/future and actual future.
6. Serve read-only dashboard and verify no-scroll desktop layouts + Tailscale URL. Record evidence and limitations.

If code changes during runs, stop/reopen coding, reverify and issue fresh run ID. Do not start research/controller campaign.
