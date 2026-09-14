# Scripted practice and goal generation

Requested extension, 2026-09-14. Design recorded; implementation not started. Actor scope is awaiting clarification: player, enemies/targets, or both. Recommended first delivery: player trajectories in the existing safe scenarios.

A practice configuration should select a scenario, seed, and bounded script. Scripts can specify named actions with decision counts, moves between authored coordinates, and a stationary hold. Coordinates belong to the native authoring/driver layer. The learner receives rendered pixels and the actions actually executed, never coordinates, entity identities, waypoint indices, or script progress.

Moves must use the game's ordinary action and physics path. Initial-state placement, if supported, happens before the episode's first observation. Teleports, checkpoint restoration, and resets cannot become apparent action-driven transitions. A waypoint is not automatically reachable around obstacles: report arrival or timeout, and never label a timeout as a successful goal demonstration.

Capture start observations, the executed trajectory, and selected goal frames or clips. Save hashes and exact scenario/script provenance. A stationary capture should include enough settling time to distinguish stopping from passing through. A motion capture retains an ordered clip and action history. Goal images should come from actual native renderings of reached states.

LeWM learns dynamics from these trajectories. It does not learn a reaching policy merely because demonstrations have goals. The paper's planner holds the world model fixed and minimizes final predicted latent distance to an encoded goal image. Matching a moving goal clip or handling changing distractors needs a separately specified Dodge planning objective; it is not already supported by the MVP. Source: [paper §3.2](https://arxiv.org/html/2603.19312v3#S3.SS2), local `references/paper/lewm-v3.md`.

Death remains an episode boundary:
- Keep the action and rendered outcome of the transition into death.
- Record native termination separately from a collection time cap.
- End the episode at termination; reset starts a new episode and history.
- LeWM keeps its pixel/action inputs and predictive objective. A death-to-reset prediction pair is invalid.
- Later DDQN/PPO uses termination to stop return bootstrapping, with ordinary native rewards. A pure planner needs a separately designed terminal/risk mechanism; goal-image matching alone does not guarantee death avoidance.
- A possible learned terminal head can consume frozen latents and predict termination. Its labels and gradients stay outside representation training. Adequate terminal information in the frozen representation must be tested, not assumed.

Proposed delivery gates:

| Phase | Work | Acceptance |
|---|---|---|
| G0: contract | Resolve actors, action timing, arrival/hold criteria and goal artifact format | Concrete schema and native scope reviewed |
| G1: implementation | Native script execution and capture/provenance plumbing | Deterministic actions, bounded timeouts, valid reached goals, no state features in learner samples |
| G2: collection verification | Small safe practice corpus from frozen G1 code | Replay and captured goals agree; no reset crossings; no model training |
| G3: later design | Stationary/moving goal planning and terminal handling | Separate controller contract after representation validation |

Existing evidence: scenario implementation commit `1075e89`; 255 Python and 91 native tests passed. Four configuration-driven collection checks totaled 256 decisions. The existing dataset validates episode boundaries and records terminal versus truncated outcomes. No goal generator or reaching controller is implemented yet.
