# Scripted practice and goal generation

Requested extension, 2026-09-14. User selected both player and enemies/targets. G0-G2 complete; G3 controller design remains deferred. First delivery: native player action/waypoint scripts, static or linearly moving normal square enemies, and real rendered goal captures.

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

Delivery gates:

| Phase | Work | Acceptance |
|---|---|---|
| G0: contract | Resolve actors, action timing, arrival/hold criteria and goal artifact format | Concrete schema and native scope reviewed |
| G1: implementation | Native script execution and capture/provenance plumbing | Deterministic actions, bounded timeouts, valid reached goals, no state features in learner samples |
| G2: collection verification | Small safe practice corpus from frozen G1 code | Replay and captured goals agree; no reset crossings; no model training |
| G3: later design | Stationary/moving goal planning and terminal handling | Separate controller contract after representation validation |

Existing evidence: scenario implementation commit `1075e89`; 255 Python and 91 native tests passed. Four configuration-driven collection checks totaled 256 decisions. The existing dataset validates episode boundaries and records terminal versus truncated outcomes. The scripted goal generator is implemented; learned reaching controllers remain deferred.

G0 review: use a dedicated native practice driver with externally controlled enemy mode3; ordinary modes0-2 retain their existing behavior. Scripted actors use real enemy rendering/collision shapes, but their authored paths replace pursuit AI. Full replay uses seed plus script; a bare native snapshot does not contain future external commands. Player MoveTo is an authoring driver using native motion prediction, not a learned policy or obstacle planner. Arrival requires a small position/speed tolerance; holds use neutral actions. Native and Python validate resource bounds independently. Goal images/clips are separate practice artifacts, not automatically added to LeWM training. Keep immortal collision examples separate from lethal dynamics to avoid contradictory pixel/action targets.

Termination reference: [Gymnasium termination versus truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). Completing or timing out an authored script stops collection while Dodge remains alive, so its final transition is truncated. A successful capture can therefore be both `finished=true` and `truncated=true`. Native death alone sets `terminated=true`; the two episode-end flags are never both true.

G1 engineering checks: parent reviewed Luna generator/schema code and validated the native driver. 276 Python tests and 95 native tests passed; Ruff and whitespace checks pass. Native checks include exact action/pixel replay, static and moving actors, arrival/timeout separation, and death taking precedence over a simultaneous timeout. No model training performed.

G2 evidence (2026-09-14): `history/dodge/gymnasium/pixel-repr-ddqn-practice/g2-validation-20260914/`. Example and exact replay each completed in 13 decisions; death and timeout checks each used one decision (28 total). All artifact bytes/hashes and manifests matched on replay. PNG contents matched recorded RGB arrays; initial/final images visually inspected. Death kept its terminal frame and blocked another action; timeout/death produced no valid goal or goal GIF. Native binary SHA256 `61a625a862b2a390d45255b476a6abc0b8bf7ea378f6d40395ef1fc43f2b55a7`; generator SHA256 `4679145917fdfe9f8060b14588b636a42da972d63c7acfea2c6894033ac5ddf0`. No model/controller training or automatic corpus ingestion.
