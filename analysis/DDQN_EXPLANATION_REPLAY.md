# DDQN explanation replay

The explanation viewer runs a saved checkpoint in fresh native lanes. It never
reads a live learner or changes a training run. It selects the best, median, and
worst episodes **within the recorded holdout split**, using the run's game
configuration and evaluation decision cap. These are descriptive examples, not
proof of a particular avoidance strategy.

Start it from the repository:

```bash
scripts/uv-run python -m dodge_native_game.variants.cnn_image_ddqn.explanation_server \
  --run-dir history/dodge/gymnasium/cnn-image-ddqn/seedpool-700-500k-s42-v1 \
  --port 8891
```

The default address is the machine's Tailscale IPv4 address when available.
This is a read-only viewer for trusted network users, not an authenticated
public service. Checkpoints are local, trusted PyTorch artifacts. Requests cannot
select arbitrary checkpoint paths or start training.

## What each decision means

The native game image, input stack, Q-values, and feature maps describe the state
**before** the selected action. Reward, events, and terminal flags describe the
transition **after** that action. The stack is ordered oldest to newest. Four
frames with four native frames per decision span 12 native frames between the
oldest and newest observations; they are not four consecutive game frames.

The UI plots raw Q-values, not probabilities. For the dueling head:

```text
Q(a) = V + A(a) - mean(A)
mean(Q) = V
Q(a) - Q(b) = sum_j [(Wadv[a,j] - Wadv[b,j]) * h[j]]
             + badv[a] - badv[b]
```

The last identity is an exact decomposition of a fixed pair's preference,
subject to floating-point rounding. It is not a semantic explanation of each
feature. V is a learned common baseline, not measured safety. An action gap is
preference, not confidence. Plain-head checkpoints have no value stream; the
viewer must not manufacture one.

## Sensitivity and intervention

For a paused decision, select an alternative action and perturb either all stack
frames at the same spatial location or one selected frame. The original chosen
and alternative actions remain fixed across the scan, even if the edited input
changes argmax. The signed maps report:

```text
value sensitivity    = V(original) - V(perturbed)
decision sensitivity = [Q(chosen)-Q(alternative)](original)
                     - [Q(chosen)-Q(alternative)](perturbed)
```

Positive decision sensitivity means the edit weakened the original preference;
negative means it strengthened it. Zero does not establish that the region is
irrelevant under other edits. Scales and perturbation parameters must be read
with the image. Local blur can produce inputs outside the training distribution.
These are edit-sensitivity maps, not attention weights, object detectors, or
proof of causal understanding.

The default scan has 7×7 sample locations, centered at pixel coordinates
6, 18, …, 78 on each axis. Each edit replaces a square of radius 6 with a
13×13 box-blurred version; blur uses replicated image edges. Radius 3 and 12
are available for comparison. The displayed cells are coarse samples, not
pixel-resolution attributions. All-stack edits apply the same location to every
frame, but blur each frame's own content; they do not average time together.

The feature view shows the 64 final convolution channels. Strong-example search
ranks real captured observations by channel maximum or shared-feature activation.
An activation that is zero throughout the captured episode returns no strong examples.
A channel's spatial peak is a 7×7 feature-grid coordinate; its input receptive
field is 36×36 with stride 8, not an exact detected-object boundary. Suppressing
one channel replaces its activation with zero during offline inference, then
recomputes the heads. This measures an intervention on this network at this
state. Redundant features can mask an individual channel's role.

The implementation is inspired by the value/advantage distinction in
[Wang et al. (2016)](https://proceedings.mlr.press/v48/wangf16.html) and the
perturbation approach of
[Greydanus et al. (2018)](https://proceedings.mlr.press/v80/greydanus18a.html).
It is not a reproduction of either paper's visualization method.

## Evidence and limits

The trace records checkpoint SHA-256, game settings, seed, action cadence,
termination/censoring, and generation time. Regenerated reward is compared with
the recorded evaluation reward; a mismatch is evidence to investigate, not a
score to replace with the expected value. Native reward is separate from Q's
discounted training-reward estimate. Shaped-reward runs are rejected until their
replay reward contract can be reproduced faithfully.

Only native-exposed events may be labeled. Missing pickup types or reward
components remain unavailable; they are not inferred from image brightness,
score changes, or a zero-filled legacy dashboard field.

Explanation computation is on demand, serialized, and cached with a bounded
cache. It does not run for every playback frame. Native replay generation speed,
explanation latency, browser playback cadence, and training throughput have
different denominators and must not be presented as one FPS number.

Use these episodes to generate hypotheses, then test those hypotheses on new
seeds. Once held-out examples influence architecture or reward choices, use a
fresh locked evaluation set for the next generalization claim. This viewer does
not itself establish better learning or explain the entire performance gap.

## Verification record

The reproducible check is:

```bash
scripts/uv-run python scripts/ddqn-explanation-proof.py \
  --run-dir history/dodge/gymnasium/cnn-image-ddqn/seedpool-700-500k-s42-v1 \
  --output analysis/DDQN_EXPLANATION_PROOF.json
scripts/uv-run pytest -q
scripts/uv-run ruff check .
node tests/variant_cnn_image_ddqn/explain_ui.test.cjs
node tests/variant_cnn_image_ddqn/explain_playback.test.cjs
```

The real-checkpoint check reproduces held-out rewards 722, 343, and 193 on seeds
20152, 20155, and 20159. It probes the first, middle, and last decision in each
episode, in both all-stack and single-frame modes: 18 explanation probes.
It checks the value/mean-Q identity and feature-contribution reconstruction to
an absolute tolerance of 0.0001. The JSON retains measured residuals, timings,
checkpoint provenance, and hashes of the explanation source files.

Independent wire-format tests compare channel-63 suppression with manual
convolution-channel zeroing and require an identity blur to produce zero signed
sensitivity. Both dueling and plain heads are tested. This catches errors that
could otherwise produce plausible but incorrectly labeled visualizations.

The JavaScript checks exercise packed-input decoding, null handling, signed
colors, event presence, rendering all 64 maps and 512 contributions, playback
cadence, and pause cancellation with a small DOM stub. Live HTTP checks exercise
the native PNGs, final post-action PNG, trace, explanation, and example endpoints.
Final validation passed 116 Python tests, Ruff, the Node contracts, and a
32-step CPU training smoke run in a temporary history directory.
The original T13 check could not access the browser preview. The follow-up
playback fix was checked in a live browser: 12 normal-load samples and 16 samples
with an added 250 ms image-load delay all showed a visible native frame whose
index matched the replay cursor. Controlled JavaScript tests also cover pending
loads interrupted by pause, seek, or episode replacement, failed-load retry, and
the 24-image cache bound.

The native view is rendered by the native game, not reconstructed from the
collision image. Both capture lanes receive the same chosen action. A separate
32-decision regression resets a fresh native pixel lane and feeds it the stored
actions with model inference disabled; every pre-action PNG and the final PNG
must match byte for byte, along with native frame numbers and termination flags.
An action-only replay of the three served held-out episodes also passed: all
316 decision images (181 best, 86 median, 49 worst), their post-action frame
numbers and termination flags, and all three final images matched. That check
used fresh native lanes and fetched the stored actions and PNGs from the viewer;
it did not load a policy. The follow-up Python suite passed 117 tests, Ruff
passed, and a 32-step CPU smoke run completed in a temporary history directory.

## Playback buffering

The viewer preloads four upcoming native images and retains at most 24 decoded
or pending images. It draws ready images onto a canvas. If the next image is
late, the current native frame, collision input, and predictions stay together
until it arrives. A failed image pauses playback and can be retried with Play.
Pause and seek invalidate pending playback callbacks.

Playback targets the recorded native-frame cadence: four native frames per
decision at 60 Hz means about 15 displayed decisions per second, not 60 distinct
images per second. Slow image loads can reduce playback speed; the viewer does
not advance the predictions ahead of the native image to conceal that delay.
This change affects offline display only, not training or game behavior.

## Viewport layout

Native playback, the input stack, transport controls, and timeline remain on the
left. Q values, events, heatmaps, convolution channels, ablation, shared features,
help, and the final native state have separate inspector tabs. Channel and
feature lists show 16 entries per page without discarding the remaining entries.

Live browser checks at 1280×720 and 1366×768 found no document overflow and no
overflow in the replay cards or any of the eight initial inspector views. Tabs
and pagination also have JavaScript contract tests. Expanded help and loaded
example lists can scroll within their pane; small screens use Replay/Inspector
tabs instead of squeezing both columns together.
