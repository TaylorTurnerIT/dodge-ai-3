# RGB target-refresh experiment

Both treatments use source commit `e7afc03`, native 128×128 RGB, four frames,
700 game seeds, learner seed 42, learning rate 1e-4, batch size 32, 100k replay
capacity, one optimizer update per four decisions, and unchanged native rewards.
The only treatment difference is target refresh every 1,000 or 10,000 optimizer
updates. Each fresh learner runs 500k decisions, saving 10k and 200k checkpoints.
Warmup is 5k; epsilon decays over 500k decisions. Snapshot evaluation happens
after training so it does not disturb the training RNG sequence.

Each checkpoint is evaluated on 128 inner and 128 holdout scenarios with seed
base 512 and the existing disjoint offsets. Evaluation allows 4,096 decisions
per episode; results must report censoring. This is a one-learner-seed screen,
not evidence of robustness across learner initializations. Grayscale follows
as a separate ablation, as agreed with the user.

## GPU gate

NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu128. Three alternating pairs of 10k
training decisions compared routine logging every 1,000 decisions with minimal
logging. Each run performed 1,251 optimizer updates. A separate warmup run was
excluded. Added training-loop wall time was 0.93%, 0.31%, and 1.00%, averaging
0.75%; all pairs passed the 5% budget. This measures incremental routine logging
over retained mandatory counters and first/final logs, not the cost of all
observability against a telemetry-free implementation.

The GPU microbenchmark also passed exact normalized-input and weight-update
parity across dense, packed, and diagnostic-sampling modes. Packed sampling
averaged 2.03 ms per 32-transition batch. Packed transfer, decode, and optimizer
updates averaged 5.40 ms per update over three 50-update trials. These component
timings are not end-to-end training throughput.

## Execution

- Colab session: `dodge-rgb-target-v1`.
- First run: `rgb700-ts1000-500k-s42-v1`; observed running beyond 43k decisions.
- Queued next: `rgb700-ts10000-500k-s42-v1`.
- Remote log: `/content/rgb-campaign.log`.
- Local collector log: `/tmp/ddqn-rgb-collector.log`.
- Downloaded gate evidence: `/tmp/ddqn-rgb-telemetry-gate.json`.
- Downloaded microbenchmark: `/tmp/ddqn-rgb-gpu-microbenchmark.json`.

The collector downloads each completed archive into
`history/dodge/gymnasium/cnn-image-ddqn/` and releases only the assigned A100,
after both archives have been collected. It has a three-hour deadline and does
not treat missing or failed runs as complete. The campaign stops before long
runs if the telemetry gate fails. No learning improvement is claimed yet.
