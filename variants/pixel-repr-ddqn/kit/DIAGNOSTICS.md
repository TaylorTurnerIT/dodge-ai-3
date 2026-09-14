# Reading the reconstruction panels

The two bottom panels are outputs of a separate decoder fitted on training frames while LeWM is frozen. They are not segmentation masks or the world model's training targets. Implementation: `src/dodge_native_game/variants/pixel_repr_ddqn/probe.py`.

| Panel | Input to decoder | Compare with | Useful evidence |
|---|---|---|---|
| Bottom left: decoded current | Actual current encoder embedding | Top left: actual current frame | Scene detail can be recovered from the embedding by this decoder |
| Bottom right: decoded prediction | Predicted next embedding, using observed history and executed actions | Top right: actual next frame | Predicted embedding can recover the next scene through this decoder |

Early outputs may be mostly background colour and blurry average shapes. With a useful representation and sufficient decoder fitting, current reconstructions should become more faithful to object positions, sizes and boundaries. Next-frame reconstructions should place moving objects where they actually move, rather than merely reproduce their current positions. This is a desired outcome, not a guaranteed sequence of learning stages.

The paper reports early reconstructions dominated by slowly changing features in [Appendix F.2, Figure 10](https://arxiv.org/html/2603.19312v3#A6.F10). For Dodge, background and stationary patterns appearing before small fast objects would be consistent with that observation; it is not a guarantee that the missing details will eventually emerge. The pinned text is in `references/paper/lewm-v3.md`.

If current reconstructions improve while predictions remain misplaced, dynamics learning deserves investigation. The decoder's response to predicted embeddings is another possible cause: it was fitted on actual embeddings. If both panels show nearly the same scene across different inputs, inspect feature spread/rank and decoder underfitting before declaring representation collapse.

The decoder outputs only 32×32 RGB, downsampled from 128×128 native frames during fitting. Small enemies, hollow centres, trails and preview halos can disappear at this resolution. Background pixels also dominate plain mean squared error. A convincing blue background with missing hazards is not enough.

These snapshots are exported after separate decoder fitting; they do not update continuously during world-model optimization. The current MVP used 32 world-model updates followed by 32 decoder updates. That checks the pipeline, not useful Dodge understanding. Compare the same validation windows and equal decoder-fitting budgets across checkpoints. Better pictures can result from decoder learning alone.

The right panel is a one-step forecast with actual observed history. It does not demonstrate a long imagined rollout. Meaningful dynamics evaluation still needs persistence, shuffled-action and temporal controls, plus open-loop forecasts that never receive the actual future. Sharp reconstructions alone do not establish action use, object separation, planning ability or survival.
