"""CNN image-observation variant for native Dodge training."""

from .agent import DDQNUpdate, DoubleDQNAgent
from .diagnostics import (
    action_balance_ratio,
    action_histogram,
    best_greedy_episode,
    decide_gate,
    effective_decay_steps,
    eval_seed_list,
    greedy_eval_rows,
    q_spread_stats,
    reward_mix_stats,
    td_error_stats,
)
from .env import (
    CNNImageDDQNEnv,
    NativeBatchResultError,
    NativeCollisionImageUnavailable,
)
from .image import (
    COLLISION_IMAGE_SHAPE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    collision_image_from_result,
)
from .model import IMAGE_SHAPE, AtariCnnQNetwork
from .native_replay import NativeReplay, ReplayFrame, generate_native_replay
from .replay import ReplayBatch, ReplayBuffer
from .run_replay import (
    ComparisonEpisode,
    RunReplayConfig,
    generate_run_comparison,
    run_training_curves,
    select_comparison_episodes,
)
from .session import DodgeDDQNSession
from .temporal import TemporalFrameStack

__all__ = [
    "CNNImageDDQNEnv",
    "COLLISION_IMAGE_SHAPE",
    "DDQNUpdate",
    "DoubleDQNAgent",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "IMAGE_SHAPE",
    "NativeBatchResultError",
    "NativeCollisionImageUnavailable",
    "AtariCnnQNetwork",
    "ComparisonEpisode",
    "NativeReplay",
    "ReplayBatch",
    "ReplayBuffer",
    "ReplayFrame",
    "RunReplayConfig",
    "DodgeDDQNSession",
    "TemporalFrameStack",
    "action_balance_ratio",
    "action_histogram",
    "best_greedy_episode",
    "collision_image_from_result",
    "decide_gate",
    "effective_decay_steps",
    "eval_seed_list",
    "generate_native_replay",
    "generate_run_comparison",
    "greedy_eval_rows",
    "q_spread_stats",
    "run_training_curves",
    "select_comparison_episodes",
    "reward_mix_stats",
    "td_error_stats",
]
