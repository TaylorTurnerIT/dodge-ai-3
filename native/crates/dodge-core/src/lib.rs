#![doc = "Engine-free deterministic core for the native Dodge runtime."]

mod action;
mod config;
mod error;
mod fixed;
mod game;
mod input;
mod lifecycle;
mod patterns;
mod rng;
mod snapshot;
mod state;

pub use action::{Action, BUTTON_X_MASK};
pub use config::NativeConfig;
pub use error::CoreError;
pub use fixed::{pico_ceil, pico_floor, pico_mid, pico_mod, PicoFixed};
pub use game::{AudioEvent, FrameEvent, FrameResult, MlFrameResult, NativeGame, PixelFrameResult};
pub use input::{Button, InputState, BUTTON_MASK_LIMIT};
pub use lifecycle::{LifecycleState, Mode};
pub use patterns::{PatternRect, PatternState, PatternTarget, SpawnPoint, WarningLine};
pub use rng::{PicoRng, RngCheckpoint};
pub use snapshot::{
    FullState, IndexedFramebuffer, RenderState, Snapshot, SnapshotProvenance,
    CARTRIDGE_SOURCE_SHA256, FRAMEBUFFER_HEIGHT, FRAMEBUFFER_SIZE, FRAMEBUFFER_WIDTH, PALETTE_SIZE,
};
pub use state::{EnemyState, ParticleState, PlayerState, SettingsState};

/// Workspace-level version of native state contract.
pub const CORE_SCHEMA_VERSION: u32 = 1;
