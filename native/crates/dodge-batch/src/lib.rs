#![doc = "Deterministic serial and parallel native Dodge training lanes."]

use std::fmt::{Display, Formatter};

use dodge_core::{
    Action, AudioEvent, CoreError, FrameEvent, FullState, MlFrameResult, Mode, NativeConfig,
    NativeGame, PicoFixed, PixelFrameResult, RenderState, Snapshot, BUTTON_X_MASK,
    FRAMEBUFFER_SIZE,
};
use rayon::prelude::*;

mod board;
mod collision_image;
mod hazard;
mod ml;

pub use board::{
    Board19x16, BOARD_CHANNELS, BOARD_HEIGHT, BOARD_SIZE, BOARD_VALUES, BOARD_WIDTH,
    FULL_BOARD_CHANNELS, FULL_BOARD_VALUES,
};
pub use collision_image::{
    encode_collision_image, CollisionImage84x84, COLLISION_IMAGE_AREA, COLLISION_IMAGE_BACKGROUND,
    COLLISION_IMAGE_CHANNELS, COLLISION_IMAGE_CONFIG, COLLISION_IMAGE_HEIGHT,
    COLLISION_IMAGE_HOSTILE, COLLISION_IMAGE_OBSERVATION_VERSION, COLLISION_IMAGE_PLAYER,
    COLLISION_IMAGE_VALUES, COLLISION_IMAGE_WIDTH,
};
pub use hazard::{
    HazardObservation, HAZARD_AOE_EXPLOSION, HAZARD_AOE_PATTERN, HAZARD_AOE_PRESENCE,
    HAZARD_AOE_STAGE, HAZARD_AOE_VX, HAZARD_AOE_VY, HAZARD_CHANNELS, HAZARD_DEFAULT_HORIZON,
    HAZARD_DEFAULT_SPAWN_HALO_RADIUS, HAZARD_ENEMY_PRESENCE, HAZARD_ENEMY_TYPE, HAZARD_ENEMY_VX,
    HAZARD_ENEMY_VY, HAZARD_MAX_GRID_SIZE, HAZARD_OBSERVATION_VERSION, HAZARD_PLAYER_PRESENCE,
    HAZARD_SCALARS, HAZARD_SPAWN_CORNER_MASK, HAZARD_SPAWN_HALO, HAZARD_TTC,
};
pub use ml::{
    encode_waypoint_observation, encode_waypoint_observation_from_game, DEFAULT_GRID_SPACING,
    ML_OBSERVATION_SIZE,
};

const START_HOLD_FRAMES: usize = 13;
const AI_STARTUP_MAX_MOVE_FRAMES: usize = 240;
const AI_STARTUP_MAX_WAIT_FRAMES: usize = 240;
const HAZARD_CELL_PARALLEL_MAX_LANES: usize = 16;

pub const PIXEL_WIDTH: usize = dodge_core::FRAMEBUFFER_WIDTH;
pub const PIXEL_HEIGHT: usize = dodge_core::FRAMEBUFFER_HEIGHT;
pub const PIXEL_VALUES: usize = FRAMEBUFFER_SIZE;
pub const ACTION_COUNT: usize = Action::ALL.len();

/// Selects whether independent environment lanes execute serially or with Rayon.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionMode {
    #[default]
    Serial,
    Parallel,
}

/// Chooses which observation buffers a batch call materializes.
///
/// The core always computes a canonical snapshot internally so the game and
/// render boundaries remain the same. Flags control which owned views are
/// retained in the batch result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationFlags {
    pub full_state: bool,
    pub pixels: bool,
    pub board: bool,
    pub collision_image: bool,
    pub include_offscreen_board: bool,
    pub preserve_offscreen_coordinates: bool,
    pub ml: bool,
    pub ml_grid_spacing: u32,
}

impl ObservationFlags {
    pub const fn all() -> Self {
        Self {
            full_state: true,
            pixels: true,
            board: true,
            collision_image: false,
            include_offscreen_board: false,
            preserve_offscreen_coordinates: false,
            ml: false,
            ml_grid_spacing: DEFAULT_GRID_SPACING,
        }
    }

    pub const fn training_board() -> Self {
        Self {
            full_state: false,
            pixels: false,
            board: true,
            collision_image: false,
            include_offscreen_board: false,
            preserve_offscreen_coordinates: false,
            ml: false,
            ml_grid_spacing: DEFAULT_GRID_SPACING,
        }
    }

    pub const fn pixels_and_state() -> Self {
        Self {
            full_state: true,
            pixels: true,
            board: false,
            collision_image: false,
            include_offscreen_board: false,
            preserve_offscreen_coordinates: false,
            ml: false,
            ml_grid_spacing: DEFAULT_GRID_SPACING,
        }
    }
}

impl Default for ObservationFlags {
    fn default() -> Self {
        Self::all()
    }
}

/// Configuration shared by every lane in one batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchConfig {
    pub native: NativeConfig,
    pub step_frames: u32,
    pub observations: ObservationFlags,
    pub execution: ExecutionMode,
}

impl BatchConfig {
    pub const fn new(step_frames: u32) -> Self {
        Self {
            native: NativeConfig::new(42),
            step_frames,
            observations: ObservationFlags::all(),
            execution: ExecutionMode::Serial,
        }
    }

    fn validate(self) -> Result<Self, BatchError> {
        if !(3..=5).contains(&self.step_frames) {
            return Err(BatchError::InvalidStepFrames(self.step_frames));
        }
        if self.observations.ml && self.observations.ml_grid_spacing == 0 {
            return Err(BatchError::InvalidMlGridSpacing(
                self.observations.ml_grid_spacing,
            ));
        }
        Ok(self)
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new(4)
    }
}

/// One ordered result from reset or step.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchObservation {
    pub lane: usize,
    pub seed: u32,
    pub frame: u32,
    pub frames_advanced: u32,
    pub reward: u32,
    pub survival_frames: u32,
    pub shattered: u32,
    pub score: f32,
    pub done: bool,
    pub mode: Mode,
    pub events: Vec<FrameEvent>,
    pub audio: Vec<AudioEvent>,
    pub state_hash: u64,
    pub pixel_hash: u64,
    pub full_state: Option<FullState>,
    pub render_state: Option<RenderState>,
    pub pixels: Option<[u8; FRAMEBUFFER_SIZE]>,
    pub board: Option<Board19x16>,
    pub collision_image: Option<CollisionImage84x84>,
    pub canonical_snapshot: Option<Vec<u8>>,
    pub ml_observation: Option<[f32; ML_OBSERVATION_SIZE]>,
    pub player_position: Option<[f32; 2]>,
}

/// One ordered ML-only result with no render, hash, pixel, board, or snapshot
/// payload.
#[derive(Clone, Debug, PartialEq)]
pub struct MlBatchObservation {
    pub lane: usize,
    pub seed: u32,
    pub frame: u32,
    pub frames_advanced: u32,
    pub reward: u32,
    pub done: bool,
    pub mode: Mode,
    pub ml_observation: [f32; ML_OBSERVATION_SIZE],
    pub player_position: [f32; 2],
}

/// One render-free ML step result that omits the observation vector.
///
/// The waypoint learner holds each selected target for several native steps;
/// intermediate steps only need reward, terminal state, and player position.
/// A full `MlBatchObservation` is still materialized at the decision boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct MlPositionBatchObservation {
    pub lane: usize,
    pub seed: u32,
    pub frame: u32,
    pub frames_advanced: u32,
    pub reward: u32,
    pub done: bool,
    pub mode: Mode,
    pub player_position: [f32; 2],
}

/// One ordered rendered result for pixel-only ML consumers.
///
/// This boundary intentionally carries no snapshot, hashes, board, events,
/// audio, or derived waypoint vector. It is the smallest native result that
/// can drive a pixel learner and the hidden movement controller.
#[derive(Debug, PartialEq)]
pub struct PixelBatchObservation {
    pub lane: usize,
    pub seed: u32,
    pub frame: u32,
    pub frames_advanced: u32,
    pub reward: u32,
    pub survival_frames: u32,
    pub done: bool,
    pub mode: Mode,
    pub pixels: [u8; FRAMEBUFFER_SIZE],
    pub player_position: [f32; 2],
}

impl BatchObservation {
    pub fn is_game_ready(&self) -> bool {
        self.full_state
            .as_ref()
            .is_some_and(|state| state.lifecycle.game_ready)
    }

    pub fn is_dead(&self) -> bool {
        self.full_state
            .as_ref()
            .is_some_and(|state| state.lifecycle.dead)
    }
}

/// Errors raised at the batch boundary before a result can be returned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchError {
    InvalidStepFrames(u32),
    InvalidMlGridSpacing(u32),
    InvalidHazardGridSize(u32),
    InvalidHazardHorizon(u32),
    MlObservationDisabled,
    InvalidLookaheadSteps(u32),
    InvalidWaypointTolerance,
    EmptyBatch,
    EmptySnapshots,
    LaneCountMismatch { expected: usize, actual: usize },
    LaneIndexOutOfBounds { lane: usize, lane_count: usize },
    DuplicateLane(usize),
    LaneAlreadyDone(usize),
    Core(CoreError),
}

impl Display for BatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidStepFrames(frames) => {
                write!(
                    formatter,
                    "batch step_frames must be between 3 and 5: {frames}"
                )
            }
            Self::InvalidMlGridSpacing(spacing) => {
                write!(formatter, "ML grid spacing must be positive: {spacing}")
            }
            Self::InvalidHazardGridSize(size) => {
                write!(
                    formatter,
                    "hazard grid size must be between 1 and {0}: {size}",
                    HAZARD_MAX_GRID_SIZE
                )
            }
            Self::InvalidHazardHorizon(horizon) => {
                write!(
                    formatter,
                    "hazard prediction horizon must be positive: {horizon}"
                )
            }
            Self::MlObservationDisabled => {
                formatter.write_str("fast ML batch operations require ML observation support")
            }
            Self::InvalidLookaheadSteps(steps) => {
                write!(
                    formatter,
                    "counterfactual lookahead must be positive: {steps}"
                )
            }
            Self::InvalidWaypointTolerance => formatter.write_str(
                "waypoint tolerance must be finite, non-negative, and below half spacing",
            ),
            Self::EmptyBatch => formatter.write_str("batch must contain at least one lane"),
            Self::EmptySnapshots => {
                formatter.write_str("counterfactual scoring requires at least one snapshot")
            }
            Self::LaneCountMismatch { expected, actual } => write!(
                formatter,
                "batch lane count mismatch: expected {expected}, got {actual}"
            ),
            Self::LaneIndexOutOfBounds { lane, lane_count } => write!(
                formatter,
                "batch lane index {lane} is outside lane count {lane_count}"
            ),
            Self::DuplicateLane(lane) => write!(formatter, "batch lane {lane} was repeated"),
            Self::LaneAlreadyDone(lane) => write!(formatter, "batch lane {lane} is complete"),
            Self::Core(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for BatchError {}

impl From<CoreError> for BatchError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

/// Persistent independent native environments with stable lane ordering.
#[derive(Clone, Debug)]
pub struct BatchEnvironment {
    config: BatchConfig,
    games: Vec<NativeGame>,
    seeds: Vec<u32>,
    last_frames: Vec<u32>,
    last_survival_frames: Vec<u32>,
    done: Vec<bool>,
}

impl BatchEnvironment {
    pub fn new(config: BatchConfig) -> Result<Self, BatchError> {
        Ok(Self {
            config: config.validate()?,
            games: Vec::new(),
            seeds: Vec::new(),
            last_frames: Vec::new(),
            last_survival_frames: Vec::new(),
            done: Vec::new(),
        })
    }

    pub const fn config(&self) -> BatchConfig {
        self.config
    }

    pub fn lane_count(&self) -> usize {
        self.games.len()
    }

    pub fn reset(&mut self, seeds: &[u32]) -> Result<Vec<BatchObservation>, BatchError> {
        if seeds.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        self.seeds = seeds.to_vec();
        self.games = seeds
            .iter()
            .copied()
            .map(|seed| {
                let mut config = self.config.native;
                config.seed = seed;
                NativeGame::new(config)
            })
            .collect();
        self.last_frames = vec![0; seeds.len()];
        self.last_survival_frames = vec![0; seeds.len()];
        self.done = vec![false; seeds.len()];

        let mut observations = Vec::with_capacity(seeds.len());
        let game_count = self.games.len();
        for lane in 0..seeds.len() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneCountMismatch {
                    expected: seeds.len(),
                    actual: game_count,
                })?;
            let mut snapshot = game.reset();
            for _ in 0..START_HOLD_FRAMES {
                snapshot = game.advance_frame(BUTTON_X_MASK)?.snapshot;
            }
            let state = snapshot.logical_state();
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = state.lifecycle.frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = state.survival_frames;
            }
            observations.push(observation_from_snapshot(
                lane,
                0,
                false,
                Vec::new(),
                Vec::new(),
                &snapshot,
                self.config.observations,
            ));
        }
        Ok(observations)
    }

    /// Reset lanes for AI episodes after the scripted startup boundary.
    ///
    /// The ordinary reset remains at the exact game-ready frame. This variant
    /// moves the player out of the cartridge's center `fyou` trigger and then
    /// advances neutral frames until the first enemy is on-screen, keeping
    /// those non-learning frames inside the native loop.
    pub fn reset_with_startup(
        &mut self,
        seeds: &[u32],
    ) -> Result<Vec<BatchObservation>, BatchError> {
        self.reset(seeds)?;
        let grid_spacing = self.config.observations.ml_grid_spacing;
        for game in &mut self.games {
            prepare_render_startup(game, grid_spacing)?;
        }
        self.refresh_render_reset_observations()
    }

    /// Reset all lanes through the render-free ML boundary.
    pub fn reset_ml(&mut self, seeds: &[u32]) -> Result<Vec<MlBatchObservation>, BatchError> {
        let grid_spacing = self.ml_grid_spacing()?;
        if seeds.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        self.seeds = seeds.to_vec();
        self.games = seeds
            .iter()
            .copied()
            .map(|seed| {
                let mut config = self.config.native;
                config.seed = seed;
                NativeGame::new(config)
            })
            .collect();
        self.last_frames = vec![0; seeds.len()];
        self.last_survival_frames = vec![0; seeds.len()];
        self.done = vec![false; seeds.len()];

        let mut observations = Vec::with_capacity(seeds.len());
        let game_count = self.games.len();
        for (lane, seed) in seeds.iter().copied().enumerate() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneCountMismatch {
                    expected: seeds.len(),
                    actual: game_count,
                })?;
            game.reset_ml();
            for _ in 0..START_HOLD_FRAMES {
                game.advance_frame_ml(BUTTON_X_MASK, BUTTON_X_MASK)?;
            }
            let frame = game.lifecycle().frame;
            let survival_frames = game.survival_frames();
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = survival_frames;
            }
            observations.push(ml_observation_at_reset(lane, seed, game, grid_spacing)?);
        }
        Ok(observations)
    }

    /// Render-free AI reset with native startup movement and hazard wait.
    pub fn reset_ml_with_startup(
        &mut self,
        seeds: &[u32],
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        self.reset_ml(seeds)?;
        let grid_spacing = self.config.observations.ml_grid_spacing;
        for game in &mut self.games {
            prepare_ml_startup(game, grid_spacing)?;
        }
        self.refresh_ml_reset_observations()
    }

    /// Render-free AI reset using centered fixed-N hazard waypoints.
    pub fn reset_ml_with_centered_startup(
        &mut self,
        seeds: &[u32],
        grid_size: u32,
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        if !hazard::validate_grid_size(grid_size) {
            return Err(BatchError::InvalidHazardGridSize(grid_size));
        }
        self.reset_ml(seeds)?;
        for game in &mut self.games {
            prepare_ml_centered_startup(game, grid_size)?;
        }
        self.refresh_ml_reset_observations()
    }

    /// Reset only the requested lanes, preserving every other lane's state.
    pub fn reset_lanes(
        &mut self,
        lanes: &[usize],
        seeds: &[u32],
    ) -> Result<Vec<BatchObservation>, BatchError> {
        if lanes.is_empty() || seeds.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        if lanes.len() != seeds.len() {
            return Err(BatchError::LaneCountMismatch {
                expected: lanes.len(),
                actual: seeds.len(),
            });
        }
        for (position, lane) in lanes.iter().copied().enumerate() {
            if lane >= self.games.len() {
                return Err(BatchError::LaneIndexOutOfBounds {
                    lane,
                    lane_count: self.games.len(),
                });
            }
            if lanes
                .iter()
                .take(position)
                .any(|candidate| *candidate == lane)
            {
                return Err(BatchError::DuplicateLane(lane));
            }
        }

        let mut observations = Vec::with_capacity(lanes.len());
        let lane_count = self.games.len();
        for (lane, seed) in lanes.iter().copied().zip(seeds.iter().copied()) {
            let mut config = self.config.native;
            config.seed = seed;
            let mut game = NativeGame::new(config);
            let mut snapshot = game.reset();
            for _ in 0..START_HOLD_FRAMES {
                snapshot = game.advance_frame(BUTTON_X_MASK)?.snapshot;
            }
            let state = snapshot.logical_state();
            let game_slot = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            *game_slot = game;
            if let Some(seed_slot) = self.seeds.get_mut(lane) {
                *seed_slot = seed;
            }
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = state.lifecycle.frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = state.survival_frames;
            }
            if let Some(done) = self.done.get_mut(lane) {
                *done = false;
            }
            observations.push(observation_from_snapshot(
                lane,
                0,
                false,
                Vec::new(),
                Vec::new(),
                &snapshot,
                self.config.observations,
            ));
        }
        Ok(observations)
    }

    /// Reset selected lanes through the AI startup path.
    pub fn reset_lanes_with_startup(
        &mut self,
        lanes: &[usize],
        seeds: &[u32],
    ) -> Result<Vec<BatchObservation>, BatchError> {
        self.reset_lanes(lanes, seeds)?;
        let lane_count = self.games.len();
        let grid_spacing = self.config.observations.ml_grid_spacing;
        for lane in lanes.iter().copied() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            prepare_render_startup(game, grid_spacing)?;
        }
        self.refresh_render_reset_lane_observations(lanes)
    }

    /// Reset only selected lanes through the render-free ML boundary.
    pub fn reset_ml_lanes(
        &mut self,
        lanes: &[usize],
        seeds: &[u32],
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        let grid_spacing = self.ml_grid_spacing()?;
        validate_reset_lanes(&self.games, lanes, seeds)?;

        let mut observations = Vec::with_capacity(lanes.len());
        let lane_count = self.games.len();
        for (lane, seed) in lanes.iter().copied().zip(seeds.iter().copied()) {
            let mut config = self.config.native;
            config.seed = seed;
            let mut game = NativeGame::new(config);
            game.reset_ml();
            for _ in 0..START_HOLD_FRAMES {
                game.advance_frame_ml(BUTTON_X_MASK, BUTTON_X_MASK)?;
            }
            let frame = game.lifecycle().frame;
            let survival_frames = game.survival_frames();
            let game_slot = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            *game_slot = game;
            if let Some(seed_slot) = self.seeds.get_mut(lane) {
                *seed_slot = seed;
            }
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = survival_frames;
            }
            if let Some(done) = self.done.get_mut(lane) {
                *done = false;
            }
            observations.push(ml_observation_at_reset(
                lane,
                seed,
                self.games
                    .get(lane)
                    .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?,
                grid_spacing,
            )?);
        }
        Ok(observations)
    }

    /// Reset selected lanes through the render-free AI startup path.
    pub fn reset_ml_lanes_with_startup(
        &mut self,
        lanes: &[usize],
        seeds: &[u32],
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        self.reset_ml_lanes(lanes, seeds)?;
        let lane_count = self.games.len();
        let grid_spacing = self.config.observations.ml_grid_spacing;
        for lane in lanes.iter().copied() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            prepare_ml_startup(game, grid_spacing)?;
        }
        self.refresh_ml_reset_lane_observations(lanes)
    }

    /// Reset selected ML lanes using centered fixed-N hazard waypoints.
    pub fn reset_ml_lanes_with_centered_startup(
        &mut self,
        lanes: &[usize],
        seeds: &[u32],
        grid_size: u32,
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        if !hazard::validate_grid_size(grid_size) {
            return Err(BatchError::InvalidHazardGridSize(grid_size));
        }
        self.reset_ml_lanes(lanes, seeds)?;
        let lane_count = self.games.len();
        for lane in lanes.iter().copied() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            prepare_ml_centered_startup(game, grid_size)?;
        }
        self.refresh_ml_reset_lane_observations(lanes)
    }

    pub fn step(&mut self, actions: &[Action]) -> Result<Vec<BatchObservation>, BatchError> {
        self.validate_actions(actions)?;
        match self.config.execution {
            ExecutionMode::Serial => self.step_serial(actions),
            ExecutionMode::Parallel => self.step_parallel(actions),
        }
    }

    /// Advance all lanes through the rendered pixel-only boundary.
    pub fn step_pixels(
        &mut self,
        actions: &[Action],
    ) -> Result<Vec<PixelBatchObservation>, BatchError> {
        self.validate_actions(actions)?;
        match self.config.execution {
            ExecutionMode::Serial => self.step_pixels_serial(actions, None),
            ExecutionMode::Parallel => self.step_pixels_parallel(actions, None),
        }
    }

    /// Step selected lanes only; excluded games and their bookkeeping stay frozen.
    /// Results retain original lane IDs and contain only selected lanes.
    pub fn step_pixels_active(
        &mut self,
        actions: &[Action],
        active: &[bool],
    ) -> Result<Vec<PixelBatchObservation>, BatchError> {
        if actions.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        for actual in [actions.len(), active.len()] {
            if actual != self.games.len() {
                return Err(BatchError::LaneCountMismatch {
                    expected: self.games.len(),
                    actual,
                });
            }
        }
        for (lane, (complete, selected)) in self.done.iter().zip(active).enumerate() {
            if *complete && *selected {
                return Err(BatchError::LaneAlreadyDone(lane));
            }
        }
        match self.config.execution {
            ExecutionMode::Serial => self.step_pixels_serial(actions, Some(active)),
            ExecutionMode::Parallel => self.step_pixels_parallel(actions, Some(active)),
        }
    }

    /// Advance all lanes through the render-free ML boundary.
    pub fn step_ml(&mut self, actions: &[Action]) -> Result<Vec<MlBatchObservation>, BatchError> {
        let grid_spacing = self.ml_grid_spacing()?;
        self.validate_actions(actions)?;
        match self.config.execution {
            ExecutionMode::Serial => self.step_ml_serial(actions, grid_spacing),
            ExecutionMode::Parallel => self.step_ml_parallel(actions, grid_spacing),
        }
    }

    /// Advance all lanes through the render-free ML boundary without encoding
    /// the 225-value observation vector.
    pub fn step_ml_positions(
        &mut self,
        actions: &[Action],
    ) -> Result<Vec<MlPositionBatchObservation>, BatchError> {
        let grid_spacing = self.ml_grid_spacing()?;
        self.validate_actions(actions)?;
        match self.config.execution {
            ExecutionMode::Serial => self.step_ml_positions_serial(actions, grid_spacing),
            ExecutionMode::Parallel => self.step_ml_positions_parallel(actions, grid_spacing),
        }
    }

    /// Materialize the current render-free ML observation without advancing.
    pub fn observe_ml(&self) -> Result<Vec<MlBatchObservation>, BatchError> {
        let grid_spacing = self.ml_grid_spacing()?;
        if self.games.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        self.games
            .iter()
            .enumerate()
            .map(|(lane, game)| {
                let seed = *self
                    .seeds
                    .get(lane)
                    .ok_or(BatchError::LaneIndexOutOfBounds {
                        lane,
                        lane_count: self.games.len(),
                    })?;
                ml_observation_at_reset(lane, seed, game, grid_spacing)
            })
            .collect()
    }

    /// Compute the slow/reference frozen-center hazard field for every lane.
    ///
    /// The live games are borrowed immutably. Each TTC query clones its lane
    /// inside the core, so this operation cannot advance or mutate training
    /// state. Rayon is used only for independent lane work; result ordering
    /// remains the stable lane order.
    pub fn hazard_observations(
        &self,
        grid_size: u32,
        horizon: u32,
        spawn_halo_radius: u32,
    ) -> Result<Vec<HazardObservation>, BatchError> {
        if self.games.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        if !hazard::validate_grid_size(grid_size) {
            return Err(BatchError::InvalidHazardGridSize(grid_size));
        }
        if horizon == 0 {
            return Err(BatchError::InvalidHazardHorizon(horizon));
        }
        let observe_lane = |(lane, game): (usize, &NativeGame)| {
            hazard::observe(lane, game, grid_size, horizon, spawn_halo_radius)
                .map_err(BatchError::Core)
        };
        let observe_parallel_lane = |(lane, game): (usize, &NativeGame)| {
            hazard::observe_parallel(lane, game, grid_size, horizon, spawn_halo_radius)
                .map_err(BatchError::Core)
        };
        match self.config.execution {
            ExecutionMode::Serial => self.games.iter().enumerate().map(observe_lane).collect(),
            ExecutionMode::Parallel => {
                if self.games.len() <= HAZARD_CELL_PARALLEL_MAX_LANES {
                    self.games
                        .par_iter()
                        .enumerate()
                        .map(observe_parallel_lane)
                        .collect()
                } else {
                    self.games
                        .par_iter()
                        .enumerate()
                        .map(observe_lane)
                        .collect()
                }
            }
        }
    }

    /// Score every action from each supplied canonical state independently.
    ///
    /// This method intentionally borrows the live batch immutably. Each
    /// counterfactual restores its own `NativeGame`, so scoring cannot advance,
    /// reset, or otherwise mutate any active training lane.
    pub fn score_actions(
        &self,
        snapshots: &[Vec<u8>],
        lookahead_steps: u32,
    ) -> Result<Vec<[f32; ACTION_COUNT]>, BatchError> {
        if snapshots.is_empty() {
            return Err(BatchError::EmptySnapshots);
        }
        if lookahead_steps == 0 {
            return Err(BatchError::InvalidLookaheadSteps(lookahead_steps));
        }

        let score_snapshot = |bytes: &Vec<u8>| -> Result<[f32; ACTION_COUNT], BatchError> {
            let snapshot = Snapshot::from_canonical_bytes(bytes)?;
            let initial_survival = snapshot.logical_state().survival_frames;
            let base_game = NativeGame::restore(&snapshot)?;
            {
                let mut scores = [0.0; ACTION_COUNT];
                for (score, action) in scores.iter_mut().zip(Action::ALL) {
                    let mut game = base_game.clone();
                    for _ in 0..lookahead_steps {
                        let frame_result = game.step(action, self.config.step_frames)?;
                        if frame_result.done {
                            break;
                        }
                    }
                    *score = game.survival_frames().saturating_sub(initial_survival) as f32;
                }
                Ok(scores)
            }
        };

        match self.config.execution {
            ExecutionMode::Serial => snapshots.iter().map(score_snapshot).collect(),
            ExecutionMode::Parallel => snapshots.par_iter().map(score_snapshot).collect(),
        }
    }

    /// Score the nine relative waypoint choices under the training controller.
    /// Each choice fixes one neighboring target, then recomputes native steering
    /// from the simulated player position for every held decision.
    pub fn score_waypoint_actions(
        &self,
        snapshots: &[Vec<u8>],
        hold_decisions: u32,
        grid_spacing: u32,
        tolerance: f32,
        arrival_latching: bool,
        ban_corner_nodes: bool,
    ) -> Result<Vec<[f32; ACTION_COUNT]>, BatchError> {
        if snapshots.is_empty() {
            return Err(BatchError::EmptySnapshots);
        }
        if hold_decisions == 0 {
            return Err(BatchError::InvalidLookaheadSteps(hold_decisions));
        }
        if grid_spacing == 0 {
            return Err(BatchError::InvalidMlGridSpacing(grid_spacing));
        }
        if !tolerance.is_finite() || tolerance < 0.0 || tolerance >= grid_spacing as f32 / 2.0 {
            return Err(BatchError::InvalidWaypointTolerance);
        }
        let tolerance = PicoFixed::from_f32(tolerance);
        let points = waypoint_axis_points(grid_spacing);
        let score_snapshot = |bytes: &Vec<u8>| -> Result<[f32; ACTION_COUNT], BatchError> {
            let snapshot = Snapshot::from_canonical_bytes(bytes)?;
            let initial_survival = snapshot.logical_state().survival_frames;
            let base_game = NativeGame::restore(&snapshot)?;
            let mut scores = [0.0; ACTION_COUNT];
            for (score, waypoint_action) in scores.iter_mut().zip(Action::ALL) {
                let mut game = base_game.clone();
                let target = waypoint_target(&game, &points, waypoint_action, ban_corner_nodes);
                let mut arrived =
                    arrival_latching && waypoint_reached_with_tolerance(&game, target, tolerance);
                for _ in 0..hold_decisions {
                    let action = waypoint_steering_action(
                        &game,
                        target,
                        tolerance,
                        arrival_latching && arrived,
                    );
                    let result = game.step(action, self.config.step_frames)?;
                    if result.done {
                        break;
                    }
                    if arrival_latching && !arrived {
                        arrived = waypoint_reached_with_tolerance(&game, target, tolerance);
                    }
                }
                *score = game.survival_frames().saturating_sub(initial_survival) as f32;
            }
            Ok(scores)
        };
        match self.config.execution {
            ExecutionMode::Serial => snapshots.iter().map(score_snapshot).collect(),
            ExecutionMode::Parallel => snapshots.par_iter().map(score_snapshot).collect(),
        }
    }

    fn validate_actions(&self, actions: &[Action]) -> Result<(), BatchError> {
        if actions.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        if actions.len() != self.games.len() {
            return Err(BatchError::LaneCountMismatch {
                expected: self.games.len(),
                actual: actions.len(),
            });
        }
        for (lane, complete) in self.done.iter().copied().enumerate() {
            if complete {
                return Err(BatchError::LaneAlreadyDone(lane));
            }
        }
        Ok(())
    }

    fn ml_grid_spacing(&self) -> Result<u32, BatchError> {
        if !self.config.observations.ml {
            return Err(BatchError::MlObservationDisabled);
        }
        Ok(self.config.observations.ml_grid_spacing)
    }

    fn step_serial(&mut self, actions: &[Action]) -> Result<Vec<BatchObservation>, BatchError> {
        let mut observations = Vec::with_capacity(actions.len());
        let game_count = self.games.len();
        for (lane, action) in actions.iter().copied().enumerate() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: game_count,
                })?;
            let result = game.step(action, self.config.step_frames)?;
            observations.push(self.record_step(lane, result)?);
        }
        Ok(observations)
    }

    fn step_parallel(&mut self, actions: &[Action]) -> Result<Vec<BatchObservation>, BatchError> {
        let previous_frames = self.last_frames.clone();
        let previous_survival_frames = self.last_survival_frames.clone();
        let flags = self.config.observations;
        let step_frames = self.config.step_frames;
        let results: Vec<Result<IndexedStep, BatchError>> =
            self.games
                .par_iter_mut()
                .enumerate()
                .zip(actions.par_iter().copied())
                .map(|((lane, game), action)| {
                    let previous_frame = previous_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_frames.len(),
                        },
                    )?;
                    let previous_survival = previous_survival_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_survival_frames.len(),
                        },
                    )?;
                    let result = game.step(action, step_frames)?;
                    Ok(IndexedStep {
                        lane,
                        previous_frame,
                        previous_survival,
                        result,
                        flags,
                    })
                })
                .collect();

        let mut observations = Vec::with_capacity(results.len());
        for indexed in results {
            let indexed = indexed?;
            let lane = indexed.lane;
            let frame_result = indexed.result;
            let observation = observation_from_frame_result(
                lane,
                indexed.previous_frame,
                indexed.previous_survival,
                &frame_result,
                indexed.flags,
            );
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = frame_result.frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = frame_result.snapshot.logical_state().survival_frames;
            }
            if let Some(done) = self.done.get_mut(lane) {
                *done = frame_result.done;
            }
            observations.push(observation);
        }
        Ok(observations)
    }

    fn step_pixels_serial(
        &mut self,
        actions: &[Action],
        active: Option<&[bool]>,
    ) -> Result<Vec<PixelBatchObservation>, BatchError> {
        let mut observations = Vec::with_capacity(actions.len());
        let game_count = self.games.len();
        for (lane, action) in actions.iter().copied().enumerate() {
            if active.is_some_and(|mask| !mask.get(lane).copied().unwrap_or(false)) {
                continue;
            }
            let previous_frame =
                self.last_frames
                    .get(lane)
                    .copied()
                    .ok_or(BatchError::LaneCountMismatch {
                        expected: actions.len(),
                        actual: game_count,
                    })?;
            let previous_survival = self.last_survival_frames.get(lane).copied().ok_or(
                BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: self.last_survival_frames.len(),
                },
            )?;
            let seed = self
                .seeds
                .get(lane)
                .copied()
                .ok_or(BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: self.seeds.len(),
                })?;
            let (observation, frame, survival, done) = {
                let game = self
                    .games
                    .get_mut(lane)
                    .ok_or(BatchError::LaneCountMismatch {
                        expected: actions.len(),
                        actual: game_count,
                    })?;
                let result = game.step_pixels(action, self.config.step_frames)?;
                let frame = result.frame;
                let survival = result.survival_frames;
                let done = result.done;
                let observation = pixel_observation_from_result(
                    lane,
                    seed,
                    previous_frame,
                    previous_survival,
                    result,
                );
                (observation, frame, survival, done)
            };
            self.record_pixel_step(lane, frame, survival, done);
            observations.push(observation);
        }
        Ok(observations)
    }

    fn step_pixels_parallel(
        &mut self,
        actions: &[Action],
        active: Option<&[bool]>,
    ) -> Result<Vec<PixelBatchObservation>, BatchError> {
        let previous_frames = self.last_frames.clone();
        let previous_survival_frames = self.last_survival_frames.clone();
        let seeds = self.seeds.clone();
        let step_frames = self.config.step_frames;
        let results: Vec<Result<IndexedPixelStep, BatchError>> =
            self.games
                .par_iter_mut()
                .enumerate()
                .zip(actions.par_iter().copied())
                .filter(|((lane, _), _)| {
                    active.is_none_or(|mask| mask.get(*lane).copied().unwrap_or(false))
                })
                .map(|((lane, game), action)| {
                    let previous_frame = previous_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_frames.len(),
                        },
                    )?;
                    let previous_survival = previous_survival_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_survival_frames.len(),
                        },
                    )?;
                    let seed = seeds
                        .get(lane)
                        .copied()
                        .ok_or(BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: seeds.len(),
                        })?;
                    let result = game.step_pixels(action, step_frames)?;
                    let observation = pixel_observation_from_result(
                        lane,
                        seed,
                        previous_frame,
                        previous_survival,
                        result,
                    );
                    Ok(IndexedPixelStep { observation })
                })
                .collect();

        let mut observations = Vec::with_capacity(results.len());
        for indexed in results {
            let observation = indexed?.observation;
            self.record_pixel_step(
                observation.lane,
                observation.frame,
                observation.survival_frames,
                observation.done,
            );
            observations.push(observation);
        }
        Ok(observations)
    }

    fn step_ml_serial(
        &mut self,
        actions: &[Action],
        grid_spacing: u32,
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        let mut observations = Vec::with_capacity(actions.len());
        let game_count = self.games.len();
        for (lane, action) in actions.iter().copied().enumerate() {
            let previous_frame =
                self.last_frames
                    .get(lane)
                    .copied()
                    .ok_or(BatchError::LaneCountMismatch {
                        expected: actions.len(),
                        actual: game_count,
                    })?;
            let previous_survival = self.last_survival_frames.get(lane).copied().ok_or(
                BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: self.last_survival_frames.len(),
                },
            )?;
            let seed = self
                .seeds
                .get(lane)
                .copied()
                .ok_or(BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: self.seeds.len(),
                })?;
            let observation = {
                let game = self
                    .games
                    .get_mut(lane)
                    .ok_or(BatchError::LaneCountMismatch {
                        expected: actions.len(),
                        actual: game_count,
                    })?;
                let result = game.step_ml(action, self.config.step_frames)?;
                ml_observation_from_game(
                    lane,
                    seed,
                    previous_frame,
                    previous_survival,
                    result,
                    game,
                    grid_spacing,
                )?
            };
            self.record_ml_step(&observation);
            observations.push(observation);
        }
        Ok(observations)
    }

    fn step_ml_parallel(
        &mut self,
        actions: &[Action],
        grid_spacing: u32,
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        let previous_frames = self.last_frames.clone();
        let previous_survival_frames = self.last_survival_frames.clone();
        let seeds = self.seeds.clone();
        let step_frames = self.config.step_frames;
        let results: Vec<Result<IndexedMlStep, BatchError>> =
            self.games
                .par_iter_mut()
                .enumerate()
                .zip(actions.par_iter().copied())
                .map(|((lane, game), action)| {
                    let previous_frame = previous_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_frames.len(),
                        },
                    )?;
                    let previous_survival = previous_survival_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_survival_frames.len(),
                        },
                    )?;
                    let seed = seeds
                        .get(lane)
                        .copied()
                        .ok_or(BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: seeds.len(),
                        })?;
                    let result = game.step_ml(action, step_frames)?;
                    let observation = ml_observation_from_game(
                        lane,
                        seed,
                        previous_frame,
                        previous_survival,
                        result,
                        game,
                        grid_spacing,
                    )?;
                    Ok(IndexedMlStep { observation })
                })
                .collect();

        let mut observations = Vec::with_capacity(results.len());
        for indexed in results {
            let indexed = indexed?;
            self.record_ml_step(&indexed.observation);
            observations.push(indexed.observation);
        }
        Ok(observations)
    }

    fn step_ml_positions_serial(
        &mut self,
        actions: &[Action],
        grid_spacing: u32,
    ) -> Result<Vec<MlPositionBatchObservation>, BatchError> {
        let mut observations = Vec::with_capacity(actions.len());
        let game_count = self.games.len();
        for (lane, action) in actions.iter().copied().enumerate() {
            let previous_frame =
                self.last_frames
                    .get(lane)
                    .copied()
                    .ok_or(BatchError::LaneCountMismatch {
                        expected: actions.len(),
                        actual: game_count,
                    })?;
            let previous_survival = self.last_survival_frames.get(lane).copied().ok_or(
                BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: self.last_survival_frames.len(),
                },
            )?;
            let seed = self
                .seeds
                .get(lane)
                .copied()
                .ok_or(BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: self.seeds.len(),
                })?;
            let observation = {
                let game = self
                    .games
                    .get_mut(lane)
                    .ok_or(BatchError::LaneCountMismatch {
                        expected: actions.len(),
                        actual: game_count,
                    })?;
                let result = game.step_ml(action, self.config.step_frames)?;
                ml_position_observation_from_game(
                    lane,
                    seed,
                    previous_frame,
                    previous_survival,
                    result,
                    game,
                    grid_spacing,
                )?
            };
            self.record_ml_position_step(&observation);
            observations.push(observation);
        }
        Ok(observations)
    }

    fn step_ml_positions_parallel(
        &mut self,
        actions: &[Action],
        grid_spacing: u32,
    ) -> Result<Vec<MlPositionBatchObservation>, BatchError> {
        let previous_frames = self.last_frames.clone();
        let previous_survival_frames = self.last_survival_frames.clone();
        let seeds = self.seeds.clone();
        let step_frames = self.config.step_frames;
        let results: Vec<Result<IndexedMlPositionStep, BatchError>> =
            self.games
                .par_iter_mut()
                .enumerate()
                .zip(actions.par_iter().copied())
                .map(|((lane, game), action)| {
                    let previous_frame = previous_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_frames.len(),
                        },
                    )?;
                    let previous_survival = previous_survival_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_survival_frames.len(),
                        },
                    )?;
                    let seed = seeds
                        .get(lane)
                        .copied()
                        .ok_or(BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: seeds.len(),
                        })?;
                    let result = game.step_ml(action, step_frames)?;
                    let observation = ml_position_observation_from_game(
                        lane,
                        seed,
                        previous_frame,
                        previous_survival,
                        result,
                        game,
                        grid_spacing,
                    )?;
                    Ok(IndexedMlPositionStep { observation })
                })
                .collect();

        let mut observations = Vec::with_capacity(results.len());
        for indexed in results {
            let observation = indexed?.observation;
            self.record_ml_position_step(&observation);
            observations.push(observation);
        }
        Ok(observations)
    }

    fn record_ml_step(&mut self, observation: &MlBatchObservation) {
        if let Some(last_frame) = self.last_frames.get_mut(observation.lane) {
            *last_frame = observation.frame;
        }
        if let Some(last_survival) = self.last_survival_frames.get_mut(observation.lane) {
            *last_survival = last_survival.saturating_add(observation.reward);
        }
        if let Some(done) = self.done.get_mut(observation.lane) {
            *done = observation.done;
        }
    }

    fn record_ml_position_step(&mut self, observation: &MlPositionBatchObservation) {
        if let Some(last_frame) = self.last_frames.get_mut(observation.lane) {
            *last_frame = observation.frame;
        }
        if let Some(last_survival) = self.last_survival_frames.get_mut(observation.lane) {
            *last_survival = last_survival.saturating_add(observation.reward);
        }
        if let Some(done) = self.done.get_mut(observation.lane) {
            *done = observation.done;
        }
    }

    fn record_pixel_step(&mut self, lane: usize, frame: u32, survival: u32, done: bool) {
        if let Some(last_frame) = self.last_frames.get_mut(lane) {
            *last_frame = frame;
        }
        if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
            *last_survival = survival;
        }
        if let Some(done_slot) = self.done.get_mut(lane) {
            *done_slot = done;
        }
    }

    fn record_step(
        &mut self,
        lane: usize,
        result: dodge_core::FrameResult,
    ) -> Result<BatchObservation, BatchError> {
        let previous_frame =
            self.last_frames
                .get(lane)
                .copied()
                .ok_or(BatchError::LaneCountMismatch {
                    expected: self.games.len(),
                    actual: lane + 1,
                })?;
        let previous_survival =
            self.last_survival_frames
                .get(lane)
                .copied()
                .ok_or(BatchError::LaneCountMismatch {
                    expected: self.games.len(),
                    actual: lane + 1,
                })?;
        let observation = observation_from_frame_result(
            lane,
            previous_frame,
            previous_survival,
            &result,
            self.config.observations,
        );
        if let Some(last_frame) = self.last_frames.get_mut(lane) {
            *last_frame = result.frame;
        }
        if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
            *last_survival = result.snapshot.logical_state().survival_frames;
        }
        if let Some(done) = self.done.get_mut(lane) {
            *done = result.done;
        }
        Ok(observation)
    }

    fn refresh_render_reset_observations(&mut self) -> Result<Vec<BatchObservation>, BatchError> {
        let flags = self.config.observations;
        let lane_count = self.games.len();
        let mut observations = Vec::with_capacity(lane_count);
        for lane in 0..lane_count {
            let snapshot = self
                .games
                .get(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?
                .snapshot();
            let state = snapshot.logical_state();
            let done = state.lifecycle.dead;
            if let Some(done_slot) = self.done.get_mut(lane) {
                *done_slot = done;
            }
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = state.lifecycle.frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = state.survival_frames;
            }
            observations.push(observation_from_snapshot(
                lane,
                0,
                done,
                Vec::new(),
                Vec::new(),
                &snapshot,
                flags,
            ));
        }
        Ok(observations)
    }

    fn refresh_render_reset_lane_observations(
        &mut self,
        lanes: &[usize],
    ) -> Result<Vec<BatchObservation>, BatchError> {
        let flags = self.config.observations;
        let lane_count = self.games.len();
        let mut observations = Vec::with_capacity(lanes.len());
        for lane in lanes.iter().copied() {
            let snapshot = self
                .games
                .get(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?
                .snapshot();
            let state = snapshot.logical_state();
            let done = state.lifecycle.dead;
            if let Some(done_slot) = self.done.get_mut(lane) {
                *done_slot = done;
            }
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = state.lifecycle.frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = state.survival_frames;
            }
            observations.push(observation_from_snapshot(
                lane,
                0,
                done,
                Vec::new(),
                Vec::new(),
                &snapshot,
                flags,
            ));
        }
        Ok(observations)
    }

    fn refresh_ml_reset_observations(&mut self) -> Result<Vec<MlBatchObservation>, BatchError> {
        let grid_spacing = self.ml_grid_spacing()?;
        let lane_count = self.games.len();
        let mut observations = Vec::with_capacity(lane_count);
        for lane in 0..lane_count {
            let seed = *self
                .seeds
                .get(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            let (frame, survival_frames, done) = self
                .games
                .get(lane)
                .map(|game| {
                    (
                        game.lifecycle().frame,
                        game.survival_frames(),
                        game.lifecycle().dead,
                    )
                })
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = survival_frames;
            }
            if let Some(done_slot) = self.done.get_mut(lane) {
                *done_slot = done;
            }
            let game = self
                .games
                .get(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            observations.push(ml_observation_at_reset(lane, seed, game, grid_spacing)?);
        }
        Ok(observations)
    }

    fn refresh_ml_reset_lane_observations(
        &mut self,
        lanes: &[usize],
    ) -> Result<Vec<MlBatchObservation>, BatchError> {
        let grid_spacing = self.ml_grid_spacing()?;
        let lane_count = self.games.len();
        let mut observations = Vec::with_capacity(lanes.len());
        for lane in lanes.iter().copied() {
            let seed = *self
                .seeds
                .get(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            let (frame, survival_frames, done) = self
                .games
                .get(lane)
                .map(|game| {
                    (
                        game.lifecycle().frame,
                        game.survival_frames(),
                        game.lifecycle().dead,
                    )
                })
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = survival_frames;
            }
            if let Some(done_slot) = self.done.get_mut(lane) {
                *done_slot = done;
            }
            let game = self
                .games
                .get(lane)
                .ok_or(BatchError::LaneIndexOutOfBounds { lane, lane_count })?;
            observations.push(ml_observation_at_reset(lane, seed, game, grid_spacing)?);
        }
        Ok(observations)
    }
}

struct IndexedStep {
    lane: usize,
    previous_frame: u32,
    previous_survival: u32,
    result: dodge_core::FrameResult,
    flags: ObservationFlags,
}

struct IndexedMlStep {
    observation: MlBatchObservation,
}

struct IndexedMlPositionStep {
    observation: MlPositionBatchObservation,
}

struct IndexedPixelStep {
    observation: PixelBatchObservation,
}

fn validate_reset_lanes(
    games: &[NativeGame],
    lanes: &[usize],
    seeds: &[u32],
) -> Result<(), BatchError> {
    if lanes.is_empty() || seeds.is_empty() {
        return Err(BatchError::EmptyBatch);
    }
    if lanes.len() != seeds.len() {
        return Err(BatchError::LaneCountMismatch {
            expected: lanes.len(),
            actual: seeds.len(),
        });
    }
    for (position, lane) in lanes.iter().copied().enumerate() {
        if lane >= games.len() {
            return Err(BatchError::LaneIndexOutOfBounds {
                lane,
                lane_count: games.len(),
            });
        }
        if lanes
            .iter()
            .take(position)
            .any(|candidate| *candidate == lane)
        {
            return Err(BatchError::DuplicateLane(lane));
        }
    }
    Ok(())
}

fn prepare_render_startup(game: &mut NativeGame, grid_spacing: u32) -> Result<(), BatchError> {
    let target = startup_up_target(game, grid_spacing);
    for _ in 0..AI_STARTUP_MAX_MOVE_FRAMES {
        if waypoint_reached(game, target) {
            break;
        }
        let result = game.advance_frame(Action::Up.mask())?;
        if result.done {
            return Ok(());
        }
    }
    for _ in 0..AI_STARTUP_MAX_WAIT_FRAMES {
        if has_visible_enemy(game) {
            break;
        }
        let result = game.advance_frame(Action::Neutral.mask())?;
        if result.done {
            break;
        }
    }
    Ok(())
}

fn prepare_ml_startup(game: &mut NativeGame, grid_spacing: u32) -> Result<(), BatchError> {
    let target = startup_up_target(game, grid_spacing);
    prepare_ml_startup_to(game, target)
}

fn prepare_ml_centered_startup(game: &mut NativeGame, grid_size: u32) -> Result<(), BatchError> {
    let target = centered_startup_up_target(game, grid_size);
    for _ in 0..AI_STARTUP_MAX_MOVE_FRAMES {
        if waypoint_reached(game, target) {
            break;
        }
        let action = waypoint_steering_action(game, target, PicoFixed::from_int(2), false);
        let result = game.advance_frame_ml(action.mask(), action.mask())?;
        if result.done {
            return Ok(());
        }
    }
    for _ in 0..AI_STARTUP_MAX_WAIT_FRAMES {
        if has_visible_enemy(game) {
            break;
        }
        let result = game.advance_frame_ml(Action::Neutral.mask(), Action::Neutral.mask())?;
        if result.done {
            break;
        }
    }
    Ok(())
}

fn prepare_ml_startup_to(
    game: &mut NativeGame,
    target: (PicoFixed, PicoFixed),
) -> Result<(), BatchError> {
    for _ in 0..AI_STARTUP_MAX_MOVE_FRAMES {
        if waypoint_reached(game, target) {
            break;
        }
        let result = game.advance_frame_ml(Action::Up.mask(), Action::Up.mask())?;
        if result.done {
            return Ok(());
        }
    }
    for _ in 0..AI_STARTUP_MAX_WAIT_FRAMES {
        if has_visible_enemy(game) {
            break;
        }
        let result = game.advance_frame_ml(Action::Neutral.mask(), Action::Neutral.mask())?;
        if result.done {
            break;
        }
    }
    Ok(())
}

fn startup_up_target(game: &NativeGame, spacing: u32) -> (PicoFixed, PicoFixed) {
    let points = waypoint_axis_points(spacing);
    startup_up_target_from_points(game, &points)
}

fn centered_startup_up_target(game: &NativeGame, grid_size: u32) -> (PicoFixed, PicoFixed) {
    let points = hazard::centered_points(grid_size as usize);
    startup_up_target_from_points(game, &points)
}

fn startup_up_target_from_points(
    game: &NativeGame,
    points: &[PicoFixed],
) -> (PicoFixed, PicoFixed) {
    let player = game.player();
    let column = nearest_waypoint_index(points, player.x);
    let row = nearest_waypoint_index(points, player.y);
    (
        points.get(column).copied().unwrap_or(PicoFixed::ZERO),
        points
            .get(row.saturating_sub(1))
            .copied()
            .unwrap_or(PicoFixed::ZERO),
    )
}

fn waypoint_axis_points(spacing: u32) -> Vec<PicoFixed> {
    let mut points = Vec::new();
    let mut point = 2_u32;
    while point < 125 {
        points.push(PicoFixed::from_int(point as i32));
        let next = point.saturating_add(spacing);
        if next == point {
            break;
        }
        point = next;
    }
    if points.last().copied() != Some(PicoFixed::from_int(125)) {
        points.push(PicoFixed::from_int(125));
    }
    points
}

fn nearest_waypoint_index(points: &[PicoFixed], value: PicoFixed) -> usize {
    let mut nearest = 0;
    let mut distance = i64::MAX;
    for (index, point) in points.iter().copied().enumerate() {
        let difference = i64::from(value.raw()) - i64::from(point.raw());
        let candidate_distance = difference.abs();
        if candidate_distance < distance {
            nearest = index;
            distance = candidate_distance;
        }
    }
    nearest
}

fn waypoint_reached(game: &NativeGame, target: (PicoFixed, PicoFixed)) -> bool {
    waypoint_reached_with_tolerance(game, target, PicoFixed::from_int(2))
}

fn waypoint_reached_with_tolerance(
    game: &NativeGame,
    target: (PicoFixed, PicoFixed),
    tolerance: PicoFixed,
) -> bool {
    let player = game.player();
    player.x.sub(target.0).raw().abs() <= tolerance.raw()
        && player.y.sub(target.1).raw().abs() <= tolerance.raw()
}

fn waypoint_target(
    game: &NativeGame,
    points: &[PicoFixed],
    action: Action,
    ban_corner_nodes: bool,
) -> (PicoFixed, PicoFixed) {
    let player = game.player();
    let current = (
        nearest_waypoint_index(points, player.x),
        nearest_waypoint_index(points, player.y),
    );
    let (horizontal, vertical) = action_delta(action);
    let last = points.len().saturating_sub(1);
    let target = (
        current.0.saturating_add_signed(horizontal).min(last),
        current.1.saturating_add_signed(vertical).min(last),
    );
    let target =
        if ban_corner_nodes && target.0.is_multiple_of(last) && target.1.is_multiple_of(last) {
            current
        } else {
            target
        };
    (
        points.get(target.0).copied().unwrap_or(PicoFixed::ZERO),
        points.get(target.1).copied().unwrap_or(PicoFixed::ZERO),
    )
}

const fn action_delta(action: Action) -> (isize, isize) {
    match action {
        Action::Neutral => (0, 0),
        Action::Left => (-1, 0),
        Action::Right => (1, 0),
        Action::Up => (0, -1),
        Action::Down => (0, 1),
        Action::UpLeft => (-1, -1),
        Action::UpRight => (1, -1),
        Action::DownLeft => (-1, 1),
        Action::DownRight => (1, 1),
    }
}

fn waypoint_steering_action(
    game: &NativeGame,
    target: (PicoFixed, PicoFixed),
    tolerance: PicoFixed,
    arrived: bool,
) -> Action {
    if arrived {
        return Action::Neutral;
    }
    let player = game.player();
    let axis = |delta: i32| {
        if delta < -tolerance.raw() {
            -1
        } else if delta > tolerance.raw() {
            1
        } else {
            0
        }
    };
    match (
        axis(target.0.sub(player.x).raw()),
        axis(target.1.sub(player.y).raw()),
    ) {
        (0, 0) => Action::Neutral,
        (-1, 0) => Action::Left,
        (1, 0) => Action::Right,
        (0, -1) => Action::Up,
        (0, 1) => Action::Down,
        (-1, -1) => Action::UpLeft,
        (1, -1) => Action::UpRight,
        (-1, 1) => Action::DownLeft,
        (1, 1) => Action::DownRight,
        _ => Action::Neutral,
    }
}

fn has_visible_enemy(game: &NativeGame) -> bool {
    game.enemies().iter().any(|enemy| enemy.inside)
}

fn ml_observation_at_reset(
    lane: usize,
    seed: u32,
    game: &NativeGame,
    grid_spacing: u32,
) -> Result<MlBatchObservation, BatchError> {
    let ml_observation = ml::encode_waypoint_observation_from_game(game, grid_spacing)
        .ok_or(BatchError::InvalidMlGridSpacing(grid_spacing))?;
    let player = game.player();
    Ok(MlBatchObservation {
        lane,
        seed,
        frame: game.lifecycle().frame,
        frames_advanced: 0,
        reward: 0,
        done: game.lifecycle().dead,
        mode: game.lifecycle().mode,
        ml_observation,
        player_position: [player.x.to_f32(), player.y.to_f32()],
    })
}

fn ml_observation_from_game(
    lane: usize,
    seed: u32,
    previous_frame: u32,
    previous_survival: u32,
    result: MlFrameResult,
    game: &NativeGame,
    grid_spacing: u32,
) -> Result<MlBatchObservation, BatchError> {
    let ml_observation = ml::encode_waypoint_observation_from_game(game, grid_spacing)
        .ok_or(BatchError::InvalidMlGridSpacing(grid_spacing))?;
    let player = game.player();
    let survival_now = game.survival_frames();
    Ok(MlBatchObservation {
        lane,
        seed,
        frame: result.frame,
        frames_advanced: result.frame.saturating_sub(previous_frame),
        reward: survival_now.saturating_sub(previous_survival),
        done: result.done,
        mode: result.mode,
        ml_observation,
        player_position: [player.x.to_f32(), player.y.to_f32()],
    })
}

fn ml_position_observation_from_game(
    lane: usize,
    seed: u32,
    previous_frame: u32,
    previous_survival: u32,
    result: MlFrameResult,
    game: &NativeGame,
    grid_spacing: u32,
) -> Result<MlPositionBatchObservation, BatchError> {
    let player = game.player();
    if grid_spacing == 0 {
        return Err(BatchError::InvalidMlGridSpacing(grid_spacing));
    }
    Ok(MlPositionBatchObservation {
        lane,
        seed,
        frame: result.frame,
        frames_advanced: result.frame.saturating_sub(previous_frame),
        reward: game.survival_frames().saturating_sub(previous_survival),
        done: result.done,
        mode: result.mode,
        player_position: [player.x.to_f32(), player.y.to_f32()],
    })
}

fn pixel_observation_from_result(
    lane: usize,
    seed: u32,
    previous_frame: u32,
    previous_survival: u32,
    result: PixelFrameResult,
) -> PixelBatchObservation {
    let [player_x, player_y] = result.player_position;
    PixelBatchObservation {
        lane,
        seed,
        frame: result.frame,
        frames_advanced: result.frame.saturating_sub(previous_frame),
        reward: result.survival_frames.saturating_sub(previous_survival),
        survival_frames: result.survival_frames,
        done: result.done,
        mode: result.mode,
        pixels: result.pixels.into_array(),
        player_position: [player_x.to_f32(), player_y.to_f32()],
    }
}

fn observation_from_frame_result(
    lane: usize,
    previous_frame: u32,
    previous_survival: u32,
    result: &dodge_core::FrameResult,
    flags: ObservationFlags,
) -> BatchObservation {
    let frames_advanced = result.frame.saturating_sub(previous_frame);
    let survival_now = result.snapshot.logical_state().survival_frames;
    let reward = survival_now.saturating_sub(previous_survival);
    observation_from_snapshot(
        lane,
        frames_advanced,
        result.done,
        result.events.clone(),
        result.audio.clone(),
        &result.snapshot,
        flags,
    )
    .with_reward(reward)
}

impl BatchObservation {
    fn with_reward(mut self, reward: u32) -> Self {
        self.reward = reward;
        self
    }
}

fn observation_from_snapshot(
    lane: usize,
    frames_advanced: u32,
    done: bool,
    events: Vec<FrameEvent>,
    audio: Vec<AudioEvent>,
    snapshot: &Snapshot,
    flags: ObservationFlags,
) -> BatchObservation {
    let logical_state = snapshot.logical_state();
    BatchObservation {
        lane,
        seed: logical_state.seed,
        frame: logical_state.lifecycle.frame,
        frames_advanced,
        reward: 0,
        survival_frames: logical_state.survival_frames,
        shattered: logical_state.shattered,
        score: logical_state.score.to_f32(),
        done,
        mode: logical_state.lifecycle.mode,
        events,
        audio,
        state_hash: snapshot.state_hash(),
        pixel_hash: snapshot.pixel_hash(),
        full_state: flags.full_state.then(|| logical_state.clone()),
        render_state: flags.full_state.then(|| snapshot.render_state().clone()),
        pixels: flags.pixels.then(|| *snapshot.pixels()),
        board: flags.board.then(|| {
            if flags.preserve_offscreen_coordinates {
                Board19x16::from_full_state_with_coordinates(logical_state)
            } else {
                Board19x16::from_full_state_with_offscreen(
                    logical_state,
                    flags.include_offscreen_board,
                )
            }
        }),
        collision_image: flags
            .collision_image
            .then(|| encode_collision_image(logical_state)),
        canonical_snapshot: flags.full_state.then(|| snapshot.canonical_bytes()),
        ml_observation: flags
            .ml
            .then(|| encode_waypoint_observation(logical_state, flags.ml_grid_spacing))
            .flatten(),
        player_position: flags.ml.then(|| {
            [
                logical_state.player.x.to_f32(),
                logical_state.player.y.to_f32(),
            ]
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Action, BatchConfig, BatchEnvironment, ExecutionMode, ObservationFlags, BOARD_VALUES,
        HAZARD_CHANNELS, HAZARD_DEFAULT_HORIZON, HAZARD_DEFAULT_SPAWN_HALO_RADIUS, HAZARD_SCALARS,
    };

    fn configured(execution: ExecutionMode) -> BatchConfig {
        BatchConfig {
            native: dodge_core::NativeConfig::new(42),
            step_frames: 4,
            observations: ObservationFlags::all(),
            execution,
        }
    }

    fn configured_ml(execution: ExecutionMode, full_state: bool) -> BatchConfig {
        let mut config = configured(execution);
        config.observations = ObservationFlags {
            full_state,
            pixels: false,
            board: false,
            collision_image: false,
            include_offscreen_board: false,
            preserve_offscreen_coordinates: false,
            ml: true,
            ml_grid_spacing: 32,
        };
        config
    }

    fn action_at(index: usize) -> Action {
        Action::ALL
            .get(index % Action::ALL.len())
            .copied()
            .unwrap_or_else(|| unreachable!("action table index is bounded"))
    }

    #[test]
    fn render_free_ml_batch_matches_canonical_batch_trace() {
        let seeds = [42, 13, 30_100];
        let mut canonical = BatchEnvironment::new(configured_ml(ExecutionMode::Serial, true))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut fast = BatchEnvironment::new(configured_ml(ExecutionMode::Serial, false))
            .unwrap_or_else(|_| unreachable!("valid batch config"));

        let canonical_reset = canonical
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("canonical reset should succeed"));
        let fast_reset = fast
            .reset_ml(&seeds)
            .unwrap_or_else(|_| unreachable!("ML reset should succeed"));
        assert_eq!(canonical_reset.len(), fast_reset.len());
        for (canonical, fast) in canonical_reset.iter().zip(fast_reset.iter()) {
            assert_eq!(canonical.lane, fast.lane);
            assert_eq!(canonical.seed, fast.seed);
            assert_eq!(canonical.frame, fast.frame);
            assert_eq!(canonical.mode, fast.mode);
            assert_eq!(canonical.ml_observation, Some(fast.ml_observation));
            assert_eq!(canonical.player_position, Some(fast.player_position));
        }

        for step in 0..90 {
            let actions = [action_at(step), action_at(step + 2), action_at(step + 5)];
            let canonical_step = canonical
                .step(&actions)
                .unwrap_or_else(|_| unreachable!("canonical step should succeed"));
            let fast_step = fast
                .step_ml(&actions)
                .unwrap_or_else(|_| unreachable!("ML step should succeed"));
            for (canonical, fast) in canonical_step.iter().zip(fast_step.iter()) {
                assert_eq!(canonical.lane, fast.lane);
                assert_eq!(canonical.seed, fast.seed);
                assert_eq!(canonical.frame, fast.frame);
                assert_eq!(canonical.frames_advanced, fast.frames_advanced);
                assert_eq!(canonical.reward, fast.reward);
                assert_eq!(canonical.done, fast.done);
                assert_eq!(canonical.mode, fast.mode);
                assert_eq!(canonical.ml_observation, Some(fast.ml_observation));
                assert_eq!(canonical.player_position, Some(fast.player_position));
            }
            if canonical_step.iter().any(|observation| observation.done) {
                break;
            }
        }
    }

    #[test]
    fn render_free_ml_serial_and_parallel_batches_match() {
        let seeds = [42, 13, 30_100, 7];
        let mut serial = BatchEnvironment::new(configured_ml(ExecutionMode::Serial, false))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut parallel = BatchEnvironment::new(configured_ml(ExecutionMode::Parallel, false))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        assert_eq!(serial.reset_ml(&seeds), parallel.reset_ml(&seeds));

        for step in 0..90 {
            let actions = seeds
                .iter()
                .enumerate()
                .map(|(lane, _)| action_at(step + lane * 3))
                .collect::<Vec<_>>();
            let serial_step = serial
                .step_ml(&actions)
                .unwrap_or_else(|_| unreachable!("serial ML step should succeed"));
            let parallel_step = parallel
                .step_ml(&actions)
                .unwrap_or_else(|_| unreachable!("parallel ML step should succeed"));
            assert_eq!(serial_step, parallel_step);
            if serial_step.iter().any(|observation| observation.done) {
                break;
            }
        }
    }

    #[test]
    fn compact_ml_steps_preserve_state_until_observation_is_requested() {
        let seeds = [42, 13, 30_100];
        let mut full = BatchEnvironment::new(configured_ml(ExecutionMode::Serial, false))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut compact = BatchEnvironment::new(configured_ml(ExecutionMode::Serial, false))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        assert_eq!(full.reset_ml(&seeds), compact.reset_ml(&seeds));

        for step in 0..90 {
            let actions = [action_at(step), action_at(step + 2), action_at(step + 5)];
            let full_step = full
                .step_ml(&actions)
                .unwrap_or_else(|_| unreachable!("full ML step should succeed"));
            let compact_step = compact
                .step_ml_positions(&actions)
                .unwrap_or_else(|_| unreachable!("compact ML step should succeed"));
            for (full, compact) in full_step.iter().zip(compact_step.iter()) {
                assert_eq!(full.lane, compact.lane);
                assert_eq!(full.seed, compact.seed);
                assert_eq!(full.frame, compact.frame);
                assert_eq!(full.frames_advanced, compact.frames_advanced);
                assert_eq!(full.reward, compact.reward);
                assert_eq!(full.done, compact.done);
                assert_eq!(full.mode, compact.mode);
                assert_eq!(full.player_position, compact.player_position);
            }
            let materialized = compact
                .observe_ml()
                .unwrap_or_else(|_| unreachable!("current ML observation should succeed"));
            for (full, current) in full_step.iter().zip(materialized.iter()) {
                assert_eq!(full.ml_observation, current.ml_observation);
                assert_eq!(full.player_position, current.player_position);
            }
            if full_step.iter().any(|observation| observation.done) {
                break;
            }
        }
    }

    #[test]
    fn rendered_pixel_boundary_matches_canonical_batch_trace() {
        let seeds = [42, 13, 30_100];
        let mut canonical = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut pixels = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        canonical
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("canonical reset should succeed"));
        pixels
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("pixel reset should succeed"));

        for step in 0..90 {
            let actions = [action_at(step), action_at(step + 2), action_at(step + 5)];
            let canonical_step = canonical
                .step(&actions)
                .unwrap_or_else(|_| unreachable!("canonical step should succeed"));
            let pixel_step = pixels
                .step_pixels(&actions)
                .unwrap_or_else(|_| unreachable!("pixel step should succeed"));
            for (canonical, pixel) in canonical_step.iter().zip(pixel_step.iter()) {
                assert_eq!(canonical.lane, pixel.lane);
                assert_eq!(canonical.seed, pixel.seed);
                assert_eq!(canonical.frame, pixel.frame);
                assert_eq!(canonical.frames_advanced, pixel.frames_advanced);
                assert_eq!(canonical.reward, pixel.reward);
                assert_eq!(canonical.survival_frames, pixel.survival_frames);
                assert_eq!(canonical.done, pixel.done);
                assert_eq!(canonical.mode, pixel.mode);
                assert_eq!(canonical.pixels.as_ref(), Some(&pixel.pixels));
                let canonical_position = canonical
                    .full_state
                    .as_ref()
                    .map(|state| [state.player.x.to_f32(), state.player.y.to_f32()]);
                assert_eq!(canonical_position, Some(pixel.player_position));
            }
            if canonical_step.iter().any(|observation| observation.done) {
                break;
            }
        }
    }

    #[test]
    fn reset_starts_every_lane_at_the_game_ready_boundary() {
        let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let observations = environment
            .reset(&[13, 27])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        assert_eq!(observations.len(), 2);
        assert!(observations
            .iter()
            .all(|observation| observation.frame == 13));
        assert!(observations
            .iter()
            .all(|observation| observation.is_game_ready()));
        assert!(observations
            .iter()
            .all(|observation| observation.reward == 0));
    }

    #[test]
    fn serial_and_parallel_lanes_match_in_order() {
        let seeds = [13, 27, 58, 101];
        let actions = [
            Action::Neutral,
            Action::Left,
            Action::UpRight,
            Action::DownLeft,
        ];
        let mut serial = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut parallel = BatchEnvironment::new(configured(ExecutionMode::Parallel))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let serial_reset = serial
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        let parallel_reset = parallel
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        assert_eq!(serial_reset, parallel_reset);

        for _ in 0..90 {
            let serial_step = serial
                .step(&actions)
                .unwrap_or_else(|_| unreachable!("serial step should succeed"));
            let parallel_step = parallel
                .step(&actions)
                .unwrap_or_else(|_| unreachable!("parallel step should succeed"));
            assert_eq!(serial_step, parallel_step);
            if serial_step.iter().any(|observation| observation.done) {
                break;
            }
        }
    }

    #[test]
    fn reset_lanes_preserves_unselected_lane_progress() {
        let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        environment
            .reset(&[13, 27])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        environment
            .step(&[Action::Neutral, Action::Left])
            .unwrap_or_else(|_| unreachable!("first step should succeed"));

        let reset = environment
            .reset_lanes(&[1], &[99])
            .unwrap_or_else(|_| unreachable!("lane reset should succeed"));
        let reset_observation = reset
            .first()
            .unwrap_or_else(|| unreachable!("one lane should be reset"));
        assert_eq!(reset.len(), 1);
        assert_eq!(reset_observation.lane, 1);
        assert_eq!(reset_observation.seed, 99);
        assert_eq!(reset_observation.frame, 13);

        let mixed = environment
            .step(&[Action::Neutral, Action::Neutral])
            .unwrap_or_else(|_| unreachable!("step after lane reset should succeed"));
        let mixed_lane0 = mixed
            .first()
            .unwrap_or_else(|| unreachable!("lane zero result should exist"));
        let mixed_lane1 = mixed
            .get(1)
            .unwrap_or_else(|| unreachable!("lane one result should exist"));
        assert_eq!(mixed_lane0.seed, 13);
        assert_eq!(mixed_lane0.frame, 21);
        assert_eq!(mixed_lane1.seed, 99);
        assert_eq!(mixed_lane1.frame, 17);

        let mut control = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        control
            .reset(&[13])
            .unwrap_or_else(|_| unreachable!("control reset should succeed"));
        control
            .step(&[Action::Neutral])
            .unwrap_or_else(|_| unreachable!("control first step should succeed"));
        let control_step = control
            .step(&[Action::Neutral])
            .unwrap_or_else(|_| unreachable!("control second step should succeed"));
        let control_observation = control_step
            .first()
            .unwrap_or_else(|| unreachable!("control result should exist"));
        assert_eq!(mixed_lane0.state_hash, control_observation.state_hash);
        assert_eq!(mixed_lane0.pixel_hash, control_observation.pixel_hash);
    }

    #[test]
    fn board_buffer_has_the_documented_flat_length() {
        let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let observations = environment
            .reset(&[42])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        let board = observations
            .first()
            .and_then(|observation| observation.board.as_ref())
            .unwrap_or_else(|| unreachable!("board requested"));
        assert_eq!(board.as_slice().len(), BOARD_VALUES);
        assert!(board.as_slice().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn collision_image_reset_and_step_are_exposed_without_render_pixels() {
        let mut config = configured(ExecutionMode::Serial);
        let mut flags = ObservationFlags::all();
        flags.full_state = false;
        flags.pixels = false;
        flags.board = false;
        flags.collision_image = true;
        config.observations = flags;

        let mut environment = BatchEnvironment::new(config)
            .unwrap_or_else(|_| unreachable!("valid collision image config"));
        let reset = environment
            .reset(&[42])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        let reset_observation = reset
            .first()
            .unwrap_or_else(|| unreachable!("one reset lane should exist"));
        let reset_image = reset_observation
            .collision_image
            .as_ref()
            .unwrap_or_else(|| unreachable!("collision image requested"));
        assert!(reset_observation.pixels.is_none());
        assert_eq!(reset_image.shape(), (84, 84));
        assert_eq!(reset_image.as_slice().len(), super::COLLISION_IMAGE_VALUES);

        let step = environment
            .step(&[Action::Neutral])
            .unwrap_or_else(|_| unreachable!("step should succeed"));
        let step_observation = step
            .first()
            .unwrap_or_else(|| unreachable!("one step lane should exist"));
        assert_eq!(step_observation.frames_advanced, 4);
        assert!(step_observation.collision_image.is_some());
        assert!(step_observation.pixels.is_none());
    }

    #[test]
    fn hazard_reference_is_parallel_deterministic_and_live_state_preserving() {
        let seeds = [13, 27];
        let mut serial = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut parallel = BatchEnvironment::new(configured(ExecutionMode::Parallel))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        serial
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("serial reset should succeed"));
        parallel
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("parallel reset should succeed"));

        let serial_field = serial
            .hazard_observations(4, HAZARD_DEFAULT_HORIZON, HAZARD_DEFAULT_SPAWN_HALO_RADIUS)
            .unwrap_or_else(|_| unreachable!("serial hazard observation should succeed"));
        let parallel_field = parallel
            .hazard_observations(4, HAZARD_DEFAULT_HORIZON, HAZARD_DEFAULT_SPAWN_HALO_RADIUS)
            .unwrap_or_else(|_| unreachable!("parallel hazard observation should succeed"));
        assert_eq!(serial_field, parallel_field);
        assert_eq!(serial_field.len(), seeds.len());
        assert_eq!(
            serial_field
                .first()
                .map(|observation| observation.hazard_observation.len()),
            Some(HAZARD_CHANNELS * 4 * 4 + HAZARD_SCALARS)
        );

        let serial_step = serial
            .step(&[Action::Neutral, Action::Left])
            .unwrap_or_else(|_| unreachable!("serial step should succeed"));
        let parallel_step = parallel
            .step(&[Action::Neutral, Action::Left])
            .unwrap_or_else(|_| unreachable!("parallel step should succeed"));
        assert_eq!(serial_step, parallel_step);
    }

    #[test]
    fn observation_flags_expose_owned_typed_buffers_without_implicit_views() {
        let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let observations = environment
            .reset(&[42])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        let observation = observations
            .first()
            .unwrap_or_else(|| unreachable!("one lane should be present"));
        let state = observation
            .full_state
            .as_ref()
            .unwrap_or_else(|| unreachable!("full state requested"));
        assert_eq!(state.lifecycle.frame, observation.frame);
        assert_eq!(
            observation
                .render_state
                .as_ref()
                .unwrap_or_else(|| unreachable!("render state accompanies full state"))
                .clip_width,
            128
        );
        assert_eq!(
            observation
                .pixels
                .as_ref()
                .unwrap_or_else(|| unreachable!("pixels requested"))
                .len(),
            super::PIXEL_VALUES
        );
        assert_eq!(
            observation
                .board
                .as_ref()
                .unwrap_or_else(|| unreachable!("board requested"))
                .shape(),
            (19, 16, 16)
        );
        let bytes = observation
            .canonical_snapshot
            .as_ref()
            .unwrap_or_else(|| unreachable!("canonical snapshot requested"));
        let restored = dodge_core::Snapshot::from_canonical_bytes(bytes)
            .unwrap_or_else(|_| unreachable!("owned snapshot should decode"));
        assert_eq!(restored.state_hash(), observation.state_hash);
        assert_eq!(restored.pixel_hash(), observation.pixel_hash);
    }

    #[test]
    fn counterfactual_scores_are_repeatable_and_do_not_mutate_live_lanes() {
        let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        environment
            .reset(&[42])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        let before = environment
            .step(&[Action::Neutral])
            .unwrap_or_else(|_| unreachable!("step should succeed"));
        let snapshot = before
            .first()
            .and_then(|observation| observation.canonical_snapshot.clone())
            .unwrap_or_else(|| unreachable!("full snapshot should be present"));

        let first = environment
            .score_actions(std::slice::from_ref(&snapshot), 8)
            .unwrap_or_else(|_| unreachable!("counterfactual scoring should succeed"));
        let second = environment
            .score_actions(std::slice::from_ref(&snapshot), 8)
            .unwrap_or_else(|_| unreachable!("counterfactual scoring should repeat"));
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        let first_scores = first
            .first()
            .unwrap_or_else(|| unreachable!("one score row should be present"));
        assert_eq!(first_scores.len(), Action::ALL.len());
        assert!(first_scores.iter().all(|score| score.is_finite()));

        let after = environment
            .step(&[Action::Left])
            .unwrap_or_else(|_| unreachable!("live lane should remain usable"));
        let mut control = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        control
            .reset(&[42])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        control
            .step(&[Action::Neutral])
            .unwrap_or_else(|_| unreachable!("step should succeed"));
        let control_after = control
            .step(&[Action::Left])
            .unwrap_or_else(|_| unreachable!("control step should succeed"));
        assert_eq!(after, control_after);
    }

    #[test]
    fn counterfactual_serial_and_parallel_scores_match() {
        let snapshot = {
            let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
                .unwrap_or_else(|_| unreachable!("valid batch config"));
            environment
                .reset(&[13, 27])
                .unwrap_or_else(|_| unreachable!("reset should succeed"))
                .into_iter()
                .map(|observation| {
                    observation
                        .canonical_snapshot
                        .unwrap_or_else(|| unreachable!("full snapshot should be present"))
                })
                .collect::<Vec<_>>()
        };
        let mut serial = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut parallel = BatchEnvironment::new(configured(ExecutionMode::Parallel))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        serial
            .reset(&[13, 27])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        parallel
            .reset(&[13, 27])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        assert_eq!(
            serial
                .score_actions(&snapshot, 8)
                .unwrap_or_else(|_| unreachable!("serial scoring should succeed")),
            parallel
                .score_actions(&snapshot, 8)
                .unwrap_or_else(|_| unreachable!("parallel scoring should succeed"))
        );
    }

    #[test]
    fn counterfactual_scoring_rejects_empty_and_zero_lookahead() {
        let environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        assert_eq!(
            environment.score_actions(&[], 1),
            Err(super::BatchError::EmptySnapshots)
        );
        assert_eq!(
            environment.score_actions(&[vec![0]], 0),
            Err(super::BatchError::InvalidLookaheadSteps(0))
        );
    }
}
