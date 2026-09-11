use crate::{
    pico_mid, Action, Button, CoreError, EnemyState, FullState, IndexedFramebuffer, InputState,
    LifecycleState, Mode, NativeConfig, ParticleState, PatternState, PicoFixed, PicoRng,
    PlayerState, SettingsState, Snapshot, SpawnPoint,
};

const PLAYER_MIN: PicoFixed = PicoFixed::from_int(2);
const PLAYER_MAX: PicoFixed = PicoFixed::from_int(125);
const PLAYER_SPEED: PicoFixed = PicoFixed::from_raw(32_768);
const PLAYER_FRICTION: PicoFixed = PicoFixed::from_raw(52_428);
const ENEMY_FRICTION: PicoFixed = PicoFixed::from_raw(64_880);
const ENEMY_ACCELERATION: PicoFixed = PicoFixed::from_raw(655);
const ENEMY_INITIAL_MAX_SIZE: PicoFixed = PicoFixed::from_int(3);
const SPAWN_EDGE: PicoFixed = PicoFixed::from_int(-10);
const SPAWN_EDGE_FAR: PicoFixed = PicoFixed::from_int(138);
const SPAWN_INTERVAL: PicoFixed = PicoFixed::from_int(60);
const ENEMY_HALF_STEP: PicoFixed = PicoFixed::from_raw(32_768);
const DIFFICULTY_SPEED_TARGET: PicoFixed = PicoFixed::from_int(3);
const DIFFICULTY_EST_TARGET: PicoFixed = PicoFixed::from_raw(14_417);
const SCORE_PER_SHATTER: PicoFixed = PicoFixed::from_raw(32_768);
const ENEMY_SPEED_STEP: PicoFixed = PicoFixed::from_raw(655);
const KAMIKAZE_RADIUS_SQUARED: PicoFixed = PicoFixed::from_int(625);
const INITIAL_PATTERN_DELAY_FRAMES: u32 = 420;
const ACTIVE_PATTERN_DELAY_FRAMES: u32 = 600;

#[derive(Clone, Copy)]
struct DifficultyCurve {
    speed_increment: PicoFixed,
    enemy_increment: PicoFixed,
    static_increment: PicoFixed,
    static_target: PicoFixed,
    moving_increment: PicoFixed,
    moving_target: PicoFixed,
}

const DIFFICULTY_CURVES: [DifficultyCurve; 3] = [
    DifficultyCurve {
        speed_increment: PicoFixed::from_raw(131),
        enemy_increment: PicoFixed::from_raw(327),
        static_increment: PicoFixed::from_raw(32),
        static_target: PicoFixed::from_raw(29_491),
        moving_increment: PicoFixed::from_raw(85),
        moving_target: PicoFixed::from_raw(29_491),
    },
    DifficultyCurve {
        speed_increment: PicoFixed::from_raw(452),
        enemy_increment: PicoFixed::from_raw(655),
        static_increment: PicoFixed::from_raw(1_245),
        static_target: PicoFixed::from_raw(29_491),
        moving_increment: PicoFixed::from_raw(1_245),
        moving_target: PicoFixed::from_raw(29_491),
    },
    DifficultyCurve {
        speed_increment: PicoFixed::from_raw(1_835),
        enemy_increment: PicoFixed::from_raw(1_310),
        static_increment: PicoFixed::from_raw(1_572),
        static_target: PicoFixed::from_raw(26_214),
        moving_increment: PicoFixed::from_raw(1_572),
        moving_target: PicoFixed::from_raw(26_214),
    },
];

/// Result of one native simulation frame and its canonical observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameResult {
    pub frame: u32,
    pub mode: Mode,
    pub input_mask: u8,
    pub previous_input_mask: u8,
    pub game_ready: bool,
    pub started: bool,
    pub dead: bool,
    pub done: bool,
    pub reward: PicoFixed,
    pub events: Vec<FrameEvent>,
    pub audio: Vec<AudioEvent>,
    pub snapshot: Snapshot,
}

/// Result of one native simulation frame without render or snapshot capture.
///
/// This boundary is intentionally smaller than `FrameResult`: ML callers use
/// the logical game state and do not need the canonical framebuffer payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlFrameResult {
    pub frame: u32,
    pub mode: Mode,
    pub done: bool,
    pub reward: PicoFixed,
}

/// Result of one rendered frame for pixel-only ML consumers.
///
/// Unlike `FrameResult`, this boundary does not materialize a canonical
/// snapshot, hashes, events, or audio. The simulation and render boundary are
/// shared with the canonical path; only the owned pixel payload and the
/// controller position are returned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PixelFrameResult {
    pub frame: u32,
    pub mode: Mode,
    pub done: bool,
    pub reward: PicoFixed,
    pub survival_frames: u32,
    pub pixels: IndexedFramebuffer,
    pub player_position: [PicoFixed; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureMode {
    None,
    Snapshot,
    Pixels,
}

impl CaptureMode {
    const fn renders(self) -> bool {
        !matches!(self, Self::None)
    }
}

struct FrameBoundary {
    reward: PicoFixed,
    events: Vec<FrameEvent>,
    audio: Vec<AudioEvent>,
    capture: Option<FrameCapture>,
}

struct FrameCapture {
    framebuffer: IndexedFramebuffer,
    render_state: crate::RenderState,
}

/// Source-visible side effects emitted at one completed frame boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameEvent {
    EnemySpawn,
    Collision,
    Death,
    PatternActive,
    Terminal,
}

impl FrameEvent {
    pub const fn name(self) -> &'static str {
        match self {
            Self::EnemySpawn => "enemy_spawn",
            Self::Collision => "collision",
            Self::Death => "death",
            Self::PatternActive => "pattern_active",
            Self::Terminal => "terminal",
        }
    }
}

/// An ordered source audio command emitted at a completed frame boundary.
///
/// The native core exposes identity and channel/timing metadata. It does not
/// synthesize a host audio device, so waveform playback remains a viewer
/// concern just as Macroquad remains outside the simulation core.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioEvent {
    Music { track: u8 },
    Sfx { id: u8, channel: Option<i8> },
}

impl AudioEvent {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Music { .. } => "music",
            Self::Sfx { .. } => "sfx",
        }
    }

    pub const fn id(self) -> u8 {
        match self {
            Self::Music { track } => track,
            Self::Sfx { id, .. } => id,
        }
    }

    pub const fn channel(self) -> Option<i8> {
        match self {
            Self::Music { .. } => None,
            Self::Sfx { channel, .. } => channel,
        }
    }
}

/// Deterministic, engine-free native game boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGame {
    config: NativeConfig,
    lifecycle: LifecycleState,
    input: InputState,
    rng: PicoRng,
    player: PlayerState,
    enemies: Vec<EnemyState>,
    particles: Vec<ParticleState>,
    patterns: Vec<PatternState>,
    active_pattern: Option<usize>,
    spawns: Vec<SpawnPoint>,
    screen: IndexedFramebuffer,
    enemy_timer: PicoFixed,
    enemy_est: PicoFixed,
    enemy_stats: [PicoFixed; 5],
    friendly_timer: u32,
    friendly_enabled: bool,
    enemy_max_size: PicoFixed,
    speed: PicoFixed,
    freeze_rate: PicoFixed,
    freeze_active: bool,
    freeze_timer: u32,
    size_timer: PicoFixed,
    patterns_enabled: bool,
    powerups_enabled: bool,
    pattern_timer: u32,
    pattern_delay_frames: u32,
    pattern_active: bool,
    new_highscore: bool,
    can_click: bool,
    has_played: bool,
    should_collide: bool,
    enemy_should_collide: bool,
    bounce_cap_static: PicoFixed,
    bounce_cap_moving: PicoFixed,
    bounce_cap: PicoFixed,
    score: PicoFixed,
    survival_frames: u32,
    shattered: u32,
    shake: PicoFixed,
    camera_x: PicoFixed,
    camera_y: PicoFixed,
    transition_render_y: i16,
    transition_from: Mode,
    settings: SettingsState,
    highscores: [PicoFixed; 12],
    frame_audio: Vec<AudioEvent>,
}

impl NativeGame {
    pub fn new(config: NativeConfig) -> Self {
        let mut rng = PicoRng::new(config.seed);
        let patterns = crate::patterns::init_patterns(&mut rng);
        Self {
            rng,
            config,
            lifecycle: LifecycleState::new(),
            input: InputState::new(),
            player: PlayerState::new(),
            enemies: Vec::new(),
            particles: Vec::new(),
            patterns,
            active_pattern: None,
            spawns: initial_spawns(),
            screen: IndexedFramebuffer::filled(12),
            enemy_timer: PicoFixed::ZERO,
            enemy_est: initial_enemy_est(config.difficulty),
            enemy_stats: initial_enemy_stats(config.powerups_enabled),
            friendly_timer: 0,
            friendly_enabled: true,
            enemy_max_size: ENEMY_INITIAL_MAX_SIZE,
            speed: initial_speed(config.difficulty),
            freeze_rate: PicoFixed::ONE,
            freeze_active: false,
            freeze_timer: 0,
            size_timer: PicoFixed::ZERO,
            patterns_enabled: config.patterns_enabled,
            powerups_enabled: config.powerups_enabled,
            pattern_timer: 0,
            pattern_delay_frames: INITIAL_PATTERN_DELAY_FRAMES,
            pattern_active: false,
            new_highscore: false,
            can_click: true,
            has_played: false,
            should_collide: true,
            enemy_should_collide: true,
            bounce_cap_static: initial_bounce_static(config.difficulty),
            bounce_cap_moving: initial_bounce_moving(config.difficulty),
            bounce_cap: PicoFixed::from_f32(1.45),
            score: PicoFixed::ZERO,
            survival_frames: 0,
            shattered: 0,
            shake: PicoFixed::ZERO,
            camera_x: PicoFixed::ZERO,
            camera_y: PicoFixed::ZERO,
            transition_render_y: -128,
            transition_from: Mode::Menu,
            settings: SettingsState::new(config),
            highscores: config.highscores,
            frame_audio: Vec::new(),
        }
    }

    pub fn reset(&mut self) -> Snapshot {
        self.reset_state();
        self.snapshot()
    }

    /// Reset the logical game state without creating a render snapshot.
    pub fn reset_ml(&mut self) {
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.lifecycle = LifecycleState::new();
        self.input = InputState::new();
        self.rng.seed(self.config.seed);
        self.patterns = crate::patterns::init_patterns(&mut self.rng);
        self.player = PlayerState::new();
        self.enemies.clear();
        self.particles.clear();
        self.active_pattern = None;
        self.spawns = initial_spawns();
        self.screen = IndexedFramebuffer::filled(12);
        self.enemy_timer = PicoFixed::ZERO;
        self.enemy_est = initial_enemy_est(self.config.difficulty);
        self.enemy_stats = initial_enemy_stats(self.config.powerups_enabled);
        self.friendly_timer = 0;
        self.friendly_enabled = true;
        self.enemy_max_size = ENEMY_INITIAL_MAX_SIZE;
        self.speed = initial_speed(self.config.difficulty);
        self.freeze_rate = PicoFixed::ONE;
        self.freeze_active = false;
        self.freeze_timer = 0;
        self.size_timer = PicoFixed::ZERO;
        self.patterns_enabled = self.config.patterns_enabled;
        self.powerups_enabled = self.config.powerups_enabled;
        self.pattern_timer = 0;
        self.pattern_delay_frames = INITIAL_PATTERN_DELAY_FRAMES;
        self.pattern_active = false;
        self.new_highscore = false;
        self.can_click = true;
        self.should_collide = true;
        self.enemy_should_collide = true;
        self.bounce_cap_static = initial_bounce_static(self.config.difficulty);
        self.bounce_cap_moving = initial_bounce_moving(self.config.difficulty);
        self.bounce_cap = PicoFixed::from_f32(1.45);
        self.score = PicoFixed::ZERO;
        self.survival_frames = 0;
        self.shattered = 0;
        self.shake = PicoFixed::ZERO;
        self.camera_x = PicoFixed::ZERO;
        self.camera_y = PicoFixed::ZERO;
        self.transition_render_y = -128;
        self.transition_from = Mode::Menu;
        self.frame_audio.clear();
    }

    /// Rebuild a native instance at the exact logical and render boundary held
    /// by a canonical snapshot. The snapshot provenance and projected pixels
    /// are validated before any mutable game state is installed.
    pub fn restore(snapshot: &Snapshot) -> Result<Self, CoreError> {
        if snapshot.provenance() != crate::SnapshotProvenance::current() {
            return Err(CoreError::InvalidSnapshotValue);
        }
        let state = snapshot.logical_state().clone();
        let mut rng = PicoRng::new(state.seed);
        rng.restore(state.rng)?;
        let config = NativeConfig {
            seed: state.seed,
            difficulty: state.settings.difficulty,
            patterns_enabled: state.patterns_enabled,
            powerups_enabled: state.powerups_enabled,
            theme_background: state.settings.theme_background,
            theme_shadow: state.settings.theme_shadow,
            highscores: state.highscores,
        };
        Ok(Self {
            config,
            lifecycle: state.lifecycle,
            input: state.input,
            rng,
            player: state.player,
            enemies: state.enemies,
            particles: state.particles,
            patterns: state.patterns,
            active_pattern: state.active_pattern,
            spawns: state.spawns,
            screen: state.physical_screen,
            enemy_timer: state.enemy_timer,
            enemy_est: state.enemy_est,
            enemy_stats: state.enemy_stats,
            friendly_timer: state.friendly_timer,
            friendly_enabled: state.friendly_enabled,
            enemy_max_size: state.enemy_max_size,
            speed: state.speed,
            freeze_rate: state.freeze_rate,
            freeze_active: state.freeze_active,
            freeze_timer: state.freeze_timer,
            size_timer: state.size_timer,
            patterns_enabled: state.patterns_enabled,
            powerups_enabled: state.powerups_enabled,
            pattern_timer: state.pattern_timer,
            pattern_delay_frames: state.pattern_delay_frames,
            pattern_active: state.pattern_active,
            new_highscore: state.new_highscore,
            can_click: state.can_click,
            has_played: state.has_played,
            should_collide: state.should_collide,
            enemy_should_collide: state.enemy_should_collide,
            bounce_cap_static: state.bounce_cap_static,
            bounce_cap_moving: state.bounce_cap_moving,
            bounce_cap: state.bounce_cap,
            score: state.score,
            survival_frames: state.survival_frames,
            shattered: state.shattered,
            shake: state.shake,
            camera_x: state.camera_x,
            camera_y: state.camera_y,
            transition_render_y: state.transition_render_y,
            transition_from: state.transition_from,
            settings: state.settings,
            highscores: state.highscores,
            frame_audio: Vec::new(),
        })
    }

    pub fn restore_bytes(bytes: &[u8]) -> Result<Self, CoreError> {
        let snapshot = Snapshot::from_canonical_bytes(bytes)?;
        Self::restore(&snapshot)
    }

    pub fn advance_frame(&mut self, input_mask: u8) -> Result<FrameResult, CoreError> {
        self.advance_frame_with_post_mask(input_mask, input_mask)
    }

    pub fn advance_frame_with_post_mask(
        &mut self,
        input_mask: u8,
        post_frame_mask: u8,
    ) -> Result<FrameResult, CoreError> {
        let boundary =
            self.advance_frame_internal(input_mask, post_frame_mask, CaptureMode::Snapshot)?;
        let FrameBoundary {
            reward,
            events,
            audio,
            capture,
        } = boundary;
        let Some(FrameCapture {
            framebuffer,
            render_state,
        }) = capture
        else {
            return Err(CoreError::InvalidSnapshotValue);
        };
        let snapshot = Snapshot::with_framebuffer_from_game(self, framebuffer, render_state);
        Ok(self.result_with_snapshot(reward, events, audio, snapshot))
    }

    /// Advance one frame while omitting render and snapshot materialization.
    ///
    /// The simulation, camera RNG, draw-side-effect RNG, and particle updates
    /// remain shared with the canonical path. The physical framebuffer is not
    /// updated because this boundary is only valid for ML consumers that do
    /// not request pixels, hashes, or snapshots.
    pub fn advance_frame_ml(
        &mut self,
        input_mask: u8,
        post_frame_mask: u8,
    ) -> Result<MlFrameResult, CoreError> {
        let boundary =
            self.advance_frame_internal(input_mask, post_frame_mask, CaptureMode::None)?;
        Ok(MlFrameResult {
            frame: self.lifecycle.frame,
            mode: self.lifecycle.mode,
            done: self.lifecycle.dead,
            reward: boundary.reward,
        })
    }

    /// Advance one rendered frame without canonical snapshot materialization.
    ///
    /// Pixel-only learners need the exact indexed framebuffer and player
    /// position, but do not need full-state cloning, hashes, events, or audio.
    /// The capture is taken at the same render boundary as `advance_frame`.
    pub fn advance_frame_pixels(
        &mut self,
        input_mask: u8,
        post_frame_mask: u8,
    ) -> Result<PixelFrameResult, CoreError> {
        let boundary =
            self.advance_frame_internal(input_mask, post_frame_mask, CaptureMode::Pixels)?;
        let FrameBoundary {
            reward, capture, ..
        } = boundary;
        let Some(FrameCapture { framebuffer, .. }) = capture else {
            return Err(CoreError::InvalidSnapshotValue);
        };
        Ok(PixelFrameResult {
            frame: self.lifecycle.frame,
            mode: self.lifecycle.mode,
            done: self.lifecycle.dead,
            reward,
            survival_frames: self.survival_frames,
            pixels: framebuffer,
            player_position: [self.player.x, self.player.y],
        })
    }

    fn advance_frame_internal(
        &mut self,
        input_mask: u8,
        post_frame_mask: u8,
        capture_mode: CaptureMode,
    ) -> Result<FrameBoundary, CoreError> {
        InputState::validate_mask(input_mask)?;
        InputState::validate_mask(post_frame_mask)?;
        self.frame_audio.clear();
        self.input.advance(input_mask)?;
        self.refresh_enemy_stats();
        let mode_before = self.lifecycle.mode;
        let mut events = Vec::new();
        let start_pressed = mode_before == Mode::Menu && self.input.btnp(Button::X);
        if start_pressed {
            self.has_played = true;
            self.transition_from = Mode::Menu;
            self.emit_sfx(55, Some(-2));
        }
        if mode_before == Mode::Menu && self.input.btnp(Button::O) {
            self.begin_transition(Mode::TransitionToSettings);
            self.emit_sfx(55, Some(-2));
            self.emit_sfx(55, Some(0));
        }
        if matches!(mode_before, Mode::Game | Mode::Terminal) && self.input.btnp(Button::O) {
            self.begin_transition(Mode::TransitionToSettings);
            if self.lifecycle.dead {
                self.emit_music(3);
            }
            self.emit_sfx(55, Some(-2));
            self.emit_sfx(55, Some(0));
        }
        self.lifecycle.advance(self.input);
        if start_pressed {
            self.transition_render_y = -108;
        }

        let game_update_from_draw = match mode_before {
            Mode::Menu => false,
            Mode::TransitionToGame => {
                self.transition_render_y += 20;
                self.transition_render_y >= 0
            }
            Mode::TransitionToSettings | Mode::TransitionToMenu => {
                self.transition_render_y -= 20;
                false
            }
            Mode::Game => true,
            Mode::Settings | Mode::Terminal => false,
        };
        match mode_before {
            Mode::TransitionToGame | Mode::TransitionToMenu if self.transition_render_y >= 128 => {
                self.lifecycle.mode = if mode_before == Mode::TransitionToGame {
                    Mode::Game
                } else {
                    Mode::Menu
                };
                self.lifecycle.transition_y = 0;
                self.lifecycle.game_ready = mode_before == Mode::TransitionToGame;
            }
            Mode::TransitionToSettings if self.transition_render_y <= -128 => {
                self.lifecycle.mode = Mode::Settings;
                self.lifecycle.transition_y = 0;
            }
            _ => {}
        }
        if game_update_from_draw {
            self.update_game_frame(&mut events);
        }
        let game_update_from_transition_draw = self.lifecycle.mode == Mode::TransitionToSettings
            && self.transition_render_y > 0
            && matches!(self.transition_from, Mode::Game | Mode::Terminal);
        if game_update_from_transition_draw {
            self.update_game_frame(&mut events);
        }
        if mode_before == Mode::Settings {
            self.update_settings();
        }
        if matches!(mode_before, Mode::Settings) && self.input.btnp(Button::X) {
            let target = if self.has_played {
                Mode::TransitionToGame
            } else {
                Mode::TransitionToMenu
            };
            self.begin_transition(target);
            if self.has_played {
                if self.lifecycle.dead {
                    self.emit_music(22);
                }
                self.emit_sfx(55, Some(-2));
            } else {
                self.emit_sfx(55, Some(2));
            }
        }
        if self.active_pattern.is_some() && mode_before == Mode::Game {
            events.push(FrameEvent::PatternActive);
        }
        if self.lifecycle.dead {
            events.push(FrameEvent::Terminal);
        }
        let reward = if mode_before == Mode::Game && !self.lifecycle.dead {
            self.survival_frames += 1;
            PicoFixed::ONE
        } else {
            PicoFixed::ZERO
        };
        self.update_camera();
        let capture = if capture_mode.renders() {
            let render_state = self.render_current_frame();
            Some(render_state)
        } else {
            None
        };
        self.apply_draw_side_effects(mode_before);
        // The cartridge exits from the update loop immediately after death,
        // before it can advance to the next command. Preserve the terminal
        // frame's simulation mask in both input slots; ordinary command
        // boundaries still expose the scheduled post-frame mask.
        let observed_post_mask = if self.lifecycle.dead {
            input_mask
        } else {
            post_frame_mask
        };
        self.input.finalize_frame(observed_post_mask);
        // Pemsa can schedule a second visible draw on the transition boundary
        // after the canonical capture callback has observed the first draw.
        // Preserve that source-side particle mutation for the next frame while
        // keeping this frame's snapshot at the capture boundary.
        if mode_before == Mode::TransitionToGame
            && self.lifecycle.mode == Mode::Game
            && self.transition_from == Mode::Menu
        {
            self.add_player_trail();
        }
        Ok(FrameBoundary {
            reward,
            events,
            audio: if capture_mode == CaptureMode::Snapshot {
                self.frame_audio.clone()
            } else {
                Vec::new()
            },
            capture,
        })
    }

    pub fn step(&mut self, action: Action, frames: u32) -> Result<FrameResult, CoreError> {
        if frames == 0 {
            return Err(CoreError::InvalidFrameCount(frames));
        }
        let mut result = self.result(PicoFixed::ZERO, Vec::new());
        for _ in 0..frames {
            result = self.advance_frame(action.mask())?;
            if result.done {
                break;
            }
        }
        Ok(result)
    }

    /// Advance several frames through the rendered pixel-only boundary.
    pub fn step_pixels(
        &mut self,
        action: Action,
        frames: u32,
    ) -> Result<PixelFrameResult, CoreError> {
        if frames == 0 {
            return Err(CoreError::InvalidFrameCount(frames));
        }
        let mut result = self.advance_frame_pixels(action.mask(), action.mask())?;
        for _ in 1..frames {
            if result.done {
                break;
            }
            result = self.advance_frame_pixels(action.mask(), action.mask())?;
        }
        Ok(result)
    }

    pub fn step_ml(&mut self, action: Action, frames: u32) -> Result<MlFrameResult, CoreError> {
        if frames == 0 {
            return Err(CoreError::InvalidFrameCount(frames));
        }
        let mut result = MlFrameResult {
            frame: self.lifecycle.frame,
            mode: self.lifecycle.mode,
            done: self.lifecycle.dead,
            reward: PicoFixed::ZERO,
        };
        for _ in 0..frames {
            result = self.advance_frame_ml(action.mask(), action.mask())?;
            if result.done {
                break;
            }
        }
        Ok(result)
    }

    pub const fn lifecycle(&self) -> LifecycleState {
        self.lifecycle
    }

    pub const fn input(&self) -> InputState {
        self.input
    }

    pub const fn rng_checkpoint(&self) -> crate::RngCheckpoint {
        self.rng.checkpoint()
    }

    pub const fn player(&self) -> PlayerState {
        self.player
    }

    pub fn enemies(&self) -> &[EnemyState] {
        self.enemies.as_slice()
    }

    pub fn particles(&self) -> &[ParticleState] {
        self.particles.as_slice()
    }

    pub fn patterns(&self) -> &[PatternState] {
        self.patterns.as_slice()
    }

    pub fn spawns(&self) -> &[SpawnPoint] {
        self.spawns.as_slice()
    }

    pub const fn active_pattern_index(&self) -> Option<usize> {
        self.active_pattern
    }

    pub const fn seed(&self) -> u32 {
        self.config.seed
    }

    pub const fn score(&self) -> PicoFixed {
        self.score
    }

    pub const fn survival_frames(&self) -> u32 {
        self.survival_frames
    }

    pub const fn shattered(&self) -> u32 {
        self.shattered
    }

    pub const fn transition_render_y(&self) -> i16 {
        self.transition_render_y
    }

    pub(crate) fn full_state(&self) -> FullState {
        FullState {
            seed: self.config.seed,
            lifecycle: self.lifecycle,
            input: self.input,
            rng: self.rng.checkpoint(),
            player: self.player,
            enemies: self.enemies.clone(),
            particles: self.particles.clone(),
            patterns: self.patterns.clone(),
            active_pattern: self.active_pattern,
            spawns: self.spawns.clone(),
            physical_screen: self.screen.clone(),
            enemy_timer: self.enemy_timer,
            enemy_est: self.enemy_est,
            enemy_stats: self.enemy_stats,
            friendly_timer: self.friendly_timer,
            friendly_enabled: self.friendly_enabled,
            enemy_max_size: self.enemy_max_size,
            speed: self.speed,
            freeze_rate: self.freeze_rate,
            freeze_active: self.freeze_active,
            freeze_timer: self.freeze_timer,
            size_timer: self.size_timer,
            patterns_enabled: self.patterns_enabled,
            powerups_enabled: self.powerups_enabled,
            pattern_timer: self.pattern_timer,
            pattern_delay_frames: self.pattern_delay_frames,
            pattern_active: self.pattern_active,
            new_highscore: self.new_highscore,
            can_click: self.can_click,
            has_played: self.has_played,
            should_collide: self.should_collide,
            enemy_should_collide: self.enemy_should_collide,
            bounce_cap_static: self.bounce_cap_static,
            bounce_cap_moving: self.bounce_cap_moving,
            bounce_cap: self.bounce_cap,
            score: self.score,
            survival_frames: self.survival_frames,
            shattered: self.shattered,
            shake: self.shake,
            camera_x: self.camera_x,
            camera_y: self.camera_y,
            transition_render_y: self.transition_render_y,
            transition_from: self.transition_from,
            settings: self.settings,
            highscores: self.highscores,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot::from_game(self)
    }

    /// Return the first native frame at which a frozen hypothetical player
    /// would die, or `None` when the declared horizon contains no death.
    ///
    /// The live game is cloned before simulation. The hypothetical player is
    /// placed at the supplied native center coordinate with zero velocity, so
    /// this query cannot move or otherwise mutate the active lane.
    pub fn time_to_death_at(
        &self,
        x: PicoFixed,
        y: PicoFixed,
        horizon: u32,
    ) -> Result<Option<u32>, CoreError> {
        let mut hypothetical = self.clone();
        hypothetical.player.x = x;
        hypothetical.player.y = y;
        hypothetical.player.vx = PicoFixed::ZERO;
        hypothetical.player.vy = PicoFixed::ZERO;
        if hypothetical.player_would_die_now() {
            return Ok(Some(0));
        }
        for frame in 1..=horizon {
            let result = hypothetical.advance_frame_ml(0, 0)?;
            if result.done {
                return Ok(Some(frame));
            }
        }
        Ok(None)
    }

    fn update_game_frame(&mut self, events: &mut Vec<FrameEvent>) {
        if self.friendly_enabled {
            self.update_fyou(events);
        }
        if self.lifecycle.dead && self.can_click && self.input.btnp(Button::X) {
            self.restart_gameplay();
            self.emit_music(3);
            self.emit_sfx(55, Some(-2));
        }
        self.can_click = !self.input.btnp(Button::X);
        self.collision_check(events);
        self.update_player();
        self.update_particles();
        self.update_freeze();
        self.update_size();
        if !self.lifecycle.dead {
            self.spawn_enemies(events);
            self.update_enemies();
            self.update_pattern_schedule();
        }
    }

    fn begin_transition(&mut self, mode: Mode) {
        let from = self.lifecycle.mode;
        self.transition_from = match from {
            Mode::Menu | Mode::Game | Mode::Terminal | Mode::Settings => from,
            _ => self.transition_from,
        };
        match mode {
            Mode::TransitionToGame => {
                self.lifecycle.mode = Mode::TransitionToGame;
                self.lifecycle.transition_y = -128;
                self.transition_render_y = -108;
                self.lifecycle.game_ready = false;
            }
            Mode::TransitionToSettings => {
                self.lifecycle.mode = Mode::TransitionToSettings;
                self.lifecycle.transition_y = 128;
                self.transition_render_y = 108;
            }
            Mode::TransitionToMenu => {
                self.lifecycle.mode = Mode::TransitionToMenu;
                self.lifecycle.transition_y = -128;
                self.transition_render_y = -108;
                self.lifecycle.game_ready = false;
            }
            Mode::Menu | Mode::Game | Mode::Terminal | Mode::Settings => {}
        }
    }

    fn update_settings(&mut self) {
        let gameplay_editable = self.lifecycle.dead || !self.has_played;
        let cursor = self.settings.cursor;
        if self.input.btnp(Button::Left) || self.input.btnp(Button::Right) {
            let increase = self.input.btnp(Button::Right);
            if cursor == 1 {
                self.settings.theme_index = cycle_index(self.settings.theme_index, 13, increase);
                let (background, shadow) = theme_values(self.settings.theme_index);
                self.config.theme_background = background;
                self.config.theme_shadow = shadow;
                self.settings.theme_background = background;
                self.settings.theme_shadow = shadow;
                self.emit_sfx(58, None);
            } else if gameplay_editable {
                match cursor {
                    2 => {
                        self.settings.difficulty =
                            cycle_index(self.settings.difficulty, 3, increase);
                        self.config.difficulty = self.settings.difficulty;
                        self.apply_initial_difficulty();
                        self.emit_sfx(58, None);
                    }
                    3 => {
                        self.settings.patterns_enabled = !self.settings.patterns_enabled;
                        self.config.patterns_enabled = self.settings.patterns_enabled;
                        self.patterns_enabled = self.settings.patterns_enabled;
                        self.emit_sfx(58, None);
                    }
                    4 => {
                        self.settings.powerups_enabled = !self.settings.powerups_enabled;
                        self.config.powerups_enabled = self.settings.powerups_enabled;
                        self.powerups_enabled = self.settings.powerups_enabled;
                        self.emit_sfx(58, None);
                    }
                    _ => {}
                }
            }
            self.settings.message_sprite = if increase { 193 } else { 192 };
            self.settings.message_x = if increase { 80 } else { 96 };
            self.settings.message_y = settings_row(cursor) - 1;
            self.settings.message_timer = 6;
        }
        let max_cursor = if gameplay_editable { 4 } else { 1 };
        if self.input.btnp(Button::Down) {
            self.settings.cursor = self.settings.cursor.saturating_add(1).min(max_cursor);
        }
        if self.input.btnp(Button::Up) {
            self.settings.cursor = self.settings.cursor.saturating_sub(1).max(1);
        }
        if self.settings.message_timer > 0 {
            self.settings.message_timer -= 1;
        }
    }

    fn apply_initial_difficulty(&mut self) {
        self.enemy_est = initial_enemy_est(self.settings.difficulty);
        self.speed = initial_speed(self.settings.difficulty);
        self.bounce_cap_static = initial_bounce_static(self.settings.difficulty);
        self.bounce_cap_moving = initial_bounce_moving(self.settings.difficulty);
    }

    fn restart_gameplay(&mut self) {
        let counters = self
            .patterns
            .iter()
            .map(|pattern| pattern.counter)
            .collect::<Vec<_>>();
        self.lifecycle.dead = false;
        self.score = PicoFixed::ZERO;
        self.survival_frames = 0;
        self.shattered = 0;
        self.shake = PicoFixed::ZERO;
        self.enemy_timer = PicoFixed::ZERO;
        self.enemies.clear();
        self.particles.clear();
        self.apply_initial_difficulty();
        self.size_timer = PicoFixed::ZERO;
        self.player = PlayerState::new();
        self.freeze_active = false;
        self.freeze_timer = 0;
        self.freeze_rate = PicoFixed::ONE;
        self.active_pattern = None;
        self.pattern_active = false;
        self.pattern_timer = 0;
        self.pattern_delay_frames = INITIAL_PATTERN_DELAY_FRAMES;
        self.friendly_enabled = true;
        self.spawns = initial_spawns();
        self.new_highscore = false;
        self.bounce_cap = PicoFixed::from_f32(1.45);

        self.patterns = crate::patterns::init_patterns(&mut self.rng);
        for (index, pattern) in self.patterns.iter_mut().enumerate() {
            let Some(counter) = counters.get(index).copied() else {
                continue;
            };
            let base = if pattern.pattern_type == 1 { 17 } else { 15 };
            let divisor = i32::try_from(pattern.variants.len() + 1).unwrap_or(1);
            let probability = PicoFixed::from_int(base)
                .div_fixed(PicoFixed::from_int(divisor))
                .unwrap_or(PicoFixed::ZERO);
            let mut adjusted = probability.sub(PicoFixed::from_int(counter as i32));
            if probability == PicoFixed::ONE {
                adjusted = adjusted.sub(PicoFixed::from_int(6 * counter as i32));
            }
            pattern.counter = counter;
            pattern.probability = adjusted.add(PicoFixed::from_int(2));
        }
    }

    fn refresh_enemy_stats(&mut self) {
        let powerup_value = if self.powerups_enabled {
            PicoFixed::from_int(2)
        } else {
            PicoFixed::ZERO
        };
        let mut stats = [
            PicoFixed::from_f32(76.5),
            PicoFixed::from_f32(17.5),
            powerup_value,
            powerup_value,
            powerup_value,
        ];
        for enemy in &self.enemies {
            if enemy.personality >= 2 {
                let personality_index = usize::try_from(enemy.personality).unwrap_or(usize::MAX);
                if let Some(value) = stats.get_mut(personality_index) {
                    *value = value.sub(PicoFixed::from_int(2));
                }
                for (index, value) in stats.iter_mut().enumerate().skip(2) {
                    if *value != PicoFixed::ZERO && enemy.personality != index as i8 {
                        *value = value.add(PicoFixed::ONE);
                    }
                    *value = pico_mid(PicoFixed::ZERO, PicoFixed::from_int(3), *value);
                }
            }
        }
        if self.score <= PicoFixed::from_int(10) {
            stats[3] = PicoFixed::ZERO;
        }
        self.enemy_stats = stats;
    }

    fn update_freeze(&mut self) {
        if !self.freeze_active {
            return;
        }
        self.freeze_timer += 1;
        if self.freeze_timer <= 420 {
            self.freeze_rate = PicoFixed::from_f32(0.4);
        } else {
            self.freeze_rate = self.freeze_rate.add(PicoFixed::from_f32(0.6 / 60.0));
        }
        if self.freeze_timer > 480 {
            self.freeze_active = false;
            self.freeze_rate = PicoFixed::ONE;
        }
    }

    fn update_size(&mut self) {
        if self.player.size == PicoFixed::from_int(4) {
            return;
        }
        self.size_timer = self.size_timer.add(self.freeze_rate);
        if self.size_timer <= PicoFixed::from_int(600) {
            return;
        }
        if self.player.size < PicoFixed::from_int(4) {
            self.player.size = self
                .player
                .size
                .add(PicoFixed::from_f32(0.5).mul_fixed(self.freeze_rate));
        } else {
            self.player.size = PicoFixed::from_int(4);
        }
    }

    fn render_current_frame(&mut self) -> FrameCapture {
        let state = self.full_state();
        let mut render = crate::RenderState::new(state.transition_render_y);
        render.camera_x = state.camera_x;
        render.camera_y = state.camera_y;
        render.screen_palette[12] = state.settings.theme_background;
        render.screen_palette[1] = state.settings.theme_shadow;
        crate::snapshot::render_full_state_into(&state, &render, &mut self.screen);
        let framebuffer = self.screen.project(&render);
        FrameCapture {
            framebuffer,
            render_state: render,
        }
    }

    fn update_particles(&mut self) {
        let mut index = 0;
        while index < self.particles.len() {
            let Some(mut particle) = self.particles.get(index).copied() else {
                break;
            };
            particle.age += 1;
            if PicoFixed::from_int(particle.age as i32) > particle.max_age {
                self.particles.remove(index);
                continue;
            }
            particle.color = particle_color(particle);
            particle.x = particle.x.add(particle.dx);
            particle.y = particle.y.add(particle.dy);
            if particle.kind != 2 {
                particle.radius = particle.radius.mul_fixed(PicoFixed::from_f32(0.9));
            }
            if particle.radius < PicoFixed::ZERO && particle.kind == 0 {
                self.particles.remove(index);
                continue;
            }
            if let Some(slot) = self.particles.get_mut(index) {
                *slot = particle;
            }
            index += 1;
        }
    }

    fn update_fyou(&mut self, events: &mut Vec<FrameEvent>) {
        let in_center = self
            .player
            .x
            .add(self.player.size)
            .sub(PicoFixed::from_int(2))
            > PicoFixed::from_int(60)
            && self
                .player
                .y
                .add(self.player.size)
                .sub(PicoFixed::from_int(2))
                > PicoFixed::from_int(60)
            && self
                .player
                .x
                .sub(self.player.size)
                .add(PicoFixed::from_int(2))
                < PicoFixed::from_int(68)
            && self
                .player
                .y
                .sub(self.player.size)
                .add(PicoFixed::from_int(2))
                < PicoFixed::from_int(68);
        if !in_center {
            self.friendly_timer = 0;
            return;
        }
        self.friendly_timer += 1;
        if self.friendly_timer > 50 {
            self.friendly_timer = 0;
            self.add_corner_enemies(events);
        }
    }

    fn add_corner_enemies(&mut self, events: &mut Vec<FrameEvent>) {
        for spawn in self.spawns.clone() {
            self.enemies
                .push(EnemyState::normal(spawn.x, spawn.y, self.enemy_max_size));
            events.push(FrameEvent::EnemySpawn);
        }
    }

    fn collision_check(&mut self, events: &mut Vec<FrameEvent>) {
        if self.lifecycle.dead || !self.should_collide {
            return;
        }
        let mut index = 0;
        while index < self.enemies.len() {
            let Some(enemy) = self.enemies.get(index).copied() else {
                break;
            };
            let player = self.player;
            let collision = player.x.add(player.size).sub(PicoFixed::ONE) > enemy.x
                && player.y.add(player.size).sub(PicoFixed::ONE) > enemy.y
                && player.x.sub(player.size).add(PicoFixed::ONE) < enemy.x.add(enemy.size)
                && player.y.sub(player.size).add(PicoFixed::ONE) < enemy.y.add(enemy.size);
            let powerup_collision = player.x.add(player.size) > enemy.x.sub(PicoFixed::from_int(4))
                && player.y.add(player.size) > enemy.y.sub(PicoFixed::from_int(4))
                && player.x.sub(player.size) < enemy.x.add(PicoFixed::from_int(4))
                && player.y.sub(player.size) < enemy.y.add(PicoFixed::from_int(4));
            let collision = if enemy.personality >= 2 {
                powerup_collision
            } else {
                collision
            };
            if collision {
                events.push(FrameEvent::Collision);
                self.collide_enemy(index, enemy, events);
            } else {
                index += 1;
            }
        }
        if let Some(pattern_index) = self.active_pattern {
            let Some(pattern) = self.patterns.get(pattern_index) else {
                return;
            };
            // Static patterns kill the player once fully expanded; moving
            // patterns are lethal throughout their active rectangle, matching
            // `collisioncheck` in the cartridge.
            for rect in &pattern.rects {
                if pattern.pattern_type != 1 && rect.sh != PicoFixed::from_int(2) {
                    continue;
                }
                if self
                    .player
                    .x
                    .sub(PicoFixed::from_int(2))
                    .add(self.player.size)
                    > rect.x
                    && self
                        .player
                        .y
                        .sub(PicoFixed::from_int(2))
                        .add(self.player.size)
                        > rect.y
                    && self
                        .player
                        .x
                        .add(PicoFixed::from_int(2))
                        .sub(self.player.size)
                        < rect.x.add(rect.width).sub(PicoFixed::ONE)
                    && self
                        .player
                        .y
                        .add(PicoFixed::from_int(2))
                        .sub(self.player.size)
                        < rect.y.add(rect.height).sub(PicoFixed::ONE)
                {
                    self.die();
                    events.push(FrameEvent::Death);
                    break;
                }
            }
        }
    }

    fn player_would_die_now(&self) -> bool {
        if self.lifecycle.dead || !self.should_collide {
            return false;
        }
        for enemy in &self.enemies {
            if enemy.personality >= 2 {
                continue;
            }
            let player = self.player;
            let collision = player.x.add(player.size).sub(PicoFixed::ONE) > enemy.x
                && player.y.add(player.size).sub(PicoFixed::ONE) > enemy.y
                && player.x.sub(player.size).add(PicoFixed::ONE) < enemy.x.add(enemy.size)
                && player.y.sub(player.size).add(PicoFixed::ONE) < enemy.y.add(enemy.size);
            if collision {
                return true;
            }
        }
        if let Some(pattern_index) = self.active_pattern {
            let Some(pattern) = self.patterns.get(pattern_index) else {
                return false;
            };
            for rect in &pattern.rects {
                if pattern.pattern_type != 1 && rect.sh != PicoFixed::from_int(2) {
                    continue;
                }
                if self
                    .player
                    .x
                    .sub(PicoFixed::from_int(2))
                    .add(self.player.size)
                    > rect.x
                    && self
                        .player
                        .y
                        .sub(PicoFixed::from_int(2))
                        .add(self.player.size)
                        > rect.y
                    && self
                        .player
                        .x
                        .add(PicoFixed::from_int(2))
                        .sub(self.player.size)
                        < rect.x.add(rect.width).sub(PicoFixed::ONE)
                    && self
                        .player
                        .y
                        .add(PicoFixed::from_int(2))
                        .sub(self.player.size)
                        < rect.y.add(rect.height).sub(PicoFixed::ONE)
                {
                    return true;
                }
            }
        }
        false
    }

    fn collide_enemy(&mut self, index: usize, enemy: EnemyState, events: &mut Vec<FrameEvent>) {
        self.shatter(enemy.x, enemy.y);
        self.shake = self.shake.add(PicoFixed::from_f32(0.07));
        match enemy.personality {
            personality if personality < 2 => {
                self.remove_enemy_at(index);
                self.die();
                events.push(FrameEvent::Death);
            }
            2 => {
                self.emit_sfx(60, None);
                self.explode_powerup_enemy(index);
            }
            3 => {
                self.remove_enemy_at(index);
                self.freeze_active = true;
                self.freeze_timer = 0;
                self.emit_sfx(62, None);
                self.score = self.score.add(PicoFixed::ONE);
                self.apply_difficulty(false);
            }
            4 => {
                self.remove_enemy_at(index);
                self.player.size = PicoFixed::from_int(2);
                self.size_timer = PicoFixed::ZERO;
                self.emit_sfx(61, None);
                self.score = self.score.add(PicoFixed::ONE);
                self.apply_difficulty(false);
            }
            _ => {
                self.remove_enemy_at(index);
            }
        }
    }

    fn explode_powerup_enemy(&mut self, index: usize) {
        if index >= self.enemies.len() {
            return;
        }
        // PICO-8's `all(enemies)` iterator observes the source list mutation:
        // deleting the current entry advances the next entry into its slot,
        // while `kamikaze` appends a new -1 entry that is visited later.
        let mut cursor = 0;
        while cursor < self.enemies.len() {
            let Some(other) = self.enemies.get(cursor).copied() else {
                break;
            };
            if other.personality == 1 {
                self.add_kamikaze(other);
            } else if other.personality != -1 {
                self.shatter(other.x, other.y);
            }
            if other.personality != -1 {
                self.remove_enemy_at(cursor);
                self.shattered = self.shattered.saturating_add(1);
            } else {
                cursor += 1;
            }
            self.score = self.score.add(PicoFixed::ONE);
            self.apply_difficulty(false);
        }
    }

    fn remove_enemy_at(&mut self, index: usize) -> Option<EnemyState> {
        if index < self.enemies.len() {
            Some(self.enemies.remove(index))
        } else {
            None
        }
    }

    fn die(&mut self) {
        self.emit_sfx(62, None);
        self.lifecycle.mark_dead();
        self.shake = self.shake.add(PicoFixed::from_f32(0.07));
        let slot = self.current_highscore_slot();
        let previous = self
            .highscores
            .get(slot)
            .copied()
            .unwrap_or(PicoFixed::ZERO);
        if self.score > previous {
            if let Some(highscore) = self.highscores.get_mut(slot) {
                *highscore = self.score.ceil();
            }
            self.new_highscore = true;
        }
        self.emit_music(22);
    }

    fn current_highscore_slot(&self) -> usize {
        let difficulty = self.settings.difficulty;
        let patterns = self.settings.patterns_enabled;
        let powerups = self.settings.powerups_enabled;
        (0..12)
            .find(|index| {
                let category = index / 4 + 1;
                let slot = index % 4;
                category == usize::from(difficulty)
                    && patterns == (slot == 0 || slot == 1)
                    && powerups == (slot == 0 || slot == 3)
            })
            .unwrap_or(0)
    }

    fn update_player(&mut self) {
        self.player.vx = self.player.vx.mul_fixed(PLAYER_FRICTION);
        self.player.vy = self.player.vy.mul_fixed(PLAYER_FRICTION);
        if self.input.btn(Button::Left) {
            self.player.vx = self.player.vx.sub(PLAYER_SPEED);
        }
        if self.input.btn(Button::Right) {
            self.player.vx = self.player.vx.add(PLAYER_SPEED);
        }
        if self.input.btn(Button::Up) {
            self.player.vy = self.player.vy.sub(PLAYER_SPEED);
        }
        if self.input.btn(Button::Down) {
            self.player.vy = self.player.vy.add(PLAYER_SPEED);
        }
        self.player.x = pico_mid(PLAYER_MIN, PLAYER_MAX, self.player.x.add(self.player.vx));
        self.player.y = pico_mid(PLAYER_MIN, PLAYER_MAX, self.player.y.add(self.player.vy));
    }

    fn spawn_enemies(&mut self, events: &mut Vec<FrameEvent>) {
        self.enemy_timer = self.enemy_timer.add(PicoFixed::ONE);
        let spawn_multiplier = self
            .active_pattern
            .and_then(|index| self.patterns.get(index))
            .map_or(PicoFixed::ONE, |pattern| {
                if pattern.special == PicoFixed::from_int(2)
                    || pattern.special == PicoFixed::from_int(3)
                {
                    PicoFixed::from_f32(1.75)
                } else {
                    PicoFixed::ONE
                }
            });
        let threshold = self
            .enemy_est
            .mul_fixed(SPAWN_INTERVAL)
            .mul_fixed(spawn_multiplier)
            .mul_fixed(PicoFixed::from_int(2).sub(self.freeze_rate));
        if self.enemy_timer <= threshold || self.spawns.is_empty() {
            return;
        }
        self.enemy_timer = PicoFixed::ZERO;
        let size_roll = self.rng.rnd(PicoFixed::from_int(100));
        let (x, y) = self.nearest_corner();
        let max_size = enemy_max_size_from_roll(size_roll);
        self.enemy_max_size = max_size;
        let personality = self.random_personality();
        self.enemies.push(EnemyState {
            personality,
            ..EnemyState::normal(x, y, max_size)
        });
        events.push(FrameEvent::EnemySpawn);
    }

    fn random_personality(&mut self) -> i8 {
        let total = self
            .enemy_stats
            .iter()
            .map(|value| value.floor().raw().max(0) >> 16)
            .sum::<i32>();
        if total <= 0 {
            return 0;
        }
        let roll = (self.rng.rnd(PicoFixed::from_int(total)).floor().raw() >> 16).max(0);
        let mut cursor = 0_i32;
        for (index, value) in self.enemy_stats.iter().enumerate() {
            cursor += value.floor().raw().max(0) >> 16;
            if roll < cursor {
                return index as i8;
            }
        }
        0
    }

    fn nearest_corner(&self) -> (PicoFixed, PicoFixed) {
        let Some(first) = self.spawns.first().copied() else {
            return (SPAWN_EDGE, SPAWN_EDGE);
        };
        let mut best = (first.x, first.y);
        let mut best_distance = f64::INFINITY;
        for spawn in &self.spawns {
            let dx = spawn.x.to_double() - self.player.x.to_double();
            let dy = spawn.y.to_double() - self.player.y.to_double();
            let distance = dx * dx + dy * dy;
            if distance < best_distance {
                best = (spawn.x, spawn.y);
                best_distance = distance;
            }
        }
        best
    }

    fn update_enemies(&mut self) {
        let player = self.player;
        let mut index = 0;
        while index < self.enemies.len() {
            let Some(original) = self.enemies.get(index).copied() else {
                break;
            };
            let x = original.x;
            let y = original.y;
            let mut vx = original.vx.mul_fixed(ENEMY_FRICTION);
            let mut vy = original.vy.mul_fixed(ENEMY_FRICTION);
            // `updateenemies` keeps the pre-growth `s` local for pattern and
            // crush checks, while later overlap/edge code reads `_e.s`.
            let source_size = original.size;
            let mut size = source_size;
            let mut next_x = original.x;
            let mut next_y = original.y;
            let mut isizing = original.isizing;
            let mut life = original.life;

            if size < original.max_size && original.personality != -1 {
                size = size.add(self.freeze_rate);
            }
            if size < original.max_size && original.personality == -1 && isizing {
                size = size.add(self.freeze_rate);
                next_x = next_x.sub(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
                next_y = next_y.sub(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
            }
            // The cartridge tests the pre-growth local `s` here.  Testing the
            // mutated size would flip an expanding kamikaze to shrinking on
            // the exact frame it reaches max size, moving it and reducing it
            // one frame too early.
            if source_size >= original.max_size && original.personality == -1 {
                isizing = false;
            }
            if original.personality == -1 && !isizing {
                size = size.sub(self.freeze_rate);
                next_x = next_x.add(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
                next_y = next_y.add(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
            }
            if original.personality >= 2 {
                size = PicoFixed::from_int(4);
            }
            if let Some(current_life) = life {
                let remaining = current_life.sub(self.freeze_rate);
                life = Some(remaining);
                if remaining <= PicoFixed::ZERO {
                    self.remove_enemy_at(index);
                    continue;
                }
            }

            let inside = original.inside || enemy_is_inside(x);
            let mut current = original;
            current.x = next_x;
            current.y = next_y;
            current.size = size;
            current.inside = inside;
            current.isizing = isizing;
            current.life = life;

            if original.is_dying {
                self.enemies.remove(index);
                self.resolve_enemy_death(current, x, y, source_size);
                continue;
            }
            if let Some(slot) = self.enemies.get_mut(index) {
                *slot = current;
            } else {
                break;
            }

            if original.personality >= 0 {
                if x < player.x {
                    vx = vx.add(ENEMY_ACCELERATION);
                } else if x > player.x {
                    vx = vx.sub(ENEMY_ACCELERATION);
                }
                if y < player.y {
                    vy = vy.add(ENEMY_ACCELERATION);
                } else if y > player.y {
                    vy = vy.sub(ENEMY_ACCELERATION);
                }
            }

            if original.personality != -1 {
                self.apply_pattern_enemy_collision(index, &mut vx, &mut vy, x, y, source_size);
            }
            self.apply_pattern_crush(index, x, y, source_size);

            let Some(current_enemy) = self.enemies.get(index).copied() else {
                break;
            };
            let current_offset = if current_enemy.personality >= 2 {
                PicoFixed::from_int(4)
            } else {
                PicoFixed::ZERO
            };
            for other_index in 0..self.enemies.len() {
                if other_index == index {
                    continue;
                }
                let Some(other) = self.enemies.get(other_index).copied() else {
                    continue;
                };
                let other_offset = if other.personality >= 2 {
                    PicoFixed::from_int(4)
                } else {
                    PicoFixed::ZERO
                };
                if self.enemy_should_collide
                    && current_enemy.inside
                    && other.inside
                    && current_enemy.x.add(current_enemy.size) > other.x.sub(other_offset)
                    && current_enemy.y.add(current_enemy.size) > other.y.sub(other_offset)
                    && current_enemy.x.sub(current_offset) < other.x.add(other.size)
                    && current_enemy.y.sub(current_offset) < other.y.add(other.size)
                {
                    if let Some(current_enemy) = self.enemies.get_mut(index) {
                        current_enemy.is_dying = true;
                    }
                    if let Some(other_enemy) = self.enemies.get_mut(other_index) {
                        other_enemy.is_dying = true;
                    }
                }
            }

            if current_enemy.inside && current_enemy.personality != -1 {
                if current_enemy.y >= PicoFixed::from_int(129).sub(current_enemy.size) {
                    vy = fixed_abs(vy).neg();
                } else if current_enemy.y <= PicoFixed::from_int(-2) {
                    vy = fixed_abs(vy);
                }
                if current_enemy.x >= PicoFixed::from_int(129).sub(current_enemy.size) {
                    vx = fixed_abs(vx).neg();
                } else if current_enemy.x <= PicoFixed::from_int(-2) {
                    vx = fixed_abs(vx);
                }
            }

            let Some(mut updated) = self.enemies.get(index).copied() else {
                break;
            };
            updated.speed = update_enemy_speed(updated, player);
            updated.x = current_enemy.x.add(
                vx.mul_fixed(self.speed)
                    .mul_fixed(self.freeze_rate)
                    .mul_fixed(updated.speed),
            );
            updated.y = current_enemy.y.add(
                vy.mul_fixed(self.speed)
                    .mul_fixed(self.freeze_rate)
                    .mul_fixed(updated.speed),
            );
            updated.vx = vx;
            updated.vy = vy;
            if let Some(slot) = self.enemies.get_mut(index) {
                *slot = updated;
            }
            index += 1;
        }
    }

    fn apply_pattern_enemy_collision(
        &self,
        _index: usize,
        vx: &mut PicoFixed,
        vy: &mut PicoFixed,
        x: PicoFixed,
        y: PicoFixed,
        size: PicoFixed,
    ) {
        let Some(pattern_index) = self.active_pattern else {
            return;
        };
        let Some(pattern) = self.patterns.get(pattern_index) else {
            return;
        };
        for rect in &pattern.rects {
            if rect.sh != PicoFixed::from_int(2) {
                continue;
            }
            let powerup = self
                .enemies
                .get(_index)
                .is_some_and(|enemy| enemy.personality > 1);
            let powerup_offset = if powerup {
                PicoFixed::from_int(3)
            } else {
                PicoFixed::ZERO
            };
            if x.add(*vx).sub(powerup_offset) < rect.x.add(rect.width)
                && x.add(size) > rect.x.add(rect.dx)
                && y.add(size) > rect.y.add(rect.dy)
                && y.add(*vy).sub(powerup_offset) < rect.y.add(rect.height)
            {
                let minimum = if pattern.pattern_type == 0 {
                    self.bounce_cap_static
                } else {
                    self.bounce_cap_moving
                };
                if x.add(*vx) < rect.x {
                    *vx = fixed_abs(*vx).max(minimum).neg();
                } else if x.add(*vx).add(size) > rect.x.add(rect.width) {
                    *vx = fixed_abs(*vx).max(minimum);
                }
                if y.add(*vy) < rect.y {
                    *vy = fixed_abs(*vy).max(minimum).neg();
                } else if y.add(*vy).add(size) > rect.y.add(rect.height) {
                    *vy = fixed_abs(*vy).max(minimum);
                }
                let mut cap = if pattern.pattern_type == 0 {
                    self.bounce_cap_static
                } else {
                    self.bounce_cap_moving
                };
                if pattern.bounce_cap {
                    cap = cap.sub(PicoFixed::from_f32(0.15));
                }
                *vx = with_sign(fixed_abs(*vx).min(cap), *vx);
                *vy = with_sign(fixed_abs(*vy).min(cap), *vy);
            }
        }
    }

    fn apply_pattern_crush(&mut self, index: usize, x: PicoFixed, y: PicoFixed, size: PicoFixed) {
        let Some(pattern_index) = self.active_pattern else {
            return;
        };
        let Some(pattern) = self.patterns.get(pattern_index) else {
            return;
        };
        let Some(enemy) = self.enemies.get(index) else {
            return;
        };
        if enemy.personality == -1 || !enemy.inside {
            return;
        }
        let mut dying = false;
        for rect in &pattern.rects {
            if rect.sh != PicoFixed::from_int(2) {
                continue;
            }
            let mut should_check = false;
            let mut add_x = PicoFixed::ZERO;
            let mut add_y = PicoFixed::ZERO;
            if x.add(size) > PicoFixed::from_int(127) {
                should_check = true;
                add_x = PicoFixed::from_int(-4);
            } else if x.sub(size) < PicoFixed::ZERO {
                should_check = true;
                add_x = PicoFixed::from_int(4);
            }
            if y.add(size) > PicoFixed::from_int(127) {
                should_check = true;
                add_y = PicoFixed::from_int(-4);
            } else if y.sub(size) < PicoFixed::ZERO {
                should_check = true;
                add_y = PicoFixed::from_int(4);
            }
            if should_check
                && x.add(add_x) > rect.x
                && y.add(add_y) > rect.y
                && x.add(add_x) < rect.x.add(rect.width)
                && y.add(add_y) < rect.y.add(rect.height)
            {
                dying = true;
                break;
            }
        }
        if dying {
            if let Some(enemy) = self.enemies.get_mut(index) {
                enemy.is_dying = true;
            }
        }
    }

    fn resolve_enemy_death(
        &mut self,
        enemy: EnemyState,
        shatter_x: PicoFixed,
        shatter_y: PicoFixed,
        source_size: PicoFixed,
    ) {
        if enemy.personality == 1 {
            // The cartridge's `kamikaze(_e)` reads `_e.x` and `_e.s`, while
            // the update locals still contain the pre-growth values. `enemy`
            // has already received this frame's growth and movement, so use
            // the source position/size explicitly here.
            self.add_kamikaze_at(shatter_x, shatter_y, source_size);
        } else {
            self.shatter(shatter_x, shatter_y);
        }
        self.emit_sfx(63, None);
        self.score = self.score.add(SCORE_PER_SHATTER);
        self.shattered = self.shattered.saturating_add(1);
        self.shake = self.shake.add(PicoFixed::from_f32(0.07));
        self.apply_difficulty(true);
    }

    fn shatter(&mut self, x: PicoFixed, y: PicoFixed) {
        for _ in 0..11 {
            let angle = self.rng.rnd(PicoFixed::ONE);
            self.particles.push(ParticleState {
                x,
                y,
                dx: pico_sine(angle),
                dy: pico_cosine(angle),
                radius: PicoFixed::ZERO,
                kind: 1,
                max_age: PicoFixed::from_int(60),
                age: 0,
                color: 0,
                colors: [7, 0, 0],
                color_count: 1,
            });
        }
    }

    fn add_kamikaze(&mut self, enemy: EnemyState) {
        self.add_kamikaze_at(enemy.x, enemy.y, enemy.size);
    }

    fn add_kamikaze_at(&mut self, x: PicoFixed, y: PicoFixed, size: PicoFixed) {
        self.enemies.push(EnemyState {
            x: x.add(size.mul_fixed(PicoFixed::from_f32(0.5))),
            y: y.add(size.mul_fixed(PicoFixed::from_f32(0.5))),
            vx: PicoFixed::ZERO,
            vy: PicoFixed::ZERO,
            size: PicoFixed::ZERO,
            max_size: PicoFixed::from_int(30),
            personality: -1,
            speed: PicoFixed::ONE,
            inside: false,
            is_dying: false,
            isizing: true,
            life: Some(PicoFixed::from_int(60)),
        });
    }

    fn apply_difficulty(&mut self, half: bool) {
        let index = difficulty_index(self.settings.difficulty);
        let Some(curve) = DIFFICULTY_CURVES.get(index).copied() else {
            return;
        };
        let speed_step = difficulty_increment(curve.speed_increment, half);
        let enemy_step = difficulty_increment(curve.enemy_increment, half);
        self.speed = pico_lerp(self.speed, DIFFICULTY_SPEED_TARGET, speed_step);
        self.enemy_est = pico_lerp(self.enemy_est, DIFFICULTY_EST_TARGET, enemy_step);
        let static_step = difficulty_increment(curve.static_increment, half);
        let moving_step = difficulty_increment(curve.moving_increment, half);
        self.bounce_cap_static =
            pico_lerp(self.bounce_cap_static, curve.static_target, static_step);
        self.bounce_cap_moving =
            pico_lerp(self.bounce_cap_moving, curve.moving_target, moving_step);
    }

    fn update_pattern_schedule(&mut self) {
        if !self.patterns_enabled {
            return;
        }
        if self.active_pattern.is_some() {
            self.update_active_pattern();
            return;
        }
        if self.freeze_active {
            return;
        }
        self.pattern_timer += 1;
        if self.pattern_timer < self.pattern_delay_frames {
            return;
        }
        self.pattern_timer = 0;
        self.pattern_delay_frames = ACTIVE_PATTERN_DELAY_FRAMES;
        let mut weighted = Vec::new();
        for (index, pattern) in self.patterns.iter().enumerate() {
            if self.score >= pattern.mins && self.score <= pattern.maxs {
                let repetitions = (pattern.probability.floor().raw() >> 16).max(0);
                for _ in 0..repetitions {
                    weighted.push(index);
                }
            }
        }
        let selected_position = self
            .rng
            .rnd(PicoFixed::from_int(weighted.len() as i32))
            .floor()
            .raw()
            / PicoFixed::ONE.raw();
        let Some(&selected) =
            weighted.get(usize::try_from(selected_position).unwrap_or(usize::MAX))
        else {
            return;
        };
        self.active_pattern = Some(selected);
        self.pattern_active = true;
        let variants = self
            .patterns
            .get(selected)
            .map(|pattern| pattern.variants.clone())
            .unwrap_or_default();
        if let Some(pattern) = self.patterns.get_mut(selected) {
            pattern.counter += 1;
            pattern.probability = PicoFixed::ONE;
            if pattern.spawn_enabled {
                self.spawns.clear();
            }
        }
        for variant in variants {
            if let Some(variant_pattern) = self.patterns.get_mut(usize::from(variant - 1)) {
                variant_pattern.probability = PicoFixed::ONE;
            }
        }
    }

    fn update_active_pattern(&mut self) {
        let Some(pattern_index) = self.active_pattern else {
            return;
        };
        let increment = PicoFixed::from_f32(0.02).mul_fixed(self.freeze_rate);
        let Some(pattern_snapshot) = self.patterns.get(pattern_index).cloned() else {
            self.active_pattern = None;
            self.pattern_active = false;
            return;
        };
        if let Some(pattern) = self.patterns.get_mut(pattern_index) {
            pattern.timer = pattern.timer.add(self.freeze_rate);
        }

        let mut finished_count = 0_usize;
        for rect_index in 0..pattern_snapshot.rects.len() {
            let Some(rect_snapshot) = pattern_snapshot.rects.get(rect_index).cloned() else {
                continue;
            };
            if pattern_snapshot.pattern_type == 1 && rect_snapshot.sh < PicoFixed::from_int(2) {
                let warnings = moving_pattern_warnings(&rect_snapshot);
                if let Some(rect) = self
                    .patterns
                    .get_mut(pattern_index)
                    .and_then(|pattern| pattern.rects.get_mut(rect_index))
                {
                    rect.warnings = warnings;
                }
            }

            if rect_snapshot.sh == PicoFixed::from_int(2) && !rect_snapshot.collision_done {
                if let Some(rect) = self
                    .patterns
                    .get_mut(pattern_index)
                    .and_then(|pattern| pattern.rects.get_mut(rect_index))
                {
                    rect.collision_done = true;
                }
                for enemy in &mut self.enemies {
                    if enemy.personality != -1
                        && enemy.x.add(enemy.size) > rect_snapshot.x
                        && enemy.y.add(enemy.size) > rect_snapshot.y
                        && enemy.x < rect_snapshot.x.add(rect_snapshot.width)
                        && enemy.y < rect_snapshot.y.add(rect_snapshot.height)
                    {
                        enemy.is_dying = true;
                    }
                }
            }

            let mut next_x = rect_snapshot.x;
            let mut next_y = rect_snapshot.y;
            let mut next_width = rect_snapshot.width;
            let mut next_height = rect_snapshot.height;
            let mut next_target_index = rect_snapshot.target_index;
            let mut next_wait = rect_snapshot.wait;
            let mut next_shown = rect_snapshot.shown;
            let mut next_sh = rect_snapshot.sh;
            let mut finished = rect_snapshot.finished;
            if rect_snapshot.sh >= PicoFixed::from_int(2) && rect_snapshot.shown {
                if let Some(target) = rect_snapshot.targets.get(rect_snapshot.target_index) {
                    match target {
                        crate::PatternTarget::Move {
                            x,
                            y,
                            width,
                            height,
                        } => {
                            if pattern_snapshot.smooth {
                                next_x = next_x.add(
                                    x.sub(next_x)
                                        .div_fixed(rect_snapshot.speed)
                                        .unwrap_or(PicoFixed::ZERO)
                                        .mul_fixed(self.freeze_rate),
                                );
                                next_y = next_y.add(
                                    y.sub(next_y)
                                        .div_fixed(rect_snapshot.speed)
                                        .unwrap_or(PicoFixed::ZERO)
                                        .mul_fixed(self.freeze_rate),
                                );
                                next_width = next_width.add(
                                    width
                                        .sub(next_width)
                                        .div_fixed(rect_snapshot.speed)
                                        .unwrap_or(PicoFixed::ZERO)
                                        .mul_fixed(self.freeze_rate),
                                );
                                next_height = next_height.add(
                                    height
                                        .sub(next_height)
                                        .div_fixed(rect_snapshot.speed)
                                        .unwrap_or(PicoFixed::ZERO)
                                        .mul_fixed(self.freeze_rate),
                                );
                            } else {
                                next_x =
                                    move_toward(next_x, *x, rect_snapshot.speed, self.freeze_rate);
                                next_y =
                                    move_toward(next_y, *y, rect_snapshot.speed, self.freeze_rate);
                                next_width = move_toward(
                                    next_width,
                                    *width,
                                    rect_snapshot.speed,
                                    self.freeze_rate,
                                );
                                next_height = move_toward(
                                    next_height,
                                    *height,
                                    rect_snapshot.speed,
                                    self.freeze_rate,
                                );
                            }
                            if next_x.round() == *x
                                && next_y.round() == *y
                                && next_width.round() == *width
                                && next_height.round() == *height
                            {
                                next_target_index += 1;
                            }
                        }
                        crate::PatternTarget::Wait(seconds) => {
                            next_wait = next_wait.add(self.freeze_rate);
                            if next_wait >= seconds.mul_fixed(PicoFixed::from_int(60)) {
                                next_wait = PicoFixed::ZERO;
                                next_target_index += 1;
                            }
                        }
                        crate::PatternTarget::SetFyou(value) => {
                            self.friendly_enabled = *value;
                            next_target_index += 1;
                        }
                        crate::PatternTarget::SetSpawns(points) => {
                            self.spawns = points.clone();
                            next_target_index += 1;
                        }
                    }
                }
            } else if rect_snapshot.shown {
                next_sh = next_sh.add(increment);
                if next_sh > PicoFixed::from_f32(1.99) {
                    next_sh = PicoFixed::from_int(2);
                    self.spawns = initial_spawns();
                }
            } else {
                next_sh = next_sh.sub(increment);
                if pattern_snapshot.spawn_enabled {
                    self.spawns.clear();
                }
                if next_sh < PicoFixed::from_f32(0.05) {
                    next_sh = PicoFixed::ZERO;
                }
            }

            if next_target_index >= rect_snapshot.targets.len() {
                if pattern_snapshot.pattern_type == 1 {
                    finished = true;
                } else {
                    next_shown = false;
                }
            }
            if !next_shown && next_sh <= PicoFixed::ZERO {
                finished = true;
            }
            if finished {
                finished_count += 1;
            }
            if let Some(rect) = self
                .patterns
                .get_mut(pattern_index)
                .and_then(|pattern| pattern.rects.get_mut(rect_index))
            {
                rect.x = next_x;
                rect.y = next_y;
                rect.width = next_width;
                rect.height = next_height;
                rect.target_index = next_target_index;
                rect.wait = next_wait;
                rect.shown = next_shown;
                rect.sh = next_sh;
                rect.finished = finished;
            }
        }
        if finished_count >= pattern_snapshot.rects.len() {
            let probabilities = self
                .patterns
                .iter()
                .map(|pattern| pattern.probability)
                .collect::<Vec<_>>();
            let counters = self
                .patterns
                .iter()
                .map(|pattern| pattern.counter)
                .collect::<Vec<_>>();
            self.patterns = crate::patterns::init_patterns(&mut self.rng);
            for (index, pattern) in self.patterns.iter_mut().enumerate() {
                if let Some(probability) = probabilities.get(index).copied() {
                    pattern.probability = probability;
                }
                if let Some(counter) = counters.get(index).copied() {
                    pattern.counter = counter;
                }
            }
            self.active_pattern = None;
            self.pattern_active = false;
            self.friendly_enabled = true;
            self.spawns = initial_spawns();
        }
    }

    fn apply_draw_side_effects(&mut self, mode_before: Mode) {
        let renders_game = match mode_before {
            Mode::Game => {
                self.lifecycle.mode == Mode::Game
                    || (self.lifecycle.mode == Mode::TransitionToSettings
                        && self.transition_render_y > 0
                        && matches!(self.transition_from, Mode::Game | Mode::Terminal))
            }
            Mode::TransitionToGame => self.transition_render_y > 0,
            Mode::TransitionToSettings => {
                self.transition_render_y > 0
                    && matches!(self.transition_from, Mode::Game | Mode::Terminal)
            }
            _ => false,
        };
        if self.lifecycle.dead || !renders_game {
            return;
        }
        for enemy in &self.enemies {
            if enemy.personality >= 2 {
                let angle = self.rng.rnd(PicoFixed::ONE);
                let x = enemy.x.floor();
                let y = enemy.y.floor();
                let offset_x = pico_sine(angle).mul_fixed(PicoFixed::from_f32(2.4));
                let offset_y = pico_cosine(angle).mul_fixed(PicoFixed::from_f32(2.4));
                let max_age = PicoFixed::from_int(20).add(self.rng.rnd(PicoFixed::from_int(15)));
                let colors = match enemy.personality {
                    2 => [8, 9, 10],
                    3 => [1, 13, 7],
                    _ => [7, 0, 0],
                };
                let color_count = if enemy.personality == 4 { 1 } else { 3 };
                self.particles.push(ParticleState {
                    x: x.add(offset_x),
                    y: y.add(offset_y),
                    dx: PicoFixed::ZERO,
                    dy: PicoFixed::ZERO,
                    radius: PicoFixed::ZERO,
                    kind: 1,
                    max_age,
                    age: 0,
                    color: 0,
                    colors,
                    color_count,
                });
            }
        }
        self.add_player_trail();
    }

    fn add_player_trail(&mut self) {
        self.particles.push(ParticleState::player_trail(
            self.player.x,
            self.player.y,
            self.player.size,
        ));
    }

    fn update_camera(&mut self) {
        let shake_limit = PicoFixed::from_f32(0.1);
        self.shake = self.shake.min(shake_limit);
        let horizontal = PicoFixed::from_int(16).sub(self.rng.rnd(PicoFixed::from_int(32)));
        let vertical = PicoFixed::from_int(16).sub(self.rng.rnd(PicoFixed::from_int(32)));
        self.camera_x = horizontal.mul_fixed(self.shake);
        self.camera_y = vertical.mul_fixed(self.shake);
        self.shake = self.shake.mul_fixed(PicoFixed::from_f32(0.95));
        if self.shake < PicoFixed::from_f32(0.05) {
            self.shake = PicoFixed::ZERO;
        }
    }

    fn result(&self, reward: PicoFixed, events: Vec<FrameEvent>) -> FrameResult {
        let snapshot = self.snapshot();
        self.result_with_snapshot(reward, events, Vec::new(), snapshot)
    }

    fn emit_music(&mut self, track: u8) {
        self.frame_audio.push(AudioEvent::Music { track });
    }

    fn emit_sfx(&mut self, id: u8, channel: Option<i8>) {
        self.frame_audio.push(AudioEvent::Sfx { id, channel });
    }

    fn result_with_snapshot(
        &self,
        reward: PicoFixed,
        events: Vec<FrameEvent>,
        audio: Vec<AudioEvent>,
        snapshot: Snapshot,
    ) -> FrameResult {
        FrameResult {
            frame: self.lifecycle.frame,
            mode: self.lifecycle.mode,
            input_mask: self.input.current_mask(),
            previous_input_mask: self.input.previous_mask(),
            game_ready: self.lifecycle.game_ready,
            started: self.lifecycle.started,
            dead: self.lifecycle.dead,
            done: self.lifecycle.dead,
            reward,
            events,
            audio,
            snapshot,
        }
    }
}

fn particle_color(particle: ParticleState) -> u8 {
    if particle.color_count <= 1 {
        return particle.colors[0];
    }
    let age = PicoFixed::from_int(particle.age as i32);
    let fraction = age.div_fixed(particle.max_age).unwrap_or(PicoFixed::ZERO);
    let scaled = fraction.mul_fixed(PicoFixed::from_int(i32::from(particle.color_count)));
    let index = (scaled.floor().raw() >> 16)
        .max(0)
        .min(i32::from(particle.color_count - 1));
    particle
        .colors
        .get(index as usize)
        .copied()
        .or_else(|| particle.colors.first().copied())
        .unwrap_or(0)
}

fn pico_sine(angle: PicoFixed) -> PicoFixed {
    let radians = f64::from(angle.to_f32()) * std::f64::consts::TAU;
    PicoFixed::from_f32(-(radians.sin() as f32))
}

fn pico_cosine(angle: PicoFixed) -> PicoFixed {
    let radians = f64::from(angle.to_f32()) * std::f64::consts::TAU;
    PicoFixed::from_f32(radians.cos() as f32)
}

fn enemy_max_size_from_roll(roll: PicoFixed) -> PicoFixed {
    if roll <= PicoFixed::from_int(20) {
        PicoFixed::from_int(3)
    } else if roll <= PicoFixed::from_int(70) {
        PicoFixed::from_int(4)
    } else if roll <= PicoFixed::from_int(90) {
        PicoFixed::from_int(5)
    } else {
        PicoFixed::from_int(6)
    }
}

#[cfg(test)]
fn normal_personality_from_roll(roll: PicoFixed) -> i8 {
    let index = roll.floor().raw() >> 16;
    if index < 76 {
        0
    } else if index < 93 {
        1
    } else if index < 95 {
        2
    } else {
        4
    }
}

fn update_enemy_speed(enemy: EnemyState, player: PlayerState) -> PicoFixed {
    if enemy.personality != 1 {
        return enemy.speed;
    }
    let dx = enemy.x.sub(player.x);
    let dy = enemy.y.sub(player.y);
    let distance_squared = dx.mul_fixed(dx).add(dy.mul_fixed(dy));
    if distance_squared <= KAMIKAZE_RADIUS_SQUARED {
        enemy.speed.sub(ENEMY_SPEED_STEP)
    } else if enemy.speed < PicoFixed::ONE {
        enemy.speed.add(ENEMY_SPEED_STEP).min(PicoFixed::ONE)
    } else {
        PicoFixed::ONE
    }
}

fn enemy_is_inside(x: PicoFixed) -> bool {
    x >= PicoFixed::from_int(1) && x <= PicoFixed::from_int(128)
}

fn fixed_abs(value: PicoFixed) -> PicoFixed {
    if value < PicoFixed::ZERO {
        value.neg()
    } else {
        value
    }
}

fn with_sign(magnitude: PicoFixed, signed_value: PicoFixed) -> PicoFixed {
    if signed_value < PicoFixed::ZERO {
        magnitude.neg()
    } else {
        magnitude
    }
}

fn pico_lerp(position: PicoFixed, target: PicoFixed, percentage: PicoFixed) -> PicoFixed {
    PicoFixed::ONE
        .sub(percentage)
        .mul_fixed(position)
        .add(percentage.mul_fixed(target))
}

fn move_toward(
    position: PicoFixed,
    target: PicoFixed,
    speed: PicoFixed,
    freeze_rate: PicoFixed,
) -> PicoFixed {
    let step = speed.mul_fixed(freeze_rate);
    if position > target {
        position.sub(step)
    } else if position < target {
        position.add(step)
    } else {
        position
    }
}

fn moving_pattern_warnings(rect: &crate::PatternRect) -> Vec<crate::WarningLine> {
    let x = rect.x;
    let y = rect.y;
    let width = rect.width;
    let height = rect.height;
    let offset = PicoFixed::from_int(6);
    let half_width = width
        .div_fixed(PicoFixed::from_int(2))
        .unwrap_or(PicoFixed::ZERO);
    let half_height = height
        .div_fixed(PicoFixed::from_int(2))
        .unwrap_or(PicoFixed::ZERO);
    let center_x = x.add(half_width);
    let center_y = y.add(half_height);
    let sh = rect.sh.min(PicoFixed::ONE);
    let mut warnings = Vec::new();
    if rect.dx > PicoFixed::ZERO {
        warnings = vec![
            crate::WarningLine {
                x0: x.add(width).add(offset),
                y0: center_y,
                x1: x.add(width).add(offset),
                y1: pico_lerp(center_y, y.add(PicoFixed::ONE), sh),
            },
            crate::WarningLine {
                x0: x.add(width).add(offset),
                y0: center_y,
                x1: x.add(width).add(offset),
                y1: pico_lerp(center_y, y.add(height).sub(PicoFixed::from_int(2)), sh),
            },
        ];
    } else if rect.dx < PicoFixed::ZERO {
        warnings = vec![
            crate::WarningLine {
                x0: x.sub(offset),
                y0: center_y,
                x1: x.sub(offset),
                y1: pico_lerp(center_y, y.add(PicoFixed::ONE), sh),
            },
            crate::WarningLine {
                x0: x.sub(offset),
                y0: center_y,
                x1: x.sub(offset),
                y1: pico_lerp(center_y, y.add(height).sub(PicoFixed::from_int(2)), sh),
            },
        ];
    }
    if rect.dy > PicoFixed::ZERO {
        warnings = vec![
            crate::WarningLine {
                x0: center_x,
                y0: y.add(height).add(offset),
                x1: pico_lerp(center_x, x.add(PicoFixed::ONE), sh),
                y1: y.add(height).add(offset),
            },
            crate::WarningLine {
                x0: center_x,
                y0: y.add(height).add(offset),
                x1: pico_lerp(center_x, x.add(width).sub(PicoFixed::from_int(2)), sh),
                y1: y.add(height).add(offset),
            },
        ];
    } else if rect.dy < PicoFixed::ZERO {
        warnings = vec![
            crate::WarningLine {
                x0: center_x,
                y0: y.sub(offset),
                x1: pico_lerp(center_x, x.add(PicoFixed::ONE), sh),
                y1: y.sub(offset),
            },
            crate::WarningLine {
                x0: center_x,
                y0: y.sub(offset),
                x1: pico_lerp(center_x, x.add(width).sub(PicoFixed::from_int(2)), sh),
                y1: y.sub(offset),
            },
        ];
    }
    warnings
}

fn initial_spawns() -> Vec<SpawnPoint> {
    vec![
        SpawnPoint {
            x: SPAWN_EDGE,
            y: SPAWN_EDGE,
        },
        SpawnPoint {
            x: SPAWN_EDGE_FAR,
            y: SPAWN_EDGE,
        },
        SpawnPoint {
            x: SPAWN_EDGE,
            y: SPAWN_EDGE_FAR,
        },
        SpawnPoint {
            x: SPAWN_EDGE_FAR,
            y: SPAWN_EDGE_FAR,
        },
    ]
}

fn difficulty_index(difficulty: u8) -> usize {
    usize::from(difficulty.clamp(1, 3) - 1)
}

// These are the cartridge's `difspd`, `difest`, `incbs`, `tarbs`, `incbm`, and
// `tarbm` tables in the accepted Q16.16 boundary. Keeping table values here,
// rather than converting through host floats at each update, makes the source
// difficulty curve part of the portable deterministic contract.

fn difficulty_increment(value: PicoFixed, half: bool) -> PicoFixed {
    if half {
        PicoFixed::from_raw(value.raw() / 2)
    } else {
        value
    }
}

fn initial_speed(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(0.8),
        1 => PicoFixed::ONE,
        _ => PicoFixed::from_f32(1.6),
    }
}

fn initial_enemy_est(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(1.4),
        1 => PicoFixed::ONE,
        _ => PicoFixed::from_f32(0.8),
    }
}

fn initial_enemy_stats(powerups_enabled: bool) -> [PicoFixed; 5] {
    let powerup_value = if powerups_enabled {
        PicoFixed::from_int(2)
    } else {
        PicoFixed::ZERO
    };
    [
        PicoFixed::from_f32(76.5),
        PicoFixed::from_f32(17.5),
        powerup_value,
        PicoFixed::ZERO,
        powerup_value,
    ]
}

fn initial_bounce_static(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(0.8),
        1 => PicoFixed::from_f32(0.65),
        _ => PicoFixed::from_f32(0.6),
    }
}

fn initial_bounce_moving(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(1.05),
        1 => PicoFixed::from_f32(0.9),
        _ => PicoFixed::from_f32(0.6),
    }
}

fn cycle_index(current: u8, count: u8, increase: bool) -> u8 {
    if increase {
        if current >= count {
            1
        } else {
            current + 1
        }
    } else if current <= 1 {
        count
    } else {
        current - 1
    }
}

fn theme_values(index: u8) -> (u8, u8) {
    [
        (12, 1),
        (1, 0),
        (3, 1),
        (13, 1),
        (2, 1),
        (9, 8),
        (14, 2),
        (6, 5),
        (5, 0),
        (0, 0),
        (0, 8),
        (0, 12),
        (0, 11),
    ]
    .get(usize::from(index.saturating_sub(1)))
    .copied()
    .unwrap_or((12, 1))
}

fn settings_row(cursor: u8) -> i16 {
    match cursor {
        1 => 29,
        2 => 59,
        3 => 69,
        4 => 79,
        _ => 29,
    }
}

#[cfg(test)]
mod tests {
    use super::{pico_cosine, pico_sine, FrameEvent, NativeGame, INITIAL_PATTERN_DELAY_FRAMES};
    use crate::{
        Action, AudioEvent, CoreError, EnemyState, Mode, NativeConfig, PatternRect, PatternState,
        PatternTarget, PicoFixed, BUTTON_X_MASK,
    };

    fn start_game(game: &mut NativeGame) {
        assert!(game.advance_frame(BUTTON_X_MASK).is_ok());
        for _ in 1..13 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().mode, Mode::Game);
    }

    #[test]
    fn reset_starts_at_menu_frame_zero() {
        let game = NativeGame::new(NativeConfig::new(42));
        assert_eq!(game.lifecycle().frame, 0);
        assert_eq!(game.lifecycle().mode, Mode::Menu);
        assert!(!game.lifecycle().started);
    }

    #[test]
    fn v102_start_crosses_transition_on_observed_thirteenth_frame() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let first = game.advance_frame(BUTTON_X_MASK);
        assert!(first.is_ok());
        assert_eq!(first.map(|value| value.mode), Ok(Mode::TransitionToGame));
        for _ in 0..11 {
            assert!(game.advance_frame(BUTTON_X_MASK).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 12);
        assert_eq!(game.lifecycle().mode, Mode::TransitionToGame);
        let ready = game.advance_frame(0);
        assert!(ready.is_ok());
        assert_eq!(ready.as_ref().map(|value| value.frame), Ok(13));
        assert_eq!(ready.as_ref().map(|value| value.mode), Ok(Mode::Game));
        assert!(game.lifecycle().game_ready);
    }

    #[test]
    fn action_step_advances_exact_frames_and_neutral_is_zero_mask() {
        let mut game = NativeGame::new(NativeConfig::default());
        let result = game.step(Action::Neutral, 4);
        assert!(result.is_ok());
        assert_eq!(result.as_ref().map(|value| value.frame), Ok(4));
        assert_eq!(result.as_ref().map(|value| value.input_mask), Ok(0));
        assert_eq!(
            result.as_ref().map(|value| value.previous_input_mask),
            Ok(0)
        );
    }

    #[test]
    fn frozen_center_time_to_death_reports_current_collision_at_zero() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(64),
            PicoFixed::from_int(64),
            PicoFixed::from_int(3),
        ));
        let before = game.snapshot().canonical_bytes();

        assert_eq!(
            game.time_to_death_at(PicoFixed::from_int(64), PicoFixed::from_int(64), 32,)
                .ok(),
            Some(Some(0))
        );
        assert_eq!(game.snapshot().canonical_bytes(), before);
    }

    #[test]
    fn frame_result_snapshot_matches_current_capture_boundary() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let inputs = [BUTTON_X_MASK, BUTTON_X_MASK, 0, 1, 4, 16, 32, 0, 0, 0];

        for input in inputs {
            let result = game.advance_frame(input);
            assert!(result.is_ok());
            if let Ok(result) = result {
                let current = game.snapshot();
                assert_eq!(result.snapshot, current);
                assert_eq!(result.snapshot.canonical_bytes(), current.canonical_bytes());
            }
        }
    }

    #[test]
    fn ml_frame_path_matches_canonical_logical_trace_without_physical_screen() {
        let mut canonical = NativeGame::new(NativeConfig::new(42));
        let mut ml = NativeGame::new(NativeConfig::new(42));

        for frame in 0..180 {
            let input = if frame == 0 {
                BUTTON_X_MASK
            } else {
                Action::ALL
                    .get(frame % Action::ALL.len())
                    .copied()
                    .unwrap_or_else(|| unreachable!("action table index is bounded"))
                    .mask()
            };
            let canonical_result = canonical.advance_frame(input);
            let ml_result = ml.advance_frame_ml(input, input);
            assert_eq!(
                canonical_result
                    .as_ref()
                    .map(|result| { (result.frame, result.mode, result.done, result.reward) }),
                ml_result.as_ref().map(|result| (
                    result.frame,
                    result.mode,
                    result.done,
                    result.reward
                ))
            );

            let expected = canonical.full_state();
            let mut actual = ml.full_state();
            actual.physical_screen = expected.physical_screen.clone();
            assert_eq!(actual, expected, "logical state diverged at frame {frame}");
        }
    }

    #[test]
    fn pixel_frame_path_matches_canonical_render_trace() {
        let mut canonical = NativeGame::new(NativeConfig::new(42));
        let mut pixels = NativeGame::new(NativeConfig::new(42));

        for frame in 0..180 {
            let input = if frame == 0 {
                BUTTON_X_MASK
            } else {
                Action::ALL
                    .get(frame % Action::ALL.len())
                    .copied()
                    .unwrap_or_else(|| unreachable!("action table index is bounded"))
                    .mask()
            };
            let canonical_result = canonical.advance_frame(input);
            let pixel_result = pixels.advance_frame_pixels(input, input);
            assert_eq!(
                canonical_result.as_ref().map(|result| (
                    result.frame,
                    result.mode,
                    result.done,
                    result.reward
                )),
                pixel_result
                    .as_ref()
                    .map(|result| { (result.frame, result.mode, result.done, result.reward) })
            );

            let Ok(canonical_result) = canonical_result else {
                unreachable!("canonical frame should succeed")
            };
            let Ok(pixel_result) = pixel_result else {
                unreachable!("pixel frame should succeed")
            };
            assert_eq!(
                pixel_result.pixels.pixels(),
                canonical_result.snapshot.pixels()
            );
            assert_eq!(
                pixel_result.survival_frames,
                canonical_result.snapshot.logical_state().survival_frames
            );
            let player = &canonical_result.snapshot.logical_state().player;
            assert_eq!(pixel_result.player_position, [player.x, player.y]);
            if pixel_result.done {
                break;
            }
        }
    }

    #[test]
    fn invalid_input_and_zero_step_fail_without_mutation() {
        let mut game = NativeGame::new(NativeConfig::default());
        assert_eq!(
            game.advance_frame(64),
            Err(CoreError::InvalidButtonMask(64))
        );
        assert_eq!(game.lifecycle().frame, 0);
        assert_eq!(
            game.step(Action::Neutral, 0),
            Err(CoreError::InvalidFrameCount(0))
        );
        assert_eq!(game.lifecycle().frame, 0);
    }

    #[test]
    fn player_movement_uses_fixed_friction_and_clamped_bounds() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        let first = game.advance_frame(1);
        assert!(first.is_ok());
        assert_eq!(game.player().x, PicoFixed::from_f32(63.5));
        assert_eq!(game.player().vx, PicoFixed::from_f32(-0.5));
        let second = game.advance_frame(1);
        assert!(second.is_ok());
        assert_eq!(game.player().x, PicoFixed::from_raw(4_102_554));
        assert_eq!(game.player().vx, PicoFixed::from_f32(-0.9));
    }

    #[test]
    fn first_friendly_spawn_and_normal_enemy_match_seed_42_slice() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        for _ in 0..44 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 57);
        assert_eq!(game.enemies().len(), 4);
        let first = game.enemies().first().copied();
        assert_eq!(
            first.map(|enemy| enemy.x),
            Some(PicoFixed::from_raw(-654_705))
        );
        assert_eq!(
            first.map(|enemy| enemy.y),
            Some(PicoFixed::from_raw(-654_705))
        );
        assert_eq!(first.map(|enemy| enemy.size), Some(PicoFixed::from_int(2)));
        assert_eq!(game.score(), PicoFixed::ZERO);
    }

    #[test]
    fn first_normal_spawn_preserves_source_init_rng_offset() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        for _ in 0..54 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 67);
        let first_normal = game.enemies().get(4).copied();
        assert_eq!(
            first_normal.map(|enemy| enemy.max_size),
            Some(PicoFixed::from_int(3))
        );
        assert_eq!(
            first_normal.map(|enemy| enemy.size),
            Some(PicoFixed::from_int(2))
        );
    }

    #[test]
    fn initial_pattern_rng_branch_is_seed_dependent() {
        let mut game = NativeGame::new(NativeConfig::new(7));
        start_game(&mut game);
        for _ in 0..57 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 70);
        assert_eq!(
            game.enemies().get(4).map(|enemy| enemy.max_size),
            Some(PicoFixed::from_int(5))
        );
    }

    #[test]
    fn first_pattern_selection_converts_sample_to_integer_index() {
        let mut game = NativeGame::new(NativeConfig::new(17));
        game.lifecycle.mode = Mode::Game;
        game.lifecycle.game_ready = true;
        game.lifecycle.started = true;
        game.should_collide = false;
        game.enemy_should_collide = false;
        game.pattern_timer = INITIAL_PATTERN_DELAY_FRAMES - 1;

        let boundary_result = game.advance_frame(0);
        assert!(boundary_result.is_ok());
        let Some(boundary) = boundary_result.ok() else {
            return;
        };

        assert_eq!(game.lifecycle().frame, 1);
        assert!(boundary.events.contains(&FrameEvent::PatternActive));
        assert!(game.snapshot().logical_state().active_pattern.is_some());
    }

    #[test]
    fn normal_spawn_size_is_reused_by_later_friendly_spawn() {
        let mut game = NativeGame::new(NativeConfig::new(7));
        start_game(&mut game);
        for _ in 0..95 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 108);
        assert_eq!(game.enemies().len(), 9);
        assert!(game
            .enemies()
            .iter()
            .skip(5)
            .all(|enemy| { enemy.max_size == PicoFixed::from_int(5) }));
    }

    #[test]
    fn normal_enemy_size_roll_uses_source_thresholds() {
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_int(20)),
            PicoFixed::from_int(3)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_raw(20 * 65_536 + 1)),
            PicoFixed::from_int(4)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_int(70)),
            PicoFixed::from_int(4)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_raw(70 * 65_536 + 1)),
            PicoFixed::from_int(5)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_int(90)),
            PicoFixed::from_int(5)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_raw(90 * 65_536 + 1)),
            PicoFixed::from_int(6)
        );
    }

    #[test]
    fn normal_personality_roll_matches_low_score_source_distribution() {
        assert_eq!(super::normal_personality_from_roll(PicoFixed::ZERO), 0);
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(75)),
            0
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(76)),
            1
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(92)),
            1
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(93)),
            2
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(95)),
            4
        );
    }

    #[test]
    fn v149_collision_flags_gate_player_and_enemy_collision() {
        let mut player_game = NativeGame::new(NativeConfig::default());
        start_game(&mut player_game);
        player_game.enemies.clear();
        player_game.should_collide = false;
        player_game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(61),
            PicoFixed::from_int(64),
            PicoFixed::from_int(3),
        ));
        assert!(player_game.advance_frame(0).is_ok());
        assert!(!player_game.lifecycle.dead);

        let mut enemy_game = NativeGame::new(NativeConfig::default());
        start_game(&mut enemy_game);
        enemy_game.enemies.clear();
        enemy_game.enemy_should_collide = false;
        let mut first = EnemyState::normal(
            PicoFixed::from_int(10),
            PicoFixed::from_int(10),
            PicoFixed::from_int(3),
        );
        first.inside = true;
        let mut second = first;
        second.x = PicoFixed::from_int(11);
        enemy_game.enemies.push(first);
        enemy_game.enemies.push(second);
        assert!(enemy_game.advance_frame(0).is_ok());
        assert!(enemy_game.enemies().iter().all(|enemy| !enemy.is_dying));
    }

    #[test]
    fn v166_enemy_collision_shatters_score_and_counts_kills() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        game.enemies.clear();
        assert_eq!(game.shattered(), 0);
        let mut first = EnemyState::normal(
            PicoFixed::from_int(10),
            PicoFixed::from_int(10),
            PicoFixed::from_int(3),
        );
        first.inside = true;
        let mut second = first;
        second.x = PicoFixed::from_int(11);
        game.enemies.push(first);
        game.enemies.push(second);
        let score_before = game.score();
        assert!(game.advance_frame(0).is_ok());
        assert!(game.enemies().iter().all(|enemy| enemy.is_dying));
        assert!(game.advance_frame(0).is_ok());
        assert!(game.enemies().is_empty());
        assert_eq!(game.shattered(), 2);
        assert!(game.score() > score_before);
    }

    #[test]
    fn v150_difficulty_curve_uses_all_source_tables_and_half_steps() {
        let expected = [
            (131, 327, 32, 29_491, 85, 29_491),
            (452, 655, 1_245, 29_491, 1_245, 29_491),
            (1_835, 1_310, 1_572, 26_214, 1_572, 26_214),
        ];
        for (difficulty_index, values) in expected.iter().enumerate() {
            let difficulty = u8::try_from(difficulty_index + 1).unwrap_or(1);
            let curve = super::DIFFICULTY_CURVES.get(difficulty_index).copied();
            assert!(curve.is_some());
            let Some(curve) = curve else {
                continue;
            };
            assert_eq!(curve.speed_increment, PicoFixed::from_raw(values.0));
            assert_eq!(curve.enemy_increment, PicoFixed::from_raw(values.1));
            assert_eq!(curve.static_increment, PicoFixed::from_raw(values.2));
            assert_eq!(curve.static_target, PicoFixed::from_raw(values.3));
            assert_eq!(curve.moving_increment, PicoFixed::from_raw(values.4));
            assert_eq!(curve.moving_target, PicoFixed::from_raw(values.5));

            let mut config = NativeConfig::new(42);
            config.difficulty = difficulty;
            let mut full = NativeGame::new(config);
            full.apply_difficulty(false);
            assert_eq!(
                full.speed,
                super::pico_lerp(
                    super::initial_speed(difficulty),
                    super::DIFFICULTY_SPEED_TARGET,
                    curve.speed_increment,
                )
            );
            assert_eq!(
                full.enemy_est,
                super::pico_lerp(
                    super::initial_enemy_est(difficulty),
                    super::DIFFICULTY_EST_TARGET,
                    curve.enemy_increment,
                )
            );
            assert_eq!(
                full.bounce_cap_static,
                super::pico_lerp(
                    super::initial_bounce_static(difficulty),
                    curve.static_target,
                    curve.static_increment,
                )
            );
            assert_eq!(
                full.bounce_cap_moving,
                super::pico_lerp(
                    super::initial_bounce_moving(difficulty),
                    curve.moving_target,
                    curve.moving_increment,
                )
            );

            let mut half = NativeGame::new(config);
            half.apply_difficulty(true);
            assert_eq!(
                half.speed,
                super::pico_lerp(
                    super::initial_speed(difficulty),
                    super::DIFFICULTY_SPEED_TARGET,
                    super::difficulty_increment(curve.speed_increment, true),
                )
            );
            assert_eq!(
                half.enemy_est,
                super::pico_lerp(
                    super::initial_enemy_est(difficulty),
                    super::DIFFICULTY_EST_TARGET,
                    super::difficulty_increment(curve.enemy_increment, true),
                )
            );
            assert_eq!(
                half.bounce_cap_static,
                super::pico_lerp(
                    super::initial_bounce_static(difficulty),
                    curve.static_target,
                    super::difficulty_increment(curve.static_increment, true),
                )
            );
            assert_eq!(
                half.bounce_cap_moving,
                super::pico_lerp(
                    super::initial_bounce_moving(difficulty),
                    curve.moving_target,
                    super::difficulty_increment(curve.moving_increment, true),
                )
            );
        }
    }

    #[test]
    fn v153_powerup_explosion_preserves_source_mutable_list_order() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        game.enemies.push(EnemyState {
            personality: 2,
            ..EnemyState::normal(
                PicoFixed::from_int(60),
                PicoFixed::from_int(60),
                PicoFixed::from_int(3),
            )
        });
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(20),
            PicoFixed::from_int(20),
            PicoFixed::from_int(3),
        ));
        game.enemies.push(EnemyState {
            personality: 1,
            ..EnemyState::normal(
                PicoFixed::from_int(30),
                PicoFixed::from_int(30),
                PicoFixed::from_int(3),
            )
        });
        game.enemies.push(EnemyState {
            personality: 3,
            ..EnemyState::normal(
                PicoFixed::from_int(40),
                PicoFixed::from_int(40),
                PicoFixed::from_int(3),
            )
        });
        let initiating = game.enemies.first().copied();
        assert!(initiating.is_some());
        let mut events = Vec::new();
        if let Some(initiating) = initiating {
            game.collide_enemy(0, initiating, &mut events);
        }

        assert_eq!(game.score, PicoFixed::from_int(5));
        assert_eq!(game.enemies.len(), 1);
        assert_eq!(
            game.enemies.first().map(|enemy| enemy.personality),
            Some(-1)
        );
        assert_eq!(game.particles.len(), 44);
        assert_eq!(
            game.enemies.first().map(|enemy| enemy.x),
            Some(PicoFixed::from_f32(30.5))
        );
    }

    #[test]
    fn v163_kamikaze_reaches_max_size_before_switching_to_shrink() {
        let mut game = NativeGame::new(NativeConfig::default());
        game.enemies.push(EnemyState {
            size: PicoFixed::from_int(29),
            max_size: PicoFixed::from_int(30),
            personality: -1,
            life: None,
            x: PicoFixed::from_int(50),
            y: PicoFixed::from_int(50),
            ..EnemyState::normal(
                PicoFixed::from_int(50),
                PicoFixed::from_int(50),
                PicoFixed::from_int(30),
            )
        });

        game.update_enemies();

        let Some(explosion) = game.enemies.first() else {
            return;
        };
        assert_eq!(explosion.size, PicoFixed::from_int(30));
        assert!(explosion.isizing);
        assert_eq!(explosion.x, PicoFixed::from_f32(49.5));
        assert_eq!(explosion.y, PicoFixed::from_f32(49.5));
    }

    #[test]
    fn v164_dying_personality_one_uses_pre_growth_kamikaze_source() {
        let mut game = NativeGame::new(NativeConfig::default());
        game.enemies.push(EnemyState {
            personality: 1,
            size: PicoFixed::from_int(3),
            max_size: PicoFixed::from_int(5),
            is_dying: true,
            x: PicoFixed::from_int(20),
            y: PicoFixed::from_int(30),
            ..EnemyState::normal(
                PicoFixed::from_int(20),
                PicoFixed::from_int(30),
                PicoFixed::from_int(3),
            )
        });

        game.update_enemies();

        let Some(explosion) = game.enemies.first() else {
            assert!(
                !game.enemies.is_empty(),
                "dying personality-1 enemy did not spawn a kamikaze"
            );
            return;
        };
        // The appended kamikaze is visited later in this same update, so its
        // first visible state has grown by one pixel and shifted by 0.5.
        assert_eq!(explosion.personality, -1);
        assert_eq!(explosion.size, PicoFixed::ONE);
        assert_eq!(explosion.x, PicoFixed::from_f32(21.0));
        assert_eq!(explosion.y, PicoFixed::from_f32(31.0));
    }

    #[test]
    fn v165_kamikaze_does_not_bounce_from_pattern_rectangles() {
        let mut game = NativeGame::new(NativeConfig::default());
        game.patterns = vec![crate::PatternState {
            id: 1,
            mins: PicoFixed::ZERO,
            maxs: PicoFixed::from_int(100),
            probability: PicoFixed::ONE,
            variants: Vec::new(),
            smooth: false,
            pattern_type: 0,
            bounce_cap: false,
            spawn_enabled: false,
            automatic_variant: None,
            special: PicoFixed::ZERO,
            counter: 0,
            timer: PicoFixed::ZERO,
            rects: vec![crate::PatternRect {
                x: PicoFixed::from_int(80),
                y: PicoFixed::from_int(16),
                width: PicoFixed::from_int(20),
                height: PicoFixed::from_int(20),
                speed: PicoFixed::from_int(12),
                dx: PicoFixed::ZERO,
                dy: PicoFixed::ZERO,
                targets: Vec::new(),
                target_index: 0,
                wait: PicoFixed::ZERO,
                shown: true,
                sh: PicoFixed::from_int(2),
                warnings: Vec::new(),
                collision_done: false,
                finished: false,
            }],
        }];
        game.active_pattern = Some(0);
        game.enemies.push(EnemyState {
            personality: -1,
            size: PicoFixed::from_int(20),
            max_size: PicoFixed::from_int(30),
            x: PicoFixed::from_int(80),
            y: PicoFixed::from_int(-4),
            ..EnemyState::normal(
                PicoFixed::from_int(80),
                PicoFixed::from_int(-4),
                PicoFixed::from_int(20),
            )
        });

        game.update_enemies();

        let Some(explosion) = game.enemies.first() else {
            assert!(
                !game.enemies.is_empty(),
                "kamikaze disappeared during pattern update"
            );
            return;
        };
        assert_eq!(explosion.vx, PicoFixed::ZERO);
        assert_eq!(explosion.vy, PicoFixed::ZERO);
        assert_eq!(explosion.size, PicoFixed::from_int(21));
        assert_eq!(explosion.x, PicoFixed::from_f32(79.5));
        assert_eq!(explosion.y, PicoFixed::from_f32(-4.5));
    }

    #[test]
    fn v154_powerup_pattern_collision_uses_pre_growth_source_size() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        game.patterns = vec![crate::PatternState {
            id: 1,
            mins: PicoFixed::ZERO,
            maxs: PicoFixed::from_int(100),
            probability: PicoFixed::ONE,
            variants: Vec::new(),
            smooth: false,
            pattern_type: 0,
            bounce_cap: false,
            spawn_enabled: false,
            automatic_variant: None,
            special: PicoFixed::ZERO,
            counter: 0,
            timer: PicoFixed::ZERO,
            rects: vec![crate::PatternRect {
                x: PicoFixed::from_int(10),
                y: PicoFixed::from_int(10),
                width: PicoFixed::from_int(10),
                height: PicoFixed::from_int(10),
                speed: PicoFixed::from_int(12),
                dx: PicoFixed::ZERO,
                dy: PicoFixed::ZERO,
                targets: Vec::new(),
                target_index: 0,
                wait: PicoFixed::ZERO,
                shown: true,
                sh: PicoFixed::from_int(2),
                warnings: Vec::new(),
                collision_done: false,
                finished: false,
            }],
        }];
        game.active_pattern = Some(0);
        game.enemies.push(EnemyState {
            personality: 2,
            inside: true,
            ..EnemyState::normal(
                PicoFixed::from_int(7),
                PicoFixed::from_int(10),
                PicoFixed::from_int(3),
            )
        });

        game.update_enemies();

        let enemy = game.enemies.first().copied();
        assert_eq!(enemy.map(|value| value.size), Some(PicoFixed::from_int(4)));
        assert_eq!(enemy.map(|value| value.vx), Some(PicoFixed::from_f32(0.01)));
        assert_eq!(
            enemy.map(|value| value.x),
            Some(PicoFixed::from_raw(459_407))
        );
    }

    #[test]
    fn v122_restore_replays_the_same_next_frame() {
        let mut original = NativeGame::new(NativeConfig::new(42));
        start_game(&mut original);
        for _ in 0..90 {
            assert!(original.advance_frame(0).is_ok());
        }
        let checkpoint = original.snapshot();
        let restored_result = NativeGame::restore(&checkpoint);
        assert!(restored_result.is_ok());
        let Some(mut restored) = restored_result.ok() else {
            return;
        };
        assert_eq!(
            restored.snapshot().canonical_bytes(),
            checkpoint.canonical_bytes()
        );

        let expected = original.advance_frame(2);
        let actual = restored.advance_frame(2);
        assert!(expected.is_ok());
        assert!(actual.is_ok());
        if let (Ok(expected), Ok(actual)) = (expected, actual) {
            assert_eq!(actual.frame, expected.frame);
            assert_eq!(actual.reward, expected.reward);
            assert_eq!(actual.events, expected.events);
            assert_eq!(
                actual.snapshot.canonical_bytes(),
                expected.snapshot.canonical_bytes()
            );
        }
    }

    #[test]
    fn v157_pattern_targets_warnings_and_completion_follow_source_order() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let moving = PatternRect {
            x: PicoFixed::from_int(10),
            y: PicoFixed::from_int(20),
            width: PicoFixed::from_int(16),
            height: PicoFixed::from_int(6),
            speed: PicoFixed::from_f32(0.7),
            dx: PicoFixed::from_int(1),
            dy: PicoFixed::ZERO,
            targets: vec![
                PatternTarget::SetFyou(false),
                PatternTarget::Wait(PicoFixed::from_f32(0.05)),
            ],
            target_index: 0,
            wait: PicoFixed::ZERO,
            shown: true,
            sh: PicoFixed::ZERO,
            warnings: Vec::new(),
            collision_done: false,
            finished: false,
        };
        game.patterns = vec![PatternState {
            id: 1,
            mins: PicoFixed::ZERO,
            maxs: PicoFixed::from_int(100),
            probability: PicoFixed::from_int(15),
            variants: Vec::new(),
            smooth: false,
            pattern_type: 1,
            bounce_cap: false,
            spawn_enabled: true,
            automatic_variant: None,
            special: PicoFixed::ZERO,
            counter: 7,
            timer: PicoFixed::ZERO,
            rects: vec![moving],
        }];
        game.active_pattern = Some(0);
        game.friendly_enabled = true;

        game.update_active_pattern();
        let moving = game
            .patterns
            .first()
            .and_then(|pattern| pattern.rects.first());
        assert_eq!(moving.map(|rect| rect.sh), Some(PicoFixed::from_f32(0.02)));
        assert_eq!(moving.map(|rect| rect.warnings.len()), Some(2));
        assert_eq!(
            moving
                .and_then(|rect| rect.warnings.first())
                .map(|warning| warning.x0),
            Some(PicoFixed::from_int(32))
        );
        assert!(game.friendly_enabled);

        for _ in 0..99 {
            game.update_active_pattern();
        }
        assert_eq!(
            game.patterns
                .first()
                .and_then(|pattern| pattern.rects.first())
                .map(|rect| rect.sh),
            Some(PicoFixed::from_int(2))
        );
        game.update_active_pattern();
        assert!(!game.friendly_enabled);
        assert_eq!(
            game.patterns
                .first()
                .and_then(|pattern| pattern.rects.first())
                .map(|rect| rect.target_index),
            Some(1)
        );
        for _ in 0..2 {
            game.update_active_pattern();
        }
        assert!(game.active_pattern.is_some());
        game.update_active_pattern();
        assert!(game.active_pattern.is_none());
        assert!(game.friendly_enabled);
        assert_eq!(game.spawns.len(), 4);
    }

    #[test]
    fn normal_enemy_collision_marks_terminal_without_survival_reward() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(61),
            PicoFixed::from_int(64),
            PicoFixed::from_int(3),
        ));
        let result = game.advance_frame(crate::Button::Right.mask());
        assert_eq!(result.as_ref().map(|value| value.done), Ok(true));
        assert!(game.lifecycle().dead);
        assert!(game.enemies().is_empty());
        assert_eq!(game.survival_frames(), 0);
        assert_eq!(
            result.as_ref().map(|value| value.reward),
            Ok(PicoFixed::ZERO)
        );
        assert_eq!(
            result.as_ref().map(|value| value.input_mask),
            Ok(crate::Button::Right.mask())
        );
        assert_eq!(
            result.as_ref().map(|value| value.previous_input_mask),
            Ok(crate::Button::Right.mask())
        );
    }

    #[test]
    fn v161_static_pattern_collision_matches_source() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        game.player.x = PicoFixed::from_int(5);
        game.player.y = PicoFixed::from_int(5);
        game.patterns = vec![PatternState {
            id: 1,
            mins: PicoFixed::ZERO,
            maxs: PicoFixed::from_int(100),
            probability: PicoFixed::ONE,
            variants: Vec::new(),
            smooth: false,
            pattern_type: 0,
            bounce_cap: false,
            spawn_enabled: false,
            automatic_variant: None,
            special: PicoFixed::ZERO,
            counter: 0,
            timer: PicoFixed::ZERO,
            rects: vec![PatternRect {
                x: PicoFixed::ZERO,
                y: PicoFixed::ZERO,
                width: PicoFixed::from_int(16),
                height: PicoFixed::from_int(16),
                speed: PicoFixed::from_int(12),
                dx: PicoFixed::ZERO,
                dy: PicoFixed::ZERO,
                targets: Vec::new(),
                target_index: 0,
                wait: PicoFixed::ZERO,
                shown: true,
                sh: PicoFixed::from_int(2),
                warnings: Vec::new(),
                collision_done: false,
                finished: false,
            }],
        }];
        game.active_pattern = Some(0);

        let result = game.advance_frame(0);

        assert_eq!(result.as_ref().map(|value| value.done), Ok(true));
        assert!(result.is_ok_and(|value| value.events.contains(&FrameEvent::Death)));
        assert!(game.lifecycle().dead);
    }

    #[test]
    fn powerup_collision_applies_size_reward_and_difficulty_without_terminal() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        game.enemies.push(EnemyState {
            personality: 4,
            ..EnemyState::normal(
                PicoFixed::from_int(61),
                PicoFixed::from_int(64),
                PicoFixed::from_int(3),
            )
        });
        let result = game.advance_frame(0);
        assert_eq!(result.as_ref().map(|value| value.done), Ok(false));
        let snapshot = game.snapshot();
        assert_eq!(snapshot.logical_state().player.size, PicoFixed::from_int(2));
        assert_eq!(game.score(), PicoFixed::ONE);
        assert!(game.enemies().is_empty());
    }

    #[test]
    fn v158_audio_events_preserve_source_order_and_restart_boundary() {
        let mut game = NativeGame::new(NativeConfig::default());
        let start = game.advance_frame(BUTTON_X_MASK);
        assert_eq!(
            start.as_ref().map(|result| result.audio.as_slice()),
            Ok([AudioEvent::Sfx {
                id: 55,
                channel: Some(-2),
            }]
            .as_slice())
        );
        start_game(&mut game);

        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(61),
            PicoFixed::from_int(64),
            PicoFixed::from_int(3),
        ));
        let death = game.advance_frame(0);
        assert_eq!(
            death.as_ref().map(|result| result.audio.as_slice()),
            Ok([
                AudioEvent::Sfx {
                    id: 62,
                    channel: None,
                },
                AudioEvent::Music { track: 22 },
            ]
            .as_slice())
        );
        assert!(game.lifecycle().dead);

        let restart = game.advance_frame(BUTTON_X_MASK);
        assert_eq!(
            restart.as_ref().map(|result| result.audio.as_slice()),
            Ok([
                AudioEvent::Music { track: 3 },
                AudioEvent::Sfx {
                    id: 55,
                    channel: Some(-2),
                },
            ]
            .as_slice())
        );
        assert!(!game.lifecycle().dead);
        assert_eq!(game.score(), PicoFixed::ZERO);
        assert_eq!(game.survival_frames(), 1);
        assert!(game.enemies().is_empty());
    }

    #[test]
    fn v162_retry_resets_player_to_center_and_clears_velocity() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        game.player.x = PicoFixed::from_int(11);
        game.player.y = PicoFixed::from_int(117);
        game.player.vx = PicoFixed::from_f32(2.5);
        game.player.vy = PicoFixed::from_f32(-1.5);
        game.lifecycle.dead = true;
        game.can_click = true;

        let retry = game.advance_frame(BUTTON_X_MASK);

        assert!(retry.is_ok());
        assert!(!game.lifecycle.dead);
        assert_eq!(game.player, crate::PlayerState::new());
    }

    #[test]
    fn v158_settings_change_emits_one_source_sfx() {
        let mut game = NativeGame::new(NativeConfig::default());
        game.lifecycle.mode = Mode::Settings;
        let result = game.advance_frame(crate::Button::Right.mask());
        assert_eq!(
            result.as_ref().map(|value| value.audio.as_slice()),
            Ok([AudioEvent::Sfx {
                id: 58,
                channel: None,
            }]
            .as_slice())
        );
    }

    #[test]
    fn v109_ordered_enemy_overlap_removes_marked_entries_and_applies_side_effects() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(10),
            PicoFixed::from_int(10),
            PicoFixed::from_int(3),
        ));
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(11),
            PicoFixed::from_int(10),
            PicoFixed::from_int(3),
        ));

        assert!(game.advance_frame(0).is_ok());
        assert_eq!(game.enemies().len(), 2);
        assert!(game.enemies().iter().all(|enemy| enemy.is_dying));

        assert!(game.advance_frame(0).is_ok());
        assert!(game.enemies().is_empty());
        let snapshot = game.snapshot();
        let state = snapshot.logical_state();
        assert_eq!(state.score, PicoFixed::ONE);
        assert!(state.speed > PicoFixed::ONE);
        assert!(state.enemy_est < PicoFixed::ONE);
    }

    #[test]
    fn draw_side_effect_rng_is_consumed_for_powerup_enemy_trails() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        game.enemies.push(EnemyState {
            personality: 4,
            ..EnemyState::normal(
                PicoFixed::from_int(10),
                PicoFixed::from_int(10),
                PicoFixed::from_int(3),
            )
        });
        let before = game.rng_checkpoint();
        assert!(game.advance_frame(0).is_ok());
        let after = game.rng_checkpoint();

        let mut expected = crate::PicoRng::new(42);
        assert!(expected.restore(before).is_ok());
        let _ = expected.rnd(PicoFixed::from_int(32));
        let _ = expected.rnd(PicoFixed::from_int(32));
        let _ = expected.rnd(PicoFixed::ONE);
        let _ = expected.rnd(PicoFixed::from_int(15));
        assert_eq!(after, expected.checkpoint());
    }

    #[test]
    fn v140_kamikaze_speed_updates_once_before_movement() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        game.enemies.push(EnemyState {
            x: PicoFixed::from_int(80),
            y: PicoFixed::from_int(64),
            personality: 1,
            ..EnemyState::normal(
                PicoFixed::from_int(80),
                PicoFixed::from_int(64),
                PicoFixed::from_int(3),
            )
        });
        assert!(game.advance_frame(0).is_ok());
        assert_eq!(
            game.enemies().last().map(|enemy| enemy.speed),
            Some(PicoFixed::from_raw(64_881))
        );
    }

    #[test]
    fn v159_trig_bridge_narrows_after_double_evaluation() {
        let angle = PicoFixed::from_f32(0.123456);
        let radians = f64::from(angle.to_f32()) * std::f64::consts::TAU;
        let expected_sine = PicoFixed::from_f32(-(radians.sin() as f32));
        let expected_cosine = PicoFixed::from_f32(radians.cos() as f32);

        assert_eq!(pico_sine(angle), expected_sine);
        assert_eq!(pico_cosine(angle), expected_cosine);
    }
}
