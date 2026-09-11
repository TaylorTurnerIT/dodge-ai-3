use crate::PicoFixed;

/// Mutable player fields used by the P3 movement slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlayerState {
    pub x: PicoFixed,
    pub y: PicoFixed,
    pub vx: PicoFixed,
    pub vy: PicoFixed,
    pub size: PicoFixed,
}

impl PlayerState {
    pub const fn new() -> Self {
        Self {
            x: PicoFixed::from_int(64),
            y: PicoFixed::from_int(64),
            vx: PicoFixed::ZERO,
            vy: PicoFixed::ZERO,
            size: PicoFixed::from_int(4),
        }
    }
}

/// A draw-side particle emitted by the cartridge's `part` list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParticleState {
    pub x: PicoFixed,
    pub y: PicoFixed,
    pub dx: PicoFixed,
    pub dy: PicoFixed,
    pub radius: PicoFixed,
    pub kind: i8,
    pub max_age: PicoFixed,
    pub age: u32,
    pub color: u8,
    pub colors: [u8; 3],
    pub color_count: u8,
}

impl ParticleState {
    pub const fn player_trail(x: PicoFixed, y: PicoFixed, radius: PicoFixed) -> Self {
        Self {
            x,
            y,
            dx: PicoFixed::ZERO,
            dy: PicoFixed::ZERO,
            radius,
            kind: 0,
            max_age: PicoFixed::from_int(10),
            age: 0,
            color: 0,
            colors: [7, 0, 0],
            color_count: 1,
        }
    }
}

impl Default for PlayerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Current settings-screen values and cursor state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SettingsState {
    pub theme_index: u8,
    pub theme_background: u8,
    pub theme_shadow: u8,
    pub difficulty: u8,
    pub patterns_enabled: bool,
    pub powerups_enabled: bool,
    pub cursor: u8,
    pub message_timer: u8,
    pub message_sprite: u8,
    pub message_x: i16,
    pub message_y: i16,
}

impl SettingsState {
    pub const fn new(config: crate::NativeConfig) -> Self {
        Self {
            theme_index: theme_index(config.theme_background, config.theme_shadow),
            theme_background: config.theme_background,
            theme_shadow: config.theme_shadow,
            difficulty: clamp_difficulty(config.difficulty),
            patterns_enabled: config.patterns_enabled,
            powerups_enabled: config.powerups_enabled,
            cursor: 1,
            message_timer: 0,
            message_sprite: 0,
            message_x: 0,
            message_y: 0,
        }
    }
}

const fn theme_index(background: u8, shadow: u8) -> u8 {
    match (background, shadow) {
        (12, 1) => 1,
        (1, 0) => 2,
        (3, 1) => 3,
        (13, 1) => 4,
        (2, 1) => 5,
        (9, 8) => 6,
        (14, 2) => 7,
        (6, 5) => 8,
        (5, 0) => 9,
        (0, 0) => 10,
        (0, 8) => 11,
        (0, 12) => 12,
        (0, 11) => 13,
        _ => 1,
    }
}

const fn clamp_difficulty(value: u8) -> u8 {
    if value < 1 {
        1
    } else if value > 3 {
        3
    } else {
        value
    }
}

/// Typed normal-enemy fields used by the P3 representative hazard slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnemyState {
    pub x: PicoFixed,
    pub y: PicoFixed,
    pub vx: PicoFixed,
    pub vy: PicoFixed,
    pub size: PicoFixed,
    pub max_size: PicoFixed,
    pub personality: i8,
    pub speed: PicoFixed,
    pub inside: bool,
    pub is_dying: bool,
    pub isizing: bool,
    pub life: Option<PicoFixed>,
}

impl EnemyState {
    pub const fn normal(x: PicoFixed, y: PicoFixed, max_size: PicoFixed) -> Self {
        Self {
            x,
            y,
            vx: PicoFixed::ZERO,
            vy: PicoFixed::ZERO,
            size: PicoFixed::ONE,
            max_size,
            personality: 0,
            speed: PicoFixed::ONE,
            inside: false,
            is_dying: false,
            isizing: true,
            life: None,
        }
    }
}
