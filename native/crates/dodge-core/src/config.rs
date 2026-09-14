use crate::PicoFixed;

/// Optional testing rules. Zero/default values preserve cartridge behavior.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScenarioRules {
    /// 0 = all enemies, 1 = none, 2 = normal only.
    pub enemy_mode: u8,
    /// Zero uses the ordinary schedule; 1..=39 freezes catalog geometry.
    pub permanent_pattern: u8,
    /// Disable player collisions for safe observation collection.
    pub invulnerable: bool,
}

impl ScenarioRules {
    pub const fn standard() -> Self {
        Self {
            enemy_mode: 0,
            permanent_pattern: 0,
            invulnerable: false,
        }
    }
    pub const fn valid(self) -> bool {
        self.enemy_mode <= 2 && self.permanent_pattern <= 39
    }
}

/// Persistent settings and cartridge data supplied to a native game instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeConfig {
    pub seed: u32,
    pub scenario: ScenarioRules,
    pub difficulty: u8,
    pub patterns_enabled: bool,
    pub powerups_enabled: bool,
    pub theme_background: u8,
    pub theme_shadow: u8,
    pub highscores: [PicoFixed; 12],
}

impl NativeConfig {
    pub const fn new(seed: u32) -> Self {
        Self {
            seed,
            scenario: ScenarioRules::standard(),
            difficulty: 2,
            patterns_enabled: true,
            powerups_enabled: true,
            theme_background: 12,
            theme_shadow: 1,
            highscores: [PicoFixed::ZERO; 12],
        }
    }
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self::new(42)
    }
}
