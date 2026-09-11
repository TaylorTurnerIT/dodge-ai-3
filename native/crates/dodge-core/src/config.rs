use crate::PicoFixed;

/// Persistent settings and cartridge data supplied to a native game instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeConfig {
    pub seed: u32,
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
