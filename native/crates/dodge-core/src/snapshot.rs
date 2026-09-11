use crate::{
    CoreError, EnemyState, InputState, LifecycleState, Mode, NativeGame, ParticleState,
    PatternRect, PatternState, PatternTarget, PicoFixed, PlayerState, RngCheckpoint, SettingsState,
    SpawnPoint, WarningLine, CORE_SCHEMA_VERSION,
};

mod embedded_assets {
    include!(concat!(env!("OUT_DIR"), "/gfx_indices.rs"));
}

pub const FRAMEBUFFER_WIDTH: usize = 128;
pub const FRAMEBUFFER_HEIGHT: usize = 128;
pub const FRAMEBUFFER_SIZE: usize = FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT;
pub const PALETTE_SIZE: usize = 16;
pub const SNAPSHOT_WIRE_VERSION: u32 = 8;
pub const CARTRIDGE_SOURCE_SHA256: [u8; 32] = [
    0x74, 0x53, 0xa9, 0x65, 0x8f, 0xd3, 0x25, 0x77, 0x38, 0x5a, 0xd7, 0x26, 0x72, 0xa5, 0x4a, 0xd8,
    0x4f, 0xf7, 0x05, 0x67, 0xfa, 0xdb, 0xde, 0x75, 0xba, 0x66, 0x34, 0xaa, 0x5c, 0xc6, 0x84, 0xa3,
];

const SNAPSHOT_MAGIC: [u8; 4] = *b"DGSN";
const MAX_SNAPSHOT_ENEMIES: u32 = 4_096;
const MAX_SNAPSHOT_PARTICLES: u32 = 16_384;
const MAX_SNAPSHOT_PATTERNS: u32 = 128;
const MAX_SNAPSHOT_PATTERN_RECTS: u32 = 4_096;
const MAX_SNAPSHOT_PATTERN_TARGETS: u32 = 256;
const MAX_SNAPSHOT_PATTERN_POINTS: u32 = 256;
const MAX_SNAPSHOT_PATTERN_WARNINGS: u32 = 256;
const MAX_SNAPSHOT_SPAWNS: u32 = 256;
const BACKGROUND_COLOR: u8 = 12;
const SHADOW_COLOR: u8 = 1;
const ENTITY_COLOR: u8 = 7;
const SPRITE_TRANSPARENT: u8 = 0;

/// Provenance carried by every native observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotProvenance {
    pub core_schema_version: u32,
    pub cartridge_sha256: [u8; 32],
}

impl SnapshotProvenance {
    pub const fn current() -> Self {
        Self {
            core_schema_version: CORE_SCHEMA_VERSION,
            cartridge_sha256: CARTRIDGE_SOURCE_SHA256,
        }
    }
}

/// Canonical mutable state exposed to native consumers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FullState {
    pub seed: u32,
    pub lifecycle: LifecycleState,
    pub input: InputState,
    pub rng: RngCheckpoint,
    pub player: PlayerState,
    pub enemies: Vec<EnemyState>,
    pub particles: Vec<ParticleState>,
    pub patterns: Vec<PatternState>,
    pub active_pattern: Option<usize>,
    pub spawns: Vec<SpawnPoint>,
    pub physical_screen: IndexedFramebuffer,
    pub enemy_timer: PicoFixed,
    pub enemy_est: PicoFixed,
    pub enemy_stats: [PicoFixed; 5],
    pub friendly_timer: u32,
    pub friendly_enabled: bool,
    pub enemy_max_size: PicoFixed,
    pub speed: PicoFixed,
    pub freeze_rate: PicoFixed,
    pub freeze_active: bool,
    pub freeze_timer: u32,
    pub size_timer: PicoFixed,
    pub patterns_enabled: bool,
    pub powerups_enabled: bool,
    pub pattern_timer: u32,
    pub pattern_delay_frames: u32,
    pub pattern_active: bool,
    pub new_highscore: bool,
    pub can_click: bool,
    pub has_played: bool,
    pub should_collide: bool,
    pub enemy_should_collide: bool,
    pub bounce_cap_static: PicoFixed,
    pub bounce_cap_moving: PicoFixed,
    pub bounce_cap: PicoFixed,
    pub score: PicoFixed,
    pub survival_frames: u32,
    pub shattered: u32,
    pub shake: PicoFixed,
    pub camera_x: PicoFixed,
    pub camera_y: PicoFixed,
    pub transition_render_y: i16,
    pub transition_from: Mode,
    pub settings: SettingsState,
    pub highscores: [PicoFixed; 12],
}

/// Indexed raster state needed to reproduce the PICO-8 draw boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderState {
    pub draw_color: u8,
    pub fill_pattern: u16,
    pub draw_palette: [u8; PALETTE_SIZE],
    pub screen_palette: [u8; PALETTE_SIZE],
    pub transparent: [bool; PALETTE_SIZE],
    pub camera_x: PicoFixed,
    pub camera_y: PicoFixed,
    pub clip_x: i16,
    pub clip_y: i16,
    pub clip_width: u16,
    pub clip_height: u16,
    pub transition_y: i16,
}

impl RenderState {
    pub const fn new(transition_y: i16) -> Self {
        Self {
            draw_color: 6,
            fill_pattern: 0,
            draw_palette: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            screen_palette: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            transparent: [
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false,
            ],
            camera_x: PicoFixed::ZERO,
            camera_y: PicoFixed::ZERO,
            clip_x: 0,
            clip_y: 0,
            clip_width: FRAMEBUFFER_WIDTH as u16,
            clip_height: FRAMEBUFFER_HEIGHT as u16,
            transition_y,
        }
    }

    pub fn validate(&self) -> bool {
        if self.draw_color >= PALETTE_SIZE as u8 {
            return false;
        }
        if self.clip_width > FRAMEBUFFER_WIDTH as u16
            || self.clip_height > FRAMEBUFFER_HEIGHT as u16
        {
            return false;
        }
        self.palette_is_valid(&self.draw_palette) && self.palette_is_valid(&self.screen_palette)
    }

    /// Map a source palette index through the draw and screen palettes.
    pub fn palette_index(&self, color: u8) -> Option<u8> {
        let drawn = self.draw_palette.get(usize::from(color)).copied()?;
        self.screen_palette.get(usize::from(drawn)).copied()
    }

    fn palette_is_valid(&self, palette: &[u8; PALETTE_SIZE]) -> bool {
        palette.iter().all(|color| *color < PALETTE_SIZE as u8)
    }
}

impl Default for RenderState {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Row-major PICO-8 palette-index framebuffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedFramebuffer {
    pixels: [u8; FRAMEBUFFER_SIZE],
}

impl IndexedFramebuffer {
    pub const fn blank() -> Self {
        Self {
            pixels: [0; FRAMEBUFFER_SIZE],
        }
    }

    pub const fn filled(color: u8) -> Self {
        Self {
            pixels: [color; FRAMEBUFFER_SIZE],
        }
    }

    pub fn pixels(&self) -> &[u8; FRAMEBUFFER_SIZE] {
        &self.pixels
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.pixels.as_slice()
    }

    /// Move the owned raster out of the framebuffer wrapper.
    pub fn into_array(self) -> [u8; FRAMEBUFFER_SIZE] {
        self.pixels
    }

    pub fn pixel(&self, x: usize, y: usize) -> Option<u8> {
        if x >= FRAMEBUFFER_WIDTH || y >= FRAMEBUFFER_HEIGHT {
            return None;
        }
        self.pixels.get(y * FRAMEBUFFER_WIDTH + x).copied()
    }

    fn clear(&mut self, color: u8, state: &RenderState) {
        let Some(mapped) = map_color(color, state) else {
            return;
        };
        let camera_x = camera_offset(state.camera_x);
        let camera_y = camera_offset(state.camera_y);
        for world_y in 0..FRAMEBUFFER_HEIGHT as i32 {
            for world_x in 0..FRAMEBUFFER_WIDTH as i32 {
                let screen_x = world_x - camera_x;
                let screen_y = world_y - camera_y;
                if (0..FRAMEBUFFER_WIDTH as i32).contains(&screen_x)
                    && (0..FRAMEBUFFER_HEIGHT as i32).contains(&screen_y)
                {
                    if let Some(pixel) = self
                        .pixels
                        .get_mut(screen_y as usize * FRAMEBUFFER_WIDTH + screen_x as usize)
                    {
                        *pixel = mapped;
                    }
                }
            }
        }
    }

    fn rect_fill(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u8, state: &RenderState) {
        let (start_x, end_x) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (start_y, end_y) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        for y in start_y..=end_y {
            for x in start_x..=end_x {
                self.plot_world(x, y, color, state);
            }
        }
    }

    fn rect_outline(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u8, state: &RenderState) {
        self.rect_line(x0, y0, x1, y0, color, state);
        self.rect_line(x1, y0, x1, y1, color, state);
        self.rect_line(x1, y1, x0, y1, color, state);
        self.rect_line(x0, y1, x0, y0, color, state);
    }

    fn rect_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u8, state: &RenderState) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        let mut x = x0;
        let mut y = y0;
        loop {
            self.plot_world(x, y, color, state);
            if x == x1 && y == y1 {
                break;
            }
            let double_error = 2 * error;
            if double_error >= dy {
                error += dy;
                x += sx;
            }
            if double_error <= dx {
                error += dx;
                y += sy;
            }
        }
    }

    fn circle_fill(
        &mut self,
        center_x: i32,
        center_y: i32,
        radius: PicoFixed,
        color: u8,
        state: &RenderState,
    ) {
        let radius = coordinate(radius).clamp(0, 8) as usize;
        for (row, half_width) in circle_scanlines(radius).iter().enumerate() {
            let dy = row as i32 - radius as i32;
            for x in center_x - i32::from(*half_width)..=center_x + i32::from(*half_width) {
                self.plot_world(x, center_y + dy, color, state);
            }
        }
    }

    fn plot_world(&mut self, x: i32, y: i32, color: u8, state: &RenderState) {
        // The oracle samples `pget` while the source camera is still active.
        // `pget` applies the same camera transform as drawing, so the captured
        // canonical buffer is camera-relative even though `cls` retains the
        // newly exposed edge of the physical screen.
        let camera_x = camera_offset(state.camera_x);
        let camera_y = camera_offset(state.camera_y);
        self.plot_screen(x - camera_x, y - camera_y, color, state);
    }

    fn plot_screen(&mut self, x: i32, y: i32, color: u8, state: &RenderState) {
        if x < 0
            || y < 0
            || x >= FRAMEBUFFER_WIDTH as i32
            || y >= FRAMEBUFFER_HEIGHT as i32
            || x < i32::from(state.clip_x)
            || y < i32::from(state.clip_y)
            || x >= i32::from(state.clip_x) + i32::from(state.clip_width)
            || y >= i32::from(state.clip_y) + i32::from(state.clip_height)
            || !pattern_bit(state.fill_pattern, x, y)
        {
            return;
        }
        let Some(mapped) = map_color(color, state) else {
            return;
        };
        if let Some(pixel) = self
            .pixels
            .get_mut(y as usize * FRAMEBUFFER_WIDTH + x as usize)
        {
            *pixel = mapped;
        }
    }

    pub(crate) fn project(&self, state: &RenderState) -> Self {
        let camera_x = camera_offset(state.camera_x);
        let camera_y = camera_offset(state.camera_y);
        let mut projected = Self::blank();
        for world_y in 0..FRAMEBUFFER_HEIGHT as i32 {
            for world_x in 0..FRAMEBUFFER_WIDTH as i32 {
                let screen_x = world_x - camera_x;
                let screen_y = world_y - camera_y;
                if (0..FRAMEBUFFER_WIDTH as i32).contains(&screen_x)
                    && (0..FRAMEBUFFER_HEIGHT as i32).contains(&screen_y)
                {
                    let source_index = screen_y as usize * FRAMEBUFFER_WIDTH + screen_x as usize;
                    let target_index = world_y as usize * FRAMEBUFFER_WIDTH + world_x as usize;
                    if let (Some(target), Some(source)) = (
                        projected.pixels.get_mut(target_index),
                        self.pixels.get(source_index),
                    ) {
                        *target = *source;
                    }
                }
            }
        }
        projected
    }

    fn sprite(
        &mut self,
        sprite_id: u8,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        state: &RenderState,
    ) {
        let sheet_x = i32::from(sprite_id % 16) * 8;
        let sheet_y = i32::from(sprite_id / 16) * 8;
        for row in 0..height * 8 {
            for column in 0..width * 8 {
                let source_x = sheet_x + column;
                let source_y = sheet_y + row;
                if !(0..128).contains(&source_x) || !(0..128).contains(&source_y) {
                    continue;
                }
                let Some(&color) =
                    embedded_assets::GFX_INDICES.get(source_y as usize * 128 + source_x as usize)
                else {
                    continue;
                };
                if color != SPRITE_TRANSPARENT {
                    self.plot_world(x + column, y + row, color, state);
                }
            }
        }
    }
}

impl Default for IndexedFramebuffer {
    fn default() -> Self {
        Self::blank()
    }
}

/// One complete native observation from one simulation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    provenance: SnapshotProvenance,
    logical_state: FullState,
    render_state: RenderState,
    framebuffer: IndexedFramebuffer,
}

impl Snapshot {
    pub(crate) fn from_game(game: &NativeGame) -> Self {
        let logical_state = game.full_state();
        let mut render_state = RenderState::new(logical_state.transition_render_y);
        render_state.camera_x = logical_state.camera_x;
        render_state.camera_y = logical_state.camera_y;
        render_state.screen_palette[12] = logical_state.settings.theme_background;
        render_state.screen_palette[1] = logical_state.settings.theme_shadow;
        let framebuffer = logical_state.physical_screen.project(&render_state);
        Self::with_parts(logical_state, render_state, framebuffer)
    }

    pub(crate) fn with_framebuffer_from_game(
        game: &NativeGame,
        framebuffer: IndexedFramebuffer,
        render_state: RenderState,
    ) -> Self {
        Self::with_parts(game.full_state(), render_state, framebuffer)
    }

    fn with_parts(
        logical_state: FullState,
        render_state: RenderState,
        framebuffer: IndexedFramebuffer,
    ) -> Self {
        Self {
            provenance: SnapshotProvenance::current(),
            logical_state,
            render_state,
            framebuffer,
        }
    }

    pub fn provenance(&self) -> SnapshotProvenance {
        self.provenance
    }

    pub fn logical_state(&self) -> &FullState {
        &self.logical_state
    }

    pub fn render_state(&self) -> &RenderState {
        &self.render_state
    }

    pub fn framebuffer(&self) -> &IndexedFramebuffer {
        &self.framebuffer
    }

    pub fn pixels(&self) -> &[u8; FRAMEBUFFER_SIZE] {
        self.framebuffer.pixels()
    }

    pub fn pixel_hash(&self) -> u64 {
        stable_hash(self.framebuffer.as_bytes())
    }

    pub fn state_hash(&self) -> u64 {
        let mut writer = Writer::new();
        write_full_state(&mut writer, &self.logical_state);
        stable_hash(writer.as_bytes())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.bytes(&SNAPSHOT_MAGIC);
        writer.u32(SNAPSHOT_WIRE_VERSION);
        writer.u32(self.provenance.core_schema_version);
        writer.bytes(&self.provenance.cartridge_sha256);
        write_full_state(&mut writer, &self.logical_state);
        write_render_state(&mut writer, &self.render_state);
        writer.bytes(self.framebuffer.as_bytes());
        writer.into_bytes()
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CoreError> {
        let mut reader = Reader::new(bytes);
        if reader.bytes(4)? != SNAPSHOT_MAGIC {
            return Err(CoreError::InvalidSnapshotMagic);
        }
        let version = reader.u32()?;
        if version != SNAPSHOT_WIRE_VERSION {
            return Err(CoreError::InvalidSnapshotVersion(version));
        }
        let core_schema_version = reader.u32()?;
        if core_schema_version != CORE_SCHEMA_VERSION {
            return Err(CoreError::InvalidSnapshotVersion(core_schema_version));
        }
        let mut cartridge_sha256 = [0_u8; 32];
        cartridge_sha256.copy_from_slice(reader.bytes(32)?);
        if cartridge_sha256 != CARTRIDGE_SOURCE_SHA256 {
            return Err(CoreError::InvalidSnapshotValue);
        }
        let logical_state = read_full_state(&mut reader)?;
        let render_state = read_render_state(&mut reader)?;
        if !render_state.validate() {
            return Err(CoreError::InvalidSnapshotValue);
        }
        let mut pixels = [0_u8; FRAMEBUFFER_SIZE];
        pixels.copy_from_slice(reader.bytes(FRAMEBUFFER_SIZE)?);
        if pixels.iter().any(|pixel| *pixel >= PALETTE_SIZE as u8) {
            return Err(CoreError::InvalidSnapshotValue);
        }
        let expected_pixels = logical_state.physical_screen.project(&render_state);
        if expected_pixels.pixels != pixels {
            return Err(CoreError::InvalidSnapshotValue);
        }
        if !reader.is_finished() {
            return Err(CoreError::InvalidSnapshotTrailingBytes);
        }
        Ok(Self {
            provenance: SnapshotProvenance {
                core_schema_version,
                cartridge_sha256,
            },
            logical_state,
            render_state,
            framebuffer: IndexedFramebuffer { pixels },
        })
    }
}

pub(crate) fn render_full_state_into(
    state: &FullState,
    render: &RenderState,
    framebuffer: &mut IndexedFramebuffer,
) {
    framebuffer.clear(BACKGROUND_COLOR, render);
    match state.lifecycle.mode {
        Mode::Menu => render_menu(state, render, framebuffer),
        Mode::TransitionToGame => {
            render_transition_base(state, render, framebuffer);
            framebuffer.rect_fill(
                0,
                i32::from(state.transition_render_y),
                (FRAMEBUFFER_WIDTH - 1) as i32,
                i32::from(state.transition_render_y) + (FRAMEBUFFER_HEIGHT - 1) as i32,
                ENTITY_COLOR,
                render,
            );
        }
        Mode::TransitionToSettings | Mode::TransitionToMenu => {
            render_transition_base(state, render, framebuffer);
            framebuffer.rect_fill(
                0,
                i32::from(state.transition_render_y),
                (FRAMEBUFFER_WIDTH - 1) as i32,
                i32::from(state.transition_render_y) + (FRAMEBUFFER_HEIGHT - 1) as i32,
                ENTITY_COLOR,
                render,
            );
        }
        Mode::Settings => render_settings(state, render, framebuffer),
        Mode::Game | Mode::Terminal => render_game(state, render, framebuffer),
    }
}

fn render_transition_base(
    state: &FullState,
    render: &RenderState,
    framebuffer: &mut IndexedFramebuffer,
) {
    let positive_side = state.transition_render_y > 0;
    let base_mode = if positive_side {
        match state.lifecycle.mode {
            Mode::TransitionToSettings | Mode::TransitionToMenu => {
                if state.lifecycle.mode == Mode::TransitionToSettings
                    && matches!(state.transition_from, Mode::Game | Mode::Terminal)
                {
                    Mode::Game
                } else {
                    Mode::Menu
                }
            }
            Mode::TransitionToGame => Mode::Game,
            _ => Mode::Menu,
        }
    } else {
        match state.lifecycle.mode {
            Mode::TransitionToSettings => Mode::Settings,
            Mode::TransitionToMenu => Mode::Settings,
            Mode::TransitionToGame => {
                if matches!(state.transition_from, Mode::Settings) {
                    Mode::Settings
                } else {
                    Mode::Menu
                }
            }
            _ => Mode::Menu,
        }
    };
    match base_mode {
        Mode::Menu => render_menu(state, render, framebuffer),
        Mode::Settings => render_settings(state, render, framebuffer),
        Mode::Game | Mode::Terminal => render_game(state, render, framebuffer),
        Mode::TransitionToGame | Mode::TransitionToSettings | Mode::TransitionToMenu => {}
    }
}

fn render_menu(state: &FullState, render: &RenderState, framebuffer: &mut IndexedFramebuffer) {
    framebuffer.sprite(17, 6, 18, 14, 7, render);
    let prompt = if state.input.current_mask() != 0 {
        "press ❎"
    } else {
        "press "
    };
    print2(framebuffer, prompt, 48, 58, render);
    if !state.input.source_input_mode() {
        framebuffer.sprite(192, 71, 56, 1, 2, render);
    }
    print2(framebuffer, "eddy rashed", 42, 96, render);
    print2(framebuffer, "oskar zanota", 40, 103, render);
}

fn render_settings(state: &FullState, render: &RenderState, framebuffer: &mut IndexedFramebuffer) {
    framebuffer.sprite(160, 38, 8, 8, 2, render);
    let categories = [("", 1_u8), ("gameplay", 3_u8)];
    let names = ["theme", "difficulty", "patterns", "powerups"];
    let values = [
        theme_name(state.settings.theme_index),
        difficulty_name(state.settings.difficulty),
        if state.settings.patterns_enabled {
            "on"
        } else {
            "off"
        },
        if state.settings.powerups_enabled {
            "on"
        } else {
            "off"
        },
    ];
    let editable_gameplay = state.lifecycle.dead || !state.has_played;
    let mut y = 0_i32;
    let mut setting_index = 0_u8;
    for (category, row_count) in categories {
        y += 20;
        let _ = row_count;
        print2(
            framebuffer,
            category,
            64 - category.len() as i32 * 2,
            y,
            render,
        );
        let count = if category.is_empty() { 1 } else { 3 };
        for _ in 0..count {
            y += 10;
            setting_index += 1;
            if state.input.source_input_mode() && state.settings.cursor == setting_index {
                framebuffer.sprite(64, 5, y, 1, 1, render);
            }
            let Some(name) = names.get(usize::from(setting_index - 1)).copied() else {
                continue;
            };
            let Some(value) = values.get(usize::from(setting_index - 1)).copied() else {
                continue;
            };
            print2(framebuffer, name, 20, y, render);
            print2(framebuffer, value, 91 - value.len() as i32 * 2, y, render);
            if (editable_gameplay || setting_index == 1) && !category.is_empty() {
                print2(framebuffer, "<           >", 65, y, render);
            }
            if category.is_empty() && setting_index == 1 {
                print2(framebuffer, "<           >", 65, y, render);
            }
        }
    }
    let highscore = current_highscore(state);
    if highscore > PicoFixed::ZERO {
        let text = format!("highscore: {}", ceil_fixed(highscore));
        print2(framebuffer, &text, 64 - text.len() as i32 * 2, 95, render);
    } else {
        print2(framebuffer, "no highscore", 40, 95, render);
    }
    if state.settings.message_timer > 0 {
        framebuffer.sprite(
            state.settings.message_sprite,
            i32::from(state.settings.message_x),
            i32::from(state.settings.message_y),
            1,
            1,
            render,
        );
    }
    let exit_text = if state.input.source_input_mode() {
        "❎ to exit"
    } else {
        "   to exit"
    };
    print2(framebuffer, exit_text, 1, 110, render);
    if !state.input.source_input_mode() {
        framebuffer.sprite(192, 2, 108, 1, 2, render);
    }
    let change_text = if state.input.source_input_mode() {
        "⬅➡ to change a setting"
    } else {
        "   to change a setting"
    };
    print2(framebuffer, change_text, 1, 120, render);
    if !state.input.source_input_mode() {
        framebuffer.sprite(193, 2, 118, 1, 2, render);
    }
}

fn theme_name(index: u8) -> &'static str {
    [
        "blue",
        "dark blue",
        "green",
        "indigo",
        "purple",
        "orange",
        "pink",
        "grey",
        "dark grey",
        "black",
        "neon red",
        "neon blue",
        "neon green",
    ]
    .get(usize::from(index.saturating_sub(1)))
    .copied()
    .unwrap_or("blue")
}

fn difficulty_name(index: u8) -> &'static str {
    match index {
        3 => "hard",
        1 => "easy",
        _ => "normal",
    }
}

fn current_highscore(state: &FullState) -> PicoFixed {
    let difficulty = state.settings.difficulty;
    let patterns = state.settings.patterns_enabled;
    let powerups = state.settings.powerups_enabled;
    let slot = (0..12).find(|index| {
        let category = index / 4 + 1;
        let slot = index % 4;
        category == usize::from(difficulty)
            && patterns == (slot == 0 || slot == 1)
            && powerups == (slot == 0 || slot == 3)
    });
    slot.and_then(|index| state.highscores.get(index).copied())
        .unwrap_or(PicoFixed::ZERO)
}

fn print2(framebuffer: &mut IndexedFramebuffer, text: &str, x: i32, y: i32, state: &RenderState) {
    print_text(framebuffer, text, x, y + 1, 1, state);
    print_text(framebuffer, text, x, y, 7, state);
}

fn print_text(
    framebuffer: &mut IndexedFramebuffer,
    text: &str,
    x: i32,
    y: i32,
    color: u8,
    state: &RenderState,
) {
    let mut cursor_x = x;
    for character in text.chars() {
        let glyph = glyph(character);
        for (row, bits) in glyph.rows.iter().enumerate() {
            for column in 0..glyph.width {
                if bits & (1 << (glyph.width - 1 - column)) != 0 {
                    framebuffer.plot_world(cursor_x + column, y + row as i32, color, state);
                }
            }
        }
        cursor_x += glyph.advance;
    }
}

#[derive(Clone, Copy)]
struct Glyph {
    rows: [u8; 5],
    width: i32,
    advance: i32,
}

const fn glyph_rows(rows: [u8; 5]) -> Glyph {
    Glyph {
        rows,
        width: 3,
        advance: 4,
    }
}

const fn wide_glyph(rows: [u8; 5]) -> Glyph {
    Glyph {
        rows,
        width: 8,
        advance: 8,
    }
}

fn glyph(character: char) -> Glyph {
    match character {
        'A' => glyph_rows([0, 7, 5, 7, 5]),
        'B' => glyph_rows([0, 6, 6, 5, 7]),
        'C' => glyph_rows([0, 7, 4, 4, 7]),
        'D' => glyph_rows([0, 6, 5, 5, 6]),
        'E' => glyph_rows([0, 7, 6, 4, 7]),
        'F' => glyph_rows([0, 7, 6, 4, 4]),
        'G' => glyph_rows([0, 7, 4, 5, 7]),
        'H' => glyph_rows([0, 5, 5, 7, 5]),
        'I' => glyph_rows([0, 7, 2, 2, 7]),
        'J' => glyph_rows([0, 7, 2, 2, 6]),
        'K' => glyph_rows([0, 5, 6, 5, 5]),
        'L' => glyph_rows([0, 4, 4, 4, 7]),
        'M' => glyph_rows([0, 7, 7, 5, 5]),
        'N' => glyph_rows([0, 6, 5, 5, 5]),
        'O' => glyph_rows([0, 3, 5, 5, 6]),
        'P' => glyph_rows([0, 7, 5, 7, 4]),
        'Q' => glyph_rows([0, 2, 5, 6, 3]),
        'R' => glyph_rows([0, 7, 5, 6, 5]),
        'S' => glyph_rows([0, 3, 4, 1, 6]),
        'T' => glyph_rows([0, 7, 2, 2, 2]),
        'U' => glyph_rows([0, 5, 5, 5, 3]),
        'V' => glyph_rows([0, 5, 5, 7, 2]),
        'W' => glyph_rows([0, 5, 5, 7, 7]),
        'X' => glyph_rows([0, 5, 2, 5, 5]),
        'Y' => glyph_rows([0, 5, 7, 1, 7]),
        'Z' => glyph_rows([0, 7, 1, 4, 7]),
        'a' => glyph_rows([7, 5, 7, 5, 5]),
        'b' => glyph_rows([7, 5, 6, 5, 7]),
        'c' => glyph_rows([3, 4, 4, 4, 3]),
        'd' => glyph_rows([6, 5, 5, 5, 7]),
        'e' => glyph_rows([7, 4, 6, 4, 7]),
        'f' => glyph_rows([7, 4, 6, 4, 4]),
        'g' => glyph_rows([3, 4, 4, 5, 7]),
        'h' => glyph_rows([5, 5, 7, 5, 5]),
        'i' => glyph_rows([7, 2, 2, 2, 7]),
        'j' => glyph_rows([7, 2, 2, 2, 6]),
        'k' => glyph_rows([5, 5, 6, 5, 5]),
        'l' => glyph_rows([4, 4, 4, 4, 7]),
        'm' => glyph_rows([7, 7, 5, 5, 5]),
        'n' => glyph_rows([6, 5, 5, 5, 5]),
        'o' => glyph_rows([3, 5, 5, 5, 6]),
        'p' => glyph_rows([7, 5, 7, 4, 4]),
        'q' => glyph_rows([2, 5, 5, 6, 3]),
        'r' => glyph_rows([7, 5, 6, 5, 5]),
        's' => glyph_rows([3, 4, 7, 1, 6]),
        't' => glyph_rows([7, 2, 2, 2, 2]),
        'u' => glyph_rows([5, 5, 5, 5, 3]),
        'v' => glyph_rows([5, 5, 5, 5, 2]),
        'w' => glyph_rows([5, 5, 5, 7, 7]),
        'x' => glyph_rows([5, 5, 2, 5, 5]),
        'y' => glyph_rows([5, 5, 7, 1, 7]),
        'z' => glyph_rows([7, 1, 2, 4, 7]),
        ' ' => glyph_rows([0; 5]),
        '0' => glyph_rows([7, 5, 5, 5, 7]),
        '1' => glyph_rows([6, 2, 2, 2, 7]),
        '2' => glyph_rows([7, 1, 7, 4, 7]),
        '3' => glyph_rows([7, 1, 3, 1, 7]),
        '4' => glyph_rows([5, 5, 7, 1, 1]),
        '5' => glyph_rows([7, 4, 7, 1, 7]),
        '6' => glyph_rows([4, 4, 7, 5, 7]),
        '7' => glyph_rows([7, 1, 1, 1, 1]),
        '8' => glyph_rows([7, 5, 7, 5, 7]),
        '9' => glyph_rows([7, 5, 7, 1, 1]),
        ':' => glyph_rows([0, 2, 0, 2, 0]),
        ';' => glyph_rows([0, 2, 0, 2, 4]),
        ',' => glyph_rows([0, 0, 0, 2, 4]),
        '.' => glyph_rows([0, 0, 0, 0, 2]),
        '!' => glyph_rows([2, 2, 2, 0, 2]),
        '?' => glyph_rows([7, 1, 3, 0, 2]),
        '+' => glyph_rows([0, 2, 7, 2, 0]),
        '-' => glyph_rows([0, 0, 7, 0, 0]),
        '>' => glyph_rows([4, 2, 1, 2, 4]),
        '<' => glyph_rows([1, 2, 4, 2, 1]),
        '(' => glyph_rows([2, 4, 4, 4, 2]),
        ')' => glyph_rows([2, 1, 1, 1, 2]),
        '❎' => wide_glyph([
            0b0111_1100,
            0b1101_0110,
            0b1110_1110,
            0b1101_0110,
            0b0111_1100,
        ]),
        '🅾' => wide_glyph([
            0b0111_1100,
            0b1100_0110,
            0b1101_0110,
            0b1100_0110,
            0b0111_1100,
        ]),
        '⬅' => wide_glyph([
            0b0111_1100,
            0b1110_0110,
            0b1100_0110,
            0b1110_0110,
            0b0111_1100,
        ]),
        '➡' => wide_glyph([
            0b0111_1100,
            0b1100_1110,
            0b1100_0110,
            0b1100_1110,
            0b0111_1100,
        ]),
        '⬆' => wide_glyph([
            0b0111_1100,
            0b1110_1110,
            0b1100_0110,
            0b1100_0110,
            0b0111_1100,
        ]),
        '⬇' => wide_glyph([
            0b0111_1100,
            0b1100_0110,
            0b1100_0110,
            0b1110_1110,
            0b0111_1100,
        ]),
        _ => glyph_rows([0; 5]),
    }
}

fn map_color(color: u8, state: &RenderState) -> Option<u8> {
    state.palette_index(color)
}

fn render_game(state: &FullState, render: &RenderState, framebuffer: &mut IndexedFramebuffer) {
    if state.lifecycle.dead {
        render_game_over(state, render, framebuffer);
        return;
    }
    for particle in &state.particles {
        let x = coordinate(particle.x);
        let y = coordinate(particle.y);
        if particle.kind == 0 {
            framebuffer.circle_fill(x, y + 1, particle.radius, SHADOW_COLOR, render);
        } else {
            framebuffer.plot_world(x, y + 1, SHADOW_COLOR, render);
            framebuffer.plot_world(x, y, particle.color, render);
        }
    }
    for particle in &state.particles {
        if particle.kind == 0 {
            framebuffer.circle_fill(
                coordinate(particle.x),
                coordinate(particle.y),
                particle.radius,
                ENTITY_COLOR,
                render,
            );
        }
    }
    for enemy in &state.enemies {
        let x = coordinate(enemy.x);
        let y = coordinate(enemy.y);
        if enemy.personality >= 2 {
            framebuffer.circle_fill(x, y + 1, PicoFixed::from_int(4), SHADOW_COLOR, render);
            framebuffer.circle_fill(x, y, PicoFixed::from_int(4), ENTITY_COLOR, render);
        } else {
            let size = coordinate(enemy.size).max(0);
            if enemy.personality == 0 {
                framebuffer.rect_fill(x, y + 1, x + size, y + size + 1, SHADOW_COLOR, render);
                framebuffer.rect_fill(x, y, x + size, y + size, ENTITY_COLOR, render);
            } else {
                framebuffer.rect_outline(x, y + 1, x + size, y + size + 1, SHADOW_COLOR, render);
                framebuffer.rect_outline(x, y, x + size, y + size, ENTITY_COLOR, render);
            }
        }
    }
    if let Some(pattern_index) = state.active_pattern {
        if let Some(pattern) = state.patterns.get(pattern_index) {
            render_pattern(pattern, render, framebuffer);
        }
    }
    let score = ceil_fixed(state.score);
    let score_text = score.to_string();
    let score_width = score_text.len() as i32 * 4;
    framebuffer.rect_fill(0, 0, score_width, 7, BACKGROUND_COLOR, render);
    print2(framebuffer, &score_text, 1, 1, render);
}

fn render_game_over(state: &FullState, render: &RenderState, framebuffer: &mut IndexedFramebuffer) {
    let button_active = state.input.source_input_mode();
    let settings_prompt = if button_active {
        "press 🅾 to open settings"
    } else {
        "press   to open settings"
    };
    let replay_prompt = if button_active {
        "press ❎ to play again"
    } else {
        "press   to play again"
    };
    print2(framebuffer, settings_prompt, 14, 110, render);
    if !button_active {
        framebuffer.sprite(192, 44, 118, 1, 2, render);
    }
    print2(framebuffer, replay_prompt, 20, 120, render);
    if !button_active {
        framebuffer.sprite(193, 38, 108, 1, 2, render);
    }
    framebuffer.sprite(128, 34, 40, 8, 2, render);
    let score_text = format!("score:{}", ceil_fixed(state.score));
    let x = 64 - score_text.len() as i32 * 2;
    print2(framebuffer, &score_text, x, 56, render);
    if state.new_highscore {
        render_wave(state, render, framebuffer);
    }
}

fn render_wave(state: &FullState, render: &RenderState, framebuffer: &mut IndexedFramebuffer) {
    // Pemsa's deterministic headless clock reports time just after the
    // completed frame: frame 501 is 8.3666 seconds, or (501 + 1) / 60.
    let time = PicoFixed::from_int(state.lifecycle.frame as i32)
        .add(PicoFixed::ONE)
        .div_fixed(PicoFixed::from_int(60))
        .unwrap_or(PicoFixed::ZERO);
    let phase = time.mul_fixed(PicoFixed::from_int(30));
    let mut x = 36_i32;
    for character in "new highscore!".chars() {
        let angle = PicoFixed::from_int(x)
            .add(phase)
            .div_fixed(PicoFixed::from_int(25))
            .unwrap_or(PicoFixed::ZERO);
        let offset = source_sine(angle).mul_fixed(PicoFixed::from_int(2));
        let y = coordinate(PicoFixed::from_int(79).add(offset));
        print2(framebuffer, &character.to_string(), x, y, render);
        x += 4;
    }
}

fn source_sine(angle: PicoFixed) -> PicoFixed {
    let radians = f64::from(angle.to_f32()) * std::f64::consts::TAU;
    PicoFixed::from_f32(-(radians.sin() as f32))
}

fn render_pattern(
    pattern: &PatternState,
    render: &RenderState,
    framebuffer: &mut IndexedFramebuffer,
) {
    for rect in &pattern.rects {
        let x = rounded_coordinate(rect.x);
        let y = rounded_coordinate(rect.y);
        let width = rounded_coordinate(rect.width);
        let height = rounded_coordinate(rect.height);
        let xw = x + width;
        let yh = y + height;
        // Lua evaluates `x + w / 2` and `y + h / 2` before `line()` gets
        // integer raster coordinates. Keep the half-pixel for odd-sized
        // rectangles instead of truncating it with integer division.
        let half_width = PicoFixed::from_int(width)
            .div_fixed(PicoFixed::from_int(2))
            .unwrap_or(PicoFixed::ZERO);
        let half_height = PicoFixed::from_int(height)
            .div_fixed(PicoFixed::from_int(2))
            .unwrap_or(PicoFixed::ZERO);
        let center_x = PicoFixed::from_int(x).add(half_width);
        let center_y = PicoFixed::from_int(y).add(half_height);
        if rect.sh < PicoFixed::from_int(2) && pattern.pattern_type == 0 {
            let mut shrink = rect.sh.min(PicoFixed::ONE);
            let mut dotted = true;
            if rect.sh > PicoFixed::ONE {
                let shadow = patterned_render_state(render, fill_pattern_dot(x, y));
                framebuffer.rect_outline(x, y + 1, xw, yh + 1, SHADOW_COLOR, &shadow);
                let actual = patterned_render_state(render, fill_pattern_dot(x, y + 1));
                framebuffer.rect_outline(x, y, xw, yh, ENTITY_COLOR, &actual);
                shrink = rect.sh.sub(PicoFixed::ONE);
                dotted = false;
            }
            line2(
                framebuffer,
                x,
                y,
                lerp_fixed(x, center_x, shrink),
                y,
                dotted,
                render,
            );
            line2(
                framebuffer,
                xw,
                y,
                lerp_fixed(xw, center_x, shrink),
                y,
                dotted,
                render,
            );
            line2(
                framebuffer,
                xw,
                y,
                x + width,
                lerp_fixed(y, center_y, shrink),
                dotted,
                render,
            );
            line2(
                framebuffer,
                xw,
                yh,
                xw,
                lerp_fixed(yh, center_y, shrink),
                dotted,
                render,
            );
            line2(
                framebuffer,
                x,
                yh,
                lerp_fixed(x, center_x, shrink),
                y + height,
                dotted,
                render,
            );
            line2(
                framebuffer,
                xw,
                yh,
                lerp_fixed(xw, center_x, shrink),
                y + height,
                dotted,
                render,
            );
            line2(
                framebuffer,
                x,
                y,
                x,
                lerp_fixed(y, center_y, shrink),
                dotted,
                render,
            );
            line2(
                framebuffer,
                x,
                yh,
                x,
                lerp_fixed(yh, center_y, shrink),
                dotted,
                render,
            );
        } else {
            if pattern.timer < PicoFixed::from_int(125) {
                for warning in &rect.warnings {
                    line2(
                        framebuffer,
                        rounded_coordinate(warning.x0),
                        rounded_coordinate(warning.y0),
                        rounded_coordinate(warning.x1),
                        rounded_coordinate(warning.y1),
                        false,
                        render,
                    );
                }
            }
            framebuffer.rect_fill(x, y + 1, xw, yh + 1, SHADOW_COLOR, render);
            framebuffer.rect_fill(x, y, xw, yh, ENTITY_COLOR, render);
        }
    }
}

fn line2(
    framebuffer: &mut IndexedFramebuffer,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    dotted: bool,
    render: &RenderState,
) {
    let shadow = if dotted {
        patterned_render_state(render, fill_pattern_dot(x0, y0 + 1))
    } else {
        render.clone()
    };
    framebuffer.rect_line(x0, y0 + 1, x1, y1 + 1, SHADOW_COLOR, &shadow);
    let actual = if dotted {
        patterned_render_state(render, fill_pattern_dot(x0, y0))
    } else {
        render.clone()
    };
    framebuffer.rect_line(x0, y0, x1, y1, ENTITY_COLOR, &actual);
}

fn fill_pattern_dot(x: i32, y: i32) -> u16 {
    if (x & 1) == (y & 1) {
        0x5a5a
    } else {
        0xa5a5
    }
}

fn patterned_render_state(render: &RenderState, fill_pattern: u16) -> RenderState {
    let mut patterned = render.clone();
    patterned.fill_pattern = fill_pattern;
    patterned
}

fn lerp_fixed(position: i32, target: PicoFixed, percentage: PicoFixed) -> i32 {
    let start = PicoFixed::from_int(position);
    coordinate(
        PicoFixed::ONE
            .sub(percentage)
            .mul_fixed(start)
            .add(percentage.mul_fixed(target)),
    )
}

fn circle_scanlines(radius: usize) -> &'static [u8] {
    match radius {
        0 => &[0],
        1 => &[0, 1, 0],
        2 => &[1, 2, 2, 2, 1],
        3 => &[1, 2, 3, 3, 3, 2, 1],
        4 => &[1, 3, 3, 4, 4, 4, 3, 3, 1],
        5 => &[2, 3, 4, 5, 5, 5, 5, 4, 3, 2],
        6 => &[2, 3, 4, 5, 6, 6, 6, 6, 5, 4, 3, 2],
        7 => &[2, 4, 5, 6, 7, 7, 7, 7, 7, 6, 5, 4, 2],
        _ => &[3, 5, 6, 7, 8, 8, 8, 8, 8, 7, 6, 5, 3],
    }
}

fn ceil_fixed(value: PicoFixed) -> i32 {
    let raw = value.raw();
    if raw <= 0 {
        return raw >> 16;
    }
    (raw + 0xffff) >> 16
}

fn coordinate(value: PicoFixed) -> i32 {
    value.raw() >> 16
}

fn rounded_coordinate(value: PicoFixed) -> i32 {
    value.round().raw() >> 16
}

fn camera_offset(value: PicoFixed) -> i32 {
    let raw = i64::from(value.raw());
    if raw >= 0 {
        ((raw + (1 << 15)) / (1 << 16)) as i32
    } else {
        -(((-raw + (1 << 15)) / (1 << 16)) as i32)
    }
}

fn pattern_bit(pattern: u16, x: i32, y: i32) -> bool {
    let bit = 15 - (((y & 3) << 2) | (x & 3));
    pattern & (1_u16 << bit) == 0
}

fn mode_tag(mode: Mode) -> u8 {
    match mode {
        Mode::Menu => 0,
        Mode::TransitionToGame => 1,
        Mode::Game => 2,
        Mode::Terminal => 3,
        Mode::Settings => 4,
        Mode::TransitionToSettings => 5,
        Mode::TransitionToMenu => 6,
    }
}

fn mode_from_tag(tag: u8) -> Result<Mode, CoreError> {
    match tag {
        0 => Ok(Mode::Menu),
        1 => Ok(Mode::TransitionToGame),
        2 => Ok(Mode::Game),
        3 => Ok(Mode::Terminal),
        4 => Ok(Mode::Settings),
        5 => Ok(Mode::TransitionToSettings),
        6 => Ok(Mode::TransitionToMenu),
        _ => Err(CoreError::InvalidSnapshotValue),
    }
}

fn write_full_state(writer: &mut Writer, state: &FullState) {
    writer.u32(state.seed);
    writer.u32(state.lifecycle.frame);
    writer.u8(mode_tag(state.lifecycle.mode));
    writer.i16(state.lifecycle.transition_y);
    writer.bool(state.lifecycle.started);
    writer.bool(state.lifecycle.game_ready);
    writer.bool(state.lifecycle.dead);
    writer.u8(state.input.current_mask());
    writer.u8(state.input.previous_mask());
    writer.bool(state.input.source_input_mode());
    writer.u32(state.rng.seed);
    for value in state.rng.state {
        writer.u32(value);
    }
    writer.u8(state.rng.front);
    writer.u8(state.rng.rear);
    write_player_state(writer, state.player);
    writer.u32(state.enemies.len() as u32);
    for enemy in &state.enemies {
        write_enemy_state(writer, *enemy);
    }
    writer.u32(state.particles.len() as u32);
    for particle in &state.particles {
        write_particle_state(writer, *particle);
    }
    writer.u32(state.patterns.len() as u32);
    for pattern in &state.patterns {
        write_pattern_state(writer, pattern);
    }
    match state.active_pattern {
        Some(index) => {
            writer.bool(true);
            writer.u32(index as u32);
        }
        None => writer.bool(false),
    }
    writer.u32(state.spawns.len() as u32);
    for spawn in &state.spawns {
        writer.i32(spawn.x.raw());
        writer.i32(spawn.y.raw());
    }
    writer.i32(state.enemy_timer.raw());
    writer.i32(state.enemy_est.raw());
    for value in state.enemy_stats {
        writer.i32(value.raw());
    }
    writer.u32(state.friendly_timer);
    writer.bool(state.friendly_enabled);
    writer.i32(state.enemy_max_size.raw());
    writer.i32(state.speed.raw());
    writer.i32(state.freeze_rate.raw());
    writer.bool(state.freeze_active);
    writer.u32(state.freeze_timer);
    writer.i32(state.size_timer.raw());
    writer.bool(state.patterns_enabled);
    writer.bool(state.powerups_enabled);
    writer.u32(state.pattern_timer);
    writer.u32(state.pattern_delay_frames);
    writer.bool(state.pattern_active);
    writer.bool(state.new_highscore);
    writer.bool(state.can_click);
    writer.bool(state.has_played);
    writer.bool(state.should_collide);
    writer.bool(state.enemy_should_collide);
    writer.i32(state.bounce_cap_static.raw());
    writer.i32(state.bounce_cap_moving.raw());
    writer.i32(state.bounce_cap.raw());
    writer.i32(state.score.raw());
    writer.u32(state.survival_frames);
    writer.u32(state.shattered);
    writer.i32(state.shake.raw());
    writer.i32(state.camera_x.raw());
    writer.i32(state.camera_y.raw());
    writer.i16(state.transition_render_y);
    writer.u8(mode_tag(state.transition_from));
    writer.u8(state.settings.theme_index);
    writer.u8(state.settings.theme_background);
    writer.u8(state.settings.theme_shadow);
    writer.u8(state.settings.difficulty);
    writer.bool(state.settings.patterns_enabled);
    writer.bool(state.settings.powerups_enabled);
    writer.u8(state.settings.cursor);
    writer.u8(state.settings.message_timer);
    writer.u8(state.settings.message_sprite);
    writer.i16(state.settings.message_x);
    writer.i16(state.settings.message_y);
    for highscore in state.highscores {
        writer.i32(highscore.raw());
    }
    writer.bytes(state.physical_screen.as_bytes());
}

fn read_full_state(reader: &mut Reader<'_>) -> Result<FullState, CoreError> {
    let seed = reader.u32()?;
    let lifecycle = LifecycleState {
        frame: reader.u32()?,
        mode: mode_from_tag(reader.u8()?)?,
        transition_y: reader.i16()?,
        started: reader.bool()?,
        game_ready: reader.bool()?,
        dead: reader.bool()?,
    };
    let input = InputState::from_wire(reader.u8()?, reader.u8()?, reader.bool()?)?;
    let rng = read_rng(reader)?;
    let player = read_player_state(reader)?;
    let enemy_count = reader.u32()?;
    if enemy_count > MAX_SNAPSHOT_ENEMIES {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut enemies = Vec::with_capacity(enemy_count as usize);
    for _ in 0..enemy_count {
        enemies.push(read_enemy_state(reader)?);
    }
    let particle_count = reader.u32()?;
    if particle_count > MAX_SNAPSHOT_PARTICLES {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut particles = Vec::with_capacity(particle_count as usize);
    for _ in 0..particle_count {
        particles.push(read_particle_state(reader)?);
    }
    let pattern_count = reader.u32()?;
    if pattern_count > MAX_SNAPSHOT_PATTERNS {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut patterns = Vec::with_capacity(pattern_count as usize);
    for _ in 0..pattern_count {
        patterns.push(read_pattern_state(reader)?);
    }
    let active_pattern = if reader.bool()? {
        let index = reader.u32()? as usize;
        if index >= patterns.len() {
            return Err(CoreError::InvalidSnapshotValue);
        }
        Some(index)
    } else {
        None
    };
    let spawn_count = reader.u32()?;
    if spawn_count > MAX_SNAPSHOT_SPAWNS {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut spawns = Vec::with_capacity(spawn_count as usize);
    for _ in 0..spawn_count {
        spawns.push(SpawnPoint {
            x: PicoFixed::from_raw(reader.i32()?),
            y: PicoFixed::from_raw(reader.i32()?),
        });
    }
    let enemy_timer = PicoFixed::from_raw(reader.i32()?);
    let enemy_est = PicoFixed::from_raw(reader.i32()?);
    let mut enemy_stats = [PicoFixed::ZERO; 5];
    for value in &mut enemy_stats {
        *value = PicoFixed::from_raw(reader.i32()?);
    }
    let friendly_timer = reader.u32()?;
    let friendly_enabled = reader.bool()?;
    let enemy_max_size = PicoFixed::from_raw(reader.i32()?);
    let speed = PicoFixed::from_raw(reader.i32()?);
    let freeze_rate = PicoFixed::from_raw(reader.i32()?);
    let freeze_active = reader.bool()?;
    let freeze_timer = reader.u32()?;
    let size_timer = PicoFixed::from_raw(reader.i32()?);
    let patterns_enabled = reader.bool()?;
    let powerups_enabled = reader.bool()?;
    let pattern_timer = reader.u32()?;
    let pattern_delay_frames = reader.u32()?;
    let pattern_active = reader.bool()?;
    let new_highscore = reader.bool()?;
    let can_click = reader.bool()?;
    let has_played = reader.bool()?;
    let should_collide = reader.bool()?;
    let enemy_should_collide = reader.bool()?;
    let bounce_cap_static = PicoFixed::from_raw(reader.i32()?);
    let bounce_cap_moving = PicoFixed::from_raw(reader.i32()?);
    let bounce_cap = PicoFixed::from_raw(reader.i32()?);
    let score = PicoFixed::from_raw(reader.i32()?);
    let survival_frames = reader.u32()?;
    let shattered = reader.u32()?;
    let shake = PicoFixed::from_raw(reader.i32()?);
    let camera_x = PicoFixed::from_raw(reader.i32()?);
    let camera_y = PicoFixed::from_raw(reader.i32()?);
    let transition_render_y = reader.i16()?;
    let transition_from = mode_from_tag(reader.u8()?)?;
    let settings = SettingsState {
        theme_index: reader.u8()?,
        theme_background: reader.u8()?,
        theme_shadow: reader.u8()?,
        difficulty: reader.u8()?,
        patterns_enabled: reader.bool()?,
        powerups_enabled: reader.bool()?,
        cursor: reader.u8()?,
        message_timer: reader.u8()?,
        message_sprite: reader.u8()?,
        message_x: reader.i16()?,
        message_y: reader.i16()?,
    };
    let mut highscores = [PicoFixed::ZERO; 12];
    for highscore in &mut highscores {
        *highscore = PicoFixed::from_raw(reader.i32()?);
    }
    let mut physical_pixels = [0_u8; FRAMEBUFFER_SIZE];
    physical_pixels.copy_from_slice(reader.bytes(FRAMEBUFFER_SIZE)?);
    if physical_pixels
        .iter()
        .any(|pixel| *pixel >= PALETTE_SIZE as u8)
    {
        return Err(CoreError::InvalidSnapshotValue);
    }
    Ok(FullState {
        seed,
        lifecycle,
        input,
        rng,
        player,
        enemies,
        particles,
        patterns,
        active_pattern,
        spawns,
        physical_screen: IndexedFramebuffer {
            pixels: physical_pixels,
        },
        enemy_timer,
        enemy_est,
        enemy_stats,
        friendly_timer,
        friendly_enabled,
        enemy_max_size,
        speed,
        freeze_rate,
        freeze_active,
        freeze_timer,
        size_timer,
        patterns_enabled,
        powerups_enabled,
        pattern_timer,
        pattern_delay_frames,
        pattern_active,
        new_highscore,
        can_click,
        has_played,
        should_collide,
        enemy_should_collide,
        bounce_cap_static,
        bounce_cap_moving,
        bounce_cap,
        score,
        survival_frames,
        shattered,
        shake,
        camera_x,
        camera_y,
        transition_render_y,
        transition_from,
        settings,
        highscores,
    })
}

fn write_player_state(writer: &mut Writer, player: PlayerState) {
    writer.i32(player.x.raw());
    writer.i32(player.y.raw());
    writer.i32(player.vx.raw());
    writer.i32(player.vy.raw());
    writer.i32(player.size.raw());
}

fn read_player_state(reader: &mut Reader<'_>) -> Result<PlayerState, CoreError> {
    Ok(PlayerState {
        x: PicoFixed::from_raw(reader.i32()?),
        y: PicoFixed::from_raw(reader.i32()?),
        vx: PicoFixed::from_raw(reader.i32()?),
        vy: PicoFixed::from_raw(reader.i32()?),
        size: PicoFixed::from_raw(reader.i32()?),
    })
}

fn write_enemy_state(writer: &mut Writer, enemy: EnemyState) {
    writer.i32(enemy.x.raw());
    writer.i32(enemy.y.raw());
    writer.i32(enemy.vx.raw());
    writer.i32(enemy.vy.raw());
    writer.i32(enemy.size.raw());
    writer.i32(enemy.max_size.raw());
    writer.i8(enemy.personality);
    writer.i32(enemy.speed.raw());
    writer.bool(enemy.inside);
    writer.bool(enemy.is_dying);
    writer.bool(enemy.isizing);
    match enemy.life {
        Some(life) => {
            writer.bool(true);
            writer.i32(life.raw());
        }
        None => writer.bool(false),
    }
}

fn write_particle_state(writer: &mut Writer, particle: ParticleState) {
    writer.i32(particle.x.raw());
    writer.i32(particle.y.raw());
    writer.i32(particle.dx.raw());
    writer.i32(particle.dy.raw());
    writer.i32(particle.radius.raw());
    writer.i8(particle.kind);
    writer.i32(particle.max_age.raw());
    writer.u32(particle.age);
    writer.u8(particle.color);
    writer.bytes(&particle.colors);
    writer.u8(particle.color_count);
}

fn write_pattern_state(writer: &mut Writer, pattern: &PatternState) {
    writer.u8(pattern.id);
    writer.i32(pattern.mins.raw());
    writer.i32(pattern.maxs.raw());
    writer.i32(pattern.probability.raw());
    writer.u32(pattern.variants.len() as u32);
    for variant in &pattern.variants {
        writer.u8(*variant);
    }
    writer.bool(pattern.smooth);
    writer.u8(pattern.pattern_type);
    writer.bool(pattern.bounce_cap);
    writer.bool(pattern.spawn_enabled);
    match pattern.automatic_variant {
        Some(variant) => {
            writer.bool(true);
            writer.u8(variant);
        }
        None => writer.bool(false),
    }
    writer.i32(pattern.special.raw());
    writer.u32(pattern.counter);
    writer.i32(pattern.timer.raw());
    writer.u32(pattern.rects.len() as u32);
    for rect in &pattern.rects {
        write_pattern_rect(writer, rect);
    }
}

fn write_pattern_rect(writer: &mut Writer, rect: &PatternRect) {
    writer.i32(rect.x.raw());
    writer.i32(rect.y.raw());
    writer.i32(rect.width.raw());
    writer.i32(rect.height.raw());
    writer.i32(rect.speed.raw());
    writer.i32(rect.dx.raw());
    writer.i32(rect.dy.raw());
    writer.u32(rect.targets.len() as u32);
    for target in &rect.targets {
        match target {
            PatternTarget::Move {
                x,
                y,
                width,
                height,
            } => {
                writer.u8(0);
                writer.i32(x.raw());
                writer.i32(y.raw());
                writer.i32(width.raw());
                writer.i32(height.raw());
            }
            PatternTarget::Wait(seconds) => {
                writer.u8(1);
                writer.i32(seconds.raw());
            }
            PatternTarget::SetFyou(value) => {
                writer.u8(2);
                writer.bool(*value);
            }
            PatternTarget::SetSpawns(points) => {
                writer.u8(3);
                writer.u32(points.len() as u32);
                for point in points {
                    writer.i32(point.x.raw());
                    writer.i32(point.y.raw());
                }
            }
        }
    }
    writer.u32(rect.target_index as u32);
    writer.i32(rect.wait.raw());
    writer.bool(rect.shown);
    writer.i32(rect.sh.raw());
    writer.u32(rect.warnings.len() as u32);
    for warning in &rect.warnings {
        writer.i32(warning.x0.raw());
        writer.i32(warning.y0.raw());
        writer.i32(warning.x1.raw());
        writer.i32(warning.y1.raw());
    }
    writer.bool(rect.collision_done);
    writer.bool(rect.finished);
}

fn read_particle_state(reader: &mut Reader<'_>) -> Result<ParticleState, CoreError> {
    let x = PicoFixed::from_raw(reader.i32()?);
    let y = PicoFixed::from_raw(reader.i32()?);
    let dx = PicoFixed::from_raw(reader.i32()?);
    let dy = PicoFixed::from_raw(reader.i32()?);
    let radius = PicoFixed::from_raw(reader.i32()?);
    let kind = reader.i8()?;
    let max_age = PicoFixed::from_raw(reader.i32()?);
    let age = reader.u32()?;
    let color = reader.u8()?;
    let mut colors = [0_u8; 3];
    colors.copy_from_slice(reader.bytes(3)?);
    let color_count = reader.u8()?;
    if color_count == 0
        || color_count > 3
        || colors
            .get(..usize::from(color_count))
            .is_none_or(|values| values.iter().any(|value| *value >= PALETTE_SIZE as u8))
    {
        return Err(CoreError::InvalidSnapshotValue);
    }
    Ok(ParticleState {
        x,
        y,
        dx,
        dy,
        radius,
        kind,
        max_age,
        age,
        color,
        colors,
        color_count,
    })
}

fn read_pattern_state(reader: &mut Reader<'_>) -> Result<PatternState, CoreError> {
    let id = reader.u8()?;
    let mins = PicoFixed::from_raw(reader.i32()?);
    let maxs = PicoFixed::from_raw(reader.i32()?);
    let probability = PicoFixed::from_raw(reader.i32()?);
    let variant_count = reader.u32()?;
    if variant_count > MAX_SNAPSHOT_PATTERN_TARGETS {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut variants = Vec::with_capacity(variant_count as usize);
    for _ in 0..variant_count {
        variants.push(reader.u8()?);
    }
    let smooth = reader.bool()?;
    let pattern_type = reader.u8()?;
    let bounce_cap = reader.bool()?;
    let spawn_enabled = reader.bool()?;
    let automatic_variant = if reader.bool()? {
        Some(reader.u8()?)
    } else {
        None
    };
    let special = PicoFixed::from_raw(reader.i32()?);
    let counter = reader.u32()?;
    let timer = PicoFixed::from_raw(reader.i32()?);
    let rect_count = reader.u32()?;
    if rect_count > MAX_SNAPSHOT_PATTERN_RECTS {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut rects = Vec::with_capacity(rect_count as usize);
    for _ in 0..rect_count {
        rects.push(read_pattern_rect(reader)?);
    }
    Ok(PatternState {
        id,
        mins,
        maxs,
        probability,
        variants,
        smooth,
        pattern_type,
        bounce_cap,
        spawn_enabled,
        automatic_variant,
        special,
        counter,
        timer,
        rects,
    })
}

fn read_pattern_rect(reader: &mut Reader<'_>) -> Result<PatternRect, CoreError> {
    let x = PicoFixed::from_raw(reader.i32()?);
    let y = PicoFixed::from_raw(reader.i32()?);
    let width = PicoFixed::from_raw(reader.i32()?);
    let height = PicoFixed::from_raw(reader.i32()?);
    let speed = PicoFixed::from_raw(reader.i32()?);
    let dx = PicoFixed::from_raw(reader.i32()?);
    let dy = PicoFixed::from_raw(reader.i32()?);
    let target_count = reader.u32()?;
    if target_count > MAX_SNAPSHOT_PATTERN_TARGETS {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut targets = Vec::with_capacity(target_count as usize);
    for _ in 0..target_count {
        let tag = reader.u8()?;
        targets.push(match tag {
            0 => PatternTarget::Move {
                x: PicoFixed::from_raw(reader.i32()?),
                y: PicoFixed::from_raw(reader.i32()?),
                width: PicoFixed::from_raw(reader.i32()?),
                height: PicoFixed::from_raw(reader.i32()?),
            },
            1 => PatternTarget::Wait(PicoFixed::from_raw(reader.i32()?)),
            2 => PatternTarget::SetFyou(reader.bool()?),
            3 => {
                let point_count = reader.u32()?;
                if point_count > MAX_SNAPSHOT_PATTERN_POINTS {
                    return Err(CoreError::InvalidSnapshotValue);
                }
                let mut points = Vec::with_capacity(point_count as usize);
                for _ in 0..point_count {
                    points.push(SpawnPoint {
                        x: PicoFixed::from_raw(reader.i32()?),
                        y: PicoFixed::from_raw(reader.i32()?),
                    });
                }
                PatternTarget::SetSpawns(points)
            }
            _ => return Err(CoreError::InvalidSnapshotValue),
        });
    }
    let target_index = reader.u32()? as usize;
    if target_index > targets.len() {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let wait = PicoFixed::from_raw(reader.i32()?);
    let shown = reader.bool()?;
    let sh = PicoFixed::from_raw(reader.i32()?);
    let warning_count = reader.u32()?;
    if warning_count > MAX_SNAPSHOT_PATTERN_WARNINGS {
        return Err(CoreError::InvalidSnapshotValue);
    }
    let mut warnings = Vec::with_capacity(warning_count as usize);
    for _ in 0..warning_count {
        warnings.push(WarningLine {
            x0: PicoFixed::from_raw(reader.i32()?),
            y0: PicoFixed::from_raw(reader.i32()?),
            x1: PicoFixed::from_raw(reader.i32()?),
            y1: PicoFixed::from_raw(reader.i32()?),
        });
    }
    Ok(PatternRect {
        x,
        y,
        width,
        height,
        speed,
        dx,
        dy,
        targets,
        target_index,
        wait,
        shown,
        sh,
        warnings,
        collision_done: reader.bool()?,
        finished: reader.bool()?,
    })
}

fn read_enemy_state(reader: &mut Reader<'_>) -> Result<EnemyState, CoreError> {
    Ok(EnemyState {
        x: PicoFixed::from_raw(reader.i32()?),
        y: PicoFixed::from_raw(reader.i32()?),
        vx: PicoFixed::from_raw(reader.i32()?),
        vy: PicoFixed::from_raw(reader.i32()?),
        size: PicoFixed::from_raw(reader.i32()?),
        max_size: PicoFixed::from_raw(reader.i32()?),
        personality: reader.i8()?,
        speed: PicoFixed::from_raw(reader.i32()?),
        inside: reader.bool()?,
        is_dying: reader.bool()?,
        isizing: reader.bool()?,
        life: if reader.bool()? {
            Some(PicoFixed::from_raw(reader.i32()?))
        } else {
            None
        },
    })
}

fn read_rng(reader: &mut Reader<'_>) -> Result<RngCheckpoint, CoreError> {
    let seed = reader.u32()?;
    let mut state = [0_u32; 31];
    for value in &mut state {
        *value = reader.u32()?;
    }
    let front = reader.u8()?;
    let rear = reader.u8()?;
    if front >= 31 || rear >= 31 {
        return Err(CoreError::InvalidRngCheckpoint);
    }
    Ok(RngCheckpoint {
        seed,
        state,
        front,
        rear,
    })
}

fn write_render_state(writer: &mut Writer, state: &RenderState) {
    writer.u8(state.draw_color);
    writer.u16(state.fill_pattern);
    writer.bytes(&state.draw_palette);
    writer.bytes(&state.screen_palette);
    for value in state.transparent {
        writer.bool(value);
    }
    writer.i32(state.camera_x.raw());
    writer.i32(state.camera_y.raw());
    writer.i16(state.clip_x);
    writer.i16(state.clip_y);
    writer.u16(state.clip_width);
    writer.u16(state.clip_height);
    writer.i16(state.transition_y);
}

fn read_render_state(reader: &mut Reader<'_>) -> Result<RenderState, CoreError> {
    let draw_color = reader.u8()?;
    let fill_pattern = reader.u16()?;
    let mut draw_palette = [0_u8; PALETTE_SIZE];
    draw_palette.copy_from_slice(reader.bytes(PALETTE_SIZE)?);
    let mut screen_palette = [0_u8; PALETTE_SIZE];
    screen_palette.copy_from_slice(reader.bytes(PALETTE_SIZE)?);
    let mut transparent = [false; PALETTE_SIZE];
    for value in &mut transparent {
        *value = reader.bool()?;
    }
    Ok(RenderState {
        draw_color,
        fill_pattern,
        draw_palette,
        screen_palette,
        transparent,
        camera_x: PicoFixed::from_raw(reader.i32()?),
        camera_y: PicoFixed::from_raw(reader.i32()?),
        clip_x: reader.i16()?,
        clip_y: reader.i16()?,
        clip_width: reader.u16()?,
        clip_height: reader.u16()?,
        transition_y: reader.i16()?,
    })
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 14_695_981_039_346_656_037_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211_u64);
    }
    hash
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn i8(&mut self, value: i8) {
        self.bytes.push(value as u8);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    fn i16(&mut self, value: i16) {
        self.bytes(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, length: usize) -> Result<&'a [u8], CoreError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(CoreError::InvalidSnapshotTruncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(CoreError::InvalidSnapshotTruncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, CoreError> {
        self.bytes(1)?
            .first()
            .copied()
            .ok_or(CoreError::InvalidSnapshotTruncated)
    }

    fn i8(&mut self) -> Result<i8, CoreError> {
        Ok(self.u8()? as i8)
    }

    fn bool(&mut self) -> Result<bool, CoreError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CoreError::InvalidSnapshotValue),
        }
    }

    fn u16(&mut self) -> Result<u16, CoreError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.bytes(2)?);
        Ok(u16::from_le_bytes(value))
    }

    fn i16(&mut self) -> Result<i16, CoreError> {
        let mut value = [0_u8; 2];
        value.copy_from_slice(self.bytes(2)?);
        Ok(i16::from_le_bytes(value))
    }

    fn u32(&mut self) -> Result<u32, CoreError> {
        let mut value = [0_u8; 4];
        value.copy_from_slice(self.bytes(4)?);
        Ok(u32::from_le_bytes(value))
    }

    fn i32(&mut self) -> Result<i32, CoreError> {
        let mut value = [0_u8; 4];
        value.copy_from_slice(self.bytes(4)?);
        Ok(i32::from_le_bytes(value))
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{lerp_fixed, IndexedFramebuffer, RenderState, Snapshot, FRAMEBUFFER_SIZE};
    use crate::{Action, NativeConfig, NativeGame, PicoFixed};

    #[test]
    fn reset_snapshot_exposes_typed_state_and_indexed_pixels() {
        let game = NativeGame::new(NativeConfig::default());
        let snapshot = game.snapshot();
        assert_eq!(snapshot.logical_state().lifecycle.frame, 0);
        assert_eq!(snapshot.logical_state().player.x, PicoFixed::from_int(64));
        assert!(snapshot.logical_state().enemies.is_empty());
        assert_eq!(snapshot.pixels().len(), FRAMEBUFFER_SIZE);
        assert_eq!(snapshot.framebuffer().pixel(0, 0), Some(12));
        assert_eq!(snapshot.render_state().draw_palette[12], 12);
        assert_eq!(snapshot.render_state().palette_index(12), Some(12));
        assert_eq!(snapshot.render_state().palette_index(16), None);
    }

    #[test]
    fn snapshot_roundtrip_preserves_bytes_state_and_pixel_hash() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let _ = game.step(Action::Right, 17);
        let snapshot = game.snapshot();
        let bytes = snapshot.canonical_bytes();
        let restored = Snapshot::from_canonical_bytes(&bytes);
        assert!(restored.is_ok());
        if let Ok(restored) = restored {
            assert_eq!(restored.canonical_bytes(), bytes);
            assert_eq!(restored.logical_state(), snapshot.logical_state());
            assert_eq!(restored.render_state(), snapshot.render_state());
            assert_eq!(restored.pixels(), snapshot.pixels());
            assert_eq!(restored.state_hash(), snapshot.state_hash());
            assert_eq!(restored.pixel_hash(), snapshot.pixel_hash());
        }
    }

    #[test]
    fn rendering_does_not_mutate_simulation_state_or_rng() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let _ = game.step(Action::Right, 17);
        let before = game.full_state();
        let first = game.snapshot();
        let second = game.snapshot();

        assert_eq!(game.full_state(), before);
        assert_eq!(first, second);
    }

    #[test]
    fn camera_pixels_remain_in_source_pget_coordinate_space() {
        let mut render = RenderState::new(0);
        render.camera_x = PicoFixed::from_int(1);
        render.camera_y = PicoFixed::from_int(-1);
        let mut framebuffer = IndexedFramebuffer::blank();
        framebuffer.clear(12, &render);
        framebuffer.plot_world(10, 10, 7, &render);
        framebuffer.plot_world(11, 10, 0, &render);
        framebuffer.plot_world(0, 0, 7, &render);
        framebuffer.plot_world(127, 127, 7, &render);
        let projected = framebuffer.project(&render);

        assert_eq!(projected.pixel(0, 10), Some(0));
        assert_eq!(projected.pixel(10, 10), Some(7));
        assert_eq!(projected.pixel(11, 10), Some(0));
        assert_eq!(projected.pixel(9, 10), Some(12));
        assert_eq!(projected.pixel(10, 9), Some(12));
        assert_eq!(projected.pixel(0, 0), Some(0));
        assert_eq!(projected.pixel(127, 127), Some(0));

        render.camera_x = PicoFixed::from_f32(0.5);
        render.camera_y = PicoFixed::from_f32(-0.5);
        let mut subpixel = IndexedFramebuffer::blank();
        subpixel.clear(12, &render);
        assert_eq!(subpixel.pixel(0, 0), Some(0));
        assert_eq!(subpixel.pixel(127, 127), Some(0));
    }

    #[test]
    fn v161_pattern_lerp_preserves_odd_midpoint_before_rasterization() {
        let midpoint = PicoFixed::from_int(63).add(
            PicoFixed::from_int(1)
                .div_fixed(PicoFixed::from_int(2))
                .unwrap_or(PicoFixed::ZERO),
        );

        assert_eq!(lerp_fixed(95, midpoint, PicoFixed::from_f32(0.22)), 88);
    }

    #[test]
    fn truncated_snapshot_is_rejected_before_decode() {
        let snapshot = NativeGame::new(NativeConfig::default()).snapshot();
        let bytes = snapshot.canonical_bytes();
        assert!(!bytes.is_empty());
        let truncated = bytes.get(..bytes.len() - 1);
        assert!(truncated.is_some());
        if let Some(truncated) = truncated {
            assert!(Snapshot::from_canonical_bytes(truncated).is_err());
        }
    }
}
