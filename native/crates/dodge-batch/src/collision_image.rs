use dodge_core::{FullState, PatternRect, PatternState, PicoFixed};

pub const COLLISION_IMAGE_WIDTH: usize = 84;
pub const COLLISION_IMAGE_HEIGHT: usize = 84;
pub const COLLISION_IMAGE_CHANNELS: usize = 1;
pub const COLLISION_IMAGE_VALUES: usize =
    COLLISION_IMAGE_WIDTH * COLLISION_IMAGE_HEIGHT * COLLISION_IMAGE_CHANNELS;
pub const COLLISION_IMAGE_OBSERVATION_VERSION: u32 = 1;
pub const COLLISION_IMAGE_CONFIG: &str = "world128-max-composite-u8-hw-v1";

pub const COLLISION_IMAGE_BACKGROUND: u8 = 0;
pub const COLLISION_IMAGE_AREA: u8 = 96;
pub const COLLISION_IMAGE_HOSTILE: u8 = 192;
pub const COLLISION_IMAGE_PLAYER: u8 = 255;

const FIXED_ONE: i64 = 1 << 16;
const WORLD_SIZE: i64 = 128;
const WORLD_RAW: i64 = WORLD_SIZE * FIXED_ONE;

/// One-channel row-major collision image for a native `FullState`.
///
/// The encoder deliberately does not inspect `FullState::physical_screen` or
/// any render-side state. It projects the same logical rectangles used by the
/// native player/enemy/pattern collision checks into an 84x84 world grid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollisionImage84x84 {
    values: [u8; COLLISION_IMAGE_VALUES],
}

impl CollisionImage84x84 {
    pub fn from_full_state(state: &FullState) -> Self {
        let mut values = [COLLISION_IMAGE_BACKGROUND; COLLISION_IMAGE_VALUES];

        paint_player(&mut values, state);
        if state.should_collide && !state.lifecycle.dead {
            for enemy in &state.enemies {
                if enemy.personality < 2 {
                    let value = if enemy.personality == -1 {
                        COLLISION_IMAGE_AREA
                    } else {
                        COLLISION_IMAGE_HOSTILE
                    };
                    paint_rect(&mut values, enemy.x, enemy.y, enemy.size, enemy.size, value);
                }
            }

            if let Some(pattern_index) = state.active_pattern {
                if let Some(pattern) = state.patterns.get(pattern_index) {
                    paint_damaging_pattern(&mut values, pattern);
                }
            }
        }

        Self { values }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.values
    }

    pub fn into_array(self) -> [u8; COLLISION_IMAGE_VALUES] {
        self.values
    }

    pub const fn shape(&self) -> (usize, usize) {
        (COLLISION_IMAGE_HEIGHT, COLLISION_IMAGE_WIDTH)
    }
}

pub fn encode_collision_image(state: &FullState) -> CollisionImage84x84 {
    CollisionImage84x84::from_full_state(state)
}

fn paint_player(values: &mut [u8; COLLISION_IMAGE_VALUES], state: &FullState) {
    // These bounds match the normal-hostile collision inequalities:
    // player.x - size + 1 < enemy.right and player.x + size - 1 > enemy.left.
    let half_extent = state.player.size.sub(PicoFixed::ONE);
    paint_rect(
        values,
        state.player.x.sub(half_extent),
        state.player.y.sub(half_extent),
        half_extent.add(half_extent),
        half_extent.add(half_extent),
        COLLISION_IMAGE_PLAYER,
    );
}

fn paint_damaging_pattern(values: &mut [u8; COLLISION_IMAGE_VALUES], pattern: &PatternState) {
    for rect in &pattern.rects {
        if pattern_rect_is_damaging(pattern, rect) {
            paint_rect(
                values,
                rect.x,
                rect.y,
                rect.width,
                rect.height,
                COLLISION_IMAGE_AREA,
            );
        }
    }
}

fn pattern_rect_is_damaging(pattern: &PatternState, rect: &PatternRect) -> bool {
    // This is the same gate used by NativeGame::collision_check for active
    // pattern rectangles: moving patterns are always live; static patterns
    // become live when their expansion reaches sh == 2.
    pattern.pattern_type == 1 || rect.sh == PicoFixed::from_int(2)
}

fn paint_rect(
    values: &mut [u8; COLLISION_IMAGE_VALUES],
    x: PicoFixed,
    y: PicoFixed,
    width: PicoFixed,
    height: PicoFixed,
    value: u8,
) {
    let Some((left, right)) = clipped_interval(x.raw(), width.raw()) else {
        return;
    };
    let Some((top, bottom)) = clipped_interval(y.raw(), height.raw()) else {
        return;
    };
    let left = scaled_floor(left);
    let right = scaled_ceil(right);
    let top = scaled_floor(top);
    let bottom = scaled_ceil(bottom);

    for row in top..bottom {
        let row_start = row * COLLISION_IMAGE_WIDTH;
        for column in left..right {
            let index = row_start + column;
            if let Some(pixel) = values.get_mut(index) {
                *pixel = (*pixel).max(value);
            }
        }
    }
}

fn clipped_interval(start_raw: i32, length_raw: i32) -> Option<(i64, i64)> {
    let start = i64::from(start_raw);
    let end = start.saturating_add(i64::from(length_raw));
    let left = start.max(0);
    let right = end.min(WORLD_RAW);
    (right > left).then_some((left, right))
}

fn scaled_floor(world_raw: i64) -> usize {
    ((world_raw * COLLISION_IMAGE_WIDTH as i64) / WORLD_RAW).clamp(0, COLLISION_IMAGE_WIDTH as i64)
        as usize
}

fn scaled_ceil(world_raw: i64) -> usize {
    ((world_raw * COLLISION_IMAGE_WIDTH as i64 + WORLD_RAW - 1) / WORLD_RAW)
        .clamp(0, COLLISION_IMAGE_WIDTH as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::{
        encode_collision_image, CollisionImage84x84, COLLISION_IMAGE_AREA,
        COLLISION_IMAGE_BACKGROUND, COLLISION_IMAGE_CONFIG, COLLISION_IMAGE_HEIGHT,
        COLLISION_IMAGE_HOSTILE, COLLISION_IMAGE_OBSERVATION_VERSION, COLLISION_IMAGE_PLAYER,
        COLLISION_IMAGE_VALUES, COLLISION_IMAGE_WIDTH,
    };
    use dodge_core::{
        EnemyState, FullState, IndexedFramebuffer, NativeConfig, NativeGame, PatternRect,
        PatternState, PicoFixed,
    };

    fn state_with_collision_geometry() -> FullState {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let mut state = game.reset().logical_state().clone();
        state.should_collide = true;
        state.enemies = vec![EnemyState::normal(
            PicoFixed::from_int(16),
            PicoFixed::from_int(16),
            PicoFixed::from_int(8),
        )];
        state.patterns = vec![PatternState {
            id: 0,
            mins: PicoFixed::ZERO,
            maxs: PicoFixed::from_int(128),
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
                x: PicoFixed::from_int(96),
                y: PicoFixed::from_int(96),
                width: PicoFixed::from_int(12),
                height: PicoFixed::from_int(12),
                speed: PicoFixed::ZERO,
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
        state.active_pattern = Some(0);
        state
    }

    #[test]
    fn collision_image_has_exact_shape_bounded_values_and_stable_metadata() {
        let state = state_with_collision_geometry();
        let image = encode_collision_image(&state);

        assert_eq!(
            image.shape(),
            (COLLISION_IMAGE_HEIGHT, COLLISION_IMAGE_WIDTH)
        );
        assert_eq!(image.as_slice().len(), COLLISION_IMAGE_VALUES);
        assert!(image.as_slice().iter().all(|value| matches!(
            *value,
            COLLISION_IMAGE_BACKGROUND
                | COLLISION_IMAGE_AREA
                | COLLISION_IMAGE_HOSTILE
                | COLLISION_IMAGE_PLAYER
        )));
        assert!(image.as_slice().contains(&COLLISION_IMAGE_PLAYER));
        assert!(image.as_slice().contains(&COLLISION_IMAGE_HOSTILE));
        assert!(image.as_slice().contains(&COLLISION_IMAGE_AREA));
        assert_eq!(encode_collision_image(&state), image);
        assert_eq!(COLLISION_IMAGE_OBSERVATION_VERSION, 1);
        assert_eq!(COLLISION_IMAGE_CONFIG, "world128-max-composite-u8-hw-v1");
    }

    #[test]
    fn collision_image_does_not_read_rendered_pixels() {
        let state = state_with_collision_geometry();
        let mut altered = state.clone();
        altered.physical_screen = IndexedFramebuffer::filled(15);

        assert_eq!(
            encode_collision_image(&state),
            encode_collision_image(&altered)
        );
    }

    #[test]
    fn collision_image_is_not_emitted_for_non_damaging_static_pattern() {
        let mut state = state_with_collision_geometry();
        let pattern = state
            .patterns
            .first_mut()
            .unwrap_or_else(|| unreachable!("test pattern exists"));
        let rect = pattern
            .rects
            .first_mut()
            .unwrap_or_else(|| unreachable!("test rect exists"));
        rect.sh = PicoFixed::ZERO;

        let image = CollisionImage84x84::from_full_state(&state);
        assert!(!image.as_slice().contains(&COLLISION_IMAGE_AREA));
        assert!(image.as_slice().contains(&COLLISION_IMAGE_HOSTILE));
    }
}
