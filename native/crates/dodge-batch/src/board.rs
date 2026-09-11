use dodge_core::{FullState, PatternRect, PicoFixed};

pub const BOARD_SIZE: usize = 16;
pub const BOARD_WIDTH: usize = BOARD_SIZE;
pub const BOARD_HEIGHT: usize = BOARD_SIZE;
pub const BOARD_CHANNELS: usize = 19;
pub const FULL_BOARD_CHANNELS: usize = 23;
pub const BOARD_VALUES: usize = BOARD_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH;
pub const FULL_BOARD_VALUES: usize = FULL_BOARD_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH;
const CELL_VALUES: usize = BOARD_HEIGHT * BOARD_WIDTH;
const VELOCITY_VALUES: usize = 6 * CELL_VALUES;
const SCREEN_SIZE: f32 = 128.0;

pub const PLAYER_PRESENCE: usize = 0;
pub const PLAYER_VX: usize = 1;
pub const PLAYER_VY: usize = 2;
pub const PLAYER_WIDTH: usize = 3;
pub const PLAYER_HEIGHT: usize = 4;
pub const ENEMY_PRESENCE: usize = 5;
pub const ENEMY_VX: usize = 6;
pub const ENEMY_VY: usize = 7;
pub const ENEMY_WIDTH: usize = 8;
pub const ENEMY_HEIGHT: usize = 9;
pub const ENEMY_STAGE: usize = 10;
pub const AOE_PRESENCE: usize = 11;
pub const AOE_VX: usize = 12;
pub const AOE_VY: usize = 13;
pub const AOE_WIDTH: usize = 14;
pub const AOE_HEIGHT: usize = 15;
pub const AOE_STAGE: usize = 16;
pub const AOE_EXPLOSION: usize = 17;
pub const AOE_PATTERN: usize = 18;
pub const ENEMY_X: usize = 19;
pub const ENEMY_Y: usize = 20;
pub const AOE_X: usize = 21;
pub const AOE_Y: usize = 22;

/// Channel-major `(19, 16, 16)` semantic board, flattened in C order.
///
/// This is a compatibility observation for the existing CNN. The canonical
/// native observation remains `FullState` plus indexed pixels. Normal enemy
/// records use the same width rule as the Python bridge: personalities 0/1
/// use their current size, while power-up personalities use an 8-pixel box.
#[derive(Clone, Debug, PartialEq)]
pub struct Board19x16 {
    values: Vec<f32>,
    channels: usize,
}

impl Board19x16 {
    pub fn from_full_state(state: &FullState) -> Self {
        Self::from_full_state_with_options(state, false, false)
    }

    pub fn from_full_state_with_offscreen(state: &FullState, include_offscreen: bool) -> Self {
        Self::from_full_state_with_options(state, include_offscreen, false)
    }

    pub fn from_full_state_with_coordinates(state: &FullState) -> Self {
        Self::from_full_state_with_options(state, true, true)
    }

    fn from_full_state_with_options(
        state: &FullState,
        include_offscreen: bool,
        preserve_coordinates: bool,
    ) -> Self {
        let channels = if preserve_coordinates {
            FULL_BOARD_CHANNELS
        } else {
            BOARD_CHANNELS
        };
        let mut board = vec![0.0; channels * BOARD_HEIGHT * BOARD_WIDTH];
        let mut velocity_sums = vec![0.0; VELOCITY_VALUES];
        let mut velocity_counts = vec![0_u32; VELOCITY_VALUES];
        paint_entity(
            &mut board,
            &mut velocity_sums,
            &mut velocity_counts,
            state.player.x,
            state.player.y,
            state.player.size,
            state.player.size,
            state.player.vx,
            state.player.vy,
            None,
            PLAYER_PRESENCE,
            PLAYER_VX,
            PLAYER_VY,
            PLAYER_WIDTH,
            PLAYER_HEIGHT,
            None,
            0,
            None,
            include_offscreen,
            None,
        );
        for enemy in &state.enemies {
            if enemy.personality == -1 {
                paint_entity(
                    &mut board,
                    &mut velocity_sums,
                    &mut velocity_counts,
                    enemy.x,
                    enemy.y,
                    enemy.size,
                    enemy.size,
                    enemy.vx,
                    enemy.vy,
                    Some(PicoFixed::ZERO),
                    AOE_PRESENCE,
                    AOE_VX,
                    AOE_VY,
                    AOE_WIDTH,
                    AOE_HEIGHT,
                    Some(AOE_STAGE),
                    4,
                    Some(AOE_EXPLOSION),
                    include_offscreen,
                    preserve_coordinates.then_some((AOE_X, AOE_Y)),
                );
            } else {
                let size = if enemy.personality >= 2 {
                    PicoFixed::from_int(8)
                } else {
                    enemy.size
                };
                paint_entity(
                    &mut board,
                    &mut velocity_sums,
                    &mut velocity_counts,
                    enemy.x,
                    enemy.y,
                    size,
                    size,
                    enemy.vx,
                    enemy.vy,
                    Some(PicoFixed::ZERO),
                    ENEMY_PRESENCE,
                    ENEMY_VX,
                    ENEMY_VY,
                    ENEMY_WIDTH,
                    ENEMY_HEIGHT,
                    Some(ENEMY_STAGE),
                    2,
                    None,
                    include_offscreen,
                    preserve_coordinates.then_some((ENEMY_X, ENEMY_Y)),
                );
            }
        }
        if let Some(pattern_index) = state.active_pattern {
            if let Some(pattern) = state.patterns.get(pattern_index) {
                for rect in &pattern.rects {
                    paint_pattern(
                        &mut board,
                        &mut velocity_sums,
                        &mut velocity_counts,
                        rect,
                        include_offscreen,
                        preserve_coordinates.then_some((AOE_X, AOE_Y)),
                    );
                }
            }
        }
        write_velocity_channel(&mut board, &velocity_sums, &velocity_counts, PLAYER_VX, 0);
        write_velocity_channel(&mut board, &velocity_sums, &velocity_counts, PLAYER_VY, 1);
        write_velocity_channel(&mut board, &velocity_sums, &velocity_counts, ENEMY_VX, 2);
        write_velocity_channel(&mut board, &velocity_sums, &velocity_counts, ENEMY_VY, 3);
        write_velocity_channel(&mut board, &velocity_sums, &velocity_counts, AOE_VX, 4);
        write_velocity_channel(&mut board, &velocity_sums, &velocity_counts, AOE_VY, 5);
        Self {
            values: board,
            channels,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Return the C-order flat index for a channel-major board coordinate.
    pub const fn flat_index(channel: usize, row: usize, column: usize) -> Option<usize> {
        if channel >= BOARD_CHANNELS || row >= BOARD_HEIGHT || column >= BOARD_WIDTH {
            return None;
        }
        Some(channel * CELL_VALUES + row * BOARD_WIDTH + column)
    }

    /// Read one semantic board cell without exposing unchecked indexing.
    pub fn value(&self, channel: usize, row: usize, column: usize) -> Option<f32> {
        Self::flat_index(channel, row, column).and_then(|index| self.values.get(index).copied())
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, BOARD_HEIGHT, BOARD_WIDTH)
    }
}

fn paint_pattern(
    board: &mut [f32],
    velocity_sums: &mut [f32],
    velocity_counts: &mut [u32],
    rect: &PatternRect,
    include_offscreen: bool,
    position_channels: Option<(usize, usize)>,
) {
    paint_entity(
        board,
        velocity_sums,
        velocity_counts,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        rect.dx,
        rect.dy,
        Some(rect.sh),
        AOE_PRESENCE,
        AOE_VX,
        AOE_VY,
        AOE_WIDTH,
        AOE_HEIGHT,
        Some(AOE_STAGE),
        4,
        Some(AOE_PATTERN),
        include_offscreen,
        position_channels,
    );
}

#[allow(clippy::too_many_arguments)]
fn paint_entity(
    board: &mut [f32],
    velocity_sums: &mut [f32],
    velocity_counts: &mut [u32],
    x: PicoFixed,
    y: PicoFixed,
    width: PicoFixed,
    height: PicoFixed,
    vx: PicoFixed,
    vy: PicoFixed,
    stage: Option<PicoFixed>,
    presence_channel: usize,
    _vx_channel: usize,
    _vy_channel: usize,
    width_channel: usize,
    height_channel: usize,
    stage_channel: Option<usize>,
    velocity_slot: usize,
    kind_channel: Option<usize>,
    include_offscreen: bool,
    position_channels: Option<(usize, usize)>,
) {
    let Some((top, bottom, left, right)) = cell_bounds(x, y, width, height, include_offscreen)
    else {
        return;
    };
    let normalized_width = width.to_f32() / SCREEN_SIZE;
    let normalized_height = height.to_f32() / SCREEN_SIZE;
    let normalized_vx = vx.to_f32() / SCREEN_SIZE;
    let normalized_vy = vy.to_f32() / SCREEN_SIZE;
    let normalized_stage = stage.map(|value| value.to_f32() / 2.0);
    let normalized_x = 0.5 + x.to_f32() / SCREEN_SIZE;
    let normalized_y = 0.5 + y.to_f32() / SCREEN_SIZE;
    for row in top..=bottom {
        for column in left..=right {
            let cell = row * BOARD_WIDTH + column;
            set_presence(board, presence_channel, cell);
            update_max(board, width_channel, cell, normalized_width);
            update_max(board, height_channel, cell, normalized_height);
            add_velocity(
                velocity_sums,
                velocity_counts,
                velocity_slot,
                cell,
                normalized_vx,
                normalized_vy,
            );
            if let (Some(channel), Some(value)) = (stage_channel, normalized_stage) {
                update_max(board, channel, cell, value);
            }
            if let Some(channel) = kind_channel {
                set_presence(board, channel, cell);
            }
            if let Some((x_channel, y_channel)) = position_channels {
                update_max(board, x_channel, cell, normalized_x);
                update_max(board, y_channel, cell, normalized_y);
            }
        }
    }
}

fn cell_bounds(
    x: PicoFixed,
    y: PicoFixed,
    width: PicoFixed,
    height: PicoFixed,
    include_offscreen: bool,
) -> Option<(usize, usize, usize, usize)> {
    let scale = BOARD_SIZE as f32 / SCREEN_SIZE;
    let left = ((x.to_f32() - width.to_f32() / 2.0) * scale).floor();
    let right = ((x.to_f32() + width.to_f32() / 2.0) * scale).floor();
    let top = ((y.to_f32() - height.to_f32() / 2.0) * scale).floor();
    let bottom = ((y.to_f32() + height.to_f32() / 2.0) * scale).floor();
    let left = left as i32;
    let right = right as i32;
    let top = top as i32;
    let bottom = bottom as i32;
    if !include_offscreen
        && (right < 0 || left >= BOARD_SIZE as i32 || bottom < 0 || top >= BOARD_SIZE as i32)
    {
        return None;
    }
    let (left, right) = if right < 0 {
        (0, 0)
    } else if left >= BOARD_SIZE as i32 {
        (BOARD_SIZE - 1, BOARD_SIZE - 1)
    } else {
        (
            left.max(0) as usize,
            right.min(BOARD_SIZE as i32 - 1) as usize,
        )
    };
    let (top, bottom) = if bottom < 0 {
        (0, 0)
    } else if top >= BOARD_SIZE as i32 {
        (BOARD_SIZE - 1, BOARD_SIZE - 1)
    } else {
        (
            top.max(0) as usize,
            bottom.min(BOARD_SIZE as i32 - 1) as usize,
        )
    };
    Some((top, bottom, left, right))
}

fn set_presence(board: &mut [f32], channel: usize, cell: usize) {
    let index = channel * CELL_VALUES + cell;
    if let Some(value) = board.get_mut(index) {
        *value = 1.0;
    }
}

fn update_max(board: &mut [f32], channel: usize, cell: usize, value: f32) {
    let index = channel * CELL_VALUES + cell;
    if let Some(current) = board.get_mut(index) {
        *current = current.max(value);
    }
}

fn add_velocity(sums: &mut [f32], counts: &mut [u32], slot: usize, cell: usize, vx: f32, vy: f32) {
    let x_index = slot * CELL_VALUES + cell;
    let y_index = (slot + 1) * CELL_VALUES + cell;
    if let Some(sum) = sums.get_mut(x_index) {
        *sum += vx;
    }
    if let Some(sum) = sums.get_mut(y_index) {
        *sum += vy;
    }
    if let Some(count) = counts.get_mut(x_index) {
        *count = count.saturating_add(1);
    }
    if let Some(count) = counts.get_mut(y_index) {
        *count = count.saturating_add(1);
    }
}

fn write_velocity_channel(
    board: &mut [f32],
    sums: &[f32],
    counts: &[u32],
    channel: usize,
    slot: usize,
) {
    for cell in 0..CELL_VALUES {
        let sum_index = slot * CELL_VALUES + cell;
        let Some(count) = counts.get(sum_index).copied() else {
            continue;
        };
        if count == 0 {
            continue;
        }
        let Some(sum) = sums.get(sum_index).copied() else {
            continue;
        };
        let board_index = channel * CELL_VALUES + cell;
        if let Some(value) = board.get_mut(board_index) {
            *value = sum / count as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{cell_bounds, Board19x16, BOARD_CHANNELS, BOARD_HEIGHT, BOARD_SIZE, BOARD_WIDTH};
    use dodge_core::{NativeConfig, NativeGame, PicoFixed};

    #[test]
    fn native_ready_state_matches_board_shape_and_finite_contract() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let mut snapshot = game.reset();
        for _ in 0..13 {
            snapshot = game
                .advance_frame(dodge_core::BUTTON_X_MASK)
                .unwrap_or_else(|_| unreachable!("startup should succeed"))
                .snapshot;
        }
        let board = Board19x16::from_full_state(snapshot.logical_state());
        assert_eq!(board.shape(), (BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH));
        assert_eq!(
            board.as_slice().len(),
            BOARD_SIZE * BOARD_SIZE * BOARD_CHANNELS
        );
        assert!(board.as_slice().iter().all(|value| value.is_finite()));
        assert_eq!(Board19x16::flat_index(18, 15, 15), Some(4_863));
        assert_eq!(Board19x16::flat_index(19, 0, 0), None);
        assert!(board.value(0, 0, 0).is_some());
    }

    #[test]
    fn offscreen_entities_are_optional_and_pin_to_the_approaching_edge() {
        let offscreen_left = PicoFixed::from_int(-10);
        let offscreen_right = PicoFixed::from_int(138);
        let visible = PicoFixed::from_int(64);
        let size = PicoFixed::from_int(8);

        assert_eq!(
            cell_bounds(offscreen_left, visible, size, size, false),
            None
        );
        assert_eq!(
            cell_bounds(offscreen_left, visible, size, size, true),
            Some((7, 8, 0, 0))
        );
        assert_eq!(
            cell_bounds(offscreen_right, visible, size, size, true),
            Some((7, 8, 15, 15))
        );
        assert_eq!(
            cell_bounds(offscreen_left, offscreen_right, size, size, true),
            Some((15, 15, 0, 0))
        );
    }
}
