//! Native ML observations shared by the batch and Python boundaries.

use std::cmp::Ordering;

use dodge_core::{EnemyState, FullState, NativeGame, PatternState, PlayerState};

pub const PLAYER_FEATURE_COUNT: usize = 5;
pub const ENTITY_SLOT_COUNT: usize = 16;
pub const AOE_SLOT_COUNT: usize = 8;
pub const ENTITY_FEATURE_COUNT: usize = 9;
pub const ML_OBSERVATION_SIZE: usize =
    PLAYER_FEATURE_COUNT + (ENTITY_SLOT_COUNT + AOE_SLOT_COUNT) * ENTITY_FEATURE_COUNT + 4;
pub const DEFAULT_GRID_SPACING: u32 = 32;

const SCREEN_SIZE: f64 = 128.0;
const PLAYER_CENTER_MIN: f64 = 2.0;
const PLAYER_CENTER_MAX: f64 = 125.0;
const TIME_TO_INTERSECTION_HORIZON: f64 = 120.0;

#[derive(Clone, Copy, Debug)]
struct Entity {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    width: f64,
    height: f64,
    stage: f64,
}

/// Encode the waypoint DQN's structured observation without crossing through
/// Python or serializing a canonical snapshot.
pub fn encode_waypoint_observation(
    state: &FullState,
    grid_spacing: u32,
) -> Option<[f32; ML_OBSERVATION_SIZE]> {
    encode_waypoint_observation_from_source(state, grid_spacing)
}

/// Encode the waypoint DQN observation directly from a live native game.
///
/// This shares the exact feature encoder used for canonical snapshots while
/// avoiding a `FullState` clone and its physical framebuffer allocation.
pub fn encode_waypoint_observation_from_game(
    game: &NativeGame,
    grid_spacing: u32,
) -> Option<[f32; ML_OBSERVATION_SIZE]> {
    encode_waypoint_observation_from_source(game, grid_spacing)
}

trait ObservationSource {
    fn player_state(&self) -> PlayerState;
    fn enemies(&self) -> &[EnemyState];
    fn patterns(&self) -> &[PatternState];
    fn active_pattern_index(&self) -> Option<usize>;
}

impl ObservationSource for FullState {
    fn player_state(&self) -> PlayerState {
        self.player
    }

    fn enemies(&self) -> &[EnemyState] {
        self.enemies.as_slice()
    }

    fn patterns(&self) -> &[PatternState] {
        self.patterns.as_slice()
    }

    fn active_pattern_index(&self) -> Option<usize> {
        self.active_pattern
    }
}

impl ObservationSource for NativeGame {
    fn player_state(&self) -> PlayerState {
        self.player()
    }

    fn enemies(&self) -> &[EnemyState] {
        self.enemies()
    }

    fn patterns(&self) -> &[PatternState] {
        self.patterns()
    }

    fn active_pattern_index(&self) -> Option<usize> {
        self.active_pattern_index()
    }
}

fn encode_waypoint_observation_from_source<S: ObservationSource>(
    state: &S,
    grid_spacing: u32,
) -> Option<[f32; ML_OBSERVATION_SIZE]> {
    if grid_spacing == 0 {
        return None;
    }

    let player_state = state.player_state();
    let player = Entity {
        x: player_state.x.to_double(),
        y: player_state.y.to_double(),
        vx: player_state.vx.to_double(),
        vy: player_state.vy.to_double(),
        width: player_state.size.to_double(),
        height: player_state.size.to_double(),
        stage: 0.0,
    };
    let mut enemies = Vec::new();
    let mut aoes = Vec::new();
    for enemy in state.enemies() {
        let size = enemy.size.to_double();
        let width = if enemy.personality >= 2 { 8.0 } else { size };
        let entity = Entity {
            x: enemy.x.to_double(),
            y: enemy.y.to_double(),
            vx: enemy.vx.to_double(),
            vy: enemy.vy.to_double(),
            width,
            height: width,
            stage: 0.0,
        };
        if enemy.personality == -1 {
            aoes.push(entity);
        } else {
            enemies.push(entity);
        }
    }
    if let Some(pattern_index) = state.active_pattern_index() {
        if let Some(pattern) = state.patterns().get(pattern_index) {
            aoes.extend(pattern.rects.iter().map(pattern_entity));
        }
    }

    enemies.sort_by(|left, right| danger_ordering(player, *left, *right));
    aoes.sort_by(|left, right| danger_ordering(player, *left, *right));

    let mut values = [0.0_f32; ML_OBSERVATION_SIZE];
    for (index, value) in [
        (player.x / SCREEN_SIZE) as f32,
        (player.y / SCREEN_SIZE) as f32,
        (player.vx / SCREEN_SIZE) as f32,
        (player.vy / SCREEN_SIZE) as f32,
        (player.width / SCREEN_SIZE) as f32,
    ]
    .into_iter()
    .enumerate()
    {
        if !write_feature(&mut values, index, value) {
            return None;
        }
    }
    let mut offset = PLAYER_FEATURE_COUNT;
    if !encode_entities(
        &mut values,
        &mut offset,
        player,
        &enemies,
        ENTITY_SLOT_COUNT,
    ) {
        return None;
    }
    if !encode_entities(&mut values, &mut offset, player, &aoes, AOE_SLOT_COUNT) {
        return None;
    }

    let axis_points = axis_points(grid_spacing);
    let denominator = (axis_points.len().saturating_sub(1).max(1)) as f64;
    let column = nearest_axis(&axis_points, player.x);
    let row = nearest_axis(&axis_points, player.y);
    for value in [
        column as f32 / denominator as f32,
        row as f32 / denominator as f32,
        f32::from(enemies.len() > ENTITY_SLOT_COUNT),
        f32::from(aoes.len() > AOE_SLOT_COUNT),
    ] {
        if !write_feature(&mut values, offset, value) {
            return None;
        }
        offset += 1;
    }
    Some(values)
}

fn pattern_entity(rect: &dodge_core::PatternRect) -> Entity {
    Entity {
        x: rect.x.to_double(),
        y: rect.y.to_double(),
        vx: rect.dx.to_double(),
        vy: rect.dy.to_double(),
        width: rect.width.to_double(),
        height: rect.height.to_double(),
        stage: rect.sh.to_double(),
    }
}

fn encode_entities(
    values: &mut [f32; ML_OBSERVATION_SIZE],
    offset: &mut usize,
    player: Entity,
    entities: &[Entity],
    slot_count: usize,
) -> bool {
    for entity in entities.iter().take(slot_count).copied() {
        for value in [
            1.0,
            normalized_time_to_intersection(player, entity) as f32,
            ((entity.x - player.x) / SCREEN_SIZE) as f32,
            ((entity.y - player.y) / SCREEN_SIZE) as f32,
            (entity.vx / SCREEN_SIZE) as f32,
            (entity.vy / SCREEN_SIZE) as f32,
            (entity.width / SCREEN_SIZE) as f32,
            (entity.height / SCREEN_SIZE) as f32,
            (entity.stage / 2.0) as f32,
        ] {
            if !write_feature(values, *offset, value) {
                return false;
            }
            *offset += 1;
        }
    }
    *offset += ENTITY_FEATURE_COUNT * (slot_count - entities.len().min(slot_count));
    true
}

fn write_feature(values: &mut [f32; ML_OBSERVATION_SIZE], index: usize, value: f32) -> bool {
    let Some(slot) = values.get_mut(index) else {
        return false;
    };
    *slot = value;
    true
}

fn danger_ordering(player: Entity, left: Entity, right: Entity) -> Ordering {
    normalized_sort_key(player, left).cmp(&normalized_sort_key(player, right))
}

fn normalized_sort_key(player: Entity, entity: Entity) -> (OrderedFloat, OrderedFloat) {
    (
        OrderedFloat(time_to_intersection(player, entity)),
        OrderedFloat(center_distance(player, entity)),
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn normalized_time_to_intersection(player: Entity, entity: Entity) -> f64 {
    let time = time_to_intersection(player, entity);
    if time.is_infinite() {
        1.0
    } else {
        (time / TIME_TO_INTERSECTION_HORIZON).min(1.0)
    }
}

fn time_to_intersection(player: Entity, entity: Entity) -> f64 {
    let player_half = player.width / 2.0;
    entity_axis_time(
        entity.x - player.x,
        entity.vx - player.vx,
        player_half + entity.width / 2.0,
    )
    .max(entity_axis_time(
        entity.y - player.y,
        entity.vy - player.vy,
        player_half + entity.height / 2.0,
    ))
}

fn entity_axis_time(distance: f64, velocity: f64, combined_half: f64) -> f64 {
    if distance.abs() <= combined_half {
        return 0.0;
    }
    if velocity == 0.0 || distance * velocity >= 0.0 {
        return f64::INFINITY;
    }
    ((distance.abs() - combined_half) / velocity.abs()).max(0.0)
}

fn center_distance(player: Entity, entity: Entity) -> f64 {
    let dx = entity.x - player.x;
    let dy = entity.y - player.y;
    dx * dx + dy * dy
}

fn axis_points(grid_spacing: u32) -> Vec<f64> {
    let mut points = Vec::new();
    let mut point = PLAYER_CENTER_MIN;
    let spacing = f64::from(grid_spacing);
    while point < PLAYER_CENTER_MAX {
        points.push(point);
        point += spacing;
    }
    if points.last().copied() != Some(PLAYER_CENTER_MAX) {
        points.push(PLAYER_CENTER_MAX);
    }
    points
}

fn nearest_axis(points: &[f64], value: f64) -> usize {
    let Some(first) = points.first() else {
        return 0;
    };
    let mut best_index = 0;
    let mut best_distance = (*first - value).abs();
    for (index, point) in points.iter().copied().enumerate().skip(1) {
        let distance = (point - value).abs();
        if distance < best_distance {
            best_index = index;
            best_distance = distance;
        }
    }
    best_index
}

#[cfg(test)]
mod tests {
    use super::{encode_waypoint_observation, ML_OBSERVATION_SIZE};
    use dodge_core::{NativeConfig, NativeGame};

    #[test]
    fn encoder_returns_finite_declared_vector() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let snapshot = game.reset();
        let observation = encode_waypoint_observation(snapshot.logical_state(), 32)
            .unwrap_or_else(|| unreachable!("positive spacing should encode"));
        assert_eq!(observation.len(), ML_OBSERVATION_SIZE);
        assert!(observation.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn encoder_rejects_zero_grid_spacing() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let snapshot = game.reset();
        assert!(encode_waypoint_observation(snapshot.logical_state(), 0).is_none());
    }

    #[test]
    fn encoder_includes_rects_from_the_active_pattern_in_aoe_slots() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let snapshot = game.reset();
        let mut state = snapshot.logical_state().clone();
        state.active_pattern = Some(0);
        let observation = encode_waypoint_observation(&state, 32)
            .unwrap_or_else(|| unreachable!("positive spacing should encode"));
        let aoe_start =
            super::PLAYER_FEATURE_COUNT + super::ENTITY_SLOT_COUNT * super::ENTITY_FEATURE_COUNT;
        assert_eq!(observation.get(aoe_start).copied().unwrap_or(-1.0), 1.0);
    }
}
