//! Bounded authoring driver. Only pixels and executed player actions leave the
//! learning boundary; waypoint coordinates are simulation inputs, not features.
use super::{EnemyState, IndexedFramebuffer, NativeConfig, NativeGame, PicoFixed};
use crate::{Action, BUTTON_X_MASK};

#[derive(Clone, Debug)]
pub enum PracticeCommand {
    Action { action: Action, decisions: u32 },
    MoveTo { x: f32, y: f32, max_decisions: u32 },
}

#[derive(Clone, Debug)]
pub struct PracticeSegment {
    pub x: f32,
    pub y: f32,
    pub frames: u32,
}

#[derive(Clone, Debug)]
pub struct PracticeEnemy {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub looping: bool,
    pub segments: Vec<PracticeSegment>,
}

pub struct PracticeStep {
    pub pixels: IndexedFramebuffer,
    pub action: Action,
    pub terminated: bool,
    pub finished: bool,
    pub failed: bool,
}

pub struct PracticeSimulation {
    game: NativeGame,
    config: NativeConfig,
    start: (f32, f32),
    commands: Vec<PracticeCommand>,
    enemies: Vec<PracticeEnemy>,
    command: usize,
    command_steps: u32,
    tick: u32,
    ended: bool,
}

fn within(value: f32, low: f32, high: f32) -> bool {
    value.is_finite() && value >= low && value <= high
}

impl PracticeSimulation {
    pub fn new(
        mut config: NativeConfig,
        start: (f32, f32),
        commands: Vec<PracticeCommand>,
        enemies: Vec<PracticeEnemy>,
        step_frames: u32,
    ) -> Result<Self, &'static str> {
        if step_frames != 4
            || !(1..=3).contains(&config.difficulty)
            || config.scenario.permanent_pattern > 39
            || !within(start.0, 4.0, 124.0)
            || !within(start.1, 4.0, 124.0)
            || commands.is_empty()
            || commands.len() > 64
            || enemies.len() > 32
        {
            return Err("invalid practice bounds or empty player script");
        }
        let mut budget = 0_u32;
        for command in &commands {
            let count = match command {
                PracticeCommand::Action { decisions, .. } => *decisions,
                PracticeCommand::MoveTo {
                    x,
                    y,
                    max_decisions,
                } => {
                    if !within(*x, 4.0, 124.0) || !within(*y, 4.0, 124.0) {
                        return Err("player waypoint outside 4..124");
                    }
                    *max_decisions
                }
            };
            if !(1..=256).contains(&count) {
                return Err("invalid command budget");
            }
            budget += count;
        }
        if budget > 256 {
            return Err("practice script exceeds 256 decisions");
        }
        for enemy in &enemies {
            let half = enemy.size / 2.0;
            if !within(enemy.size, 2.0, 16.0)
                || !within(enemy.x, half, 128.0 - half)
                || !within(enemy.y, half, 128.0 - half)
                || enemy.segments.len() > 64
            {
                return Err("invalid scripted enemy geometry");
            }
            for segment in &enemy.segments {
                if !within(segment.x, half, 128.0 - half)
                    || !within(segment.y, half, 128.0 - half)
                    || !(1..=3600).contains(&segment.frames)
                {
                    return Err("invalid enemy segment");
                }
            }
            if enemy.looping
                && enemy
                    .segments
                    .last()
                    .is_none_or(|last| last.x != enemy.x || last.y != enemy.y)
            {
                return Err("looping enemy path must end at its start");
            }
        }
        config.scenario.enemy_mode = 3;
        config.patterns_enabled = config.scenario.permanent_pattern != 0;
        config.powerups_enabled = false;
        Ok(Self {
            game: NativeGame::new(config),
            config,
            start,
            commands,
            enemies,
            command: 0,
            command_steps: 0,
            tick: 0,
            ended: true,
        })
    }

    pub fn reset(&mut self, seed: u32) -> Result<IndexedFramebuffer, &'static str> {
        if seed > 32767 {
            return Err("seed outside 0..32767");
        }
        self.config.seed = seed;
        self.game = NativeGame::new(self.config);
        for frame in 0..13 {
            let mask = if frame == 0 { BUTTON_X_MASK } else { 0 };
            self.game
                .advance_frame_pixels(mask, mask)
                .map_err(|_| "startup failed")?;
        }
        self.game.player.x = PicoFixed::from_f32(self.start.0);
        self.game.player.y = PicoFixed::from_f32(self.start.1);
        self.game.player.vx = PicoFixed::ZERO;
        self.game.player.vy = PicoFixed::ZERO;
        self.game.particles.clear();
        self.command = 0;
        self.command_steps = 0;
        self.tick = 0;
        self.ended = false;
        self.place_enemies();
        Ok(self.game.render_current_frame().framebuffer)
    }

    fn place_enemies(&mut self) {
        // External script positions are applied at each native frame. Normal
        // spawning and pursuit AI are disabled by mode3, not by pixel overlays.
        self.game.enemies = self
            .enemies
            .iter()
            .map(|actor| {
                let mut x = actor.x;
                let mut y = actor.y;
                let total: u32 = actor.segments.iter().map(|segment| segment.frames).sum();
                let mut remaining = if actor.looping {
                    self.tick % total
                } else {
                    self.tick
                };
                for segment in &actor.segments {
                    if remaining <= segment.frames {
                        let fraction = remaining as f32 / segment.frames as f32;
                        x += (segment.x - x) * fraction;
                        y += (segment.y - y) * fraction;
                        break;
                    }
                    remaining -= segment.frames;
                    x = segment.x;
                    y = segment.y;
                }
                let size = PicoFixed::from_f32(actor.size);
                EnemyState {
                    size,
                    isizing: false,
                    ..EnemyState::normal(
                        PicoFixed::from_f32(x - actor.size / 2.0),
                        PicoFixed::from_f32(y - actor.size / 2.0),
                        size,
                    )
                }
            })
            .collect();
    }

    fn arrived(&self, x: f32, y: f32) -> bool {
        let player = self.game.player;
        let dx = player.x.to_double() - f64::from(x);
        let dy = player.y.to_double() - f64::from(y);
        dx * dx + dy * dy <= 4.0 && player.vx.to_double().abs() + player.vy.to_double().abs() <= 0.2
    }

    fn waypoint_action(&self, x: f32, y: f32) -> Action {
        let mut best = (f64::INFINITY, Action::Neutral);
        for action in Action::ALL {
            let mut hypothetical = self.game.clone();
            // Evaluate the real native movement law, including friction and
            // bounds. Neutral rollout estimates stopping position; no obstacle
            // planning or learned policy is claimed by this authoring helper.
            hypothetical
                .input
                .advance(action.mask())
                .expect("valid action");
            for _ in 0..4 {
                hypothetical.update_player();
            }
            hypothetical.input.advance(0).expect("neutral action");
            for _ in 0..24 {
                hypothetical.update_player();
            }
            let dx = hypothetical.player.x.to_double() - f64::from(x);
            let dy = hypothetical.player.y.to_double() - f64::from(y);
            let cost = dx * dx + dy * dy;
            if cost < best.0 {
                best = (cost, action);
            }
        }
        best.1
    }

    pub fn step(&mut self) -> Result<PracticeStep, &'static str> {
        if self.ended {
            return Err("practice episode ended; reset before stepping");
        }
        let command = self.commands[self.command].clone();
        let action = match command {
            PracticeCommand::Action { action, .. } => action,
            PracticeCommand::MoveTo { x, y, .. } => self.waypoint_action(x, y),
        };
        let mut pixels = None;
        for _ in 0..4 {
            self.tick += 1;
            self.place_enemies();
            let frame = self
                .game
                .advance_frame_pixels(action.mask(), action.mask())
                .map_err(|_| "native step failed")?;
            pixels = Some(frame.pixels);
            if frame.done {
                break;
            }
        }
        self.command_steps += 1;
        let mut failed = false;
        let complete = match command {
            PracticeCommand::Action { decisions, .. } => self.command_steps >= decisions,
            PracticeCommand::MoveTo {
                x,
                y,
                max_decisions,
            } => {
                let arrived = self.arrived(x, y);
                failed = !arrived && self.command_steps >= max_decisions;
                arrived
            }
        };
        if complete {
            self.command += 1;
            self.command_steps = 0;
        }
        let terminated = self.game.lifecycle.dead;
        failed &= !terminated;
        let finished = !terminated && !failed && self.command == self.commands.len();
        self.ended = terminated || failed || finished;
        Ok(PracticeStep {
            pixels: pixels.expect("four-frame step produced a frame"),
            action,
            terminated,
            finished,
            failed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> NativeConfig {
        let mut config = NativeConfig::new(42);
        config.scenario.invulnerable = true;
        config
    }
    fn moving_enemy() -> PracticeEnemy {
        PracticeEnemy {
            x: 64.0,
            y: 16.0,
            size: 6.0,
            looping: true,
            segments: vec![
                PracticeSegment {
                    x: 64.0,
                    y: 112.0,
                    frames: 80,
                },
                PracticeSegment {
                    x: 64.0,
                    y: 16.0,
                    frames: 80,
                },
            ],
        }
    }
    #[test]
    fn player_waypoints_use_actions_and_replay_exactly() {
        let commands = vec![
            PracticeCommand::MoveTo {
                x: 92.0,
                y: 78.0,
                max_decisions: 64,
            },
            PracticeCommand::Action {
                action: Action::Neutral,
                decisions: 4,
            },
        ];
        let mut simulation =
            PracticeSimulation::new(config(), (32.0, 48.0), commands, vec![moving_enemy()], 4)
                .unwrap();
        let first = simulation.reset(42).unwrap();
        assert_eq!(simulation.game.player.x, PicoFixed::from_int(32));
        let mut trace = Vec::new();
        loop {
            let step = simulation.step().unwrap();
            assert!(
                !step.failed && !step.terminated,
                "position={:?}",
                simulation.game.player
            );
            trace.push((step.action, step.pixels));
            if step.finished {
                break;
            }
        }
        assert!(simulation.arrived(92.0, 78.0));
        assert!(trace.iter().any(|(action, _)| *action != Action::Neutral));
        assert!(simulation.step().is_err());
        assert_eq!(simulation.reset(42).unwrap(), first);
        for (action, pixels) in trace {
            let step = simulation.step().unwrap();
            assert_eq!((step.action, step.pixels), (action, pixels));
        }
    }
    #[test]
    fn moving_and_static_enemies_follow_native_frame_paths() {
        let stationary = PracticeEnemy {
            x: 20.0,
            y: 20.0,
            size: 4.0,
            looping: false,
            segments: vec![],
        };
        let mut simulation = PracticeSimulation::new(
            config(),
            (32.0, 64.0),
            vec![PracticeCommand::Action {
                action: Action::Neutral,
                decisions: 60,
            }],
            vec![moving_enemy(), stationary],
            4,
        )
        .unwrap();
        simulation.reset(42).unwrap();
        for _ in 0..20 {
            simulation.step().unwrap();
        }
        assert_eq!(simulation.game.enemies[0].y, PicoFixed::from_int(109));
        assert_eq!(simulation.game.enemies[1].y, PicoFixed::from_int(18));
        for _ in 0..20 {
            simulation.step().unwrap();
        }
        assert_eq!(simulation.game.enemies[0].y, PicoFixed::from_int(13));
    }
    #[test]
    fn death_and_waypoint_timeout_are_not_successful_goals() {
        let mut lethal = config();
        lethal.scenario.invulnerable = false;
        let enemy = PracticeEnemy {
            x: 64.0,
            y: 64.0,
            size: 8.0,
            looping: false,
            segments: vec![],
        };
        let commands = vec![PracticeCommand::MoveTo {
            x: 96.0,
            y: 96.0,
            max_decisions: 1,
        }];
        let mut simulation =
            PracticeSimulation::new(lethal, (64.0, 64.0), commands, vec![enemy], 4).unwrap();
        simulation.reset(42).unwrap();
        let step = simulation.step().unwrap();
        assert!(step.terminated && !step.finished && !step.failed);
        assert!(simulation.step().is_err());
        let mut simulation = PracticeSimulation::new(
            config(),
            (4.0, 4.0),
            vec![PracticeCommand::MoveTo {
                x: 124.0,
                y: 124.0,
                max_decisions: 1,
            }],
            vec![],
            4,
        )
        .unwrap();
        simulation.reset(42).unwrap();
        let step = simulation.step().unwrap();
        assert!(step.failed && !step.finished && !step.terminated);
        assert!(simulation.step().is_err());
    }
    #[test]
    fn invalid_paths_and_budgets_rejected_before_reset() {
        let mut enemy = moving_enemy();
        enemy.segments[0].frames = 0;
        assert!(PracticeSimulation::new(
            config(),
            (64.0, 64.0),
            vec![PracticeCommand::Action {
                action: Action::Neutral,
                decisions: 1
            }],
            vec![enemy],
            4
        )
        .is_err());
        assert!(PracticeSimulation::new(
            config(),
            (64.0, 64.0),
            vec![PracticeCommand::MoveTo {
                x: f32::NAN,
                y: 64.0,
                max_decisions: 1
            }],
            vec![],
            4
        )
        .is_err());
    }
}
