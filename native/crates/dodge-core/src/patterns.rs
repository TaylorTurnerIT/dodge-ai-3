use crate::{PicoFixed, PicoRng};

/// A source pattern target. String targets from the cartridge are represented
/// as explicit state transitions rather than remaining as dynamically typed
/// Lua values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PatternTarget {
    Move {
        x: PicoFixed,
        y: PicoFixed,
        width: PicoFixed,
        height: PicoFixed,
    },
    Wait(PicoFixed),
    SetFyou(bool),
    SetSpawns(Vec<SpawnPoint>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpawnPoint {
    pub x: PicoFixed,
    pub y: PicoFixed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WarningLine {
    pub x0: PicoFixed,
    pub y0: PicoFixed,
    pub x1: PicoFixed,
    pub y1: PicoFixed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternRect {
    pub x: PicoFixed,
    pub y: PicoFixed,
    pub width: PicoFixed,
    pub height: PicoFixed,
    pub speed: PicoFixed,
    pub dx: PicoFixed,
    pub dy: PicoFixed,
    pub targets: Vec<PatternTarget>,
    pub target_index: usize,
    pub wait: PicoFixed,
    pub shown: bool,
    pub sh: PicoFixed,
    pub warnings: Vec<WarningLine>,
    pub collision_done: bool,
    pub finished: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternState {
    pub id: u8,
    pub mins: PicoFixed,
    pub maxs: PicoFixed,
    pub probability: PicoFixed,
    pub variants: Vec<u8>,
    pub smooth: bool,
    pub pattern_type: u8,
    pub bounce_cap: bool,
    pub spawn_enabled: bool,
    pub automatic_variant: Option<u8>,
    pub special: PicoFixed,
    pub counter: u32,
    pub timer: PicoFixed,
    pub rects: Vec<PatternRect>,
}

impl PatternRect {
    fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            x: PicoFixed::from_int(x),
            y: PicoFixed::from_int(y),
            width: PicoFixed::from_int(width),
            height: PicoFixed::from_int(height),
            speed: PicoFixed::from_int(12),
            dx: PicoFixed::ZERO,
            dy: PicoFixed::ZERO,
            targets: vec![PatternTarget::Wait(PicoFixed::from_f32(8.5))],
            target_index: 0,
            wait: PicoFixed::ZERO,
            shown: true,
            sh: PicoFixed::ZERO,
            warnings: Vec::new(),
            collision_done: false,
            finished: false,
        }
    }

    fn with_targets(mut self, targets: Vec<PatternTarget>) -> Self {
        self.targets = targets;
        self
    }

    fn with_motion(mut self, speed: f32, dx: i32, dy: i32) -> Self {
        self.speed = PicoFixed::from_f32(speed);
        self.dx = PicoFixed::from_int(dx);
        self.dy = PicoFixed::from_int(dy);
        self
    }
}

impl PatternState {
    fn new(id: u8, rects: Vec<PatternRect>) -> Self {
        Self {
            id,
            mins: PicoFixed::ZERO,
            maxs: PicoFixed::from_int(32_767),
            probability: PicoFixed::from_int(15),
            variants: Vec::new(),
            smooth: false,
            pattern_type: 0,
            bounce_cap: false,
            spawn_enabled: false,
            automatic_variant: None,
            special: PicoFixed::ZERO,
            counter: 0,
            timer: PicoFixed::ZERO,
            rects,
        }
    }

    fn with_range(mut self, mins: i32, maxs: i32) -> Self {
        self.mins = PicoFixed::from_int(mins);
        self.maxs = PicoFixed::from_int(maxs);
        self
    }

    fn with_variants(mut self, variants: &[u8], automatic: Option<u8>) -> Self {
        self.variants = variants.to_vec();
        self.automatic_variant = automatic;
        self
    }

    fn with_type(mut self, pattern_type: u8) -> Self {
        self.pattern_type = pattern_type;
        self
    }

    fn with_special(mut self, special: f32) -> Self {
        self.special = PicoFixed::from_f32(special);
        self
    }
}

pub(crate) fn init_patterns(rng: &mut PicoRng) -> Vec<PatternState> {
    let mut patterns = vec![
        PatternState::new(
            1,
            vec![
                PatternRect::new(28, 16, 20, 20),
                PatternRect::new(80, 92, 20, 20),
            ],
        )
        .with_variants(&[2], None),
        PatternState::new(
            2,
            vec![
                PatternRect::new(28, 92, 20, 20),
                PatternRect::new(80, 16, 20, 20),
            ],
        )
        .with_variants(&[1], None),
        PatternState::new(
            3,
            vec![
                PatternRect::new(56, 0, 16, 42),
                PatternRect::new(56, 85, 16, 42),
            ],
        )
        .with_range(10, 60)
        .with_variants(&[4], Some(4)),
        PatternState::new(4, Vec::new()).with_variants(&[3], None),
        PatternState::new(
            5,
            vec![PatternRect::new(131, 32, 24, 64)
                .with_motion(0.7, -1, 0)
                .with_targets(vec![PatternTarget::Move {
                    x: PicoFixed::from_int(-26),
                    y: PicoFixed::from_int(32),
                    width: PicoFixed::from_int(24),
                    height: PicoFixed::from_int(64),
                }])],
        )
        .with_variants(&[6, 7, 8], Some(6))
        .with_type(1),
        PatternState::new(6, Vec::new())
            .with_variants(&[5, 7, 8], None)
            .with_type(1),
        PatternState::new(
            7,
            vec![PatternRect::new(-28, 32, 64, 24)
                .with_motion(0.7, 0, 1)
                .with_targets(vec![PatternTarget::Move {
                    x: PicoFixed::from_int(32),
                    y: PicoFixed::from_int(130),
                    width: PicoFixed::from_int(64),
                    height: PicoFixed::from_int(24),
                }])],
        )
        .with_variants(&[8, 5, 6], Some(8))
        .with_type(1),
        PatternState::new(8, Vec::new())
            .with_variants(&[7, 6, 5], None)
            .with_type(1),
        PatternState::new(
            9,
            vec![
                PatternRect::new(0, 131, 63, 16)
                    .with_motion(0.7, 0, -1)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::ZERO,
                        y: PicoFixed::from_int(-20),
                        width: PicoFixed::from_int(63),
                        height: PicoFixed::from_int(16),
                    }]),
                PatternRect::new(64, -20, 63, 16)
                    .with_motion(0.7, 0, 1)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::from_int(64),
                        y: PicoFixed::from_int(144),
                        width: PicoFixed::from_int(63),
                        height: PicoFixed::from_int(16),
                    }]),
            ],
        )
        .with_variants(&[10, 11, 12], Some(10))
        .with_type(1),
        PatternState::new(10, Vec::new())
            .with_variants(&[9, 11, 12], None)
            .with_type(1),
        PatternState::new(
            11,
            vec![
                PatternRect::new(131, 0, 16, 63)
                    .with_motion(0.7, -1, 0)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::from_int(-18),
                        y: PicoFixed::ZERO,
                        width: PicoFixed::from_int(16),
                        height: PicoFixed::from_int(63),
                    }]),
                PatternRect::new(-20, 64, 16, 63)
                    .with_motion(0.7, 1, 0)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::from_int(144),
                        y: PicoFixed::from_int(64),
                        width: PicoFixed::from_int(16),
                        height: PicoFixed::from_int(63),
                    }]),
            ],
        )
        .with_variants(&[9, 10, 12], Some(12))
        .with_type(1),
        PatternState::new(12, Vec::new())
            .with_variants(&[11, 10, 9], None)
            .with_type(1),
        PatternState::new(
            13,
            vec![
                PatternRect::new(0, 0, 127, 32).with_targets(vec![
                    PatternTarget::SetFyou(false),
                    PatternTarget::SetSpawns(Vec::new()),
                    PatternTarget::SetSpawns(vec![spawn(-10, 62), spawn(138, 62)]),
                    PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                    PatternTarget::SetSpawns(vec![
                        spawn(-10, -10),
                        spawn(138, -10),
                        spawn(-10, 138),
                        spawn(138, 138),
                    ]),
                ]),
                PatternRect::new(0, 96, 127, 31),
            ],
        )
        .with_range(50, 170)
        .with_variants(&[14], None),
        PatternState::new(
            14,
            vec![
                PatternRect::new(0, 0, 32, 127).with_targets(vec![
                    PatternTarget::SetFyou(false),
                    PatternTarget::SetSpawns(Vec::new()),
                    PatternTarget::SetSpawns(vec![spawn(62, -10), spawn(62, 138)]),
                    PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                    PatternTarget::SetSpawns(vec![
                        spawn(-10, -10),
                        spawn(138, -10),
                        spawn(-10, 138),
                        spawn(138, 138),
                    ]),
                ]),
                PatternRect::new(96, 0, 31, 127),
            ],
        )
        .with_range(50, 120)
        .with_variants(&[13], None),
        PatternState::new(15, Vec::new())
            .with_range(100, 200)
            .with_variants(&[16], None)
            .with_special(1.0),
        PatternState::new(16, Vec::new())
            .with_range(100, 200)
            .with_variants(&[15], None)
            .with_special(1.1),
        PatternState::new(
            17,
            vec![PatternRect::new(56, 56, 16, 16).with_motion(34.0, 0, 0)],
        )
        .with_range(25, 90),
        PatternState::new(18, Vec::new())
            .with_range(0, 60)
            .with_special(5.0),
        PatternState::new(19, Vec::new())
            .with_range(60, 120)
            .with_special(2.0)
            .with_type(0),
        PatternState::new(20, Vec::new())
            .with_range(60, 120)
            .with_special(3.0)
            .with_type(0),
        PatternState::new(
            21,
            vec![
                PatternRect::new(20, 62, 26, 4),
                PatternRect::new(81, 62, 26, 4),
                PatternRect::new(62, 20, 4, 26),
                PatternRect::new(62, 81, 4, 26),
            ],
        )
        .with_range(17, 60),
        PatternState::new(
            22,
            vec![
                PatternRect::new(19, 62, 90, 4),
                PatternRect::new(62, 19, 4, 90),
            ],
        )
        .with_range(60, 120),
        PatternState::new(
            23,
            vec![PatternRect::new(-50, -50, 0, 0).with_targets(vec![
                PatternTarget::SetSpawns(vec![
                    spawn(-10, -10),
                    spawn(138, -10),
                    spawn(-10, 138),
                    spawn(138, 138),
                    spawn(64, -10),
                    spawn(138, 64),
                    spawn(64, 138),
                    spawn(-10, 64),
                ]),
                PatternTarget::Wait(PicoFixed::from_int(10)),
            ])],
        )
        .with_range(120, 32_767)
        .with_type(1),
        PatternState::new(24, Vec::new())
            .with_range(120, 32_767)
            .with_special(6.0),
        PatternState::new(
            25,
            vec![
                PatternRect::new(0, 0, 127, 20).with_targets(vec![
                    PatternTarget::SetFyou(false),
                    PatternTarget::SetSpawns(vec![
                        spawn(138, 83),
                        spawn(-10, 83),
                        spawn(-10, 38),
                        spawn(138, 38),
                    ]),
                    PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                ]),
                PatternRect::new(0, 107, 127, 20),
                PatternRect::new(24, 62, 30, 4),
                PatternRect::new(74, 62, 30, 4),
            ],
        )
        .with_range(40, 100)
        .with_variants(&[26], None),
        PatternState::new(
            26,
            vec![
                PatternRect::new(0, 0, 20, 127).with_targets(vec![
                    PatternTarget::SetFyou(false),
                    PatternTarget::SetSpawns(vec![
                        spawn(83, 128),
                        spawn(83, -10),
                        spawn(38, -10),
                        spawn(38, 138),
                    ]),
                    PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                ]),
                PatternRect::new(107, 0, 20, 127),
                PatternRect::new(62, 24, 4, 30),
                PatternRect::new(62, 74, 4, 30),
            ],
        )
        .with_range(40, 100)
        .with_variants(&[25], None),
        PatternState::new(27, Vec::new())
            .with_range(10, 60)
            .with_special(7.0),
        PatternState::new(
            28,
            vec![
                PatternRect::new(32, 0, 63, 16),
                PatternRect::new(32, 111, 63, 16),
                PatternRect::new(0, 32, 16, 63),
                PatternRect::new(111, 32, 16, 63),
            ],
        )
        .with_range(0, 60),
        PatternState::new(
            29,
            vec![
                PatternRect::new(0, 62, 32, 4),
                PatternRect::new(95, 62, 32, 4),
                PatternRect::new(62, 0, 4, 32),
                PatternRect::new(62, 95, 4, 32),
            ],
        )
        .with_range(50, 100),
        PatternState::new(
            30,
            vec![
                PatternRect::new(0, -20, 32, 16)
                    .with_motion(0.7, 0, 1)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::ZERO,
                        y: PicoFixed::from_int(128),
                        width: PicoFixed::from_int(40),
                        height: PicoFixed::from_int(16),
                    }]),
                PatternRect::new(87, -20, 40, 16)
                    .with_motion(0.7, 0, 1)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::from_int(87),
                        y: PicoFixed::from_int(128),
                        width: PicoFixed::from_int(40),
                        height: PicoFixed::from_int(16),
                    }]),
            ],
        )
        .with_variants(&[31, 32, 33], Some(31))
        .with_type(1),
        PatternState::new(31, Vec::new())
            .with_variants(&[30, 32, 33], None)
            .with_type(1),
        PatternState::new(
            32,
            vec![
                PatternRect::new(130, 0, 16, 40)
                    .with_motion(0.7, -1, 0)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::from_int(-20),
                        y: PicoFixed::ZERO,
                        width: PicoFixed::from_int(16),
                        height: PicoFixed::from_int(40),
                    }]),
                PatternRect::new(130, 87, 16, 40)
                    .with_motion(0.7, -1, 0)
                    .with_targets(vec![PatternTarget::Move {
                        x: PicoFixed::from_int(-20),
                        y: PicoFixed::from_int(87),
                        width: PicoFixed::from_int(16),
                        height: PicoFixed::from_int(40),
                    }]),
            ],
        )
        .with_variants(&[30, 31, 33], Some(33))
        .with_type(1),
        PatternState::new(33, Vec::new())
            .with_variants(&[30, 31, 32], None)
            .with_type(1),
        PatternState::new(
            34,
            vec![
                PatternRect::new(48, 16, 16, 16),
                PatternRect::new(16, 48, 16, 16),
                PatternRect::new(84, 84, 43, 43).with_targets(vec![
                    PatternTarget::SetSpawns(vec![
                        spawn(-10, -10),
                        spawn(138, -10),
                        spawn(-10, 138),
                    ]),
                    PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                ]),
            ],
        )
        .with_range(120, 32_767)
        .with_variants(&[34, 35, 36], None),
        PatternState::new(
            35,
            vec![
                PatternRect::new(20, 48, 8, 32),
                PatternRect::new(48, 20, 32, 8),
                PatternRect::new(48, 100, 32, 8),
                PatternRect::new(100, 48, 8, 32),
            ],
        )
        .with_range(80, 160),
        PatternState::new(36, vec![PatternRect::new(54, 53, 20, 74)])
            .with_range(20, 90)
            .with_variants(&[37, 38, 39], Some(37)),
        PatternState::new(37, Vec::new()).with_variants(&[36, 38, 39], None),
        PatternState::new(38, vec![PatternRect::new(0, 54, 74, 20)])
            .with_range(20, 90)
            .with_variants(&[36, 37, 39], Some(39)),
        PatternState::new(39, Vec::new()).with_variants(&[36, 37, 38], None),
    ];

    add_special_rects(&mut patterns, rng);
    apply_defaults(&mut patterns);
    add_automatic_variants(&mut patterns);
    patterns
}

fn spawn(x: i32, y: i32) -> SpawnPoint {
    SpawnPoint {
        x: PicoFixed::from_int(x),
        y: PicoFixed::from_int(y),
    }
}

fn add_special_rects(patterns: &mut [PatternState], rng: &mut PicoRng) {
    if let Some(pattern) = patterns.get_mut(14) {
        for x in (-8..=120).step_by(32) {
            pattern
                .rects
                .push(PatternRect::new(x, 61, 16, 6).with_targets(vec![
                    PatternTarget::SetFyou(false),
                    PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                ]));
        }
    }
    if let Some(pattern) = patterns.get_mut(15) {
        for y in (-8..=120).step_by(32) {
            pattern
                .rects
                .push(PatternRect::new(60, y, 8, 16).with_targets(vec![
                    PatternTarget::SetFyou(false),
                    PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                ]));
        }
    }
    if let Some(pattern) = patterns.get_mut(18) {
        let exits = vec![spawn(64, -4), spawn(129, 64), spawn(64, 129), spawn(-4, 64)];
        for x in [16, 56, 96] {
            for y in [16, 56, 96] {
                pattern.rects.push(
                    PatternRect::new(x, y, 15, 15)
                        .with_motion(1.0, 0, 0)
                        .with_targets(vec![
                            PatternTarget::SetFyou(false),
                            PatternTarget::SetSpawns(exits.clone()),
                            PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                        ]),
                );
            }
        }
    }
    if let Some(pattern) = patterns.get_mut(19) {
        let exits = vec![spawn(64, -4), spawn(129, 64), spawn(64, 129), spawn(-4, 64)];
        for x in [16, 61, 106] {
            for y in [16, 61, 106] {
                pattern
                    .rects
                    .push(PatternRect::new(x, y, 8, 8).with_targets(vec![
                        PatternTarget::SetFyou(false),
                        PatternTarget::SetSpawns(exits.clone()),
                        PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                    ]));
            }
        }
        for x in [39, 84] {
            for y in [39, 84] {
                pattern
                    .rects
                    .push(PatternRect::new(x, y, 8, 8).with_motion(1.0, 0, 0));
            }
        }
    }
    if let Some(pattern) = patterns.get_mut(17) {
        for x in [0, 75] {
            for y in [0, 111] {
                pattern
                    .rects
                    .push(PatternRect::new(x, y, 52, 16).with_targets(vec![
                        PatternTarget::SetSpawns(vec![
                            spawn(64, -4),
                            spawn(129, 64),
                            spawn(64, 129),
                            spawn(-4, 64),
                        ]),
                        PatternTarget::Wait(PicoFixed::from_f32(8.5)),
                    ]));
            }
        }
        for x in [0, 111] {
            for y in [0, 75] {
                pattern.rects.push(PatternRect::new(x, y, 16, 52));
            }
        }
        for x in [50, 75] {
            for y in [-9, 129] {
                pattern.rects.push(PatternRect::new(x, y, 2, 7));
            }
        }
        for y in [50, 75] {
            for x in [-9, 129] {
                pattern.rects.push(PatternRect::new(x, y, 7, 2));
            }
        }
    }
    if let Some(pattern) = patterns.get_mut(23) {
        let locations = [(16, 16, 14, 14), (72, 16, 10, 10), (26, 72, 14, 10)];
        let ranges = [
            (16, 32),
            (16, 32),
            (14, 24),
            (14, 24),
            (72, 95),
            (16, 36),
            (10, 20),
            (10, 20),
            (26, 86),
            (72, 82),
            (14, 19),
            (10, 18),
        ];
        let mut values = [PicoFixed::ZERO; 12];
        for (value, (minimum, maximum)) in values.iter_mut().zip(ranges) {
            *value =
                PicoFixed::from_int(minimum).add(rng.rnd(PicoFixed::from_int(maximum - minimum)));
        }
        let mut range_iter = values.into_iter();
        for _ in locations {
            let Some(x) = range_iter.next() else { return };
            let Some(y) = range_iter.next() else { return };
            let Some(width) = range_iter.next() else {
                return;
            };
            let Some(height) = range_iter.next() else {
                return;
            };
            pattern.rects.push(PatternRect {
                x,
                y,
                width,
                height,
                ..PatternRect::new(0, 0, 0, 0)
            });
        }
    }
    if let Some(pattern) = patterns.get_mut(26) {
        let first = [
            (14, 32),
            (20, 32),
            (20, 25),
            (25, 30),
            (50, 90),
            (76, 90),
            (10, 20),
            (10, 20),
        ];
        let mut values = [PicoFixed::ZERO; 8];
        for (value, (minimum, maximum)) in values.iter_mut().zip(first) {
            *value =
                PicoFixed::from_int(minimum).add(rng.rnd(PicoFixed::from_int(maximum - minimum)));
        }
        let [first_x, first_y, first_width, first_height, second_x, second_y, second_width, second_height] =
            values;
        pattern.rects.push(PatternRect {
            x: first_x,
            y: first_y,
            width: first_width,
            height: first_height,
            ..PatternRect::new(0, 0, 0, 0)
        });
        let mut other = [second_x, second_y, second_width, second_height];
        if rng.rnd(PicoFixed::ONE) > PicoFixed::from_f32(0.5) {
            for (value, (minimum, maximum)) in
                other
                    .iter_mut()
                    .zip([(66, 90), (50, 90), (10, 20), (10, 20)])
            {
                *value = PicoFixed::from_int(minimum)
                    .add(rng.rnd(PicoFixed::from_int(maximum - minimum)));
            }
        }
        let [x, y, width, height] = other;
        pattern.rects.push(PatternRect {
            x,
            y,
            width,
            height,
            ..PatternRect::new(0, 0, 0, 0)
        });
    }
}

fn apply_defaults(patterns: &mut [PatternState]) {
    for pattern in patterns {
        let divisor = (pattern.variants.len() + 1) as f32;
        let base = if pattern.pattern_type == 1 {
            17.0
        } else {
            15.0
        };
        pattern.probability = PicoFixed::from_f32(base / divisor);
        for rect in &mut pattern.rects {
            if rect.targets.is_empty() {
                rect.targets
                    .push(PatternTarget::Wait(PicoFixed::from_f32(8.5)));
            }
            rect.target_index = 0;
            rect.wait = PicoFixed::ZERO;
            rect.shown = true;
            rect.sh = PicoFixed::ZERO;
            rect.warnings.clear();
            rect.collision_done = false;
            rect.finished = false;
        }
        pattern.counter = 0;
        pattern.timer = PicoFixed::ZERO;
    }
}

fn add_automatic_variants(patterns: &mut [PatternState]) {
    let source = patterns.to_vec();
    for pattern in source {
        let Some(target_id) = pattern.automatic_variant else {
            continue;
        };
        let Some(target) = patterns.get_mut(usize::from(target_id - 1)) else {
            continue;
        };
        for rect in pattern.rects {
            let targets = rect
                .targets
                .iter()
                .map(transpose_target)
                .collect::<Vec<_>>();
            let mut variant = PatternRect::new(0, 0, 0, 0);
            variant.x = rect.y;
            variant.y = rect.x;
            variant.width = rect.height;
            variant.height = rect.width;
            variant.speed = rect.speed;
            variant.dx = rect.dy;
            variant.dy = rect.dx;
            variant.targets = targets;
            target.rects.push(variant);
        }
        if let Some(target) = patterns.get_mut(usize::from(target_id - 1)) {
            target.mins = pattern.mins;
            target.maxs = pattern.maxs;
        }
    }
}

fn transpose_target(target: &PatternTarget) -> PatternTarget {
    match target {
        PatternTarget::Move {
            x,
            y,
            width,
            height,
        } => PatternTarget::Move {
            x: *y,
            y: *x,
            width: *height,
            height: *width,
        },
        PatternTarget::Wait(seconds) => PatternTarget::Wait(*seconds),
        PatternTarget::SetFyou(value) => PatternTarget::SetFyou(*value),
        PatternTarget::SetSpawns(points) => PatternTarget::SetSpawns(
            points
                .iter()
                .map(|point| SpawnPoint {
                    x: point.y,
                    y: point.x,
                })
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::init_patterns;
    use crate::{PatternTarget, PicoFixed, PicoRng};

    #[test]
    fn v155_special_values_and_automatic_variants_are_independent() {
        let mut rng = PicoRng::new(42);
        let patterns = init_patterns(&mut rng);
        assert_eq!(patterns.get(2).and_then(|p| p.automatic_variant), Some(4));
        assert_eq!(patterns.get(2).map(|p| p.special), Some(PicoFixed::ZERO));
        assert_eq!(
            patterns.get(14).map(|p| p.special),
            Some(PicoFixed::from_int(1))
        );
        assert_eq!(
            patterns.get(15).map(|p| p.special),
            Some(PicoFixed::from_f32(1.1))
        );
        assert_eq!(patterns.get(24).and_then(|p| p.automatic_variant), None);
        assert_eq!(patterns.get(24).map(|p| p.special), Some(PicoFixed::ZERO));
    }

    #[test]
    fn v157_source_pattern_inventory_contains_generated_and_dynamic_shapes() {
        let mut rng = PicoRng::new(42);
        let patterns = init_patterns(&mut rng);

        assert_eq!(patterns.len(), 39);
        assert!(patterns
            .iter()
            .enumerate()
            .all(|(index, pattern)| pattern.id == u8::try_from(index + 1).unwrap_or(0)));

        let id3 = patterns.get(2).and_then(|pattern| pattern.rects.first());
        let id4 = patterns.get(3).and_then(|pattern| pattern.rects.first());
        assert_eq!(id3.map(|rect| rect.x), Some(PicoFixed::from_int(56)));
        assert_eq!(id3.map(|rect| rect.y), Some(PicoFixed::ZERO));
        assert_eq!(id4.map(|rect| rect.x), Some(PicoFixed::ZERO));
        assert_eq!(id4.map(|rect| rect.y), Some(PicoFixed::from_int(56)));
        assert_eq!(id4.map(|rect| rect.width), Some(PicoFixed::from_int(42)));
        assert_eq!(id4.map(|rect| rect.height), Some(PicoFixed::from_int(16)));

        assert_eq!(patterns.get(14).map(|pattern| pattern.rects.len()), Some(5));
        assert_eq!(patterns.get(15).map(|pattern| pattern.rects.len()), Some(5));
        assert_eq!(patterns.get(16).map(|pattern| pattern.rects.len()), Some(1));
        assert_eq!(
            patterns.get(17).map(|pattern| pattern.rects.len()),
            Some(16)
        );
        assert_eq!(patterns.get(18).map(|pattern| pattern.rects.len()), Some(9));
        assert_eq!(
            patterns.get(19).map(|pattern| pattern.rects.len()),
            Some(13)
        );
        assert_eq!(patterns.get(23).map(|pattern| pattern.rects.len()), Some(3));
        assert_eq!(patterns.get(26).map(|pattern| pattern.rects.len()), Some(2));

        let special_rect = patterns.get(14).and_then(|pattern| pattern.rects.first());
        assert!(special_rect.is_some_and(|rect| {
            rect.targets
                .first()
                .is_some_and(|target| matches!(target, PatternTarget::SetFyou(false)))
        }));
        assert!(special_rect.is_some_and(|rect| {
            rect.targets
                .get(1)
                .is_some_and(|target| matches!(target, PatternTarget::Wait(_)))
        }));
    }
}
