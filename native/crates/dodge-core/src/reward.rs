//! Optional reward geometry. This module never mutates game state or observations.

/// Separate nonnegative unit costs; callers apply explicit experiment weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryCosts {
    pub edge: f32,
    pub corner: f32,
}

/// Signed components before experiment weights. Counts belong to one decision,
/// not lifetime counters; survival retains the native survival-frame scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardTerms {
    pub survival: f32,
    pub death: f32,
    pub pickups: f32,
    pub enemy_deaths: f32,
    pub edge: f32,
    pub corner: f32,
}

impl RewardTerms {
    pub fn new(
        survival_frames: u32,
        death_events: u32,
        pickups: u32,
        enemy_deaths: u32,
        boundary: BoundaryCosts,
        frames_advanced: u32,
    ) -> Option<Self> {
        if survival_frames > frames_advanced
            || ![boundary.edge, boundary.corner]
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        {
            return None;
        }
        Some(Self {
            survival: survival_frames as f32,
            // Several overlapping hazards can emit Death in one native frame.
            // They terminate one life, so charge once, not once per collision.
            death: if death_events > 0 { -1.0 } else { 0.0 },
            pickups: pickups as f32,
            enemy_deaths: enemy_deaths as f32,
            edge: -boundary.edge * frames_advanced as f32,
            corner: -boundary.corner * frames_advanced as f32,
        })
    }

    /// Order: survival, death, pickups, enemy deaths, edge, corner.
    /// Spatial terms use decision-end position times frames advanced.
    pub fn weighted(self, weights: [f32; 6]) -> Option<[f32; 6]> {
        if !weights.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return None;
        }
        let values = [
            self.survival,
            self.death,
            self.pickups,
            self.enemy_deaths,
            self.edge,
            self.corner,
        ];
        let result = std::array::from_fn(|i| {
            values.get(i).copied().unwrap_or(0.0) * weights.get(i).copied().unwrap_or(0.0)
        });
        result.iter().all(|v| v.is_finite()).then_some(result)
    }
}

/// Smooth boundary bands measured in native player-center coordinates.
/// The interior is flat. Corner cost requires proximity to two adjacent sides.
/// Widths and bounds are explicit rather than silently tied to framebuffer size.
pub fn boundary_costs(
    x: f32,
    y: f32,
    minimum: f32,
    maximum: f32,
    edge_width: f32,
    corner_width: f32,
) -> Option<BoundaryCosts> {
    if ![x, y, minimum, maximum, edge_width, corner_width]
        .iter()
        .all(|value| value.is_finite())
        || minimum >= maximum
        || edge_width <= 0.0
        || corner_width < edge_width
        || corner_width > (maximum - minimum) / 2.0
        || !(minimum..=maximum).contains(&x)
        || !(minimum..=maximum).contains(&y)
    {
        return None;
    }
    let dx = (x - minimum).min(maximum - x);
    let dy = (y - minimum).min(maximum - y);
    let band = |distance: f32, width: f32| {
        let t = (1.0 - distance / width).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let ex = band(dx, edge_width);
    let ey = band(dy, edge_width);
    Some(BoundaryCosts {
        // Smooth union, bounded even at corners.
        edge: ex + ey - ex * ey,
        corner: band(dx, corner_width) * band(dy, corner_width),
    })
}

#[cfg(test)]
mod tests {
    use super::{boundary_costs, BoundaryCosts, RewardTerms};

    #[test]
    fn v72_native_reward_components_preserve_survival_and_event_multiplicity() {
        let terms = RewardTerms::new(
            3,
            1,
            2,
            5,
            BoundaryCosts {
                edge: 1.0,
                corner: 0.5,
            },
            4,
        );
        assert!(terms.is_some());
        if let Some(terms) = terms {
            assert_eq!(
                terms.weighted([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                Some([3.0, -0.0, 0.0, 0.0, -0.0, -0.0])
            );
            assert_eq!(
                terms.weighted([1.0, 10.0, 2.0, 0.5, 0.01, 0.1]),
                Some([3.0, -10.0, 4.0, 2.5, -0.04, -0.2])
            );
            assert!(terms.weighted([f32::NAN; 6]).is_none());
            assert!(terms.weighted([-1.0; 6]).is_none());
        }
        assert!(RewardTerms::new(
            4,
            2,
            0,
            0,
            BoundaryCosts {
                edge: 0.0,
                corner: 0.0
            },
            4
        )
        .is_some_and(|terms| terms.death == -1.0));
    }

    #[test]
    fn v71_boundary_symmetry_bounds_flat_interior_and_perimeter() {
        for x in 0..=120 {
            for y in 0..=120 {
                let a = boundary_costs(x as f32, y as f32, 0.0, 120.0, 2.0, 16.0);
                let b = boundary_costs(y as f32, x as f32, 0.0, 120.0, 2.0, 16.0);
                let c = boundary_costs(120.0 - x as f32, y as f32, 0.0, 120.0, 2.0, 16.0);
                assert_eq!(a, b);
                assert_eq!(a, c);
                assert!(a.is_some_and(
                    |v| (0.0..=1.0).contains(&v.edge) && (0.0..=1.0).contains(&v.corner)
                ));
            }
        }
        let interior = boundary_costs(30.0, 40.0, 0.0, 120.0, 2.0, 16.0);
        assert!(interior.is_some_and(|v| v.edge == 0.0 && v.corner == 0.0));
        let perimeter = boundary_costs(3.0, 60.0, 0.0, 120.0, 2.0, 16.0);
        assert!(perimeter.is_some_and(|v| v.edge == 0.0 && v.corner == 0.0));
        let straight = boundary_costs(0.0, 60.0, 0.0, 120.0, 2.0, 16.0);
        assert!(straight.is_some_and(|v| v.edge == 1.0 && v.corner == 0.0));
        let corner = boundary_costs(0.0, 0.0, 0.0, 120.0, 2.0, 16.0);
        assert!(corner.is_some_and(|v| v.edge == 1.0 && v.corner == 1.0));
    }

    #[test]
    fn v71_corner_increases_toward_corner_and_rejects_invalid_inputs() {
        let mut previous = 1.0;
        for d in 0..=160 {
            let value = boundary_costs(d as f32 / 10.0, d as f32 / 10.0, 0.0, 120.0, 2.0, 16.0);
            assert!(value.is_some_and(|v| v.corner <= previous));
            if let Some(value) = value {
                previous = value.corner;
            }
        }
        assert!(boundary_costs(f32::NAN, 1.0, 0.0, 120.0, 2.0, 16.0).is_none());
        assert!(boundary_costs(-1.0, 1.0, 0.0, 120.0, 2.0, 16.0).is_none());
        assert!(boundary_costs(1.0, 1.0, 0.0, 120.0, 0.0, 16.0).is_none());
        assert!(boundary_costs(1.0, 1.0, 0.0, 120.0, 2.0, 61.0).is_none());
    }
}
