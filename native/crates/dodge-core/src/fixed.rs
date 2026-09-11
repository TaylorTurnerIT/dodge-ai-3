use crate::CoreError;

const SHIFT: u32 = 16;
const ONE_RAW: i32 = 1 << SHIFT;
const FRACTION_MASK: i32 = ONE_RAW - 1;

/// Signed Q16.16 value matching the accepted Pemsa fixmath boundary.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct PicoFixed {
    raw: i32,
}

impl PicoFixed {
    pub const ZERO: Self = Self { raw: 0 };
    pub const ONE: Self = Self { raw: ONE_RAW };

    pub const fn from_raw(raw: i32) -> Self {
        Self { raw }
    }

    pub const fn from_int(value: i32) -> Self {
        Self {
            raw: value * ONE_RAW,
        }
    }

    pub fn from_f32(value: f32) -> Self {
        Self::from_raw((value * ONE_RAW as f32) as i32)
    }

    pub const fn raw(self) -> i32 {
        self.raw
    }

    pub fn to_f32(self) -> f32 {
        self.raw as f32 / ONE_RAW as f32
    }

    pub fn to_double(self) -> f64 {
        self.raw as f64 / ONE_RAW as f64
    }

    pub fn to_pico_string(self) -> String {
        let sign = if self.raw < 0 { "-" } else { "" };
        let absolute = i64::from(self.raw).unsigned_abs();
        let whole = absolute / ONE_RAW as u64;
        let fraction = absolute % ONE_RAW as u64;
        let digits = fraction * 10_000 / ONE_RAW as u64;
        if digits == 0 {
            format!("{sign}{whole}")
        } else {
            format!("{sign}{whole}.{digits:04}")
                .trim_end_matches('0')
                .to_owned()
        }
    }

    pub const fn floor(self) -> Self {
        Self::from_raw(self.raw & !FRACTION_MASK)
    }

    pub const fn ceil(self) -> Self {
        let floor = self.floor();
        if self.raw & FRACTION_MASK == 0 {
            floor
        } else {
            Self::from_raw(floor.raw + ONE_RAW)
        }
    }

    pub fn round(self) -> Self {
        let floor = self.floor();
        let remainder = self.sub(floor);
        if remainder < Self::from_f32(0.5) {
            floor
        } else {
            floor.add(Self::ONE)
        }
    }

    pub const fn add(self, other: Self) -> Self {
        Self::from_raw(self.raw + other.raw)
    }

    pub const fn sub(self, other: Self) -> Self {
        Self::from_raw(self.raw - other.raw)
    }

    pub const fn neg(self) -> Self {
        Self::from_raw(-self.raw)
    }

    pub fn mul_fixed(self, other: Self) -> Self {
        let product = i64::from(self.raw) * i64::from(other.raw);
        Self::from_raw((product >> SHIFT) as i32)
    }

    pub fn div_fixed(self, other: Self) -> Result<Self, CoreError> {
        if other.raw == 0 {
            return Err(CoreError::DivisionByZero);
        }
        let numerator = i64::from(self.raw) << SHIFT;
        Ok(Self::from_raw((numerator / i64::from(other.raw)) as i32))
    }

    pub fn rem_fixed(self, other: Self) -> Self {
        Self::from_raw(self.raw % other.raw)
    }
}

pub const fn pico_floor(value: PicoFixed) -> PicoFixed {
    value.floor()
}

pub const fn pico_ceil(value: PicoFixed) -> PicoFixed {
    value.ceil()
}

pub fn pico_mid(first: PicoFixed, second: PicoFixed, third: PicoFixed) -> PicoFixed {
    let (low, high) = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    low.max(high.min(third))
}

pub fn pico_mod(first: PicoFixed, second: PicoFixed) -> Result<PicoFixed, CoreError> {
    if second.raw() == 0 {
        return Err(CoreError::DivisionByZero);
    }
    Ok(first.rem_fixed(second))
}

#[cfg(test)]
mod tests {
    use super::{pico_ceil, pico_floor, pico_mid, pico_mod, PicoFixed};

    #[test]
    fn fixed_boundaries_match_probe() {
        let value = PicoFixed::from_f32(-1.2);
        assert_eq!(value.raw(), -78_643);
        assert_eq!(pico_floor(value).to_pico_string(), "-2");
        assert_eq!(pico_ceil(value).to_pico_string(), "-1");
        assert_eq!(
            pico_mid(
                PicoFixed::ZERO,
                PicoFixed::from_int(9),
                PicoFixed::from_int(4)
            ),
            PicoFixed::from_int(4)
        );
        assert_eq!(
            pico_mod(PicoFixed::from_int(7), PicoFixed::from_int(4)),
            Ok(PicoFixed::from_int(3))
        );
        assert_eq!(PicoFixed::from_f32(-1.2).round(), PicoFixed::from_int(-1));
        assert_eq!(PicoFixed::from_f32(-1.7).round(), PicoFixed::from_int(-2));
    }

    #[test]
    fn fixed_math_truncates_toward_zero_for_division() {
        let value = PicoFixed::from_int(-7);
        assert_eq!(
            value.div_fixed(PicoFixed::from_int(2)),
            Ok(PicoFixed::from_f32(-3.5))
        );
        assert_eq!(
            value.rem_fixed(PicoFixed::from_int(2)),
            PicoFixed::from_int(-1)
        );
    }
}
