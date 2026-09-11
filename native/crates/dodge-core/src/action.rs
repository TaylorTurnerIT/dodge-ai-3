use crate::CoreError;

pub const BUTTON_X_MASK: u8 = 1 << 5;

/// Nine-action training space retained from the Python collector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Neutral,
    Left,
    Right,
    Up,
    Down,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
}

impl Action {
    pub const ALL: [Self; 9] = [
        Self::Neutral,
        Self::Left,
        Self::Right,
        Self::Up,
        Self::Down,
        Self::UpLeft,
        Self::UpRight,
        Self::DownLeft,
        Self::DownRight,
    ];

    pub const fn mask(self) -> u8 {
        match self {
            Self::Neutral => 0,
            Self::Left => 1,
            Self::Right => 2,
            Self::Up => 4,
            Self::Down => 8,
            Self::UpLeft => 5,
            Self::UpRight => 6,
            Self::DownLeft => 9,
            Self::DownRight => 10,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Left => "left",
            Self::Right => "right",
            Self::Up => "up",
            Self::Down => "down",
            Self::UpLeft => "up_left",
            Self::UpRight => "up_right",
            Self::DownLeft => "down_left",
            Self::DownRight => "down_right",
        }
    }

    pub fn from_name(name: &str) -> Result<Self, CoreError> {
        match name {
            "neutral" => Ok(Self::Neutral),
            "left" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            "up_left" => Ok(Self::UpLeft),
            "up_right" => Ok(Self::UpRight),
            "down_left" => Ok(Self::DownLeft),
            "down_right" => Ok(Self::DownRight),
            _ => Err(CoreError::InvalidActionName),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Action;

    #[test]
    fn action_order_and_masks_match_training_contract() {
        let masks = Action::ALL.map(Action::mask);
        assert_eq!(masks, [0, 1, 2, 4, 8, 5, 6, 9, 10]);
        assert_eq!(
            Action::from_name("up_right").map(Action::name),
            Ok("up_right")
        );
    }
}
