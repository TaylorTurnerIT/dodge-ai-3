use crate::CoreError;

pub const BUTTON_MASK_LIMIT: u8 = 0b11_1111;

/// PICO-8 button identities used by the cartridge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Button {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
    O = 4,
    X = 5,
}

impl Button {
    pub const fn index(self) -> u8 {
        self as u8
    }

    pub const fn mask(self) -> u8 {
        1 << self.index()
    }
}

/// Current/previous input masks and mouse/stat compatibility values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputState {
    current_mask: u8,
    previous_mask: u8,
    source_input_mode: bool,
}

impl InputState {
    pub const fn new() -> Self {
        Self {
            current_mask: 0,
            previous_mask: 0,
            source_input_mode: false,
        }
    }

    pub(crate) const fn from_wire(
        current_mask: u8,
        previous_mask: u8,
        source_input_mode: bool,
    ) -> Result<Self, CoreError> {
        if current_mask > BUTTON_MASK_LIMIT {
            return Err(CoreError::InvalidButtonMask(current_mask));
        }
        if previous_mask > BUTTON_MASK_LIMIT {
            return Err(CoreError::InvalidButtonMask(previous_mask));
        }
        Ok(Self {
            current_mask,
            previous_mask,
            source_input_mode,
        })
    }

    pub const fn current_mask(self) -> u8 {
        self.current_mask
    }

    pub const fn previous_mask(self) -> u8 {
        self.previous_mask
    }

    /// Whether the cartridge's `input` selector is in keyboard/button mode.
    pub const fn source_input_mode(self) -> bool {
        self.source_input_mode
    }

    pub fn validate_mask(mask: u8) -> Result<(), CoreError> {
        if mask > BUTTON_MASK_LIMIT {
            Err(CoreError::InvalidButtonMask(mask))
        } else {
            Ok(())
        }
    }

    pub fn advance(&mut self, mask: u8) -> Result<(), CoreError> {
        Self::validate_mask(mask)?;
        self.previous_mask = self.current_mask;
        self.current_mask = mask;
        if mask != 0 {
            self.source_input_mode = true;
        }
        Ok(())
    }

    pub(crate) fn finalize_frame(&mut self, post_frame_mask: u8) {
        self.current_mask = post_frame_mask;
        self.previous_mask = post_frame_mask;
    }

    pub const fn btn(self, button: Button) -> bool {
        self.current_mask & button.mask() != 0
    }

    pub const fn btnp(self, button: Button) -> bool {
        self.btn(button) && self.previous_mask & button.mask() == 0
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{Button, InputState};

    #[test]
    fn frame_finalization_preserves_post_boundary_masks() {
        let mut input = InputState::new();
        assert!(input.advance(Button::X.mask()).is_ok());
        assert!(input.btnp(Button::X));

        input.finalize_frame(Button::X.mask());

        assert_eq!(input.current_mask(), Button::X.mask());
        assert_eq!(input.previous_mask(), Button::X.mask());
        assert!(!input.btnp(Button::X));
    }
}
