use crate::{CoreError, PicoFixed};

const RAND_MAX: u64 = 2_147_483_647;
const RAND_DEGREE: usize = 31;
const RAND_SEPARATOR: usize = 3;
const RAND_WARMUP: usize = RAND_DEGREE * 10;

/// Serializable state for the Linux/Pemsa `rand()` stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RngCheckpoint {
    pub seed: u32,
    pub state: [u32; RAND_DEGREE],
    pub front: u8,
    pub rear: u8,
}

/// glibc-compatible additive RNG used by the accepted local Pemsa runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PicoRng {
    checkpoint: RngCheckpoint,
}

impl PicoRng {
    pub fn new(seed: u32) -> Self {
        let mut state = [0_u32; RAND_DEGREE];
        let normalized = if seed == 0 { 1 } else { seed };
        let mut previous = u64::from(normalized);
        if let Some(first) = state.first_mut() {
            *first = normalized;
        }
        for slot in state.iter_mut().skip(1) {
            previous = (16_807 * previous) % RAND_MAX;
            *slot = previous as u32;
        }
        let mut rng = Self {
            checkpoint: RngCheckpoint {
                seed,
                state,
                front: RAND_SEPARATOR as u8,
                rear: 0,
            },
        };
        for _ in 0..RAND_WARMUP {
            let _ = rng.rand_int();
        }
        rng
    }

    pub fn seed(&mut self, seed: u32) {
        *self = Self::new(seed);
    }

    pub fn rand_int(&mut self) -> u32 {
        let front = usize::from(self.checkpoint.front);
        let rear = usize::from(self.checkpoint.rear);
        let front_value = self.checkpoint.state.get(front).copied().unwrap_or(0);
        let rear_value = self.checkpoint.state.get(rear).copied().unwrap_or(0);
        let value = front_value.wrapping_add(rear_value);
        if let Some(slot) = self.checkpoint.state.get_mut(front) {
            *slot = value;
        }
        self.checkpoint.front = ((front + 1) % RAND_DEGREE) as u8;
        self.checkpoint.rear = ((rear + 1) % RAND_DEGREE) as u8;
        value >> 1
    }

    pub fn rnd(&mut self, limit: PicoFixed) -> PicoFixed {
        let random_float = self.rand_int() as f32 / RAND_MAX as f32;
        PicoFixed::from_f32(random_float * limit.to_f32())
    }

    pub const fn checkpoint(&self) -> RngCheckpoint {
        self.checkpoint
    }

    pub fn restore(&mut self, checkpoint: RngCheckpoint) -> Result<(), CoreError> {
        if usize::from(checkpoint.front) >= RAND_DEGREE
            || usize::from(checkpoint.rear) >= RAND_DEGREE
        {
            return Err(CoreError::InvalidRngCheckpoint);
        }
        self.checkpoint = checkpoint;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::PicoRng;
    use crate::PicoFixed;

    #[test]
    fn seed_42_matches_pemsa_rand_stream() {
        let mut rng = PicoRng::new(42);
        assert_eq!(rng.rand_int(), 71_876_166);
        assert_eq!(rng.rand_int(), 708_592_740);
        assert_eq!(rng.rand_int(), 1_483_128_881);
    }

    #[test]
    fn rnd_matches_pemsa_probe_strings() {
        let mut rng = PicoRng::new(42);
        assert_eq!(rng.rnd(PicoFixed::ONE).to_pico_string(), "0.0334");
        assert_eq!(rng.rnd(PicoFixed::from_int(10)).to_pico_string(), "3.2996");
    }

    #[test]
    fn checkpoint_restores_stream() {
        let mut rng = PicoRng::new(42);
        let checkpoint = rng.checkpoint();
        let first = rng.rand_int();
        assert!(rng.restore(checkpoint).is_ok());
        assert_eq!(rng.rand_int(), first);
    }
}
