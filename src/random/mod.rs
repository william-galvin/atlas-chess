use std::time::{SystemTime, UNIX_EPOCH};

fn get_sys_time() -> u64 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_millis() as u64 % 10_000
}

pub struct Rng {
    state: u64,
    inc: u64
}

impl Rng {

    /// Initialize the random number generator
    pub fn new() -> Self {
        Self {
            state: get_sys_time(),
            inc: !get_sys_time()
        }
    }

    /// initialize the random number generator with deterministic behavior
    pub fn new_deterministic(state: u64, inc: u64) -> Self {
        Self {
            state,
            inc
        }
    }

    /// Generate a uniform random unsigned integer
    pub fn random(&mut self) -> u64 {
        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005u64).wrapping_add(self.inc | 1);
        let xorshifted = ((old_state >> 18u64) ^ old_state) >> 27u64;
        let rot: i64 = (old_state >> 59u64).try_into().unwrap();
        (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
    }

    /// Get random number within range.
    /// Only works if low and high are both positive
    pub fn bounded_random(&mut self, low: u64, high: u64) -> u64 {
        assert!(high > low, "high must be > low");
        let bound = high - low;
        let threshold = !bound % bound;
        loop {
            let r = self.random();
            if r >= threshold {
                return (r % bound) + low
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructor() {
        let rng = Rng::new();
        assert_ne!(rng.state, rng.inc);
    }

    #[test]
    fn test_bounded_random() {
        let mut rng = Rng::new();
        let mut low = 10;
        let mut high = 100;
        for i in 0..100 {
            let x = rng.bounded_random(low, high);
            assert!(x >= low);
            assert!(x < high);
        }
    }
}