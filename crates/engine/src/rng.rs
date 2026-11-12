//! Small RNG helpers: unbiased bounded integers and deterministic seed mixing.
//!
//! The engine only depends on [`rand_core::Rng`]; callers always pass an explicitly
//! seeded generator.

use rand_core::Rng;

/// Uniform integer in `0..n` (Lemire's nearly-divisionless method, unbiased). `n` must be > 0.
#[inline]
pub fn below<R: Rng + ?Sized>(rng: &mut R, n: u32) -> u32 {
    debug_assert!(n > 0);
    let mut m = (rng.next_u32() as u64) * (n as u64);
    let mut lo = m as u32;
    if lo < n {
        let t = n.wrapping_neg() % n;
        while lo < t {
            m = (rng.next_u32() as u64) * (n as u64);
            lo = m as u32;
        }
    }
    (m >> 32) as u32
}

/// Uniform `f64` in `[0, 1)`.
#[inline]
pub fn unit_f64<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// SplitMix64 finalizer: a strong 64-bit mixing function.
#[inline]
pub const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministically derives a seed from a master seed and two indices
/// (e.g. `mix_seed(master, round, table_id)`).
#[inline]
pub const fn mix_seed(master: u64, a: u64, b: u64) -> u64 {
    splitmix64(splitmix64(splitmix64(master) ^ a) ^ b.wrapping_mul(0xD6E8_FEB8_6659_FD93))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter(u64);
    impl rand_core::TryRng for Counter {
        type Error = core::convert::Infallible;
        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            Ok(self.try_next_u64()? as u32)
        }
        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            self.0 = self.0.wrapping_add(1);
            Ok(splitmix64(self.0))
        }
        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
            for b in dst {
                *b = self.try_next_u32()? as u8;
            }
            Ok(())
        }
    }

    #[test]
    fn below_in_range_and_roughly_uniform() {
        let mut r = Counter(0);
        let mut counts = [0u32; 7];
        for _ in 0..70_000 {
            counts[below(&mut r, 7) as usize] += 1;
        }
        for c in counts {
            assert!((9_000..11_000).contains(&c), "{counts:?}");
        }
    }

    #[test]
    fn mix_seed_distinct() {
        assert_ne!(mix_seed(1, 0, 1), mix_seed(1, 1, 0));
        assert_ne!(mix_seed(1, 0, 0), mix_seed(2, 0, 0));
        assert_eq!(mix_seed(5, 6, 7), mix_seed(5, 6, 7));
    }
}
