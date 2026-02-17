//! Compact monomial representation for modular polynomial GCD.
//!
//! Uses fixed-size array [u16; 8] for up to 8 variables.
//! This is much faster than Vec<u32> for hashing and comparison.

/// Maximum variables supported by compact monomials
pub const MAX_VARS: usize = 8;

/// Exponent type (u16 supports degrees up to 65535)
pub type Exp = u16;

/// Compact monomial: fixed-size array of exponents.
/// Unused positions (for fewer than 8 vars) are 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Mono(pub [Exp; MAX_VARS]);

impl Mono {
    /// Create zero monomial (constant term)
    #[inline]
    pub const fn zero() -> Self {
        Mono([0; MAX_VARS])
    }

    /// Create monomial for single variable: x_i^1
    #[inline]
    pub fn var(i: usize) -> Self {
        debug_assert!(i < MAX_VARS);
        let mut m = [0; MAX_VARS];
        m[i] = 1;
        Mono(m)
    }

    /// Add monomials (multiply terms): x^a * x^b = x^(a+b)
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [0; MAX_VARS];
        for (dst, (&a, &b)) in result.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *dst = a + b;
        }
        Mono(result)
    }

    /// Total degree (sum of all exponents)
    #[inline]
    pub fn total_degree(&self) -> u32 {
        self.0.iter().map(|&e| e as u32).sum()
    }

    /// Degree in a specific variable
    #[inline]
    pub fn deg_var(&self, i: usize) -> Exp {
        self.0.get(i).copied().unwrap_or(0)
    }

    /// Check if this is the zero monomial (constant)
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.0.iter().all(|&e| e == 0)
    }

    /// Number of active variables (non-zero exponents)
    pub fn num_vars(&self) -> usize {
        self.0.iter().filter(|&&e| e > 0).count()
    }

    /// Create from a slice of exponents (pads with zeros)
    pub fn from_slice(exps: &[Exp]) -> Self {
        let mut m = [0; MAX_VARS];
        for (i, &e) in exps.iter().take(MAX_VARS).enumerate() {
            m[i] = e;
        }
        Mono(m)
    }

    /// Substitute variable i with 0 (remove it from monomial, keep for x_i^0 = 1)
    /// Returns None if x_i has non-zero exponent (would need coefficient adjustment)
    /// Returns Some(new_mono) if x_i^0 (no change needed except marker)
    #[inline]
    pub fn eval_var_zero(&self, i: usize) -> Option<Self> {
        if self.0[i] > 0 {
            None // Evaluating x_i^k at 0 gives 0 if k > 0
        } else {
            Some(*self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_zero() {
        let m = Mono::zero();
        assert!(m.is_constant());
        assert_eq!(m.total_degree(), 0);
    }

    #[test]
    fn test_mono_var() {
        let x0 = Mono::var(0);
        let x1 = Mono::var(1);
        assert_eq!(x0.deg_var(0), 1);
        assert_eq!(x0.deg_var(1), 0);
        assert_eq!(x1.deg_var(1), 1);
    }

    #[test]
    fn test_mono_add() {
        let x0 = Mono::var(0);
        let x1 = Mono::var(1);
        let x0x1 = x0.add(&x1);
        assert_eq!(x0x1.deg_var(0), 1);
        assert_eq!(x0x1.deg_var(1), 1);
        assert_eq!(x0x1.total_degree(), 2);
    }

    #[test]
    fn test_mono_hash() {
        use std::collections::HashMap;
        let mut map: HashMap<Mono, u64> = HashMap::new();
        let m1 = Mono::from_slice(&[1, 2, 3]);
        let m2 = Mono::from_slice(&[1, 2, 3]);
        let m3 = Mono::from_slice(&[1, 2, 4]);

        map.insert(m1, 100);
        assert_eq!(map.get(&m2), Some(&100));
        assert_eq!(map.get(&m3), None);
    }

    #[test]
    fn test_mono_ord() {
        let m1 = Mono::from_slice(&[1, 0, 0]);
        let m2 = Mono::from_slice(&[0, 1, 0]);
        let m3 = Mono::from_slice(&[1, 1, 0]);

        // Lex order: [1,0,0] > [0,1,0]
        assert!(m1 > m2);
        assert!(m3 > m1);
    }
}
