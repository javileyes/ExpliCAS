//! Multivariate polynomial over Fp (mod p).
//!
//! Uses compact monomials (Mono) for fast hashing.
//! Designed for Zippel GCD algorithm.

use crate::modp::{add_mod, mul_mod, neg_mod, sub_mod};
use crate::mono::{Exp, Mono, MAX_VARS};
use crate::unipoly_modp::UniPolyModP;
use std::collections::HashMap;

/// Multivariate polynomial mod p
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPolyModP {
    pub p: u64,
    pub num_vars: usize,
    /// Terms sorted by monomial (lex order), no zero coefficients
    pub terms: Vec<(Mono, u64)>,
}

impl MultiPolyModP {
    /// Create zero polynomial
    pub fn zero(p: u64, num_vars: usize) -> Self {
        Self {
            p,
            num_vars,
            terms: vec![],
        }
    }

    /// Create constant polynomial
    pub fn constant(c: u64, p: u64, num_vars: usize) -> Self {
        let c = c % p;
        if c == 0 {
            Self::zero(p, num_vars)
        } else {
            Self {
                p,
                num_vars,
                terms: vec![(Mono::zero(), c)],
            }
        }
    }

    /// Create monomial x_i
    pub fn var(i: usize, p: u64, num_vars: usize) -> Self {
        debug_assert!(i < num_vars && i < MAX_VARS);
        Self {
            p,
            num_vars,
            terms: vec![(Mono::var(i), 1)],
        }
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Total degree
    pub fn total_degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|(m, _)| m.total_degree())
            .max()
            .unwrap_or(0)
    }

    /// Degree in a specific variable
    pub fn degree_in(&self, var: usize) -> Exp {
        self.terms
            .iter()
            .map(|(m, _)| m.deg_var(var))
            .max()
            .unwrap_or(0)
    }

    /// Leading coefficient (of first term in lex order)
    pub fn lc(&self) -> u64 {
        self.terms.first().map(|(_, c)| *c).unwrap_or(0)
    }

    /// Normalize: sort by monomial, combine like terms, remove zeros
    pub fn normalize(&mut self) {
        if self.terms.is_empty() {
            return;
        }

        // Accumulate in HashMap
        let mut map: HashMap<Mono, u64> = HashMap::with_capacity(self.terms.len());
        for (mono, coeff) in &self.terms {
            let entry = map.entry(*mono).or_insert(0);
            *entry = add_mod(*entry, *coeff, self.p);
        }

        // Convert back, sort, remove zeros
        self.terms = map.into_iter().filter(|(_, c)| *c != 0).collect();
        self.terms.sort_by(|a, b| b.0.cmp(&a.0)); // Descending lex for leading term
    }

    /// Make monic (divide by leading coefficient)
    pub fn make_monic(&mut self) {
        let lc = self.lc();
        if lc == 0 || lc == 1 {
            return;
        }
        if let Some(lc_inv) = crate::modp::inv_mod(lc, self.p) {
            for (_, c) in &mut self.terms {
                *c = mul_mod(*c, lc_inv, self.p);
            }
        }
    }

    /// Negate
    pub fn neg(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(m, c)| (*m, neg_mod(*c, self.p)))
            .collect();
        Self {
            p: self.p,
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Add two polynomials
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.p, other.p);
        debug_assert_eq!(self.num_vars, other.num_vars);

        let mut map: HashMap<Mono, u64> =
            HashMap::with_capacity(self.terms.len() + other.terms.len());

        for (m, c) in &self.terms {
            *map.entry(*m).or_insert(0) = add_mod(map.get(m).copied().unwrap_or(0), *c, self.p);
        }
        for (m, c) in &other.terms {
            *map.entry(*m).or_insert(0) = add_mod(map.get(m).copied().unwrap_or(0), *c, self.p);
        }

        let mut terms: Vec<_> = map.into_iter().filter(|(_, c)| *c != 0).collect();
        terms.sort_by(|a, b| b.0.cmp(&a.0));

        Self {
            p: self.p,
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Subtract
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Multiply (fast version using HashMap)
    pub fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.p, other.p);
        debug_assert_eq!(self.num_vars, other.num_vars);

        if self.is_zero() || other.is_zero() {
            return Self::zero(self.p, self.num_vars);
        }

        let capacity = self.terms.len() * other.terms.len() / 2;
        let mut map: HashMap<Mono, u64> = HashMap::with_capacity(capacity);

        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let new_mono = m1.add(m2);
                let new_coeff = mul_mod(*c1, *c2, self.p);
                let entry = map.entry(new_mono).or_insert(0);
                *entry = add_mod(*entry, new_coeff, self.p);
            }
        }

        let mut terms: Vec<_> = map.into_iter().filter(|(_, c)| *c != 0).collect();
        terms.sort_by(|a, b| b.0.cmp(&a.0));

        Self {
            p: self.p,
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, k: u64) -> Self {
        if k == 0 {
            return Self::zero(self.p, self.num_vars);
        }
        let terms = self
            .terms
            .iter()
            .map(|(m, c)| (*m, mul_mod(*c, k, self.p)))
            .filter(|(_, c)| *c != 0)
            .collect();
        Self {
            p: self.p,
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Power (by squaring)
    pub fn pow(&self, mut exp: u32) -> Self {
        if exp == 0 {
            return Self::constant(1, self.p, self.num_vars);
        }

        let mut result = Self::constant(1, self.p, self.num_vars);
        let mut base = self.clone();

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// Evaluate specific variables at given values.
    /// Returns a new polynomial with fewer effective variables.
    /// `assignments` is a list of (var_index, value) pairs.
    pub fn eval_vars(&self, assignments: &[(usize, u64)]) -> Self {
        if assignments.is_empty() {
            return self.clone();
        }

        let mut result_map: HashMap<Mono, u64> = HashMap::new();

        for (mono, coeff) in &self.terms {
            let mut new_mono = *mono;
            let mut new_coeff = *coeff;

            for &(var_idx, val) in assignments {
                let exp = new_mono.0[var_idx];
                if exp > 0 {
                    // Multiply coefficient by val^exp
                    let val_pow = crate::modp::pow_mod(val, exp as u64, self.p);
                    new_coeff = mul_mod(new_coeff, val_pow, self.p);
                    new_mono.0[var_idx] = 0; // Remove variable
                }
            }

            if new_coeff != 0 {
                let entry = result_map.entry(new_mono).or_insert(0);
                *entry = add_mod(*entry, new_coeff, self.p);
            }
        }

        let mut terms: Vec<_> = result_map.into_iter().filter(|(_, c)| *c != 0).collect();
        terms.sort_by(|a, b| b.0.cmp(&a.0));

        Self {
            p: self.p,
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Evaluate all variables except `main_var`, returning a univariate polynomial.
    /// `assignments` should contain values for all vars except `main_var`.
    pub fn eval_to_univar(&self, main_var: usize, assignments: &[(usize, u64)]) -> UniPolyModP {
        // First evaluate all other variables
        let partial = self.eval_vars(assignments);

        // Now convert to univariate in main_var
        let max_deg = partial.degree_in(main_var) as usize;
        let mut coeffs = vec![0u64; max_deg + 1];

        for (mono, coeff) in &partial.terms {
            // After eval, only main_var should have non-zero exponent
            let deg = mono.deg_var(main_var) as usize;
            coeffs[deg] = add_mod(coeffs[deg], *coeff, self.p);
        }

        let mut result = UniPolyModP { p: self.p, coeffs };
        result.trim();
        result
    }

    /// Check if other divides self exactly (no remainder)
    pub fn div_exact(&self, other: &Self) -> Option<Self> {
        // For now, simple check: trial division
        // This is a simplified version; full impl would use multivariate division
        if other.is_zero() {
            return None;
        }
        // TODO: Implement proper multivariate division
        // For GCD verification, we'll use a simpler approach
        None // Placeholder
    }

    /// Add a constant to the polynomial (modifies constant term)
    pub fn add_const(&self, k: u64) -> Self {
        if k == 0 {
            return self.clone();
        }
        let k = k % self.p;
        let const_mono = Mono::zero();

        let mut terms = self.terms.clone();
        let mut found = false;
        for (mono, coeff) in &mut terms {
            if *mono == const_mono {
                *coeff = add_mod(*coeff, k, self.p);
                found = true;
                break;
            }
        }
        if !found {
            terms.push((const_mono, k));
        }

        // Remove zero coefficients and sort
        terms.retain(|(_, c)| *c != 0);
        terms.sort_by(|a, b| b.0.cmp(&a.0));

        Self {
            p: self.p,
            num_vars: self.num_vars,
            terms,
        }
    }
}

// =============================================================================
// Direct construction of (c0 + c1*x1 + ... + cn*xn)^e using multinomial theorem
// =============================================================================

/// Build (c0 + c1*x1 + c2*x2 + ... + c_n*x_n)^exp directly using multinomial coefficients.
/// This is O(num_terms) instead of O(num_terms^2) for repeated multiplication.
///
/// `coeffs` has length num_vars + 1: coeffs[0] is constant, coeffs[1..] are xi coefficients.
pub fn build_linear_pow_direct(coeffs: &[u64], exp: u32, p: u64, num_vars: usize) -> MultiPolyModP {
    debug_assert_eq!(coeffs.len(), num_vars + 1);

    // Precompute factorials and inverse factorials mod p
    let (fact, inv_fact) = precompute_factorials(exp as usize, p);

    let mut terms: Vec<(Mono, u64)> = Vec::new();

    // Enumerate all compositions k0 + k1 + ... + k_n = exp
    enumerate_compositions(num_vars + 1, exp as usize, |k| {
        // Multinomial coefficient: exp! / (k0! * k1! * ... * kn!)
        let mut mult = fact[exp as usize];
        for &ki in &k {
            mult = mul_mod(mult, inv_fact[ki], p);
        }

        // Coefficient contribution: mult * Π(coeffs[i]^k[i])
        let mut coef = mult;
        for (i, &ki) in k.iter().enumerate() {
            if ki > 0 {
                let ci_pow = crate::modp::pow_mod(coeffs[i], ki as u64, p);
                coef = mul_mod(coef, ci_pow, p);
            }
        }

        if coef != 0 {
            // Build monomial: exponents for x1..xn are k[1..n]
            let mut mono = Mono::zero();
            for (i, &ki) in k.iter().skip(1).enumerate() {
                if i < MAX_VARS {
                    mono.0[i] = ki as u16;
                }
            }
            terms.push((mono, coef));
        }
    });

    // Sort by monomial (descending lex)
    terms.sort_by(|a, b| b.0.cmp(&a.0));

    MultiPolyModP { p, num_vars, terms }
}

/// Precompute factorials and their modular inverses up to n.
fn precompute_factorials(n: usize, p: u64) -> (Vec<u64>, Vec<u64>) {
    let mut fact = vec![1u64; n + 1];
    for i in 1..=n {
        fact[i] = mul_mod(fact[i - 1], i as u64, p);
    }

    // Compute inverse of n! using Fermat's little theorem
    let mut inv_fact = vec![1u64; n + 1];
    inv_fact[n] = crate::modp::pow_mod(fact[n], p - 2, p);
    for i in (1..n).rev() {
        inv_fact[i] = mul_mod(inv_fact[i + 1], (i + 1) as u64, p);
    }
    inv_fact[0] = 1;

    (fact, inv_fact)
}

/// Enumerate all compositions of `parts` non-negative integers that sum to `total`.
/// Calls `f` with each composition as a Vec<usize>.
fn enumerate_compositions<F: FnMut(Vec<usize>)>(parts: usize, total: usize, mut f: F) {
    let mut current = vec![0usize; parts];
    enumerate_compositions_rec(parts, total, 0, &mut current, &mut f);
}

fn enumerate_compositions_rec<F: FnMut(Vec<usize>)>(
    parts: usize,
    remaining: usize,
    index: usize,
    current: &mut [usize],
    f: &mut F,
) {
    if index == parts - 1 {
        current[index] = remaining;
        f(current.to_vec());
        return;
    }

    for k in 0..=remaining {
        current[index] = k;
        enumerate_compositions_rec(parts, remaining - k, index + 1, current, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 17;

    #[test]
    fn test_add() {
        // x + y
        let x = MultiPolyModP::var(0, P, 2);
        let y = MultiPolyModP::var(1, P, 2);
        let sum = x.add(&y);
        assert_eq!(sum.num_terms(), 2);
    }

    #[test]
    fn test_mul() {
        // (x + 1) * (y + 1) = xy + x + y + 1
        let x = MultiPolyModP::var(0, P, 2);
        let y = MultiPolyModP::var(1, P, 2);
        let one = MultiPolyModP::constant(1, P, 2);

        let xp1 = x.add(&one);
        let yp1 = y.add(&one);
        let prod = xp1.mul(&yp1);

        assert_eq!(prod.num_terms(), 4);
    }

    #[test]
    fn test_pow() {
        // (x + 1)^2 = x^2 + 2x + 1
        let x = MultiPolyModP::var(0, P, 2);
        let one = MultiPolyModP::constant(1, P, 2);
        let xp1 = x.add(&one);

        let sq = xp1.pow(2);
        assert_eq!(sq.num_terms(), 3);
        assert_eq!(sq.total_degree(), 2);
    }

    #[test]
    fn test_eval_to_univar() {
        // f(x,y) = xy + x + y + 1
        // f(x, 2) = 2x + x + 2 + 1 = 3x + 3
        let x = MultiPolyModP::var(0, P, 2);
        let y = MultiPolyModP::var(1, P, 2);
        let one = MultiPolyModP::constant(1, P, 2);

        let f = x.mul(&y).add(&x).add(&y).add(&one);
        let uni = f.eval_to_univar(0, &[(1, 2)]); // y = 2

        // 3x + 3
        assert_eq!(uni.coeffs.len(), 2);
        assert_eq!(uni.coeffs[0], 3); // constant
        assert_eq!(uni.coeffs[1], 3); // x coefficient
    }
}
