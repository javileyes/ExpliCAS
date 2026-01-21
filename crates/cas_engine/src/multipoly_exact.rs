//! MultiPolyExact - Multivariate Polynomial with Exact Coefficients
//!
//! Provides exact arithmetic (no modular reduction) for polynomial operations.
//! Uses BigRational for arbitrary precision coefficients.
//!
//! # Key Operations
//! - `add`, `sub`, `mul`, `pow` - Polynomial arithmetic
//! - `from_expr` / `to_expr` - Conversion to/from AST
//!
//! # Design Notes
//! - HashMap-based sparse representation
//! - Monomial = Vec<u16> exponents (one per variable)
//! - Coefficients stored as BigRational (exact)

use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;
use std::ops::{Add, Mul, Neg, Sub};

/// Monomial represented as exponent vector.
/// `[e0, e1, e2, ...]` represents `x0^e0 * x1^e1 * x2^e2 * ...`
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Monomial(pub Vec<u16>);

impl Monomial {
    /// Create constant monomial (all exponents zero)
    pub fn constant(n_vars: usize) -> Self {
        Monomial(vec![0; n_vars])
    }

    /// Create single variable monomial: x_i^1
    pub fn var(n_vars: usize, var_idx: usize) -> Self {
        let mut exps = vec![0; n_vars];
        if var_idx < n_vars {
            exps[var_idx] = 1;
        }
        Monomial(exps)
    }

    /// Total degree = sum of exponents
    pub fn total_degree(&self) -> u32 {
        self.0.iter().map(|&e| e as u32).sum()
    }

    /// Is this the constant monomial (all zeros)?
    pub fn is_constant(&self) -> bool {
        self.0.iter().all(|&e| e == 0)
    }

    /// Multiply two monomials: add exponents
    pub fn mul(&self, other: &Monomial) -> Monomial {
        let len = self.0.len().max(other.0.len());
        let mut result = vec![0u16; len];
        for (i, &e) in self.0.iter().enumerate() {
            result[i] = result[i].saturating_add(e);
        }
        for (i, &e) in other.0.iter().enumerate() {
            result[i] = result[i].saturating_add(e);
        }
        Monomial(result)
    }

    /// Raise monomial to power: multiply exponents
    pub fn pow(&self, n: u32) -> Monomial {
        Monomial(self.0.iter().map(|&e| (e as u32 * n) as u16).collect())
    }

    /// Pad exponents to match n_vars
    pub fn pad_to(&mut self, n_vars: usize) {
        self.0.resize(n_vars, 0);
    }

    /// Remap variables according to mapping: old_idx -> new_idx
    pub fn remap(&self, mapping: &[u8], new_n_vars: usize) -> Monomial {
        let mut new_exps = vec![0u16; new_n_vars];
        for (old_idx, &exp) in self.0.iter().enumerate() {
            if old_idx < mapping.len() {
                let new_idx = mapping[old_idx] as usize;
                if new_idx < new_n_vars {
                    new_exps[new_idx] = exp;
                }
            }
        }
        Monomial(new_exps)
    }
}

/// Multivariate polynomial with exact (BigRational) coefficients.
#[derive(Debug, Clone)]
pub struct MultiPolyExact {
    /// Terms: monomial -> coefficient
    pub terms: FxHashMap<Monomial, BigRational>,
    /// Number of variables
    pub n_vars: usize,
}

impl MultiPolyExact {
    /// Create zero polynomial
    pub fn zero(n_vars: usize) -> Self {
        Self {
            terms: FxHashMap::default(),
            n_vars,
        }
    }

    /// Create constant polynomial
    pub fn constant(c: BigRational, n_vars: usize) -> Self {
        let mut poly = Self::zero(n_vars);
        if !c.is_zero() {
            poly.terms.insert(Monomial::constant(n_vars), c);
        }
        poly
    }

    /// Create polynomial from single variable: x_i
    pub fn var(n_vars: usize, var_idx: usize) -> Self {
        let mut poly = Self::zero(n_vars);
        poly.terms
            .insert(Monomial::var(n_vars, var_idx), BigRational::one());
        poly
    }

    /// Number of non-zero terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Maximum total degree among all terms
    pub fn max_total_degree(&self) -> u32 {
        self.terms
            .keys()
            .map(|m| m.total_degree())
            .max()
            .unwrap_or(0)
    }

    /// Add a single term (monomial, coefficient)
    pub fn add_term(&mut self, mono: Monomial, coeff: BigRational) {
        if coeff.is_zero() {
            return;
        }
        self.terms
            .entry(mono)
            .and_modify(|c| *c = &*c + &coeff)
            .or_insert(coeff);
        // Clean up zero
        self.terms.retain(|_, c| !c.is_zero());
    }

    /// Negate all coefficients
    pub fn negate(&self) -> Self {
        Self {
            terms: self.terms.iter().map(|(m, c)| (m.clone(), -c)).collect(),
            n_vars: self.n_vars,
        }
    }

    /// Polynomial addition
    pub fn add(&self, other: &Self) -> Self {
        let n_vars = self.n_vars.max(other.n_vars);
        let mut result = Self::zero(n_vars);

        for (mono, coeff) in &self.terms {
            let mut m = mono.clone();
            m.pad_to(n_vars);
            result.add_term(m, coeff.clone());
        }
        for (mono, coeff) in &other.terms {
            let mut m = mono.clone();
            m.pad_to(n_vars);
            result.add_term(m, coeff.clone());
        }

        result
    }

    /// Polynomial subtraction
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    /// Polynomial multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let n_vars = self.n_vars.max(other.n_vars);
        let mut result = Self::zero(n_vars);

        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                let new_mono = m1.mul(m2);
                let new_coeff = c1 * c2;
                result.add_term(new_mono, new_coeff);
            }
        }

        result
    }

    /// Polynomial power (binary exponentiation)
    pub fn pow(&self, n: u32) -> Self {
        if n == 0 {
            return Self::constant(BigRational::one(), self.n_vars);
        }
        if n == 1 {
            return self.clone();
        }

        let mut result = self.clone();
        let mut exp = n - 1;
        let mut base = self.clone();

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }

        result
    }

    /// Remap variables to new VarTable order
    pub fn remap_vars(&self, mapping: &[u8], new_n_vars: usize) -> Self {
        let mut result = Self::zero(new_n_vars);
        for (mono, coeff) in &self.terms {
            let new_mono = mono.remap(mapping, new_n_vars);
            result.add_term(new_mono, coeff.clone());
        }
        result
    }
}

impl Add for &MultiPolyExact {
    type Output = MultiPolyExact;
    fn add(self, other: Self) -> MultiPolyExact {
        MultiPolyExact::add(self, other)
    }
}

impl Sub for &MultiPolyExact {
    type Output = MultiPolyExact;
    fn sub(self, other: Self) -> MultiPolyExact {
        MultiPolyExact::sub(self, other)
    }
}

impl Mul for &MultiPolyExact {
    type Output = MultiPolyExact;
    fn mul(self, other: Self) -> MultiPolyExact {
        MultiPolyExact::mul(self, other)
    }
}

impl Neg for &MultiPolyExact {
    type Output = MultiPolyExact;
    fn neg(self) -> MultiPolyExact {
        self.negate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;
    use num_traits::FromPrimitive;

    fn rat(n: i64) -> BigRational {
        BigRational::from_i64(n).unwrap()
    }

    #[test]
    fn test_constant() {
        let p = MultiPolyExact::constant(rat(5), 2);
        assert_eq!(p.num_terms(), 1);
        assert_eq!(p.max_total_degree(), 0);
    }

    #[test]
    fn test_var() {
        let x = MultiPolyExact::var(2, 0);
        assert_eq!(x.num_terms(), 1);
        assert_eq!(x.max_total_degree(), 1);
    }

    #[test]
    fn test_add() {
        let x = MultiPolyExact::var(2, 0);
        let y = MultiPolyExact::var(2, 1);
        let sum = &x + &y;
        assert_eq!(sum.num_terms(), 2);
    }

    #[test]
    fn test_mul() {
        let x = MultiPolyExact::var(2, 0);
        let y = MultiPolyExact::var(2, 1);
        let prod = &x * &y;
        assert_eq!(prod.num_terms(), 1);
        assert_eq!(prod.max_total_degree(), 2);
    }

    #[test]
    fn test_pow() {
        // (1 + x)^3 = 1 + 3x + 3x^2 + x^3
        let one = MultiPolyExact::constant(rat(1), 1);
        let x = MultiPolyExact::var(1, 0);
        let p = &one + &x;
        let p3 = p.pow(3);
        assert_eq!(p3.num_terms(), 4);
        assert_eq!(p3.max_total_degree(), 3);
    }

    #[test]
    fn test_cancellation() {
        let x = MultiPolyExact::var(1, 0);
        let neg_x = -&x;
        let sum = &x + &neg_x;
        assert!(sum.is_zero());
    }
}
