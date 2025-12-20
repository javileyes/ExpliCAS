//! Univariate polynomial over Fp (mod p).
//!
//! Coefficients stored low-to-high: coeffs[i] is coefficient of x^i.

use crate::modp::{add_mod, mul_mod, sub_mod};

/// Univariate polynomial mod p
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniPolyModP {
    pub p: u64,
    /// Coefficients from degree 0 upward: coeffs[i] = coefficient of x^i
    pub coeffs: Vec<u64>,
}

impl UniPolyModP {
    /// Create zero polynomial
    pub fn zero(p: u64) -> Self {
        Self { p, coeffs: vec![] }
    }

    /// Create constant polynomial
    pub fn constant(c: u64, p: u64) -> Self {
        if c % p == 0 {
            Self::zero(p)
        } else {
            Self {
                p,
                coeffs: vec![c % p],
            }
        }
    }

    /// Create monomial x^deg
    pub fn monomial(deg: usize, p: u64) -> Self {
        let mut coeffs = vec![0u64; deg + 1];
        coeffs[deg] = 1;
        Self { p, coeffs }
    }

    /// Remove trailing zeros
    pub fn trim(&mut self) {
        while self.coeffs.last() == Some(&0) {
            self.coeffs.pop();
        }
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0)
    }

    /// Degree (-1 if zero, represented as 0 for simplicity)
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            let mut d = self.coeffs.len() - 1;
            while d > 0 && self.coeffs[d] == 0 {
                d -= 1;
            }
            if self.coeffs[d] == 0 {
                0
            } else {
                d
            }
        }
    }

    /// Leading coefficient
    pub fn lc(&self) -> u64 {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs[self.degree()]
        }
    }

    /// Make monic (divide by leading coefficient)
    pub fn make_monic(&mut self) {
        self.trim();
        let lc = self.lc();
        if lc == 0 || lc == 1 {
            return;
        }
        if let Some(lc_inv) = crate::modp::inv_mod(lc, self.p) {
            for c in &mut self.coeffs {
                *c = mul_mod(*c, lc_inv, self.p);
            }
        }
    }

    /// Add two polynomials
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.p, other.p);
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0u64; max_len];
        for (i, c) in result.iter_mut().enumerate() {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            *c = add_mod(a, b, self.p);
        }
        let mut poly = Self {
            p: self.p,
            coeffs: result,
        };
        poly.trim();
        poly
    }

    /// Subtract
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.p, other.p);
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0u64; max_len];
        for (i, c) in result.iter_mut().enumerate() {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            *c = sub_mod(a, b, self.p);
        }
        let mut poly = Self {
            p: self.p,
            coeffs: result,
        };
        poly.trim();
        poly
    }

    /// Multiply
    pub fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.p, other.p);
        if self.is_zero() || other.is_zero() {
            return Self::zero(self.p);
        }
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut result = vec![0u64; n];
        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0 {
                continue;
            }
            for (j, &b) in other.coeffs.iter().enumerate() {
                let prod = mul_mod(a, b, self.p);
                result[i + j] = add_mod(result[i + j], prod, self.p);
            }
        }
        let mut poly = Self {
            p: self.p,
            coeffs: result,
        };
        poly.trim();
        poly
    }

    /// Euclidean division: returns (quotient, remainder)
    /// such that self = quotient * divisor + remainder
    pub fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        debug_assert_eq!(self.p, divisor.p);
        if divisor.is_zero() {
            return None;
        }

        let mut remainder = self.clone();
        remainder.trim();

        let divisor_deg = divisor.degree();
        let divisor_lc = divisor.lc();
        let divisor_lc_inv = crate::modp::inv_mod(divisor_lc, self.p)?;

        if remainder.degree() < divisor_deg {
            return Some((Self::zero(self.p), remainder));
        }

        let quot_len = remainder.degree() - divisor_deg + 1;
        let mut quotient = vec![0u64; quot_len];

        while !remainder.is_zero() && remainder.degree() >= divisor_deg {
            let r_deg = remainder.degree();
            let r_lc = remainder.lc();

            // q_coef = lc(remainder) / lc(divisor)
            let q_coef = mul_mod(r_lc, divisor_lc_inv, self.p);
            let q_deg = r_deg - divisor_deg;

            quotient[q_deg] = q_coef;

            // remainder -= q_coef * x^q_deg * divisor
            for (i, &d_coef) in divisor.coeffs.iter().enumerate() {
                let term = mul_mod(q_coef, d_coef, self.p);
                let pos = q_deg + i;
                remainder.coeffs[pos] = sub_mod(remainder.coeffs[pos], term, self.p);
            }
            remainder.trim();
        }

        Some((
            Self {
                p: self.p,
                coeffs: quotient,
            },
            remainder,
        ))
    }

    /// GCD using Euclidean algorithm
    pub fn gcd(&self, other: &Self) -> Self {
        debug_assert_eq!(self.p, other.p);
        let mut a = self.clone();
        let mut b = other.clone();
        a.trim();
        b.trim();

        while !b.is_zero() {
            if let Some((_, rem)) = a.div_rem(&b) {
                a = b;
                b = rem;
            } else {
                break;
            }
        }

        a.make_monic();
        a
    }

    /// Evaluate at a point
    pub fn eval(&self, x: u64) -> u64 {
        let mut result = 0u64;
        let mut x_pow = 1u64;
        for &c in &self.coeffs {
            result = add_mod(result, mul_mod(c, x_pow, self.p), self.p);
            x_pow = mul_mod(x_pow, x, self.p);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 17;

    #[test]
    fn test_add() {
        // (x + 1) + (x + 2) = 2x + 3
        let a = UniPolyModP {
            p: P,
            coeffs: vec![1, 1],
        };
        let b = UniPolyModP {
            p: P,
            coeffs: vec![2, 1],
        };
        let c = a.add(&b);
        assert_eq!(c.coeffs, vec![3, 2]);
    }

    #[test]
    fn test_mul() {
        // (x + 1) * (x + 2) = x^2 + 3x + 2
        let a = UniPolyModP {
            p: P,
            coeffs: vec![1, 1],
        };
        let b = UniPolyModP {
            p: P,
            coeffs: vec![2, 1],
        };
        let c = a.mul(&b);
        assert_eq!(c.coeffs, vec![2, 3, 1]);
    }

    #[test]
    fn test_div_rem() {
        // (x^2 + 3x + 2) / (x + 1) = (x + 2), rem 0
        let dividend = UniPolyModP {
            p: P,
            coeffs: vec![2, 3, 1],
        };
        let divisor = UniPolyModP {
            p: P,
            coeffs: vec![1, 1],
        };
        let (q, r) = dividend.div_rem(&divisor).unwrap();
        assert_eq!(q.coeffs, vec![2, 1]); // x + 2
        assert!(r.is_zero());
    }

    #[test]
    fn test_gcd() {
        // gcd((x+1)(x+2), (x+1)(x+3)) = x+1 (monic)
        let f1 = UniPolyModP {
            p: P,
            coeffs: vec![1, 1],
        }; // x+1
        let f2 = UniPolyModP {
            p: P,
            coeffs: vec![2, 1],
        }; // x+2
        let f3 = UniPolyModP {
            p: P,
            coeffs: vec![3, 1],
        }; // x+3

        let a = f1.mul(&f2); // (x+1)(x+2)
        let b = f1.mul(&f3); // (x+1)(x+3)

        let g = a.gcd(&b);
        // Should be monic (x+1)
        assert_eq!(g.degree(), 1);
        assert_eq!(g.lc(), 1);
        // Verify: (x+1) has root at x = -1 = 16 mod 17
        assert_eq!(g.eval(16), 0);
    }

    #[test]
    fn test_eval() {
        // f(x) = x^2 + 2x + 3, f(2) = 4 + 4 + 3 = 11
        let f = UniPolyModP {
            p: P,
            coeffs: vec![3, 2, 1],
        };
        assert_eq!(f.eval(2), 11);
    }
}
