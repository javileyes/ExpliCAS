//! Arithmetic operations, content/GCD Layer 1, and exact division for MultiPoly.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::BTreeMap;

use super::{Monomial, MultiPoly, PolyBudget, PolyError, Term};

// =============================================================================
// Arithmetic
// =============================================================================

impl MultiPoly {
    /// Negate
    pub fn neg(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(c, m)| (-c.clone(), m.clone()))
            .collect();
        Self {
            vars: self.vars.clone(),
            terms,
        }
    }

    /// Add two polynomials (must have same vars)
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        let mut map = self.to_map();
        for (c, m) in &other.terms {
            let entry = map.entry(m.clone()).or_insert_with(BigRational::zero);
            *entry = entry.clone() + c.clone();
        }
        Ok(Self::from_map(self.vars.clone(), map))
    }

    /// Subtract
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        self.add(&other.neg())
    }

    /// Multiply by scalar
    pub fn mul_scalar(&self, k: &BigRational) -> Self {
        if k.is_zero() {
            return Self::zero(self.vars.clone());
        }
        let terms = self.terms.iter().map(|(c, m)| (c * k, m.clone())).collect();
        Self {
            vars: self.vars.clone(),
            terms,
        }
    }

    /// Multiply by monomial
    pub fn mul_monomial(&self, mono: &Monomial) -> Result<Self, PolyError> {
        if mono.len() != self.vars.len() {
            return Err(PolyError::VarMismatch);
        }
        let terms = self
            .terms
            .iter()
            .map(|(c, m)| {
                let new_mono: Monomial = m.iter().zip(mono.iter()).map(|(a, b)| a + b).collect();
                (c.clone(), new_mono)
            })
            .collect();
        Ok(Self {
            vars: self.vars.clone(),
            terms,
        })
    }

    /// Multiply two polynomials with budget
    pub fn mul(&self, other: &Self, budget: &PolyBudget) -> Result<Self, PolyError> {
        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero(self.vars.clone()));
        }

        let mut map: BTreeMap<Monomial, BigRational> = BTreeMap::new();

        for (c1, m1) in &self.terms {
            for (c2, m2) in &other.terms {
                let new_coeff = c1 * c2;
                let new_mono: Monomial = m1.iter().zip(m2.iter()).map(|(a, b)| a + b).collect();

                // Check total degree
                let td: u32 = new_mono.iter().sum();
                if td > budget.max_total_degree {
                    return Err(PolyError::BudgetExceeded);
                }

                let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
                *entry = entry.clone() + new_coeff;

                // Check term count
                if map.len() > budget.max_terms {
                    return Err(PolyError::BudgetExceeded);
                }
            }
        }

        Ok(Self::from_map(self.vars.clone(), map))
    }

    /// Fast multiplication using HashMap accumulation.
    /// Much faster than mul() for large polynomials (O(1) amortized per term vs O(log n)).
    /// Result is still normalized (sorted, no zero coeffs).
    pub fn mul_fast(&self, other: &Self, budget: &PolyBudget) -> Result<Self, PolyError> {
        use std::collections::HashMap;

        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero(self.vars.clone()));
        }

        // Estimate result size and pre-allocate
        let estimated_terms = self.terms.len() * other.terms.len() / 2;
        let mut map: HashMap<Monomial, BigRational> = HashMap::with_capacity(estimated_terms);

        for (c1, m1) in &self.terms {
            for (c2, m2) in &other.terms {
                let new_coeff = c1 * c2;
                let new_mono: Monomial = m1.iter().zip(m2.iter()).map(|(a, b)| a + b).collect();

                // Check total degree
                let td: u32 = new_mono.iter().sum();
                if td > budget.max_total_degree {
                    return Err(PolyError::BudgetExceeded);
                }

                // O(1) amortized insertion
                map.entry(new_mono)
                    .and_modify(|c| *c = c.clone() + &new_coeff)
                    .or_insert(new_coeff);

                // Check term count (less frequently to reduce overhead)
                if map.len() > budget.max_terms {
                    return Err(PolyError::BudgetExceeded);
                }
            }
        }

        // Convert to sorted Vec, remove zeros, swap tuple order (HashMap has (mono, coeff), Term is (coeff, mono))
        let mut terms: Vec<Term> = map
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(mono, coeff)| (coeff, mono))
            .collect();
        terms.sort_by(|a, b| a.1.cmp(&b.1)); // Sort by monomial (lex)

        Ok(Self {
            vars: self.vars.clone(),
            terms,
        })
    }

    /// Multiply two polynomials with budget tracking, returning PassStats.
    ///
    /// This is the instrumented version of `mul_fast` for unified budget charging.
    pub fn mul_with_stats(
        &self,
        other: &Self,
        budget: &PolyBudget,
    ) -> Result<(Self, crate::budget::PassStats), PolyError> {
        let result = self.mul_fast(other, budget)?;

        let stats = crate::budget::PassStats {
            op: crate::budget::Operation::PolyOps,
            rewrite_count: 0,
            nodes_delta: 0, // Poly ops don't create AST nodes
            terms_materialized: result.num_terms() as u64,
            poly_ops: 1, // Count this as one poly operation
            stop_reason: None,
        };

        Ok((result, stats))
    }
}

// =============================================================================
// Content / Primitive Part / Monomial GCD (Layer 1)
// =============================================================================

/// GCD of two BigRational (as fractions)
pub(crate) fn gcd_bigrational(a: &BigRational, b: &BigRational) -> BigRational {
    if a.is_zero() {
        return b.abs();
    }
    if b.is_zero() {
        return a.abs();
    }
    // gcd(a/b, c/d) = gcd(a*d, c*b) / (b*d)
    // But for content, we want gcd of numerators / lcm of denominators
    let num_gcd = a.numer().gcd(b.numer());
    let den_lcm = lcm_bigint(a.denom(), b.denom());
    BigRational::new(num_gcd, den_lcm)
}

fn lcm_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    if a.is_zero() || b.is_zero() {
        return BigInt::zero();
    }
    let g = a.gcd(b);
    (a / &g) * b
}

impl MultiPoly {
    /// Content: GCD of all coefficients
    pub fn content(&self) -> BigRational {
        if self.is_zero() {
            return BigRational::zero();
        }
        let mut g = BigRational::zero();
        for (c, _) in &self.terms {
            g = gcd_bigrational(&g, c);
        }
        // Ensure positive
        if g < BigRational::zero() {
            g = -g;
        }
        g
    }

    /// Primitive part: self / content
    pub fn primitive_part(&self) -> (BigRational, Self) {
        let c = self.content();
        if c.is_zero() || c.is_one() {
            return (c, self.clone());
        }
        let pp = self.div_scalar_exact(&c).unwrap_or_else(|| self.clone());
        (c, pp)
    }

    /// Divide by scalar (exact)
    pub fn div_scalar_exact(&self, k: &BigRational) -> Option<Self> {
        if k.is_zero() {
            return None;
        }
        let terms = self.terms.iter().map(|(c, m)| (c / k, m.clone())).collect();
        Some(Self {
            vars: self.vars.clone(),
            terms,
        })
    }

    /// Monomial GCD of self: min exponent per variable across all terms
    pub fn monomial_gcd(&self) -> Monomial {
        if self.is_zero() || self.vars.is_empty() {
            return vec![0; self.vars.len()];
        }
        let n = self.vars.len();
        let mut result = vec![u32::MAX; n];
        for (_, m) in &self.terms {
            for i in 0..n {
                result[i] = result[i].min(m[i]);
            }
        }
        // Replace MAX with 0 (shouldn't happen if non-empty)
        result.iter_mut().for_each(|e| {
            if *e == u32::MAX {
                *e = 0;
            }
        });
        result
    }

    /// Monomial GCD between two polys
    pub fn monomial_gcd_with(&self, other: &Self) -> Result<Monomial, PolyError> {
        if self.vars != other.vars {
            return Err(PolyError::VarMismatch);
        }
        let mg1 = self.monomial_gcd();
        let mg2 = other.monomial_gcd();
        Ok(mg1
            .iter()
            .zip(mg2.iter())
            .map(|(a, b)| (*a).min(*b))
            .collect())
    }

    /// Divide by monomial (exact - all terms must be divisible)
    pub fn div_monomial_exact(&self, mono: &Monomial) -> Option<Self> {
        if mono.len() != self.vars.len() {
            return None;
        }
        // Check all terms are divisible
        for (_, m) in &self.terms {
            for (i, &e) in mono.iter().enumerate() {
                if m[i] < e {
                    return None;
                }
            }
        }
        let terms = self
            .terms
            .iter()
            .map(|(c, m)| {
                let new_mono: Monomial = m.iter().zip(mono.iter()).map(|(a, b)| a - b).collect();
                (c.clone(), new_mono)
            })
            .collect();
        Some(Self {
            vars: self.vars.clone(),
            terms,
        })
    }
}

// =============================================================================
// Exact Division (sparse multivariate)
// =============================================================================

impl MultiPoly {
    /// Exact division: self / divisor
    /// Returns None if doesn't divide exactly
    pub fn div_exact(&self, divisor: &Self) -> Option<Self> {
        if divisor.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(Self::zero(self.vars.clone()));
        }
        if self.vars != divisor.vars {
            return None;
        }

        // Handle constant divisor
        if divisor.is_constant() {
            let c = divisor.constant_value()?;
            return self.div_scalar_exact(&c);
        }

        // Sparse division algorithm (lex order)
        let n_vars = self.vars.len();
        let mut remainder = self.to_map();
        let mut quotient: BTreeMap<Monomial, BigRational> = BTreeMap::new();

        // Leading term of divisor (largest in lex order)
        let (d_coeff, d_mono) = divisor.terms.last()?;

        while !remainder.is_empty() {
            // Get largest remaining term
            let (r_mono, r_coeff) = remainder.iter().next_back()?;
            let r_mono = r_mono.clone();
            let r_coeff = r_coeff.clone();

            // Check if divisor's leading monomial divides remainder's leading monomial
            let mut can_divide = true;
            let mut q_mono = vec![0u32; n_vars];
            for i in 0..n_vars {
                if r_mono[i] < d_mono[i] {
                    can_divide = false;
                    break;
                }
                q_mono[i] = r_mono[i] - d_mono[i];
            }

            if !can_divide {
                // Cannot divide - not exact
                return None;
            }

            // Quotient term
            let q_coeff = &r_coeff / d_coeff;

            // Subtract divisor * q_term from remainder
            for (dc, dm) in &divisor.terms {
                let sub_mono: Monomial = dm.iter().zip(q_mono.iter()).map(|(a, b)| a + b).collect();
                let sub_coeff = dc * &q_coeff;
                let entry = remainder.entry(sub_mono).or_insert_with(BigRational::zero);
                *entry = entry.clone() - sub_coeff;
            }

            // Clean zeros
            remainder.retain(|_, c| !c.is_zero());

            // Add to quotient
            let qe = quotient.entry(q_mono).or_insert_with(BigRational::zero);
            *qe = qe.clone() + q_coeff;
        }

        Some(Self::from_map(self.vars.clone(), quotient))
    }

    /// Exact division with budget tracking, returning PassStats.
    ///
    /// This is the instrumented version of `div_exact` for unified budget charging.
    pub fn div_exact_with_stats(&self, divisor: &Self) -> (Option<Self>, crate::budget::PassStats) {
        let result = self.div_exact(divisor);

        let terms = result.as_ref().map_or(0, |q| q.num_terms() as u64);

        let stats = crate::budget::PassStats {
            op: crate::budget::Operation::PolyOps,
            rewrite_count: 0,
            nodes_delta: 0,
            terms_materialized: terms,
            poly_ops: 1,
            stop_reason: None,
        };

        (result, stats)
    }
}
