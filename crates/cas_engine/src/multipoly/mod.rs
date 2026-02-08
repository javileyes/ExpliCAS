//! Sparse Multivariate Polynomial over Q
//!
//! Representation: terms stored as (coefficient, monomial) pairs.
//! Monomial = Vec<u32> of exponents aligned with vars vector.
//! Uses BTreeMap internally for normalization, Vec for storage.
//!
//! # Module Structure
//!
//! - [`arithmetic`]: Arithmetic operations, content/GCD Layer 1, and exact division
//! - [`gcd`]: Multivariate GCD via evaluation-interpolation (Layers 2 and 2.5)
//! - [`conversion`]: AST ↔ MultiPoly conversion

pub mod arithmetic;
pub mod conversion;
pub mod gcd;

use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;

// Re-export public API so `crate::multipoly::X` continues to work
pub use conversion::{collect_poly_vars, multipoly_from_expr, multipoly_to_expr};
pub use gcd::{
    gcd_multivar_layer2, gcd_multivar_layer25, gcd_multivar_layer2_with_stats, GcdBudget, GcdLayer,
    Layer25Budget,
};

// =============================================================================
// Types
// =============================================================================

pub type VarIdx = usize;
pub type Exp = u32;

/// Monomial = vector of exponents aligned with `vars`
/// Invariant: monomial.len() == vars.len()
pub type Monomial = Vec<Exp>;

/// Term: (coeff != 0, monomial)
pub type Term = (BigRational, Monomial);

/// Limits to avoid explosion during conversion/operations
#[derive(Clone, Debug)]
pub struct PolyBudget {
    pub max_terms: usize,
    pub max_total_degree: u32,
    /// Maximum exponent for Pow(sum, n) conversion. Prevents expensive expansions.
    pub max_pow_exp: u32,
}

impl Default for PolyBudget {
    fn default() -> Self {
        Self {
            max_terms: 200,
            max_total_degree: 32,
            max_pow_exp: 2, // Conservative: only expand x^0, x^1, x^2 during GCD
        }
    }
}

/// Error types for polynomial operations
#[derive(Clone, Debug)]
pub enum PolyError {
    NonPolynomial,
    BadExponent,
    NonConstantDivision,
    BudgetExceeded,
    VarMismatch,
}

impl std::fmt::Display for PolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolyError::NonPolynomial => write!(f, "expression is not a polynomial over Q"),
            PolyError::BadExponent => write!(f, "non-integer or negative exponent"),
            PolyError::NonConstantDivision => write!(f, "division by non-constant expression"),
            PolyError::BudgetExceeded => write!(f, "budget exceeded"),
            PolyError::VarMismatch => write!(f, "variable mismatch"),
        }
    }
}

impl std::error::Error for PolyError {}

impl From<PolyError> for crate::error::CasError {
    fn from(e: PolyError) -> Self {
        match e {
            PolyError::BudgetExceeded => {
                use crate::budget::{BudgetExceeded, Metric, Operation};
                crate::error::CasError::BudgetExceeded(BudgetExceeded {
                    op: Operation::PolyOps,
                    metric: Metric::TermsMaterialized,
                    used: 0,
                    limit: 0,
                })
            }
            _ => crate::error::CasError::PolynomialError(e.to_string()),
        }
    }
}

// =============================================================================
// MultiPoly struct
// =============================================================================

/// Sparse multivariate polynomial over Q
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPoly {
    /// Ordered variable list (lexicographic)
    pub vars: Vec<String>,
    /// Terms in lex order by monomial, no zero coefficients
    pub terms: Vec<Term>,
}

impl MultiPoly {
    /// Create zero polynomial
    pub fn zero(vars: Vec<String>) -> Self {
        Self {
            vars,
            terms: Vec::new(),
        }
    }

    /// Create constant 1
    pub fn one(vars: Vec<String>) -> Self {
        let mono = vec![0; vars.len()];
        Self {
            vars,
            terms: vec![(BigRational::one(), mono)],
        }
    }

    /// Create from constant (no variables)
    pub fn from_const(c: BigRational) -> Self {
        if c.is_zero() {
            Self::zero(Vec::new())
        } else {
            Self {
                vars: Vec::new(),
                terms: vec![(c, Vec::new())],
            }
        }
    }

    /// Create monomial x_i (single variable with exp 1)
    pub fn from_var(vars: Vec<String>, var: &str) -> Result<Self, PolyError> {
        let idx = vars
            .iter()
            .position(|v| v == var)
            .ok_or(PolyError::VarMismatch)?;
        let mut mono = vec![0; vars.len()];
        mono[idx] = 1;
        Ok(Self {
            vars,
            terms: vec![(BigRational::one(), mono)],
        })
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if constant 1
    pub fn is_one(&self) -> bool {
        self.terms.len() == 1
            && self.terms[0].0 == BigRational::one()
            && self.terms[0].1.iter().all(|&e| e == 0)
    }

    /// Check if constant (possibly 0)
    pub fn is_constant(&self) -> bool {
        self.terms.is_empty() || (self.terms.len() == 1 && self.terms[0].1.iter().all(|&e| e == 0))
    }

    /// Get constant value if constant
    pub fn constant_value(&self) -> Option<BigRational> {
        if self.is_zero() {
            Some(BigRational::zero())
        } else if self.is_constant() {
            Some(self.terms[0].0.clone())
        } else {
            None
        }
    }

    /// Degree in a specific variable
    pub fn degree_in(&self, v: VarIdx) -> u32 {
        self.terms
            .iter()
            .map(|(_, m)| m.get(v).copied().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    /// Total degree (max sum of exponents)
    pub fn total_degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|(_, m)| m.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    /// Leading term in lex order (first in sorted list)
    pub fn leading_term_lex(&self) -> Option<&Term> {
        self.terms.last() // terms are in ascending lex order, so last is largest
    }

    /// Number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Get variable index by name.
    ///
    /// Returns `Some(idx)` if the variable exists in this polynomial's variable list.
    pub fn var_index(&self, name: &str) -> Option<VarIdx> {
        self.vars.iter().position(|v| v == name)
    }

    /// Leading coefficient in a specific variable.
    ///
    /// Returns a new polynomial containing the sum of all terms with max degree in `v`,
    /// with the variable `v` factored out (set to degree 0).
    ///
    /// # Example
    /// For `P(x,y) = (2y+1)*x^3 + x^2 + y`, the leading coefficient in `x` is `2y+1`.
    ///
    /// # Returns
    /// - If `max_deg == 0`, returns the full polynomial (constant wrt var)
    /// - If polynomial is zero, returns zero
    pub fn leading_coeff_in(&self, v: VarIdx) -> Self {
        if self.is_zero() {
            return Self::zero(self.vars.clone());
        }
        if v >= self.vars.len() {
            return self.clone(); // var not in this poly → return self
        }

        let max_deg = self.degree_in(v);

        let mut map: std::collections::BTreeMap<Monomial, BigRational> =
            std::collections::BTreeMap::new();

        for (coeff, mono) in &self.terms {
            if mono.get(v).copied().unwrap_or(0) == max_deg {
                // Create new monomial with v's exponent set to 0
                let mut new_mono = mono.clone();
                new_mono[v] = 0;

                // Accumulate coefficient
                let entry = map.entry(new_mono).or_insert_with(BigRational::zero);
                *entry = entry.clone() + coeff.clone();
            }
        }

        Self::from_map(self.vars.clone(), map)
    }
}

// =============================================================================
// Normalization (combine like terms, remove zeros, sort lex)
// =============================================================================

impl MultiPoly {
    /// Build from BTreeMap (map has Monomial as key, coeff as value)
    pub fn from_map(vars: Vec<String>, map: BTreeMap<Monomial, BigRational>) -> Self {
        let terms: Vec<Term> = map
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(m, c)| (c, m)) // Convert to (coeff, mono) for Term
            .collect();
        Self { vars, terms }
    }

    /// Convert to BTreeMap for operations (key=Monomial, value=coeff)
    pub fn to_map(&self) -> BTreeMap<Monomial, BigRational> {
        self.terms
            .iter()
            .map(|(c, m)| (m.clone(), c.clone()))
            .collect()
    }

    /// Normalize in place (if needed)
    pub fn normalize(&mut self) {
        let map = self.to_map();
        self.terms = map
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(m, c)| (c, m))
            .collect();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn parse_to_multipoly(input: &str) -> MultiPoly {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(input, &mut ctx).expect("parse failed");
        multipoly_from_expr(&ctx, expr, &PolyBudget::default()).expect("conversion failed")
    }

    #[test]
    fn test_from_expr_simple() {
        let p = parse_to_multipoly("x + 1");
        assert_eq!(p.vars, vec!["x".to_string()]);
        assert_eq!(p.num_terms(), 2);
    }

    #[test]
    fn test_from_expr_multivar() {
        let p = parse_to_multipoly("x*y + x + y + 1");
        assert_eq!(p.vars, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(p.num_terms(), 4);
    }

    #[test]
    fn test_content() {
        let p = parse_to_multipoly("2*x + 4*y");
        let c = p.content();
        assert_eq!(c, BigRational::from_integer(2.into()));
    }

    #[test]
    fn test_monomial_gcd() {
        let p = parse_to_multipoly("x^2*y + x^3*y^2");
        let mg = p.monomial_gcd();
        assert_eq!(mg, vec![2, 1]); // x^2 * y
    }

    #[test]
    fn test_div_monomial_exact() {
        let p = parse_to_multipoly("x^2*y + x^3*y^2");
        let mg = p.monomial_gcd();
        let q = p.div_monomial_exact(&mg).expect("should divide");
        // Result: 1 + x*y
        assert_eq!(q.num_terms(), 2);
    }

    #[test]
    fn test_div_exact() {
        let p = parse_to_multipoly("x^2 - 1");
        let d = parse_to_multipoly("x - 1");
        let q = p.div_exact(&d).expect("should divide exactly");
        // Result: x + 1
        assert_eq!(q.num_terms(), 2);
    }

    #[test]
    fn test_div_exact_fails() {
        let p = parse_to_multipoly("x^2 + 1");
        let d = parse_to_multipoly("x - 1");
        assert!(p.div_exact(&d).is_none());
    }

    #[test]
    fn test_mul() {
        let p = parse_to_multipoly("x + 1");
        let q = parse_to_multipoly("x - 1");
        let r = p.mul(&q, &PolyBudget::default()).expect("mul");
        // Result: x^2 - 1
        assert_eq!(r.num_terms(), 2);
    }
}
