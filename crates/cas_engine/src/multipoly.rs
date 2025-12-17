//! Sparse Multivariate Polynomial over Q
//!
//! Representation: terms stored as (coefficient, monomial) pairs.
//! Monomial = Vec<u32> of exponents aligned with vars vector.
//! Uses BTreeMap internally for normalization, Vec for storage.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::{BTreeMap, BTreeSet};

use cas_ast::{Context, Expr, ExprId};

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
}

impl Default for PolyBudget {
    fn default() -> Self {
        Self {
            max_terms: 200,
            max_total_degree: 32,
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
}

// =============================================================================
// Content / Primitive Part / Monomial GCD (Layer 1)
// =============================================================================

/// GCD of two BigRational (as fractions)
fn gcd_bigrational(a: &BigRational, b: &BigRational) -> BigRational {
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
            let (r_mono, r_coeff) = remainder.iter().next_back()?.clone();
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
}

// =============================================================================
// AST Conversion
// =============================================================================

/// Collect all variable names from expression
pub fn collect_poly_vars(ctx: &Context, expr: ExprId) -> BTreeSet<String> {
    let mut vars = BTreeSet::new();
    collect_vars_recursive(ctx, expr, &mut vars);
    vars
}

fn collect_vars_recursive(ctx: &Context, expr: ExprId, vars: &mut BTreeSet<String>) {
    match ctx.get(expr) {
        Expr::Variable(name) => {
            vars.insert(name.clone());
        }
        Expr::Number(_) | Expr::Constant(_) => {}
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_vars_recursive(ctx, *a, vars);
            collect_vars_recursive(ctx, *b, vars);
        }
        Expr::Neg(a) => {
            collect_vars_recursive(ctx, *a, vars);
        }
        _ => {} // Functions, matrices, etc. - ignore
    }
}

/// Convert expression to MultiPoly
pub fn multipoly_from_expr(
    ctx: &Context,
    expr: ExprId,
    budget: &PolyBudget,
) -> Result<MultiPoly, PolyError> {
    // Collect and sort variables
    let vars_set = collect_poly_vars(ctx, expr);
    let vars: Vec<String> = vars_set.into_iter().collect();

    // Convert
    from_expr_recursive(ctx, expr, &vars, budget)
}

fn from_expr_recursive(
    ctx: &Context,
    expr: ExprId,
    vars: &[String],
    budget: &PolyBudget,
) -> Result<MultiPoly, PolyError> {
    match ctx.get(expr) {
        Expr::Number(n) => Ok(MultiPoly {
            vars: vars.to_vec(),
            terms: if n.is_zero() {
                vec![]
            } else {
                vec![(n.clone(), vec![0; vars.len()])]
            },
        }),

        Expr::Variable(name) => {
            let idx = vars
                .iter()
                .position(|v| v == name)
                .ok_or(PolyError::NonPolynomial)?;
            let mut mono = vec![0; vars.len()];
            mono[idx] = 1;
            Ok(MultiPoly {
                vars: vars.to_vec(),
                terms: vec![(BigRational::one(), mono)],
            })
        }

        Expr::Neg(a) => {
            let p = from_expr_recursive(ctx, *a, vars, budget)?;
            Ok(p.neg())
        }

        Expr::Add(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            let result = pa.add(&pb)?;
            check_budget(&result, budget)?;
            Ok(result)
        }

        Expr::Sub(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            let result = pa.sub(&pb)?;
            check_budget(&result, budget)?;
            Ok(result)
        }

        Expr::Mul(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            pa.mul(&pb, budget)
        }

        Expr::Div(a, b) => {
            let pa = from_expr_recursive(ctx, *a, vars, budget)?;
            let pb = from_expr_recursive(ctx, *b, vars, budget)?;
            // Only allow division by constants
            if let Some(c) = pb.constant_value() {
                if c.is_zero() {
                    return Err(PolyError::NonPolynomial);
                }
                Ok(pa.mul_scalar(&(BigRational::one() / c)))
            } else {
                Err(PolyError::NonConstantDivision)
            }
        }

        Expr::Pow(base, exp) => {
            let pb = from_expr_recursive(ctx, *base, vars, budget)?;
            // Exponent must be non-negative integer constant
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n >= BigRational::zero() {
                    let e = n
                        .to_integer()
                        .try_into()
                        .map_err(|_| PolyError::BadExponent)?;
                    return pow_poly(&pb, e, budget);
                }
            }
            Err(PolyError::BadExponent)
        }

        Expr::Constant(_) => {
            // Constants like pi, e - treat as non-polynomial
            Err(PolyError::NonPolynomial)
        }

        _ => Err(PolyError::NonPolynomial),
    }
}

fn pow_poly(p: &MultiPoly, exp: u32, budget: &PolyBudget) -> Result<MultiPoly, PolyError> {
    if exp == 0 {
        return Ok(MultiPoly::one(p.vars.clone()));
    }
    if exp == 1 {
        return Ok(p.clone());
    }

    // Binary exponentiation
    let mut result = MultiPoly::one(p.vars.clone());
    let mut base = p.clone();
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&base, budget)?;
        }
        e >>= 1;
        if e > 0 {
            base = base.mul(&base, budget)?;
        }
    }

    Ok(result)
}

fn check_budget(p: &MultiPoly, budget: &PolyBudget) -> Result<(), PolyError> {
    if p.num_terms() > budget.max_terms {
        return Err(PolyError::BudgetExceeded);
    }
    if p.total_degree() > budget.max_total_degree {
        return Err(PolyError::BudgetExceeded);
    }
    Ok(())
}

/// Convert MultiPoly back to expression
pub fn multipoly_to_expr(p: &MultiPoly, ctx: &mut Context) -> ExprId {
    if p.is_zero() {
        return ctx.num(0);
    }

    let mut term_exprs: Vec<ExprId> = Vec::new();

    for (coeff, mono) in &p.terms {
        let term = build_term_expr(ctx, coeff, mono, &p.vars);
        term_exprs.push(term);
    }

    // Sum all terms
    let mut result = term_exprs[0];
    for &t in &term_exprs[1..] {
        result = ctx.add_raw(Expr::Add(result, t));
    }

    result
}

fn build_term_expr(
    ctx: &mut Context,
    coeff: &BigRational,
    mono: &Monomial,
    vars: &[String],
) -> ExprId {
    // Build monomial part: x^a * y^b * ...
    let mut factors: Vec<ExprId> = Vec::new();

    for (i, &exp) in mono.iter().enumerate() {
        if exp > 0 {
            let var_expr = ctx.add(Expr::Variable(vars[i].clone()));
            if exp == 1 {
                factors.push(var_expr);
            } else {
                let exp_expr = ctx.num(exp as i64);
                factors.push(ctx.add_raw(Expr::Pow(var_expr, exp_expr)));
            }
        }
    }

    // Combine monomial factors
    let mono_expr = if factors.is_empty() {
        None
    } else {
        let mut m = factors[0];
        for &f in &factors[1..] {
            m = ctx.add_raw(Expr::Mul(m, f));
        }
        Some(m)
    };

    // Combine with coefficient
    if coeff.is_one() {
        mono_expr.unwrap_or_else(|| ctx.num(1))
    } else if *coeff == -BigRational::one() {
        let m = mono_expr.unwrap_or_else(|| ctx.num(1));
        ctx.add_raw(Expr::Neg(m))
    } else {
        let c_expr = ctx.add_raw(Expr::Number(coeff.clone()));
        if let Some(m) = mono_expr {
            ctx.add_raw(Expr::Mul(c_expr, m))
        } else {
            c_expr
        }
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
        let mut ctx = Context::new();
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
