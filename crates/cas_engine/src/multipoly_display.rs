//! Display utilities for MultiPoly → Expr conversion.
//!
//! Used by the didactic narrator to show normalized polynomial forms
//! in a readable way.

use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use crate::multipoly::MultiPoly;

/// Proof data for polynomial identity cancellation.
/// Attached to Step for didactic display in explain mode.
#[derive(Debug, Clone)]
pub struct PolynomialProofData {
    /// Number of monomials in the normalized form
    pub monomials: usize,
    /// Maximum degree of the polynomial
    pub degree: usize,
    /// Variable names used
    pub vars: Vec<String>,
    /// The normalized form as an expression (if small enough to display)
    /// None if the polynomial is too large (> MAX_MONOMIALS_FOR_DISPLAY)
    pub normal_form_expr: Option<ExprId>,
}

/// Budget limits for display conversion
const MAX_MONOMIALS_FOR_DISPLAY: usize = 30;
const MAX_NODES_BUILD: usize = 400;

impl PolynomialProofData {
    /// Create proof data from a MultiPoly, optionally building display expression
    pub fn from_multipoly(ctx: &mut Context, poly: &MultiPoly, vars: Vec<String>) -> Self {
        let monomials = poly.terms.len();
        let degree = poly.total_degree() as usize;

        // Only build display expression if small enough
        let normal_form_expr = if monomials <= MAX_MONOMIALS_FOR_DISPLAY && monomials > 0 {
            multipoly_to_expr_budgeted(ctx, poly, MAX_NODES_BUILD)
        } else {
            None
        };

        PolynomialProofData {
            monomials,
            degree,
            vars,
            normal_form_expr,
        }
    }

    /// Format a summary string (for when expression is too big)
    pub fn summary(&self) -> String {
        format!(
            "{} monomials, degree {} in {} variable{}",
            self.monomials,
            self.degree,
            self.vars.len(),
            if self.vars.len() == 1 { "" } else { "s" }
        )
    }
}

/// Convert MultiPoly to expression for display, with budget limit.
/// Returns None if budget exceeded.
///
/// Uses grevlex ordering (degree descending, then lex) for deterministic output.
/// Handles special cases: 1·x → x, (-1)·x → -x, x^1 → x
fn multipoly_to_expr_budgeted(
    ctx: &mut Context,
    poly: &MultiPoly,
    max_nodes: usize,
) -> Option<ExprId> {
    if poly.terms.is_empty() {
        return Some(ctx.num(0));
    }

    // Sort terms by grevlex: total degree descending, then lex on exponents
    let mut sorted_terms: Vec<_> = poly.terms.iter().collect();
    sorted_terms.sort_by(|(_, mono_a), (_, mono_b)| {
        let deg_a: u32 = mono_a.iter().sum();
        let deg_b: u32 = mono_b.iter().sum();
        // Descending degree first
        match deg_b.cmp(&deg_a) {
            std::cmp::Ordering::Equal => {
                // Then lex on exponents (reversed for standard order)
                mono_b.cmp(mono_a)
            }
            other => other,
        }
    });

    let mut nodes_built = 0;
    let mut terms_built = Vec::new();

    for (coeff, exponents) in sorted_terms {
        if coeff.is_zero() {
            continue;
        }

        // Build monomial: coeff * x1^e1 * x2^e2 * ...
        let monomial = build_monomial(ctx, coeff, exponents, &poly.vars, &mut nodes_built)?;

        if nodes_built > max_nodes {
            return None; // Budget exceeded
        }

        terms_built.push(monomial);
    }

    if terms_built.is_empty() {
        return Some(ctx.num(0));
    }

    // Combine terms with addition (using add_raw to avoid canonicalization)
    let mut result = terms_built[0];
    for term in terms_built.into_iter().skip(1) {
        result = ctx.add(Expr::Add(result, term));
        nodes_built += 1;
        if nodes_built > max_nodes {
            return None;
        }
    }

    Some(result)
}

/// Build a single monomial: coeff * x1^e1 * x2^e2 * ...
fn build_monomial(
    ctx: &mut Context,
    coeff: &BigRational,
    exponents: &[u32],
    vars: &[String],
    nodes_built: &mut usize,
) -> Option<ExprId> {
    if coeff.is_zero() {
        *nodes_built += 1;
        return Some(ctx.num(0));
    }

    // Build variable factors: x1^e1 * x2^e2 * ...
    let mut var_factors = Vec::new();
    for (i, &exp) in exponents.iter().enumerate() {
        if exp == 0 {
            continue;
        }
        if i >= vars.len() {
            continue;
        }

        let var_expr = ctx.var(&vars[i]);
        *nodes_built += 1;

        let factor = if exp == 1 {
            var_expr
        } else {
            let exp_num = ctx.num(exp as i64);
            *nodes_built += 1;
            let pow = ctx.add(Expr::Pow(var_expr, exp_num));
            *nodes_built += 1;
            pow
        };

        var_factors.push(factor);
    }

    // Handle coefficient
    let is_one = coeff == &BigRational::one();
    let is_neg_one = coeff == &(-BigRational::one());

    if var_factors.is_empty() {
        // Pure constant
        let num_expr = bigrational_to_expr(ctx, coeff);
        *nodes_built += 1;
        return Some(num_expr);
    }

    // Combine variable factors with multiplication
    let mut var_product = var_factors[0];
    for factor in var_factors.into_iter().skip(1) {
        var_product = ctx.add(Expr::Mul(var_product, factor));
        *nodes_built += 1;
    }

    if is_one {
        // 1·x → x
        Some(var_product)
    } else if is_neg_one {
        // (-1)·x → -x
        let neg = ctx.add(Expr::Neg(var_product));
        *nodes_built += 1;
        Some(neg)
    } else if coeff.is_negative() {
        // Negative: -|c|·x
        let abs_coeff = -coeff.clone();
        let coeff_expr = bigrational_to_expr(ctx, &abs_coeff);
        *nodes_built += 1;
        let product = ctx.add(Expr::Mul(coeff_expr, var_product));
        *nodes_built += 1;
        let neg = ctx.add(Expr::Neg(product));
        *nodes_built += 1;
        Some(neg)
    } else {
        // Positive: c·x
        let coeff_expr = bigrational_to_expr(ctx, coeff);
        *nodes_built += 1;
        let product = ctx.add(Expr::Mul(coeff_expr, var_product));
        *nodes_built += 1;
        Some(product)
    }
}

/// Convert BigRational to expression
fn bigrational_to_expr(ctx: &mut Context, r: &BigRational) -> ExprId {
    use num_traits::ToPrimitive;

    if r.denom().is_one() {
        // Integer - use num() for small values, or direct Number construction for big
        if let Some(i) = r.numer().to_i64() {
            ctx.num(i)
        } else {
            // Large integer: construct directly
            ctx.add(Expr::Number(BigRational::from_integer(r.numer().clone())))
        }
    } else {
        // Fraction: construct Number directly with the rational
        ctx.add(Expr::Number(r.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multipoly::MultiPoly;
    use num_rational::BigRational;

    #[test]
    fn test_zero_poly() {
        let mut ctx = Context::new();
        let poly = MultiPoly::zero(vec!["x".to_string()]);
        let vars = vec!["x".to_string()];

        let proof = PolynomialProofData::from_multipoly(&mut ctx, &poly, vars);
        assert_eq!(proof.monomials, 0);
        assert_eq!(proof.degree, 0);
    }

    #[test]
    fn test_constant_poly() {
        let mut ctx = Context::new();
        let poly = MultiPoly::from_const(BigRational::from_integer(5.into()));
        let vars = vec![];

        let proof = PolynomialProofData::from_multipoly(&mut ctx, &poly, vars);
        assert_eq!(proof.monomials, 1);
        assert_eq!(proof.degree, 0);
        assert!(proof.normal_form_expr.is_some());
    }

    #[test]
    fn test_summary_format() {
        let mut ctx = Context::new();
        let poly = MultiPoly::one(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        let vars = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let proof = PolynomialProofData::from_multipoly(&mut ctx, &poly, vars);
        assert!(proof.summary().contains("variable"));
    }
}
