//! Display utilities for MultiPoly → Expr conversion.
//!
//! Used by the didactic narrator to show normalized polynomial forms
//! in a readable way.

use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use cas_math::multipoly::MultiPoly;

/// Stats for a single polynomial normal form
#[derive(Debug, Clone)]
pub struct PolyNormalFormStats {
    /// Number of monomials
    pub monomials: usize,
    /// Maximum degree
    pub degree: usize,
    /// The normalized form as an expression (if small enough)
    pub expr: Option<ExprId>,
}

impl PolyNormalFormStats {
    pub fn from_multipoly(ctx: &mut Context, poly: &MultiPoly) -> Self {
        let monomials = poly.terms.len();
        let degree = poly.total_degree() as usize;
        let expr = if monomials <= MAX_MONOMIALS_FOR_DISPLAY && monomials > 0 {
            multipoly_to_expr_budgeted(ctx, poly, MAX_NODES_BUILD)
        } else {
            None
        };
        PolyNormalFormStats {
            monomials,
            degree,
            expr,
        }
    }
}

/// Proof data for polynomial identity cancellation.
/// Attached to Step for didactic display in explain mode.
#[derive(Debug, Clone)]
pub struct PolynomialProofData {
    /// Number of monomials in the difference (result = 0)
    pub monomials: usize,
    /// Maximum degree of the polynomial
    pub degree: usize,
    /// Variable names used
    pub vars: Vec<String>,
    /// The normalized form as an expression (if small enough to display)
    /// None if the polynomial is too large (> MAX_MONOMIALS_FOR_DISPLAY)
    pub normal_form_expr: Option<ExprId>,
    /// The fully expanded form as an expression (each product expanded,
    /// but terms not yet cancelled). Shows e.g. `t² + 3t + 2 - t² - 3t - 2`.
    pub expanded_form_expr: Option<ExprId>,

    /// LHS normal form (for Sub(lhs, rhs) patterns)
    /// This is the "left side" before cancellation
    pub lhs_stats: Option<PolyNormalFormStats>,
    /// RHS normal form (for Sub(lhs, rhs) patterns)
    /// This is the "right side" before cancellation
    pub rhs_stats: Option<PolyNormalFormStats>,

    /// Opaque substitution mapping for didactic display.
    /// Each entry is (temp_var_name, original_expr_id) — e.g., ("t₀", sin(u))
    /// Empty for non-opaque polynomial identities.
    pub opaque_substitutions: Vec<(String, ExprId)>,
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
            expanded_form_expr: None,
            lhs_stats: None,
            rhs_stats: None,
            opaque_substitutions: Vec::new(),
        }
    }

    /// Create proof data for an identity with LHS and RHS normal forms
    pub fn from_identity(
        ctx: &mut Context,
        lhs_poly: &MultiPoly,
        rhs_poly: &MultiPoly,
        vars: Vec<String>,
    ) -> Self {
        let lhs_stats = PolyNormalFormStats::from_multipoly(ctx, lhs_poly);
        let rhs_stats = PolyNormalFormStats::from_multipoly(ctx, rhs_poly);

        PolynomialProofData {
            monomials: 0, // Identity means difference = 0
            degree: 0,
            vars,
            normal_form_expr: Some(ctx.num(0)),
            expanded_form_expr: None,
            lhs_stats: Some(lhs_stats),
            rhs_stats: Some(rhs_stats),
            opaque_substitutions: Vec::new(),
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

/// Expand each top-level additive term independently through the multipoly system.
///
/// This produces the "expanded but not cancelled" form:
/// `(t+1)(t+2) - t² - 3t - 2` → `t² + 3t + 2 - t² - 3t - 2`
///
/// Each term is expanded to its polynomial normal form, then converted back
/// to an expression. The results are combined with addition (using the sign
/// from the original expression).
///
/// Returns None if: too many terms, budget exceeded, or any term is not polynomial.
pub fn expand_additive_terms(
    ctx: &mut Context,
    expr: ExprId,
    display_vars: &[String],
) -> Option<ExprId> {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    // Phase 1 (immutable borrow): collect terms and convert to multipolys
    let mut signed_terms: Vec<(bool, ExprId)> = Vec::new();
    collect_signed_terms(ctx, expr, true, &mut signed_terms);

    if signed_terms.is_empty() || signed_terms.len() > 20 {
        return None;
    }

    let budget = PolyBudget {
        max_terms: 200,
        max_total_degree: 32,
        max_pow_exp: 12,
    };

    // Convert each term to multipoly (needs &Context only)
    let mut poly_results: Vec<(bool, Option<MultiPoly>, ExprId)> = Vec::new();
    for (positive, term) in &signed_terms {
        match multipoly_from_expr(ctx, *term, &budget) {
            Ok(poly) => {
                let display_poly = remap_poly_vars(&poly, display_vars);
                poly_results.push((*positive, Some(display_poly), *term));
            }
            Err(_) => {
                poly_results.push((*positive, None, *term));
            }
        }
    }

    // Phase 2 (mutable borrow): build expressions from multipolys
    let mut expanded_exprs: Vec<(bool, ExprId)> = Vec::new();
    for (positive, poly_opt, original_term) in poly_results {
        if let Some(poly) = poly_opt {
            if let Some(term_expr) = multipoly_to_expr_budgeted(ctx, &poly, MAX_NODES_BUILD) {
                expanded_exprs.push((positive, term_expr));
            } else {
                return None;
            }
        } else {
            expanded_exprs.push((positive, original_term));
        }
    }

    if expanded_exprs.is_empty() {
        return None;
    }

    // Combine expanded terms: result = ±t1 ± t2 ± ...
    let (first_positive, first_expr) = expanded_exprs[0];
    let mut result = if first_positive {
        first_expr
    } else {
        ctx.add(Expr::Neg(first_expr))
    };

    for (positive, term_expr) in expanded_exprs.into_iter().skip(1) {
        if positive {
            result = ctx.add(Expr::Add(result, term_expr));
        } else {
            result = ctx.add(Expr::Sub(result, term_expr));
        }
    }

    Some(result)
}

/// Remap poly variable names to display-friendly names.
///
/// Variable names like `__opq0`, `__opq1` are converted to display names
/// like `t`, `t₀`, `t₁` based on position in the display_vars list.
/// Other variable names are kept as-is.
fn remap_poly_vars(poly: &MultiPoly, display_vars: &[String]) -> MultiPoly {
    // If all vars already match display_vars, return as-is
    if poly.vars == display_vars {
        return poly.clone();
    }

    // Build new var list with display names
    let new_vars: Vec<String> = poly
        .vars
        .iter()
        .map(|v| {
            // Check if this is an opaque var (__opqN) and remap to display name
            if let Some(idx_str) = v.strip_prefix("__opq") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx < display_vars.len() {
                        return display_vars[idx].clone();
                    }
                }
            }
            // Check if this var exists in display_vars
            if display_vars.contains(v) {
                return v.clone();
            }
            v.clone()
        })
        .collect();

    MultiPoly {
        vars: new_vars,
        terms: poly.terms.clone(),
    }
}

/// Collect signed additive terms from an expression.
/// `Add(a, b)` splits into `+a, +b`
/// `Sub(a, b)` splits into `+a, -b`
/// `Neg(a)` flips sign
fn collect_signed_terms(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<(bool, ExprId)>,
) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_signed_terms(ctx, *l, positive, terms);
            collect_signed_terms(ctx, *r, positive, terms);
        }
        Expr::Sub(l, r) => {
            collect_signed_terms(ctx, *l, positive, terms);
            collect_signed_terms(ctx, *r, !positive, terms);
        }
        Expr::Neg(e) => {
            collect_signed_terms(ctx, *e, !positive, terms);
        }
        _ => {
            terms.push((positive, expr));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_math::multipoly::MultiPoly;
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
