//! Algebraic polynomial GCD over ℚ[x1,...,xn].
//!
//! `poly_gcd_exact(a, b)` computes the actual polynomial GCD by converting
//! expressions to `MultiPoly` and using the Layer 1/2/2.5 pipeline.

use crate::multipoly::{
    gcd_multivar_layer2, gcd_multivar_layer25, multipoly_from_expr, multipoly_to_expr, GcdBudget,
    Layer25Budget, MultiPoly, PolyBudget,
};
use crate::phase::PhaseMask;
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Budget for poly_gcd_exact computation
#[derive(Clone, Debug)]
pub struct GcdExactBudget {
    pub max_vars: usize,
    pub max_terms_input: usize,
    pub max_total_degree: usize,
}

impl Default for GcdExactBudget {
    fn default() -> Self {
        GcdExactBudget {
            max_vars: 5,
            max_terms_input: 500,
            max_total_degree: 50,
        }
    }
}

/// Result of poly_gcd_exact computation
#[derive(Clone, Debug)]
pub struct GcdExactResult {
    pub gcd: ExprId,
    pub layer_used: GcdExactLayer,
    pub warnings: Vec<String>,
}

/// Which GCD algorithm layer succeeded
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GcdExactLayer {
    /// Input was zero or constant
    Trivial,
    /// Univariate Euclidean GCD
    Univariate,
    /// Layer 1: monomial + content GCD only
    Layer1MonomialContent,
    /// Layer 2: heuristic seeds interpolation
    Layer2HeuristicSeeds,
    /// Layer 2.5: tensor grid interpolation
    Layer25TensorGrid,
    /// Budget exceeded, returned 1
    BudgetExceeded,
}

/// Strip __hold() wrapper(s) from an expression. __hold is an internal barrier
/// that should be transparent for algebraic operations like poly_gcd_exact.
/// Uses canonical implementation from cas_ast::hold
fn strip_hold(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, expr);
        if unwrapped == expr {
            return expr;
        }
        expr = unwrapped;
    }
}

/// Compute exact polynomial GCD over ℚ[x1,...,xn].
///
/// # Contract
/// - Result is **primitive**: GCD of coefficients = 1
/// - Result has **positive leading coefficient** (in lex monomial order)
/// - `gcd(0, p) = normalize(p)`
/// - `gcd(p, 0) = normalize(p)`
/// - `gcd(c1, c2) = 1` for non-zero constants (over ℚ, any non-zero constant divides any other)
pub fn gcd_exact(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    budget: &GcdExactBudget,
) -> GcdExactResult {
    // Strip __hold wrappers first (makes expand() output transparent)
    let a = strip_hold(ctx, a);
    let b = strip_hold(ctx, b);

    let mut warnings = Vec::new();

    // Convert to MultiPoly
    let poly_budget = PolyBudget::default();

    let p_a = match multipoly_from_expr(ctx, a, &poly_budget) {
        Ok(p) => p,
        Err(_) => {
            warnings.push("poly_gcd_exact: first argument is not a polynomial".to_string());
            return GcdExactResult {
                gcd: ctx.num(1),
                layer_used: GcdExactLayer::BudgetExceeded,
                warnings,
            };
        }
    };

    let p_b = match multipoly_from_expr(ctx, b, &poly_budget) {
        Ok(p) => p,
        Err(_) => {
            warnings.push("poly_gcd_exact: second argument is not a polynomial".to_string());
            return GcdExactResult {
                gcd: ctx.num(1),
                layer_used: GcdExactLayer::BudgetExceeded,
                warnings,
            };
        }
    };

    // Handle zero cases: gcd(0, p) = p, gcd(p, 0) = p
    if p_a.is_zero() {
        let normalized = normalize_multipoly(ctx, &p_b);
        return GcdExactResult {
            gcd: normalized,
            layer_used: GcdExactLayer::Trivial,
            warnings,
        };
    }
    if p_b.is_zero() {
        let normalized = normalize_multipoly(ctx, &p_a);
        return GcdExactResult {
            gcd: normalized,
            layer_used: GcdExactLayer::Trivial,
            warnings,
        };
    }

    // Handle constant cases: gcd(c1, c2) = 1 for non-zero constants over ℚ
    if p_a.is_constant() && p_b.is_constant() {
        return GcdExactResult {
            gcd: ctx.num(1),
            layer_used: GcdExactLayer::Trivial,
            warnings,
        };
    }

    // Budget checks
    if p_a.vars.len() > budget.max_vars || p_b.vars.len() > budget.max_vars {
        warnings.push(format!(
            "poly_gcd_exact: too many variables (max {})",
            budget.max_vars
        ));
        return GcdExactResult {
            gcd: ctx.num(1),
            layer_used: GcdExactLayer::BudgetExceeded,
            warnings,
        };
    }
    if p_a.num_terms() > budget.max_terms_input || p_b.num_terms() > budget.max_terms_input {
        warnings.push(format!(
            "poly_gcd_exact: too many terms (max {})",
            budget.max_terms_input
        ));
        return GcdExactResult {
            gcd: ctx.num(1),
            layer_used: GcdExactLayer::BudgetExceeded,
            warnings,
        };
    }

    // Align variables
    let (p_a, p_b) = align_variables(&p_a, &p_b);

    // Try univariate path using Polynomial::gcd
    if p_a.vars.len() == 1 {
        if let Some(gcd) = try_univariate_gcd(ctx, a, b, &p_a.vars[0]) {
            return GcdExactResult {
                gcd,
                layer_used: GcdExactLayer::Univariate,
                warnings,
            };
        }
    }

    // Try Layer 1: monomial + content GCD
    if let Some(gcd) = try_layer1_gcd(&p_a, &p_b) {
        if !gcd.is_constant() {
            let normalized = normalize_multipoly(ctx, &gcd);
            return GcdExactResult {
                gcd: normalized,
                layer_used: GcdExactLayer::Layer1MonomialContent,
                warnings,
            };
        }
    }

    // Try Layer 2: heuristic seeds
    let gcd_budget = GcdBudget::default();
    if let Some(gcd) = gcd_multivar_layer2(&p_a, &p_b, &gcd_budget) {
        let normalized = normalize_multipoly(ctx, &gcd);
        return GcdExactResult {
            gcd: normalized,
            layer_used: GcdExactLayer::Layer2HeuristicSeeds,
            warnings,
        };
    }

    // Try Layer 2.5: tensor grid
    let layer25_budget = Layer25Budget::default();
    if let Some(gcd) = gcd_multivar_layer25(&p_a, &p_b, &layer25_budget) {
        let normalized = normalize_multipoly(ctx, &gcd);
        return GcdExactResult {
            gcd: normalized,
            layer_used: GcdExactLayer::Layer25TensorGrid,
            warnings,
        };
    }

    // No GCD found - return 1
    GcdExactResult {
        gcd: ctx.num(1),
        layer_used: GcdExactLayer::Trivial,
        warnings,
    }
}

/// Normalize a MultiPoly: primitive part + positive leading coefficient
fn normalize_multipoly(ctx: &mut Context, poly: &MultiPoly) -> ExprId {
    if poly.is_zero() {
        return ctx.num(0);
    }

    // Get primitive part
    let (_, primitive) = poly.primitive_part();

    // Check leading coefficient sign and negate if needed
    let normalized = if let Some((coeff, _)) = primitive.terms.first() {
        if coeff.is_negative() {
            primitive.neg()
        } else {
            primitive
        }
    } else {
        primitive
    };

    multipoly_to_expr(&normalized, ctx)
}

/// Align variables between two polynomials (ensure same variable set)
fn align_variables(p: &MultiPoly, q: &MultiPoly) -> (MultiPoly, MultiPoly) {
    if p.vars == q.vars {
        return (p.clone(), q.clone());
    }

    // Merge variable sets
    let mut all_vars: Vec<String> = p.vars.clone();
    for v in &q.vars {
        if !all_vars.contains(v) {
            all_vars.push(v.clone());
        }
    }
    all_vars.sort();

    // Extend p to all_vars
    let p_extended = extend_to_vars(p, &all_vars);
    let q_extended = extend_to_vars(q, &all_vars);

    (p_extended, q_extended)
}

/// Extend a polynomial to a larger variable set
fn extend_to_vars(poly: &MultiPoly, new_vars: &[String]) -> MultiPoly {
    if poly.vars == new_vars {
        return poly.clone();
    }

    // Map old indices to new indices
    let mut index_map: Vec<usize> = Vec::with_capacity(poly.vars.len());
    for v in &poly.vars {
        match new_vars.iter().position(|x| x == v) {
            Some(idx) => index_map.push(idx),
            None => {
                // Invariant violation: `new_vars` should contain all original variables.
                // Fall back to returning the original polynomial unchanged.
                return poly.clone();
            }
        }
    }

    let new_terms: Vec<(BigRational, Vec<u32>)> = poly
        .terms
        .iter()
        .map(|(coeff, exps)| {
            let mut new_exps = vec![0u32; new_vars.len()];
            for (old_idx, &exp) in exps.iter().enumerate() {
                new_exps[index_map[old_idx]] = exp;
            }
            (coeff.clone(), new_exps)
        })
        .collect();

    MultiPoly {
        vars: new_vars.to_vec(),
        terms: new_terms,
    }
}

/// Try univariate Euclidean GCD using Polynomial struct
fn try_univariate_gcd(ctx: &mut Context, a: ExprId, b: ExprId, var: &str) -> Option<ExprId> {
    use crate::polynomial::Polynomial;

    let p_a = Polynomial::from_expr(ctx, a, var).ok()?;
    let p_b = Polynomial::from_expr(ctx, b, var).ok()?;

    let gcd = p_a.gcd(&p_b);

    // Check if trivial
    if gcd.degree() == 0 && gcd.leading_coeff().is_one() {
        return Some(ctx.num(1));
    }

    Some(gcd.to_expr(ctx))
}

/// Try Layer 1: monomial + content GCD
fn try_layer1_gcd(p: &MultiPoly, q: &MultiPoly) -> Option<MultiPoly> {
    // Compute monomial GCD (min exponents for each variable)
    if p.vars != q.vars || p.vars.is_empty() {
        return None;
    }

    let n_vars = p.vars.len();

    // Find min exponents across all terms
    let mut min_exps_p = vec![u32::MAX; n_vars];
    let mut min_exps_q = vec![u32::MAX; n_vars];

    for (_, exps) in &p.terms {
        for (i, &e) in exps.iter().enumerate() {
            min_exps_p[i] = min_exps_p[i].min(e);
        }
    }
    for (_, exps) in &q.terms {
        for (i, &e) in exps.iter().enumerate() {
            min_exps_q[i] = min_exps_q[i].min(e);
        }
    }

    // GCD of exponents = min of mins
    let mono_gcd: Vec<u32> = min_exps_p
        .iter()
        .zip(min_exps_q.iter())
        .map(|(&a, &b)| {
            let a = if a == u32::MAX { 0 } else { a };
            let b = if b == u32::MAX { 0 } else { b };
            a.min(b)
        })
        .collect();

    // Compute content GCD
    let content_p = p.content();
    let content_q = q.content();
    let content_gcd = gcd_rational(&content_p, &content_q);

    // Check if we have any non-trivial GCD
    let has_mono_gcd = mono_gcd.iter().any(|&e| e > 0);
    let has_content_gcd = !content_gcd.is_one();

    if !has_mono_gcd && !has_content_gcd {
        return None;
    }

    // Build GCD polynomial
    let gcd = MultiPoly {
        vars: p.vars.clone(),
        terms: vec![(content_gcd, mono_gcd)],
    };

    Some(gcd)
}

/// GCD of two rationals
fn gcd_rational(a: &BigRational, b: &BigRational) -> BigRational {
    use num_integer::Integer;

    if a.is_zero() {
        return b.abs();
    }
    if b.is_zero() {
        return a.abs();
    }

    // gcd(a/b, c/d) = gcd(a*d, c*b) / lcm(b, d)
    // Simplified: gcd of numerators / lcm of denominators
    let num_gcd = a.numer().gcd(b.numer());
    let den_lcm = a.denom().lcm(b.denom());

    BigRational::new(num_gcd, den_lcm)
}

// =============================================================================
// Rule definition
// =============================================================================

/// Rule for poly_gcd_exact(a, b) function.
/// Computes algebraic GCD of two polynomial expressions over ℚ.
pub struct PolyGcdExactRule;

impl Rule for PolyGcdExactRule {
    fn name(&self) -> &str {
        "Polynomial GCD Exact"
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM
    }

    fn priority(&self) -> i32 {
        200 // High priority to evaluate early
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let fn_expr = ctx.get(expr).clone();

        if let Expr::Function(fn_id, args) = fn_expr {
            let name = ctx.sym_name(fn_id);
            // Match poly_gcd_exact, pgcdx with 2 arguments
            let is_gcd_exact = name == "poly_gcd_exact" || name == "pgcdx";

            if is_gcd_exact && args.len() == 2 {
                let a = args[0];
                let b = args[1];

                let result = gcd_exact(ctx, a, b, &GcdExactBudget::default());

                return Some(Rewrite::simple(
                    result.gcd,
                    format!(
                        "poly_gcd_exact({}, {}) [{:?}]",
                        DisplayExpr {
                            context: ctx,
                            id: a
                        },
                        DisplayExpr {
                            context: ctx,
                            id: b
                        },
                        result.layer_used
                    ),
                ));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_gcd_exact_zero_a() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let b = parse("x + 1", &mut ctx).expect("parse");

        let result = gcd_exact(&mut ctx, zero, b, &GcdExactBudget::default());
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: result.gcd
            }
        );

        // gcd(0, x+1) = x+1
        assert!(
            result_str.contains("x"),
            "Expected x in result: {}",
            result_str
        );
    }

    #[test]
    fn test_gcd_exact_constants() {
        let mut ctx = Context::new();
        let a = ctx.num(6);
        let b = ctx.num(15);

        let result = gcd_exact(&mut ctx, a, b, &GcdExactBudget::default());

        // gcd(6, 15) = 1 over ℚ
        if let Expr::Number(n) = ctx.get(result.gcd) {
            assert!(n.is_one(), "Expected 1, got: {}", n);
        } else {
            panic!("Expected number");
        }
    }

    #[test]
    fn test_gcd_exact_univar() {
        let mut ctx = Context::new();
        // gcd(x^2 - 1, x - 1) = x - 1
        let a = parse("x^2 - 1", &mut ctx).expect("parse");
        let b = parse("x - 1", &mut ctx).expect("parse");

        let result = gcd_exact(&mut ctx, a, b, &GcdExactBudget::default());
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: result.gcd
            }
        );

        // Should contain x - 1 or -1 + x
        assert!(
            result_str.contains("x") && result_str.contains("1"),
            "Expected x-1, got: {} (layer: {:?})",
            result_str,
            result.layer_used
        );
    }
}
