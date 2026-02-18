//! Core fraction rules and helpers.
//!
//! This module contains the main simplification and cancellation rules,
//! along with helper functions for polynomial comparison and factor collection.

use crate::build::mul2_raw;
use crate::rules::algebra::helpers::gcd_rational;
use cas_ast::{Context, Expr, ExprId};
use cas_math::multipoly::{
    gcd_multivar_layer2, gcd_multivar_layer25, multipoly_from_expr, multipoly_to_expr, GcdBudget,
    GcdLayer, Layer25Budget, MultiPoly, PolyBudget,
};
use num_traits::{One, Zero};

// =============================================================================
// Context-aware helpers for AddFractionsRule gating
// =============================================================================

/// Check if a function name is trigonometric (sin, cos, tan and inverses/hyperbolics)
pub(super) fn is_trig_function_name(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "csc"
            | "sec"
            | "cot"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
    )
}

/// Check if a function (by SymbolId) is trigonometric
pub(super) fn is_trig_function(ctx: &Context, fn_id: usize) -> bool {
    ctx.builtin_of(fn_id)
        .is_some_and(|b| is_trig_function_name(b.name()))
}

/// Check if expression is a constant involving π (e.g., pi, pi/9, 2*pi/3)
pub(super) fn is_pi_constant(ctx: &Context, id: ExprId) -> bool {
    crate::helpers::extract_rational_pi_multiple(ctx, id).is_some()
}

// =============================================================================
// Polynomial equality helper (for canonical comparison ignoring AST order)
// =============================================================================

/// Compare two expressions as polynomials (ignoring AST structure/order).
/// Returns true if both convert to the same canonical polynomial form.
/// Falls back to false if either is not a polynomial (budget exceeded, non-poly, etc).
pub(super) fn poly_eq(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Use a tight budget for recognizer comparisons
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 10,
        max_pow_exp: 5,
    };

    let pa = match multipoly_from_expr(ctx, a, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let pb = match multipoly_from_expr(ctx, b, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };

    pa == pb
}

/// Relation between two polynomial expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SignRelation {
    Same,    // a == b
    Negated, // a == -b (e.g., x-y vs y-x)
}

/// Compare two expressions to detect if they are equal or negated.
/// Returns Some(Same) if a == b, Some(Negated) if a == -b, None otherwise.
pub(super) fn poly_relation(ctx: &Context, a: ExprId, b: ExprId) -> Option<SignRelation> {
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 10,
        max_pow_exp: 5,
    };

    let pa = multipoly_from_expr(ctx, a, &budget).ok()?;
    let pb = multipoly_from_expr(ctx, b, &budget).ok()?;

    if pa == pb {
        return Some(SignRelation::Same);
    }

    // Check if pa == -pb
    let neg_pb = pb.neg();
    if pa == neg_pb {
        return Some(SignRelation::Negated);
    }

    None
}

// =============================================================================
// A1: Structural Factor Cancellation (without polynomial expansion)
// =============================================================================

/// Collect multiplicative factors with integer exponents from an expression.
/// - Mul(...) is flattened
/// - Pow(base, k) with integer k becomes (base, k)
/// - Neg(x) is unwrapped and factors collected from x (for intersection purposes)
/// - Everything else becomes (expr, 1)
pub(super) fn collect_mul_factors_int_pow(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    let mut factors = Vec::new();
    // Unwrap top-level Neg for factor collection (enables intersection with positive terms)
    let actual_expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    collect_mul_factors_recursive(ctx, actual_expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_recursive(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            // Binary Mul: recursively collect both sides
            collect_mul_factors_recursive(ctx, *left, mult, factors);
            collect_mul_factors_recursive(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            // Check if exponent is an integer
            if let Some(k) = get_integer_exponent_a1(ctx, *exp) {
                factors.push((*base, mult * k));
            } else {
                factors.push((expr, mult));
            }
        }
        _ => {
            factors.push((expr, mult));
        }
    }
}

/// Extract integer from exponent expression (Number or Neg(Number))
fn get_integer_exponent_a1(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent_a1(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Build a product from factors with integer exponents.
///
/// Uses canonical `MulBuilder` (right-fold with exponents).
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
pub(super) fn build_mul_from_factors_a1(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    use cas_ast::views::MulBuilder;

    let mut builder = MulBuilder::new_simple();
    for &(base, exp) in factors {
        if exp > 0 {
            builder.push_pow(base, exp);
        }
        // Negative exponents shouldn't appear in numerator/denominator factors
    }
    builder.build(ctx)
}

// =============================================================================
// Multivariate GCD (Layers 1 + 2 + 2.5)
// =============================================================================

/// Align MultiPoly to a target variable set by embedding it into the larger space.
/// For example, a polynomial in [h] can be embedded into [h, x] by adding zero exponents for x.
fn align_multipoly_vars(p: &MultiPoly, target_vars: &[String]) -> MultiPoly {
    use std::collections::BTreeMap;

    if p.vars == target_vars {
        return p.clone();
    }

    // Map old var indices to new indices
    let mapping: Vec<Option<usize>> = p
        .vars
        .iter()
        .map(|v| target_vars.iter().position(|tv| tv == v))
        .collect();

    let mut new_terms: Vec<(num_rational::BigRational, Vec<u32>)> = Vec::new();
    let mut term_map: BTreeMap<Vec<u32>, num_rational::BigRational> = BTreeMap::new();

    for (coeff, mono) in &p.terms {
        let mut new_mono = vec![0u32; target_vars.len()];
        for (old_idx, &exp) in mono.iter().enumerate() {
            if let Some(Some(new_idx)) = mapping.get(old_idx) {
                new_mono[*new_idx] = exp;
            }
        }
        let entry = term_map
            .entry(new_mono)
            .or_insert_with(num_rational::BigRational::zero);
        *entry = entry.clone() + coeff.clone();
    }

    for (mono, coeff) in term_map {
        if !coeff.is_zero() {
            new_terms.push((coeff, mono));
        }
    }

    MultiPoly {
        vars: target_vars.to_vec(),
        terms: new_terms,
    }
}

/// Try to compute GCD of two expressions using multivariate polynomial representation.
/// Returns None if expressions can't be converted to polynomials or if GCD is trivial (1).
/// Returns Some((quotient_num, quotient_den, gcd_expr, layer)) if non-trivial GCD found.
pub(super) fn try_multivar_gcd(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ExprId, ExprId, ExprId, GcdLayer)> {
    let budget = PolyBudget::default();

    // Try to convert both to MultiPoly
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Skip if not multivariate (let univariate path handle it)
    if p_num.vars.len() <= 1 {
        return None;
    }

    // Align variables if needed: embed both into union of variables
    let (p_num, p_den) = if p_num.vars != p_den.vars {
        // Compute union of variables (sorted for consistency)
        let mut all_vars: Vec<String> = p_num
            .vars
            .iter()
            .chain(p_den.vars.iter())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        all_vars.sort();

        // Embed both polynomials into the shared variable space
        let p_num_aligned = align_multipoly_vars(&p_num, &all_vars);
        let p_den_aligned = align_multipoly_vars(&p_den, &all_vars);
        (p_num_aligned, p_den_aligned)
    } else {
        (p_num, p_den)
    };

    // Layer 1: Monomial GCD
    let mono_gcd = p_num.monomial_gcd_with(&p_den).ok()?;
    let has_mono_gcd = mono_gcd.iter().any(|&e| e > 0);

    // Layer 1: Content GCD
    let content_num = p_num.content();
    let content_den = p_den.content();
    let content_gcd = gcd_rational(content_num.clone(), content_den.clone());
    let has_content_gcd = !content_gcd.is_one();

    // If no GCD found at Layer 1, try Layer 2 and Layer 2.5
    if !has_mono_gcd && !has_content_gcd {
        if p_num.vars.len() >= 2 {
            // Try Layer 2 first (bivar with seeds)
            let gcd_budget = GcdBudget::default();
            if let Some(gcd_poly) = gcd_multivar_layer2(&p_num, &p_den, &gcd_budget) {
                if !gcd_poly.is_one() && !gcd_poly.is_constant() {
                    let q_num = p_num.div_exact(&gcd_poly)?;
                    let q_den = p_den.div_exact(&gcd_poly)?;
                    let new_num = multipoly_to_expr(&q_num, ctx);
                    let new_den = multipoly_to_expr(&q_den, ctx);
                    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);
                    return Some((new_num, new_den, gcd_expr, GcdLayer::Layer2HeuristicSeeds));
                }
            }

            // Try Layer 2.5 (tensor grid for multi-param factors)
            let layer25_budget = Layer25Budget::default();
            if let Some(gcd_poly) = gcd_multivar_layer25(&p_num, &p_den, &layer25_budget) {
                if !gcd_poly.is_one() && !gcd_poly.is_constant() {
                    let q_num = p_num.div_exact(&gcd_poly)?;
                    let q_den = p_den.div_exact(&gcd_poly)?;
                    let new_num = multipoly_to_expr(&q_num, ctx);
                    let new_den = multipoly_to_expr(&q_den, ctx);
                    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);
                    return Some((new_num, new_den, gcd_expr, GcdLayer::Layer25TensorGrid));
                }
            }
        }
        return None;
    }

    // Divide by monomial GCD
    let (p_num, p_den) = if has_mono_gcd {
        (
            p_num.div_monomial_exact(&mono_gcd)?,
            p_den.div_monomial_exact(&mono_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    // Divide by content GCD
    let (p_num, p_den) = if has_content_gcd {
        (
            p_num.div_scalar_exact(&content_gcd)?,
            p_den.div_scalar_exact(&content_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    // Build GCD expression (monomial * content)
    let mut gcd_poly = MultiPoly::one(p_num.vars.clone());

    if has_mono_gcd {
        gcd_poly = gcd_poly.mul_monomial(&mono_gcd).ok()?;
    }

    if has_content_gcd {
        gcd_poly = gcd_poly.mul_scalar(&content_gcd);
    }

    // Convert back to expressions
    let new_num = multipoly_to_expr(&p_num, ctx);
    let new_den = multipoly_to_expr(&p_den, ctx);
    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);

    Some((new_num, new_den, gcd_expr, GcdLayer::Layer1MonomialContent))
}

// ========== Helper to extract fraction parts from both Div and Mul(1/n,x) ==========
// This is needed because canonicalization may convert Div(x,n) to Mul(1/n,x)

/// Extract (numerator, denominator, is_fraction) from an expression.
/// Recognizes:
/// - Div(num, den) → (num, den, true)
/// - Mul(Number(1/n), x) or Mul(x, Number(1/n)) → (x, n, true) where numerator of coeff is ±1
/// - Mul(Div(1,den), x) or Mul(x, Div(1,den)) → (x, den, true) for symbolic denominators
/// - anything else → (expr, 1, false)
pub(super) fn extract_as_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId, bool) {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::Signed;

    // Case 1: Direct Div - use zero-clone destructuring
    if let Some((num, den)) = crate::helpers::as_div(ctx, expr) {
        return (num, den, true);
    }

    // Case 2 & 3: Mul with fractional coefficient
    if let Some((l, r)) = crate::helpers::as_mul(ctx, expr) {
        // Helper to check if a Number is ±1/n and extract denominator
        let check_unit_fraction = |n: &BigRational| -> Option<(BigInt, bool)> {
            if n.is_integer() {
                return None;
            }
            let numer = n.numer();
            let abs_numer: BigInt = if numer < &BigInt::from(0) {
                -numer.clone()
            } else {
                numer.clone()
            };
            if abs_numer == BigInt::from(1) {
                let is_negative = numer.is_negative();
                return Some((n.denom().clone(), is_negative));
            }
            None
        };

        // Helper to check if expression is Div(1, den) or Div(-1, den)
        let check_unit_div = |factor: ExprId| -> Option<(ExprId, bool)> {
            let (num, den) = crate::helpers::as_div(ctx, factor)?;
            let n = cas_math::numeric::as_number(ctx, num)?;
            if n.is_integer() {
                let n_val = n.numer();
                if *n_val == BigInt::from(1) {
                    return Some((den, false));
                } else if *n_val == BigInt::from(-1) {
                    return Some((den, true));
                }
            }
            None
        };

        // Case 2: Check for Number(±1/n)
        if let Some(n) = cas_math::numeric::as_number(ctx, l) {
            if let Some((denom, is_neg)) = check_unit_fraction(n) {
                let denom_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    let neg_r = ctx.add(Expr::Neg(r));
                    return (neg_r, denom_expr, true);
                }
                return (r, denom_expr, true);
            }
        }
        if let Some(n) = cas_math::numeric::as_number(ctx, r) {
            if let Some((denom, is_neg)) = check_unit_fraction(n) {
                let denom_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    let neg_l = ctx.add(Expr::Neg(l));
                    return (neg_l, denom_expr, true);
                }
                return (l, denom_expr, true);
            }
        }

        // Case 3: Check for Div(1, den) or Div(-1, den) - symbolic denominator
        if let Some((den, is_neg)) = check_unit_div(l) {
            if is_neg {
                let neg_r = ctx.add(Expr::Neg(r));
                return (neg_r, den, true);
            }
            return (r, den, true);
        }
        if let Some((den, is_neg)) = check_unit_div(r) {
            if is_neg {
                let neg_l = ctx.add(Expr::Neg(l));
                return (neg_l, den, true);
            }
            return (l, den, true);
        }
    }

    // Not a recognized fraction form
    let one = ctx.num(1);
    (expr, one, false)
}
/// Check if one denominator divides the other
/// Returns (new_n1, new_n2, common_den, is_divisible)
///
/// For example:
/// - d1=2, d2=2n → d2 = n·d1, so multiply n1 by n: (n1·n, n2, 2n, true)
/// - d1=2n, d2=2 → d1 = n·d2, so multiply n2 by n: (n1, n2·n, 2n, true)
pub(super) fn check_divisible_denominators(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
) -> (ExprId, ExprId, ExprId, bool) {
    // Try to find if d2 = k * d1 (d1 divides d2)
    if let Some(k) = try_extract_factor(ctx, d2, d1) {
        // d2 = k * d1, so use d2 as common denominator
        // n1/d1 = n1*k/d2
        let new_n1 = mul2_raw(ctx, n1, k);
        return (new_n1, n2, d2, true);
    }

    // Try to find if d1 = k * d2 (d2 divides d1)
    if let Some(k) = try_extract_factor(ctx, d1, d2) {
        // d1 = k * d2, so use d1 as common denominator
        // n2/d2 = n2*k/d1
        let new_n2 = mul2_raw(ctx, n2, k);
        return (n1, new_n2, d1, true);
    }

    // Not divisible
    (n1, n2, d1, false)
}

/// Returns Some(k) if expr = k * factor, None otherwise
fn try_extract_factor(ctx: &Context, expr: ExprId, factor: ExprId) -> Option<ExprId> {
    // Direct equality check (same ExprId)
    if expr == factor {
        return None; // k would be 1, not useful
    }

    // Check if expr is a Mul containing factor
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check l * r where one equals factor (using ExprId equality or structural)
        if *l == factor || exprs_equal(ctx, *l, factor) {
            return Some(*r); // expr = factor * r, so k = r
        }
        if *r == factor || exprs_equal(ctx, *r, factor) {
            return Some(*l); // expr = l * factor, so k = l
        }

        // For nested Mul, we'd need a more sophisticated approach
        // For now, only handle simple a*b case where one of them is the factor
    }

    None
}

/// Check if two expressions are structurally equal
fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    match (ctx.get(a), ctx.get(b)) {
        (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
        (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
        (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Div(l1, r1), Expr::Div(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Neg(e1), Expr::Neg(e2)) => exprs_equal(ctx, *e1, *e2),
        _ => false,
    }
}
