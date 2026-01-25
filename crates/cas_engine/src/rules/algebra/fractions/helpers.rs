//! Shared helper functions for the fractions module.
//!
//! These functions are used across multiple submodules (core, cancel, rationalize, more_rules)
//! and are extracted here to enable proper module separation.
//!
//! **Phase 1 Note:** This module is additive - the original files still contain
//! their own copies. Future phases will remove duplicates.

use crate::multipoly::{multipoly_from_expr, PolyBudget};
use cas_ast::views::MulBuilder;
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::cmp::Ordering;

// =============================================================================
// Trig and Pi helpers
// =============================================================================

/// Check if a function name is trigonometric (sin, cos, tan and inverses/hyperbolics)
#[allow(dead_code)]
pub fn is_trig_function_name(name: &str) -> bool {
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

/// Check if expression is a constant involving π (e.g., pi, pi/9, 2*pi/3)
#[allow(dead_code)]
pub fn is_pi_constant(ctx: &Context, id: ExprId) -> bool {
    crate::helpers::extract_rational_pi_multiple(ctx, id).is_some()
}

// =============================================================================
// Polynomial equality helpers
// =============================================================================

/// Relation between two polynomial expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SignRelation {
    Same,    // a == b
    Negated, // a == -b (e.g., x-y vs y-x)
}

/// Compare two expressions as polynomials (ignoring AST structure/order).
/// Returns true if both convert to the same canonical polynomial form.
#[allow(dead_code)]
pub fn poly_eq(ctx: &Context, a: ExprId, b: ExprId) -> bool {
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

/// Compare two expressions to detect if they are equal or negated.
/// Returns Some(Same) if a == b, Some(Negated) if a == -b, None otherwise.
#[allow(dead_code)]
pub fn poly_relation(ctx: &Context, a: ExprId, b: ExprId) -> Option<SignRelation> {
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

    let neg_pb = pb.neg();
    if pa == neg_pb {
        return Some(SignRelation::Negated);
    }

    None
}

// =============================================================================
// Factor collection helpers (with integer exponents)
// =============================================================================

/// Collect multiplicative factors with integer exponents from an expression.
/// For `a * b^2 * c`, returns [(a, 1), (b, 2), (c, 1)].
#[allow(dead_code)]
pub fn collect_mul_factors_int_pow(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    let mut factors = Vec::new();
    // Unwrap top-level Neg for factor collection
    let actual_expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    collect_mul_factors_int_pow_recursive(ctx, actual_expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_int_pow_recursive(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_int_pow_recursive(ctx, *left, mult, factors);
            collect_mul_factors_int_pow_recursive(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exponent(ctx, *exp) {
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
#[allow(dead_code)]
pub fn get_integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Compare expressions for sorting (by string representation)
#[allow(dead_code)]
pub fn compare_expr_for_sort(ctx: &Context, a: ExprId, b: ExprId) -> Ordering {
    let a_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: a
        }
    );
    let b_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: b
        }
    );
    a_str.cmp(&b_str)
}

/// Build a product from factors with integer exponents.
#[allow(dead_code)]
pub fn build_mul_from_factors(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    let mut builder = MulBuilder::new_simple();
    for &(base, exp) in factors {
        if exp > 0 {
            builder.push_pow(base, exp);
        }
    }
    builder.build(ctx)
}

// =============================================================================
// Factor collection helpers (flat, no exponents)
// =============================================================================

/// Collect all multiplicative factors from an expression (flat).
/// For `a * b * c`, returns [a, b, c].
#[allow(dead_code)]
pub fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    collect_mul_factors_flat_recursive(ctx, expr, &mut factors);
    factors
}

fn collect_mul_factors_flat_recursive(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_mul_factors_flat_recursive(ctx, *l, factors);
            collect_mul_factors_flat_recursive(ctx, *r, factors);
        }
        _ => {
            factors.push(expr);
        }
    }
}

// =============================================================================
// Additive term collection helpers
// =============================================================================

/// Collect all additive terms from an expression.
/// For `a + b + c`, returns [a, b, c].
#[allow(dead_code)]
pub fn collect_additive_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_additive_terms_recursive(ctx, expr, &mut terms);
    terms
}

fn collect_additive_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_additive_terms_recursive(ctx, *l, terms);
            collect_additive_terms_recursive(ctx, *r, terms);
        }
        _ => {
            terms.push(expr);
        }
    }
}

/// Build a sum from a list of terms.
#[allow(dead_code)]
pub fn build_sum(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut result = terms[0];
    for &term in terms.iter().skip(1) {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

// =============================================================================
// Irrational/root detection helpers
// =============================================================================

/// Check if an expression contains an irrational (root).
#[allow(dead_code)]
pub fn contains_irrational(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                !n.is_integer()
            } else {
                false
            }
        }
        Expr::Function(name, _) => ctx.sym_name(*name) == "sqrt",
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_irrational(ctx, *l) || contains_irrational(ctx, *r)
        }
        Expr::Neg(e) => contains_irrational(ctx, *e),
        _ => false,
    }
}

/// Extract root from expression: sqrt(n) or n^(1/k)
/// Returns (radicand, index) where expr = radicand^(1/index)
#[allow(dead_code)]
pub fn extract_root_base(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    // Check for sqrt(n) function
    if let Some(arg) = crate::helpers::as_fn1(ctx, expr, "sqrt") {
        let two = ctx.num(2);
        return Some((arg, two));
    }

    // Check for Pow(base, exp)
    if let Some((base, exp)) = crate::helpers::as_pow(ctx, expr) {
        if let Expr::Number(n) = ctx.get(exp) {
            if !n.is_integer() && n.numer() == &BigInt::from(1) {
                let index_val = n.denom().clone();
                let index = ctx.add(Expr::Number(BigRational::from_integer(index_val)));
                return Some((base, index));
            }
        }
        // Check for Pow(base, Div(1, index))
        if let Some((num, index)) = crate::helpers::as_div(ctx, exp) {
            if let Expr::Number(n) = ctx.get(num) {
                if n == &BigRational::one() {
                    return Some((base, index));
                }
            }
        }
    }

    None
}

// =============================================================================
// Fraction extraction helper
// =============================================================================

/// Extract (numerator, denominator, is_fraction) from an expression.
/// Recognizes:
/// - Div(num, den) → (num, den, true)
/// - Mul(Number(1/n), x) or Mul(x, Number(1/n)) → (x, n, true)
/// - Mul(Div(1,den), x) → (x, den, true)
/// - anything else → (expr, 1, false)
#[allow(dead_code)]
pub fn extract_as_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId, bool) {
    // Case 1: Direct Div
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
            let n = crate::helpers::as_number(ctx, num)?;
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

        // Check left for Number(1/n)
        if let Expr::Number(n) = ctx.get(l) {
            if let Some((denom_val, is_neg)) = check_unit_fraction(n) {
                let denom = ctx.add(Expr::Number(BigRational::from_integer(denom_val)));
                let result_num = if is_neg { ctx.add(Expr::Neg(r)) } else { r };
                return (result_num, denom, true);
            }
        }

        // Check right for Number(1/n)
        if let Expr::Number(n) = ctx.get(r) {
            if let Some((denom_val, is_neg)) = check_unit_fraction(n) {
                let denom = ctx.add(Expr::Number(BigRational::from_integer(denom_val)));
                let result_num = if is_neg { ctx.add(Expr::Neg(l)) } else { l };
                return (result_num, denom, true);
            }
        }

        // Check left for Div(1, den)
        if let Some((den, is_neg)) = check_unit_div(l) {
            let result_num = if is_neg { ctx.add(Expr::Neg(r)) } else { r };
            return (result_num, den, true);
        }

        // Check right for Div(1, den)
        if let Some((den, is_neg)) = check_unit_div(r) {
            let result_num = if is_neg { ctx.add(Expr::Neg(l)) } else { l };
            return (result_num, den, true);
        }
    }

    // Default: not a fraction
    let one = ctx.num(1);
    (expr, one, false)
}

// =============================================================================
// Divisibility check helper
// =============================================================================

/// Check if denominators are divisible (d1 | d2 or d2 | d1)
/// Returns Some((larger_den, d1_divides_d2)) if divisible, None otherwise
#[allow(dead_code)]
pub fn check_divisible_denominators(
    ctx: &Context,
    d1: ExprId,
    d2: ExprId,
) -> Option<(ExprId, bool)> {
    // Only handle numeric denominators for now
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(d1), ctx.get(d2)) {
        if n1.is_integer() && n2.is_integer() {
            let v1 = n1.to_integer();
            let v2 = n2.to_integer();
            if &v2 % &v1 == BigInt::from(0) {
                // d1 | d2, so d2 is the LCD
                return Some((d2, true));
            }
            if &v1 % &v2 == BigInt::from(0) {
                // d2 | d1, so d1 is the LCD
                return Some((d1, false));
            }
        }
    }
    None
}
