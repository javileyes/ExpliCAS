//! Polynomial GCD structural rule.
//!
//! Implements `poly_gcd(a, b)` which finds the structural GCD of two expressions
//! by collecting multiplicative factors and intersecting them.
//!
//! Example:
//! ```text
//! poly_gcd((1+x)^3 * (2+y), (1+x)^2 * (3+z)) = (1+x)^2
//! poly_gcd(a*g, b*g) = g
//! ```
//!
//! This allows Mathematica/Symbolica-style polynomial GCD without expanding.

use crate::build::mul2_raw;
use crate::phase::PhaseMask;
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_rational::BigRational;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// =============================================================================
// AC-Canonical Key for expression comparison
// =============================================================================

/// A hash-based key for AC (associative-commutative) comparison of expressions.
/// Two expressions with the same ExprKey are considered structurally equivalent
/// even if they have different parenthesization or term ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprKey(u64);

/// Compute AC-canonical key for an expression.
/// This flattens Add and Mul chains, sorts children by their keys, and produces
/// a stable hash that's independent of parenthesization and term order.
fn expr_key_ac(ctx: &Context, expr: ExprId) -> ExprKey {
    let mut hasher = DefaultHasher::new();
    expr_key_hash(ctx, expr, &mut hasher);
    ExprKey(hasher.finish())
}

/// Core hashing logic for AC-canonical key
fn expr_key_hash<H: Hasher>(ctx: &Context, expr: ExprId, hasher: &mut H) {
    match ctx.get(expr) {
        Expr::Add(_, _) => {
            // Flatten and sort by key
            let mut children = Vec::new();
            flatten_add(ctx, expr, &mut children);
            let mut keys: Vec<ExprKey> = children.iter().map(|&e| expr_key_ac(ctx, e)).collect();
            keys.sort();

            "Add".hash(hasher);
            keys.len().hash(hasher);
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Mul(_, _) => {
            // Flatten and sort by key
            let mut children = Vec::new();
            flatten_mul(ctx, expr, &mut children);
            let mut keys: Vec<ExprKey> = children.iter().map(|&e| expr_key_ac(ctx, e)).collect();
            keys.sort();

            "Mul".hash(hasher);
            keys.len().hash(hasher);
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Pow(base, exp) => {
            "Pow".hash(hasher);
            expr_key_ac(ctx, *base).hash(hasher);
            expr_key_ac(ctx, *exp).hash(hasher);
        }
        Expr::Neg(inner) => {
            "Neg".hash(hasher);
            expr_key_ac(ctx, *inner).hash(hasher);
        }
        Expr::Sub(a, b) => {
            // Treat as Add(a, Neg(b)) for AC equivalence
            "Add".hash(hasher);
            2usize.hash(hasher);
            let mut keys = vec![expr_key_ac(ctx, *a), expr_key_neg(ctx, *b)];
            keys.sort();
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Div(a, b) => {
            "Div".hash(hasher);
            expr_key_ac(ctx, *a).hash(hasher);
            expr_key_ac(ctx, *b).hash(hasher);
        }
        Expr::Number(n) => {
            "Number".hash(hasher);
            n.numer().hash(hasher);
            n.denom().hash(hasher);
        }
        Expr::Variable(name) => {
            "Variable".hash(hasher);
            name.hash(hasher);
        }
        Expr::Constant(c) => {
            "Constant".hash(hasher);
            format!("{:?}", c).hash(hasher);
        }
        Expr::Function(name, args) => {
            "Function".hash(hasher);
            name.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                expr_key_ac(ctx, *arg).hash(hasher);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            "Matrix".hash(hasher);
            rows.hash(hasher);
            cols.hash(hasher);
            for d in data {
                expr_key_ac(ctx, *d).hash(hasher);
            }
        }
        Expr::SessionRef(id) => {
            "SessionRef".hash(hasher);
            id.hash(hasher);
        }
    }
}

/// Hash for Neg(expr) - used when treating Sub as Add(a, Neg(b))
fn expr_key_neg(ctx: &Context, expr: ExprId) -> ExprKey {
    let mut hasher = DefaultHasher::new();
    "Neg".hash(&mut hasher);
    expr_key_ac(ctx, expr).hash(&mut hasher);
    ExprKey(hasher.finish())
}

/// Flatten Add chain (associative)
fn flatten_add(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            flatten_add(ctx, *left, out);
            flatten_add(ctx, *right, out);
        }
        _ => {
            out.push(expr);
        }
    }
}

/// Flatten Mul chain (associative)
fn flatten_mul(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            flatten_mul(ctx, *left, out);
            flatten_mul(ctx, *right, out);
        }
        _ => {
            out.push(expr);
        }
    }
}

/// Check if two expressions are AC-equivalent
fn expr_equal_ac(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    expr_key_ac(ctx, a) == expr_key_ac(ctx, b)
}

// =============================================================================
// __hold transparency helper
// =============================================================================

/// Strip __hold() wrapper(s) from an expression. __hold is an internal barrier
/// that should be transparent for structural operations like poly_gcd.
fn strip_hold(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        match ctx.get(expr) {
            Expr::Function(name, args) if name == "__hold" && args.len() == 1 => {
                expr = args[0];
            }
            _ => return expr,
        }
    }
}

/// Collect multiplicative factors with integer exponents from an expression.
/// - Mul(...) is flattened
/// - Pow(base, k) with integer k becomes (base, k)
/// - Everything else becomes (expr, 1)
/// - __hold wrappers are stripped transparently
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    // Strip __hold wrapper first
    let expr = strip_hold(ctx, expr);
    let mut factors = Vec::new();
    collect_mul_factors_rec(ctx, expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_rec(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_rec(ctx, *left, mult, factors);
            collect_mul_factors_rec(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exp(ctx, *exp) {
                if k > 0 {
                    factors.push((*base, mult * k));
                } else {
                    // Negative exponents: treat whole as factor
                    factors.push((expr, mult));
                }
            } else {
                factors.push((expr, mult));
            }
        }
        _ => {
            factors.push((expr, mult));
        }
    }
}

/// Extract integer from exponent expression
fn get_integer_exp(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exp(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

// Keep for backwards compatibility but now unused
#[allow(dead_code)]
fn compare_expr_for_sort(ctx: &Context, a: ExprId, b: ExprId) -> Ordering {
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

/// Build a product from factors with positive exponents.
fn build_mul_from_factors(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    if factors.is_empty() {
        return ctx.add(Expr::Number(BigRational::from_integer(1.into())));
    }

    let mut terms: Vec<ExprId> = Vec::new();

    for &(base, exp) in factors {
        if exp == 0 {
            continue;
        } else if exp == 1 {
            terms.push(base);
        } else if exp > 1 {
            let exp_expr = ctx.add(Expr::Number(BigRational::from_integer(exp.into())));
            terms.push(ctx.add(Expr::Pow(base, exp_expr)));
        }
        // Negative exponents shouldn't appear in GCD factors
    }

    if terms.is_empty() {
        ctx.add(Expr::Number(BigRational::from_integer(1.into())))
    } else if terms.len() == 1 {
        terms[0]
    } else {
        let mut iter = terms.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, t| mul2_raw(ctx, acc, t))
    }
}

// =============================================================================
// Structural GCD computation
// =============================================================================

/// Compute structural GCD by intersecting factor lists.
/// Returns the GCD expression (or 1 if no common factors).
///
/// Uses AC-canonical key for factor matching - handles different parenthesization
/// and term ordering by flattening Add/Mul chains and sorting by canonical hash.
fn poly_gcd_structural(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    let a_factors = collect_mul_factors(ctx, a);
    let b_factors = collect_mul_factors(ctx, b);

    // Find common factors by AC-canonical key comparison
    let mut gcd_factors: Vec<(ExprId, i64)> = Vec::new();
    let mut used_b: Vec<bool> = vec![false; b_factors.len()];

    for (a_base, a_exp) in &a_factors {
        // Find matching factor in b_factors (by AC-canonical equality)
        for (j, (b_base, b_exp)) in b_factors.iter().enumerate() {
            if !used_b[j] && expr_equal_ac(ctx, *a_base, *b_base) {
                // Common factor found: take min exponent
                let min_exp = (*a_exp).min(*b_exp);
                if min_exp > 0 {
                    gcd_factors.push((*a_base, min_exp));
                }
                used_b[j] = true;
                break;
            }
        }
    }

    build_mul_from_factors(ctx, &gcd_factors)
}

// =============================================================================
// REPL function rule
// =============================================================================

/// Rule for poly_gcd(a, b) function.
/// Computes structural GCD of two polynomial expressions.
pub struct PolyGcdRule;

impl Rule for PolyGcdRule {
    fn name(&self) -> &str {
        "Polynomial GCD"
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

        if let Expr::Function(name, args) = fn_expr {
            // Match poly_gcd, pgcd with 2 arguments
            let is_poly_gcd = name == "poly_gcd" || name == "pgcd";

            if is_poly_gcd && args.len() == 2 {
                let a = args[0];
                let b = args[1];

                let gcd = poly_gcd_structural(ctx, a, b);

                // Wrap result in __hold() to prevent further simplification (invisible barrier)
                let held_gcd = ctx.add(Expr::Function("__hold".to_string(), vec![gcd]));

                return Some(Rewrite::simple(
                    held_gcd,
                    format!(
                        "poly_gcd({}, {})",
                        DisplayExpr {
                            context: ctx,
                            id: a
                        },
                        DisplayExpr {
                            context: ctx,
                            id: b
                        }
                    ),
                ));
            }
        }

        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn setup_ctx() -> Context {
        Context::new()
    }

    #[test]
    fn test_poly_gcd_simple_common_factor() {
        let mut ctx = setup_ctx();

        // x+1
        let x = ctx.add(Expr::Variable("x".into()));
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));

        // y+2
        let y = ctx.add(Expr::Variable("y".into()));
        let two = ctx.num(2);
        let y_plus_2 = ctx.add(Expr::Add(y, two));

        // (x+1) * (y+2)
        let a = ctx.add(Expr::Mul(x_plus_1, y_plus_2));

        // z+3
        let z = ctx.add(Expr::Variable("z".into()));
        let three = ctx.num(3);
        let z_plus_3 = ctx.add(Expr::Add(z, three));

        // (x+1) * (z+3)
        let b = ctx.add(Expr::Mul(x_plus_1, z_plus_3));

        // GCD should be (x+1)
        let gcd = poly_gcd_structural(&mut ctx, a, b);

        // Verify it's x+1
        assert_eq!(gcd, x_plus_1);
    }

    #[test]
    fn test_poly_gcd_with_powers() {
        let mut ctx = setup_ctx();

        // x+1
        let x = ctx.add(Expr::Variable("x".into()));
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));

        // (x+1)^3
        let three = ctx.num(3);
        let pow3 = ctx.add(Expr::Pow(x_plus_1, three));

        // (x+1)^2
        let two = ctx.num(2);
        let pow2 = ctx.add(Expr::Pow(x_plus_1, two));

        // GCD((x+1)^3, (x+1)^2) = (x+1)^2
        let gcd = poly_gcd_structural(&mut ctx, pow3, pow2);

        // Should be (x+1)^2
        let gcd_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: gcd
            }
        );
        assert!(gcd_str.contains("x") && gcd_str.contains("1"));
    }

    #[test]
    fn test_poly_gcd_no_common() {
        let mut ctx = setup_ctx();

        // x
        let x = ctx.add(Expr::Variable("x".into()));
        // y
        let y = ctx.add(Expr::Variable("y".into()));

        // GCD(x, y) = 1 (no structural common factor)
        let gcd = poly_gcd_structural(&mut ctx, x, y);

        // Should be 1
        if let Expr::Number(n) = ctx.get(gcd) {
            assert_eq!(*n, BigRational::from_integer(BigInt::from(1)));
        } else {
            panic!("Expected number 1");
        }
    }
}
