//! Helper functions for constant folding.
//!
//! # Allowlist Functions
//!
//! These functions implement the allowlist-only constant folding:
//! - `is_constant_literal`: check if expression is constant
//! - `fold_neg`: fold negation of literal
//! - `fold_sqrt`: fold sqrt(literal) respecting ValueDomain
//! - `fold_mul_imaginary`: fold i*i→-1 pattern

use crate::semantics::ValueDomain;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Get children of an expression for traversal.
pub fn get_children(ctx: &Context, id: ExprId) -> Vec<ExprId> {
    match ctx.get(id) {
        // Binary operations
        Expr::Add(a, b) => vec![*a, *b],
        Expr::Sub(a, b) => vec![*a, *b],
        Expr::Mul(a, b) => vec![*a, *b],
        Expr::Div(num, den) => vec![*num, *den],
        Expr::Pow(base, exp) => vec![*base, *exp],
        Expr::Neg(inner) => vec![*inner],
        Expr::Function(_, args) => args.clone(),
        Expr::Matrix { data, .. } => data.clone(),
        // Leaves
        _ => vec![],
    }
}

/// Check if an expression is a constant literal (no variables).
///
/// Returns true only for:
/// - Numeric literals (integers, rationals)
/// - Constants (π, e, i, ∞, -∞, undefined)
pub fn is_constant_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) => true,
        Expr::Constant(c) => matches!(
            c,
            cas_ast::Constant::Pi
                | cas_ast::Constant::E
                | cas_ast::Constant::I
                | cas_ast::Constant::Infinity
                | cas_ast::Constant::Undefined
        ),
        _ => false,
    }
}

/// Check if an expression is the imaginary unit i.
pub fn is_imaginary_unit(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Constant(cas_ast::Constant::I))
}

/// Fold negation of a literal.
pub fn fold_neg(ctx: &mut Context, inner: ExprId) -> Option<ExprId> {
    match ctx.get(inner) {
        Expr::Number(n) => Some(ctx.add(Expr::Number(-n.clone()))),
        _ => None,
    }
}

/// Fold sqrt(literal) respecting ValueDomain.
///
/// # Behavior
///
/// - Positive literal with perfect square: return exact integer
/// - Negative literal in RealOnly: return undefined
/// - Negative literal in ComplexEnabled: return i*sqrt(-n)
/// - Otherwise: return None (leave residual)
pub fn fold_sqrt(ctx: &mut Context, base: ExprId, value_domain: ValueDomain) -> Option<ExprId> {
    // Only fold numeric literals
    let n = match ctx.get(base) {
        Expr::Number(n) => n.clone(),
        _ => return None,
    };

    // Check sign
    if n.is_negative() {
        match value_domain {
            ValueDomain::RealOnly => {
                // sqrt(-n) is undefined in reals
                Some(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)))
            }
            ValueDomain::ComplexEnabled => {
                // sqrt(-n) = i * sqrt(n) for n > 0
                let pos_n = -n;
                let pos_n_expr = ctx.add(Expr::Number(pos_n));
                let sqrt_pos = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![pos_n_expr]);
                let i = ctx.add(Expr::Constant(cas_ast::Constant::I));
                Some(ctx.add(Expr::Mul(i, sqrt_pos)))
            }
        }
    } else if n.is_zero() {
        Some(ctx.num(0))
    } else {
        // Positive: try exact integer square root
        try_exact_sqrt(ctx, &n)
    }
}

/// Try to compute exact integer square root if perfect square.
fn try_exact_sqrt(ctx: &mut Context, n: &BigRational) -> Option<ExprId> {
    // Only for integers
    if !n.is_integer() {
        return None;
    }

    let num = n.numer();
    let sqrt_num = num.sqrt();

    // Check if perfect square
    if &(&sqrt_num * &sqrt_num) == num {
        Some(ctx.add(Expr::Number(BigRational::from_integer(sqrt_num))))
    } else {
        None
    }
}

/// Fold multiplication patterns involving imaginary unit.
///
/// Currently handles binary Mul only:
/// - i * i → -1
/// - (-i) * i → 1
/// - i * (-i) → 1
pub fn fold_mul_imaginary(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    value_domain: ValueDomain,
) -> Option<ExprId> {
    // Only in complex mode
    if value_domain != ValueDomain::ComplexEnabled {
        return None;
    }

    // i * i → -1
    if is_imaginary_unit(ctx, a) && is_imaginary_unit(ctx, b) {
        return Some(ctx.num(-1));
    }

    // Check for Neg(i) patterns
    let a_is_neg_i = is_neg_of_i(ctx, a);
    let b_is_neg_i = is_neg_of_i(ctx, b);

    // (-i) * i → 1 or i * (-i) → 1
    if (a_is_neg_i && is_imaginary_unit(ctx, b)) || (is_imaginary_unit(ctx, a) && b_is_neg_i) {
        return Some(ctx.num(1));
    }

    // (-i) * (-i) → -1
    if a_is_neg_i && b_is_neg_i {
        return Some(ctx.num(-1));
    }

    None
}

/// Check if expression is Neg(i).
fn is_neg_of_i(ctx: &Context, id: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(id) {
        is_imaginary_unit(ctx, *inner)
    } else {
        false
    }
}

/// Extract rational from a Number literal.
/// Also handles Neg(Number) for cases like `-1/2`.
pub fn literal_rat(ctx: &Context, id: ExprId) -> Option<BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => {
            // Handle Neg(n) as -n
            if let Expr::Number(n) = ctx.get(*inner) {
                return Some(-n.clone());
            }
            None
        }
        _ => None,
    }
}

/// Fold power of literal constants.
///
/// # Allowlist (V1):
/// - integer ^ integer (n≥0): compute a^n
/// - 0^0 → undefined
/// - a^0 (a≠0) → 1
/// - 0^n (n>0) → 0
/// - (-1)^(1/2): RealOnly→undefined, ComplexEnabled+Principal→i
///
/// Returns None if:
/// - Either operand is not a constant literal
/// - Exponent is negative (PR2.2)
/// - Exponent is non-trivial rational (PR2.2+)
pub fn fold_pow(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    value_domain: ValueDomain,
    _branch: crate::semantics::BranchPolicy, // Wired for future use
) -> Option<ExprId> {
    // First: delegate integer exponent case to canonical const_eval helper
    // This covers: b^n for ℚ base and ℤ exponent, including edge cases (0^0, 0^-n)
    if let Some(result) = cas_math::const_eval::try_eval_pow_literal(ctx, base, exp) {
        return Some(result);
    }

    // Case: semantic evaluation (depends on ValueDomain)
    // Handle sqrt(-1) which const_eval doesn't cover
    let base_rat = literal_rat(ctx, base)?;
    let exp_rat = literal_rat(ctx, exp)?;

    // Normalize exponent: ensure denominator is positive
    let exp_rat = if exp_rat.denom().is_negative() {
        BigRational::new(-exp_rat.numer().clone(), -exp_rat.denom().clone())
    } else {
        exp_rat
    };

    let exp_numer = exp_rat.numer();
    let exp_denom = exp_rat.denom();

    // Special case: (-1)^(1/2) - depends on ValueDomain
    if exp_numer == &num_bigint::BigInt::from(1) && exp_denom == &num_bigint::BigInt::from(2) {
        // Check if base is exactly -1
        if base_rat == BigRational::from_integer((-1).into()) {
            match value_domain {
                ValueDomain::RealOnly => {
                    // sqrt(-1) undefined in reals
                    return Some(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)));
                }
                ValueDomain::ComplexEnabled => {
                    // sqrt(-1) = i in complex (principal branch)
                    return Some(ctx.add(Expr::Constant(cas_ast::Constant::I)));
                }
            }
        }
    }

    // All other cases: leave residual (not in allowlist)
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_constant_literal() {
        let mut ctx = Context::new();

        let num = ctx.num(42);
        assert!(is_constant_literal(&ctx, num));

        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
        assert!(is_constant_literal(&ctx, pi));

        let x = ctx.var("x");
        assert!(!is_constant_literal(&ctx, x));
    }

    #[test]
    fn test_fold_sqrt_positive_perfect() {
        let mut ctx = Context::new();
        let four = ctx.num(4);

        let result = fold_sqrt(&mut ctx, four, ValueDomain::RealOnly);
        assert!(result.is_some());

        // Should be 2
        if let Some(r) = result {
            assert!(
                matches!(ctx.get(r), Expr::Number(n) if n == &BigRational::from_integer(2.into()))
            );
        }
    }

    #[test]
    fn test_fold_sqrt_negative_real() {
        let mut ctx = Context::new();
        let neg_one = ctx.num(-1);

        let result = fold_sqrt(&mut ctx, neg_one, ValueDomain::RealOnly);
        assert!(result.is_some());

        // Should be undefined
        if let Some(r) = result {
            assert!(matches!(
                ctx.get(r),
                Expr::Constant(cas_ast::Constant::Undefined)
            ));
        }
    }

    #[test]
    fn test_fold_sqrt_negative_complex() {
        let mut ctx = Context::new();
        let neg_one = ctx.num(-1);

        let result = fold_sqrt(&mut ctx, neg_one, ValueDomain::ComplexEnabled);
        assert!(result.is_some());

        // Should be i * sqrt(1) = Mul(i, sqrt(1))
        if let Some(r) = result {
            assert!(matches!(ctx.get(r), Expr::Mul(_, _)));
        }
    }

    #[test]
    fn test_fold_mul_i_i() {
        let mut ctx = Context::new();
        let i1 = ctx.add(Expr::Constant(cas_ast::Constant::I));
        let i2 = ctx.add(Expr::Constant(cas_ast::Constant::I));

        let result = fold_mul_imaginary(&mut ctx, i1, i2, ValueDomain::ComplexEnabled);
        assert!(result.is_some());

        // Should be -1
        if let Some(r) = result {
            assert!(
                matches!(ctx.get(r), Expr::Number(n) if n == &BigRational::from_integer((-1).into()))
            );
        }
    }
}
