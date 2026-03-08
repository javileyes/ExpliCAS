//! Canonical constant evaluation for domain-neutral literal operations.
//!
//! This module provides the **single source of truth** for evaluating
//! literal rational expressions. Both the simplifier (`EvaluatePowerRule`)
//! and `const_fold` delegate to these helpers to avoid duplication.
//!
//! # Contract
//!
//! - **Domain-neutral**: No ValueDomain/BranchPolicy checks
//! - **Literals only**: Both operands must be literal rationals/integers
//! - **Safe edge cases**: 0^0, 0^(-n) → undefined
//! - **Bounded**: MAX_ABS_POW prevents explosion

use cas_ast::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

/// Maximum absolute value for exponents to prevent materialization explosion.
pub const MAX_ABS_POW: i64 = 1000;

/// Try to evaluate a literal power expression: base^exp where both are literals.
///
/// # Returns
/// - `Some(result)` if evaluation succeeds (including undefined for edge cases)
/// - `None` if not applicable (non-literal, symbolic, or exp too large)
///
/// # Edge Cases
/// - `0^0` → undefined
/// - `0^(-n)` → undefined (division by zero)
/// - `a^0` → 1 (when a ≠ 0)
/// - `1^n` → 1 (for any n)
pub fn try_eval_pow_literal(ctx: &mut Context, base: ExprId, exp: ExprId) -> Option<ExprId> {
    let base_q = literal_rational(ctx, base)?;
    let exp_i = literal_integer_i64(ctx, exp)?;

    // Prevent absurd materialization
    if exp_i.unsigned_abs() > MAX_ABS_POW as u64 {
        return None;
    }

    // Edge case: a^0
    if exp_i == 0 {
        if base_q.is_zero() {
            // 0^0 is undefined
            return Some(mk_undefined(ctx));
        }
        // a^0 = 1 for a ≠ 0
        return Some(mk_one(ctx));
    }

    // Edge case: 0^(-n) is undefined (division by zero)
    if base_q.is_zero() && exp_i < 0 {
        return Some(mk_undefined(ctx));
    }

    // Compute base^|exp|
    let abs_e = exp_i.unsigned_abs() as u32;
    let mut result = pow_rational_exact(&base_q, abs_e);

    // Invert if negative exponent
    if exp_i < 0 {
        // base ≠ 0 already guaranteed by earlier check
        result = BigRational::one() / result;
    }

    Some(mk_rational(ctx, result))
}

// =============================================================================
// Literal extraction helpers
// =============================================================================

/// Extract a literal rational from an expression.
/// Handles Number and Neg(Number).
fn literal_rational(ctx: &Context, id: ExprId) -> Option<BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => literal_rational(ctx, *inner).map(|v| -v),
        _ => None,
    }
}

/// Extract a literal integer from an expression (must be exact integer).
/// Handles Number and Neg(Number).
fn literal_integer_i64(ctx: &Context, id: ExprId) -> Option<i64> {
    match ctx.get(id) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().to_i64()
            } else {
                None
            }
        }
        Expr::Neg(inner) => literal_integer_i64(ctx, *inner).map(|v| -v),
        _ => None,
    }
}

// =============================================================================
// Computation helpers
// =============================================================================

/// Compute base^exp for BigRational using square-and-multiply.
fn pow_rational_exact(base: &BigRational, exp: u32) -> BigRational {
    if exp == 0 {
        return BigRational::one();
    }
    if exp == 1 {
        return base.clone();
    }

    let mut result = BigRational::one();
    let mut b = base.clone();
    let mut e = exp;

    while e > 0 {
        if (e & 1) == 1 {
            result *= &b;
        }
        e >>= 1;
        if e > 0 {
            b = &b * &b;
        }
    }
    result
}

// =============================================================================
// Expression constructors
// =============================================================================

fn mk_one(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Number(BigRational::one()))
}

fn mk_rational(ctx: &mut Context, q: BigRational) -> ExprId {
    ctx.add(Expr::Number(q))
}

fn mk_undefined(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Constant(Constant::Undefined))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Context {
        Context::new()
    }

    fn display(ctx: &Context, id: ExprId) -> String {
        use cas_formatter::DisplayExpr;
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn test_pow_positive_integer() {
        let mut ctx = setup();
        let base = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
        let exp = ctx.add(Expr::Number(BigRational::from_integer(10.into())));

        let result = try_eval_pow_literal(&mut ctx, base, exp).unwrap();
        assert_eq!(display(&ctx, result), "1024");
    }

    #[test]
    fn test_pow_negative_integer() {
        let mut ctx = setup();
        let base = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
        let neg_three = ctx.add(Expr::Number(BigRational::from_integer((-3).into())));

        let result = try_eval_pow_literal(&mut ctx, base, neg_three).unwrap();
        assert_eq!(display(&ctx, result), "1/8");
    }

    #[test]
    fn test_pow_rational_base() {
        let mut ctx = setup();
        // (3/4)^(-2) = 16/9
        let base = ctx.add(Expr::Number(BigRational::new(3.into(), 4.into())));
        let exp = ctx.add(Expr::Number(BigRational::from_integer((-2).into())));

        let result = try_eval_pow_literal(&mut ctx, base, exp).unwrap();
        assert_eq!(display(&ctx, result), "16/9");
    }

    #[test]
    fn test_zero_to_zero_undefined() {
        let mut ctx = setup();
        let zero = ctx.add(Expr::Number(BigRational::zero()));
        let zero_exp = ctx.add(Expr::Number(BigRational::zero()));

        let result = try_eval_pow_literal(&mut ctx, zero, zero_exp).unwrap();
        assert_eq!(display(&ctx, result), "undefined");
    }

    #[test]
    fn test_zero_to_negative_undefined() {
        let mut ctx = setup();
        let zero = ctx.add(Expr::Number(BigRational::zero()));
        let neg = ctx.add(Expr::Number(BigRational::from_integer((-3).into())));

        let result = try_eval_pow_literal(&mut ctx, zero, neg).unwrap();
        assert_eq!(display(&ctx, result), "undefined");
    }

    #[test]
    fn test_any_to_zero_is_one() {
        let mut ctx = setup();
        let base = ctx.add(Expr::Number(BigRational::from_integer(42.into())));
        let zero = ctx.add(Expr::Number(BigRational::zero()));

        let result = try_eval_pow_literal(&mut ctx, base, zero).unwrap();
        assert_eq!(display(&ctx, result), "1");
    }

    #[test]
    fn test_one_to_any() {
        let mut ctx = setup();
        let one = ctx.add(Expr::Number(BigRational::one()));
        let exp = ctx.add(Expr::Number(BigRational::from_integer(100.into())));

        let result = try_eval_pow_literal(&mut ctx, one, exp).unwrap();
        assert_eq!(display(&ctx, result), "1");
    }

    #[test]
    fn test_max_abs_pow_guard() {
        let mut ctx = setup();
        let base = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
        let huge_exp = ctx.add(Expr::Number(BigRational::from_integer(100_000.into())));

        // Should return None (not applicable) due to limit
        assert!(try_eval_pow_literal(&mut ctx, base, huge_exp).is_none());
    }

    #[test]
    fn test_non_literal_returns_none() {
        let mut ctx = setup();
        let x = ctx.var("x");
        let two = ctx.add(Expr::Number(BigRational::from_integer(2.into())));

        // x^2 - not literal
        assert!(try_eval_pow_literal(&mut ctx, x, two).is_none());
        // 2^x - not literal exp
        assert!(try_eval_pow_literal(&mut ctx, two, x).is_none());
    }
}
