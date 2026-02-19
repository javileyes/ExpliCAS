//! Contract tests for const_fold module.
//!
//! These tests verify the "load-bearing" behaviors of constant folding:
//! 1. Safe mode only folds constant subtrees
//! 2. Off mode never changes anything  
//! 3. ValueDomain controls sqrt(-1) semantics
//! 4. i*i -> -1 only in ComplexEnabled

use cas_ast::{Context, Expr};
use cas_engine::semantics::{EvalConfig, ValueDomain};
use cas_engine::Budget;
use cas_engine::{fold_constants, ConstFoldMode, ConstFoldResult};
use num_rational::BigRational;
use num_traits::Zero;

/// Helper to fold with given mode and value_domain
fn fold(
    ctx: &mut Context,
    expr: cas_ast::ExprId,
    mode: ConstFoldMode,
    value_domain: ValueDomain,
) -> ConstFoldResult {
    let cfg = EvalConfig {
        value_domain,
        ..Default::default()
    };
    let mut budget = Budget::preset_unlimited();
    fold_constants(ctx, expr, &cfg, mode, &mut budget).unwrap()
}

// ============================================================================
// Off mode tests
// ============================================================================

#[test]
fn off_mode_is_noop() {
    let mut ctx = Context::new();
    let expr = ctx.num(42);

    let result = fold(&mut ctx, expr, ConstFoldMode::Off, ValueDomain::RealOnly);

    assert_eq!(result.expr, expr);
    assert_eq!(result.folds_performed, 0);
    assert_eq!(result.nodes_created, 0);
}

#[test]
fn off_mode_preserves_sqrt_negative() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let sqrt_neg_one = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![neg_one]);

    let result = fold(
        &mut ctx,
        sqrt_neg_one,
        ConstFoldMode::Off,
        ValueDomain::RealOnly,
    );

    // Should not change
    assert_eq!(result.expr, sqrt_neg_one);
    assert_eq!(result.folds_performed, 0);
}

#[test]
fn off_mode_complex_preserves_sqrt_negative() {
    // KEY CONTRACT: off mode is noop even when complex is enabled
    // ValueDomain defines available semantics, but const_fold(off) doesn't apply them
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let sqrt_neg_one = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![neg_one]);

    let result = fold(
        &mut ctx,
        sqrt_neg_one,
        ConstFoldMode::Off,
        ValueDomain::ComplexEnabled, // Complex enabled but fold is OFF
    );

    // Should not change - off means no folding regardless of ValueDomain
    assert_eq!(result.expr, sqrt_neg_one);
    assert_eq!(result.folds_performed, 0);
}

// ============================================================================
// Safe mode - RealOnly tests
// ============================================================================

#[test]
fn realonly_sqrt_negative_becomes_undefined() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let sqrt_neg_one = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![neg_one]);

    let result = fold(
        &mut ctx,
        sqrt_neg_one,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // Should fold to undefined
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Constant(cas_ast::Constant::Undefined)
    ));
    assert!(result.folds_performed > 0);
}

#[test]
fn realonly_sqrt_perfect_square() {
    let mut ctx = Context::new();
    let four = ctx.num(4);
    let sqrt_four = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![four]);

    let result = fold(
        &mut ctx,
        sqrt_four,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // Should fold to 2
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(2.into())
    ));
}

// ============================================================================
// Safe mode - ComplexEnabled tests
// ============================================================================

#[test]
fn complex_sqrt_minus_one_produces_i_times_sqrt() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let sqrt_neg_one = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![neg_one]);

    let result = fold(
        &mut ctx,
        sqrt_neg_one,
        ConstFoldMode::Safe,
        ValueDomain::ComplexEnabled,
    );

    // Should fold to i * sqrt(1) = Mul(i, sqrt(1))
    assert!(matches!(ctx.get(result.expr), Expr::Mul(_, _)));
}

#[test]
fn complex_i_times_i_is_minus_one() {
    let mut ctx = Context::new();
    let i1 = ctx.add(Expr::Constant(cas_ast::Constant::I));
    let i2 = ctx.add(Expr::Constant(cas_ast::Constant::I));
    let i_times_i = ctx.add(Expr::Mul(i1, i2));

    let result = fold(
        &mut ctx,
        i_times_i,
        ConstFoldMode::Safe,
        ValueDomain::ComplexEnabled,
    );

    // Should fold to -1
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer((-1).into())
    ));
}

#[test]
fn realonly_i_times_i_not_folded() {
    let mut ctx = Context::new();
    let i1 = ctx.add(Expr::Constant(cas_ast::Constant::I));
    let i2 = ctx.add(Expr::Constant(cas_ast::Constant::I));
    let i_times_i = ctx.add(Expr::Mul(i1, i2));

    let result = fold(
        &mut ctx,
        i_times_i,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // Should NOT fold (not in complex mode)
    assert!(matches!(ctx.get(result.expr), Expr::Mul(_, _)));
}

// ============================================================================
// Safe mode - no invention tests
// ============================================================================

#[test]
fn safe_does_not_fold_nonconstant() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

    let result = fold(&mut ctx, sqrt_x, ConstFoldMode::Safe, ValueDomain::RealOnly);

    // Should not change (x is not constant)
    assert_eq!(result.expr, sqrt_x);
}

#[test]
fn safe_preserves_non_perfect_sqrt() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let sqrt_two = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![two]);

    let result = fold(
        &mut ctx,
        sqrt_two,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // sqrt(2) is not a perfect square, should not fold
    assert!(
        matches!(ctx.get(result.expr), Expr::Function(name, _) if ctx.sym_name(*name) == "sqrt")
    );
}

// ============================================================================
// PR2.1: Pow folding tests
// ============================================================================

#[test]
fn pow_off_mode_noop() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let three = ctx.num(3);
    let pow_expr = ctx.add(Expr::Pow(two, three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Off,
        ValueDomain::RealOnly,
    );

    // Should not change in Off mode
    assert_eq!(result.expr, pow_expr);
    assert_eq!(result.folds_performed, 0);
}

#[test]
fn real_pow_int() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let three = ctx.num(3);
    let pow_expr = ctx.add(Expr::Pow(two, three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 2^3 = 8
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(8.into())
    ));
}

#[test]
fn real_pow_neg_base_odd() {
    let mut ctx = Context::new();
    let neg_two = ctx.num(-2);
    let three = ctx.num(3);
    let pow_expr = ctx.add(Expr::Pow(neg_two, three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // (-2)^3 = -8
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer((-8).into())
    ));
}

#[test]
fn real_pow_neg_base_even() {
    let mut ctx = Context::new();
    let neg_two = ctx.num(-2);
    let four = ctx.num(4);
    let pow_expr = ctx.add(Expr::Pow(neg_two, four));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // (-2)^4 = 16
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(16.into())
    ));
}

#[test]
fn real_pow_zero_exp() {
    let mut ctx = Context::new();
    let five = ctx.num(5);
    let zero = ctx.num(0);
    let pow_expr = ctx.add(Expr::Pow(five, zero));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 5^0 = 1
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(1.into())
    ));
}

#[test]
fn real_pow_zero_zero_undefined() {
    let mut ctx = Context::new();
    let zero1 = ctx.num(0);
    let zero2 = ctx.num(0);
    let pow_expr = ctx.add(Expr::Pow(zero1, zero2));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 0^0 = undefined
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Constant(cas_ast::Constant::Undefined)
    ));
}

#[test]
fn real_pow_zero_positive() {
    let mut ctx = Context::new();
    let zero = ctx.num(0);
    let five = ctx.num(5);
    let pow_expr = ctx.add(Expr::Pow(zero, five));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 0^5 = 0
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n.is_zero()
    ));
}

#[test]
fn real_pow_neg_half_undefined() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    // Create 1/2 as rational
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let pow_expr = ctx.add(Expr::Pow(neg_one, half));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // (-1)^(1/2) = undefined in RealOnly
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Constant(cas_ast::Constant::Undefined)
    ));
}

#[test]
fn complex_pow_neg_half_is_i() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let pow_expr = ctx.add(Expr::Pow(neg_one, half));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::ComplexEnabled,
    );

    // (-1)^(1/2) = i in ComplexEnabled
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Constant(cas_ast::Constant::I)
    ));
}

#[test]
fn complex_pow_int_still_works() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let three = ctx.num(3);
    let pow_expr = ctx.add(Expr::Pow(two, three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::ComplexEnabled,
    );

    // 2^3 = 8 (same in complex mode)
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(8.into())
    ));
}

#[test]
fn no_fold_symbolic_pow() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let two = ctx.num(2);
    let pow_expr = ctx.add(Expr::Pow(x, two));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // x^2 should not change (x is not constant)
    assert_eq!(result.expr, pow_expr);
}

#[test]
fn no_fold_nontrivial_root() {
    let mut ctx = Context::new();
    let neg_two = ctx.num(-2);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let pow_expr = ctx.add(Expr::Pow(neg_two, half));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::ComplexEnabled,
    );

    // (-2)^(1/2) is not in allowlist, should not fold
    assert!(matches!(ctx.get(result.expr), Expr::Pow(_, _)));
}

#[test]
fn no_fold_general_rational_exp() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let third = ctx.add(Expr::Number(BigRational::new(1.into(), 3.into())));
    let pow_expr = ctx.add(Expr::Pow(two, third));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 2^(1/3) is not in allowlist, should not fold
    assert!(matches!(ctx.get(result.expr), Expr::Pow(_, _)));
}

// ============================================================================
// PR2.2: Negative integer exponent tests
// ============================================================================

#[test]
fn pow_neg_int_off_noop() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let neg_three = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(two, neg_three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Off,
        ValueDomain::RealOnly,
    );

    assert_eq!(result.expr, pow_expr);
    assert_eq!(result.folds_performed, 0);
}

#[test]
fn real_pow_neg_int() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let neg_three = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(two, neg_three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 2^(-3) = 1/8
    let expected = BigRational::new(1.into(), 8.into());
    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n == &expected));
}

#[test]
fn real_pow_neg_int_neg_base_odd() {
    let mut ctx = Context::new();
    let neg_two = ctx.num(-2);
    let neg_three = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(neg_two, neg_three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // (-2)^(-3) = -1/8
    let expected = BigRational::new((-1).into(), 8.into());
    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n == &expected));
}

#[test]
fn real_pow_neg_int_neg_base_even() {
    let mut ctx = Context::new();
    let neg_two = ctx.num(-2);
    let neg_four = ctx.num(-4);
    let pow_expr = ctx.add(Expr::Pow(neg_two, neg_four));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // (-2)^(-4) = 1/16
    let expected = BigRational::new(1.into(), 16.into());
    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n == &expected));
}

#[test]
fn real_pow_zero_neg_undefined() {
    let mut ctx = Context::new();
    let zero = ctx.num(0);
    let neg_three = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(zero, neg_three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 0^(-3) = undefined
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Constant(cas_ast::Constant::Undefined)
    ));
}

#[test]
fn real_pow_one_neg() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let neg_999 = ctx.num(-999);
    let pow_expr = ctx.add(Expr::Pow(one, neg_999));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // 1^(-999) = 1
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(1.into())
    ));
}

#[test]
fn real_pow_minus_one_neg_odd() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let neg_three = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(neg_one, neg_three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // (-1)^(-3) = -1
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer((-1).into())
    ));
}

#[test]
fn real_pow_minus_one_neg_even() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let neg_four = ctx.num(-4);
    let pow_expr = ctx.add(Expr::Pow(neg_one, neg_four));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // (-1)^(-4) = 1
    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(1.into())
    ));
}

#[test]
fn no_fold_symbolic_neg_pow() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let neg_three = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(x, neg_three));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // x^(-3) should not fold
    assert_eq!(result.expr, pow_expr);
}

// ============================================================================
// PR2.3: Rational base with integer exponent tests
// ============================================================================

#[test]
fn rational_base_pos_exp() {
    let mut ctx = Context::new();
    // (3/4)^2 = 9/16
    let base = ctx.add(Expr::Number(BigRational::new(3.into(), 4.into())));
    let exp = ctx.num(2);
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    let expected = BigRational::new(9.into(), 16.into());
    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n == &expected));
}

#[test]
fn rational_base_neg_base_odd_exp() {
    let mut ctx = Context::new();
    // (-3/4)^3 = -27/64
    let base = ctx.add(Expr::Number(BigRational::new((-3).into(), 4.into())));
    let exp = ctx.num(3);
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    let expected = BigRational::new((-27).into(), 64.into());
    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n == &expected));
}

#[test]
fn rational_base_neg_exp() {
    let mut ctx = Context::new();
    // (3/4)^(-2) = 16/9
    let base = ctx.add(Expr::Number(BigRational::new(3.into(), 4.into())));
    let exp = ctx.num(-2);
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    let expected = BigRational::new(16.into(), 9.into());
    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n == &expected));
}

#[test]
fn rational_base_neg_base_neg_odd_exp() {
    let mut ctx = Context::new();
    // (-3/4)^(-3) = -64/27
    let base = ctx.add(Expr::Number(BigRational::new((-3).into(), 4.into())));
    let exp = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    let expected = BigRational::new((-64).into(), 27.into());
    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n == &expected));
}

#[test]
fn rational_base_zero_pos_exp() {
    let mut ctx = Context::new();
    // (0/5)^3 = 0
    let base = ctx.add(Expr::Number(BigRational::new(0.into(), 5.into())));
    let exp = ctx.num(3);
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    assert!(matches!(ctx.get(result.expr), Expr::Number(n) if n.is_zero()));
}

#[test]
fn rational_base_zero_neg_exp_undefined() {
    let mut ctx = Context::new();
    // (0/5)^(-3) = undefined
    let base = ctx.add(Expr::Number(BigRational::new(0.into(), 5.into())));
    let exp = ctx.num(-3);
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    assert!(matches!(
        ctx.get(result.expr),
        Expr::Constant(cas_ast::Constant::Undefined)
    ));
}

#[test]
fn no_fold_rational_base_fractional_exp() {
    let mut ctx = Context::new();
    // (2/3)^(1/2) should not fold (exp not integer)
    let base = ctx.add(Expr::Number(BigRational::new(2.into(), 3.into())));
    let exp = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // Should not fold
    assert!(matches!(ctx.get(result.expr), Expr::Pow(_, _)));
}

#[test]
fn rational_base_one_any_exp() {
    let mut ctx = Context::new();
    // (5/5)^(-999) = 1^(-999) = 1
    let base = ctx.add(Expr::Number(BigRational::new(5.into(), 5.into()))); // = 1
    let exp = ctx.num(-999);
    let pow_expr = ctx.add(Expr::Pow(base, exp));

    let result = fold(
        &mut ctx,
        pow_expr,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    assert!(matches!(
        ctx.get(result.expr),
        Expr::Number(n) if n == &BigRational::from_integer(1.into())
    ));
}
