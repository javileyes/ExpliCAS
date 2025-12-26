//! Contract tests for const_fold module.
//!
//! These tests verify the "load-bearing" behaviors of constant folding:
//! 1. Safe mode only folds constant subtrees
//! 2. Off mode never changes anything  
//! 3. ValueDomain controls sqrt(-1) semantics
//! 4. i*i -> -1 only in ComplexEnabled

use cas_ast::{Context, Expr};
use cas_engine::budget::Budget;
use cas_engine::const_fold::{fold_constants, ConstFoldMode, ConstFoldResult};
use cas_engine::semantics::{EvalConfig, ValueDomain};
use num_rational::BigRational;

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
    let sqrt_neg_one = ctx.add(Expr::Function("sqrt".to_string(), vec![neg_one]));

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

// ============================================================================
// Safe mode - RealOnly tests
// ============================================================================

#[test]
fn realonly_sqrt_negative_becomes_undefined() {
    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let sqrt_neg_one = ctx.add(Expr::Function("sqrt".to_string(), vec![neg_one]));

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
    let sqrt_four = ctx.add(Expr::Function("sqrt".to_string(), vec![four]));

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
    let sqrt_neg_one = ctx.add(Expr::Function("sqrt".to_string(), vec![neg_one]));

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
    let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));

    let result = fold(&mut ctx, sqrt_x, ConstFoldMode::Safe, ValueDomain::RealOnly);

    // Should not change (x is not constant)
    assert_eq!(result.expr, sqrt_x);
}

#[test]
fn safe_preserves_non_perfect_sqrt() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let sqrt_two = ctx.add(Expr::Function("sqrt".to_string(), vec![two]));

    let result = fold(
        &mut ctx,
        sqrt_two,
        ConstFoldMode::Safe,
        ValueDomain::RealOnly,
    );

    // sqrt(2) is not a perfect square, should not fold
    assert!(matches!(ctx.get(result.expr), Expr::Function(name, _) if name == "sqrt"));
}
