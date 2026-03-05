use super::*;
use cas_ast::{Constant, Expr};
use cas_math::ground_eval_guard::GroundEvalGuard;

fn make_ctx() -> Context {
    Context::new()
}

#[test]
fn test_ground_nonzero_constant_sqrt2() {
    // sqrt(2) = 2^(1/2) is non-zero
    let mut ctx = make_ctx();
    let two = ctx.num(2);
    let half = ctx.rational(1, 2);
    let sqrt2 = ctx.add(Expr::Pow(two, half));

    let result = try_ground_nonzero(&ctx, sqrt2);
    assert_eq!(result, Some(Proof::Proven));
}

#[test]
fn test_ground_nonzero_zero() {
    // 0 should be disproven
    let mut ctx = make_ctx();
    let zero = ctx.num(0);

    let result = try_ground_nonzero(&ctx, zero);
    assert_eq!(result, Some(Proof::Disproven));
}

#[test]
fn test_ground_nonzero_pi() {
    // π is non-zero
    let mut ctx = make_ctx();
    let pi = ctx.add(Expr::Constant(Constant::Pi));

    let result = try_ground_nonzero(&ctx, pi);
    assert_eq!(result, Some(Proof::Proven));
}

#[test]
fn test_ground_nonzero_sum_of_constants() {
    // 1 + 2 = 3 is non-zero
    let mut ctx = make_ctx();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let sum = ctx.add(Expr::Add(one, two));

    let result = try_ground_nonzero(&ctx, sum);
    assert_eq!(result, Some(Proof::Proven));
}

#[test]
fn test_ground_nonzero_diff_equal() {
    // 5 - 5 = 0 should be disproven
    let mut ctx = make_ctx();
    let five_a = ctx.num(5);
    let five_b = ctx.num(5);
    let diff = ctx.add(Expr::Sub(five_a, five_b));

    let result = try_ground_nonzero(&ctx, diff);
    assert_eq!(result, Some(Proof::Disproven));
}

#[test]
fn test_reentrancy_guard_prevents_recursion() {
    // Manually hold the guard and verify that enter() returns None
    let _guard = GroundEvalGuard::enter();
    assert!(_guard.is_some(), "First enter should succeed");

    let second = GroundEvalGuard::enter();
    assert!(second.is_none(), "Second enter should be blocked");
}
