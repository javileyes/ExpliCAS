use super::cancel_common_terms::cancel_common_additive_terms;
use cas_ast::{Context, Expr};

#[test]
fn test_cancel_simple() {
    // (x^2 + y) - y → x^2 (on LHS), 0 (on RHS)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let two = ctx.num(2);
    let x2 = ctx.add(Expr::Pow(x, two));
    let y = ctx.var("y");
    let lhs = ctx.add(Expr::Add(x2, y));
    let rhs = y;
    let result = cancel_common_additive_terms(&mut ctx, lhs, rhs).unwrap();
    assert_eq!(result.cancelled_count, 1);
    assert!(matches!(ctx.get(result.new_lhs), Expr::Pow(_, _)));
    assert!(matches!(ctx.get(result.new_rhs), Expr::Number(_))); // 0
}

#[test]
fn test_no_cancel_different_terms() {
    // (x + y) vs z → no cancellation
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let z = ctx.var("z");
    let lhs = ctx.add(Expr::Add(x, y));
    assert!(cancel_common_additive_terms(&mut ctx, lhs, z).is_none());
}

#[test]
fn test_cancel_with_duplicates() {
    // (a + b + b) vs b → cancels one b, leaves (a + b) vs 0
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let b2 = ctx.var("b");
    let ab = ctx.add(Expr::Add(a, b));
    let lhs = ctx.add(Expr::Add(ab, b2));
    let result = cancel_common_additive_terms(&mut ctx, lhs, b).unwrap();
    assert_eq!(result.cancelled_count, 1);
}

#[test]
fn test_cancel_symmetric() {
    // (a + b + c) vs (b + c) → a vs 0, cancelled=2
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");
    let ab = ctx.add(Expr::Add(a, b));
    let lhs = ctx.add(Expr::Add(ab, c));
    let rhs = ctx.add(Expr::Add(b, c));
    let result = cancel_common_additive_terms(&mut ctx, lhs, rhs).unwrap();
    assert_eq!(result.cancelled_count, 2);
    // new_lhs should be a, new_rhs should be 0
    assert!(matches!(ctx.get(result.new_lhs), Expr::Variable(_)));
    assert!(matches!(ctx.get(result.new_rhs), Expr::Number(_)));
}
