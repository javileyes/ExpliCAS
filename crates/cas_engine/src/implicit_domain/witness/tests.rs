use super::*;
use crate::semantics::ValueDomain;
use cas_ast::{Context, Expr};

#[test]
fn test_infer_sqrt_implies_nonnegative() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

    let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::RealOnly);

    assert!(domain.contains_nonnegative(x));
    assert!(!domain.contains_positive(x));
}

#[test]
fn test_infer_ln_implies_positive() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![x]);

    let domain = infer_implicit_domain(&ctx, ln_x, ValueDomain::RealOnly);

    assert!(domain.contains_positive(x));
}

#[test]
fn test_infer_div_implies_nonzero() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let one_over_x = ctx.add(Expr::Div(one, x));

    let domain = infer_implicit_domain(&ctx, one_over_x, ValueDomain::RealOnly);

    assert!(domain.contains_nonzero(x));
}

#[test]
fn test_witness_survives_sqrt() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
    let y = ctx.var("y");
    let output = ctx.add(Expr::Add(sqrt_x, y)); // sqrt(x) + y

    assert!(witness_survives(&ctx, x, output, WitnessKind::Sqrt));
}

#[test]
fn test_witness_not_survives() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    // Output is just x, no sqrt(x) witness

    assert!(!witness_survives(&ctx, x, x, WitnessKind::Sqrt));
}

#[test]
fn test_complex_enabled_returns_empty() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

    let domain = infer_implicit_domain(&ctx, sqrt_x, ValueDomain::ComplexEnabled);

    assert!(domain.is_empty());
}

#[test]
fn test_domain_delta_sqrt_square_to_x() {
    // sqrt(x)^2 -> x should be detected as ExpandsAnalytic
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
    let two = ctx.num(2);
    let sqrt_x_squared = ctx.add(Expr::Pow(sqrt_x, two));

    // Check input has NonNegative(x)
    let d_in = infer_implicit_domain(&ctx, sqrt_x_squared, ValueDomain::RealOnly);
    assert!(
        d_in.contains_nonnegative(x),
        "Input should have NonNegative(x)"
    );

    // Check output (x) has no NonNegative
    let d_out = infer_implicit_domain(&ctx, x, ValueDomain::RealOnly);
    assert!(
        d_out.is_empty(),
        "Output (just x) should have no constraints"
    );

    // Check domain_delta_check detects this as ExpandsAnalytic
    let delta = domain_delta_check(&ctx, sqrt_x_squared, x, ValueDomain::RealOnly);
    assert!(
        matches!(delta, DomainDelta::ExpandsAnalytic(_)),
        "sqrt(x)^2 -> x should be detected as ExpandsAnalytic, got {:?}",
        delta
    );
}

#[test]
fn test_domain_delta_safe_with_witness_preserved() {
    // (x-y)/(sqrt(x)-sqrt(y)) -> sqrt(x)+sqrt(y) preserves sqrt witnesses
    // This is a simplified version - we just test that sqrt in output means safe
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
    let y = ctx.var("y");
    let sqrt_y = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![y]);

    // Input: sqrt(x) - sqrt(y)
    let input = ctx.add(Expr::Sub(sqrt_x, sqrt_y));
    // Output: sqrt(x) + sqrt(y)
    let output = ctx.add(Expr::Add(sqrt_x, sqrt_y));

    let delta = domain_delta_check(&ctx, input, output, ValueDomain::RealOnly);
    assert_eq!(
        delta,
        DomainDelta::Safe,
        "sqrt witnesses preserved should be safe"
    );
}

#[test]
fn test_search_witness_deep_expression_no_overflow() {
    // Regression test: a 500-deep nested expression would overflow
    // the stack with the old recursive implementation.
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);

    // Build: Add(Add(Add(... sqrt(x) ..., 1), 1), 1) — depth 500
    let mut expr = sqrt_x;
    for _ in 0..500 {
        let one = ctx.num(1);
        expr = ctx.add(Expr::Add(expr, one));
    }

    assert!(
        witness_survives(&ctx, x, expr, WitnessKind::Sqrt),
        "sqrt(x) witness should survive deep nesting"
    );

    // Also verify the in-context variant doesn't overflow
    let dummy_replaced = ctx.num(999);
    assert!(
        witness_survives_in_context(&ctx, x, expr, dummy_replaced, None, WitnessKind::Sqrt,),
        "in-context search should survive deep nesting"
    );
}
