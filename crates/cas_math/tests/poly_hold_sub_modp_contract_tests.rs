use cas_ast::{hold, Context, Expr};
use cas_math::poly_modp_calls::try_rewrite_hold_poly_sub_to_zero;
use cas_math::poly_modp_conv::DEFAULT_PRIME;
use num_traits::Zero;

#[test]
fn hold_subtraction_equal_polynomials_rewrites_to_zero() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let one = ctx.num(1);
    let x_plus_1 = ctx.add(Expr::Add(x, one));
    let hold_a = hold::wrap_hold(&mut ctx, x_plus_1);
    let hold_b = hold::wrap_hold(&mut ctx, x_plus_1);
    let sub_expr = ctx.add(Expr::Sub(hold_a, hold_b));

    let rewritten = try_rewrite_hold_poly_sub_to_zero(&mut ctx, sub_expr, DEFAULT_PRIME)
        .expect("expected rewrite to zero");

    match ctx.get(rewritten) {
        Expr::Number(n) => assert!(n.is_zero(), "rewrite result should be 0"),
        other => panic!("expected numeric zero, got {:?}", other),
    }
}

#[test]
fn hold_subtraction_non_hold_operands_do_not_rewrite() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let y = ctx.var("y");
    let sub_expr = ctx.add(Expr::Sub(x, y));

    let rewritten = try_rewrite_hold_poly_sub_to_zero(&mut ctx, sub_expr, DEFAULT_PRIME);
    assert!(
        rewritten.is_none(),
        "non-hold subtraction must not rewrite through mod-p path"
    );
}
