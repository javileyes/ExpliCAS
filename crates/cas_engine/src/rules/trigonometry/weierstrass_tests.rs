use super::weierstrass::ReverseWeierstrassRule;
use crate::rule::SimpleRule;
use cas_ast::Context;
use cas_formatter::{render_expr, DisplayExpr};
use cas_math::trig_weierstrass_support::{build_weierstrass_sin, extract_tan_half_angle_like};
use cas_parser::parse;

#[test]
fn test_weierstrass_sin() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let t = build_weierstrass_sin(&mut ctx, x);
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: t
        }
    );
    assert!(result.contains("2") && result.contains("x"));
}

#[test]
fn test_extract_tan_half_angle_like() {
    let mut ctx = Context::new();
    let expr = parse("sin(x/2) / cos(x/2)", &mut ctx).unwrap();
    let result = extract_tan_half_angle_like(&ctx, expr);
    assert!(result.is_some());
}

#[test]
fn test_reverse_weierstrass_accepts_commuted_mul() {
    let mut ctx = Context::new();
    let expr = parse("(tan(x/2)*2)/(1+tan(x/2)^2)", &mut ctx).unwrap();
    let expected = parse("sin(x)", &mut ctx).unwrap();

    let rule = ReverseWeierstrassRule;
    let rewrite = rule
        .apply_simple(&mut ctx, expr)
        .expect("reverse weierstrass should match");

    assert_eq!(
        render_expr(&ctx, rewrite.new_expr),
        render_expr(&ctx, expected)
    );
}
