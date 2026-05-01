use super::hyperbolic::{
    HyperbolicDoubleAngleRule, RecognizeHyperbolicFromExpRule, SinhDoubleAngleExpansionRule,
    TanhDoubleAngleExpansionRule,
};
use crate::parent_context::ParentContext;
use crate::rule::Rule;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn denominator_of_div(ctx: &Context, expr: ExprId) -> ExprId {
    match ctx.get(expr) {
        Expr::Div(_, denominator) => *denominator,
        other => panic!("expected division, got {other:?}"),
    }
}

fn base_of_pow(ctx: &Context, expr: ExprId) -> ExprId {
    match ctx.get(expr) {
        Expr::Pow(base, _) => *base,
        other => panic!("expected power, got {other:?}"),
    }
}

#[test]
fn test_recognize_cosh_from_exp() {
    // (e^x + e^(-x))/2 -> cosh(x)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let neg_x = ctx.add(Expr::Neg(x));
    let exp_x = ctx.add(Expr::Pow(e, x));
    let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
    let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
    let two = ctx.num(2);
    let expr = ctx.add(Expr::Div(sum, two));

    let rule = RecognizeHyperbolicFromExpRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should recognize cosh(x)");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert_eq!(result, "cosh(x)");
}

#[test]
fn test_recognize_sinh_from_exp() {
    // (e^x - e^(-x))/2 -> sinh(x)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let neg_x = ctx.add(Expr::Neg(x));
    let exp_x = ctx.add(Expr::Pow(e, x));
    let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
    let diff = ctx.add(Expr::Sub(exp_x, exp_neg_x));
    let two = ctx.num(2);
    let expr = ctx.add(Expr::Div(diff, two));

    let rule = RecognizeHyperbolicFromExpRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should recognize sinh(x)");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert_eq!(result, "sinh(x)");
}

#[test]
fn test_recognize_direct_cosh_from_exp_sum_with_reciprocal() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let exp_x = ctx.add(Expr::Pow(e, x));
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, exp_x));
    let expr = ctx.add(Expr::Add(exp_x, reciprocal));

    let rule = RecognizeHyperbolicFromExpRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should recognize 2*cosh(x)");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert_eq!(result, "2 * cosh(x)");
}

#[test]
fn test_recognize_direct_sinh_from_exp_difference_with_reciprocal() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let exp_x = ctx.add(Expr::Pow(e, x));
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, exp_x));
    let expr = ctx.add(Expr::Sub(exp_x, reciprocal));

    let rule = RecognizeHyperbolicFromExpRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should recognize 2*sinh(x)");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert_eq!(result, "2 * sinh(x)");
}

#[test]
fn test_recognize_neg_sinh_from_exp() {
    // (e^(-x) - e^x)/2 -> -sinh(x)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let neg_x = ctx.add(Expr::Neg(x));
    let exp_x = ctx.add(Expr::Pow(e, x));
    let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
    // Note: order is reversed
    let diff = ctx.add(Expr::Sub(exp_neg_x, exp_x));
    let two = ctx.num(2);
    let expr = ctx.add(Expr::Div(diff, two));

    let rule = RecognizeHyperbolicFromExpRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should recognize -sinh(x)");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert!(
        result.contains("sinh") && result.contains("-"),
        "Should be -sinh(x), got: {}",
        result
    );
}

#[test]
fn test_no_match_different_args() {
    // (e^x + e^(-y))/2 should NOT match (different args)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let neg_y = ctx.add(Expr::Neg(y));
    let exp_x = ctx.add(Expr::Pow(e, x));
    let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let exp_neg_y = ctx.add(Expr::Pow(e2, neg_y));
    let sum = ctx.add(Expr::Add(exp_x, exp_neg_y));
    let two = ctx.num(2);
    let expr = ctx.add(Expr::Div(sum, two));

    let rule = RecognizeHyperbolicFromExpRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_none(), "Should NOT match different args");
}

#[test]
fn test_no_match_wrong_divisor() {
    // (e^x + e^(-x))/3 should NOT match (not divided by 2)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let neg_x = ctx.add(Expr::Neg(x));
    let exp_x = ctx.add(Expr::Pow(e, x));
    let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
    let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
    let three = ctx.num(3);
    let expr = ctx.add(Expr::Div(sum, three));

    let rule = RecognizeHyperbolicFromExpRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_none(), "Should NOT match divisor != 2");
}

#[test]
fn test_hyperbolic_double_angle_rule() {
    // cosh(x)^2 + sinh(x)^2 -> cosh(2*x)
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let cosh_x = ctx.call_builtin(cas_ast::BuiltinFn::Cosh, vec![x]);
    let sinh_x = ctx.call_builtin(cas_ast::BuiltinFn::Sinh, vec![x]);
    let two = ctx.num(2);
    let two2 = ctx.num(2);
    let cosh_sq = ctx.add(Expr::Pow(cosh_x, two));
    let sinh_sq = ctx.add(Expr::Pow(sinh_x, two2));
    let expr = ctx.add(Expr::Add(cosh_sq, sinh_sq));

    let rule = HyperbolicDoubleAngleRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should apply cosh²+sinh² -> cosh(2x)");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert!(
        result.contains("cosh") && result.contains("2"),
        "Should be cosh(2*x), got: {}",
        result
    );
}

#[test]
fn sinh_double_angle_expansion_still_applies_at_root() {
    let mut ctx = Context::new();
    let expr = parse("sinh(2*x)", &mut ctx).expect("parse");

    let rewrite = SinhDoubleAngleExpansionRule.apply(&mut ctx, expr, &ParentContext::root());

    assert!(
        rewrite.is_some(),
        "root-level sinh double-angle expansion should remain available"
    );
}

#[test]
fn hyperbolic_double_angle_expansion_skips_division_denominators() {
    let mut ctx = Context::new();
    let sinh_quotient = parse("1/sinh(2*x)^2", &mut ctx).expect("parse sinh quotient");
    let sinh_denominator = denominator_of_div(&ctx, sinh_quotient);
    let sinh = base_of_pow(&ctx, sinh_denominator);
    let sinh_parent_ctx = ParentContext::root()
        .extend_with_div_check(sinh_quotient, &ctx)
        .extend_with_div_check(sinh_denominator, &ctx);

    assert!(
        SinhDoubleAngleExpansionRule
            .apply(&mut ctx, sinh, &sinh_parent_ctx)
            .is_none(),
        "sinh double-angle expansion should not expand denominator internals"
    );

    let tanh_quotient = parse("1/tanh(2*x)", &mut ctx).expect("parse tanh quotient");
    let tanh = denominator_of_div(&ctx, tanh_quotient);
    let tanh_parent_ctx = ParentContext::root().extend_with_div_check(tanh_quotient, &ctx);

    assert!(
        TanhDoubleAngleExpansionRule
            .apply(&mut ctx, tanh, &tanh_parent_ctx)
            .is_none(),
        "tanh double-angle expansion should not expand denominator internals"
    );
}
