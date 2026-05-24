use super::*;
use crate::rule::Rule;
use cas_ast::{target_kind::TargetKind, Context};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn sum_diff_cubes_contraction_rule_targets_mul_only() {
    let targets = SumDiffCubesContractionRule
        .target_types()
        .expect("SumDiffCubesContractionRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn combine_like_terms_rule_targets_add_and_mul_only() {
    let targets = CombineLikeTermsRule
        .target_types()
        .expect("CombineLikeTermsRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Add));
    assert!(targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn sqrt_perfect_square_rule_targets_pow_and_function_only() {
    let targets = SqrtPerfectSquareRule
        .target_types()
        .expect("SqrtPerfectSquareRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Pow));
    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
}

#[test]
fn test_distribute() {
    let mut ctx = Context::new();
    let rule = DistributeRule;
    // x^2 * (x + 3) - use x^2 (not an integer) so guard doesn't block
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let x_sq = ctx.add(Expr::Pow(x, two));
    let add = ctx.add(Expr::Add(x, three));
    let expr = ctx.add(Expr::Mul(x_sq, add));

    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    // Should be (x^2 * x) + (x^2 * 3) before further simplification
    // Note: x^2*x -> x^3 happens in a later pass, not in DistributeRule
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "x^2 * x + x^2 * 3" // Canonical: polynomial order (x terms before constants)
    );
}

#[test]
fn test_distribute_preserves_nested_binomial_factor_chain() {
    let mut ctx = Context::new();
    let rule = DistributeRule;
    let expr = parse("((x+3)*(x+4))*(x+2)", &mut ctx).expect("parse");

    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(
        rewrite.is_none(),
        "default simplification should preserve compact nested binomial products"
    );
}

#[test]
fn test_annihilation() {
    let mut ctx = Context::new();
    let rule = AnnihilationRule;
    let x = ctx.var("x");
    let expr = ctx.add(Expr::Sub(x, x));
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
}

#[test]
fn test_annihilation_hold_sum_pattern() {
    let mut ctx = Context::new();
    let rule = AnnihilationRule;
    let x = ctx.var("x");
    let y = ctx.var("y");
    let sum = ctx.add(Expr::Add(x, y));
    let held = cas_ast::hold::wrap_hold(&mut ctx, sum);
    let neg_x = ctx.add(Expr::Neg(x));
    let neg_y = ctx.add(Expr::Neg(y));
    let rhs = ctx.add(Expr::Add(neg_x, neg_y));
    let expr = ctx.add(Expr::Add(held, rhs));

    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
}

#[test]
fn test_combine_like_terms() {
    let mut ctx = Context::new();
    let rule = CombineLikeTermsRule;

    // 2x + 3x -> 5x
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let term1 = ctx.add(Expr::Mul(two, x));
    let term2 = ctx.add(Expr::Mul(three, x));
    let expr = ctx.add(Expr::Add(term1, term2));

    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "5 * x"
    );

    // x + 2x -> 3x
    let term1 = x;
    let term2 = ctx.add(Expr::Mul(two, x));
    let expr2 = ctx.add(Expr::Add(term1, term2));
    let rewrite2 = rule
        .apply(
            &mut ctx,
            expr2,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite2.new_expr
            }
        ),
        "3 * x"
    );

    // ln(x) + ln(x) -> 2 * ln(x)
    let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![x]);
    let expr3 = ctx.add(Expr::Add(ln_x, ln_x));
    let rewrite3 = rule
        .apply(
            &mut ctx,
            expr3,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    // ln(x) is log(e, x), prints as ln(x)
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite3.new_expr
            }
        ),
        "2 * ln(x)"
    );
}

#[test]
fn test_polynomial_identity_zero_rule() {
    // Test: (a+b)^2 - (a^2 + 2ab + b^2) = 0
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");

    // (a+b)^2
    let a_plus_b = ctx.add(Expr::Add(a, b));
    let two = ctx.num(2);
    let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

    // a^2 + 2ab + b^2
    let a_sq = ctx.add(Expr::Pow(a, two));
    let b_sq = ctx.add(Expr::Pow(b, two));
    let ab = ctx.add(Expr::Mul(a, b));
    let two_ab = ctx.add(Expr::Mul(two, ab));
    let sum1 = ctx.add(Expr::Add(a_sq, two_ab));
    let rhs = ctx.add(Expr::Add(sum1, b_sq));

    // (a+b)^2 - (a^2 + 2ab + b^2)
    let expr = ctx.add(Expr::Sub(a_plus_b_sq, rhs));

    let rule = PolynomialIdentityZeroRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    // Should simplify to 0
    assert!(rewrite.is_some(), "Polynomial identity should be detected");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        ),
        "0"
    );
}

#[test]
fn test_polynomial_identity_zero_rule_non_identity() {
    // Test: (a+b)^2 - a^2 ≠ 0 (not an identity)
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");

    // (a+b)^2
    let a_plus_b = ctx.add(Expr::Add(a, b));
    let two = ctx.num(2);
    let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

    // a^2
    let a_sq = ctx.add(Expr::Pow(a, two));

    // (a+b)^2 - a^2
    let expr = ctx.add(Expr::Sub(a_plus_b_sq, a_sq));

    let rule = PolynomialIdentityZeroRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    // Should NOT return a rewrite (not an identity to 0)
    assert!(rewrite.is_none(), "Non-identity should not trigger rule");
}

#[test]
fn test_polynomial_product_normalize_rule_geometric_difference() {
    let mut ctx = Context::new();
    let rule = PolynomialProductNormalizeRule;
    let expr = cas_parser::parse("(x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .expect("rewrite");
    let expected = cas_parser::parse("x^6 - 1", &mut ctx).unwrap();
    assert!(cas_math::poly_compare::poly_eq(
        &ctx,
        rewrite.new_expr,
        expected
    ));
}
