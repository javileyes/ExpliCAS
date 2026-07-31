//! Tests de las reglas aritméticas: familia `powers` (troceo P1).

use super::*;

#[test]
fn expand_odd_half_power_to_enable_cancellation_rule_matches_nonnegative_target() {
    let mut ctx = Context::new();
    let expr =
        parse("sqrt(x^5) - x^2*sqrt(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expected = parse("x^2*sqrt(x) - x^2*sqrt(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse expected: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandOddHalfPowerToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expected
            }
        )
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn expand_odd_half_power_to_enable_cancellation_rule_matches_reversed_side() {
    let mut ctx = Context::new();
    let expr =
        parse("x^3*sqrt(x) - sqrt(x^7)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expected = parse("x^3*sqrt(x) - x^3*sqrt(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse expected: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandOddHalfPowerToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expected
            }
        )
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn additive_scope_contains_zero_term_rejects_abs_sqrt_pair_without_zero_regression() {
    let mut ctx = Context::new();
    let expr = parse("sqrt(x + 2*sqrt(x-1)) - abs(1 + sqrt(x-1))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::additive_scope_contains_zero_term(
        &mut ctx, expr
    ));
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_sine_sixth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(x)^6 - ((10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Power Reduction Identity");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_scaled_sine_fourth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("8*sin(x)^4 - (3-4*cos(2*x)+cos(4*x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Power Reduction Identity");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_cosine_twenty_fourth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse(
        "cos(x)^24 - ((1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Power Reduction Identity");
}
#[test]
fn small_direct_zero_core_rewrite_matches_rationalized_sum_of_sqrts_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "1/(sqrt(a) + sqrt(b)) - (sqrt(a) - sqrt(b))/(a - b)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.final_expr()
            }
        ),
        "0"
    );
    assert_eq!(rewrite.required_conditions.len(), 4);
}
#[test]
fn small_direct_zero_core_rewrite_matches_odd_half_power_core() {
    let mut ctx = Context::new();
    let expr =
        parse("sqrt(x^5) - x^2*sqrt(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.final_expr()
            }
        ),
        "0"
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn small_direct_zero_core_rewrite_matches_symbolic_root_denesting_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.final_expr()
            }
        ),
        "0"
    );
    assert_eq!(rewrite.required_conditions.len(), 2);
}
#[test]
fn small_structural_poly_zero_core_rewrite_matches_factor_difference_squares() {
    let mut ctx = Context::new();
    let expr =
        parse("p^2-q^2 - (p-q)*(p+q)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_structural_poly_zero_core_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.final_expr()
            }
        ),
        "0"
    );
}
#[test]
fn exact_zero_identity_rewrite_matches_symbolic_root_denesting_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.final_expr()
            }
        ),
        "0"
    );
}
#[test]
fn direct_reciprocal_half_power_shared_denominator_rewrite_matches_root_form() {
    let mut ctx = Context::new();
    let lhs =
        parse("tan(x)^(-1/2)/(2*cos(x)^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("1/(2*cos(x)^2*sqrt(tan(x)))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_reciprocal_half_power_shared_denominator_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_reciprocal_half_power_product_rewrite_matches_expanded_factor() {
    let mut ctx = Context::new();
    let lhs = parse("asinh(2*x+1)^(-1/2)*((2*x+1)^2+1)^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(asinh(2*x+1)*(4*x^2+4*x+2))^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_reciprocal_half_power_product_rewrite_matches_sqrt_denominator_form() {
    let mut ctx = Context::new();
    let lhs = parse("asinh(2*x+1)^(-1/2)*((2*x+1)^2+1)^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1/(sqrt(asinh(2*x+1))*sqrt((2*x+1)^2+1))", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_reciprocal_half_power_product_rewrite_matches_sqrt_half_power_base_mix() {
    let mut ctx = Context::new();
    let lhs =
        parse("1/(sqrt(x)*sqrt(sqrt(x)-x))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(x*(x^(1/2)-x))^(-1/2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_reciprocal_half_power_product_rewrite_matches_scaled_sqrt_denominator_form() {
    let mut ctx = Context::new();
    let lhs = parse("(2*asinh(2*x+1)^(-1/2)/sqrt((2*x+1)^2+1))/2", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1/(sqrt(asinh(2*x+1))*sqrt((2*x+1)^2+1))", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_negative_even_root_power_reciprocal_rewrite_matches_scaled_polynomial() {
    let mut ctx = Context::new();
    let lhs =
        parse("(2*x^2+2*x-3)^(-3/2)*(4*x+2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("(4*x+2)/(2*x^2+2*x-3)^(3/2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_negative_even_root_power_reciprocal_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Negative Even-Root Power Reciprocal Cancellation",
    );
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
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_negative_even_root_power_reciprocal_rewrite_matches_sqrt_power_denominator() {
    let mut ctx = Context::new();
    let lhs = parse("cos(x)*sin(x)^(-3/2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cos(x)/sqrt(sin(x)^3)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_negative_even_root_power_reciprocal_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Negative Even-Root Power Reciprocal Cancellation",
    );
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
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn exact_zero_identity_rewrite_matches_negative_even_root_power_reciprocal_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*x^2+2*x-3)^(-3/2)*(4*x+2) - (4*x+2)/(2*x^2+2*x-3)^(3/2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Negative Even-Root Power Reciprocal Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_negative_even_root_power_reciprocal_product_cofactor() {
    let mut ctx = Context::new();
    let expr = parse(
        "(y+1)*(2*x^2+2*x-3)^(-3/2)*(4*x+2) - ((y+1)*(4*x+2))/(2*x^2+2*x-3)^(3/2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Negative Even-Root Power Reciprocal Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_product_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "asinh(2*x+1)^(-1/2)*((2*x+1)^2+1)^(-1/2) - (asinh(2*x+1)*(4*x^2+4*x+2))^(-1/2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_scaled_reciprocal_half_power_product_residual() {
    let mut ctx = Context::new();
    let expr = parse("(a/(2*sqrt(x)*sqrt(1-x)))*2 - a*(x*(1-x))^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_quotient_product_one_residual() {
    let mut ctx = Context::new();
    let expr = parse("1 - x^(-1/2)*((x-1)/x)^(-1/2)*(x-1)^(1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Quotient Product Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_quotient_over_base_residual() {
    let mut ctx = Context::new();
    let expr = parse("(x-4)^(1/2)/(x*(x-4)) - x^(-3/2)*(1-4/x)^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Quotient/Base Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_linear_residual() {
    let mut ctx = Context::new();
    let expr = parse("x^(-1/2) - (2*x-1)/(2*x^(3/2)) - 1/2*x^(-3/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));
    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Linear Residual Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_common_sqrt_denominator_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "-(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1)) + \
         (3*x+1)/(2*x^(1/2)+2*x^(3/2)*(x+1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Sqrt Denominator Factor Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_scaled_common_sqrt_denominator_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(-1/2)*((3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1))) + \
         (3*x+1)/(4*x^(1/2)+4*x^(3/2)*(x+1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Sqrt Denominator Factor Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_sqrt_over_base_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x^(-1/2) - 3*x^(3/2))/(2*(x^2+1)^2) - \
         (x^(1/2)-3*x^(5/2))/(2*x*(x^2+1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Sqrt/Base Fraction Cancellation");
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
fn exact_zero_identity_rewrite_matches_rationalized_common_sqrt_denominator_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "-(3*x+1)/((x*(x+1)^2+1)*sqrt(x)) + \
         (3*x+1)*(x^(1/2)-x^(3/2)*(x+1)^2)/(x-x^3*(x+1)^4)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Rationalized Sqrt Denominator Cancellation",
    );
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
fn exact_zero_identity_rewrite_matches_scaled_reciprocal_half_power_over_base_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x*(log2(x^2+1)+1)^(-1/2))/(ln(2)*(x^2+1)) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(log2(x^2+1)+1)*(x^2+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_scaled_reciprocal_half_power_over_base_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x*(log2(x^2+1)+1)^(-1/2))/(ln(2)*(x^2+1)) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(log2(x^2+1)+1)*(x^2+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_scaled_reciprocal_half_power_over_base_residual_with_unfactored_common_denominators(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((log2(x^2+1)+1)^(-1/2)*x*2)/(ln(2)*(2*q*x^2+2*q)) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(x^2+1)*(q*log2(x^2+1)+q))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_product_wrapped_scaled_reciprocal_half_power_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "2*(((log2(x^2+1)+1)^(-1/2)*x)/(ln(2)*(2*q*x^2+2*q))) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(x^2+1)*(q*log2(x^2+1)+q))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_scaled_reciprocal_half_power_shared_denominator_residual_with_unfactored_common_denominator(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "2*(x/sqrt(log2(x^2+1)+1))/(2*q*ln(2)+2*q*ln(2)*x^2) - \
         x/(q*ln(2)*sqrt(log2(x^2+1)+1)*(x^2+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn collapse_exact_zero_product_factor_matches_scaled_reciprocal_half_power_product_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "1/2*((a/(2*sqrt(x)*sqrt(1-x)))*2 - a*(x*(1-x))^(-1/2))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rewrite =
        super::super::try_build_exact_zero_product_factor_rewrite(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
fn div_zero_rule_matches_scaled_reciprocal_half_power_product_residual_over_constant() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a/(2*sqrt(x)*sqrt(1-x)))*2 - a*(x*(1-x))^(-1/2))/2",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_radical_numerator_const_division_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
    );
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
fn direct_sub_fraction_combination_equivalence_rewrite_matches_symbolic_difference_squares_pair() {
    let mut ctx = Context::new();
    let lhs =
        parse("1/(2*a)*(1/(x-a) - 1/(x+a))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1/(x^2-a^2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_sub_fraction_combination_equivalence_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Subtract Fractions");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shared_passthrough_square_base_equivalence_keeps_quartic_conditional_factor_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((a*x^4 + b*x^3 + c*x^2 + d)^2) + m) - (((x^2*(a*x^2 + b*x + c + d/x^2))^2) + m)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::has_plausible_shared_additive_passthrough_difference_shape(&mut ctx, expr)
    );
    let rewrite =
        super::super::try_build_exact_zero_shared_passthrough_difference_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

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
fn collapse_exact_zero_additive_subexpression_matches_quartic_conditional_factor_square_difference()
{
    let mut ctx = Context::new();
    let expr = parse(
        "(a*x^4 + b*x^3 + c*x^2 + d)^2 - (x^2*(a*x^2 + b*x + c + d/x^2))^2",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
fn collapse_exact_zero_additive_subexpression_matches_telescoping_fraction_double_squared_passthrough(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((((1/(x+b) - 1/(x+c))^2) + m)^2) - (((((c-b)/(x^2+(b+c)*x+b*c))^2) + m)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Subtract Fractions");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_morrie_double_squared_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((((cos(x)*cos(2*x)*cos(4*x))^2) + m)^2) - ((((sin(8*x)/(8*sin(x)))^2) + m)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Apply Morrie's law");
    assert!(!rewrite.required_conditions.is_empty());
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_complete_square_with_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^2 - b*x + c) + m) - ((a*(x - b/(2*a))^2 + c - b^2/(4*a)) + m)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Complete the Square");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a + b + c)^2 - (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}
#[test]
fn small_polynomial_expand_helper_expands_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse("(a + b + c)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expanded = cas_math::expansion_rule_support::try_expand_small_pow_sum_expr(
        &mut ctx,
        expr,
        cas_math::expansion_rule_support::SmallPowExpandPolicy {
            max_vars: 3,
            ..cas_math::expansion_rule_support::SmallPowExpandPolicy::default()
        },
    )
    .unwrap_or_else(|| panic!("expanded"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expanded
            }
        ),
        "a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c"
    );
}
#[test]
fn fast_small_polynomial_expansion_zero_scope_rewrite_matches_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a + b + c)^2 - (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_fast_small_polynomial_expansion_zero_scope_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}
#[test]
fn fast_small_polynomial_expansion_zero_scope_rewrite_matches_signed_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a - b + c)^2 - (a^2 + b^2 + c^2 - 2*a*b + 2*a*c - 2*b*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_fast_small_polynomial_expansion_zero_scope_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}
#[test]
fn small_polynomial_expand_helper_expands_signed_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse("(a - b + c)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expanded = cas_math::expansion_rule_support::try_expand_small_pow_sum_expr(
        &mut ctx,
        expr,
        cas_math::expansion_rule_support::SmallPowExpandPolicy {
            max_vars: 3,
            ..cas_math::expansion_rule_support::SmallPowExpandPolicy::default()
        },
    )
    .unwrap_or_else(|| panic!("expanded"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expanded
            }
        ),
        "a^2 + b^2 + c^2 + -2 * a * b + -2 * b * c + 2 * a * c"
    );
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_trinomial_square_with_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a + b + c)^2 + m) - ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) + m)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_complete_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(a*x^2 - b*x + c) - k*(a*(x - b/(2*a))^2 + c - b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Complete the Square");
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(a + b + c)^2 - k*(a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}
#[test]
fn collapse_exact_zero_same_denominator_rule_matches_complete_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^2 + b*x + c)/q) - ((a*(x + b/(2*a))^2 + c - b^2/(4*a))/q)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Complete the Square");
}
#[test]
fn direct_sum_diff_cubes_core_equivalence_rewrite_matches_difference_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(a^3-b^3)/(a-b)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("a^2 + a*b + b^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        rewrite.description,
        "Subtract Expanded Sum/Difference of Cubes Quotient"
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn direct_sum_diff_cubes_core_equivalence_rewrite_matches_sum_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(a^3+b^3)/(a+b)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("a^2 - a*b + b^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        rewrite.description,
        "Subtract Expanded Sum/Difference of Cubes Quotient"
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}
