//! Calculus rules: differentiation, integration, summation, and products.
//!
//! This module is split into submodules:
//! - `differentiation`: symbolic derivative computation
//! - `integration`: symbolic integral computation + helpers
//! - `summation`: finite sum/product evaluation (SumRule, ProductRule)

mod acosh_affine_derivative_presentation;
mod acosh_over_sqrt_derivative_presentation;
mod acosh_sqrt_derivative_presentation;
mod acosh_strictly_positive_polynomial_derivative_presentation;
mod affine_inverse_family_post_calculus_presentation;
mod arctan_integrand_preservation;
mod arctan_polynomial_integrand_presentation;
mod arctan_sqrt_additive_derivative_presentation;
mod arctan_sqrt_additive_post_calculus_presentation;
mod arctan_sqrt_constant_over_polynomial_presentation;
mod arctan_sqrt_positive_shift_derivative_presentation;
mod arctan_sqrt_quotient_derivative_presentation;
mod arctan_surd_derivative_presentation;
mod asinh_polynomial_derivative_presentation;
mod asinh_sqrt_constant_over_polynomial_presentation;
mod asinh_sqrt_derivative_presentation;
mod asinh_surd_derivative_presentation;
mod atanh_sqrt_constant_over_polynomial_presentation;
mod atanh_sqrt_derivative_presentation;
mod atanh_sqrt_quotient_derivative_presentation;
mod atanh_surd_derivative_presentation;
mod bounded_inverse_root_quotient_post_calculus_presentation;
mod bounded_inverse_trig_projection_presentation;
mod bounded_inverse_trig_shifted_sqrt_derivative_presentation;
mod bounded_inverse_trig_sqrt_derivative_presentation;
mod by_parts_integrand_preservation;
mod constant_scaled_inverse_trig_root_post_calculus_presentation;
mod derivative_integrand_factor_parts;
mod diff_post_calculus_presentation;
mod diff_rule;
mod diff_rule_support;
mod differentiation;
mod direct_trig_affine_integrand_presentation;
mod domain_checks;
mod elementary_inverse_function_post_calculus_presentation;
mod elementary_sqrt_derivative_presentation;
mod elementary_variable_term_presentation;
mod exp_by_parts_integrand_presentation;
mod exponential_derivative_presentation;
mod fractional_denominator_power_integrand_preservation;
mod gap_presentation;
mod hyperbolic_by_parts_integrand_presentation;
mod hyperbolic_power_integrand_presentation;
mod hyperbolic_primitive_derivative_presentation;
mod integral_derivative_shortcut_presentation;
mod integrate_rule;
mod integration;
mod inverse_hyperbolic_affine_integrand_preservation;
mod inverse_hyperbolic_root_post_calculus_presentation;
mod inverse_reciprocal_trig_affine_abs_derivative_presentation;
mod inverse_reciprocal_trig_positive_quadratic_derivative_presentation;
mod inverse_reciprocal_trig_sqrt_derivative_presentation;
mod inverse_sqrt_product_integrand_presentation;
mod inverse_sqrt_product_integrand_preservation;
mod inverse_surd_quotient_derivative_presentation;
mod inverse_tangent_hyperbolic_rational_affine_derivative_presentation;
mod inverse_tangent_polynomial_root_derivative_presentation;
mod inverse_tangent_polynomial_root_post_calculus_presentation;
mod inverse_tangent_reciprocal_sqrt_derivative_presentation;
mod inverse_tangent_root_args;
mod inverse_tangent_scaled_root_derivative_presentation;
mod inverse_tangent_scaled_root_quotient_post_calculus_presentation;
mod inverse_tangent_trig_affine_derivative_presentation;
mod inverse_trig_derivative_presentation;
mod log_by_parts_integrand_presentation;
mod log_derivative_presentation;
mod log_product_integrand_preservation;
mod log_reciprocal_trig_sqrt_derivative_presentation;
mod log_root_derivative_presentation;
mod log_shifted_tan_sqrt_derivative_presentation;
mod log_sqrt_quotient_derivative_presentation;
mod polynomial_over_sqrt_derivative_presentation;
mod polynomial_power_presentation;
mod polynomial_support;
mod positive_quadratic_presentation;
mod post_calculus_presentation;
mod power_result_presentation;
mod presentation_compaction;
mod presentation_utils;
mod rational_partial_fraction_integrand_preservation;
mod rationalized_sqrt_result_presentation;
mod reciprocal_sqrt_product_derivative_presentation;
mod reciprocal_trig_derivative_presentation;
mod result_presentation;
mod result_preservation;
mod result_sensitive_post_calculus_presentation;
mod root_and_inverse_family_post_calculus_presentation;
mod scalar_presentation;
mod scaled_sqrt_args;
mod self_normalized_surd_quotient_derivative_presentation;
mod shifted_sqrt_args;
mod shifted_sqrt_derivative_presentation;
mod signed_factor_presentation;
mod sqrt_additive_result_presentation;
mod sqrt_additive_tan_derivative_presentation;
mod sqrt_additive_tan_result_presentation;
mod sqrt_additive_trig_derivative_presentation;
mod sqrt_chain_factor_presentation;
mod sqrt_chain_integrand_preservation;
mod sqrt_denominator_result_presentation;
mod sqrt_derivative_post_calculus_presentation;
mod sqrt_hyperbolic_log_integrand_presentation;
mod sqrt_polynomial_quotient_derivative_presentation;
mod sqrt_polynomial_scale_presentation;
mod sqrt_product_presentation;
mod sqrt_reciprocal_trig_antiderivative_presentation;
mod sqrt_reciprocal_trig_product_integrand_presentation;
mod sqrt_small_additive_derivative_presentation;
mod sqrt_trig_log_antiderivative_presentation;
mod sqrt_trig_log_integrand_presentation;
mod summation;
mod surd_quotient_args;
mod surd_quotient_presentation;
mod tanh_primitive_derivative_presentation;
mod trig_by_parts_integrand_presentation;
mod trig_power_integrand_presentation;
mod trig_result_presentation;
mod unary_function_presentation;

#[cfg(test)]
use cas_ast::{Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

use acosh_affine_derivative_presentation::{
    acosh_affine_derivative_presentation, constant_scaled_acosh_affine_derivative_presentation,
};
use acosh_over_sqrt_derivative_presentation::{
    acosh_polynomial_over_sqrt_derivative_presentation,
    constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation,
};
use acosh_sqrt_derivative_presentation::{
    acosh_sqrt_shifted_quadratic_derivative_presentation,
    constant_scaled_acosh_sqrt_polynomial_derivative_presentation,
};
use acosh_strictly_positive_polynomial_derivative_presentation::acosh_strictly_positive_polynomial_derivative_presentation;
#[cfg(test)]
use arctan_polynomial_integrand_presentation::polynomial_times_arctan_affine_integrand_for_diff_shortcut;
pub(crate) use arctan_sqrt_additive_derivative_presentation::{
    arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation_with_domain,
    arctan_sqrt_additive_tan_polynomial_derivative_presentation_with_domain,
    arctan_sqrt_additive_trig_polynomial_derivative_presentation_with_domain,
    arctan_sqrt_small_additive_elementary_derivative_presentation_with_domain,
};
use arctan_sqrt_constant_over_polynomial_presentation::arctan_sqrt_constant_over_polynomial_presentation;
use arctan_sqrt_positive_shift_derivative_presentation::{
    arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation,
    compact_sqrt_var_over_var_times_positive_shift_square_diff_result,
};
pub(crate) use arctan_sqrt_quotient_derivative_presentation::arctan_sqrt_positive_polynomial_quotient_derivative_for_diff_call;
use arctan_sqrt_quotient_derivative_presentation::{
    arctan_sqrt_positive_polynomial_quotient_derivative_shortcut,
    arctan_sqrt_variable_over_positive_affine_derivative_presentation,
    constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation,
};
use arctan_surd_derivative_presentation::{
    arctan_self_normalized_surd_reciprocal_compact_derivative,
    arctan_surd_quotient_compact_derivative, arctan_surd_quotient_scaled_compact_derivative,
    constant_scaled_arctan_surd_quotient_scaled_compact_derivative,
};
use asinh_sqrt_constant_over_polynomial_presentation::{
    asinh_sqrt_constant_over_polynomial_presentation,
    scaled_asinh_sqrt_constant_over_polynomial_presentation,
};
use asinh_sqrt_derivative_presentation::constant_scaled_asinh_sqrt_polynomial_derivative_presentation;
use asinh_surd_derivative_presentation::asinh_surd_quotient_compact_derivative;
use atanh_sqrt_constant_over_polynomial_presentation::atanh_sqrt_constant_over_polynomial_presentation;
use atanh_sqrt_derivative_presentation::constant_scaled_atanh_sqrt_polynomial_derivative_presentation;
use atanh_surd_derivative_presentation::atanh_surd_quotient_compact_derivative;
use bounded_inverse_trig_projection_presentation::bounded_inverse_trig_self_normalized_projection_derivative_presentation;
pub use diff_rule::DiffRule;
use diff_rule_support::{
    arctan_sqrt_additive_derivative_rewrite, diff_rewrite_with_conditions,
    sign_polynomial_diff_result, sqrt_additive_derivative_shortcut, undefined_diff_rewrite,
};
use differentiation::differentiate;
pub(crate) use domain_checks::diff_target_known_undefined_over_reals;
use domain_checks::{
    acosh_sqrt_diff_required_conditions, append_positive_required_conditions,
    diff_required_conditions_for_target, diff_target_known_undefined_or_empty_domain_over_reals,
};
use elementary_sqrt_derivative_presentation::signed_elementary_sqrt_polynomial_derivative_presentation;
use elementary_variable_term_presentation::{
    scaled_ln_variable_arg_for_calculus_presentation,
    scaled_sqrt_variable_term_for_calculus_presentation,
};
use exponential_derivative_presentation::{
    exp_trig_by_parts_primitive_derivative_presentation, sqrt_shifted_exp_derivative_presentation,
};
pub(crate) use hyperbolic_primitive_derivative_presentation::affine_hyperbolic_odd_primitive_derivative_presentation;
use integral_derivative_shortcut_presentation::supported_integral_diff_shortcut_presentation;
pub use integrate_rule::IntegrateRule;
#[cfg(test)]
use integration::integrate;
use inverse_reciprocal_trig_affine_abs_derivative_presentation::constant_scaled_inverse_reciprocal_trig_affine_abs_presentation;
pub(crate) use inverse_reciprocal_trig_positive_quadratic_derivative_presentation::inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation_with_domain;
use inverse_surd_quotient_derivative_presentation::{
    constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative,
    reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative,
};
use inverse_tangent_polynomial_root_derivative_presentation::negative_arccot_sqrt_polynomial_derivative_shortcut;
use inverse_tangent_reciprocal_sqrt_derivative_presentation::{
    arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut,
    inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut,
};
use inverse_tangent_scaled_root_derivative_presentation::{
    atanh_sqrt_over_symbolic_constant_derivative_shortcut,
    constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut,
    inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut,
    inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut,
};
use inverse_tangent_trig_affine_derivative_presentation::inverse_tangent_direct_trig_affine_derivative_presentation;
use inverse_trig_derivative_presentation::{
    bounded_inverse_trig_surd_quotient_compact_derivative,
    unit_interval_bounded_inverse_trig_derivative_presentation,
};
use log_reciprocal_trig_sqrt_derivative_presentation::ln_reciprocal_trig_sqrt_derivative_presentation;
pub(crate) use log_root_derivative_presentation::ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain;
use log_root_derivative_presentation::{
    ln_sqrt_plus_polynomial_direct_derivative_presentation,
    ln_sqrt_polynomial_gap_derivative_presentation,
    ln_sqrt_positive_shift_nonpolynomial_derivative_presentation,
    ln_sum_of_equal_derivative_roots_derivative_presentation,
    sqrt_shifted_ln_derivative_presentation,
};
use log_shifted_tan_sqrt_derivative_presentation::ln_constant_shifted_tan_sqrt_derivative_presentation;
use log_sqrt_quotient_derivative_presentation::{
    log_over_sqrt_polynomial_derivative_presentation,
    sqrt_over_log_polynomial_derivative_presentation,
};
use polynomial_over_sqrt_derivative_presentation::polynomial_over_sqrt_polynomial_derivative_presentation;
pub(crate) use polynomial_over_sqrt_derivative_presentation::polynomial_over_sqrt_polynomial_derivative_presentation_with_domain;
use polynomial_support::{
    polynomial_is_strictly_positive_everywhere, polynomial_radicand_for_calculus_presentation,
};
use positive_quadratic_presentation::{
    positive_quadratic_quotient_derivative_presentation,
    positive_quadratic_square_derivative_result_presentation,
};
pub(crate) use post_calculus_presentation::try_post_calculus_presentation;
use power_result_presentation::compact_positive_quadratic_square_derivative_result;
pub(crate) use presentation_compaction::compact_double_angle_sine_products_for_calculus_presentation;
use presentation_compaction::{
    bounded_sin_cos_shift_margin_for_calculus_presentation,
    compact_numeric_mul_factors_for_calculus_presentation,
    compact_small_power_exponents_for_calculus_presentation,
    distribute_half_over_additive_numerator_for_calculus_presentation,
};
use presentation_utils::unwrap_internal_hold_for_calculus;
pub(crate) use reciprocal_sqrt_product_derivative_presentation::reciprocal_sqrt_polynomial_product_derivative_presentation_with_domain;
use reciprocal_sqrt_product_derivative_presentation::{
    constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation,
    inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation,
    reciprocal_sqrt_polynomial_product_derivative_presentation,
};
pub(crate) use reciprocal_trig_derivative_presentation::reciprocal_trig_shifted_sqrt_derivative_presentation;
use reciprocal_trig_derivative_presentation::scaled_reciprocal_trig_power_derivative_presentation;
use result_presentation::arctan_affine_by_parts_compact_derivative;
#[cfg(test)]
use result_presentation::compact_arctan_additive_terms_for_calculus_presentation;
pub(crate) use result_presentation::try_calculus_result_presentation;
#[cfg(test)]
use scalar_presentation::fold_numeric_mul_constants_for_hold;
use scalar_presentation::{
    add_one_for_calculus_presentation, add_rational_for_calculus_presentation,
};
use self_normalized_surd_quotient_derivative_presentation::direct_self_normalized_surd_quotient_post_calculus_presentation;
pub(crate) use shifted_sqrt_derivative_presentation::sqrt_over_positive_shifted_sqrt_derivative_presentation_with_domain;
use shifted_sqrt_derivative_presentation::{
    reciprocal_positive_shifted_sqrt_derivative,
    reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative,
    sqrt_over_positive_shifted_sqrt_derivative,
};
pub(crate) use sqrt_additive_tan_derivative_presentation::{
    sqrt_additive_tan_polynomial_derivative_inline_presentation,
    sqrt_additive_tan_polynomial_derivative_presentation,
};
pub(crate) use sqrt_additive_trig_derivative_presentation::sqrt_additive_trig_polynomial_derivative_presentation;
use sqrt_derivative_post_calculus_presentation::{
    polynomial_times_sqrt_polynomial_derivative_presentation,
    sqrt_bounded_trig_positive_shift_derivative_presentation,
    sqrt_elementary_function_derivative_presentation,
};
#[cfg(test)]
use sqrt_hyperbolic_log_integrand_presentation::compact_direct_sqrt_hyperbolic_log_derivative_integrand;
use sqrt_polynomial_quotient_derivative_presentation::{
    sqrt_of_polynomial_quotient_derivative_presentation,
    sqrt_polynomial_quotient_derivative_presentation,
};
pub(crate) use sqrt_polynomial_quotient_derivative_presentation::{
    sqrt_of_polynomial_quotient_derivative_presentation_with_domain,
    sqrt_polynomial_quotient_derivative_presentation_with_domain,
    sqrt_polynomial_quotient_has_powered_expanded_affine_square_denominator,
};
use sqrt_product_presentation::shared_positive_content_sqrt_product_for_calculus_presentation;
pub(crate) use sqrt_small_additive_derivative_presentation::sqrt_small_additive_elementary_derivative_presentation_with_domain;
use sqrt_trig_log_antiderivative_presentation::sqrt_trig_log_antiderivative_derivative_presentation;
#[cfg(test)]
use sqrt_trig_log_integrand_presentation::{
    compact_direct_sqrt_trig_log_derivative_integrand, compact_sqrt_trig_log_derivative_integrand,
};
use tanh_primitive_derivative_presentation::affine_tanh_even_primitive_derivative_presentation;

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(summation::SumRule));
    simplifier.add_rule(Box::new(summation::ProductRule));
}

#[cfg(test)]
mod compact_hold_tests {
    use super::result_presentation::compact_sqrt_hyperbolic_reciprocal_for_integration_presentation;
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_cancels_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("-1*3/(3*cosh((3*x+1)^(1/2)))", &mut ctx).unwrap();
        let compact =
            compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(&mut ctx, expr, "x");
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "-1 / cosh(sqrt(3 * x + 1))");
    }

    #[test]
    fn arctan_surd_quotient_compact_derivative_avoids_rationalized_route() {
        let mut ctx = Context::new();
        let expr = parse("arctan((2*x+2)/sqrt(6))", &mut ctx).unwrap();
        let derivative = arctan_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "2 * sqrt(6) / ((2 * x + 2)^2 + 6)"
        );
    }

    #[test]
    fn arctan_affine_by_parts_compact_derivative_runs_in_diff_pipeline() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "diff(((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3, x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(rendered(&simplifier.context, result), "arctan(1 - x) * x^2");
    }

    #[test]
    fn integrate_pipeline_compacts_negative_affine_arctan_by_parts_result() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("integrate(x^2*arctan(1-x), x)", &mut simplifier.context).unwrap();
        let target = match simplifier.context.get(expr) {
            Expr::Function(_, args) => args[0],
            _ => expr,
        };
        assert!(
            cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
                &mut simplifier.context,
                target,
                "x",
            )
        );
        assert!(polynomial_times_arctan_affine_integrand_for_diff_shortcut(
            &simplifier.context,
            target,
            "x"
        ));
        let raw = integrate(&mut simplifier.context, target, "x").unwrap();
        let raw = fold_numeric_mul_constants_for_hold(&mut simplifier.context, raw);
        let compact = compact_arctan_additive_terms_for_calculus_presentation(
            &mut simplifier.context,
            raw,
            "x",
        )
        .unwrap();
        assert_eq!(
            rendered(&simplifier.context, compact)
                .matches("arctan(1 - x)")
                .count(),
            1
        );
        let (result, _steps) = simplifier.simplify(expr);
        let rendered = rendered(&simplifier.context, result);

        assert_eq!(rendered.matches("arctan(1 - x)").count(), 1);
    }

    #[test]
    fn constant_divisor_compact_derivative_accepts_asinh_surd_quotient() {
        let mut ctx = Context::new();
        let expr = parse("asinh((1-x-x^2)/sqrt(5))/sqrt(5)", &mut ctx).unwrap();
        let derivative = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "(-2 * x - 1) / (sqrt(5) * sqrt((1 - x - x^2)^2 + 5))"
        );
    }

    #[test]
    fn constant_divisor_compact_derivative_accepts_atanh_surd_quotient() {
        let mut ctx = Context::new();
        let expr = parse("atanh((x^2+x+1)/sqrt(7))/sqrt(7)", &mut ctx).unwrap();
        let derivative = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "(2 * x + 1) / (7 - (x^2 + x + 1)^2)"
        );
    }

    #[test]
    fn constant_scaled_acosh_affine_derivative_keeps_compact_roots() {
        let mut ctx = Context::new();
        let expr = parse("acosh(x+1)/2", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_acosh_affine_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (2 * sqrt(x) * sqrt(x + 2))"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(
            required_conditions[0].display(&ctx),
            "x > 0",
            "constant-scaled acosh shortcut must preserve the affine real-domain guard"
        );
    }

    #[test]
    fn post_diff_presentation_compacts_sqrt_over_var_shift_square() {
        let mut ctx = Context::new();
        let target = parse("arctan(sqrt(x)) + sqrt(x)/(x+1)", &mut ctx).unwrap();
        let (direct, _required) =
            arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(rendered(&ctx, direct), "1 / ((x + 1)^2 * sqrt(x))");

        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "diff(arctan(sqrt(x)) + sqrt(x)/(x+1), x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(
            rendered(&simplifier.context, result),
            "1 / ((x + 1)^2 * sqrt(x))"
        );

        let expr = parse(
            "diff(8*arctan(2*sqrt(x)) + 4*sqrt(x)/(x+1/4), x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(
            rendered(&simplifier.context, result),
            "1 / ((x + 1/4)^2 * sqrt(x))"
        );
    }

    #[test]
    fn ln_sqrt_affine_gap_derivative_keeps_compact_radicand() {
        let mut ctx = Context::new();
        let expr = parse("ln(sqrt((2*x+1)^2-4)+(2*x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_plus_polynomial_direct_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / sqrt((2 * x + 1)^2 - 4)");
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 1/2");

        let expr = parse("ln(sqrt((2*x+1)^2-4)-(2*x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_plus_polynomial_direct_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "-2 / sqrt((2 * x + 1)^2 - 4)");
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x < -3/2");
    }

    #[test]
    fn ln_sqrt_positive_shift_nonpolynomial_diff_uses_direct_denominator() {
        let mut ctx = Context::new();
        let target = parse("ln(1+sqrt(sin(x)+2))", &mut ctx).unwrap();
        let (derivative, conditions) =
            ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "cos(x) / (2 * sqrt(sin(x) + 2) * (sqrt(sin(x) + 2) + 1))"
        );
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].display(&ctx), "sin(x) + 2 > 0");
    }

    #[test]
    fn ln_sqrt_positive_shift_exp_diff_does_not_reintroduce_ln_e() {
        let mut ctx = Context::new();
        let target = parse("ln(1+sqrt(exp(x)+1))", &mut ctx).unwrap();
        let (derivative, conditions) =
            ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "e^x / (2 * sqrt(e^x + 1) * (sqrt(e^x + 1) + 1))"
        );
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].display(&ctx), "e^x + 1 > 0");
    }

    #[test]
    fn ln_reciprocal_trig_affine_sqrt_diff_uses_held_compact_derivative() {
        let mut ctx = Context::new();
        let target = parse("ln(sec(sqrt(3*x+1))+tan(sqrt(3*x+1)))", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_reciprocal_trig_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "3 / (2 * sqrt(3 * x + 1) * cos(sqrt(3 * x + 1)))"
        );
        assert_eq!(required_conditions.len(), 3);

        let target = parse("ln(csc(sqrt(3*x+1))-cot(sqrt(3*x+1)))", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_reciprocal_trig_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "3 / (2 * sqrt(3 * x + 1) * sin(sqrt(3 * x + 1)))"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn compact_direct_sqrt_trig_log_derivative_integrand_preserves_tangent_form() {
        let mut ctx = Context::new();
        let expr = parse("tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn ln_constant_shifted_tan_sqrt_diff_uses_held_compact_derivative() {
        let mut ctx = Context::new();
        let target = parse("ln(tan(sqrt(x))+1)", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_constant_shifted_tan_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "1 / (2 * sqrt(x) * cos(sqrt(x))^2 * (tan(sqrt(x)) + 1))"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn ln_constant_shifted_tan_affine_sqrt_diff_uses_held_compact_derivative() {
        let mut ctx = Context::new();
        let target = parse("ln(1+tan(sqrt(2*x+3)))", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_constant_shifted_tan_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "1 / (sqrt(2 * x + 3) * cos(sqrt(2 * x + 3))^2 * (tan(sqrt(2 * x + 3)) + 1))"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn compact_direct_sqrt_trig_log_derivative_integrand_accepts_mul_div_chain() {
        let mut ctx = Context::new();
        let expr = parse("tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_half_power_product() {
        let mut ctx = Context::new();
        let expr = parse("1/2*tanh(sqrt(x))*x^(-1/2)", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_tanh_denominator() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)/(2*x*tanh(sqrt(x)))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "1 / (2 * tanh(sqrt(x)) * sqrt(x))");
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_sinh_cosh_quotient() {
        let mut ctx = Context::new();
        let expr = parse("-sinh(sqrt(x))*x^(-1/2)/(2*cosh(sqrt(x)))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "-tanh(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_cosh_sinh_quotient() {
        let mut ctx = Context::new();
        let expr = parse("-cosh(sqrt(x))*x^(-1/2)/(2*sinh(sqrt(x)))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "-1 / (2 * tanh(sqrt(x)) * sqrt(x))"
        );
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_preserves_negative_tangent_form() {
        let mut ctx = Context::new();
        let inner = parse("sin(sqrt(x))*x^(-1/2)/(2*cos(sqrt(x)))", &mut ctx).unwrap();
        let expr = ctx.add(Expr::Neg(inner));
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "-tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_accepts_half_power_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "((3*x+1)^(-1/2) * sin((3*x+1)^(1/2)) * 3)/(2 * cos((3*x+1)^(1/2)))",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_accepts_scaled_radicand_denominator() {
        let mut ctx = Context::new();
        let expr = parse(
            "((3*x+1)^(1/2) * sin((3*x+1)^(1/2)) * 3)/(cos((3*x+1)^(1/2)) * (6*x+2))",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn sqrt_trig_log_antiderivative_derivative_presentation_compacts_shifted_chain() {
        let mut ctx = Context::new();
        let expr = parse("-ln(abs(cos(sqrt(3*x+1))))", &mut ctx).unwrap();
        let (compact, conditions) =
            sqrt_trig_log_antiderivative_derivative_presentation(&mut ctx, expr, "x").unwrap();
        let rendered_conditions: Vec<_> = conditions
            .iter()
            .map(|condition| match condition {
                crate::ImplicitCondition::Positive(expr) => {
                    format!("{} > 0", rendered(&ctx, *expr))
                }
                crate::ImplicitCondition::NonZero(expr) => {
                    format!("{} != 0", rendered(&ctx, *expr))
                }
                other => format!("{other:?}"),
            })
            .collect();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
        assert_eq!(
            rendered_conditions,
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) != 0"]
        );
    }
}
