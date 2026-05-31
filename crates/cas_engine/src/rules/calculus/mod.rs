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
mod arctan_integrand_preservation;
mod arctan_polynomial_integrand_presentation;
mod arctan_sqrt_constant_over_polynomial_presentation;
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
mod bounded_inverse_trig_projection_presentation;
mod bounded_inverse_trig_shifted_sqrt_derivative_presentation;
mod bounded_inverse_trig_sqrt_derivative_presentation;
mod by_parts_integrand_preservation;
mod derivative_integrand_factor_parts;
mod differentiation;
mod direct_trig_affine_integrand_presentation;
mod domain_checks;
mod elementary_sqrt_derivative_presentation;
mod exp_by_parts_integrand_presentation;
mod exponential_derivative_presentation;
mod fractional_denominator_power_integrand_preservation;
mod gap_presentation;
mod hyperbolic_by_parts_integrand_presentation;
mod hyperbolic_power_integrand_presentation;
mod hyperbolic_primitive_derivative_presentation;
mod integral_derivative_shortcut_presentation;
mod integration;
mod inverse_hyperbolic_affine_integrand_preservation;
mod inverse_reciprocal_trig_affine_abs_derivative_presentation;
mod inverse_reciprocal_trig_positive_quadratic_derivative_presentation;
mod inverse_reciprocal_trig_sqrt_derivative_presentation;
mod inverse_sqrt_product_integrand_presentation;
mod inverse_tangent_hyperbolic_rational_affine_derivative_presentation;
mod inverse_tangent_reciprocal_sqrt_derivative_presentation;
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
mod power_result_presentation;
mod presentation_utils;
mod rational_partial_fraction_integrand_preservation;
mod rationalized_sqrt_result_presentation;
mod reciprocal_sqrt_product_derivative_presentation;
mod reciprocal_trig_derivative_presentation;
mod result_presentation;
mod result_preservation;
mod scalar_presentation;
mod scaled_sqrt_args;
mod shifted_sqrt_args;
mod shifted_sqrt_derivative_presentation;
mod signed_factor_presentation;
mod sqrt_chain_factor_presentation;
mod sqrt_chain_integrand_preservation;
mod sqrt_denominator_result_presentation;
mod sqrt_hyperbolic_log_integrand_presentation;
mod sqrt_polynomial_quotient_derivative_presentation;
mod sqrt_polynomial_scale_presentation;
mod sqrt_product_presentation;
mod sqrt_reciprocal_trig_antiderivative_presentation;
mod sqrt_reciprocal_trig_product_integrand_presentation;
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

use crate::define_rule;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{
    render_diff_desc_with, try_extract_diff_call, try_extract_integrate_call, NamedVarCall,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

use acosh_affine_derivative_presentation::{
    acosh_affine_derivative_presentation, constant_scaled_acosh_affine_derivative_presentation,
};
use acosh_over_sqrt_derivative_presentation::{
    acosh_polynomial_over_sqrt_derivative_presentation,
    constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation,
};
use acosh_sqrt_derivative_presentation::{
    acosh_sqrt_polynomial_derivative_presentation,
    acosh_sqrt_shifted_quadratic_derivative_presentation,
    constant_scaled_acosh_sqrt_polynomial_derivative_presentation,
    scaled_acosh_sqrt_polynomial_derivative_presentation,
};
use acosh_strictly_positive_polynomial_derivative_presentation::acosh_strictly_positive_polynomial_derivative_presentation;
use arctan_polynomial_integrand_presentation::polynomial_times_arctan_affine_integrand_for_diff_shortcut;
use arctan_sqrt_constant_over_polynomial_presentation::{
    arccot_sqrt_constant_over_polynomial_presentation,
    arctan_sqrt_constant_over_polynomial_presentation,
};
pub(crate) use arctan_sqrt_quotient_derivative_presentation::arctan_sqrt_positive_polynomial_quotient_derivative_for_diff_call;
use arctan_sqrt_quotient_derivative_presentation::{
    arctan_sqrt_affine_partition_quotient_derivative_presentation,
    arctan_sqrt_polynomial_quotient_derivative_presentation,
    arctan_sqrt_positive_polynomial_quotient_derivative_shortcut,
};
use arctan_surd_derivative_presentation::{
    arctan_self_normalized_surd_quotient_compact_derivative,
    arctan_self_normalized_surd_reciprocal_compact_derivative,
    arctan_surd_quotient_compact_derivative, arctan_surd_quotient_scaled_compact_derivative,
    constant_scaled_arctan_surd_quotient_scaled_compact_derivative,
};
use asinh_polynomial_derivative_presentation::asinh_polynomial_derivative_presentation;
use asinh_sqrt_constant_over_polynomial_presentation::{
    asinh_sqrt_constant_over_polynomial_presentation,
    scaled_asinh_sqrt_constant_over_polynomial_presentation,
};
use asinh_sqrt_derivative_presentation::{
    asinh_sqrt_polynomial_derivative_presentation,
    constant_scaled_asinh_sqrt_polynomial_derivative_presentation,
    scaled_asinh_sqrt_polynomial_derivative_presentation,
};
use asinh_surd_derivative_presentation::asinh_surd_quotient_compact_derivative;
use atanh_sqrt_constant_over_polynomial_presentation::atanh_sqrt_constant_over_polynomial_presentation;
use atanh_sqrt_derivative_presentation::{
    atanh_sqrt_polynomial_derivative_presentation,
    constant_scaled_atanh_sqrt_polynomial_derivative_presentation,
    scaled_atanh_sqrt_polynomial_derivative_presentation,
};
use atanh_sqrt_quotient_derivative_presentation::atanh_sqrt_affine_quotient_positive_gap_presentation;
use atanh_surd_derivative_presentation::{
    atanh_self_normalized_surd_quotient_compact_derivative, atanh_surd_quotient_compact_derivative,
};
use bounded_inverse_trig_projection_presentation::bounded_inverse_trig_self_normalized_projection_derivative_presentation;
use bounded_inverse_trig_shifted_sqrt_derivative_presentation::{
    constant_scaled_unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation,
    unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation,
};
use bounded_inverse_trig_sqrt_derivative_presentation::{
    bounded_inverse_trig_reciprocal_sqrt_polynomial_derivative_presentation,
    bounded_inverse_trig_sqrt_affine_quotient_derivative_presentation,
    bounded_inverse_trig_sqrt_polynomial_derivative_presentation,
    constant_scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation,
    scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation,
};
use differentiation::differentiate;
#[cfg(test)]
use domain_checks::atanh_open_interval_condition;
pub(crate) use domain_checks::diff_target_known_undefined_over_reals;
use domain_checks::{
    acosh_sqrt_diff_required_conditions, bounded_inverse_trig_known_empty_open_interval_gap,
    diff_required_conditions_for_target, diff_target_known_undefined_or_empty_domain_over_reals,
    positive_polynomial_radicand_and_nonzero_required_conditions,
    positive_polynomial_radicand_required_conditions,
    reciprocal_trig_and_log_diff_required_conditions,
};
use elementary_sqrt_derivative_presentation::signed_elementary_sqrt_polynomial_derivative_presentation;
use exponential_derivative_presentation::{
    exp_trig_by_parts_primitive_derivative_presentation, sqrt_shifted_exp_derivative_presentation,
};
use gap_presentation::squared_expr_for_compact_gap_presentation;
pub(crate) use hyperbolic_primitive_derivative_presentation::affine_hyperbolic_odd_primitive_derivative_presentation;
use integral_derivative_shortcut_presentation::{
    supported_integral_derivative_presentation, supported_integral_diff_shortcut_presentation,
};
use integration::{integrate, integrate_rewrite_with_conditions, IntegrationRequiredConditions};
use inverse_reciprocal_trig_affine_abs_derivative_presentation::{
    constant_scaled_inverse_reciprocal_trig_affine_abs_presentation,
    inverse_reciprocal_trig_affine_abs_presentation,
};
pub(crate) use inverse_reciprocal_trig_positive_quadratic_derivative_presentation::inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation_with_domain;
use inverse_reciprocal_trig_positive_quadratic_derivative_presentation::{
    inverse_reciprocal_trig_positive_quadratic_presentation,
    inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation,
};
use inverse_reciprocal_trig_sqrt_derivative_presentation::{
    inverse_reciprocal_trig_sqrt_affine_derivative_presentation,
    inverse_reciprocal_trig_sqrt_quadratic_derivative_presentation,
};
use inverse_tangent_hyperbolic_rational_affine_derivative_presentation::{
    arctan_rational_affine_derivative_presentation, atanh_rational_affine_derivative_presentation,
};
use inverse_tangent_reciprocal_sqrt_derivative_presentation::{
    arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut,
    arctan_sqrt_reciprocal_content_presentation,
    inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation,
    inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut,
};
use inverse_tangent_trig_affine_derivative_presentation::inverse_tangent_direct_trig_affine_derivative_presentation;
use inverse_trig_derivative_presentation::{
    bounded_inverse_trig_polynomial_derivative_presentation,
    bounded_inverse_trig_surd_quotient_compact_derivative,
    unit_interval_bounded_inverse_trig_derivative_presentation,
};
use log_derivative_presentation::{
    ln_power_derivative_numeric_presentation, variable_base_constant_argument_log_presentation,
};
use log_reciprocal_trig_sqrt_derivative_presentation::ln_reciprocal_trig_sqrt_derivative_presentation;
pub(crate) use log_root_derivative_presentation::ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain;
use log_root_derivative_presentation::{
    ln_sqrt_negative_polynomial_gap_target, ln_sqrt_plus_polynomial_direct_derivative_presentation,
    ln_sqrt_polynomial_gap_derivative_presentation,
    ln_sqrt_positive_shift_nonpolynomial_derivative_presentation,
    ln_sqrt_shift_derivative_presentation,
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
    nonzero_affine_variable_derivative, polynomial_is_strictly_positive_everywhere,
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use positive_quadratic_presentation::{
    inverse_reciprocal_trig_positive_quadratic_square_presentation,
    positive_quadratic_quotient_derivative_presentation,
    positive_quadratic_square_derivative_result_presentation,
};
use power_result_presentation::{
    compact_negative_half_power_result_for_integration_presentation,
    compact_positive_quadratic_square_derivative_result,
};
use presentation_utils::{
    calculus_sqrt_like_radicand, is_half_power_exponent,
    scaled_sqrt_argument_for_calculus_presentation, small_rational_const_for_calculus_presentation,
    sqrt_raw_for_calculus_presentation, unwrap_internal_hold_for_calculus, variable_named,
};
pub(crate) use reciprocal_sqrt_product_derivative_presentation::reciprocal_sqrt_polynomial_product_derivative_presentation_with_domain;
use reciprocal_sqrt_product_derivative_presentation::{
    inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation,
    reciprocal_sqrt_polynomial_product_derivative_presentation,
};
pub(crate) use reciprocal_trig_derivative_presentation::reciprocal_trig_shifted_sqrt_derivative_presentation;
use reciprocal_trig_derivative_presentation::{
    reciprocal_trig_affine_derivative_presentation,
    scaled_reciprocal_trig_power_derivative_presentation,
};
pub(crate) use result_presentation::try_calculus_result_presentation;
use result_presentation::{
    apply_integration_final_presentation, arctan_affine_by_parts_compact_derivative,
    cancel_denominator_content_with_numerator_for_calculus_presentation,
    compact_arctan_additive_terms_for_calculus_presentation,
    divide_compact_derivative_by_constant_factor,
    reciprocal_constant_denominator_for_calculus_presentation,
    remove_unit_mul_factors_for_calculus_presentation, scale_compact_derivative_by_rational,
    try_diff_integral_source_post_calculus_presentation, try_integrate_post_calculus_presentation,
};
#[cfg(test)]
use result_presentation::{
    compact_acosh_surd_width_arg_for_integration_presentation,
    compact_positive_cosh_log_abs_for_integration_presentation,
};
use result_preservation::{
    apply_integration_result_preservation, integration_source_preservation_gates,
};
#[cfg(test)]
use scalar_presentation::fold_numeric_mul_constants_for_hold;
use scalar_presentation::{
    add_one_for_calculus_presentation,
    add_rational_combining_additive_constant_for_calculus_presentation,
    add_rational_for_calculus_presentation, exact_positive_rational_sqrt_for_calculus_presentation,
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    rational_scaled_single_factor, rational_scaled_single_factor_allow_unit,
    scale_compact_fraction_numerator_by_rational_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
    signed_rational_const_for_calculus_presentation,
    split_outer_numeric_mul_for_calculus_presentation, subtract_expr_for_calculus_presentation,
};
use scaled_sqrt_args::{
    inverse_tangent_sqrt_over_symbolic_constant_arg_for_calculus_presentation,
    scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation,
    scaled_sqrt_polynomial_arg_for_calculus_presentation,
};
pub(crate) use shifted_sqrt_derivative_presentation::sqrt_over_positive_shifted_sqrt_derivative_presentation_with_domain;
use shifted_sqrt_derivative_presentation::{
    inverse_tangent_reciprocal_sqrt_shifted_sqrt_product_derivative_presentation,
    reciprocal_positive_shifted_sqrt_derivative,
    reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative,
    sqrt_over_positive_shifted_sqrt_derivative,
};
use sqrt_hyperbolic_log_integrand_presentation::{
    compact_direct_sqrt_hyperbolic_log_derivative_integrand, sqrt_cosh_log_derivative_presentation,
};
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
use sqrt_trig_log_antiderivative_presentation::sqrt_trig_log_antiderivative_derivative_presentation;
#[cfg(test)]
use sqrt_trig_log_integrand_presentation::{
    compact_direct_sqrt_trig_log_derivative_integrand, compact_sqrt_trig_log_derivative_integrand,
};
use tanh_primitive_derivative_presentation::affine_tanh_even_primitive_derivative_presentation;

fn positive_scaled_variable_factor(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if variable_named(ctx, target, var_name) {
        return Some(BigRational::one());
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut saw_variable = false;
    for factor in factors {
        if variable_named(ctx, factor, var_name) {
            if saw_variable {
                return None;
            }
            saw_variable = true;
            continue;
        }

        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        if !value.is_positive() {
            return None;
        }
        scale *= value;
    }

    saw_variable.then_some(scale)
}

fn arctan_sqrt_scaled_variable_arg(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let scale = positive_scaled_variable_factor(ctx, radicand, var_name)
        .or_else(|| nonzero_affine_variable_derivative(ctx, radicand, var_name))?;
    Some((radicand, scale))
}

pub(super) fn arctan_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }

    let radicand = match ctx.get(args[0]) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            sqrt_args[0]
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => *base,
        _ => return None,
    };

    Some(radicand)
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, offset) =
        if let Some(parts) = arctan_sqrt_positive_shift_primitive_parts(ctx, target, var_name) {
            (parts.primitive_scale, parts.offset)
        } else {
            (
                arctan_sqrt_plus_sqrt_over_x_plus_one_scale(ctx, target, var_name)?,
                BigRational::one(),
            )
        };
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&scale)?;

    let var = ctx.var(var_name);
    let neg_half = ctx.rational(-1, 2);
    let reciprocal_sqrt_var = ctx.add(Expr::Pow(var, neg_half));
    let offset_expr = rational_const_for_calculus_presentation(ctx, offset);
    let unit_shift = ctx.add(Expr::Add(var, offset_expr));
    let two = ctx.num(2);
    let unit_shift_square = ctx.add(Expr::Pow(unit_shift, two));
    let numerator = if numerator_coeff == BigRational::one() {
        reciprocal_sqrt_var
    } else {
        let numerator_coeff = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        ctx.add_raw(Expr::Mul(numerator_coeff, reciprocal_sqrt_var))
    };
    let denominator = if denominator_coeff == BigRational::one() {
        unit_shift_square
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add_raw(Expr::Mul(denominator_coeff, unit_shift_square))
    };
    let result = ctx.add_raw(Expr::Div(numerator, denominator));
    let var = ctx.var(var_name);
    Some((result, vec![crate::ImplicitCondition::Positive(var)]))
}

fn arctan_sqrt_variable_over_positive_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Arctan | BuiltinFn::Atan)
        )
    {
        return None;
    }

    let (argument_scale, numerator, denominator) = match ctx.get(args[0]).clone() {
        Expr::Div(numerator, denominator) => (BigRational::one(), numerator, denominator),
        _ => {
            let (argument_scale, quotient_core) = split_numeric_scale_single_core(ctx, args[0])?;
            if !argument_scale.is_positive() {
                return None;
            }
            let Expr::Div(numerator, denominator) = ctx.get(quotient_core).clone() else {
                return None;
            };
            (argument_scale, numerator, denominator)
        }
    };
    let (numerator_scale, numerator_core) = split_numeric_scale_single_core(ctx, numerator)?;
    let numerator_scale = argument_scale * numerator_scale;
    if !numerator_scale.is_positive() {
        return None;
    }
    let radicand = extract_square_root_base(ctx, numerator_core)?;
    let radicand_scale = positive_scaled_variable_factor(ctx, radicand, var_name)?;
    let numerator_derivative_scale = numerator_scale.clone() * radicand_scale.clone();
    let denominator_variable_scale =
        numerator_scale.clone() * numerator_scale * radicand_scale.clone();

    let denominator_poly = Polynomial::from_expr(ctx, denominator, var_name).ok()?;
    if denominator_poly.degree() != 1 {
        return None;
    }
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_slope = denominator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !denominator_slope.is_positive() || denominator_constant.is_negative() {
        return None;
    }

    let var_poly = Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        var_name.to_string(),
    );
    let scaled_var_poly = Polynomial::new(
        vec![BigRational::zero(), denominator_variable_scale],
        var_name.to_string(),
    );
    let scale_poly = Polynomial::new(vec![numerator_derivative_scale], var_name.to_string());
    let two = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let mut numerator_poly = denominator_poly
        .sub(&var_poly.mul(&denominator_poly.derivative()).mul(&two))
        .mul(&scale_poly);
    if numerator_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let mut denominator_coeff = BigRational::from_integer(2.into());
    let numerator_content = numerator_poly.content();
    if numerator_content.is_positive() && numerator_content.denom().is_one() {
        let common_integer = numerator_content
            .numer()
            .gcd(denominator_coeff.numer())
            .abs();
        if common_integer > BigInt::from(1) {
            let common = BigRational::from_integer(common_integer);
            numerator_poly = numerator_poly.div_scalar(&common);
            denominator_coeff /= common;
        }
    }

    let denominator_sum_poly = denominator_poly
        .mul(&denominator_poly)
        .add(&scaled_var_poly);
    let denominator_sum = denominator_sum_poly.to_expr(ctx);
    let numerator_expr = numerator_poly.to_expr(ctx);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(sqrt_var);
    denominator_factors.push(denominator_sum);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let result = ctx.add(Expr::Div(numerator_expr, denominator));
    let var = ctx.var(var_name);

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        vec![crate::ImplicitCondition::Positive(var)],
    ))
}

fn constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        let denominator_scale = cas_ast::views::as_rational_const(ctx, outer_den, 8)?;
        if denominator_scale.is_zero() {
            return None;
        }
        let (derivative, required_conditions) =
            arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                ctx, inner, var_name,
            )?;
        let scaled = scale_compact_derivative_by_rational(
            ctx,
            derivative,
            BigRational::one() / denominator_scale,
        );
        return Some((cas_ast::hold::wrap_hold(ctx, scaled), required_conditions));
    }

    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let (derivative, required_conditions) =
        arctan_sqrt_variable_over_positive_affine_derivative_presentation(ctx, inner, var_name)?;
    let scaled = scale_compact_derivative_by_rational(ctx, derivative, scale);
    Some((cas_ast::hold::wrap_hold(ctx, scaled), required_conditions))
}

fn compact_sqrt_var_over_var_times_positive_shift_square_diff_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (num_scale, shift_square, denominator_scale) =
        sqrt_var_over_var_times_positive_shift_square_parts(ctx, expr, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    let coefficient = num_scale / denominator_scale;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let var = ctx.var(var_name);
    let neg_half = ctx.rational(-1, 2);
    let reciprocal_sqrt_var = ctx.add(Expr::Pow(var, neg_half));
    let numerator = if numerator_coeff == BigRational::one() {
        reciprocal_sqrt_var
    } else {
        let numerator_coeff = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        ctx.add_raw(Expr::Mul(numerator_coeff, reciprocal_sqrt_var))
    };
    let denominator = if denominator_coeff == BigRational::one() {
        shift_square
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add_raw(Expr::Mul(denominator_coeff, shift_square))
    };
    let result = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(result)
}

fn sqrt_var_over_var_times_positive_shift_square_parts(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BigRational)> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
        let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
        if !is_calculus_var(ctx, radicand, var_name) {
            return None;
        }
        let (shift_square, denominator_scale) =
            var_times_positive_shift_square_denominator_parts(ctx, den, var_name)?;
        return Some((num_scale, shift_square, denominator_scale));
    }

    let mut num_scale = BigRational::one();
    let mut saw_sqrt_var = false;
    let mut denominator_factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            num_scale *= value;
        } else if calculus_sqrt_like_radicand(ctx, factor)
            .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        {
            if saw_sqrt_var {
                return None;
            }
            saw_sqrt_var = true;
        } else if let Expr::Pow(base, exp) = ctx.get(factor) {
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                != Some(BigRational::new((-1).into(), 1.into()))
            {
                return None;
            }
            denominator_factors.push(*base);
        } else {
            return None;
        }
    }

    if !saw_sqrt_var || denominator_factors.is_empty() {
        return None;
    }
    let denominator_factors = denominator_factors
        .into_iter()
        .flat_map(|factor| cas_math::expr_nary::mul_leaves(ctx, factor))
        .collect::<Vec<_>>();
    let (shift_square, denominator_scale) =
        var_times_positive_shift_square_denominator_factor_parts(
            ctx,
            denominator_factors,
            var_name,
        )?;
    Some((num_scale, shift_square, denominator_scale))
}

fn var_times_positive_shift_square_denominator_parts(
    ctx: &Context,
    den: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    var_times_positive_shift_square_denominator_factor_parts(
        ctx,
        cas_math::expr_nary::mul_leaves(ctx, den).to_vec(),
        var_name,
    )
}

fn var_times_positive_shift_square_denominator_factor_parts(
    ctx: &Context,
    factors: Vec<ExprId>,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    let mut denominator_scale = BigRational::one();
    let mut saw_var_factor = false;
    let mut shift_square = None;
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            denominator_scale *= value;
        } else if is_calculus_var(ctx, factor, var_name) {
            if saw_var_factor {
                return None;
            }
            saw_var_factor = true;
        } else if positive_shift_square_factor_for_calculus_presentation(ctx, factor, var_name)
            .is_some()
        {
            if shift_square.replace(factor).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    saw_var_factor.then_some((shift_square?, denominator_scale))
}

fn positive_shift_square_factor_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if cas_ast::views::as_rational_const(ctx, *exp, 8) != Some(BigRational::from_integer(2.into()))
    {
        return None;
    }
    positive_shift_denominator_scale(ctx, *base, var_name).map(|(_, offset)| offset)
}

struct ArctanSqrtPositiveShiftPrimitiveParts {
    primitive_scale: BigRational,
    offset: BigRational,
}

fn arctan_sqrt_positive_shift_primitive_parts(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtPositiveShiftPrimitiveParts> {
    if let Some((outer_scale, core)) = scaled_nontrivial_core_for_calculus_presentation(ctx, target)
    {
        let inner = arctan_sqrt_positive_shift_primitive_parts(ctx, core, var_name)?;
        return Some(ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: outer_scale * inner.primitive_scale,
            offset: inner.offset,
        });
    }

    if let Some(parts) = arctan_sqrt_positive_shift_combined_quotient_parts(ctx, target, var_name) {
        return Some(parts);
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() != 2 {
        return None;
    }

    let mut arctan_scale_and_offset = None;
    let mut quotient_scale_and_offset = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(parts) = scaled_arctan_sqrt_positive_shift_term(ctx, term, var_name) {
            if arctan_scale_and_offset.replace(parts).is_some() {
                return None;
            }
        } else if let Some(parts) = scaled_sqrt_var_over_positive_shift_term(ctx, term, var_name) {
            if quotient_scale_and_offset.replace(parts).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let (arctan_scale, offset, offset_root) = arctan_scale_and_offset?;
    let (quotient_scale, quotient_offset) = quotient_scale_and_offset?;
    if offset != quotient_offset {
        return None;
    }

    let primitive_scale_from_arctan = arctan_scale * offset.clone() * offset_root;
    let primitive_scale_from_quotient = quotient_scale * offset.clone();
    (primitive_scale_from_arctan == primitive_scale_from_quotient).then_some(
        ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: primitive_scale_from_arctan,
            offset,
        },
    )
}

fn arctan_sqrt_positive_shift_combined_quotient_parts(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtPositiveShiftPrimitiveParts> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let (den_scale, denominator_offset) = positive_shift_denominator_scale(ctx, den, var_name)?;
    if den_scale.is_zero() {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, num);
    if terms.len() != 2 {
        return None;
    }

    let mut sqrt_scale = None;
    let mut arctan_linear = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_sqrt_var_term(ctx, term, var_name) {
            if sqrt_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(parts) =
            scaled_positive_shift_times_arctan_sqrt_term(ctx, term, var_name)
        {
            if arctan_linear.replace(parts).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let quotient_offset = denominator_offset;
    let quotient_scale = sqrt_scale? / den_scale.clone();
    let (arctan_linear_scale, arctan_offset, arctan_offset_root) = arctan_linear?;
    if quotient_offset != arctan_offset {
        return None;
    }
    let arctan_scale = arctan_linear_scale / den_scale;
    let primitive_scale_from_quotient = quotient_scale * quotient_offset.clone();
    let primitive_scale_from_arctan = arctan_scale * quotient_offset.clone() * arctan_offset_root;
    (primitive_scale_from_quotient == primitive_scale_from_arctan).then_some(
        ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: primitive_scale_from_quotient,
            offset: quotient_offset,
        },
    )
}

fn scaled_positive_shift_times_arctan_sqrt_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational, BigRational)> {
    let mut scale = BigRational::one();
    let mut offset = None;
    let mut offset_root = None;
    let mut saw_arctan = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if let Some((factor_offset, factor_root)) =
            arctan_sqrt_positive_shift_arg(ctx, factor, var_name)
        {
            if saw_arctan {
                return None;
            }
            saw_arctan = true;
            offset_root = Some(factor_root);
            match &offset {
                Some(existing) if existing != &factor_offset => return None,
                Some(_) => {}
                None => offset = Some(factor_offset),
            }
        } else if let Some((linear_scale, linear_offset)) =
            positive_shift_denominator_scale(ctx, factor, var_name)
        {
            match &offset {
                Some(existing) if existing != &linear_offset => return None,
                Some(_) => {}
                None => offset = Some(linear_offset),
            }
            scale *= linear_scale;
        } else {
            return None;
        }
    }
    let offset = offset?;
    let offset_root = offset_root?;
    saw_arctan.then_some((scale, offset, offset_root))
}

fn scaled_arctan_sqrt_positive_shift_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational, BigRational)> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let (offset, offset_root) = arctan_sqrt_positive_shift_arg(ctx, core, var_name)?;
    Some((scale, offset, offset_root))
}

fn arctan_sqrt_positive_shift_arg(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }
    let (sqrt_scale, sqrt_core) = split_numeric_scale_single_core(ctx, args[0])?;
    if !sqrt_scale.is_positive() {
        return None;
    }
    let radicand = calculus_sqrt_like_radicand(ctx, sqrt_core)?;
    if !is_calculus_var(ctx, radicand, var_name) {
        return None;
    }
    let offset_root = BigRational::one() / sqrt_scale;
    let offset = offset_root.clone() * offset_root.clone();
    Some((offset, offset_root))
}

fn scaled_sqrt_var_over_positive_shift_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let (outer_scale, core) = split_outer_numeric_mul_for_calculus_presentation(ctx, expr)?;
    let core = unwrap_internal_hold_for_calculus(ctx, core);
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
    let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
    if !is_calculus_var(ctx, radicand, var_name) {
        return None;
    }
    let (den_scale, offset) = positive_shift_denominator_scale(ctx, den, var_name)?;
    Some((outer_scale * num_scale / den_scale, offset))
}

fn positive_shift_denominator_scale(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut scale = BigRational::one();
    let mut offset = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let constant = poly.coeffs.first()?.clone();
        let slope = poly.coeffs.get(1)?;
        if !slope.is_positive() {
            return None;
        }
        let candidate_offset = constant / slope.clone();
        if !candidate_offset.is_positive() {
            return None;
        }
        scale *= slope.clone();
        if offset.replace(candidate_offset).is_some() {
            return None;
        }
    }
    Some((scale, offset?))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_scale(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if let Some((outer_scale, core)) = scaled_nontrivial_core_for_calculus_presentation(ctx, target)
    {
        if let Some(inner_scale) = arctan_sqrt_plus_sqrt_over_x_plus_one_scale(ctx, core, var_name)
        {
            return Some(outer_scale * inner_scale);
        }
    }

    if let Some(scale) =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale(ctx, target, var_name)
    {
        return Some(scale);
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() != 2 {
        return None;
    }

    let mut arctan_scale = None;
    let mut rational_scale = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_arctan_sqrt_var_term(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_over_x_plus_one_term(ctx, term, var_name) {
            if rational_scale.replace(scale).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let arctan_scale = arctan_scale?;
    let rational_scale = rational_scale?;
    (arctan_scale == rational_scale).then_some(arctan_scale)
}

fn scaled_nontrivial_core_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    (!scale.is_one() && core != expr).then_some((scale, core))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let den_scale = x_plus_one_linear_scale_for_calculus_presentation(ctx, den, var_name)?;
    if den_scale.is_zero() {
        return None;
    }
    let num_scale =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale(ctx, num, var_name)?;
    Some(num_scale / den_scale)
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale(
    ctx: &mut Context,
    numerator: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, numerator);
    if terms.len() != 3 {
        return None;
    }

    let mut arctan_scale = None;
    let mut sqrt_scale = None;
    let mut x_arctan_scale = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_arctan_sqrt_var_term(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_term(ctx, term, var_name) {
            if sqrt_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_var_times_arctan_sqrt_var_term(ctx, term, var_name) {
            if x_arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let arctan_scale = arctan_scale?;
    (sqrt_scale? == arctan_scale && x_arctan_scale? == arctan_scale).then_some(arctan_scale)
}

fn scaled_arctan_sqrt_var_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    arctan_sqrt_radicand_arg(ctx, core)
        .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        .then_some(scale)
}

fn scaled_sqrt_var_term(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<BigRational> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let radicand = calculus_sqrt_like_radicand(ctx, core)?;
    is_calculus_var(ctx, radicand, var_name).then_some(scale)
}

fn scaled_var_times_arctan_sqrt_var_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let mut scale = BigRational::one();
    let mut saw_var = false;
    let mut saw_arctan = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if is_calculus_var(ctx, factor, var_name) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if arctan_sqrt_radicand_arg(ctx, factor)
            .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        {
            if saw_arctan {
                return None;
            }
            saw_arctan = true;
        } else {
            return None;
        }
    }
    (saw_var && saw_arctan).then_some(scale)
}

fn scaled_sqrt_var_over_x_plus_one_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let (outer_scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let core = unwrap_internal_hold_for_calculus(ctx, core);
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    if !is_x_plus_one_for_calculus_presentation(ctx, den, var_name) {
        return None;
    }

    let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
    let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
    is_calculus_var(ctx, radicand, var_name).then_some(outer_scale * num_scale)
}

pub(super) fn split_numeric_scale_single_core(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if den_scale.is_zero() {
            return None;
        }
        let (num_scale, core) = split_numeric_scale_single_core(ctx, num)?;
        return Some((num_scale / den_scale, core));
    }
    let mut scale = BigRational::one();
    let mut core = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if core.replace(factor).is_some() {
            return None;
        }
    }
    Some((scale, core.unwrap_or(expr)))
}

fn x_plus_one_linear_scale_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let offset = poly.coeffs.first()?;
    let slope = poly.coeffs.get(1)?;
    (offset == slope).then_some(offset.clone())
}

fn is_calculus_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

fn is_x_plus_one_for_calculus_presentation(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok();
    poly.is_some_and(|poly| {
        poly.degree() == 1
            && poly
                .coeffs
                .first()
                .is_some_and(|offset| offset == &BigRational::one())
            && poly
                .coeffs
                .get(1)
                .is_some_and(|slope| slope == &BigRational::one())
    })
}

fn inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if args.len() != 1 {
        return None;
    }
    let expected_sign = match builtin {
        BuiltinFn::Atan | BuiltinFn::Arctan => BigRational::one(),
        BuiltinFn::Acot | BuiltinFn::Arccot => -BigRational::one(),
        _ => return None,
    };
    if derivative_sign != expected_sign {
        return None;
    }

    let (radicand, radicand_poly, sqrt_scale) = if let Some(parts) =
        scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, args[0], var_name)
    {
        parts
    } else {
        let (radicand, sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, args[0])?;
        let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        (radicand, radicand_poly, sqrt_scale)
    };
    if sqrt_scale.is_zero() || sqrt_scale.is_one() {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * sqrt_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    let radicand_gap = add_one_for_calculus_presentation(ctx, scaled_radicand);
    let (numerator_coeff, denominator_coeff, radicand_gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_gap,
        );
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_coeff / denominator_coeff))?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (radicand, scale_denominator, argument_sign, sqrt_scale) =
        inverse_tangent_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
            ctx, args[0], var_name,
        )?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * argument_sign
        * sqrt_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = if derivative_core_is_one {
        scale_denominator
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[scale_denominator, derivative_core])
    };

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let scale_square = squared_expr_for_compact_gap_presentation(ctx, scale_denominator);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    let denominator_gap = ctx.add(Expr::Add(scale_square, scaled_radicand));
    let (numerator_coeff, denominator_coeff, denominator_gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        );
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_coeff / denominator_coeff))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(_, args) = ctx.get(target).clone() else {
        return None;
    };
    let [arg] = args.as_slice() else {
        return None;
    };

    let (radicand, scale_denominator, _, _) =
        inverse_tangent_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
            ctx, *arg, var_name,
        )?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result =
        inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(ctx, target, var_name)?;

    let required_conditions = positive_polynomial_radicand_and_nonzero_required_conditions(
        radicand,
        &radicand_poly,
        scale_denominator,
    );

    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

fn atanh_sqrt_over_symbolic_constant_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let (radicand, scale_denominator, argument_sign, sqrt_scale) =
        scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(ctx, args[0], var_name)?;
    if sqrt_scale.abs().is_one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = argument_sign
        * sqrt_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = if derivative_core_is_one {
        scale_denominator
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[scale_denominator, derivative_core])
    };

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let scale_square = squared_expr_for_compact_gap_presentation(ctx, scale_denominator);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    let denominator_gap =
        subtract_expr_for_calculus_presentation(ctx, scale_square, scaled_radicand);
    let (numerator_coeff, denominator_coeff, denominator_gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        );
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_coeff / denominator_coeff))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn atanh_sqrt_over_symbolic_constant_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(_, args) = ctx.get(target).clone() else {
        return None;
    };
    let [arg] = args.as_slice() else {
        return None;
    };

    let (radicand, scale_denominator, _, sqrt_scale) =
        scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(ctx, *arg, var_name)?;
    if sqrt_scale.abs().is_one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result = atanh_sqrt_over_symbolic_constant_derivative_presentation(ctx, target, var_name)?;
    let required_conditions = positive_polynomial_radicand_and_nonzero_required_conditions(
        radicand,
        &radicand_poly,
        scale_denominator,
    );

    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

fn constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let (derivative, required_conditions) =
        inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(ctx, inner, var_name)?;
    let derivative = scale_compact_fraction_numerator_by_rational_for_calculus_presentation(
        ctx, derivative, scale,
    );

    Some((ctx.add(Expr::Hold(derivative)), required_conditions))
}

fn constant_scaled_inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        inner,
        var_name,
        BigRational::one(),
    )
    .or_else(|| {
        inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            inner,
            var_name,
            -BigRational::one(),
        )
    })?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let arg = match ctx.get(target).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan | BuiltinFn::Acot | BuiltinFn::Arccot)
                ) =>
        {
            args[0]
        }
        _ => return None,
    };
    let (radicand, radicand_poly, sqrt_scale) = if let Some(parts) =
        scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, arg, var_name)
    {
        parts
    } else {
        let (radicand, sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, arg)?;
        let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        (radicand, radicand_poly, sqrt_scale)
    };
    if sqrt_scale.is_zero() || sqrt_scale.abs() == BigRational::one() {
        return None;
    }

    let result = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    )
    .or_else(|| {
        inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            target,
            var_name,
            -BigRational::one(),
        )
    })?;
    let required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, &radicand_poly);
    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

pub(super) fn arccot_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }

    let radicand = match ctx.get(args[0]) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            sqrt_args[0]
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => *base,
        _ => return None,
    };

    Some(radicand)
}

fn affine_square_root_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 2 {
        return None;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let linear_coeff = exact_positive_rational_sqrt_for_calculus_presentation(&a)?;
    let constant_abs = if c.is_zero() {
        BigRational::zero()
    } else {
        exact_positive_rational_sqrt_for_calculus_presentation(&c)?
    };
    let expected_cross =
        BigRational::from_integer(2.into()) * linear_coeff.clone() * constant_abs.clone();
    let constant = if b == expected_cross {
        constant_abs
    } else if b == -expected_cross {
        -constant_abs
    } else {
        return None;
    };

    let affine = Polynomial::new(vec![constant, linear_coeff], var_name.to_string());
    Some(affine.to_expr(ctx))
}

fn compact_squared_affine_gap_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let Expr::Sub(left, right) = ctx.get(expr).clone() else {
        return expr;
    };
    let Expr::Pow(base, exp) = ctx.get(right).clone() else {
        return expr;
    };
    if cas_ast::views::as_rational_const(ctx, exp, 8) != Some(BigRational::from_integer(2.into())) {
        return expr;
    }
    if let Expr::Pow(_, inner_exp) = ctx.get(base).clone() {
        if cas_ast::views::as_rational_const(ctx, inner_exp, 8)
            == Some(BigRational::from_integer(2.into()))
        {
            return expr;
        }
    }

    let Some(affine) = affine_square_root_for_calculus_presentation(ctx, base, var_name) else {
        return expr;
    };
    let four = ctx.num(4);
    let compact_power = ctx.add(Expr::Pow(affine, four));
    ctx.add(Expr::Sub(left, compact_power))
}

fn arctan_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if let Some(compact) = arctan_sqrt_reciprocal_content_presentation(
        ctx,
        radicand,
        &radicand_poly,
        derivative_sign.clone(),
    ) {
        return Some(compact);
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn arctan_sqrt_additive_tan_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_tan_polynomial_derivative_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

fn arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

fn arctan_sqrt_additive_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_trig_polynomial_derivative_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_rational_combining_additive_constant_for_calculus_presentation(
        ctx,
        radicand,
        BigRational::one(),
    );
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

fn arctan_sqrt_small_additive_elementary_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let (radicand_derivative, derivative_denominator, required_conditions) =
        small_additive_elementary_radicand_derivative_for_calculus_presentation(
            ctx, radicand, var_name,
        )?;

    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let mut denominator_factors = vec![two];
    if let Some(derivative_denominator) = derivative_denominator {
        denominator_factors.push(derivative_denominator);
    }
    denominator_factors.push(sqrt_radicand);
    denominator_factors.push(radicand_plus_one);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let denominator = compact_numeric_mul_factors_for_calculus_presentation(ctx, denominator);
    let compact = ctx.add_raw(Expr::Div(radicand_derivative, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

fn sqrt_small_additive_elementary_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (radicand_derivative, derivative_denominator, required_conditions) =
        small_additive_elementary_radicand_derivative_for_calculus_presentation(
            ctx, radicand, var_name,
        )?;

    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let mut denominator_factors = vec![two];
    if let Some(derivative_denominator) = derivative_denominator {
        denominator_factors.push(derivative_denominator);
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let denominator = compact_numeric_mul_factors_for_calculus_presentation(ctx, denominator);
    let compact = ctx.add_raw(Expr::Div(radicand_derivative, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

fn small_additive_elementary_radicand_derivative_for_calculus_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<(ExprId, Option<ExprId>, Vec<crate::ImplicitCondition>)> {
    small_additive_elementary_common_derivative_for_calculus_presentation(ctx, radicand, var_name)
}

fn small_additive_elementary_common_derivative_for_calculus_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<(ExprId, Option<ExprId>, Vec<crate::ImplicitCondition>)> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 4 {
        return None;
    }

    let var = ctx.var(var_name);
    let two = ctx.num(2);
    let sqrt_var = sqrt_raw_for_calculus_presentation(ctx, var);
    let common_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, var, sqrt_var]);
    let common_denominator =
        compact_numeric_mul_factors_for_calculus_presentation(ctx, common_denominator);
    let common_scale = cas_math::expr_nary::build_balanced_mul(ctx, &[two, var, sqrt_var]);

    let mut saw_ln_var = false;
    let mut saw_sqrt_var = false;
    let mut numerator_terms = Vec::new();
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };

        if let Some((scale, arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if arg != var {
                return None;
            }
            saw_ln_var = true;
            required_conditions.push(crate::ImplicitCondition::Positive(arg));
            let two_sqrt = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_var]);
            numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, two_sqrt));
            continue;
        }

        if let Some((scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if sqrt_arg != var {
                return None;
            }
            saw_sqrt_var = true;
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, var));
            continue;
        }

        if let Some(derivative) = scaled_exp_trig_variable_term_derivative_for_calculus_presentation(
            ctx,
            signed_term,
            var_name,
        ) {
            let term = ctx.add_raw(Expr::Mul(common_scale, derivative));
            numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
                ctx, term,
            ));
            continue;
        }

        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }

        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            let derivative = derivative.to_expr(ctx);
            let term = ctx.add_raw(Expr::Mul(common_scale, derivative));
            numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
                ctx, term,
            ));
        }
    }

    if !(saw_ln_var && saw_sqrt_var) || numerator_terms.is_empty() {
        return None;
    }
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    Some((numerator, Some(common_denominator), required_conditions))
}

fn scaled_exp_trig_variable_term_derivative_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, term)
        .unwrap_or((BigRational::one(), term));
    let exp_arg = match ctx.get(core).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Exp) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => exp,
        _ => return None,
    };

    let (trig_derivative, sign) = match ctx.get(exp_arg).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sin) =>
        {
            let var = unary_variable_builtin_arg_for_calculus_presentation(
                ctx,
                exp_arg,
                var_name,
                BuiltinFn::Sin,
            )?;
            (
                ctx.call_builtin(BuiltinFn::Cos, vec![var]),
                BigRational::one(),
            )
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Cos) =>
        {
            let var = unary_variable_builtin_arg_for_calculus_presentation(
                ctx,
                exp_arg,
                var_name,
                BuiltinFn::Cos,
            )?;
            (
                ctx.call_builtin(BuiltinFn::Sin, vec![var]),
                -BigRational::one(),
            )
        }
        _ => return None,
    };

    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[trig_derivative, core]);
    Some(scale_expr_for_calculus_presentation(
        ctx,
        scale * sign,
        product,
    ))
}

pub(crate) fn arctan_sqrt_additive_tan_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_additive_trig_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_small_additive_elementary_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn sqrt_small_additive_elementary_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

fn constant_scaled_arctan_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative =
        arctan_sqrt_polynomial_derivative_presentation(ctx, inner, var_name, BigRational::one())
            .or_else(|| arccot_sqrt_polynomial_derivative_presentation(ctx, inner, var_name))?;
    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn arccot_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arccot_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if let Some(compact) = arctan_sqrt_reciprocal_content_presentation(
        ctx,
        radicand,
        &radicand_poly,
        -BigRational::one(),
    ) {
        return Some(compact);
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = -derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn negative_arccot_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }

    let (radicand, argument_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, args[0])?;
    if argument_scale != -BigRational::one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, &radicand_poly);
    Some((
        ctx.add(Expr::Div(numerator, denominator)),
        required_conditions,
    ))
}

fn sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn sqrt_bounded_trig_positive_shift_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, radicand)?;
    let presentation_radicand =
        compact_double_angle_sine_products_for_calculus_presentation(ctx, radicand)
            .filter(|candidate| {
                bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, *candidate).is_some()
            })
            .unwrap_or(radicand);

    let derivative = differentiate(ctx, presentation_radicand, var_name)?;
    let derivative = compact_small_power_exponents_for_calculus_presentation(ctx, derivative);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let distributed_numerator = if coefficient == BigRational::new(1.into(), 2.into()) {
        distribute_half_over_additive_numerator_for_calculus_presentation(ctx, derivative_core)
    } else {
        None
    };
    let (numerator, denominator_coeff) = if let Some(numerator) = distributed_numerator {
        (numerator, BigRational::one())
    } else {
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        (
            scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core),
            denominator_coeff,
        )
    };
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![presentation_radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(crate) fn sqrt_additive_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let presentation_radicand =
        compact_double_angle_sine_products_for_calculus_presentation(ctx, radicand)
            .or_else(|| signed_add_terms_for_calculus_presentation(ctx, radicand))
            .unwrap_or(radicand);
    let derivative_parts = additive_trig_polynomial_sqrt_radicand_derivative_for_presentation(
        ctx,
        presentation_radicand,
        var_name,
    )?;
    let required_conditions = derivative_parts.required_conditions;
    if let Some(derivative_denominator) = derivative_parts.denominator {
        let numerator =
            compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative_parts.numerator);
        let two =
            rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
        let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, presentation_radicand);
        let denominator = cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[two, derivative_denominator, sqrt_radicand],
        );
        let compact = ctx.add_raw(Expr::Div(numerator, denominator));
        return Some((
            cas_ast::hold::wrap_hold(ctx, compact),
            radicand,
            required_conditions,
        ));
    }

    let derivative = derivative_parts.numerator;
    let derivative = compact_small_power_exponents_for_calculus_presentation(ctx, derivative);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), radicand, required_conditions));
    }

    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let distributed_numerator = if coefficient == BigRational::new(1.into(), 2.into()) {
        distribute_half_over_additive_numerator_for_calculus_presentation(ctx, derivative_core)
    } else {
        None
    };
    let (numerator, denominator_coeff) = if let Some(numerator) = distributed_numerator {
        (numerator, BigRational::one())
    } else {
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        (
            scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core),
            denominator_coeff,
        )
    };
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, presentation_radicand);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

pub(crate) fn sqrt_additive_tan_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut tan_scale = BigRational::zero();
    let mut tan_arg = None;
    let mut common_trig_denominator_builtin = None;
    let mut has_variable_dependency = false;
    let mut common_denominator = None;
    let mut sqrt_variable_derivative = None;
    let mut reciprocal_sqrt_variable_derivative = None;
    let mut reciprocal_derivative_scales = Vec::new();
    let mut other_derivatives = Vec::new();
    let mut has_reciprocal_trig_term = false;
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if let Some((scale, arg, denominator_builtin)) =
            scaled_tan_or_cot_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if tan_arg.is_some_and(|existing| existing != arg) {
                return None;
            }
            if common_trig_denominator_builtin
                .is_some_and(|existing| existing != denominator_builtin)
            {
                return None;
            }
            tan_arg = Some(arg);
            common_trig_denominator_builtin = Some(denominator_builtin);
            tan_scale += scale;
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                other_derivatives.push(derivative);
            }
            continue;
        }
        if let Some((derivative, required_condition)) =
            scaled_sec_or_csc_variable_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            has_reciprocal_trig_term = true;
            other_derivatives.push(derivative);
            required_conditions.push(required_condition);
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if common_denominator.is_some_and(|existing| existing != ln_arg) {
                return None;
            }
            common_denominator = Some(ln_arg);
            reciprocal_derivative_scales.push(ln_scale);
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if let Some((existing_scale, existing_arg)) = &mut sqrt_variable_derivative {
                if *existing_arg != sqrt_arg {
                    return None;
                }
                *existing_scale += sqrt_scale;
            } else {
                sqrt_variable_derivative = Some((sqrt_scale, sqrt_arg));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            match reciprocal_sqrt_variable_derivative.take() {
                Some((mut existing_scale, existing_arg)) if existing_arg == reciprocal_sqrt_arg => {
                    existing_scale += reciprocal_sqrt_scale;
                    reciprocal_sqrt_variable_derivative = Some((existing_scale, existing_arg));
                }
                Some((previous_scale, previous_arg)) => {
                    other_derivatives.push(
                        reciprocal_sqrt_derivative_term_for_calculus_presentation(
                            ctx,
                            previous_scale,
                            previous_arg,
                        ),
                    );
                    other_derivatives.push(
                        reciprocal_sqrt_derivative_term_for_calculus_presentation(
                            ctx,
                            reciprocal_sqrt_scale,
                            reciprocal_sqrt_arg,
                        ),
                    );
                }
                None => {
                    reciprocal_sqrt_variable_derivative =
                        Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg));
                }
            }
            required_conditions.push(crate::ImplicitCondition::Positive(reciprocal_sqrt_arg));
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            other_derivatives.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some(exp_chain_derivative) =
            scaled_exp_bounded_chain_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            other_derivatives.push(exp_chain_derivative);
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            other_derivatives.push(derivative.to_expr(ctx));
        }
    }

    if !has_variable_dependency {
        return None;
    }
    if tan_scale.is_zero() {
        let has_common_denominator_sqrt_and_reciprocal_sqrt_route = common_denominator.is_some()
            && sqrt_variable_derivative.is_some()
            && reciprocal_sqrt_variable_derivative.is_some()
            && !reciprocal_derivative_scales.is_empty()
            && !other_derivatives.is_empty();
        if !has_reciprocal_trig_term && !has_common_denominator_sqrt_and_reciprocal_sqrt_route {
            return None;
        }
        if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
            if let Some(common_denominator) = common_denominator {
                if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
                    reciprocal_sqrt_variable_derivative
                {
                    if reciprocal_sqrt_arg == sqrt_arg
                        && sqrt_arg == common_denominator
                        && !sqrt_scale.is_zero()
                        && !reciprocal_sqrt_scale.is_zero()
                        && !reciprocal_derivative_scales.is_empty()
                        && !other_derivatives.is_empty()
                    {
                        let result =
                            sqrt_additive_generic_common_denominator_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                                ctx,
                                radicand,
                                common_denominator,
                                sqrt_scale,
                                reciprocal_sqrt_scale,
                                reciprocal_derivative_scales,
                                other_derivatives,
                            )?;
                        return Some((result, radicand, required_conditions));
                    }
                    return None;
                } else if !reciprocal_derivative_scales.is_empty() && !other_derivatives.is_empty()
                {
                    let result =
                        sqrt_additive_generic_common_denominator_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            common_denominator,
                            sqrt_arg,
                            sqrt_scale,
                            reciprocal_derivative_scales,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                return None;
            }
            if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
                reciprocal_sqrt_variable_derivative
            {
                if reciprocal_sqrt_arg == sqrt_arg
                    && !sqrt_scale.is_zero()
                    && !reciprocal_sqrt_scale.is_zero()
                {
                    let result =
                        sqrt_additive_generic_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            sqrt_arg,
                            sqrt_scale,
                            reciprocal_sqrt_scale,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
            if has_reciprocal_trig_term {
                let mut derivative_terms = other_derivatives;
                derivative_terms.push(sqrt_variable_derivative_term_for_calculus_presentation(
                    ctx, sqrt_scale, sqrt_arg,
                )?);
                let result =
                    sqrt_additive_generic_derivative_presentation(ctx, radicand, derivative_terms)?;
                return Some((result, radicand, required_conditions));
            }
            let result = sqrt_additive_generic_sqrt_variable_derivative_presentation(
                ctx,
                radicand,
                sqrt_arg,
                sqrt_scale,
                other_derivatives,
            )?;
            return Some((result, radicand, required_conditions));
        }

        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            reciprocal_sqrt_variable_derivative
        {
            if common_denominator.is_none() && !reciprocal_sqrt_scale.is_zero() {
                let result =
                    sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation(
                        ctx,
                        radicand,
                        reciprocal_sqrt_arg,
                        reciprocal_sqrt_scale,
                        other_derivatives,
                    )?;
                return Some((result, radicand, required_conditions));
            }
            if let Some(common_denominator) = common_denominator {
                if reciprocal_sqrt_arg == common_denominator
                    && !reciprocal_sqrt_scale.is_zero()
                    && !reciprocal_derivative_scales.is_empty()
                    && !other_derivatives.is_empty()
                {
                    let result =
                        sqrt_additive_generic_common_denominator_reciprocal_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            common_denominator,
                            reciprocal_sqrt_scale,
                            reciprocal_derivative_scales,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
        }

        if common_denominator.is_none()
            && reciprocal_derivative_scales.is_empty()
            && !other_derivatives.is_empty()
        {
            let result =
                sqrt_additive_generic_derivative_presentation(ctx, radicand, other_derivatives)?;
            return Some((result, radicand, required_conditions));
        }
        if let Some(common_denominator) = common_denominator {
            if !reciprocal_derivative_scales.is_empty() && !other_derivatives.is_empty() {
                let result = sqrt_additive_generic_common_denominator_derivative_presentation(
                    ctx,
                    radicand,
                    common_denominator,
                    reciprocal_derivative_scales,
                    other_derivatives,
                )?;
                return Some((result, radicand, required_conditions));
            }
        }

        return None;
    }
    let tan_arg = tan_arg?;
    let common_trig_denominator_builtin = common_trig_denominator_builtin?;
    let reciprocal_trig_builtin = match common_trig_denominator_builtin {
        BuiltinFn::Cos => BuiltinFn::Sec,
        BuiltinFn::Sin => BuiltinFn::Csc,
        _ => return None,
    };
    let cos_arg = ctx.call_builtin(common_trig_denominator_builtin, vec![tan_arg]);
    let two = ctx.num(2);
    let cos_square = ctx.add_raw(Expr::Pow(cos_arg, two));

    if sqrt_variable_derivative.is_none()
        && reciprocal_sqrt_variable_derivative.is_none()
        && common_denominator.is_some()
        && matches!(reciprocal_derivative_scales.as_slice(), [scale] if !scale.is_zero())
    {
        let (result, _, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, target, var_name)?;
        return Some((result, radicand, required_conditions));
    }

    if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
        if common_denominator.is_some() {
            return None;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            reciprocal_sqrt_variable_derivative
        {
            if reciprocal_sqrt_arg == sqrt_arg
                && !sqrt_scale.is_zero()
                && !reciprocal_sqrt_scale.is_zero()
            {
                let result =
                    sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                        ctx,
                        SqrtAdditiveTanDerivativePresentationParts {
                            radicand,
                            tan_arg,
                            reciprocal_trig_builtin,
                            tan_scale: tan_scale.clone(),
                            other_derivatives,
                        },
                        sqrt_arg,
                        sqrt_scale,
                        reciprocal_sqrt_scale,
                    )?;
                required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
                return Some((result, radicand, required_conditions));
            }
            other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                ctx,
                reciprocal_sqrt_scale,
                reciprocal_sqrt_arg,
            ));
        }
        let result = sqrt_additive_tan_sqrt_variable_derivative_presentation(
            ctx,
            SqrtAdditiveTanDerivativePresentationParts {
                radicand,
                tan_arg,
                reciprocal_trig_builtin,
                tan_scale: tan_scale.clone(),
                other_derivatives,
            },
            sqrt_arg,
            sqrt_scale,
        )?;
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        return Some((result, radicand, required_conditions));
    }

    if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) = reciprocal_sqrt_variable_derivative
    {
        if common_denominator.is_none() && !reciprocal_sqrt_scale.is_zero() {
            let result = sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation(
                ctx,
                SqrtAdditiveTanDerivativePresentationParts {
                    radicand,
                    tan_arg,
                    reciprocal_trig_builtin,
                    tan_scale: tan_scale.clone(),
                    other_derivatives,
                },
                reciprocal_sqrt_arg,
                reciprocal_sqrt_scale,
            )?;
            required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
            return Some((result, radicand, required_conditions));
        }
        other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
            ctx,
            reciprocal_sqrt_scale,
            reciprocal_sqrt_arg,
        ));
    }

    if common_denominator.is_none() && reciprocal_derivative_scales.is_empty() {
        let reciprocal_trig_arg = ctx.call_builtin(reciprocal_trig_builtin, vec![tan_arg]);
        let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
        let tan_derivative =
            scale_expr_for_calculus_presentation(ctx, tan_scale.clone(), reciprocal_trig_square);
        other_derivatives.insert(0, tan_derivative);
        let result =
            sqrt_additive_generic_derivative_presentation(ctx, radicand, other_derivatives)?;
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        return Some((result, radicand, required_conditions));
    }

    let mut numerator_terms = Vec::new();
    let common_denominator =
        common_denominator.filter(|_| !reciprocal_derivative_scales.is_empty());
    let tan_numerator = if let Some(denominator) = common_denominator {
        scale_expr_for_calculus_presentation(ctx, tan_scale, denominator)
    } else {
        rational_const_for_calculus_presentation(ctx, tan_scale)
    };
    numerator_terms.push(tan_numerator);
    for scale in reciprocal_derivative_scales {
        numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, cos_square));
    }
    for derivative in other_derivatives {
        let mut term = compact_tan_sqrt_common_denominator_numerator_term(
            ctx, cos_arg, cos_square, derivative,
        );
        if let Some(denominator) = common_denominator {
            term = ctx.add_raw(Expr::Mul(denominator, term));
            term = compact_numeric_mul_factors_for_calculus_presentation(ctx, term);
        }
        numerator_terms.push(term);
    }
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let denominator_scale =
        rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if let Some(common_denominator) = common_denominator {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[
                denominator_scale,
                common_denominator,
                cos_square,
                sqrt_radicand,
            ],
        )
    } else {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[denominator_scale, cos_square, sqrt_radicand],
        )
    };
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some((cas_ast::hold::wrap_hold(ctx, compact), radicand, {
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        required_conditions
    }))
}

pub(crate) fn sqrt_additive_tan_polynomial_derivative_inline_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut tan_scale = BigRational::zero();
    let mut tan_arg = None;
    let mut common_trig_denominator_builtin = None;
    let mut sqrt_variable_derivative = None;
    let mut other_derivatives = Vec::new();
    let mut has_variable_dependency = false;
    let mut has_ln_derivative = false;
    let mut has_reciprocal_sqrt_derivative = false;
    let mut required_conditions = Vec::new();

    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if let Some((scale, arg, denominator_builtin)) =
            scaled_tan_or_cot_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if tan_arg.is_some_and(|existing| existing != arg) {
                return None;
            }
            if common_trig_denominator_builtin
                .is_some_and(|existing| existing != denominator_builtin)
            {
                return None;
            }
            tan_arg = Some(arg);
            common_trig_denominator_builtin = Some(denominator_builtin);
            tan_scale += scale;
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if let Some((existing_scale, existing_arg)) = &mut sqrt_variable_derivative {
                if *existing_arg != sqrt_arg {
                    return None;
                }
                *existing_scale += sqrt_scale;
            } else {
                sqrt_variable_derivative = Some((sqrt_scale, sqrt_arg));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            has_ln_derivative |= !ln_scale.is_zero();
            let numerator = rational_const_for_calculus_presentation(ctx, ln_scale);
            let reciprocal = ctx.add_raw(Expr::Div(numerator, ln_arg));
            other_derivatives.push(reciprocal);
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            if has_reciprocal_sqrt_derivative {
                return None;
            }
            has_reciprocal_sqrt_derivative = true;
            if !reciprocal_sqrt_scale.is_zero() {
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(reciprocal_sqrt_arg));
            continue;
        }
        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                other_derivatives.push(derivative);
            }
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            other_derivatives.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some(exp_chain_derivative) =
            scaled_exp_bounded_chain_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            other_derivatives.push(exp_chain_derivative);
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            other_derivatives.push(derivative.to_expr(ctx));
        }
    }

    if !has_variable_dependency || tan_scale.is_zero() {
        return None;
    }
    if sqrt_variable_derivative.is_none() && !has_ln_derivative {
        return None;
    }
    if sqrt_variable_derivative.is_some() && has_reciprocal_sqrt_derivative {
        return None;
    }
    if sqrt_variable_derivative
        .as_ref()
        .is_some_and(|(sqrt_scale, _)| sqrt_scale.is_zero())
    {
        return None;
    }
    let tan_arg = tan_arg?;
    let common_trig_denominator_builtin = common_trig_denominator_builtin?;
    let reciprocal_trig_builtin = match common_trig_denominator_builtin {
        BuiltinFn::Cos => BuiltinFn::Sec,
        BuiltinFn::Sin => BuiltinFn::Csc,
        _ => return None,
    };

    let cos_arg = ctx.call_builtin(common_trig_denominator_builtin, vec![tan_arg]);
    let two = ctx.num(2);
    let reciprocal_trig_arg = ctx.call_builtin(reciprocal_trig_builtin, vec![tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let mut derivative_terms = Vec::new();
    derivative_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        tan_scale,
        reciprocal_trig_square,
    ));
    if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
        derivative_terms.push(sqrt_variable_derivative_term_for_calculus_presentation(
            ctx, sqrt_scale, sqrt_arg,
        )?);
    }
    derivative_terms.extend(other_derivatives);

    let result = sqrt_additive_generic_derivative_presentation(ctx, radicand, derivative_terms)?;
    required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
    Some((result, radicand, required_conditions))
}

fn sqrt_variable_derivative_term_for_calculus_presentation(
    ctx: &mut Context,
    sqrt_scale: BigRational,
    sqrt_arg: ExprId,
) -> Option<ExprId> {
    let coefficient = sqrt_scale * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_arg_root
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_arg_root])
    };
    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

fn reciprocal_sqrt_derivative_term_for_calculus_presentation(
    ctx: &mut Context,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_sqrt_arg: ExprId,
) -> ExprId {
    let neg_three_half = ctx.rational(-3, 2);
    let reciprocal_sqrt_cubed = ctx.add_raw(Expr::Pow(reciprocal_sqrt_arg, neg_three_half));
    scale_expr_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale * BigRational::new(1.into(), 2.into()),
        reciprocal_sqrt_cubed,
    )
}

fn sqrt_additive_generic_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if derivative_terms.is_empty() {
        return None;
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &derivative_terms);
    let numerator = compact_small_power_exponents_for_calculus_presentation(ctx, numerator);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_derivative_scales.is_empty() || derivative_terms.is_empty() {
        return None;
    }

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(common_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        numerator_terms.push(rational_const_for_calculus_presentation(ctx, scale));
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, common_denominator, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, common_denominator, sqrt_arg_root]);
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_arg);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        sqrt_scale,
        common_denominator,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[four, common_denominator, sqrt_arg_root, sqrt_radicand],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_common_denominator = sqrt_raw_for_calculus_presentation(ctx, common_denominator);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[two, common_denominator, sqrt_common_denominator],
    );
    let two_sqrt_denominator = ctx.add_raw(Expr::Mul(two, sqrt_common_denominator));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_denominator);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[
            four,
            common_denominator,
            sqrt_common_denominator,
            sqrt_radicand,
        ],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero()
        || reciprocal_sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_common_denominator = sqrt_raw_for_calculus_presentation(ctx, common_denominator);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[two, common_denominator, sqrt_common_denominator],
    );
    let two_sqrt_denominator = ctx.add_raw(Expr::Mul(two, sqrt_common_denominator));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_denominator);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        sqrt_scale,
        common_denominator,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[
            four,
            common_denominator,
            sqrt_common_denominator,
            sqrt_radicand,
        ],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(ctx, sqrt_scale));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, sqrt_arg_root, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || reciprocal_sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx, sqrt_scale, sqrt_arg,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    reciprocal_sqrt_arg: ExprId,
    reciprocal_sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, reciprocal_sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[reciprocal_sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, reciprocal_sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn compact_tan_sqrt_common_denominator_numerator_term(
    ctx: &mut Context,
    cos_arg: ExprId,
    cos_square: ExprId,
    derivative: ExprId,
) -> ExprId {
    let (scale, core) =
        split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, derivative)
            .unwrap_or((BigRational::one(), derivative));
    if cas_ast::ordering::compare_expr(ctx, core, cos_arg).is_eq() {
        let three = ctx.num(3);
        let cos_cube = ctx.add_raw(Expr::Pow(cos_arg, three));
        return scale_expr_for_calculus_presentation(ctx, scale, cos_cube);
    }

    let term = ctx.add_raw(Expr::Mul(cos_square, derivative));
    let term = compact_small_power_exponents_for_calculus_presentation(ctx, term);
    let term =
        combine_matching_cos_powers_for_calculus_presentation(ctx, cos_arg, term).unwrap_or(term);
    compact_numeric_mul_factors_for_calculus_presentation(ctx, term)
}

struct SqrtAdditiveTanDerivativePresentationParts {
    radicand: ExprId,
    tan_arg: ExprId,
    reciprocal_trig_builtin: BuiltinFn,
    tan_scale: BigRational,
    other_derivatives: Vec<ExprId>,
}

fn sqrt_additive_tan_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    let tan_term =
        scale_expr_for_calculus_presentation(ctx, parts.tan_scale, reciprocal_trig_square);
    let tan_term = ctx.add_raw(Expr::Mul(two_sqrt_arg, tan_term));
    numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
        ctx, tan_term,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(ctx, sqrt_scale));
    for derivative in parts.other_derivatives {
        let term = ctx.add_raw(Expr::Mul(two_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, sqrt_arg_root, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || reciprocal_sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    let tan_term =
        scale_expr_for_calculus_presentation(ctx, parts.tan_scale, reciprocal_trig_square);
    let tan_term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, tan_term));
    numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
        ctx, tan_term,
    ));
    for derivative in parts.other_derivatives {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx, sqrt_scale, sqrt_arg,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    reciprocal_sqrt_arg: ExprId,
    reciprocal_sqrt_scale: BigRational,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, reciprocal_sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[reciprocal_sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, reciprocal_sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    let tan_term =
        scale_expr_for_calculus_presentation(ctx, parts.tan_scale, reciprocal_trig_square);
    let tan_term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, tan_term));
    numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
        ctx, tan_term,
    ));
    for derivative in parts.other_derivatives {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn combine_matching_cos_powers_for_calculus_presentation(
    ctx: &mut Context,
    cos_arg: ExprId,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let combined = combine_matching_cos_powers_for_calculus_presentation(ctx, cos_arg, inner)?;
        return Some(ctx.add_raw(Expr::Neg(combined)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut cos_power_sum: i64 = 0;
    let mut non_cos_factors = Vec::new();
    for factor in factors {
        if let Some(power) =
            matching_cos_integer_power_for_calculus_presentation(ctx, cos_arg, factor)
        {
            cos_power_sum += power;
        } else {
            non_cos_factors.push(factor);
        }
    }

    if cos_power_sum <= 1 {
        return None;
    }

    let cos_power = if cos_power_sum == 1 {
        cos_arg
    } else {
        let exponent = ctx.num(cos_power_sum);
        ctx.add_raw(Expr::Pow(cos_arg, exponent))
    };
    non_cos_factors.push(cos_power);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &non_cos_factors,
    ))
}

fn matching_cos_integer_power_for_calculus_presentation(
    ctx: &Context,
    cos_arg: ExprId,
    expr: ExprId,
) -> Option<i64> {
    if cas_ast::ordering::compare_expr(ctx, expr, cos_arg).is_eq() {
        return Some(1);
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !cas_ast::ordering::compare_expr(ctx, *base, cos_arg).is_eq() {
        return None;
    }
    let value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    value
        .is_integer()
        .then(|| value.to_integer().to_i64())
        .flatten()
}

struct AdditiveTrigPolynomialDerivativeForPresentation {
    numerator: ExprId,
    denominator: Option<ExprId>,
    required_conditions: Vec<crate::ImplicitCondition>,
}

fn additive_trig_polynomial_sqrt_radicand_derivative_for_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<AdditiveTrigPolynomialDerivativeForPresentation> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut has_trig_term = false;
    let mut has_variable_dependency = false;
    let mut derivative_terms = Vec::new();
    let mut denominator = None;
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            has_trig_term = true;
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                derivative_terms.push(derivative);
            }
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            derivative_terms.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if denominator.is_some_and(|existing| existing != ln_arg) {
                return None;
            }
            denominator = Some(ln_arg);
            derivative_terms.push(rational_const_for_calculus_presentation(ctx, ln_scale));
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            let neg_half = ctx.rational(-1, 2);
            let reciprocal_sqrt = ctx.add_raw(Expr::Pow(sqrt_arg, neg_half));
            let derivative = scale_expr_for_calculus_presentation(
                ctx,
                sqrt_scale * BigRational::new(1.into(), 2.into()),
                reciprocal_sqrt,
            );
            derivative_terms.push(derivative);
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((reciprocal_scale, reciprocal_arg)) =
            scaled_reciprocal_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            let two = ctx.num(2);
            let reciprocal_denominator = ctx.add(Expr::Pow(reciprocal_arg, two));
            if denominator.is_some_and(|existing| existing != reciprocal_denominator) {
                return None;
            }
            denominator = Some(reciprocal_denominator);
            derivative_terms.push(rational_const_for_calculus_presentation(
                ctx,
                -reciprocal_scale,
            ));
            required_conditions.push(crate::ImplicitCondition::NonZero(reciprocal_arg));
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            derivative_terms.push(derivative.to_expr(ctx));
        }
    }

    if !has_trig_term || !has_variable_dependency {
        return None;
    }

    let numerator = if let Some(denominator) = denominator {
        let scaled_terms: Vec<_> = derivative_terms
            .into_iter()
            .map(|term| {
                if cas_ast::views::as_rational_const(ctx, term, 8).is_some() {
                    term
                } else {
                    ctx.add(Expr::Mul(denominator, term))
                }
            })
            .collect();
        if scaled_terms.is_empty() {
            ctx.num(0)
        } else {
            cas_math::expr_nary::build_balanced_add(ctx, &scaled_terms)
        }
    } else if derivative_terms.is_empty() {
        ctx.num(0)
    } else {
        cas_math::expr_nary::build_balanced_add(ctx, &derivative_terms)
    };
    Some(AdditiveTrigPolynomialDerivativeForPresentation {
        numerator,
        denominator,
        required_conditions,
    })
}

fn tan_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Tan) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_tan_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = tan_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Cot) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = cot_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn scaled_sin_over_cos_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let sin_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, num, var_name, BuiltinFn::Sin)?;
    let cos_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, den, var_name, BuiltinFn::Cos)?;
    (sin_arg == cos_arg).then_some((scale, sin_arg))
}

fn scaled_cos_over_sin_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let cos_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, num, var_name, BuiltinFn::Cos)?;
    let sin_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, den, var_name, BuiltinFn::Sin)?;
    (cos_arg == sin_arg).then_some((scale, cos_arg))
}

fn scaled_tan_or_cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BuiltinFn)> {
    if let Some((scale, arg)) =
        scaled_tan_variable_arg_for_calculus_presentation(ctx, expr, var_name).or_else(|| {
            scaled_sin_over_cos_variable_arg_for_calculus_presentation(ctx, expr, var_name)
        })
    {
        return Some((scale, arg, BuiltinFn::Cos));
    }

    scaled_cot_variable_arg_for_calculus_presentation(ctx, expr, var_name)
        .or_else(|| scaled_cos_over_sin_variable_arg_for_calculus_presentation(ctx, expr, var_name))
        .map(|(scale, arg)| (-scale, arg, BuiltinFn::Sin))
}

fn scaled_sec_or_csc_variable_derivative_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let core = cas_ast::hold::unwrap_internal_hold(ctx, core);
    let Expr::Function(fn_id, args) = ctx.get(core).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    if ctx.sym_name(*sym_id) != var_name {
        return None;
    }

    match ctx.builtin_of(fn_id)? {
        BuiltinFn::Sec => {
            let sec = ctx.call_builtin(BuiltinFn::Sec, vec![args[0]]);
            let tan = ctx.call_builtin(BuiltinFn::Tan, vec![args[0]]);
            let derivative = cas_math::expr_nary::build_balanced_mul(ctx, &[sec, tan]);
            let derivative = scale_expr_for_calculus_presentation(ctx, scale, derivative);
            let cos = ctx.call_builtin(BuiltinFn::Cos, vec![args[0]]);
            Some((derivative, crate::ImplicitCondition::NonZero(cos)))
        }
        BuiltinFn::Csc => {
            let csc = ctx.call_builtin(BuiltinFn::Csc, vec![args[0]]);
            let cot = ctx.call_builtin(BuiltinFn::Cot, vec![args[0]]);
            let derivative = scale_ordered_product_for_calculus_presentation(ctx, -scale, csc, cot);
            let sin = ctx.call_builtin(BuiltinFn::Sin, vec![args[0]]);
            Some((derivative, crate::ImplicitCondition::NonZero(sin)))
        }
        _ => None,
    }
}

fn scale_ordered_product_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    left: ExprId,
    right: ExprId,
) -> ExprId {
    let product = ctx.add_raw(Expr::Mul(left, right));
    if coeff.is_one() {
        return product;
    }
    if coeff == -BigRational::one() {
        return ctx.add_raw(Expr::Neg(product));
    }
    let coeff = rational_const_for_calculus_presentation(ctx, coeff);
    ctx.add_raw(Expr::Mul(coeff, product))
}

fn unary_variable_builtin_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(builtin) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn ln_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_ln_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = ln_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn exp_linear_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Exp) {
                return None;
            }
            let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, args[0], var_name)?;
            if arg_poly.degree() != 1 {
                return None;
            }
            let slope = arg_poly.coeffs.get(1)?.clone();
            (!slope.is_zero()).then_some((slope, expr))
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, exp, var_name)?;
            if arg_poly.degree() != 1 {
                return None;
            }
            let slope = arg_poly.coeffs.get(1)?.clone();
            (!slope.is_zero()).then_some((slope, expr))
        }
        _ => None,
    }
}

fn scaled_exp_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let (chain_scale, exp) = exp_linear_term_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale * chain_scale, exp))
}

fn exp_bounded_chain_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let inner = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Exp) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => exp,
        _ => return None,
    };

    bounded_sin_cos_term_bound_for_calculus_presentation(ctx, inner)?;
    let inner_derivative = differentiate(ctx, inner, var_name)?;
    if cas_ast::views::as_rational_const(ctx, inner_derivative, 8)
        .is_some_and(|value| value.is_zero())
    {
        return None;
    }
    Some((inner_derivative, expr))
}

fn scaled_exp_bounded_chain_derivative_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let (inner_derivative, exp_term) =
        exp_bounded_chain_term_for_calculus_presentation(ctx, core, var_name)?;
    let derivative = cas_math::expr_nary::build_balanced_mul(ctx, &[inner_derivative, exp_term]);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    Some(scale_expr_for_calculus_presentation(ctx, scale, derivative))
}

fn sqrt_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let radicand = calculus_sqrt_like_radicand(ctx, expr)?;
    let Expr::Variable(sym_id) = ctx.get(radicand) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(radicand)
}

fn scaled_sqrt_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let radicand = sqrt_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, radicand))
}

fn reciprocal_sqrt_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            sqrt_variable_arg_for_calculus_presentation(ctx, den, var_name)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            let Expr::Variable(sym_id) = ctx.get(base) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(base)
        }
        _ => None,
    }
}

fn scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, radicand) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(ctx, inner, var_name)?;
        return Some((-scale, radicand));
    }

    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    if let Some(radicand) =
        reciprocal_sqrt_variable_arg_for_calculus_presentation(ctx, core, var_name)
    {
        return Some((scale, radicand));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    let radicand = sqrt_variable_arg_for_calculus_presentation(ctx, den, var_name)?;
    Some((numerator_scale, radicand))
}

fn reciprocal_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            let Expr::Variable(sym_id) = ctx.get(den) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(den)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            let Expr::Variable(sym_id) = ctx.get(base) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(base)
        }
        _ => None,
    }
}

fn scaled_reciprocal_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, arg) =
            scaled_reciprocal_variable_term_for_calculus_presentation(ctx, inner, var_name)?;
        return Some((-scale, arg));
    }

    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    if let Some(arg) = reciprocal_variable_arg_for_calculus_presentation(ctx, core, var_name) {
        return Some((scale, arg));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    let Expr::Variable(sym_id) = ctx.get(den) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some((numerator_scale, den))
}

fn split_signed_numeric_scale_single_core_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr) {
        let (scale, core) = split_numeric_scale_single_core(ctx, *inner)?;
        return Some((-scale, core));
    }
    split_numeric_scale_single_core(ctx, expr)
}

fn distribute_half_over_additive_numerator_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let half = BigRational::new(1.into(), 2.into());
    let mut improves_integer_scale = false;
    for (term, _) in terms.iter().copied() {
        let (term_scale, _) =
            split_numeric_scale_single_core(ctx, term).unwrap_or((BigRational::one(), term));
        let scaled = term_scale * half.clone();
        if scaled.is_integer() {
            improves_integer_scale = true;
            break;
        }
    }
    if !improves_integer_scale {
        return None;
    }

    let mut scaled_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (term_scale, term_core) =
            split_numeric_scale_single_core(ctx, term).unwrap_or((BigRational::one(), term));
        let coeff = match sign {
            cas_math::expr_nary::Sign::Pos => half.clone(),
            cas_math::expr_nary::Sign::Neg => -half.clone(),
        } * term_scale;
        scaled_terms.push(scale_expr_for_calculus_presentation(ctx, coeff, term_core));
    }

    Some(cas_math::expr_nary::build_balanced_add(ctx, &scaled_terms))
}

fn compact_small_power_exponents_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Div(left, right))
        }
        Expr::Neg(inner) => {
            let inner = compact_small_power_exponents_for_calculus_presentation(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Pow(base, exp) => {
            let base = compact_small_power_exponents_for_calculus_presentation(ctx, base);
            if let Some(exponent) = small_rational_const_for_calculus_presentation(ctx, exp) {
                if exponent.is_zero() {
                    return ctx.num(1);
                }
                if exponent.is_one() {
                    return base;
                }
                let exp = rational_const_for_calculus_presentation(ctx, exponent);
                return ctx.add(Expr::Pow(base, exp));
            }
            let exp = compact_small_power_exponents_for_calculus_presentation(ctx, exp);
            ctx.add(Expr::Pow(base, exp))
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| compact_small_power_exponents_for_calculus_presentation(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    }
}

fn compact_numeric_mul_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            let product = ctx.add(Expr::Mul(left, right));
            compact_numeric_product_for_calculus_presentation(ctx, product)
        }
        Expr::Div(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Div(left, right))
        }
        Expr::Neg(inner) => {
            let inner = compact_numeric_mul_factors_for_calculus_presentation(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Pow(base, exp) => {
            let base = compact_numeric_mul_factors_for_calculus_presentation(ctx, base);
            let exp = compact_numeric_mul_factors_for_calculus_presentation(ctx, exp);
            ctx.add(Expr::Pow(base, exp))
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| compact_numeric_mul_factors_for_calculus_presentation(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    }
}

fn compact_numeric_product_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return expr;
    }

    let mut scale = BigRational::one();
    let mut non_numeric_factors = Vec::with_capacity(factors.len());
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if let Expr::Neg(inner) = ctx.get(factor) {
            scale = -scale;
            non_numeric_factors.push(*inner);
        } else {
            non_numeric_factors.push(factor);
        }
    }

    if scale.is_zero() {
        return ctx.num(0);
    }
    if non_numeric_factors.is_empty() {
        return rational_const_for_calculus_presentation(ctx, scale);
    }

    let core = if non_numeric_factors.len() == 1 {
        non_numeric_factors[0]
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric_factors)
    };
    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(core));
    }
    scale_expr_for_calculus_presentation(ctx, scale, core)
}

pub(crate) fn compact_double_angle_sine_products_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut changed = false;
    let mut rebuilt = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let compact = double_angle_sine_product_for_calculus_presentation(ctx, term);
        changed |= compact.is_some();
        let mut rebuilt_term = compact.unwrap_or(term);
        if sign == cas_math::expr_nary::Sign::Neg {
            rebuilt_term = ctx.add(Expr::Neg(rebuilt_term));
        }
        rebuilt.push(rebuilt_term);
    }

    changed.then(|| cas_math::expr_nary::build_balanced_add(ctx, &rebuilt))
}

fn signed_add_terms_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut saw_negative = false;
    let rebuilt = terms
        .into_iter()
        .map(|(term, sign)| {
            if sign == cas_math::expr_nary::Sign::Neg {
                saw_negative = true;
                ctx.add_raw(Expr::Neg(term))
            } else {
                term
            }
        })
        .collect::<Vec<_>>();

    saw_negative.then(|| build_balanced_add_raw_for_calculus_presentation(ctx, &rebuilt))
}

fn build_balanced_add_raw_for_calculus_presentation(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    match terms.len() {
        0 => ctx.num(0),
        1 => terms[0],
        2 => ctx.add_raw(Expr::Add(terms[0], terms[1])),
        n => {
            let mid = n / 2;
            let left = build_balanced_add_raw_for_calculus_presentation(ctx, &terms[..mid]);
            let right = build_balanced_add_raw_for_calculus_presentation(ctx, &terms[mid..]);
            ctx.add_raw(Expr::Add(left, right))
        }
    }
}

fn double_angle_sine_product_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let mut scale = BigRational::one();
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        match ctx.builtin_of(*fn_id) {
            Some(BuiltinFn::Sin) if sin_arg.replace(args[0]).is_none() => {}
            Some(BuiltinFn::Cos) if cos_arg.replace(args[0]).is_none() => {}
            _ => return None,
        }
    }

    if scale != BigRational::from_integer(2.into()) {
        return None;
    }
    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    if compare_expr(ctx, sin_arg, cos_arg) != std::cmp::Ordering::Equal {
        return None;
    }

    let two = rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
    let doubled_arg = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sin_arg]);
    Some(ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg]))
}

fn bounded_sin_cos_shift_margin_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    let mut constant_shift = BigRational::zero();
    let mut trig_bound = BigRational::zero();
    let mut has_bounded_trig = false;

    for term in cas_math::expr_nary::add_terms_no_sign(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, term, 8) {
            constant_shift += value;
            continue;
        }

        let bound = bounded_sin_cos_term_bound_for_calculus_presentation(ctx, term)?;
        trig_bound += bound;
        has_bounded_trig = true;
    }

    if has_bounded_trig && constant_shift > trig_bound {
        Some(constant_shift - trig_bound)
    } else {
        None
    }
}

fn bounded_sin_cos_term_bound_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if bounded_sin_cos_unit_factor_for_calculus_presentation(ctx, expr) {
        return Some(BigRational::one());
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return bounded_sin_cos_term_bound_for_calculus_presentation(ctx, *inner);
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let mut scale = BigRational::one();
    let mut has_bounded_factor = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if bounded_sin_cos_unit_factor_for_calculus_presentation(ctx, factor) {
            has_bounded_factor = true;
        } else {
            return None;
        }
    }

    has_bounded_factor.then(|| scale.abs())
}

fn bounded_sin_cos_unit_factor_for_calculus_presentation(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Sin | BuiltinFn::Cos)
                )
        }
        Expr::Pow(base, exp)
            if bounded_sin_cos_unit_factor_for_calculus_presentation(ctx, *base) =>
        {
            cas_ast::views::as_rational_const(ctx, *exp, 8)
                .is_some_and(|value| value.is_integer() && value > BigRational::zero())
        }
        _ => false,
    }
}

fn sqrt_elementary_function_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    #[derive(Clone, Copy)]
    enum SqrtElementaryDerivativeShape {
        Function(BuiltinFn),
        DenominatorSquare(BuiltinFn),
        OnePlusArgSquare,
        OneMinusArgSquare,
        SqrtOneMinusArgSquare,
        SqrtOnePlusArgSquare,
        SqrtArgMinusOneTimesArgPlusOne,
        Log,
        LogConstantBase(i64),
    }

    let radicand = extract_square_root_base(ctx, target)?;
    let Expr::Function(fn_id, args) = ctx.get(radicand).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (shape, sign) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Cos),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cos) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Sin),
            -BigRational::one(),
        ),
        Some(BuiltinFn::Exp) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Exp),
            BigRational::one(),
        ),
        Some(BuiltinFn::Tan) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Cos),
            BigRational::one(),
        ),
        Some(BuiltinFn::Tanh) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Cosh),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cot) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Sin),
            -BigRational::one(),
        ),
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => (
            SqrtElementaryDerivativeShape::OnePlusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Atanh) => (
            SqrtElementaryDerivativeShape::OneMinusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Asin | BuiltinFn::Arcsin) => (
            SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Acos | BuiltinFn::Arccos) => (
            SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare,
            -BigRational::one(),
        ),
        Some(BuiltinFn::Ln) => (SqrtElementaryDerivativeShape::Log, BigRational::one()),
        Some(BuiltinFn::Log2) => (
            SqrtElementaryDerivativeShape::LogConstantBase(2),
            BigRational::one(),
        ),
        Some(BuiltinFn::Log10) => (
            SqrtElementaryDerivativeShape::LogConstantBase(10),
            BigRational::one(),
        ),
        Some(BuiltinFn::Asinh) => (
            SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Acosh) => (
            SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne,
            BigRational::one(),
        ),
        Some(BuiltinFn::Sinh) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Cosh),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cosh) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Sinh),
            BigRational::one(),
        ),
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (mut derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let mut coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    if let Some(core_value) = signed_rational_const_for_calculus_presentation(ctx, derivative_core)
    {
        coefficient *= core_value;
        derivative_core = ctx.num(1);
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_function = match shape {
        SqrtElementaryDerivativeShape::Function(derivative_fn) => {
            Some(ctx.call_builtin(derivative_fn, vec![args[0]]))
        }
        SqrtElementaryDerivativeShape::DenominatorSquare(_) => None,
        SqrtElementaryDerivativeShape::OnePlusArgSquare => None,
        SqrtElementaryDerivativeShape::OneMinusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne => None,
        SqrtElementaryDerivativeShape::Log => None,
        SqrtElementaryDerivativeShape::LogConstantBase(_) => None,
    };
    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = match derivative_function {
        Some(derivative_function) if derivative_core_is_one => derivative_function,
        Some(derivative_function) => {
            cas_math::expr_nary::build_balanced_mul(ctx, &[derivative_core, derivative_function])
        }
        None => derivative_core,
    };
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::Log) {
        denominator_factors.push(args[0]);
    }
    if let SqrtElementaryDerivativeShape::LogConstantBase(base) = shape {
        denominator_factors.push(args[0]);
        let base_expr = ctx.num(base);
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]));
    }
    if let SqrtElementaryDerivativeShape::DenominatorSquare(denominator_fn) = shape {
        let denominator_arg = ctx.call_builtin(denominator_fn, vec![args[0]]);
        let two = ctx.num(2);
        denominator_factors.push(ctx.add(Expr::Pow(denominator_arg, two)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::OnePlusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        denominator_factors.push(ctx.add(Expr::Add(arg_square, one)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::OneMinusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let neg_arg_square = ctx.add(Expr::Neg(arg_square));
        denominator_factors.push(ctx.add(Expr::Add(one, neg_arg_square)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let neg_arg_square = ctx.add(Expr::Neg(arg_square));
        let radicand = ctx.add(Expr::Add(one, neg_arg_square));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let radicand = ctx.add(Expr::Add(arg_square, one));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]));
    }
    if matches!(
        shape,
        SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne
    ) {
        let one_poly = Polynomial::one(var_name.to_string());
        let arg_minus_one = arg_poly.sub(&one_poly).to_expr(ctx);
        let arg_plus_one = arg_poly.add(&one_poly).to_expr(ctx);
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_minus_one]));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_plus_one]));
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);

    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

fn sqrt_reciprocal_trig_function_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let Expr::Function(fn_id, args) = ctx.get(radicand).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (derivative_fn, sign) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sec) => (BuiltinFn::Tan, BigRational::one()),
        Some(BuiltinFn::Csc) => (BuiltinFn::Cot, -BigRational::one()),
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (mut derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let mut coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    if let Some(core_value) = signed_rational_const_for_calculus_presentation(ctx, derivative_core)
    {
        coefficient *= core_value;
        derivative_core = ctx.num(1);
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let trig_factor = ctx.call_builtin(derivative_fn, vec![args[0]]);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut numerator_factors = Vec::new();
    if !derivative_core_is_one {
        numerator_factors.push(derivative_core);
    }
    numerator_factors.push(trig_factor);
    numerator_factors.push(sqrt_radicand);
    let numerator_core = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    if denominator_coeff == BigRational::one() {
        return Some(numerator);
    }

    let denominator = rational_const_for_calculus_presentation(ctx, denominator_coeff);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn polynomial_times_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(target) else {
        return None;
    };

    let mut polynomial_factors = Vec::new();
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
        } else {
            polynomial_factors.push(factor);
        }
    }

    let radicand = radicand?;
    if polynomial_factors.is_empty() {
        return None;
    }

    let polynomial_expr = if polynomial_factors.len() == 1 {
        polynomial_factors[0]
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &polynomial_factors)
    };
    let multiplier_poly =
        polynomial_radicand_for_calculus_presentation(ctx, polynomial_expr, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let multiplier_derivative = multiplier_poly.derivative();
    let radicand_derivative = radicand_poly.derivative();

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let numerator_poly = multiplier_derivative
        .mul(&radicand_poly)
        .mul(&two_poly)
        .add(&multiplier_poly.mul(&radicand_derivative));
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_content * BigRational::new(1.into(), 2.into())))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(crate) fn try_post_calculus_presentation(
    ctx: &mut Context,
    source: ExprId,
    result: ExprId,
) -> Option<ExprId> {
    let unwrapped_result = unwrap_internal_hold_for_calculus(ctx, result);
    if matches!(
        ctx.get(unwrapped_result),
        Expr::Constant(Constant::Undefined)
    ) {
        return None;
    }

    if let Some(call) = try_extract_integrate_call(ctx, source) {
        if let Some(compact) = try_integrate_post_calculus_presentation(ctx, &call, result) {
            return Some(compact);
        }
    }

    let call = try_extract_diff_call(ctx, source)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if bounded_inverse_trig_known_empty_open_interval_gap(ctx, target, &call.var_name).is_some() {
        return None;
    }
    if let Some((compact, _, _)) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _, _)) =
        arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _, _)) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _, _)) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _)) =
        ln_constant_shifted_tan_sqrt_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        try_diff_integral_source_post_calculus_presentation(ctx, target, result, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_cosh_log_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if ln_sqrt_negative_polynomial_gap_target(ctx, target, &call.var_name) {
        if let Some(compact) =
            compact_negative_half_power_result_for_integration_presentation(ctx, result)
        {
            return Some(compact);
        }
    }
    if let Some((compact, _)) =
        sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = supported_integral_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_trig_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = reciprocal_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        sqrt_over_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        log_over_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_over_log_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_over_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_polynomial_quotient_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_of_polynomial_quotient_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_exp_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_ln_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _, _)) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) =
        sqrt_elementary_function_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_reciprocal_trig_function_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        signed_elementary_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_arctan_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_arctan_surd_quotient_scaled_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        constant_scaled_acosh_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_sqrt_affine_quotient_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_reciprocal_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) = unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        arctan_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_sqrt_affine_quotient_positive_gap_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        arctan_self_normalized_surd_reciprocal_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        -BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = arctan_sqrt_affine_partition_quotient_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = arctan_sqrt_polynomial_quotient_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        asinh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        scaled_asinh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_asinh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        scaled_atanh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_atanh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        scaled_acosh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        acosh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_acosh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) = constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_sqrt_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_sqrt_quadratic_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_affine_abs_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_positive_quadratic_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_positive_quadratic_square_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = ln_sqrt_shift_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        variable_base_constant_argument_log_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) = ln_power_derivative_numeric_presentation(ctx, target, result) {
        return Some(compact);
    }
    if let Some(compact) =
        unit_interval_bounded_inverse_trig_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        arctan_rational_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_rational_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = asinh_polynomial_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) = acosh_affine_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        acosh_strictly_positive_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        asinh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _)) =
        atanh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) = arctan_sqrt_constant_over_polynomial_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        arccot_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        arccot_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = arctan_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }

    let (radicand, derivative_scale) =
        arctan_sqrt_scaled_variable_arg(ctx, target, &call.var_name)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let half = BigRational::new(1.into(), 2.into());
    let coefficient = derivative_scale * half;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let denominator_head = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add(Expr::Mul(denominator_scale, sqrt_radicand))
    };
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let denominator = ctx.add(Expr::Mul(denominator_head, radicand_plus_one));
    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some(compact)
}

fn constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(inner, outer_den) = ctx.get(target).clone() else {
        return None;
    };
    if contains_named_var(ctx, outer_den, var_name) {
        return None;
    }

    let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
    let inner_derivative =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
            .or_else(|| arctan_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))?;
    Some(divide_compact_derivative_by_constant_factor(
        ctx,
        inner_derivative,
        outer_den,
    ))
}

fn reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    for idx in 0..factors.len() {
        let inner = factors[idx];
        let mut constant_factors = factors.clone();
        constant_factors.remove(idx);
        let [constant_factor] = constant_factors.as_slice() else {
            continue;
        };
        let Some(outer_den) = reciprocal_constant_denominator_for_calculus_presentation(
            ctx,
            *constant_factor,
            var_name,
        ) else {
            continue;
        };
        let Some(inner_derivative) =
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
                .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
                .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))
        else {
            continue;
        };
        return Some(divide_compact_derivative_by_constant_factor(
            ctx,
            inner_derivative,
            outer_den,
        ));
    }

    None
}

fn constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, inner) = rational_scaled_single_factor_allow_unit(ctx, target)?;
    let (derivative, required_conditions) =
        inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
            ctx, inner, var_name,
        )
        .or_else(|| {
            inverse_tangent_reciprocal_sqrt_shifted_sqrt_product_derivative_presentation(
                ctx, inner, var_name,
            )
        })?;
    let derivative = if scale.is_one() {
        unwrap_internal_hold_for_calculus(ctx, derivative)
    } else {
        scale_compact_derivative_by_rational(ctx, derivative, scale)
    };
    Some((ctx.add(Expr::Hold(derivative)), required_conditions))
}

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    let call = try_extract_integrate_call(ctx, expr)?;
    if let Some((mut result, required_nonzero)) =
        cas_math::symbolic_integration_support::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        if polynomial_times_arctan_affine_integrand_for_diff_shortcut(
            ctx,
            call.target,
            &call.var_name,
        ) {
            if let Some(compact) =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name)
            {
                result = compact;
            }
        }
        return Some(integrate_rewrite_with_conditions(
            ctx,
            &call,
            result,
            std::iter::once(crate::ImplicitCondition::NonZero(required_nonzero)),
        ));
    }

    let mut required_conditions =
        IntegrationRequiredConditions::from_target(ctx, call.target, &call.var_name);
    let source_preservation =
        integration_source_preservation_gates(ctx, call.target, &call.var_name);
    let mut result = integrate(ctx, call.target, &call.var_name)?;
    let compact_polynomial_arctan_by_parts_result =
        compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name);
    required_conditions.extend_atanh_result_conditions_if_source_positive_absent(ctx, result);
    result = apply_integration_result_preservation(
        ctx,
        result,
        &call.var_name,
        required_conditions.has_positive(),
        &source_preservation,
        compact_polynomial_arctan_by_parts_result,
    );
    result = apply_integration_final_presentation(ctx, result, &call.var_name);
    Some(integrate_rewrite_with_conditions(
        ctx,
        &call,
        result,
        required_conditions.into_implicit_conditions(),
    ))
});

fn sign_polynomial_diff_result(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Sign) {
        return None;
    }

    let arg = cas_ast::hold::unwrap_internal_hold(ctx, args[0]);
    if !contains_named_var(ctx, arg, var_name) {
        return None;
    }

    let polynomial = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    let zero = ctx.num(0);
    if polynomial.is_zero() || polynomial.degree() == 0 {
        return Some((zero, Vec::new()));
    }

    Some((zero, vec![crate::ImplicitCondition::NonZero(arg)]))
}

fn undefined_diff_rewrite(ctx: &mut Context, call: &NamedVarCall) -> Rewrite {
    let undefined = ctx.add(Expr::Constant(Constant::Undefined));
    diff_rewrite_with_conditions(
        ctx,
        call,
        undefined,
        std::iter::empty::<crate::ImplicitCondition>(),
    )
}

fn diff_rewrite_with_conditions<I>(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
    required_conditions: I,
) -> Rewrite
where
    I: IntoIterator<Item = crate::ImplicitCondition>,
{
    let desc = render_diff_desc_with(call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Rewrite::new(result)
        .desc(desc)
        .requires_all(required_conditions)
}

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    let call = try_extract_diff_call(ctx, expr)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if diff_target_known_undefined_or_empty_domain_over_reals(ctx, target, &call.var_name) {
        return Some(undefined_diff_rewrite(ctx, &call));
    }
    if let Some((result, required_conditions)) =
        sign_polynomial_diff_result(ctx, target, &call.var_name)
    {
        return Some(diff_rewrite_with_conditions(
            ctx,
            &call,
            result,
            required_conditions,
        ));
    }
    let mut shortcut_required_conditions = Vec::new();
    if let Some((result, required_conditions)) =
        reciprocal_trig_shifted_sqrt_derivative_presentation(ctx, target, &call.var_name)
    {
        shortcut_required_conditions.extend(required_conditions);
        return Some(diff_rewrite_with_conditions(
            ctx,
            &call,
            result,
            shortcut_required_conditions,
        ));
    }
    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        shortcut_required_conditions.push(crate::ImplicitCondition::Positive(required_positive));
        shortcut_required_conditions.extend(required_conditions);
        return Some(diff_rewrite_with_conditions(
            ctx,
            &call,
            result,
            shortcut_required_conditions,
        ));
    }
    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        shortcut_required_conditions.push(crate::ImplicitCondition::Positive(required_positive));
        shortcut_required_conditions.extend(required_conditions);
        let required_conditions =
            reciprocal_trig_and_log_diff_required_conditions(ctx, target, &call.var_name)
                .into_iter()
                .chain(shortcut_required_conditions);
        return Some(diff_rewrite_with_conditions(
            ctx,
            &call,
            result,
            required_conditions,
        ));
    }
    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, &call.var_name)
    {
        shortcut_required_conditions.push(crate::ImplicitCondition::Positive(required_positive));
        shortcut_required_conditions.extend(required_conditions);
        return Some(diff_rewrite_with_conditions(
            ctx,
            &call,
            result,
            shortcut_required_conditions,
        ));
    }
    let mut result = ln_reciprocal_trig_sqrt_derivative_presentation(ctx, target, &call.var_name)
        .map(|(result, required_conditions)| {
            shortcut_required_conditions.extend(required_conditions);
            result
        })
        .or_else(|| {
            ln_constant_shifted_tan_sqrt_derivative_presentation(ctx, target, &call.var_name).map(
                |(result, required_conditions)| {
                    shortcut_required_conditions.extend(required_conditions);
                    result
                },
            )
        })
        .or_else(|| {
            supported_integral_diff_shortcut_presentation(ctx, target, &call.var_name).map(
                |(result, required_conditions)| {
                    shortcut_required_conditions.extend(required_conditions);
                    result
                },
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                atanh_sqrt_over_symbolic_constant_derivative_shortcut(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| reciprocal_positive_shifted_sqrt_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_conditions) =
                reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive) =
                sqrt_over_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_conditions) =
                sqrt_of_polynomial_quotient_derivative_presentation_with_domain(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            constant_scaled_arctan_surd_quotient_scaled_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_acosh_affine_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation(ctx, target, &call.var_name)
                .map(|compact| cas_ast::hold::wrap_hold(ctx, compact))
        })
        .or_else(|| {
            bounded_inverse_trig_self_normalized_projection_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_condition) =
                asinh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| {
            let (result, required_condition) =
                scaled_asinh_sqrt_constant_over_polynomial_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| arctan_surd_quotient_scaled_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| arctan_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            arctan_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| {
            atanh_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_condition) =
                arctan_self_normalized_surd_reciprocal_compact_derivative(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| arctan_affine_by_parts_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| atanh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_condition) =
                atanh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| {
            constant_scaled_asinh_sqrt_polynomial_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            constant_scaled_atanh_sqrt_polynomial_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            constant_scaled_acosh_sqrt_polynomial_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                negative_arccot_sqrt_polynomial_derivative_shortcut(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            sqrt_bounded_trig_positive_shift_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_small_additive_elementary_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let result =
                sqrt_elementary_function_derivative_presentation(ctx, target, &call.var_name)?;
            if let Some(radicand) = extract_square_root_base(ctx, target) {
                shortcut_required_conditions.push(crate::ImplicitCondition::Positive(radicand));
            }
            Some(result)
        })
        .or_else(|| {
            scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            arctan_sqrt_constant_over_polynomial_presentation(
                ctx,
                target,
                &call.var_name,
                BigRational::one(),
            )
        })
        .or_else(|| {
            arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let result =
                acosh_sqrt_shifted_quadratic_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(acosh_sqrt_diff_required_conditions(ctx, target));
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_affine_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_strictly_positive_polynomial_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            positive_quadratic_square_derivative_result_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            positive_quadratic_quotient_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_conditions) =
                ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_conditions) =
                ln_sqrt_plus_polynomial_direct_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            exp_trig_by_parts_primitive_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| affine_tanh_even_primitive_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| {
            affine_hyperbolic_odd_primitive_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| log_over_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| sqrt_over_log_polynomial_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| differentiate(ctx, target, &call.var_name))?;
    if let Some(compact) =
        compact_positive_quadratic_square_derivative_result(ctx, result, &call.var_name)
    {
        result = compact;
    }
    if let Some(compact) = compact_sqrt_var_over_var_times_positive_shift_square_diff_result(
        ctx,
        result,
        &call.var_name,
    ) {
        result = compact;
    }
    let required_conditions = diff_required_conditions_for_target(ctx, target, &call.var_name)
        .into_iter()
        .chain(shortcut_required_conditions);
    Some(diff_rewrite_with_conditions(
        ctx,
        &call,
        result,
        required_conditions,
    ))
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(summation::SumRule));
    simplifier.add_rule(Box::new(summation::ProductRule));
}

#[cfg(test)]
mod compact_hold_tests {
    use super::domain_checks::atanh_known_empty_open_interval_gap;
    use super::power_result_presentation::{
        compact_negative_three_half_power_result_for_integration_presentation,
        compact_positive_half_power_result_for_integration_presentation,
    };
    use super::result_presentation::compact_sqrt_hyperbolic_reciprocal_for_integration_presentation;
    use super::scalar_presentation::fold_numeric_mul_constants_for_hold_additive_terms;
    use super::sqrt_denominator_result_presentation::inverse_sqrt_quotient_arg_result;
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn bounded_inverse_trig_empty_open_interval_gap_detects_root_quadratic() {
        let mut ctx = Context::new();
        let target = parse("arcsin(sqrt(x^2+1))", &mut ctx).unwrap();
        let gap = bounded_inverse_trig_known_empty_open_interval_gap(&mut ctx, target, "x")
            .expect("empty open interval gap");

        assert_eq!(rendered(&ctx, gap), "-(x^2)");

        let negated_target = parse("-arcsin(sqrt(x^2+1))", &mut ctx).unwrap();
        let negated_gap =
            bounded_inverse_trig_known_empty_open_interval_gap(&mut ctx, negated_target, "x")
                .expect("negated empty open interval gap");
        assert_eq!(rendered(&ctx, negated_gap), "-(x^2)");

        let shifted_target = parse("arcsin((x+1)^2+1)", &mut ctx).unwrap();
        assert!(
            bounded_inverse_trig_known_empty_open_interval_gap(&mut ctx, shifted_target, "x")
                .is_some()
        );

        let shifted_negative_target = parse("arccos(-((x+1)^2+1))", &mut ctx).unwrap();
        assert!(bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            shifted_negative_target,
            "x"
        )
        .is_some());

        let finite_boundary_target = parse("arccos(-1)", &mut ctx).unwrap();
        assert!(bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            finite_boundary_target,
            "x"
        )
        .is_none());

        let symbolic_constant_target = parse("arcsin(pi)", &mut ctx).unwrap();
        let symbolic_gap = bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            symbolic_constant_target,
            "x",
        )
        .expect("symbolic constant outside unit interval");
        assert_eq!(rendered(&ctx, symbolic_gap), "1 - pi^2");

        let negative_symbolic_constant_target = parse("arccos(-e)", &mut ctx).unwrap();
        assert!(bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            negative_symbolic_constant_target,
            "x"
        )
        .is_some());

        let shifted_symbolic_constant_target = parse("pi - arccos(e)", &mut ctx).unwrap();
        assert!(bounded_inverse_trig_known_empty_open_interval_gap(
            &mut ctx,
            shifted_symbolic_constant_target,
            "x"
        )
        .is_some());
    }

    #[test]
    fn atanh_empty_open_interval_gap_detects_direct_quadratic_and_wrappers() {
        let mut ctx = Context::new();
        let target = parse("atanh(x^2+1)", &mut ctx).unwrap();
        let gap =
            atanh_known_empty_open_interval_gap(&mut ctx, target).expect("empty interval gap");

        assert_eq!(rendered(&ctx, gap), "-x^4 - 2 * x^2");

        let scaled_target = parse("2*atanh(x^2+1)", &mut ctx).unwrap();
        let scaled_gap = atanh_known_empty_open_interval_gap(&mut ctx, scaled_target)
            .expect("scaled empty interval gap");
        assert_eq!(rendered(&ctx, scaled_gap), "-x^4 - 2 * x^2");

        let shifted_target = parse("atanh((x+1)^2+1)", &mut ctx).unwrap();
        assert!(atanh_known_empty_open_interval_gap(&mut ctx, shifted_target).is_some());

        let symbolic_constant_target = parse("atanh(pi)", &mut ctx).unwrap();
        let symbolic_gap = atanh_known_empty_open_interval_gap(&mut ctx, symbolic_constant_target)
            .expect("symbolic constant outside open interval");
        assert_eq!(rendered(&ctx, symbolic_gap), "1 - pi^2");
    }

    #[test]
    fn atanh_open_interval_condition_compacts_scaled_shifted_sqrt_arg() {
        let mut ctx = Context::new();
        let arg = parse("a*sqrt(x+1)", &mut ctx).unwrap();
        let condition = atanh_open_interval_condition(&mut ctx, arg);

        assert_eq!(rendered(&ctx, condition), "1 - a^2 * x - a^2");
    }

    #[test]
    fn atanh_open_interval_condition_compacts_symbolic_denominator_scaled_sqrt_arg() {
        let mut ctx = Context::new();
        let arg = parse("sqrt(x+1)/a", &mut ctx).unwrap();
        let condition = atanh_open_interval_condition(&mut ctx, arg);

        assert_eq!(rendered(&ctx, condition), "a^2 - x - 1");
    }

    #[test]
    fn atanh_open_interval_condition_compacts_external_denominator_scaled_sqrt_arg() {
        let mut ctx = Context::new();
        let arg = parse("2*sqrt(x+1)/a", &mut ctx).unwrap();
        let condition = atanh_open_interval_condition(&mut ctx, arg);

        assert_eq!(rendered(&ctx, condition), "a^2 - 4 * x - 4");
    }

    #[test]
    fn diff_target_known_undefined_over_reals_detects_nonfinite_log_and_root_domains() {
        let mut ctx = Context::new();
        for source in [
            "-infinity",
            "ln(0)",
            "log2(0)",
            "log(2, 0)",
            "log(1, 2)",
            "log(1, x)",
            "log(-2, x)",
        ] {
            let target = parse(source, &mut ctx).unwrap();
            assert!(
                diff_target_known_undefined_over_reals(&mut ctx, target, "x"),
                "source: {source}"
            );
        }

        for source in ["sqrt(-1)", "sqrt(-x^2-1)", "sqrt(-x^2)"] {
            let target = parse(source, &mut ctx).unwrap();
            assert!(
                diff_target_known_undefined_over_reals(&mut ctx, target, "x"),
                "source: {source}"
            );
        }

        for source in ["sqrt(0)", "sqrt(4)", "ln(1)", "log2(1)", "log(2, 1)"] {
            let target = parse(source, &mut ctx).unwrap();
            assert!(
                !diff_target_known_undefined_over_reals(&mut ctx, target, "x"),
                "source: {source}"
            );
        }
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_collapses_rational_noise() {
        let mut ctx = Context::new();
        let expr = parse("(atanh(x^2/sqrt(3)) * 1/2 * 2)/sqrt(3)", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_absorbs_outer_scale_into_quotient() {
        let mut ctx = Context::new();
        let expr = parse("2 * ((atanh(x^2/sqrt(3))/2)/sqrt(3))", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
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
    fn compact_sqrt_hyperbolic_reciprocal_preserves_shifted_sqrt_display() {
        let mut ctx = Context::new();
        let expr = parse("-k/cosh(x^(1/2)-b)", &mut ctx).unwrap();
        let compact =
            compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "-k / cosh(sqrt(x) - b)");
    }

    #[test]
    fn compact_sqrt_hyperbolic_reciprocal_preserves_negative_shift_orientation() {
        let mut ctx = Context::new();
        let expr = parse("-k/sinh(b-x^(1/2))", &mut ctx).unwrap();
        let compact =
            compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "-k / sinh(b - sqrt(x))");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_extracts_scaled_sqrt_square_factor() {
        let mut ctx = Context::new();
        let expr = parse("25*sqrt(12/25)", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "10 * sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_keeps_fractional_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("-1/(3*(x^2+x-1))", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "-1 / (3 * (x^2 + x - 1))");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_additive_terms_recurses_into_terms() {
        let mut ctx = Context::new();
        let expr = parse("1/2*ln(abs(x+1)) + 1/2*(x^2/2) - 1/2*x", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold_additive_terms(&mut ctx, expr);

        assert_eq!(
            rendered(&ctx, folded),
            "1/2 * ln(|x + 1|) + 1/4 * x^2 - 1/2 * x"
        );
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_direct_arg() {
        let mut ctx = Context::new();
        let expr = parse("x - tanh(x) - tanh(x)^3/3 - tanh(x)^5/5", &mut ctx).unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(x)^6");
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_positive_affine_arg() {
        let mut ctx = Context::new();
        let expr = parse(
            "x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(2 * x + 1)^6");
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_negative_affine_arg() {
        let mut ctx = Context::new();
        let expr = parse(
            "x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(1 - 2 * x)^6");
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_eighth_power() {
        let mut ctx = Context::new();
        let expr = parse(
            "x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5 + tanh(2*x+1)^7/7)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();
        assert_eq!(rendered(&ctx, compact), "tanh(2 * x + 1)^8");

        let expr = parse(
            "x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5 + tanh(1-2*x)^7/7)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();
        assert_eq!(rendered(&ctx, compact), "tanh(1 - 2 * x)^8");
    }

    #[test]
    fn ln_sum_of_equal_derivative_roots_presentation_accepts_scaled_affines() {
        let mut ctx = Context::new();
        let expr = parse("ln(sqrt(2*x+1)+sqrt(2*x+3))", &mut ctx).unwrap();
        let compact = ln_sum_of_equal_derivative_roots_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| {
                panic!("scaled affine equal-derivative root sum should be recognized")
            });

        assert_eq!(
            rendered(&ctx, compact),
            "1 / (sqrt(2 * x + 1) * sqrt(2 * x + 3))"
        );
    }

    #[test]
    fn bounded_trig_positive_shift_sqrt_derivative_presentation_accepts_multi_function_sum() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(cos(x)+2*sin(x)*cos(x)+4)", &mut ctx).unwrap();
        let compact = sqrt_bounded_trig_positive_shift_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("positive shifted bounded trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(cos(2 * x) - 1/2 * sin(x)) / sqrt(sin(2 * x) + cos(x) + 4)"
        );
    }

    #[test]
    fn additive_trig_reciprocal_subtraction_sqrt_derivative_presentation_is_direct() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sin(2*x)+cos(x)-2/x)", &mut ctx).unwrap();
        let (compact, required_positive, required_conditions) =
            sqrt_additive_trig_polynomial_derivative_presentation(&mut ctx, expr, "x")
                .unwrap_or_else(|| panic!("subtracted reciprocal trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(2 * cos(2 * x) * x^2 + 2 - sin(x) * x^2) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) - 2 / x))"
        );
        assert_eq!(
            rendered(&ctx, required_positive),
            "sin(2 * x) + cos(x) - 2 / x"
        );
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn additive_trig_reciprocal_addition_sqrt_derivative_presentation_stays_direct() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sin(2*x)+cos(x)+1/x)", &mut ctx).unwrap();
        let (compact, required_positive, required_conditions) =
            sqrt_additive_trig_polynomial_derivative_presentation(&mut ctx, expr, "x")
                .unwrap_or_else(|| panic!("added reciprocal trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(2 * cos(2 * x) * x^2 - sin(x) * x^2 - 1) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) + 1 / x))"
        );
        assert_eq!(
            rendered(&ctx, required_positive),
            "sin(2 * x) + cos(x) + 1 / x"
        );
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn inverse_sqrt_quotient_arg_result_detects_compact_inverse_sqrt_substitution() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(x^2/sqrt(3))", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, expr));

        let rationalized = parse("arcsin(1/3 * sqrt(3) * x^2)", &mut ctx).unwrap();

        assert!(!inverse_sqrt_quotient_arg_result(&ctx, rationalized));

        let arctan = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, arctan));
    }

    #[test]
    fn compact_negative_half_power_result_for_integration_presentation_uses_sqrt_denominator() {
        let mut ctx = Context::new();
        let expr = parse("-2*(x^2+x+1)^(-1/2)", &mut ctx).unwrap();
        let compact =
            compact_negative_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "-2 / sqrt(x^2 + x + 1)");
    }

    #[test]
    fn compact_negative_three_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(3*(x^2+x+1)^(3/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))"
        );
    }

    #[test]
    fn compact_negative_five_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(5*(x^2+x+1)^(5/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (5 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^2)"
        );
    }

    #[test]
    fn compact_negative_seven_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(7*(x^2+x+1)^(7/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (7 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^3)"
        );
    }

    #[test]
    fn compact_negative_nine_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(9*(x^2+x+1)^(9/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (9 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^4)"
        );
    }

    #[test]
    fn compact_negative_eleven_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(11*(x^2+x+1)^(11/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (11 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^5)"
        );
    }

    #[test]
    fn compact_negative_thirteen_half_power_result_for_integration_presentation_uses_sqrt_product()
    {
        let mut ctx = Context::new();
        let expr = parse("-2/(13*(x^2+x+1)^(13/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (13 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^6)"
        );
    }

    #[test]
    fn compact_negative_three_half_power_result_requires_conditional_domain_signal() {
        let mut ctx = Context::new();
        let expr = parse("-2/(3*(x^2-1)^(3/2))", &mut ctx).unwrap();

        assert!(
            compact_negative_three_half_power_result_for_integration_presentation(
                &mut ctx, expr, "x", false,
            )
            .is_none()
        );

        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", true,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (3 * sqrt(x^2 - 1) * (x^2 - 1))"
        );
    }

    #[test]
    fn compact_positive_half_power_result_for_integration_presentation_uses_sqrt() {
        let mut ctx = Context::new();
        let expr = parse("2*(x^2+x+1)^(1/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "2 * sqrt(x^2 + x + 1)");
    }

    #[test]
    fn compact_positive_three_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("2/3*(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1)"
        );
    }

    #[test]
    fn compact_negative_three_half_power_result_for_integration_presentation_keeps_outer_sign() {
        let mut ctx = Context::new();
        let expr = parse("-2/3*(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-(2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))"
        );
    }

    #[test]
    fn compact_acosh_surd_width_arg_for_integration_presentation_uses_sqrt_denominator() {
        let mut ctx = Context::new();
        let expr = parse("acosh(sqrt(5)*(x^2+x)/5)", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, expr).unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let normalized = parse("acosh(1/5*sqrt(5)*(x^2+x))", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, normalized)
                .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let normalized_power = parse("acosh(1/5*5^(1/2)*(x^2+x))", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, normalized_power)
                .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let negative_half_power = parse("acosh(5^(-1/2)*(x^2+x))", &mut ctx).unwrap();
        let compact = compact_acosh_surd_width_arg_for_integration_presentation(
            &mut ctx,
            negative_half_power,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");
    }

    #[test]
    fn self_normalized_projection_presentation_accepts_quadratic_numerator() {
        let mut ctx = Context::new();
        let expr = parse("arccos((x^2+x+1)/sqrt((x^2+x+1)^2+5))", &mut ctx).unwrap();
        let derivative = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-sqrt(5) * (2 * x + 1) / ((x^2 + x + 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_self_normalized_surd_quotient_accepts_inverse_sqrt_product_arg() {
        let mut ctx = Context::new();
        let expr = parse("arctan((4*x^2+4*x+2)^(-1/2)*(2*x+1))", &mut ctx).unwrap();
        let derivative =
            arctan_self_normalized_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "2 / (sqrt(4 * x^2 + 4 * x + 2) * (2 * (2 * x + 1)^2 + 1))"
        );
    }

    #[test]
    fn atanh_self_normalized_surd_quotient_accepts_inverse_sqrt_product_arg() {
        let mut ctx = Context::new();
        let expr = parse("atanh(((2*x+1)^2+3)^(-1/2)*(2*x+1))", &mut ctx).unwrap();
        let derivative =
            atanh_self_normalized_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / sqrt((2 * x + 1)^2 + 3)");
    }

    #[test]
    fn arctan_self_normalized_surd_reciprocal_accepts_inverse_denominator_arg() {
        let mut ctx = Context::new();
        let expr = parse("arctan((x^2+1)^(1/2)*x^(-1))", &mut ctx).unwrap();
        let (derivative, required_condition) =
            arctan_self_normalized_surd_reciprocal_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-1 / (sqrt(x^2 + 1) * (2 * x^2 + 1))"
        );
        assert!(matches!(
            required_condition,
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "x"
        ));
    }

    #[test]
    fn self_normalized_projection_presentation_accepts_negated_argument() {
        let mut ctx = Context::new();
        let expr = parse("arccos(-x/sqrt(x^2+1))", &mut ctx).unwrap();
        let derivative = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "1 / (x^2 + 1)");
    }

    #[test]
    fn self_normalized_projection_presentation_normalizes_negated_quadratic_content() {
        let mut ctx = Context::new();
        let expr = parse("arccos(-(x^2+x+1)/sqrt((x^2+x+1)^2+5))", &mut ctx).unwrap();
        let derivative = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "sqrt(5) * (2 * x + 1) / ((x^2 + x + 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_surd_quotient_scaled_compact_derivative_avoids_rationalized_route() {
        let mut ctx = Context::new();
        let expr = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();
        let derivative =
            arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / ((2 * x + 2)^2 + 6)");
    }

    #[test]
    fn constant_scaled_arctan_surd_quotient_scaled_derivative_reuses_compact_route() {
        let mut ctx = Context::new();
        let expr = parse("7*arctan((2*x+1)/sqrt(3))/sqrt(3)", &mut ctx).unwrap();
        let derivative =
            constant_scaled_arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x")
                .unwrap();

        assert_eq!(rendered(&ctx, derivative), "7 / (2 * (x^2 + x + 1))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_keeps_parameter_scale_compact() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (2 * sqrt(x) * (a^2 + x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_compacts_arccot_dual() {
        let mut ctx = Context::new();
        let expr = parse("arccot(sqrt(x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "-a / (2 * sqrt(x) * (a^2 + x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_numerator_scale() {
        let mut ctx = Context::new();
        let expr = parse("arctan(2*sqrt(x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/(2*a))", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (4 * a^2 + x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_shortcut_keeps_affine_domain_minimal() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(2*x+2)/(2*a))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(&mut ctx, expr, "x")
                .unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "a / (sqrt(2 * x + 2) * (2 * a^2 + x + 1))"
        );
        assert_eq!(required_conditions.len(), 2);
        assert!(matches!(
            required_conditions[0],
            crate::ImplicitCondition::Positive(required) if rendered(&ctx, required) == "2 * x + 2"
        ));
        assert!(matches!(
            required_conditions[1],
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "a"
        ));
    }

    #[test]
    fn constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_shortcut_scales_affine_numerator(
    ) {
        let mut ctx = Context::new();
        let expr = parse("3*arctan(sqrt(2*x+2)/(2*a))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
                &mut ctx, expr, "x",
            )
            .unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "3 * a / (sqrt(2 * x + 2) * (2 * a^2 + x + 1))"
        );
        assert_eq!(required_conditions.len(), 2);
        assert!(matches!(
            required_conditions[0],
            crate::ImplicitCondition::Positive(required) if rendered(&ctx, required) == "2 * x + 2"
        ));
        assert!(matches!(
            required_conditions[1],
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "a"
        ));
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_fractional_denominator_scale(
    ) {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/(a/2))", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_square_content() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(4*x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_accepts_external_scale() {
        let mut ctx = Context::new();
        let expr = parse("arctan(2*(sqrt(x)/a))", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn atanh_sqrt_over_symbolic_constant_derivative_compacts_exact_square_scale() {
        let mut ctx = Context::new();
        let expr = parse("atanh(2*(sqrt(x+1)/a))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            atanh_sqrt_over_symbolic_constant_derivative_shortcut(&mut ctx, expr, "x").unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "a / (sqrt(x + 1) * (a^2 - 4 * x - 4))"
        );
        assert_eq!(required_conditions.len(), 2);
        assert!(matches!(
            required_conditions[0],
            crate::ImplicitCondition::Positive(required) if rendered(&ctx, required) == "x + 1"
        ));
        assert!(matches!(
            required_conditions[1],
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "a"
        ));
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_rejects_x_dependent_denominator() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/x)", &mut ctx).unwrap();

        assert!(
            inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
                &mut ctx, expr, "x"
            )
            .is_none()
        );
    }

    #[test]
    fn constant_scaled_reciprocal_sqrt_product_arctan_derivative_reuses_compact_route() {
        let mut ctx = Context::new();
        let expr = parse("2*arctan(1/(sqrt(x)*(x+1)))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
                &mut ctx, expr, "x",
            )
            .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(3 * x + 1) / ((x * (x + 1)^2 + 1) * sqrt(x))"
        );
        assert_eq!(required_conditions.len(), 2);
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
    fn arctan_affine_by_parts_compact_derivative_accepts_polynomial_remainder() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3",
            &mut ctx,
        )
        .unwrap();
        let derivative = arctan_affine_by_parts_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "arctan(1 - x) * x^2");

        let normalized = parse(
            "1/6*(2*ln(x^2+2-2*x) + 2*arctan(1-x)*x^3 + 4*arctan(1-x) + x^2 + 4*x)",
            &mut ctx,
        )
        .unwrap();
        let derivative =
            arctan_affine_by_parts_compact_derivative(&mut ctx, normalized, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "x^2 * arctan(1 - x)");
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
    fn sqrt_additive_tan_exp_polynomial_derivative_presentation_accepts_exp_term() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+exp(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(e^x + sec(x)^2 + 1) / (2 * sqrt(tan(x) + e^x + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + e^x + x");
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn sqrt_additive_tan_cos_square_polynomial_derivative_compacts_power_exponent() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+cos(x)^2+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(sec(x)^2 + 1 - 2 * cos(x) * sin(x)) / (2 * sqrt(tan(x) + cos(x)^2 + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + cos(x)^2 + x");
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn sqrt_additive_tan_ln_polynomial_derivative_inline_presentation_accepts_log_term() {
        for (
            input,
            expected_result,
            expected_radicand,
            expected_required_conditions_len,
        ) in [
            (
                "sqrt(tan(x)+ln(x)+x)",
                "(sec(x)^2 + 1 / x + 1) / (2 * sqrt(tan(x) + ln(x) + x))",
                "tan(x) + ln(x) + x",
                2,
            ),
            (
                "sqrt(tan(x)+2*ln(x)+x)",
                "(sec(x)^2 + 2 / x + 1) / (2 * sqrt(tan(x) + 2 * ln(x) + x))",
                "tan(x) + 2 * ln(x) + x",
                2,
            ),
            (
                "sqrt(tan(x)-ln(x)+x)",
                "(sec(x)^2 + 1 - 1 / x) / (2 * sqrt(tan(x) - ln(x) + x))",
                "tan(x) - ln(x) + x",
                2,
            ),
            (
                "sqrt(tan(x)+ln(x)+sqrt(x)+x)",
                "(sec(x)^2 + 1 / x + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(tan(x) + ln(x) + sqrt(x) + x))",
                "tan(x) + ln(x) + sqrt(x) + x",
                3,
            ),
            (
                "sqrt(tan(x)+ln(x)+1/sqrt(x)+x)",
                "(sec(x)^2 + 1 / x + 1 - 1/2 * x^(-3/2)) / (2 * sqrt(tan(x) + ln(x) + 1 / sqrt(x) + x))",
                "tan(x) + ln(x) + 1 / sqrt(x) + x",
                3,
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_inline_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(
                rendered(&ctx, radicand),
                expected_radicand,
                "input: {input}"
            );
            assert_eq!(
                required_conditions.len(),
                expected_required_conditions_len,
                "input: {input}"
            );
        }
    }

    #[test]
    fn sqrt_additive_tan_exp_linear_polynomial_derivative_presentation_accepts_chain_factor() {
        for (input, expected_result, expected_radicand) in [
            (
                "sqrt(tan(x)+exp(2*x)+x)",
                "(sec(x)^2 + 2 * e^(2 * x) + 1) / (2 * sqrt(tan(x) + e^(2 * x) + x))",
                "tan(x) + e^(2 * x) + x",
            ),
            (
                "sqrt(tan(x)+exp(2*x+1)+x)",
                "(sec(x)^2 + 2 * e^(2 * x + 1) + 1) / (2 * sqrt(tan(x) + e^(2 * x + 1) + x))",
                "tan(x) + e^(2 * x + 1) + x",
            ),
            (
                "sqrt(tan(x)+exp(-2*x)+x)",
                "(sec(x)^2 + 1 - 2 * e^(-2 * x)) / (2 * sqrt(tan(x) + e^(-2 * x) + x))",
                "tan(x) + e^(-2 * x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(
                rendered(&ctx, radicand),
                expected_radicand,
                "input: {input}"
            );
            assert_eq!(required_conditions.len(), 1, "input: {input}");
        }
    }

    #[test]
    fn sqrt_additive_tan_reciprocal_sqrt_derivative_presentation_accepts_inverse_sqrt_term() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+1/sqrt(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + 1 / sqrt(x) + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + 1 / sqrt(x) + x");
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn sqrt_additive_tan_negative_reciprocal_sqrt_derivative_presentation_accepts_signed_inverse_sqrt_term(
    ) {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)-1/sqrt(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 1) / (4 * x * sqrt(x) * sqrt(tan(x) - 1 / sqrt(x) + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) - 1 / sqrt(x) + x");
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn sqrt_additive_tan_mixed_sqrt_and_reciprocal_sqrt_derivative_presentation_uses_common_denominator(
    ) {
        for (input, expected_result, expected_radicand) in [
            (
                "sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x))",
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x",
            ),
            (
                "sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x))",
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(rendered(&ctx, radicand), expected_radicand, "input: {input}");
            assert_eq!(required_conditions.len(), 3, "input: {input}");
        }
    }

    #[test]
    fn arctan_sqrt_additive_tan_mixed_sqrt_derivative_presentation_reuses_inner_common_denominator()
    {
        for (input, expected_result, expected_radicand) in [
            (
                "arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x))",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x) * (tan(x) + sqrt(x) + 1 / sqrt(x) + x + 1))",
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x",
            ),
            (
                "arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x))",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x) * (tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x + 1))",
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                arctan_sqrt_additive_tan_polynomial_derivative_presentation(
                    &mut ctx, target, "x",
                )
                .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(rendered(&ctx, radicand), expected_radicand, "input: {input}");
            assert_eq!(required_conditions.len(), 3, "input: {input}");
        }
    }

    #[test]
    fn compact_arctan_additive_terms_accepts_negative_affine_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "1/3*x^3*arctan(1-x) + 1/3*ln(x^2+2-2*x) + 2/3*arctan(1-x) + 1/6*x^2 + 2/3*x",
            &mut ctx,
        )
        .unwrap();
        let compact =
            compact_arctan_additive_terms_for_calculus_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact).matches("arctan(1 - x)").count(), 1);

        let raw_by_parts = parse(
            "1/3*x^3*arctan(1-x) - (-1/3*ln(x^2+2-2*x) - 2/3*arctan(1-x) - 1/6*x^2 - 2/3*x)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            compact_arctan_additive_terms_for_calculus_presentation(&mut ctx, raw_by_parts, "x")
                .unwrap();
        assert_eq!(rendered(&ctx, compact).matches("arctan(1 - x)").count(), 1);

        let duplicate_companions = parse(
            "1/3*ln(x^2+2-2*x) + 1/2*ln(x^2+2-2*x) + 1/3*x^3*arctan(1-x) + 1/2*x^2*arctan(1-x) + 2/3*arctan(1-x) + 1/6*x^2 + 1/2*x + 2/3*x",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_arctan_additive_terms_for_calculus_presentation(
            &mut ctx,
            duplicate_companions,
            "x",
        )
        .unwrap();
        let rendered = rendered(&ctx, compact);
        assert_eq!(rendered.matches("ln(x^2 + 2 - 2 * x)").count(), 1);
        assert!(rendered.contains("5/6 * ln(x^2 + 2 - 2 * x)"));
        assert!(rendered.contains("7/6 * x"));
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
    fn arctan_surd_quotient_compact_derivative_normalizes_negative_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("arctan(-(x^2+x+1)/sqrt(5))", &mut ctx).unwrap();
        let derivative = arctan_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * x + 1) * sqrt(5) / ((x^2 + x + 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_surd_quotient_scaled_compact_derivative_normalizes_negative_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("arctan((1-x-x^2)/sqrt(5))/sqrt(5)", &mut ctx).unwrap();
        let derivative =
            arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * x + 1) / ((x^2 + x - 1)^2 + 5)"
        );
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

    #[test]
    fn compact_positive_cosh_log_presentation_accepts_half_power_argument() {
        let mut ctx = Context::new();
        let expr = parse("ln(cosh((3*x+1)^(1/2)))", &mut ctx).unwrap();
        let compact =
            compact_positive_cosh_log_abs_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "ln(cosh(sqrt(3 * x + 1)))");
    }

    #[test]
    fn compact_positive_cosh_log_presentation_preserves_sinh_abs_shift() {
        let mut ctx = Context::new();
        let expr = parse("ln(abs(sinh(x^(1/2)-b)))", &mut ctx).unwrap();
        let compact =
            compact_positive_cosh_log_abs_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "ln(|sinh(sqrt(x) - b)|)");
    }

    #[test]
    fn calculus_result_presentation_expands_trig_odd_power_primitive_coefficients() {
        let cases = [
            ("1/3*(cos(x)^3 - 3*cos(x))", "1/3 * cos(x)^3 - cos(x)"),
            ("1/3*(3*sin(x) - sin(x)^3)", "sin(x) - 1/3 * sin(x)^3"),
            (
                "1/6*(cos(2*x+1)^3 - 3*cos(2*x+1))",
                "1/6 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1)",
            ),
            (
                "1/5*(10/3*cos(x)^3 - cos(x)^5 - 5*cos(x))",
                "2/3 * cos(x)^3 - cos(x) - 1/5 * cos(x)^5",
            ),
            (
                "1/5*(sin(x)^5 + 5*sin(x) - 10/3*sin(x)^3)",
                "sin(x) + 1/5 * sin(x)^5 - 2/3 * sin(x)^3",
            ),
            (
                "1/10*(10/3*cos(2*x+1)^3 - cos(2*x+1)^5 - 5*cos(2*x+1))",
                "1/3 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1) - 1/10 * cos(2 * x + 1)^5",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn calculus_result_presentation_expands_trig_square_primitive_coefficients() {
        let cases = [
            ("1/4*(2*x - sin(2*x))", "1/2 * x - 1/4 * sin(2 * x)"),
            ("1/4*(sin(2*x) + 2*x)", "1/4 * sin(2 * x) + 1/2 * x"),
            ("1/8*(4*x - sin(4*x+2))", "1/2 * x - 1/8 * sin(4 * x + 2)"),
            ("1/8*(sin(4*x+2) + 4*x)", "1/8 * sin(4 * x + 2) + 1/2 * x"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn calculus_result_presentation_compacts_negative_half_power_product_denominator() {
        let cases = [
            (
                "cos(x)/2*(sin(x)+1)^(1/2-1)",
                "cos(x) / (2 * sqrt(sin(x) + 1))",
            ),
            (
                "((ln(x)+1)^(1/2-1)/2)*(1/x)",
                "1 / (2 * x * sqrt(ln(x) + 1))",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn calculus_result_presentation_compacts_rationalized_symbolic_sqrt_denominator() {
        let cases = [
            (
                "cos(x)*sqrt(sin(x)+1)/(2*sin(x)+2)",
                "cos(x) / (2 * sqrt(sin(x) + 1))",
            ),
            ("sqrt(ln(x)+1)/(ln(x)+1)", "1 / sqrt(ln(x) + 1)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn post_calculus_presentation_compacts_nested_integral_source() {
        let mut ctx = Context::new();
        let source = parse(
            "diff(integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x), x)",
            &mut ctx,
        )
        .unwrap();
        let result = parse(
            "((3*x+1)^(1/2) * sin((3*x+1)^(1/2)) * 3)/(cos((3*x+1)^(1/2)) * (6*x+2))",
            &mut ctx,
        )
        .unwrap();
        let compact = try_post_calculus_presentation(&mut ctx, source, result).unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }
}
