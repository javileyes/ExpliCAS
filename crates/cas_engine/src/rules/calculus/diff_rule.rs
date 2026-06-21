use crate::define_rule;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{try_desugar_higher_order_diff, try_extract_diff_call};

use super::acosh_derivative_routes::constant_scaled_acosh_derivative_rewrite;
use super::arctan_sqrt_positive_affine_derivative_routes::arctan_sqrt_positive_affine_derivative_rewrite;
use super::arctan_sqrt_positive_shift_derivative_presentation::arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_rewrite;
use super::asinh_conditioned_sqrt_derivative_routes::conditioned_asinh_sqrt_constant_over_polynomial_derivative_rewrite;
use super::atanh_sqrt_constant_over_polynomial_presentation::atanh_sqrt_constant_over_polynomial_derivative_rewrite;
use super::compact_derivative_presentation_routes::compact_derivative_presentation_rewrite;
use super::compact_inverse_surd_derivative_routes::compact_inverse_surd_derivative_rewrite;
use super::diff_late_derivative_routes::late_fallback_derivative_rewrite;
use super::diff_rule_support::{
    arctan_sqrt_additive_derivative_rewrite, sign_polynomial_diff_rewrite, undefined_diff_rewrite,
};
use super::domain_checks::diff_target_known_undefined_or_empty_domain_over_reals;
use super::integral_derivative_shortcut_presentation::{
    definite_integral_leibniz_diff_rewrite,
    reciprocal_trig_derivative_product_integral_diff_shortcut_rewrite,
    supported_integral_diff_shortcut_rewrite,
};
use super::inverse_reciprocal_trig_affine_abs_derivative_presentation::constant_scaled_inverse_reciprocal_trig_affine_abs_rewrite;
use super::inverse_surd_quotient_derivative_presentation::constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative_rewrite;
use super::inverse_tangent_scaled_root_derivative_routes::inverse_tangent_scaled_root_derivative_rewrite;
use super::ln_sum_equal_roots_derivative_presentation::ln_sum_of_equal_derivative_roots_derivative_rewrite;
use super::log_sqrt_trig_derivative_routes::log_sqrt_trig_derivative_rewrite;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::reciprocal_sqrt_product_derivative_presentation::constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_rewrite;
use super::reciprocal_trig_derivative_presentation::reciprocal_trig_shifted_sqrt_derivative_rewrite;
use super::shifted_sqrt_derivative_routes::shifted_sqrt_derivative_rewrite;
use super::sqrt_early_derivative_routes::sqrt_early_derivative_rewrite;
use super::sqrt_polynomial_quotient_derivative_presentation::sqrt_of_polynomial_quotient_derivative_rewrite;
use super::surd_quotient_derivative_routes::{
    constant_scaled_surd_quotient_derivative_rewrite, surd_quotient_derivative_rewrite,
};

define_rule!(
    IntegralDiffShortcutRule,
    "Integral Diff Shortcut",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::TRANSFORM,
    priority: 100,
    |ctx, expr| {
        let call = try_extract_diff_call(ctx, expr)?;
        let target = unwrap_internal_hold_for_calculus(ctx, call.target);
        if let Some(rewrite) =
            reciprocal_trig_derivative_product_integral_diff_shortcut_rewrite(ctx, &call, target)
        {
            return Some(rewrite);
        }
        supported_integral_diff_shortcut_rewrite(ctx, &call, target)
    }
);

define_rule!(
    HigherOrderDiffRule,
    "Higher-Order Differentiation",
    |ctx, expr| {
        // `diff(f, x, n)` / `diff(f, x, y)` desugar to nested two-argument diffs so the
        // ordinary differentiation cascade evaluates each layer. Two-argument calls are
        // left to `DiffRule` (the desugar only matches 3+ args).
        let desugared = try_desugar_higher_order_diff(ctx, expr)?;
        Some(Rewrite::new(desugared).desc("derivada de orden superior"))
    }
);

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    let call = try_extract_diff_call(ctx, expr)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if diff_target_known_undefined_or_empty_domain_over_reals(ctx, target, &call.var_name) {
        return Some(undefined_diff_rewrite(ctx, &call));
    }
    if let Some(rewrite) = sign_polynomial_diff_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = reciprocal_trig_shifted_sqrt_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = arctan_sqrt_additive_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = log_sqrt_trig_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) =
        reciprocal_trig_derivative_product_integral_diff_shortcut_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }
    if let Some(rewrite) = supported_integral_diff_shortcut_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = inverse_tangent_scaled_root_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) =
        arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }
    if let Some(rewrite) = arctan_sqrt_positive_affine_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = ln_sum_of_equal_derivative_roots_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = shifted_sqrt_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = sqrt_early_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) = sqrt_of_polynomial_quotient_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }

    if let Some(rewrite) = constant_scaled_surd_quotient_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) =
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_rewrite(
            ctx, &call, target,
        )
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative_rewrite(
            ctx, &call, target,
        )
    {
        return Some(rewrite);
    }
    if let Some(rewrite) = constant_scaled_acosh_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) =
        constant_scaled_inverse_reciprocal_trig_affine_abs_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = compact_inverse_surd_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) =
        conditioned_asinh_sqrt_constant_over_polynomial_derivative_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = surd_quotient_derivative_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }

    if let Some(rewrite) = compact_derivative_presentation_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }
    if let Some(rewrite) =
        atanh_sqrt_constant_over_polynomial_derivative_rewrite(ctx, &call, target)
    {
        return Some(rewrite);
    }

    // Fundamental theorem / Leibniz rule for a definite integral with
    // variable bounds and an opaque integrand: tried late, after the
    // known-integrand shortcuts, so it only catches the residual cases.
    if let Some(rewrite) = definite_integral_leibniz_diff_rewrite(ctx, &call, target) {
        return Some(rewrite);
    }

    late_fallback_derivative_rewrite(ctx, &call, target)
});
