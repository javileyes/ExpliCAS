//! Source-side trigonometric power integrand detection.

use cas_ast::{Context, ExprId};

pub(super) fn affine_trig_power_integrand_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let preserve_compact_tan_fourth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_tan_fourth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_cot_fourth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_cot_fourth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_tan_sixth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_tan_sixth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_cot_sixth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_cot_sixth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_tan_eighth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_tan_eighth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_cot_eighth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_cot_eighth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_sec_fourth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sec_fourth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_csc_fourth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_csc_fourth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_sec_sixth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sec_sixth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_csc_sixth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_csc_sixth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_sec_eighth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sec_eighth_affine_target(
            ctx, expr, var_name,
        );
    let preserve_compact_csc_eighth_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_csc_eighth_affine_target(
            ctx, expr, var_name,
        );

    preserve_compact_tan_fourth_affine
        || preserve_compact_cot_fourth_affine
        || preserve_compact_tan_sixth_affine
        || preserve_compact_cot_sixth_affine
        || preserve_compact_tan_eighth_affine
        || preserve_compact_cot_eighth_affine
        || preserve_compact_sec_fourth_affine
        || preserve_compact_csc_fourth_affine
        || preserve_compact_sec_sixth_affine
        || preserve_compact_csc_sixth_affine
        || preserve_compact_sec_eighth_affine
        || preserve_compact_csc_eighth_affine
}
