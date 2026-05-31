use cas_ast::{Context, ExprId};

pub(super) fn log_product_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let preserve_compact_log_cube_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_log_cube_product_substitution_target(
            ctx, target, var_name,
        );
    let preserve_compact_log_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_log_product_substitution_target(
            ctx, target, var_name,
        );

    preserve_compact_log_cube_product || preserve_compact_log_product
}
