use cas_ast::{Context, ExprId};

pub(super) fn rational_linear_partial_fraction_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
        ctx, target, var_name,
    )
}
