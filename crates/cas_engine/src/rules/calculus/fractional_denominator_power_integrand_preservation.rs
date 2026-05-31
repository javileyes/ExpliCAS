use cas_ast::{Context, ExprId};

pub(super) fn fractional_denominator_power_substitution_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_fractional_denominator_power_substitution_target(
        ctx, target, var_name,
    )
}
