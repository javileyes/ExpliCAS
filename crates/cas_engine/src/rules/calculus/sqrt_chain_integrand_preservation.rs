//! Source-side sqrt-chain integrand preservation gates.

use super::sqrt_reciprocal_trig_product_integrand_presentation::sqrt_reciprocal_trig_product_integrand_target;
use cas_ast::{Context, ExprId};

pub(super) struct SqrtChainIntegrandPreservation {
    active: bool,
    pub(super) preserve_compact_sqrt_trig_log: bool,
    pub(super) preserve_compact_sqrt_hyperbolic_log: bool,
    pub(super) preserve_compact_sqrt_hyperbolic_reciprocal_derivative: bool,
}

impl SqrtChainIntegrandPreservation {
    pub(super) fn should_preserve_compact_result(&self) -> bool {
        self.active
    }
}

pub(super) fn sqrt_chain_integrand_preservation_gates(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> SqrtChainIntegrandPreservation {
    let preserve_compact_sqrt_reciprocal_trig_product =
        sqrt_reciprocal_trig_product_integrand_target(ctx, target, var_name);
    let preserve_compact_sqrt_trig_reciprocal = cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_reciprocal_derivative_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_sqrt_trig_log = cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_log_derivative_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_sqrt_hyperbolic_log = cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_sqrt_hyperbolic_reciprocal_square =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
            ctx,
            target,
            var_name,
        );
    let preserve_compact_sqrt_hyperbolic_reciprocal_derivative =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
            ctx,
            target,
            var_name,
        );

    SqrtChainIntegrandPreservation {
        active: preserve_compact_sqrt_reciprocal_trig_product
            || preserve_compact_sqrt_trig_reciprocal
            || preserve_compact_sqrt_trig_log
            || preserve_compact_sqrt_hyperbolic_log
            || preserve_compact_sqrt_hyperbolic_reciprocal_square
            || preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
        preserve_compact_sqrt_trig_log,
        preserve_compact_sqrt_hyperbolic_log,
        preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
    }
}
