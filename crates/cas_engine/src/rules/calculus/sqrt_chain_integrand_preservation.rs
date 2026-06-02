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

#[derive(Default)]
struct SqrtChainIntegrandPreservationBuilder {
    active: bool,
    preserve_compact_sqrt_trig_log: bool,
    preserve_compact_sqrt_hyperbolic_log: bool,
    preserve_compact_sqrt_hyperbolic_reciprocal_derivative: bool,
}

impl SqrtChainIntegrandPreservationBuilder {
    fn record_hit(mut self, hit: bool) -> Self {
        self.active |= hit;
        self
    }

    fn record_sqrt_trig_log(mut self, hit: bool) -> Self {
        self.active |= hit;
        self.preserve_compact_sqrt_trig_log = hit;
        self
    }

    fn record_sqrt_hyperbolic_log(mut self, hit: bool) -> Self {
        self.active |= hit;
        self.preserve_compact_sqrt_hyperbolic_log = hit;
        self
    }

    fn record_sqrt_hyperbolic_reciprocal_derivative(mut self, hit: bool) -> Self {
        self.active |= hit;
        self.preserve_compact_sqrt_hyperbolic_reciprocal_derivative = hit;
        self
    }

    fn build(self) -> SqrtChainIntegrandPreservation {
        SqrtChainIntegrandPreservation {
            active: self.active,
            preserve_compact_sqrt_trig_log: self.preserve_compact_sqrt_trig_log,
            preserve_compact_sqrt_hyperbolic_log: self.preserve_compact_sqrt_hyperbolic_log,
            preserve_compact_sqrt_hyperbolic_reciprocal_derivative: self
                .preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
        }
    }
}

pub(super) fn sqrt_chain_integrand_preservation_gates(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> SqrtChainIntegrandPreservation {
    let preserve_compact_sqrt_reciprocal_trig_product =
        sqrt_reciprocal_trig_product_integrand_target(ctx, target, var_name);
    let preserve_compact_sqrt_trig_reciprocal =
        sqrt_trig_reciprocal_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_sqrt_trig_log =
        sqrt_trig_log_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_sqrt_hyperbolic_log =
        sqrt_hyperbolic_log_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_sqrt_hyperbolic_reciprocal_square =
        sqrt_hyperbolic_reciprocal_square_integrand_for_calculus_presentation(
            ctx, target, var_name,
        );
    let preserve_compact_sqrt_hyperbolic_reciprocal_derivative =
        sqrt_hyperbolic_reciprocal_derivative_integrand_for_calculus_presentation(
            ctx, target, var_name,
        );

    SqrtChainIntegrandPreservationBuilder::default()
        .record_hit(preserve_compact_sqrt_reciprocal_trig_product)
        .record_hit(preserve_compact_sqrt_trig_reciprocal)
        .record_sqrt_trig_log(preserve_compact_sqrt_trig_log)
        .record_sqrt_hyperbolic_log(preserve_compact_sqrt_hyperbolic_log)
        .record_hit(preserve_compact_sqrt_hyperbolic_reciprocal_square)
        .record_sqrt_hyperbolic_reciprocal_derivative(
            preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
        )
        .build()
}

pub(super) fn sqrt_trig_log_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_log_derivative_target(
        ctx, target, var_name,
    )
}

pub(super) fn sqrt_hyperbolic_log_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
        ctx, target, var_name,
    )
}

pub(super) fn sqrt_hyperbolic_reciprocal_square_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
        ctx, target, var_name,
    )
}

pub(super) fn sqrt_hyperbolic_reciprocal_derivative_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
        ctx, target, var_name,
    )
}

fn sqrt_trig_reciprocal_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_reciprocal_derivative_target(
        ctx, target, var_name,
    )
}

#[cfg(test)]
mod tests {
    use super::SqrtChainIntegrandPreservationBuilder;

    #[test]
    fn sqrt_chain_builder_activates_when_fielded_gate_hits() {
        let preservation = SqrtChainIntegrandPreservationBuilder::default()
            .record_sqrt_trig_log(true)
            .build();

        assert!(preservation.should_preserve_compact_result());
        assert!(preservation.preserve_compact_sqrt_trig_log);
    }

    #[test]
    fn sqrt_chain_builder_tracks_activation_only_hits() {
        let preservation = SqrtChainIntegrandPreservationBuilder::default()
            .record_hit(true)
            .build();

        assert!(preservation.should_preserve_compact_result());
        assert!(!preservation.preserve_compact_sqrt_trig_log);
        assert!(!preservation.preserve_compact_sqrt_hyperbolic_log);
        assert!(!preservation.preserve_compact_sqrt_hyperbolic_reciprocal_derivative);
    }
}
