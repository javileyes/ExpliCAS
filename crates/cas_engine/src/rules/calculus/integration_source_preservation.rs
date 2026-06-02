use super::integration_source_route_hits::IntegrationSourcePreservationRouteHits;
use cas_ast::{Context, ExprId};

pub(super) struct IntegrationSourcePreservation {
    active: bool,
    pub(super) preserve_compact_fractional_denominator_power: bool,
    pub(super) preserve_compact_sqrt_trig_log: bool,
    pub(super) preserve_compact_sqrt_hyperbolic_log: bool,
    pub(super) preserve_compact_sqrt_hyperbolic_reciprocal_derivative: bool,
    pub(super) preserve_compact_rational_linear_partial_fraction: bool,
    pub(super) preserve_compact_log_by_parts: bool,
}

impl IntegrationSourcePreservation {
    pub(super) fn should_preserve_compact_result(&self) -> bool {
        self.active
    }
}

#[derive(Default)]
pub(super) struct IntegrationSourcePreservationBuilder {
    active: bool,
    preserve_compact_fractional_denominator_power: bool,
    preserve_compact_sqrt_trig_log: bool,
    preserve_compact_sqrt_hyperbolic_log: bool,
    preserve_compact_sqrt_hyperbolic_reciprocal_derivative: bool,
    preserve_compact_rational_linear_partial_fraction: bool,
    preserve_compact_log_by_parts: bool,
}

impl IntegrationSourcePreservationBuilder {
    pub(super) fn record_activation(mut self, active: bool) -> Self {
        self.active |= active;
        self
    }

    pub(super) fn record_fractional_denominator_power(mut self, hit: bool) -> Self {
        self.active |= hit;
        self.preserve_compact_fractional_denominator_power = hit;
        self
    }

    pub(super) fn record_sqrt_chain_flags(
        mut self,
        active: bool,
        preserve_compact_sqrt_trig_log: bool,
        preserve_compact_sqrt_hyperbolic_log: bool,
        preserve_compact_sqrt_hyperbolic_reciprocal_derivative: bool,
    ) -> Self {
        self.active |= active;
        self.preserve_compact_sqrt_trig_log = preserve_compact_sqrt_trig_log;
        self.preserve_compact_sqrt_hyperbolic_log = preserve_compact_sqrt_hyperbolic_log;
        self.preserve_compact_sqrt_hyperbolic_reciprocal_derivative =
            preserve_compact_sqrt_hyperbolic_reciprocal_derivative;
        self
    }

    pub(super) fn record_rational_partial_fraction(
        mut self,
        preserve_compact_rational_linear_partial_fraction: bool,
    ) -> Self {
        self.active |= preserve_compact_rational_linear_partial_fraction;
        self.preserve_compact_rational_linear_partial_fraction =
            preserve_compact_rational_linear_partial_fraction;
        self
    }

    pub(super) fn record_by_parts_flags(
        mut self,
        active: bool,
        preserve_compact_log_by_parts: bool,
    ) -> Self {
        self.active |= active;
        self.preserve_compact_log_by_parts = preserve_compact_log_by_parts;
        self
    }

    pub(super) fn build(self) -> IntegrationSourcePreservation {
        IntegrationSourcePreservation {
            active: self.active,
            preserve_compact_fractional_denominator_power: self
                .preserve_compact_fractional_denominator_power,
            preserve_compact_sqrt_trig_log: self.preserve_compact_sqrt_trig_log,
            preserve_compact_sqrt_hyperbolic_log: self.preserve_compact_sqrt_hyperbolic_log,
            preserve_compact_sqrt_hyperbolic_reciprocal_derivative: self
                .preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
            preserve_compact_rational_linear_partial_fraction: self
                .preserve_compact_rational_linear_partial_fraction,
            preserve_compact_log_by_parts: self.preserve_compact_log_by_parts,
        }
    }
}

pub(super) fn integration_source_preservation_gates(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> IntegrationSourcePreservation {
    IntegrationSourcePreservationRouteHits::from_target(ctx, target, var_name).into_preservation()
}

#[cfg(test)]
mod tests {
    use super::IntegrationSourcePreservationBuilder;

    #[test]
    fn source_preservation_builder_keeps_activation_with_fielded_hits() {
        let preservation = IntegrationSourcePreservationBuilder::default()
            .record_fractional_denominator_power(true)
            .build();

        assert!(preservation.should_preserve_compact_result());
        assert!(preservation.preserve_compact_fractional_denominator_power);
    }

    #[test]
    fn source_preservation_builder_stays_inactive_without_hits() {
        let preservation = IntegrationSourcePreservationBuilder::default().build();

        assert!(!preservation.should_preserve_compact_result());
        assert!(!preservation.preserve_compact_fractional_denominator_power);
        assert!(!preservation.preserve_compact_rational_linear_partial_fraction);
        assert!(!preservation.preserve_compact_log_by_parts);
    }
}
