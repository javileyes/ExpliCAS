use super::exp_by_parts_integrand_presentation::linear_exp_by_parts_integrand_for_calculus_presentation;
use super::hyperbolic_by_parts_integrand_presentation::{
    linear_hyperbolic_by_parts_integrand_for_calculus_presentation,
    repeated_hyperbolic_by_parts_integrand_for_calculus_presentation,
};
use super::log_by_parts_integrand_presentation::log_by_parts_integrand_for_calculus_presentation;
use super::trig_by_parts_integrand_presentation::{
    linear_trig_by_parts_integrand_for_calculus_presentation,
    repeated_trig_by_parts_integrand_for_calculus_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) struct ByPartsIntegrandPreservation {
    active: bool,
    pub(super) preserve_compact_log_by_parts: bool,
}

impl ByPartsIntegrandPreservation {
    pub(super) fn should_preserve_compact_result(&self) -> bool {
        self.active
    }
}

#[derive(Default)]
struct ByPartsIntegrandPreservationBuilder {
    active: bool,
    preserve_compact_log_by_parts: bool,
}

impl ByPartsIntegrandPreservationBuilder {
    fn record_hit(mut self, hit: bool) -> Self {
        self.active |= hit;
        self
    }

    fn record_log_by_parts(mut self, hit: bool) -> Self {
        self.active |= hit;
        self.preserve_compact_log_by_parts = hit;
        self
    }

    fn build(self) -> ByPartsIntegrandPreservation {
        ByPartsIntegrandPreservation {
            active: self.active,
            preserve_compact_log_by_parts: self.preserve_compact_log_by_parts,
        }
    }
}

pub(super) fn by_parts_integrand_preservation_gates(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> ByPartsIntegrandPreservation {
    let preserve_compact_linear_exp_by_parts =
        linear_exp_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_linear_trig_by_parts =
        linear_trig_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_linear_hyperbolic_by_parts =
        linear_hyperbolic_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_repeated_trig_by_parts =
        repeated_trig_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_repeated_hyperbolic_by_parts =
        repeated_hyperbolic_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_log_by_parts =
        log_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);

    ByPartsIntegrandPreservationBuilder::default()
        .record_hit(preserve_compact_linear_exp_by_parts)
        .record_hit(preserve_compact_linear_trig_by_parts)
        .record_hit(preserve_compact_linear_hyperbolic_by_parts)
        .record_hit(preserve_compact_repeated_trig_by_parts)
        .record_hit(preserve_compact_repeated_hyperbolic_by_parts)
        .record_log_by_parts(preserve_compact_log_by_parts)
        .build()
}

#[cfg(test)]
mod tests {
    use super::ByPartsIntegrandPreservationBuilder;

    #[test]
    fn by_parts_builder_activates_when_log_by_parts_hits() {
        let preservation = ByPartsIntegrandPreservationBuilder::default()
            .record_log_by_parts(true)
            .build();

        assert!(preservation.should_preserve_compact_result());
        assert!(preservation.preserve_compact_log_by_parts);
    }

    #[test]
    fn by_parts_builder_tracks_activation_only_hits() {
        let preservation = ByPartsIntegrandPreservationBuilder::default()
            .record_hit(true)
            .build();

        assert!(preservation.should_preserve_compact_result());
        assert!(!preservation.preserve_compact_log_by_parts);
    }
}
