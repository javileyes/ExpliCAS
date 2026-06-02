use super::integration_source_preservation::IntegrationSourcePreservation;
use super::sqrt_denominator_result_presentation::{
    has_sqrt_denominator_result, inverse_sqrt_quotient_arg_result,
};
use super::sqrt_hyperbolic_result_presentation::{
    compact_positive_cosh_log_abs_for_integration_presentation,
    compact_sqrt_hyperbolic_reciprocal_for_integration_presentation,
    has_compactable_sqrt_hyperbolic_reciprocal_result,
};
use super::sqrt_reciprocal_trig_antiderivative_presentation::sqrt_reciprocal_trig_antiderivative_result;
use super::sqrt_trig_result_presentation::{
    compact_sqrt_trig_log_abs_for_integration_presentation, has_compactable_ln_abs_trig_sqrt,
};
use cas_ast::{Context, ExprId};

pub(super) struct IntegrationSqrtResultPreservation {
    active: bool,
    preserve_compact_sqrt_trig_log_result: bool,
    preserve_compact_sqrt_hyperbolic_reciprocal_result: bool,
}

#[derive(Clone, Copy)]
struct IntegrationSqrtResultPresentationPolicy {
    apply_sqrt_hyperbolic_log: bool,
    apply_sqrt_trig_log: bool,
    apply_sqrt_hyperbolic_reciprocal: bool,
}

impl IntegrationSqrtResultPresentationPolicy {
    fn from_preservations(
        source_preservation: &IntegrationSourcePreservation,
        result_preservation: &IntegrationSqrtResultPreservation,
    ) -> Self {
        Self {
            apply_sqrt_hyperbolic_log: source_preservation.preserve_compact_sqrt_hyperbolic_log,
            apply_sqrt_trig_log: source_preservation.preserve_compact_sqrt_trig_log
                || result_preservation.preserve_compact_sqrt_trig_log_result,
            apply_sqrt_hyperbolic_reciprocal: source_preservation
                .preserve_compact_sqrt_hyperbolic_reciprocal_derivative
                || result_preservation.preserve_compact_sqrt_hyperbolic_reciprocal_result,
        }
    }

    fn should_apply_sqrt_hyperbolic_log(&self) -> bool {
        self.apply_sqrt_hyperbolic_log
    }

    fn should_apply_sqrt_trig_log(&self) -> bool {
        self.apply_sqrt_trig_log
    }

    fn should_apply_sqrt_hyperbolic_reciprocal(&self) -> bool {
        self.apply_sqrt_hyperbolic_reciprocal
    }

    fn apply(self, ctx: &mut Context, mut result: ExprId, var_name: &str) -> ExprId {
        if self.should_apply_sqrt_hyperbolic_log() {
            result =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, result, var_name);
        }
        if self.should_apply_sqrt_trig_log() {
            result = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, result, var_name);
        }
        if self.should_apply_sqrt_hyperbolic_reciprocal() {
            result = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, result, var_name,
            );
        }
        result
    }
}

#[derive(Default)]
struct IntegrationSqrtResultPreservationBuilder {
    active: bool,
    preserve_compact_sqrt_trig_log_result: bool,
    preserve_compact_sqrt_hyperbolic_reciprocal_result: bool,
}

impl IntegrationSqrtResultPreservationBuilder {
    fn record_hit(mut self, hit: bool) -> Self {
        self.active |= hit;
        self
    }

    fn record_sqrt_trig_log_result(mut self, hit: bool) -> Self {
        self.active |= hit;
        self.preserve_compact_sqrt_trig_log_result = hit;
        self
    }

    fn record_sqrt_hyperbolic_reciprocal_result(mut self, hit: bool) -> Self {
        self.active |= hit;
        self.preserve_compact_sqrt_hyperbolic_reciprocal_result = hit;
        self
    }

    fn build(self) -> IntegrationSqrtResultPreservation {
        IntegrationSqrtResultPreservation {
            active: self.active,
            preserve_compact_sqrt_trig_log_result: self.preserve_compact_sqrt_trig_log_result,
            preserve_compact_sqrt_hyperbolic_reciprocal_result: self
                .preserve_compact_sqrt_hyperbolic_reciprocal_result,
        }
    }
}

impl IntegrationSqrtResultPreservation {
    pub(super) fn from_result(ctx: &mut Context, result: ExprId, var_name: &str) -> Self {
        let preserve_compact_inverse_sqrt_arg = inverse_sqrt_quotient_arg_result(ctx, result);
        let preserve_compact_sqrt_denominator_result = has_sqrt_denominator_result(ctx, result);
        let preserve_compact_sqrt_reciprocal_trig_result =
            sqrt_reciprocal_trig_antiderivative_result(ctx, result, var_name);
        let preserve_compact_sqrt_trig_log_result =
            has_compactable_ln_abs_trig_sqrt(ctx, result, var_name);
        let preserve_compact_sqrt_hyperbolic_reciprocal_result =
            has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, result, var_name);

        IntegrationSqrtResultPreservationBuilder::default()
            .record_hit(preserve_compact_inverse_sqrt_arg)
            .record_hit(preserve_compact_sqrt_denominator_result)
            .record_hit(preserve_compact_sqrt_reciprocal_trig_result)
            .record_sqrt_trig_log_result(preserve_compact_sqrt_trig_log_result)
            .record_sqrt_hyperbolic_reciprocal_result(
                preserve_compact_sqrt_hyperbolic_reciprocal_result,
            )
            .build()
    }

    pub(super) fn should_preserve_compact_result(&self) -> bool {
        self.active
    }

    pub(super) fn apply(
        &self,
        ctx: &mut Context,
        result: ExprId,
        var_name: &str,
        source_preservation: &IntegrationSourcePreservation,
    ) -> ExprId {
        IntegrationSqrtResultPresentationPolicy::from_preservations(source_preservation, self)
            .apply(ctx, result, var_name)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        IntegrationSqrtResultPresentationPolicy, IntegrationSqrtResultPreservationBuilder,
    };

    #[test]
    fn sqrt_result_builder_activates_on_specific_result_hit() {
        let preservation = IntegrationSqrtResultPreservationBuilder::default()
            .record_sqrt_trig_log_result(true)
            .build();

        assert!(preservation.should_preserve_compact_result());
        assert!(preservation.preserve_compact_sqrt_trig_log_result);
        assert!(!preservation.preserve_compact_sqrt_hyperbolic_reciprocal_result);
    }

    #[test]
    fn sqrt_result_builder_stays_inactive_without_hits() {
        let preservation = IntegrationSqrtResultPreservationBuilder::default().build();

        assert!(!preservation.should_preserve_compact_result());
        assert!(!preservation.preserve_compact_sqrt_trig_log_result);
        assert!(!preservation.preserve_compact_sqrt_hyperbolic_reciprocal_result);
    }

    #[test]
    fn sqrt_result_presentation_policy_keeps_routes_independent() {
        let policy = IntegrationSqrtResultPresentationPolicy {
            apply_sqrt_hyperbolic_log: true,
            apply_sqrt_trig_log: false,
            apply_sqrt_hyperbolic_reciprocal: true,
        };

        assert!(policy.should_apply_sqrt_hyperbolic_log());
        assert!(!policy.should_apply_sqrt_trig_log());
        assert!(policy.should_apply_sqrt_hyperbolic_reciprocal());
    }
}
