use super::arctan_integrand_preservation::arctan_integrand_for_calculus_presentation;
use super::by_parts_integrand_preservation::{
    by_parts_integrand_preservation_gates, ByPartsIntegrandPreservation,
};
use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::hyperbolic_power_integrand_presentation::hyperbolic_power_integrand_for_calculus_presentation;
use super::integration_source_preservation::{
    IntegrationSourcePreservation, IntegrationSourcePreservationBuilder,
};
use super::inverse_hyperbolic_affine_integrand_preservation::inverse_hyperbolic_affine_integrand_for_calculus_presentation;
use super::inverse_sqrt_product_integrand_preservation::{
    affine_sqrt_product_derivative_integrand_for_calculus_presentation,
    arcsin_inverse_sqrt_product_integrand_for_calculus_presentation,
};
use super::log_product_integrand_preservation::log_product_integrand_for_calculus_presentation;
use super::rational_partial_fraction_integrand_preservation::rational_linear_partial_fraction_integrand_for_calculus_presentation;
use super::sqrt_chain_integrand_preservation::{
    sqrt_chain_integrand_preservation_gates, SqrtChainIntegrandPreservation,
};
use super::trig_power_integrand_presentation::affine_trig_power_integrand_for_calculus_presentation;
use cas_ast::{Context, ExprId};

#[derive(Default)]
struct IntegrationSourceDirectActivationRouteHits {
    preserve_compact_reciprocal: bool,
    preserve_compact_arctan_integrand: bool,
    preserve_compact_atanh_polynomial: bool,
    preserve_compact_inverse_hyperbolic_affine: bool,
    preserve_compact_bounded_inverse_trig: bool,
    preserve_compact_trig_polynomial: bool,
    preserve_compact_affine_trig_power: bool,
}

#[derive(Clone, Copy)]
struct IntegrationSourceDirectRoutePreservationPolicy {
    activate_reciprocal_presentation: bool,
    activate_inverse_family_presentation: bool,
    activate_trig_polynomial_presentation: bool,
}

impl IntegrationSourceDirectRoutePreservationPolicy {
    fn from_route_hits(direct_activation: &IntegrationSourceDirectActivationRouteHits) -> Self {
        Self {
            activate_reciprocal_presentation: direct_activation.preserve_compact_reciprocal,
            activate_inverse_family_presentation: direct_activation
                .preserve_compact_arctan_integrand
                || direct_activation.preserve_compact_atanh_polynomial
                || direct_activation.preserve_compact_inverse_hyperbolic_affine
                || direct_activation.preserve_compact_bounded_inverse_trig,
            activate_trig_polynomial_presentation: direct_activation
                .preserve_compact_trig_polynomial
                || direct_activation.preserve_compact_affine_trig_power,
        }
    }

    fn activate_direct_presentation(&self) -> bool {
        self.activate_reciprocal_presentation
            || self.activate_inverse_family_presentation
            || self.activate_trig_polynomial_presentation
    }
}

#[derive(Default)]
struct IntegrationSourceSqrtChainRouteHits {
    active: bool,
    preserve_compact_sqrt_trig_log: bool,
    preserve_compact_sqrt_hyperbolic_log: bool,
    preserve_compact_sqrt_hyperbolic_reciprocal_derivative: bool,
}

impl IntegrationSourceSqrtChainRouteHits {
    fn from_preservation(preservation: SqrtChainIntegrandPreservation) -> Self {
        Self {
            active: preservation.should_preserve_compact_result(),
            preserve_compact_sqrt_trig_log: preservation.preserve_compact_sqrt_trig_log,
            preserve_compact_sqrt_hyperbolic_log: preservation.preserve_compact_sqrt_hyperbolic_log,
            preserve_compact_sqrt_hyperbolic_reciprocal_derivative: preservation
                .preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
        }
    }
}

#[derive(Default)]
struct IntegrationSourceByPartsRouteHits {
    active: bool,
    preserve_compact_log_by_parts: bool,
}

impl IntegrationSourceByPartsRouteHits {
    fn from_preservation(preservation: ByPartsIntegrandPreservation) -> Self {
        Self {
            active: preservation.should_preserve_compact_result(),
            preserve_compact_log_by_parts: preservation.preserve_compact_log_by_parts,
        }
    }
}

#[derive(Default)]
struct IntegrationSourceInverseSqrtProductRouteHits {
    preserve_compact_inverse_hyperbolic_sqrt_reciprocal: bool,
    preserve_compact_affine_sqrt_product_derivative: bool,
    preserve_compact_arcsin_inverse_sqrt_product: bool,
}

#[derive(Clone, Copy)]
struct IntegrationSourceInverseSqrtProductRoutePreservationPolicy {
    activate_inverse_root_family_presentation: bool,
    activate_affine_sqrt_product_derivative_presentation: bool,
}

impl IntegrationSourceInverseSqrtProductRoutePreservationPolicy {
    fn from_route_hits(
        inverse_sqrt_product: &IntegrationSourceInverseSqrtProductRouteHits,
    ) -> Self {
        Self {
            activate_inverse_root_family_presentation: inverse_sqrt_product
                .preserve_compact_inverse_hyperbolic_sqrt_reciprocal
                || inverse_sqrt_product.preserve_compact_arcsin_inverse_sqrt_product,
            activate_affine_sqrt_product_derivative_presentation: inverse_sqrt_product
                .preserve_compact_affine_sqrt_product_derivative,
        }
    }

    fn activate_inverse_sqrt_product_presentation(&self) -> bool {
        self.activate_inverse_root_family_presentation
            || self.activate_affine_sqrt_product_derivative_presentation
    }
}

#[derive(Default)]
struct IntegrationSourceLateActivationRouteHits {
    preserve_compact_log_product_integrand: bool,
    preserve_compact_hyperbolic_power: bool,
}

impl IntegrationSourceLateActivationRouteHits {
    fn active(&self) -> bool {
        self.preserve_compact_log_product_integrand || self.preserve_compact_hyperbolic_power
    }
}

#[derive(Default)]
struct IntegrationSourceRationalPartialFractionRouteHits {
    preserve_compact_rational_linear_partial_fraction: bool,
}

#[derive(Clone, Copy)]
struct IntegrationSourceLateRoutePreservationPolicy {
    activate_late_presentation: bool,
    preserve_rational_partial_fraction: bool,
    by_parts_active: bool,
    preserve_log_by_parts: bool,
}

impl IntegrationSourceLateRoutePreservationPolicy {
    fn from_route_hits(
        late_activation: &IntegrationSourceLateActivationRouteHits,
        rational_partial_fraction: &IntegrationSourceRationalPartialFractionRouteHits,
        by_parts: &IntegrationSourceByPartsRouteHits,
    ) -> Self {
        Self {
            activate_late_presentation: late_activation.active(),
            preserve_rational_partial_fraction: rational_partial_fraction
                .preserve_compact_rational_linear_partial_fraction,
            by_parts_active: by_parts.active,
            preserve_log_by_parts: by_parts.preserve_compact_log_by_parts,
        }
    }

    fn activate_late_presentation(&self) -> bool {
        self.activate_late_presentation
    }

    fn preserve_rational_partial_fraction(&self) -> bool {
        self.preserve_rational_partial_fraction
    }

    fn by_parts_active(&self) -> bool {
        self.by_parts_active
    }

    fn preserve_log_by_parts(&self) -> bool {
        self.preserve_log_by_parts
    }
}

#[derive(Default)]
pub(super) struct IntegrationSourcePreservationRouteHits {
    direct_activation: IntegrationSourceDirectActivationRouteHits,
    preserve_compact_fractional_denominator_power: bool,
    sqrt_chain: IntegrationSourceSqrtChainRouteHits,
    inverse_sqrt_product: IntegrationSourceInverseSqrtProductRouteHits,
    late_activation: IntegrationSourceLateActivationRouteHits,
    rational_partial_fraction: IntegrationSourceRationalPartialFractionRouteHits,
    by_parts: IntegrationSourceByPartsRouteHits,
}

#[derive(Default)]
struct IntegrationSourceRouteHitScan {
    hits: IntegrationSourcePreservationRouteHits,
}

impl IntegrationSourceRouteHitScan {
    fn record_direct_route_hits(&mut self, ctx: &mut Context, target: ExprId, var_name: &str) {
        self.record_compact_reciprocal(
            cas_math::symbolic_integration_support::integrate_symbolic_is_reciprocal_negative_power_denominator_quotient_target(
                ctx, target, var_name,
            ),
        );
        self.record_compact_arctan_integrand(arctan_integrand_for_calculus_presentation(
            ctx, target, var_name,
        ));
        self.record_compact_atanh_polynomial(
            cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_polynomial_substitution_target(
                ctx, target, var_name,
            ),
        );
        self.record_compact_inverse_hyperbolic_affine(
            inverse_hyperbolic_affine_integrand_for_calculus_presentation(ctx, target, var_name),
        );
        self.record_compact_bounded_inverse_trig(
            cas_math::symbolic_integration_support::integrate_symbolic_is_bounded_inverse_trig_variable_target(
                ctx, target, var_name,
            ),
        );
        self.record_compact_trig_polynomial(
            cas_math::symbolic_integration_support::integrate_symbolic_is_trig_polynomial_substitution_target(
                ctx, target, var_name,
            ),
        );
        self.record_compact_affine_trig_power(
            affine_trig_power_integrand_for_calculus_presentation(ctx, target, var_name),
        );
    }

    fn record_sqrt_chain_routes(&mut self, ctx: &mut Context, target: ExprId, var_name: &str) {
        self.record_sqrt_chain(sqrt_chain_integrand_preservation_gates(
            ctx, target, var_name,
        ));
    }

    fn record_inverse_sqrt_product_routes(
        &mut self,
        ctx: &mut Context,
        target: ExprId,
        var_name: &str,
    ) {
        self.record_compact_inverse_hyperbolic_sqrt_reciprocal(
            cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
                ctx,
                target,
                var_name,
            ),
        );
        self.record_compact_affine_sqrt_product_derivative(
            affine_sqrt_product_derivative_integrand_for_calculus_presentation(
                ctx, target, var_name,
            ),
        );
        self.record_compact_arcsin_inverse_sqrt_product(
            arcsin_inverse_sqrt_product_integrand_for_calculus_presentation(ctx, target, var_name),
        );
    }

    fn record_late_route_hits(&mut self, ctx: &mut Context, target: ExprId, var_name: &str) {
        self.record_compact_log_product_integrand(log_product_integrand_for_calculus_presentation(
            ctx, target, var_name,
        ));
        self.record_compact_rational_linear_partial_fraction(
            rational_linear_partial_fraction_integrand_for_calculus_presentation(
                ctx, target, var_name,
            ),
        );
        self.record_compact_hyperbolic_power(hyperbolic_power_integrand_for_calculus_presentation(
            ctx, target, var_name,
        ));
        self.record_by_parts(by_parts_integrand_preservation_gates(ctx, target, var_name));
    }

    fn record_compact_reciprocal(&mut self, hit: bool) {
        self.hits.direct_activation.preserve_compact_reciprocal = hit;
    }

    fn record_fractional_denominator_power(&mut self, hit: bool) {
        self.hits.preserve_compact_fractional_denominator_power = hit;
    }

    fn record_compact_arctan_integrand(&mut self, hit: bool) {
        self.hits
            .direct_activation
            .preserve_compact_arctan_integrand = hit;
    }

    fn record_compact_atanh_polynomial(&mut self, hit: bool) {
        self.hits
            .direct_activation
            .preserve_compact_atanh_polynomial = hit;
    }

    fn record_compact_inverse_hyperbolic_affine(&mut self, hit: bool) {
        self.hits
            .direct_activation
            .preserve_compact_inverse_hyperbolic_affine = hit;
    }

    fn record_compact_bounded_inverse_trig(&mut self, hit: bool) {
        self.hits
            .direct_activation
            .preserve_compact_bounded_inverse_trig = hit;
    }

    fn record_compact_trig_polynomial(&mut self, hit: bool) {
        self.hits.direct_activation.preserve_compact_trig_polynomial = hit;
    }

    fn record_compact_affine_trig_power(&mut self, hit: bool) {
        self.hits
            .direct_activation
            .preserve_compact_affine_trig_power = hit;
    }

    fn record_sqrt_chain(&mut self, preservation: SqrtChainIntegrandPreservation) {
        self.hits.sqrt_chain = IntegrationSourceSqrtChainRouteHits::from_preservation(preservation);
    }

    fn record_compact_inverse_hyperbolic_sqrt_reciprocal(&mut self, hit: bool) {
        self.hits
            .inverse_sqrt_product
            .preserve_compact_inverse_hyperbolic_sqrt_reciprocal = hit;
    }

    fn record_compact_affine_sqrt_product_derivative(&mut self, hit: bool) {
        self.hits
            .inverse_sqrt_product
            .preserve_compact_affine_sqrt_product_derivative = hit;
    }

    fn record_compact_arcsin_inverse_sqrt_product(&mut self, hit: bool) {
        self.hits
            .inverse_sqrt_product
            .preserve_compact_arcsin_inverse_sqrt_product = hit;
    }

    fn record_compact_log_product_integrand(&mut self, hit: bool) {
        self.hits
            .late_activation
            .preserve_compact_log_product_integrand = hit;
    }

    fn record_compact_rational_linear_partial_fraction(&mut self, hit: bool) {
        self.hits
            .rational_partial_fraction
            .preserve_compact_rational_linear_partial_fraction = hit;
    }

    fn record_compact_hyperbolic_power(&mut self, hit: bool) {
        self.hits.late_activation.preserve_compact_hyperbolic_power = hit;
    }

    fn record_by_parts(&mut self, preservation: ByPartsIntegrandPreservation) {
        self.hits.by_parts = IntegrationSourceByPartsRouteHits::from_preservation(preservation);
    }

    fn finish(self) -> IntegrationSourcePreservationRouteHits {
        self.hits
    }
}

impl IntegrationSourcePreservationRouteHits {
    pub(super) fn from_target(ctx: &mut Context, target: ExprId, var_name: &str) -> Self {
        let mut scan = IntegrationSourceRouteHitScan::default();

        scan.record_direct_route_hits(ctx, target, var_name);
        scan.record_fractional_denominator_power(
            fractional_denominator_power_substitution_integrand_for_calculus_presentation(
                ctx, target, var_name,
            ),
        );
        scan.record_sqrt_chain_routes(ctx, target, var_name);
        scan.record_inverse_sqrt_product_routes(ctx, target, var_name);
        scan.record_late_route_hits(ctx, target, var_name);

        scan.finish()
    }

    pub(super) fn into_preservation(self) -> IntegrationSourcePreservation {
        let direct_policy = IntegrationSourceDirectRoutePreservationPolicy::from_route_hits(
            &self.direct_activation,
        );
        let late_policy = IntegrationSourceLateRoutePreservationPolicy::from_route_hits(
            &self.late_activation,
            &self.rational_partial_fraction,
            &self.by_parts,
        );
        let inverse_sqrt_product_policy =
            IntegrationSourceInverseSqrtProductRoutePreservationPolicy::from_route_hits(
                &self.inverse_sqrt_product,
            );

        IntegrationSourcePreservationBuilder::default()
            .record_activation(direct_policy.activate_direct_presentation())
            .record_fractional_denominator_power(self.preserve_compact_fractional_denominator_power)
            .record_sqrt_chain_flags(
                self.sqrt_chain.active,
                self.sqrt_chain.preserve_compact_sqrt_trig_log,
                self.sqrt_chain.preserve_compact_sqrt_hyperbolic_log,
                self.sqrt_chain
                    .preserve_compact_sqrt_hyperbolic_reciprocal_derivative,
            )
            .record_activation(
                inverse_sqrt_product_policy.activate_inverse_sqrt_product_presentation(),
            )
            .record_activation(late_policy.activate_late_presentation())
            .record_rational_partial_fraction(late_policy.preserve_rational_partial_fraction())
            .record_by_parts_flags(
                late_policy.by_parts_active(),
                late_policy.preserve_log_by_parts(),
            )
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        IntegrationSourceByPartsRouteHits, IntegrationSourceDirectActivationRouteHits,
        IntegrationSourceDirectRoutePreservationPolicy,
        IntegrationSourceInverseSqrtProductRouteHits,
        IntegrationSourceInverseSqrtProductRoutePreservationPolicy,
        IntegrationSourceLateActivationRouteHits, IntegrationSourceLateRoutePreservationPolicy,
        IntegrationSourcePreservationRouteHits, IntegrationSourceRationalPartialFractionRouteHits,
        IntegrationSourceSqrtChainRouteHits,
    };

    #[test]
    fn source_route_hits_preserve_sqrt_chain_flags() {
        let preservation = IntegrationSourcePreservationRouteHits {
            sqrt_chain: IntegrationSourceSqrtChainRouteHits {
                active: true,
                preserve_compact_sqrt_hyperbolic_log: true,
                ..Default::default()
            },
            ..Default::default()
        }
        .into_preservation();

        assert!(preservation.should_preserve_compact_result());
        assert!(preservation.preserve_compact_sqrt_hyperbolic_log);
        assert!(!preservation.preserve_compact_sqrt_trig_log);
    }

    #[test]
    fn source_route_hits_preserve_direct_activation_without_fielded_flags() {
        let preservation = IntegrationSourcePreservationRouteHits {
            direct_activation: IntegrationSourceDirectActivationRouteHits {
                preserve_compact_arctan_integrand: true,
                ..Default::default()
            },
            ..Default::default()
        }
        .into_preservation();

        assert!(preservation.should_preserve_compact_result());
        assert!(!preservation.preserve_compact_fractional_denominator_power);
        assert!(!preservation.preserve_compact_rational_linear_partial_fraction);
        assert!(!preservation.preserve_compact_log_by_parts);
    }

    #[test]
    fn source_direct_route_policy_groups_direct_activation_families() {
        let policy = IntegrationSourceDirectRoutePreservationPolicy::from_route_hits(
            &IntegrationSourceDirectActivationRouteHits {
                preserve_compact_atanh_polynomial: true,
                preserve_compact_trig_polynomial: true,
                ..Default::default()
            },
        );

        assert!(policy.activate_direct_presentation());
        assert!(!policy.activate_reciprocal_presentation);
        assert!(policy.activate_inverse_family_presentation);
        assert!(policy.activate_trig_polynomial_presentation);
    }

    #[test]
    fn source_route_hits_preserve_by_parts_flags() {
        let preservation = IntegrationSourcePreservationRouteHits {
            by_parts: IntegrationSourceByPartsRouteHits {
                active: true,
                preserve_compact_log_by_parts: true,
            },
            ..Default::default()
        }
        .into_preservation();

        assert!(preservation.should_preserve_compact_result());
        assert!(preservation.preserve_compact_log_by_parts);
    }

    #[test]
    fn source_route_hits_preserve_rational_partial_fraction_flags() {
        let preservation = IntegrationSourcePreservationRouteHits {
            rational_partial_fraction: IntegrationSourceRationalPartialFractionRouteHits {
                preserve_compact_rational_linear_partial_fraction: true,
            },
            ..Default::default()
        }
        .into_preservation();

        assert!(preservation.should_preserve_compact_result());
        assert!(preservation.preserve_compact_rational_linear_partial_fraction);
        assert!(!preservation.preserve_compact_log_by_parts);
    }

    #[test]
    fn source_route_hits_preserve_inverse_sqrt_product_activation() {
        let preservation = IntegrationSourcePreservationRouteHits {
            inverse_sqrt_product: IntegrationSourceInverseSqrtProductRouteHits {
                preserve_compact_affine_sqrt_product_derivative: true,
                ..Default::default()
            },
            ..Default::default()
        }
        .into_preservation();

        assert!(preservation.should_preserve_compact_result());
        assert!(!preservation.preserve_compact_sqrt_trig_log);
        assert!(!preservation.preserve_compact_log_by_parts);
    }

    #[test]
    fn source_inverse_sqrt_product_route_policy_splits_primitive_families() {
        let policy = IntegrationSourceInverseSqrtProductRoutePreservationPolicy::from_route_hits(
            &IntegrationSourceInverseSqrtProductRouteHits {
                preserve_compact_inverse_hyperbolic_sqrt_reciprocal: true,
                preserve_compact_affine_sqrt_product_derivative: false,
                preserve_compact_arcsin_inverse_sqrt_product: true,
            },
        );

        assert!(policy.activate_inverse_sqrt_product_presentation());
        assert!(policy.activate_inverse_root_family_presentation);
        assert!(!policy.activate_affine_sqrt_product_derivative_presentation);
    }

    #[test]
    fn source_route_hits_preserve_late_activation_without_fielded_flags() {
        let preservation = IntegrationSourcePreservationRouteHits {
            late_activation: IntegrationSourceLateActivationRouteHits {
                preserve_compact_log_product_integrand: true,
                ..Default::default()
            },
            ..Default::default()
        }
        .into_preservation();

        assert!(preservation.should_preserve_compact_result());
        assert!(!preservation.preserve_compact_rational_linear_partial_fraction);
        assert!(!preservation.preserve_compact_log_by_parts);
    }

    #[test]
    fn source_late_route_policy_keeps_specific_preservations_independent() {
        let policy = IntegrationSourceLateRoutePreservationPolicy::from_route_hits(
            &IntegrationSourceLateActivationRouteHits {
                preserve_compact_log_product_integrand: true,
                preserve_compact_hyperbolic_power: false,
            },
            &IntegrationSourceRationalPartialFractionRouteHits {
                preserve_compact_rational_linear_partial_fraction: false,
            },
            &IntegrationSourceByPartsRouteHits {
                active: true,
                preserve_compact_log_by_parts: true,
            },
        );

        assert!(policy.activate_late_presentation());
        assert!(!policy.preserve_rational_partial_fraction());
        assert!(policy.by_parts_active());
        assert!(policy.preserve_log_by_parts());
    }
}
