use super::arctan_additive_result_presentation::{
    compact_arctan_additive_terms_for_calculus_presentation,
    flatten_subtracting_additive_group_for_calculus_presentation,
};
use super::integration_power_result_preservation::IntegrationPowerResultPreservation;
use super::integration_source_preservation::IntegrationSourcePreservation;
use super::integration_sqrt_result_preservation::IntegrationSqrtResultPreservation;
use super::scalar_presentation::{
    fold_numeric_mul_constants_for_hold, fold_numeric_mul_constants_for_hold_additive_terms,
};
use cas_ast::{Context, Expr, ExprId};

#[derive(Clone, Copy)]
struct IntegrationResultPreservationActivation {
    source_preservation_active: bool,
    preserve_compact_polynomial_arctan_by_parts_result: bool,
    power_result_preservation_active: bool,
    sqrt_result_preservation_active: bool,
}

impl IntegrationResultPreservationActivation {
    fn should_apply_compact_cleanup(&self) -> bool {
        self.source_preservation_active
            || self.preserve_compact_polynomial_arctan_by_parts_result
            || self.power_result_preservation_active
            || self.sqrt_result_preservation_active
    }
}

struct IntegrationResultPreservationPlan<'a> {
    source_preservation: &'a IntegrationSourcePreservation,
    preserve_compact_polynomial_arctan_by_parts_result: bool,
    compact_polynomial_arctan_by_parts_result: Option<ExprId>,
    power_result_preservation: IntegrationPowerResultPreservation,
    sqrt_result_preservation: IntegrationSqrtResultPreservation,
}

impl<'a> IntegrationResultPreservationPlan<'a> {
    fn from_result(
        ctx: &mut Context,
        result: ExprId,
        var_name: &str,
        has_required_positive_conditions: bool,
        source_preservation: &'a IntegrationSourcePreservation,
        compact_polynomial_arctan_by_parts_result: Option<ExprId>,
    ) -> Self {
        let preserve_compact_polynomial_arctan_by_parts_result =
            compact_polynomial_arctan_by_parts_result.is_some();
        let power_result_preservation = IntegrationPowerResultPreservation::from_result(
            ctx,
            result,
            var_name,
            has_required_positive_conditions,
        );
        let sqrt_result_preservation =
            IntegrationSqrtResultPreservation::from_result(ctx, result, var_name);

        Self {
            source_preservation,
            preserve_compact_polynomial_arctan_by_parts_result,
            compact_polynomial_arctan_by_parts_result,
            power_result_preservation,
            sqrt_result_preservation,
        }
    }

    fn activation(&self) -> IntegrationResultPreservationActivation {
        IntegrationResultPreservationActivation {
            source_preservation_active: self.source_preservation.should_preserve_compact_result(),
            preserve_compact_polynomial_arctan_by_parts_result: self
                .preserve_compact_polynomial_arctan_by_parts_result,
            power_result_preservation_active: self
                .power_result_preservation
                .should_preserve_compact_result(),
            sqrt_result_preservation_active: self
                .sqrt_result_preservation
                .should_preserve_compact_result(),
        }
    }

    fn should_apply_compact_cleanup(&self) -> bool {
        self.activation().should_apply_compact_cleanup()
    }

    fn apply_compact_cleanup(
        &self,
        ctx: &mut Context,
        mut result: ExprId,
        var_name: &str,
    ) -> ExprId {
        result = self.power_result_preservation.apply(result);
        result =
            self.sqrt_result_preservation
                .apply(ctx, result, var_name, self.source_preservation);
        if let Some(compact) = self.compact_polynomial_arctan_by_parts_result {
            result = compact;
        }
        result = if self
            .source_preservation
            .preserve_compact_rational_linear_partial_fraction
        {
            fold_numeric_mul_constants_for_hold_additive_terms(ctx, result)
        } else {
            fold_numeric_mul_constants_for_hold(ctx, result)
        };
        if self.preserve_compact_polynomial_arctan_by_parts_result {
            if let Some(compact) =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, result, var_name)
            {
                result = compact;
            }
        }
        if self.source_preservation.preserve_compact_log_by_parts {
            if let Some(compact) =
                flatten_subtracting_additive_group_for_calculus_presentation(ctx, result, var_name)
            {
                result = compact;
            }
        }
        if self
            .source_preservation
            .preserve_compact_fractional_denominator_power
        {
            ctx.add(Expr::Hold(result))
        } else {
            cas_ast::hold::wrap_hold(ctx, result)
        }
    }
}

pub(super) fn apply_integration_result_preservation(
    ctx: &mut Context,
    mut result: ExprId,
    var_name: &str,
    has_required_positive_conditions: bool,
    source_preservation: &IntegrationSourcePreservation,
    compact_polynomial_arctan_by_parts_result: Option<ExprId>,
) -> ExprId {
    let plan = IntegrationResultPreservationPlan::from_result(
        ctx,
        result,
        var_name,
        has_required_positive_conditions,
        source_preservation,
        compact_polynomial_arctan_by_parts_result,
    );
    if plan.should_apply_compact_cleanup() {
        result = plan.apply_compact_cleanup(ctx, result, var_name);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::IntegrationResultPreservationActivation;

    #[test]
    fn result_preservation_activation_applies_for_any_route_hit() {
        let activation = IntegrationResultPreservationActivation {
            source_preservation_active: false,
            preserve_compact_polynomial_arctan_by_parts_result: false,
            power_result_preservation_active: false,
            sqrt_result_preservation_active: true,
        };

        assert!(activation.should_apply_compact_cleanup());
    }

    #[test]
    fn result_preservation_activation_stays_inactive_without_hits() {
        let activation = IntegrationResultPreservationActivation {
            source_preservation_active: false,
            preserve_compact_polynomial_arctan_by_parts_result: false,
            power_result_preservation_active: false,
            sqrt_result_preservation_active: false,
        };

        assert!(!activation.should_apply_compact_cleanup());
    }
}
