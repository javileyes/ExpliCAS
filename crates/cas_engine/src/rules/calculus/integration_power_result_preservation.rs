use super::half_power_sum_result_presentation::compact_half_power_sum_root_product_for_integration_presentation;
use super::negative_half_power_result_presentation::compact_negative_half_power_result_for_integration_presentation;
use super::negative_odd_half_power_result_presentation::compact_negative_three_half_power_result_for_integration_presentation;
use super::positive_half_power_result_presentation::compact_positive_half_power_result_for_integration_presentation;
use cas_ast::{Context, ExprId};

pub(super) struct IntegrationPowerResultPreservation {
    compact_half_power_sum_root_product_result: Option<ExprId>,
    compact_negative_half_power_result: Option<ExprId>,
    compact_negative_three_half_power_result: Option<ExprId>,
    compact_positive_half_power_result: Option<ExprId>,
}

impl IntegrationPowerResultPreservation {
    pub(super) fn from_result(
        ctx: &mut Context,
        result: ExprId,
        var_name: &str,
        has_required_positive_conditions: bool,
    ) -> Self {
        let compact_negative_half_power_result =
            compact_negative_half_power_result_for_integration_presentation(ctx, result);
        let compact_negative_three_half_power_result =
            compact_negative_three_half_power_result_for_integration_presentation(
                ctx,
                result,
                var_name,
                has_required_positive_conditions,
            );
        let compact_positive_half_power_result =
            compact_positive_half_power_result_for_integration_presentation(ctx, result);
        let compact_half_power_sum_root_product_result =
            compact_half_power_sum_root_product_for_integration_presentation(ctx, result, var_name);

        Self {
            compact_half_power_sum_root_product_result,
            compact_negative_half_power_result,
            compact_negative_three_half_power_result,
            compact_positive_half_power_result,
        }
    }

    pub(super) fn should_preserve_compact_result(&self) -> bool {
        self.compact_half_power_sum_root_product_result.is_some()
            || self.compact_negative_half_power_result.is_some()
            || self.compact_negative_three_half_power_result.is_some()
            || self.compact_positive_half_power_result.is_some()
    }

    pub(super) fn apply(&self, mut result: ExprId) -> ExprId {
        if let Some(compact) = self.compact_half_power_sum_root_product_result {
            result = compact;
        }
        if let Some(compact) = self.compact_negative_half_power_result {
            result = compact;
        }
        if let Some(compact) = self.compact_negative_three_half_power_result {
            result = compact;
        }
        if let Some(compact) = self.compact_positive_half_power_result {
            result = compact;
        }
        result
    }
}
