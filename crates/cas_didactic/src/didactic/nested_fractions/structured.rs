mod denominator_sum;
mod scalar_division;

use super::super::nested_fraction_analysis::NestedFractionPattern;
use super::SubStep;
use cas_ast::{Context, ExprId};

pub(super) fn generate_structured_nested_fraction_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    pattern: NestedFractionPattern,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    match pattern {
        NestedFractionPattern::OneOverSumWithUnitFraction
        | NestedFractionPattern::OneOverSumWithFraction => {
            denominator_sum::generate_one_over_sum_substeps(ctx, before_expr, after_expr, hints)
        }
        NestedFractionPattern::FractionOverSumWithFraction => {
            denominator_sum::generate_fraction_over_sum_substeps(
                ctx,
                before_expr,
                after_expr,
                hints,
            )
        }
        NestedFractionPattern::SumWithFractionOverScalar => {
            scalar_division::generate_sum_over_scalar_substeps(ctx, before_expr, after_expr, hints)
        }
        NestedFractionPattern::General => Vec::new(),
    }
}
