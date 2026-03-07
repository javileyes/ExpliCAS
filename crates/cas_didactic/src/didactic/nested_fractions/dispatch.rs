use super::super::nested_fraction_analysis::NestedFractionPattern;
use super::super::SubStep;
use cas_ast::{Context, ExprId};

pub(super) fn generate_nested_fraction_substeps_for_pattern(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    pattern: NestedFractionPattern,
    hints: &cas_formatter::DisplayContext,
    generate_general_nested_fraction_substeps: fn(
        &Context,
        ExprId,
        ExprId,
        &cas_formatter::DisplayContext,
    ) -> Vec<SubStep>,
    generate_structured_nested_fraction_substeps: fn(
        &Context,
        ExprId,
        ExprId,
        NestedFractionPattern,
        &cas_formatter::DisplayContext,
    ) -> Vec<SubStep>,
) -> Vec<SubStep> {
    if matches!(pattern, NestedFractionPattern::General) {
        return generate_general_nested_fraction_substeps(ctx, before_expr, after_expr, hints);
    }

    generate_structured_nested_fraction_substeps(ctx, before_expr, after_expr, pattern, hints)
}
