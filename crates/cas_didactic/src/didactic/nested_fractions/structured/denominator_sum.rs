mod fraction_over;
mod one_over;

use super::super::{nested_fraction_latex, SubStep};
use cas_ast::{Context, ExprId};

pub(super) fn generate_one_over_sum_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    one_over::generate_one_over_sum_substeps(
        ctx,
        before_expr,
        after_expr,
        hints,
        nested_fraction_latex,
    )
}

pub(super) fn generate_fraction_over_sum_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    fraction_over::generate_fraction_over_sum_substeps(
        ctx,
        before_expr,
        after_expr,
        hints,
        nested_fraction_latex,
    )
}
