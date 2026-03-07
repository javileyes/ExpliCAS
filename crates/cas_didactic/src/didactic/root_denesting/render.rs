mod latex;
mod substeps;

use super::super::SubStep;
use super::analysis::RootDenestingAnalysis;
use cas_ast::{Context, ExprId};
use num_rational::BigRational;

pub(super) fn build_identify_denesting_substep(
    ctx: &Context,
    before_expr: ExprId,
    analysis: &RootDenestingAnalysis,
) -> SubStep {
    substeps::build_identify_denesting_substep(ctx, before_expr, analysis)
}

pub(super) fn build_denesting_delta_substep(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
    delta: &BigRational,
) -> SubStep {
    substeps::build_denesting_delta_substep(ctx, analysis, delta)
}

pub(super) fn build_apply_denesting_substep(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
    after_expr: ExprId,
) -> SubStep {
    substeps::build_apply_denesting_substep(ctx, analysis, after_expr)
}
