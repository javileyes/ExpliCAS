mod apply;
mod delta;
mod identify;

use super::super::super::SubStep;
use super::super::analysis::RootDenestingAnalysis;
use cas_ast::{Context, ExprId};
use num_rational::BigRational;

pub(super) fn build_identify_denesting_substep(
    ctx: &Context,
    before_expr: ExprId,
    analysis: &RootDenestingAnalysis,
) -> SubStep {
    identify::build_identify_denesting_substep(ctx, before_expr, analysis)
}

pub(super) fn build_denesting_delta_substep(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
    delta: &BigRational,
) -> SubStep {
    delta::build_denesting_delta_substep(ctx, analysis, delta)
}

pub(super) fn build_apply_denesting_substep(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
    after_expr: ExprId,
) -> SubStep {
    apply::build_apply_denesting_substep(ctx, analysis, after_expr)
}
