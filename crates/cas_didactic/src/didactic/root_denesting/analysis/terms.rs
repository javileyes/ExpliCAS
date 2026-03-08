mod analysis;
mod extract;

use super::RootDenestingAnalysis;
use cas_ast::{Context, ExprId};
use num_rational::BigRational;

pub(super) fn extract_denesting_terms(
    ctx: &Context,
    inner_expr: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    extract::extract_denesting_terms(ctx, inner_expr)
}

pub(super) fn build_root_denesting_analysis(
    ctx: &Context,
    inner_expr: ExprId,
    a_term: ExprId,
    b_term: ExprId,
    is_add: bool,
    analyze_surd: fn(&Context, ExprId) -> Option<(BigRational, ExprId)>,
) -> Option<RootDenestingAnalysis> {
    analysis::build_root_denesting_analysis(ctx, inner_expr, a_term, b_term, is_add, analyze_surd)
}
