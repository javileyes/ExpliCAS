mod delta;
mod sqrt;
mod surd;
mod terms;

use self::sqrt::get_sqrt_inner;
use self::surd::analyze_surd;
use cas_ast::{Context, ExprId};
use num_rational::BigRational;

pub(super) struct RootDenestingAnalysis {
    pub inner_expr: ExprId,
    pub a_expr: ExprId,
    pub c_coeff: BigRational,
    pub d_expr: ExprId,
    pub is_add: bool,
}

pub(super) fn analyze_root_denesting(
    ctx: &Context,
    before_expr: ExprId,
) -> Option<RootDenestingAnalysis> {
    let inner_expr = get_sqrt_inner(ctx, before_expr)?;
    let (a_term, b_term, is_add) = terms::extract_denesting_terms(ctx, inner_expr)?;
    terms::build_root_denesting_analysis(ctx, inner_expr, a_term, b_term, is_add, analyze_surd)
}

pub(super) fn compute_denesting_delta(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
) -> Option<BigRational> {
    delta::compute_denesting_delta(ctx, analysis)
}
