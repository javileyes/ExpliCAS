use super::super::RootDenestingAnalysis;
use cas_ast::{Context, ExprId};
use num_rational::BigRational;

pub(super) fn build_root_denesting_analysis(
    ctx: &Context,
    inner_expr: ExprId,
    a_term: ExprId,
    b_term: ExprId,
    is_add: bool,
    analyze_surd: fn(&Context, ExprId) -> Option<(BigRational, ExprId)>,
) -> Option<RootDenestingAnalysis> {
    if let Some((coeff, rad)) = analyze_surd(ctx, a_term) {
        return Some(RootDenestingAnalysis {
            inner_expr,
            a_expr: b_term,
            c_coeff: coeff,
            d_expr: rad,
            is_add,
        });
    }

    if let Some((coeff, rad)) = analyze_surd(ctx, b_term) {
        return Some(RootDenestingAnalysis {
            inner_expr,
            a_expr: a_term,
            c_coeff: coeff,
            d_expr: rad,
            is_add,
        });
    }

    None
}
