use super::RootDenestingAnalysis;
use cas_ast::{Context, Expr};
use num_rational::BigRational;

pub(super) fn compute_denesting_delta(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
) -> Option<BigRational> {
    let Expr::Number(a_num) = ctx.get(analysis.a_expr) else {
        return None;
    };
    let Expr::Number(d_num) = ctx.get(analysis.d_expr) else {
        return None;
    };

    let c_sq = &analysis.c_coeff * &analysis.c_coeff;
    Some(a_num * a_num - &c_sq * d_num)
}
