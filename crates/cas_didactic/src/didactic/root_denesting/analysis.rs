mod sqrt;
mod surd;

use self::sqrt::get_sqrt_inner;
use self::surd::analyze_surd;
use cas_ast::{Context, Expr, ExprId};
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

    let (a_term, b_term, is_add) = match ctx.get(inner_expr) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return None,
    };

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
