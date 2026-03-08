use super::integers::IsOne;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Zero;

pub(super) fn try_as_fraction(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Div(numer, denom) => {
            if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*numer), ctx.get(*denom)) {
                if !d.is_zero() {
                    return Some(n / d);
                }
            }
            None
        }
        _ => None,
    }
}

pub(super) fn format_fraction(r: &BigRational) -> String {
    if r.denom().is_one() {
        format!("{}", r.numer())
    } else {
        format!("\\frac{{{}}}{{{}}}", r.numer(), r.denom())
    }
}
