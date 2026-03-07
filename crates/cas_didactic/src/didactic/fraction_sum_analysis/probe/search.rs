use super::add::find_fraction_sum_in_add;
use super::FractionSumInfo;
use cas_ast::{Context, Expr, ExprId};

pub(super) fn find_fraction_sum_in_expr(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    match ctx.get(expr) {
        Expr::Add(_, _) => find_fraction_sum_in_add(ctx, expr),
        Expr::Pow(_, e) => find_fraction_sum_in_expr(ctx, *e),
        Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Sub(l, r) => {
            find_fraction_sum_in_expr(ctx, *l).or_else(|| find_fraction_sum_in_expr(ctx, *r))
        }
        Expr::Neg(e) | Expr::Hold(e) => find_fraction_sum_in_expr(ctx, *e),
        Expr::Function(_, args) => {
            for arg in args {
                if let Some(info) = find_fraction_sum_in_expr(ctx, *arg) {
                    return Some(info);
                }
            }
            None
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                if let Some(info) = find_fraction_sum_in_expr(ctx, *elem) {
                    return Some(info);
                }
            }
            None
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => None,
    }
}
