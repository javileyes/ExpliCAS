use super::{find_fraction_sum_in_expr, FractionSumInfo};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn find_all_fraction_sums_recursive(
    ctx: &Context,
    expr: ExprId,
    results: &mut Vec<FractionSumInfo>,
) {
    if let Some(info) = find_fraction_sum_in_expr(ctx, expr) {
        results.push(info);
    }

    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            find_all_fraction_sums_recursive(ctx, *l, results);
            find_all_fraction_sums_recursive(ctx, *r, results);
        }
        Expr::Pow(b, e) => {
            find_all_fraction_sums_recursive(ctx, *b, results);
            find_all_fraction_sums_recursive(ctx, *e, results);
        }
        Expr::Neg(e) | Expr::Hold(e) => find_all_fraction_sums_recursive(ctx, *e, results),
        Expr::Function(_, args) => {
            for arg in args {
                find_all_fraction_sums_recursive(ctx, *arg, results);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                find_all_fraction_sums_recursive(ctx, *elem, results);
            }
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}
