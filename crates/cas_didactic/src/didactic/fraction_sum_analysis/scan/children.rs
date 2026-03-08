use super::super::FractionSumInfo;
use cas_ast::{Context, Expr, ExprId};

pub(super) fn scan_fraction_sum_children(
    ctx: &Context,
    expr: ExprId,
    results: &mut Vec<FractionSumInfo>,
    recurse: fn(&Context, ExprId, &mut Vec<FractionSumInfo>),
) {
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            recurse(ctx, *l, results);
            recurse(ctx, *r, results);
        }
        Expr::Pow(b, e) => {
            recurse(ctx, *b, results);
            recurse(ctx, *e, results);
        }
        Expr::Neg(e) | Expr::Hold(e) => recurse(ctx, *e, results),
        Expr::Function(_, args) => {
            for arg in args {
                recurse(ctx, *arg, results);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                recurse(ctx, *elem, results);
            }
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}
