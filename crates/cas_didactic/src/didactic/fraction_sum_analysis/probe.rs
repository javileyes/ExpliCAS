use super::{FractionSumInfo, IsOne};
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;

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

fn find_fraction_sum_in_add(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    let mut terms = Vec::new();
    super::super::collect_add_terms(ctx, expr, &mut terms);

    let mut fractions = Vec::new();
    for term in &terms {
        if let Some(frac) = super::super::try_as_fraction(ctx, *term) {
            fractions.push(frac);
        } else {
            return None;
        }
    }

    if fractions.len() < 2 {
        return None;
    }

    let has_actual_fraction = fractions.iter().any(|f| !f.denom().is_one());
    if !has_actual_fraction {
        return None;
    }

    let result: BigRational = fractions.iter().cloned().sum();
    Some(FractionSumInfo { fractions, result })
}
