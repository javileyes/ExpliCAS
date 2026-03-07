use cas_ast::{Context, Expr, ExprId};
use cas_solver::Step;
use num_rational::BigRational;

use super::IsOne;

/// Information about a fraction sum that was computed.
#[derive(Debug)]
pub(crate) struct FractionSumInfo {
    /// The fractions that were summed.
    pub fractions: Vec<BigRational>,
    /// The result of the sum.
    pub result: BigRational,
}

/// Find all fraction sums in an expression tree.
pub(crate) fn find_all_fraction_sums(ctx: &Context, expr: ExprId) -> Vec<FractionSumInfo> {
    let mut results = Vec::new();
    find_all_fraction_sums_recursive(ctx, expr, &mut results);
    results
}

fn find_all_fraction_sums_recursive(
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

/// Detect if between this step and the previous one, an exponent changed
/// due to fraction arithmetic.
pub(crate) fn detect_exponent_fraction_change(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
) -> Option<FractionSumInfo> {
    let current_step = &steps[step_idx];

    if current_step.rule_name.contains("Inverse") || current_step.rule_name.contains("Power") {
        let global_expr = if step_idx > 0 {
            steps[step_idx - 1]
                .global_after
                .unwrap_or(steps[step_idx - 1].after)
        } else {
            current_step.global_after.unwrap_or(current_step.before)
        };

        if let Some(info) = find_fraction_sum_in_expr(ctx, global_expr) {
            return Some(info);
        }
    }

    None
}

fn find_fraction_sum_in_expr(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    match ctx.get(expr) {
        Expr::Add(_, _) => {
            let mut terms = Vec::new();
            super::collect_add_terms(ctx, expr, &mut terms);

            let mut fractions = Vec::new();
            for term in &terms {
                if let Some(frac) = super::try_as_fraction(ctx, *term) {
                    fractions.push(frac);
                } else {
                    return None;
                }
            }

            if fractions.len() >= 2 {
                let has_actual_fraction = fractions.iter().any(|f| !f.denom().is_one());
                if !has_actual_fraction {
                    return None;
                }
                let result: BigRational = fractions.iter().cloned().sum();
                return Some(FractionSumInfo { fractions, result });
            }
            None
        }
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
