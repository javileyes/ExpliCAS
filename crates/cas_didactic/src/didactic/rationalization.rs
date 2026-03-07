mod binomial;
mod grouping;
mod product;

use cas_ast::{Context, Expr, ExprId};
use cas_solver::Step;

use super::SubStep;

/// Generate sub-steps explaining rationalization process.
/// Uses LaTeXExprWithHints for proper sqrt notation rendering.
pub(crate) fn generate_rationalization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let hints = cas_formatter::DisplayContext::with_root_index(2);

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if step.description.contains("group") {
        return grouping::generate_grouped_rationalization_substeps(ctx, before, after, &hints);
    }

    if step.description.contains("product") {
        return product::generate_product_rationalization_substeps(ctx, before, after, &hints);
    }

    binomial::generate_binomial_rationalization_substeps(ctx, before, after, &hints)
}

fn rationalization_latex(
    ctx: &Context,
    hints: &cas_formatter::DisplayContext,
    id: ExprId,
) -> String {
    cas_formatter::LaTeXExprWithHints {
        context: ctx,
        id,
        hints,
    }
    .to_latex()
}

fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_add_terms_recursive(ctx, expr, &mut terms);
    terms
}

fn collect_add_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_add_terms_recursive(ctx, *l, terms);
            collect_add_terms_recursive(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}
