mod add_terms;
mod binomial;
mod grouping;
mod latex;
mod product;

use crate::cas_solver::Step;
use cas_ast::{Context, ExprId};

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
    latex::rationalization_latex(ctx, hints, id)
}

fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    add_terms::collect_add_terms(ctx, expr)
}
