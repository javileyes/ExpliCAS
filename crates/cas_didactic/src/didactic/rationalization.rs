mod add_terms;
mod binomial;
mod cube_denominator;
mod exact_quotient;
mod grouping;
mod latex;
mod product;

use crate::runtime::Step;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

use super::SubStep;

/// Generate sub-steps explaining rationalization process.
/// Uses LaTeXExprWithHints for proper sqrt notation rendering.
pub(crate) fn generate_rationalization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    // Some engine rules reuse a rationalization matcher internally but surface the step
    // as an opaque notable-quotient recognition. In those cases, conjugate-narrative
    // substeps are misleading and should stay hidden.
    if step
        .description
        .to_ascii_lowercase()
        .contains("opaque substitution")
    {
        return Vec::new();
    }

    let hints = cas_formatter::DisplayContext::with_root_index(2);

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    let exact_quotient =
        exact_quotient::generate_exact_cube_quotient_substeps(ctx, before, after, &hints);
    if !exact_quotient.is_empty() {
        return exact_quotient;
    }

    let cube_denominator =
        cube_denominator::generate_cube_root_denominator_substeps(ctx, before, after, &hints);
    if !cube_denominator.is_empty() {
        return cube_denominator;
    }

    if step.description.contains("group") {
        return grouping::generate_grouped_rationalization_substeps(ctx, before, after, &hints);
    }

    if step.description.contains("product") {
        return product::generate_product_rationalization_substeps(ctx, before, after, &hints);
    }

    if let Some((focused_before, focused_after)) =
        contextual_binomial_rationalization_focus(ctx, before, after)
    {
        let sub_steps = binomial::generate_binomial_rationalization_substeps(
            ctx,
            focused_before,
            focused_after,
            &hints,
        );
        if !sub_steps.is_empty() {
            return sub_steps;
        }
    }

    binomial::generate_binomial_rationalization_substeps(ctx, before, after, &hints)
}

fn contextual_binomial_rationalization_focus(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId)> {
    let same_shape_terms = match (ctx.get(before), ctx.get(after)) {
        (Expr::Add(before_left, before_right), Expr::Add(after_left, after_right))
        | (Expr::Sub(before_left, before_right), Expr::Sub(after_left, after_right)) => {
            Some((*before_left, *before_right, *after_left, *after_right))
        }
        _ => None,
    }?;

    let (before_left, before_right, after_left, after_right) = same_shape_terms;
    if same_expr(ctx, before_right, after_right)
        && !same_expr(ctx, before_left, after_left)
        && is_fraction_pair(ctx, before_left, after_left)
    {
        return Some((before_left, after_left));
    }

    if same_expr(ctx, before_left, after_left)
        && !same_expr(ctx, before_right, after_right)
        && is_fraction_pair(ctx, before_right, after_right)
    {
        return Some((before_right, after_right));
    }

    None
}

fn is_fraction_pair(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    matches!(ctx.get(before), Expr::Div(_, _)) && matches!(ctx.get(after), Expr::Div(_, _))
}

fn same_expr(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    left == right || compare_expr(ctx, left, right) == Ordering::Equal
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
