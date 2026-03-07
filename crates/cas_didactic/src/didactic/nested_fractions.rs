mod general;
mod structured;

use cas_ast::{Context, ExprId};
use cas_solver::Step;

use super::nested_fraction_analysis::{classify_nested_fraction, NestedFractionPattern};
use super::SubStep;

/// Generate sub-steps explaining nested fraction simplification
/// For example: 1/(1 + 1/x) shows:
///   1. Combine terms in denominator: 1 + 1/x → (x+1)/x
///   2. Invert the fraction: 1/((x+1)/x) → x/(x+1)
pub(crate) fn generate_nested_fraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before_expr = step.before;
    let after_expr = step.after;
    let pattern = match classify_nested_fraction(ctx, before_expr) {
        Some(pattern) => pattern,
        None => return Vec::new(),
    };
    let hints = cas_formatter::DisplayContext::default();

    if matches!(pattern, NestedFractionPattern::General) {
        return general::generate_general_nested_fraction_substeps(
            ctx,
            before_expr,
            after_expr,
            &hints,
        );
    }

    structured::generate_structured_nested_fraction_substeps(
        ctx,
        before_expr,
        after_expr,
        pattern,
        &hints,
    )
}

fn nested_fraction_latex(
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
