use super::super::super::SubStep;
use cas_ast::{Context, ExprId};

pub(super) fn build_divide_by_scalar_substep(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
    den_str: &str,
    nested_fraction_latex: fn(&Context, &cas_formatter::DisplayContext, ExprId) -> String,
) -> SubStep {
    SubStep {
        description: format!("Dividir por {}: multiplicar denominadores", den_str),
        before_expr: nested_fraction_latex(ctx, hints, before_expr),
        after_expr: nested_fraction_latex(ctx, hints, after_expr),
        before_latex: None,
        after_latex: None,
    }
}
