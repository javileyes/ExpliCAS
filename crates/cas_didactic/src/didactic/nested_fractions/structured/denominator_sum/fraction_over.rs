use super::super::super::SubStep;
use crate::didactic::nested_fraction_analysis::extract_combined_fraction_str;
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_fraction_over_sum_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
    nested_fraction_latex: fn(&Context, &cas_formatter::DisplayContext, ExprId) -> String,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = nested_fraction_latex(ctx, hints, *num);
        let den_str = nested_fraction_latex(ctx, hints, *den);

        sub_steps.push(SubStep {
            description: "Combinar términos del denominador (denominador común)".to_string(),
            before_expr: den_str,
            after_expr: extract_combined_fraction_str(ctx, *den),
            before_latex: None,
            after_latex: None,
        });
        sub_steps.push(SubStep {
            description: format!("Multiplicar {} por el denominador interno", num_str),
            before_expr: nested_fraction_latex(ctx, hints, before_expr),
            after_expr: nested_fraction_latex(ctx, hints, after_expr),
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}
