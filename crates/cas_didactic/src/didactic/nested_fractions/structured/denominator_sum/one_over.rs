use super::super::super::SubStep;
use crate::didactic::nested_fraction_analysis::extract_combined_fraction_str;
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_one_over_sum_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
    nested_fraction_latex: fn(&Context, &cas_formatter::DisplayContext, ExprId) -> String,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(_, den) = ctx.get(before_expr) {
        let den_str = nested_fraction_latex(ctx, hints, *den);
        let intermediate_str = extract_combined_fraction_str(ctx, *den);

        sub_steps.push(SubStep {
            description: "Combinar términos del denominador (denominador común)".to_string(),
            before_expr: den_str.clone(),
            after_expr: intermediate_str.clone(),
            before_latex: None,
            after_latex: None,
        });
        sub_steps.push(SubStep {
            description: "Invertir la fracción: 1/(a/b) = b/a".to_string(),
            before_expr: format!("\\frac{{1}}{{{}}}", intermediate_str),
            after_expr: nested_fraction_latex(ctx, hints, after_expr),
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}
