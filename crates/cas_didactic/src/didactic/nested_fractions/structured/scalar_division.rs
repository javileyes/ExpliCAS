use super::super::{nested_fraction_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_sum_over_scalar_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = nested_fraction_latex(ctx, hints, *num);
        let den_str = nested_fraction_latex(ctx, hints, *den);

        sub_steps.push(SubStep {
            description: "Combinar términos del numerador (denominador común)".to_string(),
            before_expr: num_str,
            after_expr: "(numerador combinado) / B".to_string(),
            before_latex: None,
            after_latex: None,
        });
        sub_steps.push(SubStep {
            description: format!("Dividir por {}: multiplicar denominadores", den_str),
            before_expr: nested_fraction_latex(ctx, hints, before_expr),
            after_expr: nested_fraction_latex(ctx, hints, after_expr),
            before_latex: None,
            after_latex: None,
        });
    }

    sub_steps
}
