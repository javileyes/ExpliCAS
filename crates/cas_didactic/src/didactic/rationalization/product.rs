use super::{rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_product_rationalization_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(num, den) = ctx.get(before) {
        let num_latex = rationalization_latex(ctx, hints, *num);
        let den_latex = rationalization_latex(ctx, hints, *den);

        sub_steps.push(SubStep {
            description: "Denominador con producto de radical".to_string(),
            before_expr: format!("\\frac{{{}}}{{{}}}", num_latex, den_latex),
            after_expr: "\\frac{a}{k \\cdot \\sqrt{n}}".to_string(),
            before_latex: None,
            after_latex: None,
        });

        if let Expr::Div(new_num, new_den) = ctx.get(after) {
            let after_num_latex = rationalization_latex(ctx, hints, *new_num);
            let after_den_latex = rationalization_latex(ctx, hints, *new_den);
            sub_steps.push(SubStep {
                description: "Multiplicar por \\sqrt{n}/\\sqrt{n}".to_string(),
                before_expr: format!(
                    "\\frac{{{} \\cdot \\sqrt{{n}}}}{{{} \\cdot \\sqrt{{n}}}}",
                    num_latex, den_latex
                ),
                after_expr: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
                before_latex: None,
                after_latex: None,
            });
        }
    }

    sub_steps
}
