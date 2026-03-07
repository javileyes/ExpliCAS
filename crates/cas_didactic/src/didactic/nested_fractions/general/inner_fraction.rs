use super::super::{nested_fraction_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_inner_fraction_substeps(
    ctx: &Context,
    inner_frac: ExprId,
    num_str: &str,
    before_str: &str,
    after_str: &str,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    if let Expr::Div(inner_num, inner_den) = ctx.get(inner_frac) {
        let inner_num_str = nested_fraction_latex(ctx, hints, *inner_num);
        let inner_den_str = nested_fraction_latex(ctx, hints, *inner_den);

        return vec![
            SubStep {
                description: "Identificar la fracción anidada en el denominador".to_string(),
                before_expr: format!(
                    "\\frac{{{}}}{{\\text{{...}} + \\frac{{{}}}{{{}}}}}",
                    num_str, inner_num_str, inner_den_str
                ),
                after_expr: format!("\\text{{Multiplicar por }} {}", inner_den_str),
                before_latex: None,
                after_latex: None,
            },
            SubStep {
                description: "Simplificar: 1/(a/b) = b/a".to_string(),
                before_expr: before_str.to_string(),
                after_expr: after_str.to_string(),
                before_latex: None,
                after_latex: None,
            },
        ];
    }

    vec![SubStep {
        description: "Simplificar fracción compleja (multiplicar por denominador común)"
            .to_string(),
        before_expr: before_str.to_string(),
        after_expr: after_str.to_string(),
        before_latex: None,
        after_latex: None,
    }]
}
