use super::super::nested_fraction_analysis::find_div_in_expr;
use super::{nested_fraction_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_general_nested_fraction_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let before_str = nested_fraction_latex(ctx, hints, before_expr);
    let after_str = nested_fraction_latex(ctx, hints, after_expr);

    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = nested_fraction_latex(ctx, hints, *num);

        if let Some(inner_frac) = find_div_in_expr(ctx, *den) {
            return generate_inner_fraction_substeps(
                ctx,
                inner_frac,
                &num_str,
                &before_str,
                &after_str,
                hints,
            );
        }

        return vec![SubStep {
            description: "Simplificar fracción anidada".to_string(),
            before_expr: before_str,
            after_expr: after_str,
            before_latex: None,
            after_latex: None,
        }];
    }

    vec![SubStep {
        description: "Simplificar expresión".to_string(),
        before_expr: before_str,
        after_expr: after_str,
        before_latex: None,
        after_latex: None,
    }]
}

fn generate_inner_fraction_substeps(
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
