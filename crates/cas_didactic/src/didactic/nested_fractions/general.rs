mod inner_fraction;

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
            return inner_fraction::generate_inner_fraction_substeps(
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
