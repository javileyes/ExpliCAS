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
        let _inner_num_str = nested_fraction_latex(ctx, hints, *inner_num);
        let _inner_den_str = nested_fraction_latex(ctx, hints, *inner_den);

        let _ = num_str;
        return vec![SubStep {
            description: "Dividir entre una fracción equivale a invertirla".to_string(),
            before_expr: before_str.to_string(),
            after_expr: after_str.to_string(),
            before_latex: None,
            after_latex: None,
        }];
    }

    vec![SubStep {
        description: "Dividir entre una fracción equivale a invertirla".to_string(),
        before_expr: before_str.to_string(),
        after_expr: after_str.to_string(),
        before_latex: None,
        after_latex: None,
    }]
}
