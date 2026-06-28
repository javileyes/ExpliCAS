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
        let den_str = nested_fraction_latex(ctx, hints, *den);
        let intermediate_str = extract_combined_fraction_str(ctx, *den);

        sub_steps.push(
            SubStep::keyed(
                "polynomial.common_denominator_within_denominator",
                vec![],
                display_expr(ctx, *den),
                intermediate_str.clone(),
            )
            .with_before_latex(den_str)
            .with_after_latex(intermediate_str.clone()),
        );

        sub_steps.push(
            SubStep::keyed(
                "polynomial.divide_by_fraction_is_multiply_inverse",
                vec![],
                format!(
                    "\\frac{{{}}}{{{}}}",
                    nested_fraction_latex(ctx, hints, *num),
                    intermediate_str.clone()
                ),
                display_expr(ctx, after_expr),
            )
            .with_before_latex(format!(
                "\\frac{{{}}}{{{}}}",
                nested_fraction_latex(ctx, hints, *num),
                intermediate_str
            ))
            .with_after_latex(nested_fraction_latex(ctx, hints, after_expr)),
        );
    }

    sub_steps
}

fn display_expr(ctx: &Context, expr: ExprId) -> String {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr,
        }
    )
}
