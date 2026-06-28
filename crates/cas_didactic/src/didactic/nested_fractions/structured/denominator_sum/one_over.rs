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

        sub_steps.push(
            SubStep::keyed(
                "polynomial.common_denominator_within_denominator",
                vec![],
                display_expr(ctx, *den),
                intermediate_str.clone(),
            )
            .with_before_latex(den_str.clone())
            .with_after_latex(intermediate_str.clone()),
        );

        if let Some(full_intermediate) = build_one_over_intermediate_expr(ctx, *den) {
            sub_steps.push(
                SubStep::keyed(
                    "polynomial.invert_denominator_fraction",
                    vec![],
                    full_intermediate.clone(),
                    display_expr(ctx, after_expr),
                )
                .with_before_latex(full_intermediate)
                .with_after_latex(nested_fraction_latex(ctx, hints, after_expr)),
            );
        }
    }

    sub_steps
}

fn build_one_over_intermediate_expr(ctx: &Context, denominator: ExprId) -> Option<String> {
    let intermediate_den = extract_combined_fraction_str(ctx, denominator);
    Some(format!("\\frac{{1}}{{{intermediate_den}}}"))
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
