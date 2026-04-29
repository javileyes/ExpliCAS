use super::super::SubStep;
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_inner_fraction_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    inner_frac: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let _ = hints;
    let Expr::Div(outer_num, outer_den) = ctx.get(before_expr) else {
        return Vec::new();
    };
    if *outer_den != inner_frac {
        return Vec::new();
    }
    let Expr::Div(inner_num, inner_den) = ctx.get(inner_frac) else {
        return Vec::new();
    };

    let intermediate_display = format!(
        "{} · {}/{}",
        grouped_display_expr(ctx, *outer_num),
        grouped_display_expr(ctx, *inner_den),
        grouped_display_expr(ctx, *inner_num)
    );
    let intermediate_latex = format!(
        "{}\\cdot \\frac{{{}}}{{{}}}",
        grouped_latex_expr(ctx, *outer_num),
        latex_expr(ctx, *inner_den),
        latex_expr(ctx, *inner_num)
    );

    vec![
        SubStep::new(
            "Invertir la fracción del denominador",
            display_expr(ctx, before_expr),
            intermediate_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, before_expr))
        .with_after_latex(intermediate_latex.clone()),
        SubStep::new(
            "Simplificar el producto resultante",
            intermediate_display,
            display_expr(ctx, after_expr),
        )
        .with_before_latex(intermediate_latex)
        .with_after_latex(latex_expr(ctx, after_expr)),
    ]
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

fn latex_expr(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::LaTeXExpr {
        context: ctx,
        id: expr,
    }
    .to_latex()
}

fn grouped_display_expr(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if needs_grouping(ctx, expr) {
        format!("({display})")
    } else {
        display
    }
}

fn grouped_latex_expr(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    if needs_grouping(ctx, expr) {
        format!("\\left({latex}\\right)")
    } else {
        latex
    }
}

fn needs_grouping(ctx: &Context, expr: ExprId) -> bool {
    !matches!(
        ctx.get(expr),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::Function(_, _)
    )
}
