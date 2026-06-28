mod dispatch;
mod general;
mod latex;
mod structured;

use crate::runtime::Step;
use cas_ast::{Context, Expr, ExprId};

use super::nested_fraction_analysis::{classify_nested_fraction, find_div_in_expr};
use super::SubStep;

/// Generate sub-steps explaining nested fraction simplification
/// For example: 1/(1 + 1/x) shows:
///   1. Combine terms in denominator: 1 + 1/x → (x+1)/x
///   2. Invert the fraction: 1/((x+1)/x) → x/(x+1)
pub(crate) fn generate_nested_fraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before_expr = step.before;
    let after_expr = step.after;
    let global_before = step.global_before.unwrap_or(step.before);
    let global_after = step.global_after.unwrap_or(step.after);
    let hints = cas_formatter::DisplayContext::default();

    if let Some(pattern) = classify_nested_fraction(ctx, before_expr) {
        return dispatch::generate_nested_fraction_substeps_for_pattern(
            ctx,
            before_expr,
            after_expr,
            pattern,
            &hints,
            general::generate_general_nested_fraction_substeps,
            structured::generate_structured_nested_fraction_substeps,
        );
    }

    if let Some((before_focus, after_focus, pattern)) =
        additive_nested_fraction_focus_pair(ctx, before_expr, after_expr)
    {
        return dispatch::generate_nested_fraction_substeps_for_pattern(
            ctx,
            before_focus,
            after_focus,
            pattern,
            &hints,
            general::generate_general_nested_fraction_substeps,
            structured::generate_structured_nested_fraction_substeps,
        );
    }

    if let Some(pattern) = classify_nested_fraction(ctx, global_after) {
        return generate_reverse_nested_fraction_substeps(
            ctx,
            global_before,
            global_after,
            pattern,
        );
    }

    Vec::new()
}

fn additive_nested_fraction_focus_pair(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
) -> Option<(
    ExprId,
    ExprId,
    super::nested_fraction_analysis::NestedFractionPattern,
)> {
    let before_focus = find_div_in_expr(ctx, before_expr)?;
    if before_focus == before_expr {
        return None;
    }
    let pattern = classify_nested_fraction(ctx, before_focus)?;
    let after_focus = find_div_in_expr(ctx, after_expr)?;
    Some((before_focus, after_focus, pattern))
}

fn nested_fraction_latex(
    ctx: &Context,
    hints: &cas_formatter::DisplayContext,
    id: ExprId,
) -> String {
    latex::nested_fraction_latex(ctx, hints, id)
}

fn generate_reverse_nested_fraction_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    pattern: super::nested_fraction_analysis::NestedFractionPattern,
) -> Vec<SubStep> {
    match pattern {
        super::nested_fraction_analysis::NestedFractionPattern::OneOverSumWithFraction
        | super::nested_fraction_analysis::NestedFractionPattern::FractionOverSumWithFraction => {
            let Expr::Div(_, before_den) = ctx.get(before_expr) else {
                return Vec::new();
            };
            let Expr::Div(_, after_den) = ctx.get(after_expr) else {
                return Vec::new();
            };
            let Some(common_den) = extract_single_fraction_denominator_from_add(ctx, *after_den)
            else {
                return Vec::new();
            };
            let common_den_display = display_expr(ctx, common_den);
            let common_den_grouped_display = grouped_display_expr(ctx, common_den);
            let common_den_grouped_latex = grouped_latex_expr(ctx, common_den);
            let after_den_display = display_expr(ctx, *after_den);
            let after_den_latex = latex_expr(ctx, *after_den);

            vec![SubStep::keyed(
                "nested.rewrite_denominator_common_factor",
                vec![format!("{common_den_display}")],
                display_expr(ctx, *before_den),
                format!("{common_den_grouped_display} · ({after_den_display})"),
            )
            .with_before_latex(latex_expr(ctx, *before_den))
            .with_after_latex(format!(
                "{common_den_grouped_latex}\\cdot \\left({after_den_latex}\\right)"
            ))]
        }
        super::nested_fraction_analysis::NestedFractionPattern::SumWithFractionOverScalar => {
            let Expr::Div(before_num, _) = ctx.get(before_expr) else {
                return Vec::new();
            };
            let Expr::Div(after_num, _) = ctx.get(after_expr) else {
                return Vec::new();
            };
            let Some(common_den) = extract_single_fraction_denominator_from_add(ctx, *after_num)
            else {
                return Vec::new();
            };
            let common_den_display = display_expr(ctx, common_den);
            let common_den_grouped_display = grouped_display_expr(ctx, common_den);
            let common_den_grouped_latex = grouped_latex_expr(ctx, common_den);
            let after_num_display = display_expr(ctx, *after_num);
            let after_num_latex = latex_expr(ctx, *after_num);

            vec![SubStep::keyed(
                "nested.rewrite_numerator_common_factor",
                vec![format!("{common_den_display}")],
                display_expr(ctx, *before_num),
                format!("{common_den_grouped_display} · ({after_num_display})"),
            )
            .with_before_latex(latex_expr(ctx, *before_num))
            .with_after_latex(format!(
                "{common_den_grouped_latex}\\cdot \\left({after_num_latex}\\right)"
            ))]
        }
        super::nested_fraction_analysis::NestedFractionPattern::OneOverSumWithUnitFraction
        | super::nested_fraction_analysis::NestedFractionPattern::General => Vec::new(),
    }
}

fn extract_single_fraction_denominator_from_add(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    match (ctx.get(*left), ctx.get(*right)) {
        (Expr::Div(_, left_den), _) if !matches!(ctx.get(*right), Expr::Div(_, _)) => {
            Some(*left_den)
        }
        (_, Expr::Div(_, right_den)) if !matches!(ctx.get(*left), Expr::Div(_, _)) => {
            Some(*right_den)
        }
        _ => None,
    }
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
