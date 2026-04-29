use super::{rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

struct CubeRootDenominatorPlan {
    denominator: ExprId,
    multiplier: ExprId,
}

pub(super) fn generate_cube_root_denominator_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let Some(plan) = cube_root_denominator_plan(ctx, before, after) else {
        return Vec::new();
    };

    let before_latex = rationalization_latex(ctx, hints, before);
    let after_latex = rationalization_latex(ctx, hints, after);
    let denominator_latex = rationalization_latex(ctx, hints, plan.denominator);
    let multiplier_latex = rationalization_latex(ctx, hints, plan.multiplier);
    let multiplied_latex = format!(
        "\\frac{{{multiplier_latex}}}{{\\left({denominator_latex}\\right)\\left({multiplier_latex}\\right)}}"
    );

    vec![
        SubStep::new(
            "Multiplicar por el conjugado cúbico",
            human_expr_from_latex(&before_latex),
            human_expr_from_latex(&multiplied_latex),
        )
        .with_before_latex(before_latex)
        .with_after_latex(multiplied_latex.clone()),
        SubStep::new(
            "Aplicar suma de cubos en el denominador",
            human_expr_from_latex(&multiplied_latex),
            human_expr_from_latex(&after_latex),
        )
        .with_before_latex(multiplied_latex)
        .with_after_latex(after_latex),
    ]
}

fn cube_root_denominator_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<CubeRootDenominatorPlan> {
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return None;
    };
    if !is_one(ctx, *numerator) {
        return None;
    }

    cube_root_plus_one_denominator_root(ctx, *denominator)?;

    let Expr::Div(after_numerator, _) = ctx.get(after) else {
        return None;
    };

    Some(CubeRootDenominatorPlan {
        denominator: *denominator,
        multiplier: *after_numerator,
    })
}

fn cube_root_plus_one_denominator_root(ctx: &Context, denominator: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(denominator) else {
        return None;
    };

    if is_one(ctx, *left) && cube_root_base(ctx, *right).is_some() {
        Some(*right)
    } else if is_one(ctx, *right) && cube_root_base(ctx, *left).is_some() {
        Some(*left)
    } else {
        None
    }
}

fn cube_root_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_rational_literal(ctx, *exponent, 1, 3) => Some(*base),
        _ => None,
    }
}

fn human_expr_from_latex(latex: &str) -> String {
    crate::didactic::latex_to_plain_text(latex)
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &1.into())
}

fn is_rational_literal(ctx: &Context, expr: ExprId, numerator: i64, denominator: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value)
            if value.numer() == &numerator.into() && value.denom() == &denominator.into()
    )
}
