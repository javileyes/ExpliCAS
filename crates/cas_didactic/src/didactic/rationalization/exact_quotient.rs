use super::{rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_exact_cube_quotient_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let Some(_base) = extract_exact_quotient_base(ctx, before, after) else {
        return Vec::new();
    };

    vec![SubStep::new(
        "Usar (u^3 - 1) / (u - 1) = u^2 + u + 1",
        human_expr_from_latex(&rationalization_latex(ctx, hints, before)),
        human_expr_from_latex(&rationalization_latex(ctx, hints, after)),
    )
    .with_before_latex(rationalization_latex(ctx, hints, before))
    .with_after_latex(rationalization_latex(ctx, hints, after))]
}

fn extract_exact_quotient_base(ctx: &Context, before: ExprId, after: ExprId) -> Option<ExprId> {
    let Expr::Div(_, denominator) = ctx.get(before) else {
        return None;
    };
    let base = denominator_base_minus_one(ctx, *denominator)?;
    if !after_matches_cube_quotient(ctx, after, base) {
        return None;
    }
    Some(base)
}

fn after_matches_cube_quotient(ctx: &Context, after: ExprId, base: ExprId) -> bool {
    let terms = add_terms(ctx, after);
    let mut has_one = false;
    let mut has_base = false;
    let mut has_square = false;

    for term in terms {
        if is_one(ctx, term) {
            has_one = true;
        } else if term == base {
            has_base = true;
        } else if matches!(ctx.get(term), Expr::Pow(pow_base, exponent) if *pow_base == base && is_integer_literal(ctx, *exponent, 2))
        {
            has_square = true;
        }
    }

    has_one && has_base && has_square
}

fn add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_add_terms(ctx, expr, &mut out);
    out
}

fn collect_add_terms(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_add_terms(ctx, *left, out);
            collect_add_terms(ctx, *right, out);
        }
        _ => out.push(expr),
    }
}

fn denominator_base_minus_one(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(base, rhs) if is_one(ctx, *rhs) => Some(*base),
        Expr::Add(left, right) => {
            if is_negative_one(ctx, *left) {
                Some(*right)
            } else if is_negative_one(ctx, *right) {
                Some(*left)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn human_expr_from_latex(latex: &str) -> String {
    crate::didactic::latex_to_plain_text(latex)
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &1.into())
}

fn is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &(-1).into() && value.denom() == &1.into())
}

fn is_integer_literal(ctx: &Context, expr: ExprId, expected: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if value.numer() == &expected.into() && value.denom() == &1.into()
    )
}
