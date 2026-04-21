use super::{rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_nary::{AddView, Sign};

pub(super) fn generate_exact_cube_quotient_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    if let Some(substeps) =
        generate_sum_difference_cubes_self_cancel_substeps(ctx, before, after, hints)
    {
        return substeps;
    }

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

#[derive(Clone, Copy)]
enum CubeIdentityKind {
    Sum,
    Difference,
}

struct CubeSelfCancelPlan {
    numerator: ExprId,
    denominator: ExprId,
    kind: CubeIdentityKind,
}

fn generate_sum_difference_cubes_self_cancel_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Option<Vec<SubStep>> {
    let plan = sum_difference_cubes_self_cancel_plan(ctx, before, after)?;

    let numerator_latex = rationalization_latex(ctx, hints, plan.numerator);
    let denominator_latex = rationalization_latex(ctx, hints, plan.denominator);
    let quotient_latex = format!("\\frac{{{}}}{{{}}}", denominator_latex, denominator_latex);

    let factor_title = match plan.kind {
        CubeIdentityKind::Sum => "Factorizar el numerador como suma de cubos",
        CubeIdentityKind::Difference => "Factorizar el numerador como diferencia de cubos",
    };

    Some(vec![
        SubStep::new(
            factor_title,
            human_expr_from_latex(&numerator_latex),
            human_expr_from_latex(&denominator_latex),
        )
        .with_before_latex(numerator_latex)
        .with_after_latex(denominator_latex.clone()),
        SubStep::new(
            "Numerador y denominador quedan iguales, así que el cociente vale 1",
            human_expr_from_latex(&quotient_latex),
            human_expr_from_latex("1"),
        )
        .with_before_latex(quotient_latex)
        .with_after_latex("1"),
    ])
}

fn sum_difference_cubes_self_cancel_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<CubeSelfCancelPlan> {
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return None;
    };
    if !is_one(ctx, after) {
        return None;
    }

    let (left_term, right_term, kind) = cube_identity_terms(ctx, *numerator)?;
    let left_base = cube_base_from_term(ctx, left_term)?;
    let right_base = cube_base_from_term(ctx, right_term)?;

    let Expr::Mul(first_factor, second_factor) = ctx.get(*denominator) else {
        return None;
    };
    let matches = [
        (*first_factor, *second_factor),
        (*second_factor, *first_factor),
    ]
    .into_iter()
    .any(|(linear_factor, quadratic_factor)| {
        linear_factor_matches(ctx, linear_factor, left_base, right_base, kind)
            && quadratic_factor_matches(ctx, quadratic_factor, left_base, right_base, kind)
    });
    if !matches {
        return None;
    }

    Some(CubeSelfCancelPlan {
        numerator: *numerator,
        denominator: *denominator,
        kind,
    })
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

fn cube_identity_terms(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId, CubeIdentityKind)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right, CubeIdentityKind::Difference)),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner, CubeIdentityKind::Difference)),
            _ => Some((*left, *right, CubeIdentityKind::Sum)),
        },
        _ => None,
    }
}

fn cube_base_from_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, 3) => Some(*base),
        _ if is_one(ctx, expr) => Some(expr),
        _ => None,
    }
}

fn linear_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> bool {
    match kind {
        CubeIdentityKind::Sum => match ctx.get(expr) {
            Expr::Add(left, right) => {
                (*left == left_base && *right == right_base)
                    || (*left == right_base && *right == left_base)
            }
            _ => false,
        },
        CubeIdentityKind::Difference => match ctx.get(expr) {
            Expr::Sub(left, right) => *left == left_base && *right == right_base,
            Expr::Add(left, right) => {
                *left == left_base
                    && matches!(ctx.get(*right), Expr::Neg(inner) if *inner == right_base)
            }
            _ => false,
        },
    }
}

fn quadratic_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let has_left_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, left_base));
    let has_right_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, right_base));
    let mixed_sign = match kind {
        CubeIdentityKind::Sum => Sign::Neg,
        CubeIdentityKind::Difference => Sign::Pos,
    };
    let has_mixed = terms.iter().any(|(term, sign)| {
        *sign == mixed_sign && matches_unscaled_product(ctx, *term, left_base, right_base)
    });

    has_left_square && has_right_square && has_mixed
}

fn matches_square_of(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Pow(term_base, exponent)
            if *term_base == base && is_integer_literal(ctx, *exponent, 2)
    )
}

fn matches_unscaled_product(ctx: &Context, expr: ExprId, left: ExprId, right: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Mul(lhs, rhs) if (*lhs == left && *rhs == right) || (*lhs == right && *rhs == left)
    )
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
