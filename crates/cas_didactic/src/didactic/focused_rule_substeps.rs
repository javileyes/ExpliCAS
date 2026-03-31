use super::SubStep;
use crate::runtime::Step;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

pub(crate) fn generate_focused_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    match step.rule_name.as_str() {
        "Pre-order Common Factor Cancel" => generate_common_factor_cancel_substeps(ctx, step),
        "Pre-order Difference of Squares Cancel" => {
            generate_difference_of_squares_cancel_substeps(ctx, step)
        }
        "Identity Property of Multiplication" => {
            generate_identity_multiplication_substeps(ctx, step)
        }
        "Evaluate Numeric Power" => generate_evaluate_numeric_power_substeps(ctx, step),
        "Pre-order Sum/Difference of Cubes" => generate_sum_difference_cubes_substeps(ctx, step),
        "Pre-order Sum/Difference of Cubes Cancel" => {
            generate_sum_difference_cubes_cancel_substeps(ctx, step)
        }
        "Inverse Tan Relations" => generate_inverse_tan_relation_substeps(ctx, step),
        "Subtraction Self-Cancel" => generate_subtraction_self_cancel_substeps(ctx, step),
        "Cancel Reciprocal Exponents" => generate_cancel_reciprocal_exponents_substeps(ctx, step),
        "Sqrt Perfect Square" | "Simplify Square Root" => {
            generate_sqrt_perfect_square_substeps(ctx, step)
        }
        _ => Vec::new(),
    }
}

fn generate_common_factor_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    let Some(common_factor) = first_common_factor(ctx, *numerator, *denominator) else {
        return Vec::new();
    };

    let factor_display = display_expr(ctx, common_factor);
    let before_display = display_expr(ctx, before);
    let before_latex = latex_expr(ctx, before);

    vec![SubStep::new(
        format!("Como {} aparece arriba y abajo, se cancela", factor_display),
        before_display,
        display_expr(ctx, after),
    )
    .with_before_latex(before_latex)
    .with_after_latex(latex_expr(ctx, after))]
}

fn generate_difference_of_squares_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    let Some((other_factor, canceled_factor)) =
        split_product_for_cancellation(ctx, *numerator, *denominator)
    else {
        return Vec::new();
    };
    let Some((left_term, right_term)) = difference_square_terms(ctx, other_factor, canceled_factor)
    else {
        return Vec::new();
    };

    vec![
        SubStep::new(
            "Reescribir el numerador como diferencia de cuadrados",
            difference_of_squares_display(ctx, left_term, right_term),
            display_expr(ctx, *numerator),
        )
        .with_before_latex(difference_of_squares_latex(ctx, left_term, right_term))
        .with_after_latex(latex_expr(ctx, *numerator)),
        SubStep::new(
            format!(
                "Ahora se cancela el factor {}",
                display_expr(ctx, canceled_factor)
            ),
            display_expr(ctx, before),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_inverse_tan_relation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let Some(pair_before) = step.before_local() else {
        return Vec::new();
    };
    let Some(pair_after) = step.after_local() else {
        return Vec::new();
    };

    let mut out = Vec::new();

    if step.before != pair_before {
        out.push(
            SubStep::new(
                "Juntar la pareja que encaja con la identidad",
                display_expr(ctx, step.before),
                display_expr(ctx, pair_before),
            )
            .with_before_latex(latex_expr(ctx, step.before))
            .with_after_latex(latex_expr(ctx, pair_before)),
        );
    }

    out.push(
        SubStep::new(
            "Esa pareja vale pi/2",
            display_expr(ctx, pair_before),
            display_expr(ctx, pair_after),
        )
        .with_before_latex(latex_expr(ctx, pair_before))
        .with_after_latex(latex_expr(ctx, pair_after)),
    );

    out
}

fn generate_sqrt_perfect_square_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(radicand) = sqrt_radicand(ctx, before) else {
        return Vec::new();
    };
    let Some(abs_arg) = abs_argument(ctx, after) else {
        return Vec::new();
    };

    let square_display = squared_display(ctx, abs_arg);
    let square_latex = squared_latex(ctx, abs_arg);

    vec![
        SubStep::new(
            "Reescribir el radicando como un cuadrado perfecto",
            display_expr(ctx, radicand),
            square_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, radicand))
        .with_after_latex(square_latex.clone()),
        SubStep::new(
            "La raíz de un cuadrado da un valor absoluto",
            format!("sqrt({})", square_display),
            display_expr(ctx, after),
        )
        .with_before_latex(format!("\\sqrt{{{}}}", square_latex))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_subtraction_self_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Sub(left, right) = ctx.get(before) else {
        return Vec::new();
    };
    if left != right || after == before {
        return Vec::new();
    }
    let _ = (left, right);
    // The human-visible step title plus the direct local change already explain this move.
    // Adding micro-substeps like "the two terms are the same" only creates didactic noise.
    Vec::new()
}

fn generate_cancel_reciprocal_exponents_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    let Some(plan) = reciprocal_exponent_plan(ctx, local_before) else {
        return Vec::new();
    };

    let radicand_display = display_expr(ctx, plan.radicand);
    let radicand_latex = latex_expr(ctx, plan.radicand);
    let sqrt_display = format!("sqrt({})", radicand_display);
    let sqrt_latex = format!("\\sqrt{{{}}}", radicand_latex);

    let mut out = vec![SubStep::new(
        "El cuadrado deshace la raíz",
        format!("{}^2", sqrt_display),
        radicand_display,
    )
    .with_before_latex(format!("{{{}}}^{{2}}", sqrt_latex))
    .with_after_latex(radicand_latex)];

    if let (Some(global_before), Some(global_after)) = (step.global_before, step.global_after) {
        if global_before != local_before || global_after != local_after {
            out.push(
                SubStep::new(
                    "Reemplazar ese bloque en la expresión",
                    display_expr(ctx, global_before),
                    display_expr(ctx, global_after),
                )
                .with_before_latex(latex_expr(ctx, global_before))
                .with_after_latex(latex_expr(ctx, global_after)),
            );
            return out;
        }
    }

    if step.before != local_before || step.after != local_after {
        out.push(
            SubStep::new(
                "Reemplazar ese bloque en la expresión",
                display_expr(ctx, step.before),
                display_expr(ctx, step.after),
            )
            .with_before_latex(latex_expr(ctx, step.before))
            .with_after_latex(latex_expr(ctx, step.after)),
        );
    } else if local_before != local_after {
        out.push(
            SubStep::new(
                "Reemplazar la raíz al cuadrado por el radicando",
                display_expr(ctx, local_before),
                display_expr(ctx, local_after),
            )
            .with_before_latex(latex_expr(ctx, local_before))
            .with_after_latex(latex_expr(ctx, local_after)),
        );
    }

    out
}

fn generate_identity_multiplication_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Mul(left, right) = ctx.get(before) else {
        return Vec::new();
    };
    if !is_one(ctx, *left) && !is_one(ctx, *right) {
        return Vec::new();
    }

    vec![SubStep::new(
        "Quitar el factor 1 que no cambia el valor",
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

fn generate_evaluate_numeric_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Pow(_, _) = ctx.get(before) else {
        return Vec::new();
    };

    let before_human = normalize_human_power_expr(&human_expr(ctx, before));
    let after_human = human_expr(ctx, after);

    vec![SubStep::new(
        format!("Calcular {} = {}", before_human, after_human),
        before_human,
        after_human,
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

fn generate_sum_difference_cubes_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = cube_identity_plan(ctx, before, after) else {
        return Vec::new();
    };

    let identity_description = match plan.kind {
        CubeIdentityKind::Sum => "Reconocer la forma a^3 + b^3",
        CubeIdentityKind::Difference => "Reconocer la forma a^3 - b^3",
    };
    let factor_description = match plan.kind {
        CubeIdentityKind::Sum => "Aplicar a^3 + b^3 = (a + b)(a^2 - ab + b^2)",
        CubeIdentityKind::Difference => "Aplicar a^3 - b^3 = (a - b)(a^2 + ab + b^2)",
    };

    vec![
        SubStep::new(
            identity_description,
            display_expr(ctx, before),
            cube_identity_display(ctx, plan.left_base, plan.right_base, plan.kind),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(cube_identity_latex(
            ctx,
            plan.left_base,
            plan.right_base,
            plan.kind,
        )),
        SubStep::new(
            factor_description,
            cube_identity_display(ctx, plan.left_base, plan.right_base, plan.kind),
            display_expr(ctx, after),
        )
        .with_before_latex(cube_identity_latex(
            ctx,
            plan.left_base,
            plan.right_base,
            plan.kind,
        ))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_sum_difference_cubes_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    let Some((remaining_factor, matching_factor)) =
        split_product_for_cancellation(ctx, *numerator, *denominator)
    else {
        return Vec::new();
    };

    vec![
        SubStep::new(
            "Ver que el denominador coincide con un factor del numerador",
            display_expr(ctx, before),
            display_expr(ctx, matching_factor),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, matching_factor)),
        SubStep::new(
            format!(
                "Cancelar el factor comun {}",
                display_expr(ctx, matching_factor)
            ),
            display_expr(ctx, before),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after)),
        SubStep::new(
            "Queda el cociente exacto del cubo",
            display_expr(ctx, matching_factor),
            display_expr(ctx, remaining_factor),
        )
        .with_before_latex(latex_expr(ctx, matching_factor))
        .with_after_latex(latex_expr(ctx, remaining_factor)),
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

fn human_expr(ctx: &Context, expr: ExprId) -> String {
    crate::didactic::latex_to_plain_text(&latex_expr(ctx, expr))
}

fn normalize_human_power_expr(value: &str) -> String {
    value.replace("((-1))", "(-1)")
}

fn first_common_factor(ctx: &Context, numerator: ExprId, denominator: ExprId) -> Option<ExprId> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let mut denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator).to_vec();

    for numerator_factor in numerator_factors {
        if let Some(index) = denominator_factors
            .iter()
            .position(|denominator_factor| *denominator_factor == numerator_factor)
        {
            denominator_factors.remove(index);
            return Some(numerator_factor);
        }
    }

    None
}

fn split_product_for_cancellation(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Mul(left, right) = ctx.get(numerator) else {
        return None;
    };
    if *left == denominator {
        return Some((*right, *left));
    }
    if *right == denominator {
        return Some((*left, *right));
    }
    None
}

#[derive(Clone, Copy)]
enum CubeIdentityKind {
    Sum,
    Difference,
}

struct CubeIdentityPlan {
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
}

struct ReciprocalExponentPlan {
    radicand: ExprId,
}

fn cube_identity_plan(ctx: &Context, before: ExprId, after: ExprId) -> Option<CubeIdentityPlan> {
    let (left_term, right_term, kind) = match ctx.get(before) {
        Expr::Add(left, right) => (*left, *right, CubeIdentityKind::Sum),
        Expr::Sub(left, right) => (*left, *right, CubeIdentityKind::Difference),
        _ => return None,
    };

    let left_base = cube_base_from_term(ctx, left_term)?;
    let right_base = cube_base_from_term(ctx, right_term)?;

    let Expr::Mul(first_factor, second_factor) = ctx.get(after) else {
        return None;
    };
    if !linear_factor_matches(ctx, *first_factor, left_base, right_base, kind)
        && !linear_factor_matches(ctx, *second_factor, left_base, right_base, kind)
    {
        return None;
    }

    Some(CubeIdentityPlan {
        left_base,
        right_base,
        kind,
    })
}

fn reciprocal_exponent_plan(ctx: &Context, before: ExprId) -> Option<ReciprocalExponentPlan> {
    let Expr::Pow(base, exponent) = ctx.get(before) else {
        return None;
    };
    if !is_integer_literal(ctx, *exponent, 2) {
        return None;
    }

    match ctx.get(*base) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(ReciprocalExponentPlan { radicand: args[0] })
        }
        Expr::Pow(radicand, inner_exponent) if is_one_half(ctx, *inner_exponent) => {
            Some(ReciprocalExponentPlan {
                radicand: *radicand,
            })
        }
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
            _ => false,
        },
    }
}

fn cube_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let op = match kind {
        CubeIdentityKind::Sum => " + ",
        CubeIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        cubed_display(ctx, left_base),
        op,
        cubed_display(ctx, right_base)
    )
}

fn cube_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let op = match kind {
        CubeIdentityKind::Sum => " + ",
        CubeIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        cubed_latex(ctx, left_base),
        op,
        cubed_latex(ctx, right_base)
    )
}

fn cubed_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^3")
    } else {
        format!("({display})^3")
    }
}

fn cubed_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{3}}")
}

fn difference_square_terms(
    ctx: &Context,
    sum_factor: ExprId,
    diff_factor: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Add(sum_left, sum_right) = ctx.get(sum_factor) else {
        return None;
    };
    let Expr::Sub(diff_left, diff_right) = ctx.get(diff_factor) else {
        return None;
    };

    let sum_matches_direct = (*sum_left == *diff_left && *sum_right == *diff_right)
        || (*sum_left == *diff_right && *sum_right == *diff_left);
    if !sum_matches_direct {
        return None;
    }

    Some((*diff_left, *diff_right))
}

fn difference_of_squares_display(ctx: &Context, left: ExprId, right: ExprId) -> String {
    format!(
        "{} - {}",
        squared_display(ctx, left),
        squared_term_display(ctx, right)
    )
}

fn difference_of_squares_latex(ctx: &Context, left: ExprId, right: ExprId) -> String {
    format!(
        "{} - {}",
        squared_latex(ctx, left),
        squared_term_latex(ctx, right)
    )
}

fn squared_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^2")
    } else {
        format!("({display})^2")
    }
}

fn squared_term_display(ctx: &Context, expr: ExprId) -> String {
    if is_one(ctx, expr) {
        "1".to_string()
    } else {
        squared_display(ctx, expr)
    }
}

fn squared_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{2}}")
}

fn squared_term_latex(ctx: &Context, expr: ExprId) -> String {
    if is_one(ctx, expr) {
        "1".to_string()
    } else {
        squared_latex(ctx, expr)
    }
}

fn is_simple_power_base(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::Function(_, _)
    )
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &1.into())
}

fn is_integer_literal(ctx: &Context, expr: ExprId, expected: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if value.numer() == &expected.into() && value.denom() == &1.into()
    )
}

fn is_one_half(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &2.into())
}

fn sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exponent) if is_one_half(ctx, *exponent) => Some(*base),
        _ => None,
    }
}

fn abs_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Abs) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}
