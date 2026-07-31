//! `focused_rule_substeps`: familia `powers`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn generate_same_base_power_merge_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let quotient_substeps = generate_same_base_power_quotient_substeps(ctx, before, after);
    if !quotient_substeps.is_empty() {
        return quotient_substeps;
    }
    if before == after {
        return Vec::new();
    }

    vec![concrete_expr_substep(
        ctx,
        "Sumar los exponentes de la misma base",
        before,
        after,
    )]
}

pub(super) fn generate_odd_half_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(plan) = odd_half_power_simplify_plan(ctx, before, after) {
        return build_odd_half_power_simplify_substeps(ctx, after, plan, None);
    }

    if let Some(substeps) = generate_odd_half_power_simplify_substeps(ctx, step) {
        return substeps;
    }

    Vec::new()
}

pub(super) fn matches_pow_three_times_difference(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    left_name: &str,
    right_name: &str,
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return false;
    }

    factors
        .iter()
        .any(|factor| matches_var_pow(ctx, *factor, var_name, 3))
        && factors
            .iter()
            .any(|factor| matches_linear_difference(ctx, *factor, left_name, right_name))
}

fn matches_var_pow(ctx: &Context, expr: ExprId, var_name: &str, exponent: i64) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            matches_var_name(ctx, *base, var_name)
                && matches!(
                    ctx.get(*exp),
                    Expr::Number(n) if n.is_integer() && *n.numer() == exponent.into()
                )
        }
        _ => false,
    }
}

pub(super) fn generate_odd_half_power_simplify_substeps(
    ctx: &Context,
    step: &Step,
) -> Option<Vec<SubStep>> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    if let Some(plan) = odd_half_power_simplify_plan(ctx, local_before, local_after) {
        return Some(build_odd_half_power_simplify_substeps(
            ctx,
            local_after,
            plan,
            odd_half_power_replacement_pair(step, local_before, local_after),
        ));
    }

    let (_focus_before, focus_after, plan) =
        find_additive_odd_half_power_simplify_focus(ctx, step.before, step.after)?;
    Some(build_odd_half_power_simplify_substeps(
        ctx,
        focus_after,
        plan,
        Some((step.before, step.after)),
    ))
}

fn build_odd_half_power_simplify_substeps(
    ctx: &Context,
    focus_after: ExprId,
    plan: OddHalfPowerSimplifyPlan,
    replacement_pair: Option<(ExprId, ExprId)>,
) -> Vec<SubStep> {
    let (radicand_display, radicand_latex) =
        power_display_and_latex(ctx, plan.base, plan.radicand_power);
    let (outside_power_display, _outside_power_latex) =
        power_display_and_latex(ctx, plan.base, plan.outside_power);
    let (even_power_display, even_power_latex) =
        power_display_and_latex(ctx, plan.base, 2 * plan.outside_power);
    let base_grouped_display = grouped_substitution_display(ctx, plan.base);
    let base_grouped_latex = grouped_substitution_latex(ctx, plan.base);
    let factorized_radicand_display = format!("{even_power_display} · {base_grouped_display}");
    let factorized_radicand_latex = format!("{even_power_latex}\\cdot {base_grouped_latex}");
    let factorized_root_display = format!("sqrt({factorized_radicand_display})");
    let factorized_root_latex = format!("\\sqrt{{{factorized_radicand_latex}}}");

    let mut out = vec![
        SubStep::new(
            "Separar el radicando en una potencia par y un factor",
            radicand_display,
            factorized_radicand_display.clone(),
        )
        .with_before_latex(radicand_latex)
        .with_after_latex(factorized_radicand_latex.clone()),
        SubStep::new(
            format!(
                "Como {} ≥ 0, sacar {} fuera de la raíz",
                human_expr(ctx, plan.base),
                outside_power_display
            ),
            factorized_root_display,
            human_expr(ctx, focus_after),
        )
        .with_before_latex(factorized_root_latex)
        .with_after_latex(latex_expr(ctx, focus_after)),
    ];

    if let Some((replacement_before, replacement_after)) = replacement_pair {
        out.push(
            SubStep::keyed(
                "polynomial.replace_block_in_expression",
                vec![],
                human_expr(ctx, replacement_before),
                human_expr(ctx, replacement_after),
            )
            .with_before_latex(latex_expr(ctx, replacement_before))
            .with_after_latex(latex_expr(ctx, replacement_after)),
        );
    }

    out
}

fn odd_half_power_replacement_pair(
    step: &Step,
    local_before: ExprId,
    local_after: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let (Some(global_before), Some(global_after)) = (step.global_before, step.global_after) {
        if global_before != local_before || global_after != local_after {
            return Some((global_before, global_after));
        }
    }

    ((step.before != local_before) || (step.after != local_after))
        .then_some((step.before, step.after))
}

fn odd_half_power_simplify_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<OddHalfPowerSimplifyPlan> {
    let (base, numerator) = odd_half_power_components(ctx, before)?;
    let outside_power = (numerator - 1) / 2;
    matches_odd_half_power_simplified_after(ctx, after, base, outside_power).then_some(
        OddHalfPowerSimplifyPlan {
            base,
            radicand_power: numerator,
            outside_power,
        },
    )
}

fn odd_half_power_components(ctx: &Context, before: ExprId) -> Option<(ExprId, i64)> {
    if let Some(radicand) = sqrt_radicand(ctx, before) {
        let Expr::Pow(base, exponent) = ctx.get(radicand) else {
            return None;
        };
        let numerator = small_positive_integer_value(ctx, *exponent)?;
        if numerator >= 3 && numerator % 2 == 1 {
            return Some((*base, numerator));
        }
    }

    let Expr::Pow(base, exponent) = ctx.get(before) else {
        return None;
    };
    let exponent = as_rational_const(ctx, *exponent, 8)?;
    if *exponent.denom() != 2.into() {
        return None;
    }
    let numerator = exponent.numer().to_string().parse::<i64>().ok()?;
    (numerator >= 3 && numerator % 2 == 1).then_some((*base, numerator))
}

fn matches_odd_half_power_simplified_after(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    outside_power: i64,
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return false;
    }

    let mut saw_sqrt = false;
    let mut saw_outer = false;
    for factor in factors {
        if !saw_sqrt
            && sqrt_radicand(ctx, factor)
                .is_some_and(|radicand| compare_expr(ctx, radicand, base) == Ordering::Equal)
        {
            saw_sqrt = true;
            continue;
        }

        if !saw_outer && matches_odd_half_power_outer_factor(ctx, factor, base, outside_power) {
            saw_outer = true;
            continue;
        }

        return false;
    }

    saw_sqrt && saw_outer
}

fn power_display_and_latex(ctx: &Context, base: ExprId, exponent: i64) -> (String, String) {
    if exponent == 1 {
        return (human_expr(ctx, base), latex_expr(ctx, base));
    }

    let mut temp_ctx = ctx.clone();
    let exponent_expr = temp_ctx.num(exponent);
    let power_expr = temp_ctx.add_raw(Expr::Pow(base, exponent_expr));
    (
        human_expr(&temp_ctx, power_expr),
        latex_expr(&temp_ctx, power_expr),
    )
}

fn find_additive_odd_half_power_simplify_focus(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId, OddHalfPowerSimplifyPlan)> {
    let before_terms = expr_nary::add_terms_signed(ctx, before);
    let after_terms = expr_nary::add_terms_signed(ctx, after);
    if before_terms.len() < 2 || after_terms.len() != before_terms.len() {
        return None;
    }

    for (before_index, (before_focus, _before_sign)) in before_terms.iter().copied().enumerate() {
        for (after_index, (after_focus, _after_sign)) in after_terms.iter().copied().enumerate() {
            let Some(plan) = odd_half_power_simplify_plan(ctx, before_focus, after_focus) else {
                continue;
            };

            let before_passthrough =
                collect_signed_passthrough_terms_excluding_index(&before_terms, before_index);
            let after_passthrough =
                collect_signed_passthrough_terms_excluding_index(&after_terms, after_index);
            if signed_additive_term_multiset_matches(ctx, &before_passthrough, &after_passthrough) {
                return Some((before_focus, after_focus, plan));
            }
        }
    }

    None
}

pub(super) fn detect_consecutive_telescoping_sum_base(
    ctx: &Context,
    factor1: ExprId,
    factor2: ExprId,
    var: &str,
) -> Option<ExprId> {
    for (base_candidate, other_factor) in [(factor1, factor2), (factor2, factor1)] {
        let Some(base) = extract_unit_shifted_base(ctx, base_candidate, var) else {
            continue;
        };
        let mut temp_ctx = ctx.clone();
        let shifted = shifted_expr(&mut temp_ctx, base, 1);
        if compare_expr(&temp_ctx, other_factor, shifted) == std::cmp::Ordering::Equal {
            return Some(base);
        }
    }
    None
}

pub(super) fn dirichlet_kernel_base_and_n(
    ctx: &Context,
    before: ExprId,
) -> Option<(usize, usize, Vec<ExprId>)> {
    let view = AddView::from_expr(ctx, before);
    let mut has_one = false;
    let mut multiples = Vec::new();
    let mut base_factors: Option<Vec<ExprId>> = None;

    for &(term, sign) in &view.terms {
        if sign != Sign::Pos {
            return None;
        }

        if is_one(ctx, term) {
            has_one = true;
            continue;
        }

        let (multiple, candidate_base) = dirichlet_cosine_multiple(ctx, term)?;
        if let Some(existing_base) = &base_factors {
            if !same_factor_basis(ctx, existing_base, &candidate_base) {
                return None;
            }
        } else {
            base_factors = Some(candidate_base);
        }
        multiples.push(multiple);
    }

    if !has_one || multiples.is_empty() {
        return None;
    }

    multiples.sort_unstable();
    multiples.sort_unstable();
    let n = multiples.len();
    let base_multiplier = multiples.iter().copied().reduce(gcd_usize)?;
    if multiples
        .iter()
        .enumerate()
        .any(|(idx, multiple)| *multiple != (idx + 1) * base_multiplier)
    {
        return None;
    }

    Some((n, base_multiplier, base_factors?))
}

pub(super) fn generate_power_reduction_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some((kind, arg)) = trig_power_reduction_kind_and_arg(ctx, before) else {
        return Vec::new();
    };

    vec![build_power_reduction_formula_substep(ctx, kind, arg)]
}

fn build_power_reduction_formula_substep(
    ctx: &Context,
    kind: TrigPowerReductionKind,
    arg: ExprId,
) -> SubStep {
    let arg_plain = human_formula_title_plain(&display_expr(ctx, arg));
    let (title, before_plain, after_plain, before_latex, after_latex) =
        power_reduction_formula_template(kind);
    let title = format!("{title}, con u = {arg_plain}");

    schema_substep(title, before_plain, after_plain, before_latex, after_latex)
}

fn power_reduction_formula_template(
    kind: TrigPowerReductionKind,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    match kind {
        TrigPowerReductionKind::SinEvenPower => (
            "Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente",
            "sin(u)^2",
            "(1 - cos(2u)) / 2",
            "\\sin(u)^2",
            "\\frac{1-\\cos(2u)}{2}",
        ),
        TrigPowerReductionKind::CosEvenPower => (
            "Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente",
            "cos(u)^2",
            "(1 + cos(2u)) / 2",
            "\\cos(u)^2",
            "\\frac{1+\\cos(2u)}{2}",
        ),
        TrigPowerReductionKind::SinCosSquares => (
            "Usar sin²(u)·cos²(u) = (1 - cos(4u)) / 8",
            "sin(u)^2 · cos(u)^2",
            "(1 - cos(4u)) / 8",
            "\\sin(u)^2\\cdot\\cos(u)^2",
            "\\frac{1-\\cos(4u)}{8}",
        ),
    }
}

pub(super) fn extract_additive_base_and_offset_local(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, i64)> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Some(offset) = small_integer(ctx, *left) {
                return Some((*right, offset));
            }
            if let Some(offset) = small_integer(ctx, *right) {
                return Some((*left, offset));
            }
            None
        }
        Expr::Sub(left, right) => small_integer(ctx, *right).map(|offset| (*left, -offset)),
        _ => Some((expr, 0)),
    }
}

pub(super) fn render_power2_plain(base: &str) -> String {
    format!("{}^2", group_factor_plain(base))
}

pub(super) fn render_power2_latex(base: &str) -> String {
    format!("\\left({base}\\right)^{{2}}")
}

pub(super) fn generate_negative_base_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let is_contextual = step.global_before.is_some_and(|global| global != before)
        || step.global_after.is_some_and(|global| global != after);
    if !is_contextual {
        return Vec::new();
    }

    let title = if step.description.contains("^even") {
        "Usar que una potencia par elimina el signo"
    } else if step.description.contains("^odd") {
        "Usar que una potencia impar conserva el signo negativo"
    } else {
        "Simplificar la potencia con base negativa"
    };

    vec![concrete_expr_substep(ctx, title, before, after)]
}

pub(super) fn squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_small_positive_integer(ctx, *exp, 2) => Some(*base),
        Expr::Number(n) if n.is_integer() => {
            let int = n.to_integer();
            if !int.is_negative() {
                Some(expr)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn is_base_minus_one(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Sub(lhs, rhs) if *lhs == base && is_small_positive_integer(ctx, *rhs, 1))
}

pub(super) fn geometric_series_term_exponent(
    ctx: &Context,
    base: ExprId,
    term: ExprId,
) -> Option<i64> {
    if is_small_positive_integer(ctx, term, 1) {
        return Some(0);
    }
    if term == base {
        return Some(1);
    }
    match ctx.get(term) {
        Expr::Pow(pow_base, exponent) if *pow_base == base => {
            small_positive_integer_value(ctx, *exponent)
        }
        _ => None,
    }
}

pub(super) fn fourth_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_small_positive_integer(ctx, *exp, 4) => Some(*base),
        _ => None,
    }
}

pub(super) fn four_times_fourth_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if matches!(ctx.get(*left), Expr::Number(n) if n.is_integer() && n.to_integer() == 4.into()) {
        return fourth_power_base(ctx, *right);
    }
    if matches!(ctx.get(*right), Expr::Number(n) if n.is_integer() && n.to_integer() == 4.into()) {
        return fourth_power_base(ctx, *left);
    }

    None
}

pub(super) fn matches_flattened_power_multiple(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    multiplier: i64,
) -> bool {
    let Expr::Pow(expr_base, expr_exponent) = ctx.get(expr) else {
        return false;
    };
    let Expr::Pow(base_inner, base_exponent) = ctx.get(base) else {
        return false;
    };
    if compare_expr(ctx, *expr_base, *base_inner) != Ordering::Equal {
        return false;
    }
    let Some(expr_exponent) = positive_integer_literal_value(ctx, *expr_exponent) else {
        return false;
    };
    let Some(base_exponent) = positive_integer_literal_value(ctx, *base_exponent) else {
        return false;
    };
    expr_exponent == base_exponent * multiplier
}

pub(super) fn matches_fourth_power_of(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Pow(pow_base, exp)
            if is_small_positive_integer(ctx, *exp, 4)
                && cas_ast::ordering::compare_expr(ctx, *pow_base, base)
                    == std::cmp::Ordering::Equal
    )
}

pub(super) fn generate_canonicalize_nested_power_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let mut before = step.before_local().unwrap_or(step.before);
    let mut after = step.after_local().unwrap_or(step.after);

    let uses_local_pow = matches!(ctx.get(before), Expr::Pow(base, _)
        if matches!(ctx.get(*base), Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1));

    if !uses_local_pow {
        before = step.before;
        after = step.after;
    }

    if before == after {
        return Vec::new();
    }

    vec![SubStep::new(
        "Pasar la potencia al interior de la raíz",
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

pub(super) fn generate_change_of_base_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description == "Expand the logarithm using a change-of-base chain" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);

    if let Some((base, argument)) = change_of_base_log_arguments(ctx, before) {
        let mut work = ctx.clone();
        let ln_argument = work.call_builtin(BuiltinFn::Ln, vec![argument]);
        let ln_base = work.call_builtin(BuiltinFn::Ln, vec![base]);
        return [
            applied_substep(
                "Poner el argumento en el numerador",
                &work,
                argument,
                ln_argument,
                BuiltinFn::Ln,
            ),
            applied_substep(
                "Poner la base en el denominador",
                &work,
                base,
                ln_base,
                BuiltinFn::Ln,
            ),
        ]
        .into_iter()
        .flatten()
        .collect();
    }

    if let Some((argument, base, numerator, denominator)) =
        change_of_base_quotient_arguments(ctx, before)
    {
        return vec![
            temp_ctx_substep(
                "Leer el argumento desde el numerador",
                ctx,
                numerator,
                argument,
            ),
            temp_ctx_substep("Leer la base desde el denominador", ctx, denominator, base),
        ];
    }

    Vec::new()
}

pub(super) fn generate_evaluate_numeric_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Pow(base, _) = ctx.get(before) else {
        return Vec::new();
    };
    let before_human = normalize_human_power_expr(&human_expr(ctx, before));
    if is_zero(ctx, *base)
        || is_one(ctx, *base)
        || is_negative_one(ctx, *base)
        || matches!(before_human.split_once('^'), Some(("0" | "1" | "(-1)", _)))
    {
        // Evaluating 0^n, 1^n or (-1)^n is too trivial to deserve its own didactic micro-step.
        return Vec::new();
    }
    let after_human = human_expr(ctx, after);

    vec![SubStep::new(
        format!("Calcular {} = {}", before_human, after_human),
        before_human,
        after_human,
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

pub(super) fn generate_sum_difference_sixth_powers_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = sixth_power_factorized_identity_plan(ctx, after, before) else {
        return Vec::new();
    };

    let identity_latex =
        sixth_power_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let factorized_display =
        sixth_power_factorized_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
    let factorized_latex =
        sixth_power_factorized_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let factor_description = match plan.kind {
        SixthPowerIdentityKind::Sum => "Aplicar a^6 + b^6 = (a^2 + b^2)(a^4 - a^2b^2 + b^4)",
        SixthPowerIdentityKind::Difference => "Aplicar a^6 - b^6 = (a^2 - b^2)(a^4 + a^2b^2 + b^4)",
    };

    vec![SubStep::new(
        factor_description,
        display_expr(ctx, before),
        factorized_display,
    )
    .with_before_latex(identity_latex)
    .with_after_latex(factorized_latex)]
}

pub(super) fn generate_sum_difference_sixth_powers_expansion_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before;
    let after = step.after;
    let Some(plan) = sixth_power_factorized_identity_plan(ctx, before, after) else {
        return Vec::new();
    };

    let factorized_display =
        sixth_power_factorized_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
    let factorized_latex =
        sixth_power_factorized_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let identity_display =
        sixth_power_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
    let identity_latex =
        sixth_power_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let recognize_description = match plan.kind {
        SixthPowerIdentityKind::Sum => "Reconocer el patrón (a^2 + b^2)(a^4 - a^2b^2 + b^4)",
        SixthPowerIdentityKind::Difference => "Reconocer el patrón (a^2 - b^2)(a^4 + a^2b^2 + b^4)",
    };
    let expand_description = match plan.kind {
        SixthPowerIdentityKind::Sum => "Aplicar (a^2 + b^2)(a^4 - a^2b^2 + b^4) = a^6 + b^6",
        SixthPowerIdentityKind::Difference => "Aplicar (a^2 - b^2)(a^4 + a^2b^2 + b^4) = a^6 - b^6",
    };

    vec![
        SubStep::new(
            recognize_description,
            display_expr(ctx, before),
            factorized_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(factorized_latex.clone()),
        SubStep::new(expand_description, factorized_display, identity_display)
            .with_before_latex(factorized_latex)
            .with_after_latex(identity_latex),
    ]
}

pub(super) fn negative_constant_base_variable_exponent_diff_substep(
    ctx: &Context,
    target: ExprId,
    after: ExprId,
    var_name: &str,
) -> Option<SubStep> {
    let Expr::Constant(Constant::Undefined) = ctx.get(after) else {
        return None;
    };
    let Expr::Pow(base, exponent) = ctx.get(target) else {
        return None;
    };
    if contains_named_var(ctx, *base, var_name)
        || !contains_named_var(ctx, *exponent, var_name)
        || !as_rational_const(ctx, *base, 8).is_some_and(|value| value.is_negative())
    {
        return None;
    }

    Some(
        SubStep::new(
            "Detectar base negativa con exponente variable",
            display_expr(ctx, target),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, target))
        .with_after_latex(latex_expr(ctx, after)),
    )
}

pub(super) fn zero_constant_base_variable_exponent_diff_substep(
    ctx: &Context,
    target: ExprId,
    after: ExprId,
    var_name: &str,
) -> Option<SubStep> {
    let title = match ctx.get(after) {
        Expr::Constant(Constant::Undefined) => "Detectar dominio real vacío de base cero",
        _ if as_rational_const(ctx, after, 8).is_some_and(|value| value.is_zero()) => {
            "Detectar base cero con exponente variable"
        }
        _ => return None,
    };
    let Expr::Pow(base, exponent) = ctx.get(target) else {
        return None;
    };
    if contains_named_var(ctx, *base, var_name)
        || !contains_named_var(ctx, *exponent, var_name)
        || !as_rational_const(ctx, *base, 8).is_some_and(|value| value.is_zero())
    {
        return None;
    }

    Some(
        SubStep::new(title, display_expr(ctx, target), display_expr(ctx, after))
            .with_before_latex(latex_expr(ctx, target))
            .with_after_latex(latex_expr(ctx, after)),
    )
}

pub(super) fn generate_polynomial_base_table_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let mut scratch = ctx.clone();
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_base_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) {
        return Vec::new();
    }

    let Some(table_match) = polynomial_base_table_integrand(ctx, args[0], var_name) else {
        return Vec::new();
    };

    let mut substeps = vec![SubStep::new(
        table_match.title,
        display_expr(ctx, args[0]),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, args[0]))
    .with_after_latex(latex_expr(ctx, after))];
    substeps.push(
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, table_match.base)),
            format!("du = {} dx", table_match.derivative_display),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, table_match.base)))
        .with_after_latex(format!("du = {}\\,dx", table_match.derivative_latex)),
    );
    if !table_match.scale.is_one() {
        substeps.push(
            SubStep::keyed(
                "usub.adjust_constant_factor",
                vec![],
                table_match.cofactor_display,
                format!(
                    "{} · {}",
                    rational_display(&table_match.scale),
                    table_match.derivative_display
                ),
            )
            .with_before_latex(table_match.cofactor_latex)
            .with_after_latex(format!(
                "{}\\cdot {}",
                rational_latex(&table_match.scale),
                table_match.derivative_latex
            )),
        );
    }

    substeps
}

pub(super) fn polynomial_base_table_from_div(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
) -> Option<PolynomialBaseTableMatch> {
    let (negative, numerator_factors) = signed_mul_factors(ctx, num);
    if let Some(base) = polynomial_base_sqrt_arg(ctx, den) {
        return polynomial_base_match_from_cofactor(
            ctx,
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)",
            negative,
            &numerator_factors,
            base,
            var_name,
        );
    }

    if let Some((base, exponent)) = polynomial_base_pow_factor(ctx, den) {
        let title = if exponent == BigRational::new(1.into(), 2.into()) {
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)"
        } else {
            "Usar la regla de u'/u^n -> u^(1-n)/(1-n)"
        };
        return polynomial_base_match_from_cofactor(
            ctx,
            title,
            negative,
            &numerator_factors,
            base,
            var_name,
        );
    }

    polynomial_base_match_from_cofactor(
        ctx,
        "Usar la regla de u'/u -> ln|u|",
        negative,
        &numerator_factors,
        den,
        var_name,
    )
}

pub(super) fn polynomial_base_table_from_product(
    ctx: &Context,
    negative: bool,
    factors: &[ExprId],
    var_name: &str,
) -> Option<PolynomialBaseTableMatch> {
    for (power_index, factor) in factors.iter().enumerate() {
        let Some((base, exponent)) = polynomial_base_power_factor(ctx, *factor) else {
            continue;
        };
        if exponent.is_zero() || !contains_named_var(ctx, base, var_name) {
            continue;
        }

        let remaining_factors = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != power_index).then_some(*factor))
            .collect::<Vec<_>>();
        let title = if exponent == BigRational::new((-1).into(), 2.into()) {
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)"
        } else if exponent == BigRational::from_integer((-1).into()) {
            "Usar la regla de u'/u -> ln|u|"
        } else {
            "Usar la regla de u'·u^p -> u^(p+1)/(p+1)"
        };
        if let Some(table_match) = polynomial_base_match_from_cofactor(
            ctx,
            title,
            negative,
            &remaining_factors,
            base,
            var_name,
        ) {
            return Some(table_match);
        }
    }

    None
}

fn normalize_human_power_expr(value: &str) -> String {
    value.replace("((-1))", "(-1)")
}

pub(super) fn sixth_power_identity_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, SixthPowerIdentityKind)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right, SixthPowerIdentityKind::Difference)),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner, SixthPowerIdentityKind::Difference)),
            _ => Some((*left, *right, SixthPowerIdentityKind::Sum)),
        },
        _ => None,
    }
}

pub(super) fn cube_base_from_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, 3) => Some(*base),
        _ if is_one(ctx, expr) => Some(expr),
        _ => None,
    }
}

pub(super) fn cube_base_from_term_with_witness(
    ctx: &Context,
    expr: ExprId,
    witness: ExprId,
) -> Option<ExprId> {
    cube_base_from_term(ctx, expr).or_else(|| {
        if is_negative_one(ctx, expr) {
            find_one_literal(ctx, witness)
        } else if let Some((base, cube_root_exponent)) = cube_root_power_term(ctx, expr) {
            find_power_literal(ctx, witness, base, cube_root_exponent)
        } else {
            None
        }
    })
}

fn find_power_literal(ctx: &Context, expr: ExprId, base: ExprId, exponent: i64) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(candidate_base, candidate_exponent)
            if compare_expr(ctx, *candidate_base, base) == Ordering::Equal
                && is_integer_literal(ctx, *candidate_exponent, exponent) =>
        {
            Some(expr)
        }
        Expr::Neg(inner) => find_power_literal(ctx, *inner, base, exponent),
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => find_power_literal(ctx, *left, base, exponent)
            .or_else(|| find_power_literal(ctx, *right, base, exponent)),
        Expr::Function(_, args) => args
            .iter()
            .find_map(|arg| find_power_literal(ctx, *arg, base, exponent)),
        _ => None,
    }
}

pub(super) fn sixth_power_base_from_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, 6) => Some(*base),
        _ => None,
    }
}

fn sixth_power_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let op = match kind {
        SixthPowerIdentityKind::Sum => " + ",
        SixthPowerIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        sixth_power_display(ctx, left_base),
        op,
        sixth_power_display(ctx, right_base)
    )
}

fn sixth_power_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let op = match kind {
        SixthPowerIdentityKind::Sum => " + ",
        SixthPowerIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        sixth_power_latex(ctx, left_base),
        op,
        sixth_power_latex(ctx, right_base)
    )
}

fn sixth_power_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^6")
    } else {
        format!("({display})^6")
    }
}

fn sixth_power_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{6}}")
}

pub(super) fn fourth_power_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^4")
    } else {
        format!("({display})^4")
    }
}

pub(super) fn fourth_power_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{4}}")
}

pub(super) fn is_simple_power_base(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::Function(_, _)
    )
}
