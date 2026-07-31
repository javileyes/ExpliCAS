//! `focused_rule_substeps`: familia `factoring`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn generate_factor_out_with_division_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    let factored_target = detect_factor_out_with_division_substep_target(
        ctx,
        step.after_local().unwrap_or(step.after),
    );
    let Some((factor_expr, inner_expr)) = factored_target else {
        let Some(factor_display) = detect_factor_out_with_division_substep_factor(
            ctx,
            step.after_local().unwrap_or(step.after),
        )
        .map(|expr| human_expr(ctx, expr))
        .or_else(|| {
            step.description
                .strip_prefix("Factor out ")
                .and_then(|tail| tail.strip_suffix(" from the whole expression"))
                .map(str::to_string)
        }) else {
            return Vec::new();
        };
        return vec![concrete_expr_substep(
            ctx,
            format!(
                "Reescribir los términos que no llevan {factor_display} usando el factor común"
            ),
            before,
            after,
        )];
    };

    let factor_display = human_expr(ctx, factor_expr);
    let factor_latex = latex_expr(ctx, factor_expr);
    let Some(expanded_terms) =
        factored_division_expanded_terms(ctx, inner_expr, &factor_display, &factor_latex)
    else {
        return Vec::new();
    };

    vec![
        formula_substep(
            format!("Reescribir cada término con el factor común {factor_display}"),
            &display_expr(ctx, before),
            &expanded_terms.plain,
            &latex_expr(ctx, before),
            &expanded_terms.latex,
        ),
        formula_substep(
            format!("Sacar el factor común {factor_display}"),
            &expanded_terms.plain,
            &display_expr(ctx, after),
            &expanded_terms.latex,
            &latex_expr(ctx, after),
        ),
    ]
}

fn factored_division_expanded_terms(
    ctx: &Context,
    inner_expr: ExprId,
    factor_display: &str,
    factor_latex: &str,
) -> Option<FactoredDivisionExpandedTerms> {
    let terms = collect_add_chain_terms_readonly(ctx, inner_expr);
    if terms.len() < 2 {
        return None;
    }

    let mut plain = String::new();
    let mut latex = String::new();
    for (idx, signed_term) in terms.iter().enumerate() {
        let term_plain = human_expr(ctx, signed_term.term);
        let term_latex = latex_expr(ctx, signed_term.term);
        let plain_piece = format!("{factor_display}·({term_plain})");
        let latex_piece = format!("{factor_latex}\\cdot \\left({term_latex}\\right)");

        if idx == 0 {
            if signed_term.negative {
                plain.push('-');
                latex.push('-');
            }
        } else if signed_term.negative {
            plain.push_str(" - ");
            latex.push_str(" - ");
        } else {
            plain.push_str(" + ");
            latex.push_str(" + ");
        }

        plain.push_str(&plain_piece);
        latex.push_str(&latex_piece);
    }

    Some(FactoredDivisionExpandedTerms { plain, latex })
}

fn detect_factor_out_with_division_substep_target(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    let factors = collect_mul_chain_factors_readonly(ctx, expr);
    for (idx, factor) in factors.iter().copied().enumerate() {
        let mut remaining = factors.clone();
        remaining.remove(idx);
        if remaining.is_empty() {
            continue;
        }
        if let Some(inner) = single_remaining_factor_with_division_by(ctx, &remaining, factor) {
            return Some((factor, inner));
        }
    }

    None
}

fn single_remaining_factor_with_division_by(
    ctx: &Context,
    remaining: &[ExprId],
    factor: ExprId,
) -> Option<ExprId> {
    if remaining.len() == 1 && contains_division_by_exact_factor(ctx, remaining[0], factor) {
        return Some(remaining[0]);
    }
    None
}

fn detect_factor_out_with_division_substep_factor(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    detect_factor_out_with_division_substep_target(ctx, expr)
        .map(|(factor, _inner)| factor)
        .or_else(|| detect_factor_out_with_division_substep_factor_from_flat_target(ctx, expr))
}

fn detect_factor_out_with_division_substep_factor_from_flat_target(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    let factors = collect_mul_chain_factors_readonly(ctx, expr);
    for (idx, factor) in factors.iter().copied().enumerate() {
        let mut remaining = factors.clone();
        remaining.remove(idx);
        if remaining
            .into_iter()
            .any(|inner_expr| contains_division_by_exact_factor(ctx, inner_expr, factor))
        {
            return Some(factor);
        }
    }

    None
}

pub(super) fn collect_mul_chain_factors_readonly_into(
    ctx: &Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_chain_factors_readonly_into(ctx, *left, out);
            collect_mul_chain_factors_readonly_into(ctx, *right, out);
        }
        _ => out.push(expr),
    }
}

fn contains_division_by_exact_factor(ctx: &Context, expr: ExprId, factor: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(_, den) => compare_expr(ctx, *den, factor) == std::cmp::Ordering::Equal,
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            contains_division_by_exact_factor(ctx, *left, factor)
                || contains_division_by_exact_factor(ctx, *right, factor)
        }
        Expr::Pow(base, exp) => {
            contains_division_by_exact_factor(ctx, *base, factor)
                || contains_division_by_exact_factor(ctx, *exp, factor)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_division_by_exact_factor(ctx, *inner, factor)
        }
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| contains_division_by_exact_factor(ctx, *arg, factor)),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|item| contains_division_by_exact_factor(ctx, *item, factor)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

pub(super) fn generate_expand_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let conjugate_product = generate_conjugate_product_expansion_substeps(ctx, before, after);
    if !conjugate_product.is_empty() {
        return conjugate_product;
    }

    if polynomial_product_didactic_plan(ctx, before).is_some() {
        return generate_polynomial_product_normalize_substeps(ctx, step);
    }

    let perfect_square_cancel = generate_perfect_square_fraction_cancel_substeps(ctx, step);
    if !perfect_square_cancel.is_empty() {
        return perfect_square_cancel;
    }

    let difference_of_squares_cancel = generate_difference_of_squares_cancel_substeps(ctx, step);
    if !difference_of_squares_cancel.is_empty() {
        return difference_of_squares_cancel;
    }

    let sum_difference_cubes_cancel = generate_sum_difference_cubes_cancel_substeps(ctx, step);
    if !sum_difference_cubes_cancel.is_empty() {
        return sum_difference_cubes_cancel;
    }

    if let Some((factor, kind)) = common_factor_factorization_plan(ctx, after, before) {
        let factor_display = human_expr(ctx, factor);
        let factor_latex = latex_expr(ctx, factor);
        let _ = (factor_display, factor_latex, kind);
        return vec![formula_substep(
            "Usar la distributiva",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    let sophie_germain = generate_sophie_germain_expansion_substeps(ctx, step);
    if !sophie_germain.is_empty() {
        return sophie_germain;
    }

    Vec::new()
}

pub(super) fn conjugate_product_difference_of_squares_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = expr_nary::mul_leaves(ctx, before);
    if factors.len() != 2 {
        return None;
    }

    let (left_base, right_base) =
        cas_math::expr_relations::conjugate_add_sub_pair(ctx, factors[0], factors[1])?;
    let mut work = ctx.clone();
    let expected_raw = build_difference_of_squares_expansion(&mut work, left_base, right_base);
    let expected = simplify_expr_in_context(&mut work, expected_raw);
    if poly_eq(&work, expected, after) {
        Some((left_base, right_base))
    } else {
        None
    }
}

fn build_difference_of_squares_expansion(
    ctx: &mut Context,
    left_base: ExprId,
    right_base: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let left_sq = ctx.add_raw(Expr::Pow(left_base, two));
    let right_sq = ctx.add_raw(Expr::Pow(right_base, two));
    ctx.add_raw(Expr::Sub(left_sq, right_sq))
}

pub(super) fn generate_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substeps) = generate_consecutive_telescoping_fraction_substeps(ctx, before, after) {
        return substeps;
    }

    if let Some((base, power)) = geometric_difference_factor_plan(ctx, before, after) {
        let base_display = human_expr(ctx, base);

        return vec![concrete_expr_substep(
            ctx,
            format!("Aquí la diferencia de potencias usa base {base_display} y exponente {power}"),
            before,
            after,
        )];
    }

    if let Some(base) = full_sixth_power_minus_one_factor_plan(ctx, before, after) {
        let base_display = human_expr(ctx, base);

        return vec![concrete_expr_substep(
            ctx,
            format!(
                "Aquí la diferencia de sexto grado se factoriza completamente con base {base_display}"
            ),
            before,
            after,
        )];
    }

    if let Some((factor, kind)) = common_factor_factorization_plan(ctx, before, after) {
        let factor_display = human_expr(ctx, factor);
        let _ = kind;
        return vec![concrete_expr_substep(
            ctx,
            format!("Aquí el factor común es {factor_display}"),
            before,
            after,
        )];
    }

    if let Some((left, right)) = difference_of_squares_bases(ctx, before) {
        let left_display = human_expr(ctx, left);
        let right_display = human_expr(ctx, right);

        return vec![concrete_expr_substep(
            ctx,
            format!("Aquí la diferencia de cuadrados usa bases {left_display} y {right_display}"),
            before,
            after,
        )];
    }

    if cube_identity_plan(ctx, before, after).is_some() {
        return generate_sum_difference_cubes_substeps(ctx, step);
    }

    if let Some((left, right, kind, power)) = binomial_power_terms(ctx, after) {
        if power == 3 {
            let (left, right) = prefer_non_constant_term_first(ctx, left, right);
            let left_display = human_expr(ctx, left);
            let right_display = human_expr(ctx, right);
            let left_latex = latex_expr(ctx, left);
            let right_latex = latex_expr(ctx, right);
            let (title, before_display, before_latex) = match kind {
                BinomialSquareKind::Sum => (
                    "Usar a^3 + 3a^2b + 3ab^2 + b^3 = (a + b)^3",
                    format!(
                        "{left_display}^3 + 3 · {left_display}^2 · {right_display} + 3 · {left_display} · {right_display}^2 + {right_display}^3"
                    ),
                    format!(
                        "{left_latex}^3 + 3\\cdot {left_latex}^2\\cdot {right_latex} + 3\\cdot {left_latex}\\cdot {right_latex}^2 + {right_latex}^3"
                    ),
                ),
                BinomialSquareKind::Difference => (
                    "Usar a^3 - 3a^2b + 3ab^2 - b^3 = (a - b)^3",
                    format!(
                        "{left_display}^3 - 3 · {left_display}^2 · {right_display} + 3 · {left_display} · {right_display}^2 - {right_display}^3"
                    ),
                    format!(
                        "{left_latex}^3 - 3\\cdot {left_latex}^2\\cdot {right_latex} + 3\\cdot {left_latex}\\cdot {right_latex}^2 - {right_latex}^3"
                    ),
                ),
            };

            let _ = (left_display, right_display, left_latex, right_latex);
            return vec![formula_substep(
                title,
                &before_display,
                &human_expr(ctx, after),
                &before_latex,
                &latex_expr(ctx, after),
            )];
        }
    }

    if let Some((left, right, kind)) = binomial_square_terms(ctx, after) {
        let (left, right) = prefer_non_constant_term_first(ctx, left, right);
        let left_display = human_expr(ctx, left);
        let right_display = human_expr(ctx, right);
        let left_latex = latex_expr(ctx, left);
        let right_latex = latex_expr(ctx, right);
        let (title, before_display, before_latex) = match kind {
            BinomialSquareKind::Sum => (
                "Usar a^2 + 2ab + b^2 = (a + b)^2",
                format!(
                    "{left_display}^2 + 2 · {left_display} · {right_display} + {right_display}^2"
                ),
                format!(
                    "{left_latex}^2 + 2\\cdot {left_latex}\\cdot {right_latex} + {right_latex}^2"
                ),
            ),
            BinomialSquareKind::Difference => (
                "Usar a^2 - 2ab + b^2 = (a - b)^2",
                format!(
                    "{left_display}^2 - 2 · {left_display} · {right_display} + {right_display}^2"
                ),
                format!(
                    "{left_latex}^2 - 2\\cdot {left_latex}\\cdot {right_latex} + {right_latex}^2"
                ),
            ),
        };

        let _ = (left_display, right_display, left_latex, right_latex);
        return vec![formula_substep(
            title,
            &before_display,
            &human_expr(ctx, after),
            &before_latex,
            &latex_expr(ctx, after),
        )];
    }

    let sophie_germain = generate_sophie_germain_factorization_substeps(ctx, step);
    if !sophie_germain.is_empty() {
        return sophie_germain;
    }

    if let Some(vars) = alternating_cubic_vandermonde_plan(ctx, before, after) {
        return generate_alternating_cubic_vandermonde_substeps(ctx, before, vars);
    }

    Vec::new()
}

pub(super) fn vandermonde_remaining_factor_substep(
    ctx: &Context,
    before: ExprId,
    a: &str,
    b: &str,
    c: &str,
) -> SubStep {
    let source_display = human_expr(ctx, before);
    let source_latex = latex_expr(ctx, before);
    let denominator_display = format!("({a} - {b}) · ({a} - {c}) · ({b} - {c})");
    let denominator_latex = format!("({a} - {b})({a} - {c})({b} - {c})");
    let quotient_display = format!("({source_display})/({denominator_display})");
    let quotient_latex = format!("\\frac{{{source_latex}}}{{{denominator_latex}}}");
    let remaining = format!("{a} + {b} + {c}");

    SubStep::new(
        format!("El cociente restante es {remaining}"),
        quotient_display,
        remaining.clone(),
    )
    .with_before_latex(quotient_latex)
    .with_after_latex(remaining)
}

pub(super) fn generate_binomial_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((left, right, kind, power, local_focus_before, local_focus_after)) =
        binomial_power_terms(ctx, before)
            .map(|(left, right, kind, power)| (left, right, kind, power, before, after))
            .or_else(|| find_binomial_expansion_focus_substep_sides(ctx, step.before, step.after))
    else {
        return Vec::new();
    };

    let _ = (left, right);
    let identity_title = match (kind, power) {
        (BinomialSquareKind::Sum, 2) => "Cuadrado de la suma desarrollado",
        (BinomialSquareKind::Difference, 2) => "Cuadrado de la diferencia desarrollado",
        (BinomialSquareKind::Sum, 3) => "Cubo de la suma desarrollado",
        (BinomialSquareKind::Difference, 3) => "Cubo de la diferencia desarrollado",
        _ => return Vec::new(),
    };

    vec![concrete_expr_substep(
        ctx,
        identity_title,
        local_focus_before,
        local_focus_after,
    )]
}

fn find_binomial_expansion_focus_substep_sides(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId, BinomialSquareKind, i64, ExprId, ExprId)> {
    let mut candidates = Vec::new();
    collect_subexpr_ids(ctx, before, &mut candidates);

    for candidate in candidates {
        let Some((left, right, kind, power)) = binomial_power_terms(ctx, candidate) else {
            continue;
        };

        let mut work = ctx.clone();
        let expanded = build_binomial_expansion_expr(&mut work, left, right, kind, power)?;
        let intermediate = substitute_expr_by_id(&mut work, before, candidate, expanded);
        if compare_expr(&work, intermediate, after) == Ordering::Equal
            || same_presentational_expr(&work, intermediate, &work, after)
        {
            return Some((left, right, kind, power, candidate, expanded));
        }
    }

    None
}

pub(super) fn matches_odd_half_power_outer_factor(
    ctx: &Context,
    factor: ExprId,
    base: ExprId,
    outside_power: i64,
) -> bool {
    if outside_power == 1 && compare_expr(ctx, factor, base) == Ordering::Equal {
        return true;
    }

    if outside_power == 1
        && abs_argument(ctx, factor)
            .is_some_and(|inner| compare_expr(ctx, inner, base) == Ordering::Equal)
    {
        return true;
    }

    match ctx.get(factor) {
        Expr::Pow(pow_base, exponent)
            if small_positive_integer_value(ctx, *exponent) == Some(outside_power) =>
        {
            compare_expr(ctx, *pow_base, base) == Ordering::Equal
                || abs_argument(ctx, *pow_base)
                    .is_some_and(|inner| compare_expr(ctx, inner, base) == Ordering::Equal)
        }
        _ => false,
    }
}

/// `whole ≡ factor · rest`, decided EXACTLY (build the difference and simplify
/// it to zero), not assumed.
///
/// The reverse-nested-fraction narrator was written for one direction —
/// `(c+d)/(a(c+d)+b) → 1/(a + b/(c+d))`, where `a(c+d)+b = (c+d)·(a + b/(c+d))`
/// really holds — but it fired whenever the pattern MATCHED, including when
/// `before_den` and `after_den` are the same expression rewritten. There it
/// published `A = (1-x)²·A`, false unless `(1-x)² = 1`
/// (`diff(arctan((1+x)/(1-x)), x)`, residual 86.70 / 35.17 / -6.83 measured at
/// three points). Declining is the honest outcome when the identity does not
/// hold.
pub(super) fn factors_exactly(ctx: &Context, whole: ExprId, factor: ExprId, rest: ExprId) -> bool {
    let mut scratch = ctx.clone();
    let product = scratch.add(Expr::Mul(factor, rest));
    let difference = scratch.add(Expr::Sub(whole, product));
    let simplified = simplify_expr_in_context(&mut scratch, difference);
    is_zero(&scratch, simplified)
}

pub(super) fn generate_distributive_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((factor, terms)) = distributive_product_terms(ctx, before) else {
        return Vec::new();
    };
    let Some((products_display, products_latex)) = distributive_product_list(ctx, factor, &terms)
    else {
        return Vec::new();
    };

    vec![
        formula_substep(
            "Identificar los productos que genera la distributiva",
            &human_expr(ctx, before),
            &products_display,
            &latex_expr(ctx, before),
            &products_latex,
        ),
        formula_substep(
            "Escribir los productos con los signos originales",
            &products_display,
            &human_expr(ctx, after),
            &products_latex,
            &latex_expr(ctx, after),
        ),
    ]
}

fn distributive_product_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<(ExprId, Sign)>)> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    let left_terms = AddView::from_expr(ctx, *left).terms;
    if left_terms.len() >= 2 {
        return Some((*right, left_terms.to_vec()));
    }

    let right_terms = AddView::from_expr(ctx, *right).terms;
    if right_terms.len() >= 2 {
        return Some((*left, right_terms.to_vec()));
    }

    None
}

fn distributive_product_list(
    ctx: &Context,
    factor: ExprId,
    terms: &[(ExprId, Sign)],
) -> Option<(String, String)> {
    if terms.len() < 2 {
        return None;
    }

    let products: Vec<_> = terms
        .iter()
        .map(|(term, sign)| signed_distributive_product(ctx, factor, *term, *sign))
        .collect();
    let displays = products
        .iter()
        .map(|(display, _latex)| display.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let latex = products
        .iter()
        .map(|(_display, latex)| latex.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    Some((displays, latex))
}

fn signed_distributive_product(
    ctx: &Context,
    factor: ExprId,
    term: ExprId,
    sign: Sign,
) -> (String, String) {
    let mut work = ctx.clone();
    let product = work.add_raw(Expr::Mul(factor, term));
    let signed_product = match sign {
        Sign::Pos => product,
        Sign::Neg => work.add_raw(Expr::Neg(product)),
    };
    (
        human_expr(&work, signed_product),
        latex_expr(&work, signed_product),
    )
}

pub(super) fn generate_factorized_finite_product_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(call) = try_extract_finite_aggregate_call(ctx, before, "product") else {
        return Vec::new();
    };
    let Some(u_expr) = detect_factorized_telescoping_square_base(ctx, call.term, &call.var_name)
    else {
        return Vec::new();
    };

    let mut temp_ctx = ctx.clone();
    let start_base = substitute_expr_by_id(&mut temp_ctx, u_expr, call.var_expr, call.start_expr);
    let end_base = substitute_expr_by_id(&mut temp_ctx, u_expr, call.var_expr, call.end_expr);
    let (first_plain, first_latex) = render_temp_expr(&temp_ctx, start_base);
    let (second_plain, second_latex) = shifted_expr_strings(&temp_ctx, start_base, 1);
    let (last_plain, last_latex) = render_temp_expr(&temp_ctx, end_base);
    let (last_plus_one_plain, last_plus_one_latex) = shifted_expr_strings(&temp_ctx, end_base, 1);
    let (first_minus_one_plain, first_minus_one_latex) =
        shifted_expr_strings(&temp_ctx, start_base, -1);

    let factorized_series_plain = format!(
        "{} · {} · … · {}",
        render_fraction_plain(
            &render_square_difference_plain(&first_plain),
            &render_power2_plain(&first_plain),
        ),
        render_fraction_plain(
            &render_square_difference_plain(&second_plain),
            &render_power2_plain(&second_plain),
        ),
        render_fraction_plain(
            &render_square_difference_plain(&last_plain),
            &render_power2_plain(&last_plain),
        ),
    );
    let factorized_series_latex = format!(
        "{}\\cdot {}\\cdot \\cdots \\cdot {}",
        render_fraction_latex(
            &render_square_difference_latex(&first_latex),
            &render_power2_latex(&first_latex),
        ),
        render_fraction_latex(
            &render_square_difference_latex(&second_latex),
            &render_power2_latex(&second_latex),
        ),
        render_fraction_latex(
            &render_square_difference_latex(&last_latex),
            &render_power2_latex(&last_latex),
        ),
    );
    let telescoped_plain = render_fraction_plain(
        &format!(
            "{} · {}",
            group_factor_plain(&first_minus_one_plain),
            group_factor_plain(&last_plus_one_plain)
        ),
        &format!(
            "{} · {}",
            group_factor_plain(&first_plain),
            group_factor_plain(&last_plain)
        ),
    );
    let telescoped_latex = render_fraction_latex(
        &format!(
            "{}\\cdot {}",
            group_factor_latex(&first_minus_one_latex),
            group_factor_latex(&last_plus_one_latex)
        ),
        &format!(
            "{}\\cdot {}",
            group_factor_latex(&first_latex),
            group_factor_latex(&last_latex)
        ),
    );
    let after_plain = human_expr(ctx, after);
    let after_latex = latex_expr(ctx, after);

    let mut out = vec![
        formula_substep(
            "Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2",
            &human_expr(ctx, before),
            &factorized_series_plain,
            &latex_expr(ctx, before),
            &factorized_series_latex,
        ),
        formula_substep(
            "Los factores (u + 1) y (u - 1) se cancelan telescópicamente",
            &factorized_series_plain,
            &telescoped_plain,
            &factorized_series_latex,
            &telescoped_latex,
        ),
    ];

    if !same_math_render(&telescoped_latex, &after_latex) {
        out.push(formula_substep(
            "Solo quedan el primer factor u - 1 y el último factor u + 1",
            &telescoped_plain,
            &after_plain,
            &telescoped_latex,
            &after_latex,
        ));
    }
    out
}

pub(super) fn nt_prime_factors(mut n: i64) -> Vec<(i64, u32)> {
    let mut factors = Vec::new();
    n = n.abs();
    let mut d = 2i64;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            let mut e = 0u32;
            while n % d == 0 {
                n /= d;
                e += 1;
            }
            factors.push((d, e));
        }
        d += 1;
    }
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

pub(super) fn nt_factorization_string(factors: &[(i64, u32)]) -> String {
    if factors.is_empty() {
        return "1".to_string();
    }
    factors
        .iter()
        .map(|&(p, e)| {
            if e == 1 {
                p.to_string()
            } else {
                format!("{p}^{e}")
            }
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

pub(super) fn generate_binomial_symmetry_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((n, k, complement)) = choose_symmetry_data(ctx, before, after) else {
        return Vec::new();
    };

    let complement_plain = format!("C({n}, {n} - {k})");
    let complement_latex = format!("\\binom{{{n}}}{{{n}-{k}}}");

    vec![
        formula_substep(
            format!("Usar C({n},{k}) = C({n},{n}-{k})"),
            &binom_plain(n, k),
            &complement_plain,
            &binom_latex(n, k),
            &complement_latex,
        ),
        formula_substep(
            format!("Calcular {n}-{k} = {complement}"),
            &complement_plain,
            &binom_plain(n, complement),
            &complement_latex,
            &binom_latex(n, complement),
        ),
    ]
}

pub(super) fn same_factor_basis(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(lhs, rhs)| compare_expr(ctx, *lhs, *rhs).is_eq())
}

pub(super) fn render_factor_basis(ctx: &Context, factors: &[ExprId]) -> (String, String) {
    match factors {
        [] => {
            let mut temp_ctx = ctx.clone();
            let basis = build_balanced_mul(&mut temp_ctx, factors);
            render_temp_expr(&temp_ctx, basis)
        }
        [single] => render_temp_expr(ctx, *single),
        _ => {
            let plain = factors
                .iter()
                .map(|factor| render_factor_piece_plain(ctx, *factor))
                .collect::<Vec<_>>()
                .join(" · ");
            let latex = factors
                .iter()
                .map(|factor| render_factor_piece_latex(ctx, *factor))
                .collect::<Vec<_>>()
                .join("\\cdot ");
            (plain, latex)
        }
    }
}

fn render_factor_piece_plain(ctx: &Context, expr: ExprId) -> String {
    let (plain, _) = render_temp_expr(ctx, expr);
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => format!("({plain})"),
        _ => plain,
    }
}

fn render_factor_piece_latex(ctx: &Context, expr: ExprId) -> String {
    let (_, latex) = render_temp_expr(ctx, expr);
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => format!("\\left({latex}\\right)"),
        _ => latex,
    }
}

pub(super) fn generate_pythagorean_high_power_factor_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let before = human_expr(ctx, local_before);

    if before.contains("sin(") {
        if let Some(substeps) = build_pythagorean_high_power_sine_substeps(ctx, local_before) {
            return substeps;
        }
    }

    if before.starts_with("4 · cos(") && before.contains("^3") {
        if let Some(substeps) = build_pythagorean_high_power_cos_substeps(ctx, local_before, false)
        {
            return substeps;
        }
    }

    if before.contains("cos(") && before.contains("^3") {
        if let Some(substeps) = build_pythagorean_high_power_cos_substeps(ctx, local_before, true) {
            return substeps;
        }
    }

    Vec::new()
}

/// Migrated to the matcher (2026-07-27): the old emitter routed on the
/// description PREFIX («1 - sin²…»), which missed the expansion and negated
/// variants the derive route produces (`sin² ⟹ 1 - cos²`, `sin² - 1 ⟹
/// -cos²`). The table carries every oriented form the rule applies;
/// structural-first adjudication picks the row the reader sees.
pub(super) fn generate_pythagorean_factor_form_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    const FACTOR_FORM_TEMPLATES: [(&str, &str); 4] = [
        ("1 - sin(u)^2", "cos(u)^2"),
        ("1 - cos(u)^2", "sin(u)^2"),
        ("sin(u)^2 - 1", "-cos(u)^2"),
        ("cos(u)^2 - 1", "-sin(u)^2"),
    ];
    named_identity_from_table(ctx, &FACTOR_FORM_TEMPLATES, before, after)
        .into_iter()
        .collect()
}

pub(super) fn generate_consecutive_factorial_ratio_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((expanded, gap)) = build_consecutive_factorial_ratio_expansion(ctx, before) else {
        return vec![
            schema_substep(
                "Escribir el factorial superior como el siguiente número por el factorial anterior",
                "(k + 1)! / k!",
                "((k + 1) · k!) / k!",
                "\\frac{(k+1)!}{k!}",
                "\\frac{(k+1)\\cdot k!}{k!}",
            ),
            schema_substep(
                "Cancelar el factorial común",
                "((k + 1) · k!) / k!",
                "k + 1",
                "\\frac{(k+1)\\cdot k!}{k!}",
                "k + 1",
            ),
        ];
    };

    let mut work = ctx.clone();
    let expanded_in_work = rebuild_consecutive_factorial_ratio_expansion(&mut work, before)
        .map(|(expr, _)| expr)
        .unwrap_or(expanded);
    let (before_display, before_latex) = render_temp_expr(&work, before);
    let (expanded_display, expanded_latex) = render_temp_expr(&work, expanded_in_work);
    let (after_display, after_latex) = render_temp_expr(&work, after);

    let first_title = if gap == 1 {
        "Escribir el factorial superior como el siguiente número por el factorial anterior"
            .to_string()
    } else {
        "Expandir el factorial superior hasta llegar al factorial inferior".to_string()
    };

    vec![
        SubStep::new(first_title, before_display, expanded_display)
            .with_before_latex(before_latex)
            .with_after_latex(expanded_latex),
        SubStep::new(
            "Cancelar el factorial común",
            human_expr(&work, expanded_in_work),
            after_display,
        )
        .with_before_latex(latex_expr(&work, expanded_in_work))
        .with_after_latex(after_latex),
    ]
}

fn extract_factorial_call_arg_local(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.sym_name(*fn_id), "fact" | "factorial") =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn rebuild_consecutive_factorial_ratio_expansion(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, i64)> {
    let (num, den) = as_div(ctx, expr)?;
    let num_arg = extract_factorial_call_arg_local(ctx, num)?;
    let den_arg = extract_factorial_call_arg_local(ctx, den)?;
    let (num_base, num_offset) = extract_additive_base_and_offset_local(ctx, num_arg)?;
    let (den_base, den_offset) = extract_additive_base_and_offset_local(ctx, den_arg)?;
    if compare_expr(ctx, num_base, den_base) != Ordering::Equal {
        return None;
    }

    let gap = num_offset - den_offset;
    if gap <= 0 {
        return None;
    }

    let mut descending_factors = Vec::with_capacity(gap as usize);
    for shift in (1..=gap).rev() {
        let factor_offset = den_offset + shift;
        let factor = if factor_offset == num_offset {
            num_arg
        } else {
            rebuild_expr_with_offset_local(ctx, den_base, factor_offset)
        };
        descending_factors.push(factor);
    }

    let leading = build_balanced_mul(ctx, &descending_factors);
    let expanded_num = ctx.add(Expr::Mul(leading, den));
    Some((ctx.add(Expr::Div(expanded_num, den)), gap))
}

fn build_consecutive_factorial_ratio_expansion(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, i64)> {
    let mut work = ctx.clone();
    rebuild_consecutive_factorial_ratio_expansion(&mut work, expr)
}

pub(super) fn display_literal_factors(ctx: &Context, literal_factors: &[ExprId]) -> String {
    let mut parts = literal_factors
        .iter()
        .map(|factor| display_expr(ctx, *factor))
        .collect::<Vec<_>>();
    if parts.len() == 1 {
        return parts.remove(0);
    }
    cas_formatter::clean_display_string(&parts.join(" · "))
}

/// True when the rendered fragment can sit next to `·` or under `^` without
/// parentheses. Operates on already-rendered text, so the decision is by
/// character class — the same operator set that `needs_parenthesized_power_base`
/// uses in latex_plain_text.rs; anything in it means the fragment is composite.
fn is_atomic_rendered_factor(fragment: &str) -> bool {
    !fragment
        .chars()
        .any(|ch| matches!(ch, ' ' | '+' | '-' | '·' | '/' | '^' | '*'))
}

pub(super) fn group_factor_plain(fragment: &str) -> String {
    if is_atomic_rendered_factor(fragment) {
        fragment.to_string()
    } else {
        format!("({fragment})")
    }
}

pub(super) fn group_factor_latex(fragment: &str) -> String {
    if is_atomic_rendered_factor(fragment) {
        fragment.to_string()
    } else {
        format!("\\left({fragment}\\right)")
    }
}

fn difference_of_squares_bases(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    let left_base = squared_base(ctx, *left)?;
    let right_base = squared_base(ctx, *right)?;
    Some((left_base, right_base))
}

pub(super) fn binomial_square_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, BinomialSquareKind)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_small_positive_integer(ctx, *exp, 2) {
        return None;
    }
    match ctx.get(*base) {
        Expr::Add(left, right) => Some((*left, *right, BinomialSquareKind::Sum)),
        Expr::Sub(left, right) => Some((*left, *right, BinomialSquareKind::Difference)),
        _ => None,
    }
}

fn binomial_power_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, BinomialSquareKind, i64)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let power = if is_small_positive_integer(ctx, *exp, 2) {
        2
    } else if is_small_positive_integer(ctx, *exp, 3) {
        3
    } else {
        return None;
    };
    match ctx.get(*base) {
        Expr::Add(left, right) => Some((*left, *right, BinomialSquareKind::Sum, power)),
        Expr::Sub(left, right) => Some((*left, *right, BinomialSquareKind::Difference, power)),
        _ => None,
    }
}

fn build_binomial_expansion_expr(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    kind: BinomialSquareKind,
    power: i64,
) -> Option<ExprId> {
    let add_signed = |ctx: &mut Context, terms: &[(ExprId, Sign)]| {
        let mut iter = terms.iter();
        let Some((first_term, first_sign)) = iter.next() else {
            return ctx.num(0);
        };
        let mut acc = if *first_sign == Sign::Pos {
            *first_term
        } else {
            ctx.add(Expr::Neg(*first_term))
        };
        for (term, sign) in iter {
            acc = if *sign == Sign::Pos {
                ctx.add(Expr::Add(acc, *term))
            } else {
                ctx.add(Expr::Sub(acc, *term))
            };
        }
        acc
    };
    let pow = |ctx: &mut Context, base: ExprId, exp: i64| {
        let exponent = ctx.num(exp);
        ctx.add(Expr::Pow(base, exponent))
    };
    let mul = |ctx: &mut Context, factors: &[ExprId]| build_balanced_mul(ctx, factors);

    let two = ctx.num(2);
    let three = ctx.num(3);
    let left_sq = pow(ctx, left, 2);
    let right_sq = pow(ctx, right, 2);

    Some(match (kind, power) {
        (BinomialSquareKind::Sum, 2) => {
            let cross = mul(ctx, &[two, left, right]);
            add_signed(
                ctx,
                &[
                    (left_sq, Sign::Pos),
                    (cross, Sign::Pos),
                    (right_sq, Sign::Pos),
                ],
            )
        }
        (BinomialSquareKind::Difference, 2) => {
            let cross = mul(ctx, &[two, left, right]);
            add_signed(
                ctx,
                &[
                    (left_sq, Sign::Pos),
                    (cross, Sign::Neg),
                    (right_sq, Sign::Pos),
                ],
            )
        }
        (BinomialSquareKind::Sum, 3) => {
            let left_cu = pow(ctx, left, 3);
            let right_cu = pow(ctx, right, 3);
            let left_sq_right = mul(ctx, &[three, left_sq, right]);
            let left_right_sq = mul(ctx, &[three, left, right_sq]);
            add_signed(
                ctx,
                &[
                    (left_cu, Sign::Pos),
                    (left_sq_right, Sign::Pos),
                    (left_right_sq, Sign::Pos),
                    (right_cu, Sign::Pos),
                ],
            )
        }
        (BinomialSquareKind::Difference, 3) => {
            let left_cu = pow(ctx, left, 3);
            let right_cu = pow(ctx, right, 3);
            let left_sq_right = mul(ctx, &[three, left_sq, right]);
            let left_right_sq = mul(ctx, &[three, left, right_sq]);
            add_signed(
                ctx,
                &[
                    (left_cu, Sign::Pos),
                    (left_sq_right, Sign::Neg),
                    (left_right_sq, Sign::Pos),
                    (right_cu, Sign::Neg),
                ],
            )
        }
        _ => return None,
    })
}

pub(super) fn geometric_difference_factor_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, i64)> {
    let Expr::Sub(lhs, rhs) = ctx.get(before) else {
        return None;
    };
    if !is_small_positive_integer(ctx, *rhs, 1) {
        return None;
    }
    let Expr::Pow(base, exp) = ctx.get(*lhs) else {
        return None;
    };
    let power = small_positive_integer_value(ctx, *exp)?;
    if power < 2 {
        return None;
    }

    let Expr::Mul(left, right) = ctx.get(after) else {
        return None;
    };
    let series = if is_base_minus_one(ctx, *left, *base) {
        *right
    } else if is_base_minus_one(ctx, *right, *base) {
        *left
    } else {
        return None;
    };

    let terms = AddView::from_expr(ctx, series).terms;
    if terms.len() != power as usize {
        return None;
    }

    let mut seen = BTreeMap::new();
    for (term, sign) in terms {
        if sign != Sign::Pos {
            return None;
        }
        let exponent = geometric_series_term_exponent(ctx, *base, term)?;
        seen.insert(exponent, ());
    }

    if seen.len() != power as usize {
        return None;
    }
    if !(0..power).all(|exp| seen.contains_key(&exp)) {
        return None;
    }

    Some((*base, power))
}

fn full_sixth_power_minus_one_factor_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(lhs, rhs) = ctx.get(before) else {
        return None;
    };
    if !is_integer_literal(ctx, *rhs, 1) {
        return None;
    }
    let Expr::Pow(base, exponent) = ctx.get(*lhs) else {
        return None;
    };
    if !is_integer_literal(ctx, *exponent, 6) {
        return None;
    }

    let factors = expr_nary::mul_leaves(ctx, after);
    if factors.len() != 4 {
        return None;
    }

    let mut saw_plus_one = false;
    let mut saw_minus_one = false;
    let mut saw_positive_quadratic = false;
    let mut saw_negative_quadratic = false;
    for factor in factors {
        if !saw_plus_one && linear_unit_factor_matches(ctx, factor, *base, Sign::Pos) {
            saw_plus_one = true;
            continue;
        }
        if !saw_minus_one && linear_unit_factor_matches(ctx, factor, *base, Sign::Neg) {
            saw_minus_one = true;
            continue;
        }
        if !saw_positive_quadratic
            && geometric_quadratic_factor_matches(ctx, factor, *base, Sign::Pos)
        {
            saw_positive_quadratic = true;
            continue;
        }
        if !saw_negative_quadratic
            && geometric_quadratic_factor_matches(ctx, factor, *base, Sign::Neg)
        {
            saw_negative_quadratic = true;
            continue;
        }
        return None;
    }

    (saw_plus_one && saw_minus_one && saw_positive_quadratic && saw_negative_quadratic)
        .then_some(*base)
}

fn linear_unit_factor_matches(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    constant_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return false;
    }

    let has_base = terms.iter().any(|(term, sign)| {
        *sign == Sign::Pos && compare_expr(ctx, *term, base) == Ordering::Equal
    });
    let has_unit = terms
        .iter()
        .any(|(term, sign)| *sign == constant_sign && is_integer_literal(ctx, *term, 1));

    has_base && has_unit
}

fn geometric_quadratic_factor_matches(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    linear_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let has_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, base));
    let has_linear = terms.iter().any(|(term, sign)| {
        *sign == linear_sign && compare_expr(ctx, *term, base) == Ordering::Equal
    });
    let has_unit = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && is_integer_literal(ctx, *term, 1));

    has_square && has_linear && has_unit
}

pub(super) fn factor_polynomial_terms(
    ctx: &Context,
    factor: ExprId,
    var: &str,
) -> Option<Vec<PolyContribution>> {
    let poly = Polynomial::from_expr(ctx, factor, var).ok()?;
    let mut terms = Vec::new();
    for (degree, coeff) in poly.coeffs.iter().enumerate().rev() {
        if coeff.is_zero() {
            continue;
        }
        terms.push(PolyContribution {
            coeff: coeff.clone(),
            degree,
        });
    }
    Some(terms)
}

pub(super) fn expand_polynomial_term_products(
    factors: &[Vec<PolyContribution>],
) -> Vec<PolyContribution> {
    let mut acc = vec![PolyContribution {
        coeff: BigRational::from_integer(1.into()),
        degree: 0,
    }];

    for factor in factors {
        let mut next = Vec::new();
        for partial in &acc {
            for term in factor {
                next.push(PolyContribution {
                    coeff: partial.coeff.clone() * term.coeff.clone(),
                    degree: partial.degree + term.degree,
                });
            }
        }
        acc = next;
    }

    acc.retain(|term| !term.coeff.is_zero());
    acc
}

pub(super) fn generate_common_factor_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };

    let power_factor_substeps =
        generate_power_common_factor_cancel_substeps(ctx, before, after, *numerator, *denominator);
    if !power_factor_substeps.is_empty() {
        return power_factor_substeps;
    }

    let Some(common_factor) = first_common_factor(ctx, *numerator, *denominator) else {
        return Vec::new();
    };

    let factor_display = display_expr(ctx, common_factor);
    let before_display = display_expr(ctx, before);
    let before_latex = latex_expr(ctx, before);
    let final_display = display_expr(ctx, after);
    let final_latex = latex_expr(ctx, after);

    let Some((intermediate_display, intermediate_latex)) =
        quotient_after_cancel_once(ctx, *numerator, *denominator, common_factor)
    else {
        return Vec::new();
    };

    if cas_formatter::clean_display_string(&intermediate_display)
        == cas_formatter::clean_display_string(&final_display)
        && intermediate_latex == final_latex
    {
        return Vec::new();
    }

    let mut out = vec![SubStep::new(
        format!("Cancelar el factor común {}", factor_display),
        before_display,
        intermediate_display.clone(),
    )
    .with_before_latex(before_latex)
    .with_after_latex(intermediate_latex.clone())];

    if cas_formatter::clean_display_string(&intermediate_display)
        != cas_formatter::clean_display_string(&final_display)
        || intermediate_latex != final_latex
    {
        let finish_title =
            next_common_factor_after_cancel(ctx, *numerator, *denominator, common_factor)
                .map(|next_factor| {
                    format!(
                        "Cancelar también el factor común {}",
                        display_expr(ctx, next_factor)
                    )
                })
                .unwrap_or_else(|| "Reducir la fracción que queda".to_string());
        out.push(
            SubStep::new(finish_title, intermediate_display, final_display)
                .with_before_latex(intermediate_latex)
                .with_after_latex(final_latex),
        );
    }

    out
}

fn generate_power_common_factor_cancel_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    numerator: ExprId,
    denominator: ExprId,
) -> Vec<SubStep> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator);

    for denominator_factor in denominator_factors.iter().copied() {
        for (index, numerator_factor) in numerator_factors.iter().copied().enumerate() {
            let Expr::Pow(base, exponent) = ctx.get(numerator_factor) else {
                continue;
            };
            if *base != denominator_factor {
                continue;
            }
            let Some(power) = small_positive_integer_value(ctx, *exponent) else {
                continue;
            };
            if power <= 1 {
                continue;
            }

            let mut work = ctx.clone();
            let mut expanded_numerator_factors = Vec::new();
            for (factor_index, factor) in numerator_factors.iter().copied().enumerate() {
                if factor_index == index {
                    expanded_numerator_factors.push(denominator_factor);
                    expanded_numerator_factors.push(power_factor_after_peeling_once(
                        &mut work,
                        denominator_factor,
                        power - 1,
                    ));
                } else {
                    expanded_numerator_factors.push(factor);
                }
            }

            let expanded_numerator =
                build_mul_expr_from_factors(&mut work, expanded_numerator_factors.as_slice());
            let original_denominator =
                build_mul_expr_from_factors(&mut work, denominator_factors.as_slice());
            let intermediate = if is_one_expr(&work, original_denominator) {
                expanded_numerator
            } else {
                work.add(Expr::Div(expanded_numerator, original_denominator))
            };
            let (intermediate_display, intermediate_latex) = render_temp_expr(&work, intermediate);
            let final_display = display_expr(ctx, after);
            let final_latex = latex_expr(ctx, after);
            let factor_display = display_expr(ctx, denominator_factor);

            return vec![
                SubStep::new(
                    format!(
                        "Descomponer {} para exponer el factor común {}",
                        human_expr(ctx, numerator_factor),
                        human_expr(ctx, denominator_factor)
                    ),
                    display_expr(ctx, before),
                    intermediate_display.clone(),
                )
                .with_before_latex(latex_expr(ctx, before))
                .with_after_latex(intermediate_latex.clone()),
                SubStep::new(
                    format!("Cancelar el factor común {factor_display}"),
                    intermediate_display,
                    final_display,
                )
                .with_before_latex(intermediate_latex)
                .with_after_latex(final_latex),
            ];
        }
    }

    Vec::new()
}

fn power_factor_after_peeling_once(
    ctx: &mut Context,
    base: ExprId,
    remaining_power: i64,
) -> ExprId {
    if remaining_power == 1 {
        base
    } else {
        let exponent = ctx.num(remaining_power);
        ctx.add(Expr::Pow(base, exponent))
    }
}

pub(super) fn generate_difference_of_squares_cancel_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };

    if let Some((left_term, right_term)) = difference_of_squares_bases(ctx, *numerator) {
        if is_difference_of_terms(ctx, *denominator, left_term, right_term)
            && is_sum_of_terms(ctx, after, left_term, right_term)
        {
            let denominator_display = display_expr(ctx, *denominator);
            let remaining_display = display_expr(ctx, after);
            let denominator_latex = latex_expr(ctx, *denominator);
            let remaining_latex = latex_expr(ctx, after);
            let factorized_display = format!("({denominator_display}) · ({remaining_display})");
            let factorized_latex = format!(
                "\\left({denominator_latex}\\right)\\cdot \\left({remaining_latex}\\right)"
            );

            return vec![
                SubStep::new(
                    "Factorizar el numerador como diferencia de cuadrados",
                    display_expr(ctx, *numerator),
                    factorized_display.clone(),
                )
                .with_before_latex(latex_expr(ctx, *numerator))
                .with_after_latex(factorized_latex.clone()),
                SubStep::new(
                    format!("Ahora se cancela el factor {denominator_display}"),
                    format!("({factorized_display}) / ({denominator_display})"),
                    remaining_display,
                )
                .with_before_latex(format!(
                    "\\frac{{{factorized_latex}}}{{{denominator_latex}}}"
                ))
                .with_after_latex(remaining_latex),
            ];
        }
    }

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
            "Usar la diferencia de cuadrados: a^2 - b^2 = (a - b)(a + b)",
            format!(
                "{} - {}",
                squared_display(ctx, left_term),
                squared_display(ctx, right_term)
            ),
            display_expr(ctx, *numerator),
        )
        .with_before_latex(format!(
            "{} - {}",
            squared_latex(ctx, left_term),
            squared_latex(ctx, right_term)
        ))
        .with_after_latex(latex_expr(ctx, *numerator)),
        SubStep::new(
            format!(
                "Ahora se cancela el factor {}",
                display_expr(ctx, canceled_factor)
            ),
            // Group EACH factor: the old `format!("({} · {}) / ({})", …)`
            // interpolated compound factors ungrouped, so `(x - 1 · x + 1)`
            // re-parsed to `1` and the substep asserted `1/(x-1) = x+1`
            // (auditoría 2026-07-30, fichas D2-001/D3-001 — false in TEXT
            // and in the LaTeX the web renders). The canceled factor stays
            // FIRST on purpose: showing it adjacent to the denominator is
            // the didactic content that distinguishes this substep from the
            // step header (whose canonical order the prune compares against).
            format!(
                "(({}) · ({})) / ({})",
                display_expr(ctx, canceled_factor),
                display_expr(ctx, other_factor),
                display_expr(ctx, *denominator),
            ),
            display_expr(ctx, after),
        )
        .with_before_latex(format!(
            "\\frac{{\\left({}\\right)\\cdot \\left({}\\right)}}{{{}}}",
            latex_expr(ctx, canceled_factor),
            latex_expr(ctx, other_factor),
            latex_expr(ctx, *denominator),
        ))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_sophie_germain_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((a, b)) = sophie_germain_expansion_plan(ctx, after, before) else {
        return Vec::new();
    };

    let difference_display = sophie_germain_difference_of_squares_display(ctx, a, b);
    let difference_latex = sophie_germain_difference_of_squares_latex(ctx, a, b);
    let factorized_display = sophie_germain_factorized_identity_display(ctx, a, b);
    let factorized_latex = sophie_germain_factorized_identity_latex(ctx, a, b);

    vec![
        SubStep::new(
            "Convertir la suma en diferencia de cuadrados",
            display_expr(ctx, before),
            difference_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(difference_latex.clone()),
        SubStep::new(
            "Factorizar la diferencia de cuadrados",
            difference_display,
            factorized_display,
        )
        .with_before_latex(difference_latex)
        .with_after_latex(factorized_latex),
    ]
}

/// Returns `(u = linear polynomial, dv = elementary factor)` when exactly one of
/// the two product factors is a degree-1 polynomial in `var_name` and the other
/// is a non-polynomial, non-logarithm factor. The ln family is excluded so the
/// dedicated log narrator owns it (it assigns `u = ln`, not `u = polynomial`).
pub(super) fn linear_times_elementary_factors(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    oriented_linear_times_elementary(ctx, left, right, var_name)
        .or_else(|| oriented_linear_times_elementary(ctx, right, left, var_name))
}

/// Returns `(u = polynomial of degree >= 2, dv factor = elementary)` for the
/// REPEATED integration-by-parts family `p(x) * {exp, sin, cos, sinh, cosh}`.
/// Mirrors `linear_times_elementary_factors` but accepts degree >= 2 (the linear
/// narrator keeps degree 1). The elementary factor is any non-polynomial,
/// non-logarithm factor whose antiderivative is again elementary -- the
/// dispatcher has already gated the integrand to the exp/trig/hyperbolic linear
/// targets, so the per-level `integrate`/`differentiate` calls terminate after
/// exactly `degree` reductions.
pub(super) fn repeated_polynomial_times_elementary_factors(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let (left, right) = as_mul(ctx, integrand)?;
    oriented_repeated_polynomial_times_elementary(ctx, left, right, var_name)
        .or_else(|| oriented_repeated_polynomial_times_elementary(ctx, right, left, var_name))
}

pub(super) fn push_integration_constant_factor_adjustment_substep(
    substeps: &mut Vec<SubStep>,
    adjustment: IntegrationConstantFactorAdjustment<'_>,
) {
    if let (Some(scale_display), Some(scale_latex)) = (
        adjustment.symbolic_scale_display,
        adjustment.symbolic_scale_latex,
    ) {
        substeps.push(
            SubStep::keyed(
                "usub.adjust_constant_factor",
                vec![],
                adjustment.cofactor_display,
                format!("{} · {}", scale_display, adjustment.derivative_display),
            )
            .with_before_latex(adjustment.cofactor_latex)
            .with_after_latex(format!(
                "{}\\cdot {}",
                scale_latex, adjustment.derivative_latex
            )),
        );
    } else if !adjustment.scale.is_one() {
        substeps.push(
            SubStep::keyed(
                "usub.adjust_constant_factor",
                vec![],
                adjustment.cofactor_display,
                format!(
                    "{} · {}",
                    rational_display(adjustment.scale),
                    adjustment.derivative_display
                ),
            )
            .with_before_latex(adjustment.cofactor_latex)
            .with_after_latex(format!(
                "{}\\cdot {}",
                rational_latex(adjustment.scale),
                adjustment.derivative_latex
            )),
        );
    }
}

pub(super) fn polynomial_base_pow_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let (base, exponent) = as_pow(ctx, expr)?;
    let exponent = as_rational_const(ctx, exponent, 8)?;
    Some((base, exponent))
}

pub(super) fn polynomial_base_power_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    if let Some(base) = polynomial_base_sqrt_arg(ctx, expr) {
        return Some((base, BigRational::new(1.into(), 2.into())));
    }
    polynomial_base_pow_factor(ctx, expr)
}

pub(super) fn polynomial_base_match_from_cofactor(
    ctx: &Context,
    title: &'static str,
    negative: bool,
    cofactor_factors: &[ExprId],
    base: ExprId,
    var_name: &str,
) -> Option<PolynomialBaseTableMatch> {
    let base_poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
    if base_poly.degree() == 0 {
        return None;
    }
    let derivative_poly = base_poly.derivative();
    if derivative_poly.is_zero() {
        return None;
    }

    let mut scratch = ctx.clone();
    let mut signed_cofactor_factors = Vec::new();
    if negative {
        signed_cofactor_factors
            .push(scratch.add(Expr::Number(BigRational::from_integer((-1).into()))));
    }
    signed_cofactor_factors.extend_from_slice(cofactor_factors);
    let cofactor_expr = build_quotient_from_factors(&mut scratch, &signed_cofactor_factors, &[]);
    let cofactor_simplified = simplify_expr_in_context(&mut scratch, cofactor_expr);
    let cofactor_poly = Polynomial::from_expr(&scratch, cofactor_simplified, var_name).ok()?;
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative_poly)?;
    if scale.is_zero() {
        return None;
    }

    let (derivative_display, derivative_latex) =
        polynomial_display_and_latex(ctx, &derivative_poly);
    Some(PolynomialBaseTableMatch {
        title,
        base,
        cofactor_display: display_expr(&scratch, cofactor_simplified),
        cofactor_latex: latex_expr(&scratch, cofactor_simplified),
        derivative_display,
        derivative_latex,
        scale,
    })
}

pub(super) fn polynomial_product_from_factors_trace(
    scratch: &mut Context,
    factors: &[ExprId],
    var_name: &str,
) -> Option<Polynomial> {
    if factors.is_empty() {
        return Some(Polynomial::one(var_name.to_string()));
    }
    let product = build_mul_expr_from_factors(scratch, factors);
    Polynomial::from_expr(scratch, product, var_name).ok()
}

pub(super) fn signed_mul_factors_into(
    ctx: &Context,
    expr: ExprId,
    negative: &mut bool,
    factors: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            signed_mul_factors_into(ctx, *left, negative, factors);
            signed_mul_factors_into(ctx, *right, negative, factors);
        }
        Expr::Neg(inner) => {
            *negative = !*negative;
            signed_mul_factors_into(ctx, *inner, negative, factors);
        }
        _ => factors.push(expr),
    }
}

pub(super) fn first_common_factor(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<ExprId> {
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

fn next_common_factor_after_cancel(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
    common_factor: ExprId,
) -> Option<ExprId> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator);

    let numerator_remaining = remove_first_factor(&numerator_factors, common_factor)?;
    let denominator_remaining = remove_first_factor(&denominator_factors, common_factor)?;

    numerator_remaining
        .into_iter()
        .find(|numerator_factor| denominator_remaining.contains(numerator_factor))
}

pub(super) fn remove_first_factor(factors: &[ExprId], target: ExprId) -> Option<Vec<ExprId>> {
    let index = factors.iter().position(|factor| *factor == target)?;
    let mut remaining = factors.to_vec();
    remaining.remove(index);
    Some(remaining)
}

pub(super) fn is_repeated_factor_product(ctx: &Context, expr: ExprId, factor: ExprId) -> bool {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return false;
    };
    *left == factor && *right == factor
}

pub(super) fn common_factor_factorization_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, Sign)> {
    let before_terms = AddView::from_expr(ctx, before);
    if before_terms.terms.len() < 2 {
        return None;
    }

    let (factor, grouped) = split_factorized_product(ctx, after)?;
    let grouped_terms = AddView::from_expr(ctx, grouped);
    if grouped_terms.terms.len() != before_terms.terms.len() {
        return None;
    }

    let mut stripped = before_terms
        .terms
        .iter()
        .map(|&(term, sign)| Some((strip_factor_from_term(ctx, term, factor)?, sign)))
        .collect::<Option<Vec<_>>>()?;
    let mut grouped = grouped_terms.terms.to_vec();

    sort_signed_terms_for_compare(ctx, &mut stripped);
    sort_signed_terms_for_compare(ctx, &mut grouped);

    if stripped.iter().zip(grouped.iter()).all(
        |((left_expr, left_sign), (right_expr, right_sign))| {
            left_sign == right_sign && same_presentational_expr(ctx, *left_expr, ctx, *right_expr)
        },
    ) {
        let kind = if grouped_terms
            .terms
            .iter()
            .any(|(_, sign)| *sign == Sign::Neg)
        {
            Sign::Neg
        } else {
            Sign::Pos
        };
        return Some((factor, kind));
    }

    None
}

fn split_factorized_product(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    if matches!(ctx.get(*left), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return Some((*right, *left));
    }
    if matches!(ctx.get(*right), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return Some((*left, *right));
    }
    None
}

fn strip_factor_from_term(ctx: &Context, term: ExprId, factor: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(term) else {
        return None;
    };
    if *left == factor {
        return Some(*right);
    }
    if *right == factor {
        return Some(*left);
    }
    None
}

pub(super) fn cube_factorized_identity_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<CubeIdentityPlan> {
    let (left_term, right_term, kind) = cube_identity_terms(ctx, after)?;
    let left_base = cube_base_from_term_with_witness(ctx, left_term, before)?;
    let right_base = cube_base_from_term_with_witness(ctx, right_term, before)?;
    let Expr::Mul(first_factor, second_factor) = ctx.get(before) else {
        return None;
    };

    for (linear_factor, quadratic_factor) in [
        (*first_factor, *second_factor),
        (*second_factor, *first_factor),
    ] {
        if linear_factor_matches(ctx, linear_factor, left_base, right_base, kind)
            && quadratic_factor_matches(ctx, quadratic_factor, left_base, right_base, kind)
        {
            return Some(CubeIdentityPlan {
                left_base,
                right_base,
                kind,
            });
        }
    }

    None
}

pub(super) fn sixth_power_factorized_identity_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SixthPowerIdentityPlan> {
    let (left_term, right_term, kind) = sixth_power_identity_terms(ctx, after)?;
    let left_base = sixth_power_base_from_term(ctx, left_term)?;
    let right_base = sixth_power_base_from_term(ctx, right_term)?;
    if is_one(ctx, left_base) || is_one(ctx, right_base) {
        return None;
    }

    let Expr::Mul(first_factor, second_factor) = ctx.get(before) else {
        return None;
    };

    for (binomial_factor, quartic_factor) in [
        (*first_factor, *second_factor),
        (*second_factor, *first_factor),
    ] {
        if sixth_power_binomial_factor_matches(ctx, binomial_factor, left_base, right_base, kind)
            && sixth_power_quartic_factor_matches(ctx, quartic_factor, left_base, right_base, kind)
        {
            return Some(SixthPowerIdentityPlan {
                left_base,
                right_base,
                kind,
            });
        }
    }

    None
}

pub(super) fn linear_factor_matches(
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
                (*left == left_base && is_negated_version_of(ctx, *right, right_base))
                    || (*right == left_base && is_negated_version_of(ctx, *left, right_base))
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

fn sixth_power_binomial_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return false;
    }

    let has_left_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, left_base));
    let right_sign = match kind {
        SixthPowerIdentityKind::Sum => Sign::Pos,
        SixthPowerIdentityKind::Difference => Sign::Neg,
    };
    let has_right_square = terms
        .iter()
        .any(|(term, sign)| *sign == right_sign && matches_square_of(ctx, *term, right_base));

    has_left_square && has_right_square
}

fn sixth_power_quartic_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let has_left_fourth = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_fourth_power_of(ctx, *term, left_base));
    let has_right_fourth = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_fourth_power_of(ctx, *term, right_base));
    let mixed_sign = match kind {
        SixthPowerIdentityKind::Sum => Sign::Neg,
        SixthPowerIdentityKind::Difference => Sign::Pos,
    };
    let has_mixed = terms.iter().any(|(term, sign)| {
        *sign == mixed_sign && matches_product_of_squares(ctx, *term, left_base, right_base)
    });

    has_left_fourth && has_right_fourth && has_mixed
}

pub(super) fn cube_linear_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left = display_expr(ctx, left_base);
    let right = display_expr(ctx, right_base);
    match kind {
        CubeIdentityKind::Sum => format!("({left} + {right})"),
        CubeIdentityKind::Difference => format!("({left} - {right})"),
    }
}

pub(super) fn cube_linear_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left = latex_expr(ctx, left_base);
    let right = latex_expr(ctx, right_base);
    match kind {
        CubeIdentityKind::Sum => format!("\\left({left} + {right}\\right)"),
        CubeIdentityKind::Difference => format!("\\left({left} - {right}\\right)"),
    }
}

fn cube_quadratic_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left_sq = squared_display(ctx, left_base);
    let right_sq = squared_display(ctx, right_base);
    let mixed = format!(
        "{}·{}",
        display_expr(ctx, left_base),
        display_expr(ctx, right_base)
    );
    match kind {
        CubeIdentityKind::Sum => format!("({left_sq} - {mixed} + {right_sq})"),
        CubeIdentityKind::Difference => format!("({left_sq} + {mixed} + {right_sq})"),
    }
}

fn cube_quadratic_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left_sq = format!("{{{}}}^{{2}}", latex_expr(ctx, left_base));
    let right_sq = format!("{{{}}}^{{2}}", latex_expr(ctx, right_base));
    let mixed = format!(
        "{}\\cdot {}",
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base)
    );
    match kind {
        CubeIdentityKind::Sum => {
            format!("\\left({left_sq} - {mixed} + {right_sq}\\right)")
        }
        CubeIdentityKind::Difference => {
            format!("\\left({left_sq} + {mixed} + {right_sq}\\right)")
        }
    }
}

pub(super) fn cube_factorized_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    format!(
        "{}·{}",
        cube_linear_factor_display(ctx, left_base, right_base, kind),
        cube_quadratic_factor_display(ctx, left_base, right_base, kind)
    )
}

pub(super) fn cube_factorized_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    format!(
        "{}\\cdot {}",
        cube_linear_factor_latex(ctx, left_base, right_base, kind),
        cube_quadratic_factor_latex(ctx, left_base, right_base, kind)
    )
}

fn sixth_power_binomial_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left = squared_display(ctx, left_base);
    let right = squared_display(ctx, right_base);
    match kind {
        SixthPowerIdentityKind::Sum => format!("({left} + {right})"),
        SixthPowerIdentityKind::Difference => format!("({left} - {right})"),
    }
}

fn sixth_power_binomial_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left = squared_latex(ctx, left_base);
    let right = squared_latex(ctx, right_base);
    match kind {
        SixthPowerIdentityKind::Sum => format!("\\left({left} + {right}\\right)"),
        SixthPowerIdentityKind::Difference => format!("\\left({left} - {right}\\right)"),
    }
}

fn sixth_power_quartic_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left_fourth = fourth_power_display(ctx, left_base);
    let right_fourth = fourth_power_display(ctx, right_base);
    let mixed = format!(
        "{}·{}",
        squared_display(ctx, left_base),
        squared_display(ctx, right_base)
    );
    match kind {
        SixthPowerIdentityKind::Sum => format!("({left_fourth} - {mixed} + {right_fourth})"),
        SixthPowerIdentityKind::Difference => {
            format!("({left_fourth} + {mixed} + {right_fourth})")
        }
    }
}

fn sixth_power_quartic_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left_fourth = fourth_power_latex(ctx, left_base);
    let right_fourth = fourth_power_latex(ctx, right_base);
    let mixed = format!(
        "{}\\cdot {}",
        squared_latex(ctx, left_base),
        squared_latex(ctx, right_base)
    );
    match kind {
        SixthPowerIdentityKind::Sum => {
            format!("\\left({left_fourth} - {mixed} + {right_fourth}\\right)")
        }
        SixthPowerIdentityKind::Difference => {
            format!("\\left({left_fourth} + {mixed} + {right_fourth}\\right)")
        }
    }
}

pub(super) fn sixth_power_factorized_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    format!(
        "{}·{}",
        sixth_power_binomial_factor_display(ctx, left_base, right_base, kind),
        sixth_power_quartic_factor_display(ctx, left_base, right_base, kind)
    )
}

pub(super) fn sixth_power_factorized_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    format!(
        "{}\\cdot {}",
        sixth_power_binomial_factor_latex(ctx, left_base, right_base, kind),
        sixth_power_quartic_factor_latex(ctx, left_base, right_base, kind)
    )
}

fn sophie_germain_difference_of_squares_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "({} + 2 · {})^2 - (2 · {} · {})^2",
        squared_display(ctx, left_base),
        squared_display(ctx, right_base),
        display_expr(ctx, left_base),
        display_expr(ctx, right_base)
    )
}

fn sophie_germain_difference_of_squares_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "\\left({} + 2\\cdot {}\\right)^{{2}} - \\left(2\\cdot {}\\cdot {}\\right)^{{2}}",
        squared_latex(ctx, left_base),
        squared_latex(ctx, right_base),
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base)
    )
}

fn sophie_germain_minus_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "({} - 2 · {} · {} + 2 · {})",
        squared_display(ctx, left_base),
        display_expr(ctx, left_base),
        display_expr(ctx, right_base),
        squared_display(ctx, right_base)
    )
}

fn sophie_germain_minus_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "\\left({} - 2\\cdot {}\\cdot {} + 2\\cdot {}\\right)",
        squared_latex(ctx, left_base),
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base),
        squared_latex(ctx, right_base)
    )
}

fn sophie_germain_plus_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "({} + 2 · {} · {} + 2 · {})",
        squared_display(ctx, left_base),
        display_expr(ctx, left_base),
        display_expr(ctx, right_base),
        squared_display(ctx, right_base)
    )
}

fn sophie_germain_plus_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "\\left({} + 2\\cdot {}\\cdot {} + 2\\cdot {}\\right)",
        squared_latex(ctx, left_base),
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base),
        squared_latex(ctx, right_base)
    )
}

pub(super) fn sophie_germain_factorized_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "{} · {}",
        sophie_germain_minus_factor_display(ctx, left_base, right_base),
        sophie_germain_plus_factor_display(ctx, left_base, right_base)
    )
}

pub(super) fn sophie_germain_factorized_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "{}\\cdot {}",
        sophie_germain_minus_factor_latex(ctx, left_base, right_base),
        sophie_germain_plus_factor_latex(ctx, left_base, right_base)
    )
}

fn is_difference_of_terms(ctx: &Context, expr: ExprId, left: ExprId, right: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Sub(diff_left, diff_right) if *diff_left == left && *diff_right == right)
}
