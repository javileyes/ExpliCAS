//! `focused_rule_substeps`: familia `logs_exp`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn generate_expand_log_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = expanded_log_exp_cancellation_substep(ctx, after, step.after) {
        return vec![substep];
    }

    let (title, before_display, after_display, before_latex, after_latex) =
        if let Some(snippet) = log_formula_snippet(ctx, before, true) {
            snippet
        } else {
            (
                "Usar que el logaritmo de un producto se separa en una suma".to_string(),
                "ln(u · v)".to_string(),
                "ln(u) + ln(v)".to_string(),
                "\\ln(u\\cdot v)".to_string(),
                "\\ln(u) + \\ln(v)".to_string(),
            )
        };

    if before != after {
        return vec![concrete_expr_substep(ctx, title, before, after)];
    }

    vec![formula_substep(
        title,
        &before_display,
        &after_display,
        &before_latex,
        &after_latex,
    )]
}

fn expanded_log_exp_cancellation_substep(
    ctx: &Context,
    expanded_logs: ExprId,
    final_expr: ExprId,
) -> Option<SubStep> {
    if expanded_logs == final_expr {
        return None;
    }

    let terms = AddView::from_expr(ctx, expanded_logs).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut temp_ctx = ctx.clone();
    let mut cancelled_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let log_arg = change_of_base_natural_log_argument(ctx, term)?;
        let exp_arg = extract_exp_argument(ctx, log_arg)?;
        cancelled_terms.push((exp_arg, sign));
    }

    let cancelled = build_add_from_signed_terms(&mut temp_ctx, &cancelled_terms);
    if compare_expr(&temp_ctx, cancelled, final_expr) != Ordering::Equal
        && !same_presentational_expr(&temp_ctx, cancelled, &temp_ctx, final_expr)
    {
        return None;
    }

    Some(concrete_expr_substep(
        ctx,
        "Cancelar cada logaritmo natural con su exponencial",
        expanded_logs,
        final_expr,
    ))
}

pub(super) fn generate_log_cancellation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let visible_rule = super::super::visible_rule_names::visible_rule_name_for_step(
        step.rule_name.as_str(),
        step.description.as_str(),
    );
    if visible_rule.as_ref() != "Expandir logaritmos y cancelar términos iguales" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if !is_zero(ctx, after) {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let Some((expansion_focus_before, expansion_focus_after, expanded_expr)) =
        build_log_cancellation_expansion_plan(&mut work, before)
    else {
        return Vec::new();
    };

    let mut substeps = vec![SubStep::new(
        "Expandir el logaritmo del producto o del cociente",
        human_expr(&work, expansion_focus_before),
        human_expr(&work, expansion_focus_after),
    )
    .with_before_latex(latex_expr(&work, expansion_focus_before))
    .with_after_latex(latex_expr(&work, expansion_focus_after))];

    let cancel_before = if let Some((extract_before, extract_after, extracted_expr)) =
        build_log_cancellation_exponent_plan(&mut work, expanded_expr)
    {
        substeps.push(
            SubStep::new(
                "Sacar exponentes fuera del logaritmo cuando sea necesario",
                human_expr(&work, extract_before),
                human_expr(&work, extract_after),
            )
            .with_before_latex(latex_expr(&work, extract_before))
            .with_after_latex(latex_expr(&work, extract_after)),
        );
        extracted_expr
    } else {
        expanded_expr
    };

    substeps.push(
        SubStep::new(
            "Cancelar términos iguales",
            human_expr(&work, cancel_before),
            human_expr(&work, after),
        )
        .with_before_latex(latex_expr(&work, cancel_before))
        .with_after_latex(latex_expr(&work, after)),
    );
    substeps
}

pub(super) fn generate_exponential_log_cancellation_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Exponential Sum/Difference Identity" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(arg) = extract_exp_argument(ctx, before) else {
        return Vec::new();
    };
    let terms = AddView::from_expr(ctx, arg).terms.to_vec();
    if terms.len() < 2
        || !terms
            .iter()
            .any(|(term, _sign)| expression_contains_log(ctx, *term))
    {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let Some(split_expr) = build_exp_sum_product_without_log_cancellation(&mut work, &terms) else {
        return Vec::new();
    };
    if compare_expr(&work, split_expr, after) == Ordering::Equal {
        return Vec::new();
    }

    let (before_plain, before_latex) = render_temp_expr(&work, before);
    let (split_plain, split_latex) = render_temp_expr(&work, split_expr);
    let (after_plain, after_latex) = render_temp_expr(&work, after);

    vec![
        formula_substep(
            "Separar la suma o resta del exponente en productos de exponenciales",
            &before_plain,
            &split_plain,
            &before_latex,
            &split_latex,
        ),
        formula_substep(
            "Cancelar e^(k·ln(u)) como potencia en cada factor",
            &split_plain,
            &after_plain,
            &split_latex,
            &after_latex,
        ),
    ]
}

fn build_exp_sum_product_without_log_cancellation(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> Option<ExprId> {
    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();

    for (term, sign) in terms {
        let exp_term = ctx.call_builtin(BuiltinFn::Exp, vec![*term]);
        match sign {
            Sign::Pos => numerator_factors.push(exp_term),
            Sign::Neg => denominator_factors.push(exp_term),
        }
    }

    if numerator_factors.len() + denominator_factors.len() < 2 {
        return None;
    }

    let numerator = expr_nary::build_balanced_mul(ctx, &numerator_factors);
    if denominator_factors.is_empty() {
        return Some(numerator);
    }

    let denominator = expr_nary::build_balanced_mul(ctx, &denominator_factors);
    if numerator_factors.is_empty() {
        let one = ctx.num(1);
        return Some(ctx.add(Expr::Div(one, denominator)));
    }

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn expression_contains_log(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, _)
            if {
                matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Ln | BuiltinFn::Log | BuiltinFn::Log10)
                )
            } =>
        {
            true
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expression_contains_log(ctx, *left) || expression_contains_log(ctx, *right)
        }
        Expr::Neg(inner) => expression_contains_log(ctx, *inner),
        _ => false,
    }
}

pub(super) fn generate_exponential_sum_diff_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = exponential_sum_diff_identity_substep(ctx, after, false) {
        return vec![substep];
    }

    if let Some(substep) = exponential_sum_diff_identity_substep(ctx, before, true) {
        return vec![substep];
    }

    Vec::new()
}

fn exponential_sum_diff_identity_substep(
    ctx: &Context,
    exp_expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let exp_arg = extract_exp_argument(ctx, exp_expr)?;
    let (contract_title, expand_title, product_formula, exp_formula, product_latex, exp_latex) =
        match ctx.get(exp_arg) {
            Expr::Add(_, _) => (
                "Usar e^A · e^B = e^(A+B)",
                "Usar e^(A+B) = e^A · e^B",
                "e^A · e^B",
                "e^(A+B)",
                "{e}^{A}\\cdot {e}^{B}",
                "{e}^{A+B}",
            ),
            Expr::Sub(_, _) => (
                "Usar e^A / e^B = e^(A-B)",
                "Usar e^(A-B) = e^A / e^B",
                "e^A / e^B",
                "e^(A-B)",
                "\\frac{{e}^{A}}{{e}^{B}}",
                "{e}^{A-B}",
            ),
            _ => return None,
        };

    if reverse {
        Some(schema_substep(
            expand_title,
            exp_formula,
            product_formula,
            exp_latex,
            product_latex,
        ))
    } else {
        Some(schema_substep(
            contract_title,
            product_formula,
            exp_formula,
            product_latex,
            exp_latex,
        ))
    }
}

pub(super) fn generate_exponential_reciprocal_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = exponential_reciprocal_identity_substep(ctx, after, false) {
        return vec![substep];
    }

    if let Some(substep) = exponential_reciprocal_identity_substep(ctx, before, true) {
        return vec![substep];
    }

    Vec::new()
}

fn exponential_reciprocal_identity_substep(
    ctx: &Context,
    exp_expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let exp_arg = extract_exp_argument(ctx, exp_expr)?;
    if !matches!(ctx.get(exp_arg), Expr::Neg(_)) {
        return None;
    }

    if reverse {
        Some(schema_substep(
            "Usar e^(-A) = 1/e^A",
            "e^(-A)",
            "1/e^A",
            "{e}^{-A}",
            "\\frac{1}{{e}^{A}}",
        ))
    } else {
        Some(schema_substep(
            "Usar 1/e^A = e^(-A)",
            "1/e^A",
            "e^(-A)",
            "\\frac{1}{{e}^{A}}",
            "{e}^{-A}",
        ))
    }
}

pub(super) fn generate_exponential_power_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = exponential_power_identity_substep(ctx, after, false) {
        return vec![substep];
    }

    if let Some(substep) = exponential_power_identity_substep(ctx, before, true) {
        return vec![substep];
    }

    Vec::new()
}

fn exponential_power_identity_substep(
    ctx: &Context,
    exp_expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let exp_arg = extract_exp_argument(ctx, exp_expr)?;
    if !matches!(ctx.get(exp_arg), Expr::Mul(_, _)) {
        return None;
    }

    if reverse {
        Some(schema_substep(
            "Usar e^(n·A) = (e^A)^n",
            "e^(n·A)",
            "(e^A)^n",
            "{e}^{n\\cdot A}",
            "({e}^{A})^{n}",
        ))
    } else {
        Some(schema_substep(
            "Usar (e^A)^n = e^(n·A)",
            "(e^A)^n",
            "e^(n·A)",
            "({e}^{A})^{n}",
            "{e}^{n\\cdot A}",
        ))
    }
}

pub(super) fn generate_exponential_log_power_inverse_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = exponential_log_power_inverse_plan(ctx, before) else {
        return Vec::new();
    };

    let mut temp_ctx = ctx.clone();
    let core_before = temp_ctx.add(Expr::Pow(plan.source_base, plan.log_expr));
    let outer_exponent = build_balanced_mul(&mut temp_ctx, &plan.outer_factors);
    let intermediate = temp_ctx.add(Expr::Pow(core_before, outer_exponent));
    let expected_after = temp_ctx.add(Expr::Pow(plan.log_arg, outer_exponent));
    if compare_expr(&temp_ctx, expected_after, after) != Ordering::Equal {
        return Vec::new();
    }

    let (core_before_plain, core_before_latex) = render_temp_expr(&temp_ctx, core_before);
    let (log_arg_plain, log_arg_latex) = render_temp_expr(&temp_ctx, plan.log_arg);
    let (intermediate_plain, intermediate_latex) = render_temp_expr(&temp_ctx, intermediate);
    let (after_plain, after_latex) = render_temp_expr(&temp_ctx, after);

    let identity_title = match plan.base_kind {
        LogInversePowerBaseKind::Natural => "Usar que e^(ln(u)) = u",
        LogInversePowerBaseKind::Decimal => "Usar que 10^(log10(u)) = u",
        LogInversePowerBaseKind::Explicit => "Usar que a^(log(a, u)) = u",
    };

    vec![
        formula_substep(
            identity_title,
            &core_before_plain,
            &log_arg_plain,
            &core_before_latex,
            &log_arg_latex,
        ),
        formula_substep(
            "Aplicar el factor exterior como exponente",
            &intermediate_plain,
            &after_plain,
            &intermediate_latex,
            &after_latex,
        ),
    ]
}

pub(super) fn generate_log_inverse_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = log_inverse_power_plan(ctx, before) else {
        return Vec::new();
    };
    let Some((_source_base, exponent)) = as_pow(ctx, before) else {
        return Vec::new();
    };

    let mut temp_ctx = ctx.clone();
    let target_base = match plan.base_kind {
        LogInversePowerBaseKind::Natural => temp_ctx.add(Expr::Constant(Constant::E)),
        LogInversePowerBaseKind::Decimal => temp_ctx.num(10),
        LogInversePowerBaseKind::Explicit => {
            let Some(explicit_target_base) = plan.explicit_target_base else {
                return Vec::new();
            };
            explicit_target_base
        }
    };
    let recovery = temp_ctx.add(Expr::Pow(target_base, plan.log_expr));
    let intermediate = temp_ctx.add(Expr::Pow(recovery, exponent));

    let (source_plain, source_latex) = render_temp_expr(&temp_ctx, plan.source_base);
    let (recovery_plain, recovery_latex) = render_temp_expr(&temp_ctx, recovery);
    let (intermediate_plain, intermediate_latex) = render_temp_expr(&temp_ctx, intermediate);
    let (after_plain, after_latex) = render_temp_expr(&temp_ctx, after);

    let identity_title = match plan.base_kind {
        LogInversePowerBaseKind::Natural => "Usar que e^(ln(u)) = u",
        LogInversePowerBaseKind::Decimal => "Usar que 10^(log10(u)) = u",
        LogInversePowerBaseKind::Explicit => "Usar que a^(log(a, u)) = u",
    };
    let cancel_title = match plan.base_kind {
        LogInversePowerBaseKind::Natural => {
            "El exponente exterior cancela el ln del exponente interior"
        }
        LogInversePowerBaseKind::Decimal => {
            "El exponente exterior cancela el log10 del exponente interior"
        }
        LogInversePowerBaseKind::Explicit => {
            "El exponente exterior cancela el logaritmo del exponente interior"
        }
    };

    vec![
        formula_substep(
            identity_title,
            &source_plain,
            &recovery_plain,
            &source_latex,
            &recovery_latex,
        ),
        formula_substep(
            cancel_title,
            &intermediate_plain,
            &after_plain,
            &intermediate_latex,
            &after_latex,
        ),
    ]
}

fn exponential_log_power_inverse_plan(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExponentialLogPowerInversePlan> {
    let (source_base, exponent) = as_pow(ctx, expr)?;
    let factors: Vec<ExprId> = MulView::from_expr(ctx, exponent)
        .factors
        .into_iter()
        .collect();
    let mut matched_index = None;
    let mut matched_log_expr = None;
    let mut matched_log_arg = None;
    let mut matched_base_kind = None;

    for (index, factor) in factors.iter().enumerate() {
        let Some((log_base_opt, log_arg)) = extract_log_base_argument_view(ctx, *factor) else {
            continue;
        };
        let Some(base_kind) = exponential_log_inverse_base_kind(ctx, source_base, log_base_opt)
        else {
            continue;
        };
        if matched_index.is_some() {
            return None;
        }
        matched_index = Some(index);
        matched_log_expr = Some(*factor);
        matched_log_arg = Some(log_arg);
        matched_base_kind = Some(base_kind);
    }

    let matched_index = matched_index?;
    let outer_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| (index != matched_index).then_some(*factor))
        .collect();
    if outer_factors.is_empty() {
        return None;
    }

    Some(ExponentialLogPowerInversePlan {
        source_base,
        log_expr: matched_log_expr?,
        log_arg: matched_log_arg?,
        outer_factors,
        base_kind: matched_base_kind?,
    })
}

fn exponential_log_inverse_base_kind(
    ctx: &Context,
    source_base: ExprId,
    log_base_opt: Option<ExprId>,
) -> Option<LogInversePowerBaseKind> {
    match log_base_opt {
        None if matches!(ctx.get(source_base), Expr::Constant(Constant::E)) => {
            Some(LogInversePowerBaseKind::Natural)
        }
        Some(base) if base == log10_base_sentinel() && is_integer_literal(ctx, source_base, 10) => {
            Some(LogInversePowerBaseKind::Decimal)
        }
        Some(base) if compare_expr(ctx, base, source_base) == Ordering::Equal => {
            Some(LogInversePowerBaseKind::Explicit)
        }
        _ => None,
    }
}

fn log_inverse_power_plan(ctx: &Context, expr: ExprId) -> Option<LogInversePowerPlan> {
    let (source_base, exponent) = as_pow(ctx, expr)?;

    let check_log_denom = |ctx: &Context,
                           denom: ExprId|
     -> Option<(ExprId, Option<ExprId>, LogInversePowerBaseKind)> {
        let (log_base_opt, log_arg) = extract_log_base_argument_view(ctx, denom)?;
        if compare_expr(ctx, log_arg, source_base) != Ordering::Equal {
            return None;
        }

        let (explicit_target_base, base_kind) = match log_base_opt {
            Some(base) if base == log10_base_sentinel() => (None, LogInversePowerBaseKind::Decimal),
            Some(base) => (Some(base), LogInversePowerBaseKind::Explicit),
            None => (None, LogInversePowerBaseKind::Natural),
        };
        Some((denom, explicit_target_base, base_kind))
    };

    if let Some((_coeff, denom)) = as_div(ctx, exponent) {
        let (log_expr, explicit_target_base, base_kind) = check_log_denom(ctx, denom)?;
        return Some(LogInversePowerPlan {
            source_base,
            log_expr,
            explicit_target_base,
            base_kind,
        });
    }

    if let Some((lhs, rhs)) = as_mul(ctx, exponent) {
        for maybe_inverse in [rhs, lhs] {
            let Some((den, den_exp)) = as_pow(ctx, maybe_inverse) else {
                continue;
            };
            if !is_integer_literal(ctx, den_exp, -1) {
                continue;
            }
            let (log_expr, explicit_target_base, base_kind) = check_log_denom(ctx, den)?;
            return Some(LogInversePowerPlan {
                source_base,
                log_expr,
                explicit_target_base,
                base_kind,
            });
        }
    }

    if let Some((den, den_exp)) = as_pow(ctx, exponent) {
        if is_integer_literal(ctx, den_exp, -1) {
            let (log_expr, explicit_target_base, base_kind) = check_log_denom(ctx, den)?;
            return Some(LogInversePowerPlan {
                source_base,
                log_expr,
                explicit_target_base,
                base_kind,
            });
        }
    }

    None
}

fn build_log_cancellation_expansion_plan(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let mut rebuilt_terms = Vec::new();
    let mut focus_before = None;
    let mut focus_after = None;
    let terms = AddView::from_expr(ctx, expr).terms.to_vec();

    for (term, sign) in terms {
        if focus_before.is_none() {
            if let Some((term_before, term_after, expanded_terms)) =
                expand_log_term_into_signed_terms(ctx, term)
            {
                focus_before = Some(term_before);
                focus_after = Some(term_after);
                for (expanded_term, inner_sign) in expanded_terms {
                    rebuilt_terms.push((
                        expanded_term,
                        if sign == Sign::Pos {
                            inner_sign
                        } else {
                            inner_sign.negate()
                        },
                    ));
                }
                continue;
            }
        }

        rebuilt_terms.push((term, sign));
    }

    Some((
        focus_before?,
        focus_after?,
        build_add_from_signed_terms(ctx, &rebuilt_terms),
    ))
}

fn build_log_cancellation_exponent_plan(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let mut focus_before_terms = Vec::new();
    let mut focus_after_terms = Vec::new();
    let mut rebuilt_terms = Vec::new();
    let terms = AddView::from_expr(ctx, expr).terms.to_vec();

    for (term, sign) in terms {
        if let Some(rewritten) = rewrite_log_power_term_concretely(ctx, term) {
            focus_before_terms.push((term, sign));
            focus_after_terms.push((rewritten, sign));
            rebuilt_terms.push((rewritten, sign));
        } else {
            rebuilt_terms.push((term, sign));
        }
    }

    if focus_before_terms.is_empty() {
        return None;
    }

    Some((
        build_add_from_signed_terms(ctx, &focus_before_terms),
        build_add_from_signed_terms(ctx, &focus_after_terms),
        build_add_from_signed_terms(ctx, &rebuilt_terms),
    ))
}

fn expand_log_term_into_signed_terms(
    ctx: &mut Context,
    term: ExprId,
) -> Option<ConcreteLogExpansion> {
    let (coeff, log_expr) = scaled_log_term(ctx, term).unwrap_or_else(|| (1.into(), term));
    let family = extract_log_didactic_family(ctx, log_expr)?;
    let arg = match ctx.get(log_expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            args[0]
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Log) => {
            match args.as_slice() {
                [arg] => *arg,
                [_base, arg] => *arg,
                _ => return None,
            }
        }
        _ => return None,
    };
    let (inner_arg, wrap_abs) = abs_argument(ctx, arg)
        .map(|inner| (inner, true))
        .unwrap_or((arg, false));

    let signed_args = match ctx.get(inner_arg) {
        Expr::Mul(_, _) => expr_nary::mul_factors(ctx, inner_arg)
            .into_iter()
            .map(|factor| (factor, Sign::Pos))
            .collect::<Vec<_>>(),
        Expr::Div(numerator, denominator) => {
            let mut out = expr_nary::mul_factors(ctx, *numerator)
                .into_iter()
                .map(|factor| (factor, Sign::Pos))
                .collect::<Vec<_>>();
            out.extend(
                expr_nary::mul_factors(ctx, *denominator)
                    .into_iter()
                    .map(|factor| (factor, Sign::Neg)),
            );
            out
        }
        _ => return None,
    };

    let expanded_terms = signed_args
        .into_iter()
        .map(|(factor, sign)| {
            let term_arg = if wrap_abs {
                ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Abs), vec![factor]))
            } else {
                factor
            };
            let log_term = build_log_call_for_family(ctx, family, term_arg);
            (scale_expr_by_positive_bigint(ctx, &coeff, log_term), sign)
        })
        .collect::<Vec<_>>();
    let expanded_expr = build_add_from_signed_terms(ctx, &expanded_terms);
    Some((term, expanded_expr, expanded_terms))
}

fn rewrite_log_power_term_concretely(ctx: &mut Context, term: ExprId) -> Option<ExprId> {
    let (coeff, log_expr) = scaled_log_term(ctx, term).unwrap_or_else(|| (1.into(), term));
    let (family, base, exponent) = log_power_extraction_family(ctx, log_expr)?;
    let power = positive_integer_literal_value(ctx, exponent)?;
    let target_log = if matches!(family, LogDidacticFamily::Ln) && (&power % 2) == 0.into() {
        let abs_base = ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Abs), vec![base]));
        build_log_call_for_family(ctx, family, abs_base)
    } else {
        build_log_call_for_family(ctx, family, base)
    };

    Some(scale_expr_by_positive_bigint(
        ctx,
        &(coeff * power),
        target_log,
    ))
}

fn build_log_call_for_family(ctx: &mut Context, family: LogDidacticFamily, arg: ExprId) -> ExprId {
    match family {
        LogDidacticFamily::Ln => ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Ln), vec![arg])),
        LogDidacticFamily::Log10 => {
            ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Log), vec![arg]))
        }
        LogDidacticFamily::LogBase(base) => ctx.add(Expr::Function(
            ctx.builtin_id(BuiltinFn::Log),
            vec![base, arg],
        )),
    }
}

fn log_power_extraction_family(
    ctx: &Context,
    expr: ExprId,
) -> Option<(LogDidacticFamily, ExprId, ExprId)> {
    let (family, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            (LogDidacticFamily::Ln, args[0])
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Log) => {
            match args.as_slice() {
                [arg] => (LogDidacticFamily::Log10, *arg),
                [base, arg] => (LogDidacticFamily::LogBase(*base), *arg),
                _ => return None,
            }
        }
        _ => return None,
    };

    let Expr::Pow(base, exponent) = ctx.get(arg) else {
        return None;
    };
    let Expr::Number(value) = ctx.get(*exponent) else {
        return None;
    };
    if !value.is_integer() || value <= &BigRational::zero() {
        return None;
    }

    Some((family, *base, *exponent))
}

pub(super) fn generate_log_power_contraction_substep(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    if matches_even_abs_ln_power_contraction(ctx, before, after) {
        return Some(concrete_expr_substep(
            ctx,
            "Usar n · ln(|u|) = ln(u^n) cuando n es par",
            before,
            after,
        ));
    }

    if matches_general_log_power_contraction(ctx, before, after) {
        return Some(concrete_expr_substep(
            ctx,
            "Usar n · log_b(u) = log_b(u^n)",
            before,
            after,
        ));
    }

    None
}

pub(super) fn generate_log_change_of_base_chain_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    if let Some(chain_len) = log_change_of_base_chain_contraction_len(ctx, before, after) {
        if chain_len == 2 {
            return Some(vec![schema_substep(
                "Usar log_b(a) · log_a(c) = log_b(c)",
                "log_b(a) · log_a(c)",
                "log_b(c)",
                "\\log_b(a)\\cdot \\log_a(c)",
                "\\log_b(c)",
            )]);
        }

        return Some(vec![schema_substep(
            "Encadenar los cambios de base intermedios",
            "log_{u0}(u1) · log_{u1}(u2) · ... · log_{u_{n-1}}(u_n)",
            "log_{u0}(u_n)",
            "\\log_{u_0}(u_1)\\cdot \\log_{u_1}(u_2)\\cdots \\log_{u_{n-1}}(u_n)",
            "\\log_{u_0}(u_n)",
        )]);
    }

    if let Some(chain_len) = log_change_of_base_chain_expansion_len(ctx, before, after) {
        if chain_len == 2 {
            return Some(vec![schema_substep(
                "Usar log_b(c) = log_a(c) · log_b(a)",
                "log_b(c)",
                "log_a(c) · log_b(a)",
                "\\log_b(c)",
                "\\log_a(c)\\cdot \\log_b(a)",
            )]);
        }

        return Some(vec![schema_substep(
            "Desplegar un logaritmo en una cadena de cambios de base",
            "log_{u0}(u_n)",
            "log_{u0}(u1) · log_{u1}(u2) · ... · log_{u_{n-1}}(u_n)",
            "\\log_{u_0}(u_n)",
            "\\log_{u_0}(u_1)\\cdot \\log_{u_1}(u_2)\\cdots \\log_{u_{n-1}}(u_n)",
        )]);
    }

    None
}

fn log_change_of_base_chain_contraction_len(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<usize> {
    let Some((Some(target_base), target_arg)) = general_log_base_and_arg(ctx, after) else {
        return None;
    };

    let factors = expr_nary::mul_leaves(ctx, before);
    log_change_of_base_chain_len(ctx, &factors, target_base, target_arg)
}

fn log_change_of_base_chain_expansion_len(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<usize> {
    let Some((Some(target_base), target_arg)) = general_log_base_and_arg(ctx, before) else {
        return None;
    };

    let factors = expr_nary::mul_leaves(ctx, after);
    log_change_of_base_chain_len(ctx, &factors, target_base, target_arg)
}

fn log_change_of_base_chain_len(
    ctx: &Context,
    factors: &[ExprId],
    target_base: ExprId,
    target_arg: ExprId,
) -> Option<usize> {
    if factors.len() < 2 {
        return None;
    }

    let chain_nodes: Option<Vec<(ExprId, ExprId)>> = factors
        .iter()
        .map(|factor| {
            let (Some(base), arg) = general_log_base_and_arg(ctx, *factor)? else {
                return None;
            };
            Some((base, arg))
        })
        .collect();
    let chain_nodes = chain_nodes?;

    for start in 0..chain_nodes.len() {
        if cas_ast::ordering::compare_expr(ctx, chain_nodes[start].0, target_base)
            != std::cmp::Ordering::Equal
        {
            continue;
        }

        let mut used = vec![false; chain_nodes.len()];
        used[start] = true;
        if log_change_of_base_chain_dfs(ctx, &chain_nodes, start, target_arg, 1, &mut used) {
            return Some(chain_nodes.len());
        }
    }

    None
}

fn log_change_of_base_chain_dfs(
    ctx: &Context,
    nodes: &[(ExprId, ExprId)],
    current: usize,
    target_arg: ExprId,
    depth: usize,
    used: &mut [bool],
) -> bool {
    if depth == nodes.len() {
        return cas_ast::ordering::compare_expr(ctx, nodes[current].1, target_arg)
            == std::cmp::Ordering::Equal;
    }

    let current_arg = nodes[current].1;
    for next in 0..nodes.len() {
        if used[next] {
            continue;
        }
        if cas_ast::ordering::compare_expr(ctx, current_arg, nodes[next].0)
            != std::cmp::Ordering::Equal
        {
            continue;
        }
        used[next] = true;
        if log_change_of_base_chain_dfs(ctx, nodes, next, target_arg, depth + 1, used) {
            return true;
        }
        used[next] = false;
    }

    false
}

fn matches_even_abs_ln_power_contraction(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let Some((coeff, log_expr)) = scaled_log_term(ctx, before) else {
        return false;
    };
    if coeff <= 0.into() || (&coeff % 2) != 0.into() {
        return false;
    }

    let Expr::Function(fn_id, args) = ctx.get(log_expr) else {
        return false;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Ln) || args.len() != 1 {
        return false;
    }
    let Some(inner) = abs_argument(ctx, args[0]) else {
        return false;
    };

    let Expr::Function(after_fn, after_args) = ctx.get(after) else {
        return false;
    };
    if !ctx.is_builtin(*after_fn, BuiltinFn::Ln) || after_args.len() != 1 {
        return false;
    }

    let Expr::Pow(after_base, after_exp) = ctx.get(after_args[0]) else {
        return false;
    };
    let Some(exponent) = positive_integer_literal_value(ctx, *after_exp) else {
        return false;
    };

    exponent == coeff
        && cas_ast::ordering::compare_expr(ctx, inner, *after_base) == std::cmp::Ordering::Equal
}

fn matches_general_log_power_contraction(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let Some((coeff, log_expr)) = scaled_log_term(ctx, before) else {
        return false;
    };
    if coeff <= 0.into() {
        return false;
    }

    let Some((before_base, before_arg)) = general_log_base_and_arg(ctx, log_expr) else {
        return false;
    };
    let Some((after_base, after_arg)) = general_log_base_and_arg(ctx, after) else {
        return false;
    };
    if before_base != after_base {
        return false;
    }

    let Expr::Pow(after_pow_base, after_exp) = ctx.get(after_arg) else {
        return false;
    };
    let Some(exponent) = positive_integer_literal_value(ctx, *after_exp) else {
        return false;
    };

    exponent == coeff
        && cas_ast::ordering::compare_expr(ctx, before_arg, *after_pow_base)
            == std::cmp::Ordering::Equal
}

pub(super) fn generate_factor_perfect_square_log_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    vec![concrete_expr_substep(
        ctx,
        "Sacar un exponente par fuera del logaritmo",
        step.before_local().unwrap_or(step.before),
        step.after_local().unwrap_or(step.after),
    )]
}

pub(super) fn generate_log_contraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substeps) = generate_log_change_of_base_chain_substeps(ctx, before, after) {
        return substeps;
    }

    let Some((title, before_display, after_display, before_latex, after_latex)) =
        log_formula_snippet(ctx, before, false)
    else {
        return Vec::new();
    };

    if before != after {
        return vec![concrete_expr_substep(ctx, title, before, after)];
    }

    vec![formula_substep(
        title,
        &before_display,
        &after_display,
        &before_latex,
        &after_latex,
    )]
}

/// `Split Log Exponents` migrated to the matcher (2026-07-28). The rule is
/// named after its PURPOSE (freeing `e^(ln x)` pairs so they can cancel) but
/// the identity it applies is plain exponential algebra — `e^(A+B) = e^A·e^B`,
/// FREE of domain conditions — which is why it could migrate without the
/// domain-conditional census the other log rules are waiting for.
///
/// The engine folds `e^(log_e x)` to `x` inline while splitting, so a pair
/// like `e^(log_e(x) + k) ⟹ x·e^k` no longer LOOKS like the identity. The
/// directed pass of `match_instance` rescues it, which is exactly the job the
/// cycle-4 policy assigns to it: structural-first cites what is on screen,
/// directed-after rescues instances whose shape folded away. The ambiguity
/// that policy guards against — the directed mode picking the wrong row among
/// EQUIVALENT templates — cannot arise here, because this rule cites a single
/// template and there is nothing to choose between.
pub(super) fn generate_split_log_exponents_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    named_identity_substep(ctx, "e^(A+B)", "e^A · e^B", before, after)
        .into_iter()
        .collect()
}

fn log_formula_snippet(
    ctx: &Context,
    expr: ExprId,
    expand: bool,
) -> Option<(String, String, String, String, String)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            match ctx.get(args[0]) {
                Expr::Mul(_, _) => {
                    if expand {
                        Some((
                            "Usar que el logaritmo de un producto se separa en una suma"
                                .to_string(),
                            "ln(a · b)".to_string(),
                            "ln(a) + ln(b)".to_string(),
                            "\\ln(ab)".to_string(),
                            "\\ln(a) + \\ln(b)".to_string(),
                        ))
                    } else {
                        Some((
                            "Usar que una suma de logaritmos se puede reunir en un producto"
                                .to_string(),
                            "ln(a) + ln(b)".to_string(),
                            "ln(a · b)".to_string(),
                            "\\ln(a) + \\ln(b)".to_string(),
                            "\\ln(ab)".to_string(),
                        ))
                    }
                }
                Expr::Div(_, _) if expand => Some((
                    "Usar que el logaritmo de un cociente se separa en una resta".to_string(),
                    "ln(a / b)".to_string(),
                    "ln(a) - ln(b)".to_string(),
                    "\\ln\\left(\\frac{a}{b}\\right)".to_string(),
                    "\\ln(a) - \\ln(b)".to_string(),
                )),
                _ => None,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Log) && args.len() == 2 && expand =>
        {
            let base_display = human_expr(ctx, args[0]);
            let base_latex = latex_expr(ctx, args[0]);
            match ctx.get(args[1]) {
                Expr::Mul(_, _) => Some((
                    "Usar que el logaritmo de un producto se separa en una suma".to_string(),
                    format!("log_{base_display}(a · b)"),
                    format!("log_{base_display}(a) + log_{base_display}(b)"),
                    format!("\\log_{{{base_latex}}}(ab)"),
                    format!("\\log_{{{base_latex}}}(a) + \\log_{{{base_latex}}}(b)"),
                )),
                Expr::Div(_, _) => Some((
                    "Usar que el logaritmo de un cociente se separa en una resta".to_string(),
                    format!("log_{base_display}(a / b)"),
                    format!("log_{base_display}(a) - log_{base_display}(b)"),
                    format!("\\log_{{{base_latex}}}\\left(\\frac{{a}}{{b}}\\right)"),
                    format!("\\log_{{{base_latex}}}(a) - \\log_{{{base_latex}}}(b)"),
                )),
                _ => None,
            }
        }
        Expr::Add(_, _) | Expr::Sub(_, _) if !expand => {
            if let Some(snippet) = scaled_log_formula_snippet(ctx, expr) {
                return Some(snippet);
            }

            let (left, right) = match ctx.get(expr) {
                Expr::Add(left, right) | Expr::Sub(left, right) => (*left, *right),
                _ => return None,
            };
            let (left_fn, right_fn) = (ctx.get(left), ctx.get(right));
            match (left_fn, right_fn) {
                (Expr::Function(left_id, left_args), Expr::Function(right_id, right_args))
                    if ctx.is_builtin(*left_id, BuiltinFn::Ln)
                        && ctx.is_builtin(*right_id, BuiltinFn::Ln)
                        && left_args.len() == 1
                        && right_args.len() == 1 =>
                {
                    Some(if matches!(ctx.get(expr), Expr::Add(_, _)) {
                        (
                            "Usar que una suma de logaritmos se puede reunir en un producto"
                                .to_string(),
                            "ln(a) + ln(b)".to_string(),
                            "ln(a · b)".to_string(),
                            "\\ln(a) + \\ln(b)".to_string(),
                            "\\ln(ab)".to_string(),
                        )
                    } else {
                        (
                            "Usar que una resta de logaritmos se puede reunir en un cociente"
                                .to_string(),
                            "ln(a) - ln(b)".to_string(),
                            "ln(a / b)".to_string(),
                            "\\ln(a) - \\ln(b)".to_string(),
                            "\\ln\\left(\\frac{a}{b}\\right)".to_string(),
                        )
                    })
                }
                (Expr::Function(left_id, left_args), Expr::Function(right_id, right_args))
                    if ctx.is_builtin(*left_id, BuiltinFn::Log)
                        && ctx.is_builtin(*right_id, BuiltinFn::Log)
                        && left_args.len() == 2
                        && right_args.len() == 2
                        && cas_ast::ordering::compare_expr(ctx, left_args[0], right_args[0])
                            == std::cmp::Ordering::Equal =>
                {
                    let base_display = human_expr(ctx, left_args[0]);
                    let base_latex = latex_expr(ctx, left_args[0]);
                    Some(if matches!(ctx.get(expr), Expr::Add(_, _)) {
                        (
                            "Usar que una suma de logaritmos se puede reunir en un producto"
                                .to_string(),
                            format!("log_{base_display}(a) + log_{base_display}(b)"),
                            format!("log_{base_display}(a · b)"),
                            format!("\\log_{{{base_latex}}}(a) + \\log_{{{base_latex}}}(b)"),
                            format!("\\log_{{{base_latex}}}(ab)"),
                        )
                    } else {
                        (
                            "Usar que una resta de logaritmos se puede reunir en un cociente"
                                .to_string(),
                            format!("log_{base_display}(a) - log_{base_display}(b)"),
                            format!("log_{base_display}(a / b)"),
                            format!("\\log_{{{base_latex}}}(a) - \\log_{{{base_latex}}}(b)"),
                            format!("\\log_{{{base_latex}}}\\left(\\frac{{a}}{{b}}\\right)"),
                        )
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn scaled_log_formula_snippet(
    ctx: &Context,
    expr: ExprId,
) -> Option<(String, String, String, String, String)> {
    let (left, right, is_sub) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right, false),
        Expr::Sub(left, right) => (*left, *right, true),
        _ => return None,
    };

    let left_term = extract_scaled_log_didactic_term(ctx, left)?;
    let right_term = extract_scaled_log_didactic_term(ctx, right)?;
    if left_term.coeff == 1.into() && right_term.coeff == 1.into() {
        return None;
    }
    if !same_log_didactic_family(ctx, left_term.family, right_term.family) {
        return None;
    }

    let (log_display, log_latex) = log_family_formula_name(ctx, left_term.family);
    if is_sub {
        Some((
            "Meter los coeficientes dentro de los logaritmos y reunir la resta en un cociente"
                .to_string(),
            format!(
                "{} · {log_display}(u) - {} · {log_display}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_display}(u^{} / v^{})",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{}\\cdot {log_latex}(u) - {}\\cdot {log_latex}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_latex}\\left(\\frac{{u^{}}}{{v^{}}}\\right)",
                left_term.coeff, right_term.coeff
            ),
        ))
    } else {
        Some((
            "Meter los coeficientes dentro de los logaritmos como exponentes".to_string(),
            format!(
                "{} · {log_display}(u) + {} · {log_display}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_display}(u^{} · v^{})",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{}\\cdot {log_latex}(u) + {}\\cdot {log_latex}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_latex}(u^{}\\cdot v^{})",
                left_term.coeff, right_term.coeff
            ),
        ))
    }
}

fn extract_scaled_log_didactic_term(ctx: &Context, expr: ExprId) -> Option<ScaledLogDidacticTerm> {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            if let Some(coeff) = positive_integer_literal_value(ctx, *left) {
                let family = extract_log_didactic_family(ctx, *right)?;
                return Some(ScaledLogDidacticTerm { family, coeff });
            }
            if let Some(coeff) = positive_integer_literal_value(ctx, *right) {
                let family = extract_log_didactic_family(ctx, *left)?;
                return Some(ScaledLogDidacticTerm { family, coeff });
            }
            None
        }
        _ => Some(ScaledLogDidacticTerm {
            family: extract_log_didactic_family(ctx, expr)?,
            coeff: 1.into(),
        }),
    }
}

fn extract_log_didactic_family(ctx: &Context, expr: ExprId) -> Option<LogDidacticFamily> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };

    if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 {
        return Some(LogDidacticFamily::Ln);
    }
    if ctx.is_builtin(*fn_id, BuiltinFn::Log) {
        return match args.as_slice() {
            [_arg] => Some(LogDidacticFamily::Log10),
            [base, _arg] => Some(LogDidacticFamily::LogBase(*base)),
            _ => None,
        };
    }

    None
}

fn same_log_didactic_family(
    ctx: &Context,
    left: LogDidacticFamily,
    right: LogDidacticFamily,
) -> bool {
    match (left, right) {
        (LogDidacticFamily::Ln, LogDidacticFamily::Ln) => true,
        (LogDidacticFamily::Log10, LogDidacticFamily::Log10) => true,
        (LogDidacticFamily::LogBase(left), LogDidacticFamily::LogBase(right)) => {
            cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
        }
        _ => false,
    }
}

fn log_family_formula_name(ctx: &Context, family: LogDidacticFamily) -> (String, String) {
    match family {
        LogDidacticFamily::Ln => ("ln".to_string(), "\\ln".to_string()),
        LogDidacticFamily::Log10 => ("log".to_string(), "\\log".to_string()),
        LogDidacticFamily::LogBase(base) => {
            let base_display = human_expr(ctx, base);
            let base_latex = latex_expr(ctx, base);
            (
                format!("log_{base_display}"),
                format!("\\log_{{{base_latex}}}"),
            )
        }
    }
}

pub(super) fn change_of_base_log_arguments(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 2 && ctx.is_builtin(*name, BuiltinFn::Log) {
        Some((args[0], args[1]))
    } else {
        None
    }
}

pub(super) fn change_of_base_natural_log_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1
        && (ctx.is_builtin(*name, BuiltinFn::Ln) || ctx.is_builtin(*name, BuiltinFn::Log))
    {
        Some(args[0])
    } else {
        None
    }
}

pub(super) fn generate_polynomial_affine_log_by_parts_substeps(
    ctx: &Context,
    integrand: ExprId,
    after: ExprId,
    var_name: &str,
) -> Vec<SubStep> {
    let Some((log_factor, log_arg, polynomial_factor)) =
        polynomial_affine_log_by_parts_factors(ctx, integrand, var_name)
    else {
        return Vec::new();
    };
    let Some((v_display, v_latex)) =
        polynomial_antiderivative_display(ctx, polynomial_factor, var_name)
    else {
        return Vec::new();
    };
    let Some((du_display, du_factor_display, du_latex, du_factor_latex)) =
        log_argument_derivative_fraction_display(ctx, log_arg, var_name)
    else {
        return Vec::new();
    };

    let u_display = display_expr(ctx, log_factor);
    let u_latex = latex_expr(ctx, log_factor);
    let v_display_factor = group_display_for_product(&v_display);
    let v_latex_factor = group_latex_for_product(&v_latex);
    let dv_display = format!("{} dx", display_expr(ctx, polynomial_factor));
    let dv_latex = format!("{}\\,dx", latex_expr(ctx, polynomial_factor));
    let choice_display = format!("u = {}, dv = {}", u_display, dv_display);
    let choice_latex = format!("u = {},\\; dv = {}", u_latex, dv_latex);

    vec![
        SubStep::keyed(
            "by_parts.choose_u_dv",
            vec![],
            display_expr(ctx, integrand),
            choice_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, integrand))
        .with_after_latex(choice_latex.clone()),
        SubStep::keyed(
            "by_parts.compute_du_v",
            vec![],
            choice_display,
            format!("du = {}, v = {}", du_display, v_display),
        )
        .with_before_latex(choice_latex)
        .with_after_latex(format!("du = {},\\; v = {}", du_latex, v_latex)),
        SubStep::keyed(
            "by_parts.apply_formula",
            vec![],
            format!(
                "{}·{} - integrate({}·{}, {})",
                u_display, v_display_factor, v_display_factor, du_factor_display, var_name
            ),
            display_expr(ctx, after),
        )
        .with_before_latex(format!(
            "{}\\cdot {} - \\int {}\\cdot {}\\,d{}",
            u_latex, v_latex_factor, v_latex_factor, du_factor_latex, var_name
        ))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn polynomial_affine_log_by_parts_factors(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, ExprId)> {
    let (left, right) = as_mul(ctx, integrand)?;
    if let Some(log_arg) = ln_affine_arg(ctx, left, var_name) {
        return polynomial_antiderivative_display(ctx, right, var_name)
            .map(|_| (left, log_arg, right));
    }
    if let Some(log_arg) = ln_affine_arg(ctx, right, var_name) {
        return polynomial_antiderivative_display(ctx, left, var_name)
            .map(|_| (right, log_arg, left));
    }
    None
}

fn ln_affine_arg(ctx: &Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Ln) {
        return None;
    }
    is_affine_in_var(ctx, args[0], var_name).then_some(args[0])
}

/// Narrate the normalized exponential quotients: the simplifier rewrites
/// p(x)*e^(-u) into Div(p(x), e^u), so the story is "rewrite the quotient
/// back as an exponential product" followed by the table rule (var-free
/// numerator) or integration by parts (polynomial numerator). The product
/// form is rebuilt on a scratch context and quoted as the intermediate.
pub(super) fn generate_normalized_exponential_div_integration_substeps(
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
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym).to_string();
    let Expr::Div(numerator, denominator) = ctx.get(args[0]) else {
        return Vec::new();
    };
    let (numerator, denominator) = (*numerator, *denominator);
    let Some(exponent) = didactic_exp_like_arg(ctx, denominator) else {
        return Vec::new();
    };
    let Ok(poly) = cas_math::polynomial::Polynomial::from_expr(ctx, exponent, &var_name) else {
        return Vec::new();
    };
    if poly.degree() != 1 {
        return Vec::new();
    }

    let mut result = after;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, result);
        if unwrapped == result {
            break;
        }
        result = unwrapped;
    }
    if expr_contains_integrate_call(ctx, result) {
        return Vec::new();
    }

    let mut scratch = ctx.clone();
    let negated_exponent = scratch.add(Expr::Neg(exponent));
    let exp_factor = scratch.call_builtin(BuiltinFn::Exp, vec![negated_exponent]);
    let product = scratch.add(Expr::Mul(numerator, exp_factor));

    let integration_title = if contains_named_var(ctx, numerator, &var_name) {
        "Usar integración por partes"
    } else {
        "Usar la regla de la exponencial"
    };

    vec![
        SubStep::new(
            "Reescribir el cociente como producto exponencial",
            display_expr(ctx, args[0]),
            display_expr(&scratch, product),
        )
        .with_before_latex(latex_expr(ctx, args[0]))
        .with_after_latex(latex_expr(&scratch, product)),
        SubStep::new(
            integration_title,
            display_expr(&scratch, product),
            display_expr(ctx, result),
        )
        .with_before_latex(latex_expr(&scratch, product))
        .with_after_latex(latex_expr(ctx, result)),
    ]
}

fn didactic_exp_like_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Exp)) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exponent)
            if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) =>
        {
            Some(*exponent)
        }
        _ => None,
    }
}

/// Find the Ln call inside a result term and return its argument when it
/// is a compact positive quadratic (the completed-square form).
pub(super) fn compact_quadratic_from_log_term(ctx: &Context, term: ExprId) -> Option<ExprId> {
    match ctx.get(term) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(*fn_id, BuiltinFn::Ln) {
                let arg = cas_ast::hold::unwrap_hold(ctx, args[0]);
                let arg = match ctx.get(arg) {
                    Expr::Function(abs_id, abs_args)
                        if abs_args.len() == 1 && ctx.is_builtin(*abs_id, BuiltinFn::Abs) =>
                    {
                        abs_args[0]
                    }
                    _ => arg,
                };
                return is_compact_positive_quadratic_denominator(ctx, arg).then_some(arg);
            }
            args.iter()
                .find_map(|inner| compact_quadratic_from_log_term(ctx, *inner))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            compact_quadratic_from_log_term(ctx, *l)
                .or_else(|| compact_quadratic_from_log_term(ctx, *r))
        }
        Expr::Neg(inner) | Expr::Hold(inner) => compact_quadratic_from_log_term(ctx, *inner),
        _ => None,
    }
}

pub(super) fn generate_linear_log_table_integration_substeps(
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
    let Some(denominator) = linear_log_table_denominator_arg(ctx, args[0], var_name) else {
        return Vec::new();
    };
    let Ok(denominator_poly) = Polynomial::from_expr(ctx, denominator, var_name) else {
        return Vec::new();
    };
    let slope = denominator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let mut substeps = vec![
        SubStep::new(
            "Usar la regla de ln|u| con derivada interna",
            display_expr(ctx, args[0]),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, args[0]))
        .with_after_latex(latex_expr(ctx, after)),
        SubStep::keyed(
            "usub.identify_affine_denominator",
            vec![],
            display_expr(ctx, denominator),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, denominator))
        .with_after_latex(latex_expr(ctx, after)),
    ];

    if !slope.is_one() {
        substeps.push(
            SubStep::keyed(
                "usub.adjust_constant_factor",
                vec![],
                affine_internal_derivative_display(ctx, denominator, var_name, &slope),
                display_expr(ctx, after),
            )
            .with_before_latex(affine_internal_derivative_latex(
                ctx,
                denominator,
                var_name,
                &slope,
            ))
            .with_after_latex(latex_expr(ctx, after)),
        );
    }

    substeps
}

fn linear_log_table_denominator_arg(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some((numerator, denominator)) = as_div(ctx, integrand) {
        let coefficient = as_rational_const(ctx, numerator, 8)?;
        if coefficient.is_zero() || !nontrivial_affine_argument(ctx, denominator, var_name) {
            return None;
        }
        return Some(denominator);
    }

    let (base, exponent) = as_pow(ctx, integrand)?;
    let exponent = as_rational_const(ctx, exponent, 8)?;
    if exponent != -BigRational::one() || !nontrivial_affine_argument(ctx, base, var_name) {
        return None;
    }
    Some(base)
}

pub(super) fn generate_log_power_product_table_integration_substeps(
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
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_log_product_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_log_power_product_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) {
        return Vec::new();
    }

    let Some(table_match) = log_power_product_table_integrand(ctx, args[0], var_name) else {
        return Vec::new();
    };
    let title = if table_match.natural_log && table_match.power == 1 {
        "Usar la regla de u'·ln(u) -> u·(ln(u)-1)"
    } else if table_match.natural_log {
        "Usar la regla de u'·ln(u)^n por partes"
    } else {
        "Usar la regla de u'·log_b(u)^n por partes"
    };

    let mut substeps = Vec::new();
    if let Some(step) = checked_antiderivative_substep(ctx, title, args[0], after, var_name) {
        substeps.push(step);
    }
    substeps.push(
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            display_expr(ctx, table_match.base),
            table_match.derivative_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, table_match.base))
        .with_after_latex(table_match.derivative_latex.clone()),
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

pub(super) fn log_power_product_table_from_factors(
    ctx: &Context,
    negative: bool,
    factors: &[ExprId],
    var_name: &str,
) -> Option<LogPowerProductTableMatch> {
    for (log_index, factor) in factors.iter().enumerate() {
        let Some((base, power, natural_log)) = log_power_product_factor_parts(ctx, *factor) else {
            continue;
        };
        if !contains_named_var(ctx, base, var_name) {
            continue;
        }

        let remaining_factors = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect::<Vec<_>>();
        let trace = polynomial_derivative_cofactor_trace(
            ctx,
            negative,
            &remaining_factors,
            base,
            var_name,
        )?;

        return Some(LogPowerProductTableMatch {
            power,
            natural_log,
            base,
            cofactor_display: trace.cofactor_display,
            cofactor_latex: trace.cofactor_latex,
            derivative_display: trace.derivative_display,
            derivative_latex: trace.derivative_latex,
            scale: trace.scale,
        });
    }

    None
}

fn log_power_product_factor_parts(ctx: &Context, factor: ExprId) -> Option<(ExprId, u32, bool)> {
    let (log_expr, power) = match ctx.get(factor) {
        Expr::Function(_, _) => (factor, 1),
        Expr::Pow(base, exponent) => {
            let power = as_rational_const(ctx, *exponent, 8)?;
            if !power.denom().is_one() || !power.is_positive() {
                return None;
            }
            (*base, power.to_integer().to_u32()?)
        }
        _ => return None,
    };

    let Expr::Function(fn_id, args) = ctx.get(log_expr) else {
        return None;
    };
    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Ln) if args.len() == 1 => Some((args[0], power, true)),
        Some(BuiltinFn::Log) if args.len() == 2 => {
            let natural_log = matches!(ctx.get(args[0]), Expr::Constant(Constant::E));
            if !natural_log && power == 1 {
                return None;
            }
            Some((args[1], power, natural_log))
        }
        Some(BuiltinFn::Log2 | BuiltinFn::Log10) if args.len() == 1 => {
            if power == 1 {
                return None;
            }
            Some((args[0], power, false))
        }
        _ => None,
    }
}

fn scaled_log_term(ctx: &Context, expr: ExprId) -> Option<(num_bigint::BigInt, ExprId)> {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            if let Some(coeff) = positive_integer_literal_value(ctx, *left) {
                Some((coeff, *right))
            } else {
                positive_integer_literal_value(ctx, *right).map(|coeff| (coeff, *left))
            }
        }
        _ => None,
    }
}

fn general_log_base_and_arg(ctx: &Context, expr: ExprId) -> Option<(Option<ExprId>, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Log) {
        return None;
    }
    match args.as_slice() {
        [arg] => Some((None, *arg)),
        [base, arg] => Some((Some(*base), *arg)),
        _ => None,
    }
}
