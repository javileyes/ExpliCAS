//! `arithmetic`: familia `powers`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn power_merge_base_supported_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_symbolic_atom_for_cancellation(ctx, expr)
        || matches!(ctx.get(expr), Expr::Number(_) | Expr::Constant(_))
}

pub(super) fn maybe_odd_half_power_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    match ctx.get(expr) {
        Expr::Sub(lhs, rhs) => {
            expr_contains_sqrt_or_half_power(ctx, *lhs)
                || expr_contains_sqrt_or_half_power(ctx, *rhs)
        }
        Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
            Expr::Neg(inner) => {
                expr_contains_sqrt_or_half_power(ctx, *lhs)
                    || expr_contains_sqrt_or_half_power(ctx, *inner)
            }
            _ => false,
        },
        _ => false,
    }
}

fn sqrt_factor_with_polynomial_tail(
    ctx: &mut cas_ast::Context,
    factor: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
    if let Some(base) = extract_square_root_base(ctx, factor) {
        return Some((base, ctx.num(1)));
    }

    let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
    let offset = exponent - BigRational::new(1.into(), 2.into());
    if offset.is_negative() || !offset.is_integer() {
        return None;
    }
    let offset = offset.to_integer().to_usize()?;
    if offset > 4 {
        return None;
    }

    let tail = match offset {
        0 => ctx.num(1),
        1 => base,
        _ => {
            let exp = ctx.num(offset as i64);
            ctx.add(Expr::Pow(base, exp))
        }
    };
    Some((base, tail))
}

fn split_common_sqrt_factor_from_term(
    ctx: &mut cas_ast::Context,
    term: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if let Some((base, tail)) = sqrt_factor_with_polynomial_tail(ctx, term) {
        return Some((base, tail));
    }

    let view = MulView::from_expr(ctx, term);
    if view.factors.len() < 2 {
        return None;
    }

    for (index, factor) in view.factors.iter().copied().enumerate() {
        let Some((base, tail)) = sqrt_factor_with_polynomial_tail(ctx, factor) else {
            continue;
        };
        let mut remaining = Vec::with_capacity(view.factors.len());
        if !is_one_expr(ctx, tail) {
            remaining.push(tail);
        }
        remaining.extend(
            view.factors
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor)),
        );
        return Some((base, build_scale_from_factors(ctx, &remaining)));
    }

    None
}

fn pull_common_sqrt_factor_from_additive_denominator(
    ctx: &mut cas_ast::Context,
    denominator: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let (common_base, residual_sum) =
        pull_common_sqrt_factor_from_additive_terms(ctx, denominator)?;
    let sqrt_common_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![common_base]);
    let factored = build_balanced_mul(ctx, &[sqrt_common_base, residual_sum]);
    Some((common_base, factored))
}

fn pull_common_sqrt_factor_from_additive_terms(
    ctx: &mut cas_ast::Context,
    denominator: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let denominator = cas_ast::hold::unwrap_internal_hold(ctx, denominator);
    let terms = AddView::from_expr(ctx, denominator).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut common_base = None;
    let mut residual_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (base, residual) = split_common_sqrt_factor_from_term(ctx, term)?;
        if let Some(existing_base) = common_base {
            if !exprs_match_for_cancellation(ctx, existing_base, base) {
                return None;
            }
        } else {
            common_base = Some(base);
        }
        residual_terms.push(apply_sign_to_expr(ctx, sign_to_i64(sign), residual));
    }

    let common_base = common_base?;
    let residual_sum = build_balanced_add(ctx, &residual_terms);
    Some((common_base, residual_sum))
}

fn denominators_match_after_common_sqrt_factor_pull(
    ctx: &mut cas_ast::Context,
    lhs_den: cas_ast::ExprId,
    rhs_den: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    for (expanded, factored) in [(lhs_den, rhs_den), (rhs_den, lhs_den)] {
        let Some((base, pulled)) = pull_common_sqrt_factor_from_additive_denominator(ctx, expanded)
        else {
            continue;
        };
        if exprs_match_for_cancellation(ctx, pulled, factored)
            || exprs_match_after_default_simplify(ctx, pulled, factored)
        {
            return Some(base);
        }
    }

    None
}

fn try_rewrite_odd_half_power_target_aware(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewrite) = cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    let normalized = cas_math::canonical_forms::normalize_core(ctx, expr);
    if normalized == expr {
        return None;
    }

    cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, normalized)
        .map(|rewrite| rewrite.rewritten)
}

fn try_rewrite_odd_half_power_with_optional_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewritten) = try_rewrite_odd_half_power_target_aware(ctx, expr) {
        return Some(rewritten);
    }

    let simplified = run_default_simplify(ctx, expr);
    if simplified == expr {
        return None;
    }

    try_rewrite_odd_half_power_target_aware(ctx, simplified)
}

fn extract_odd_half_power_outer_factor(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, i64)> {
    if let Some(inner) = abs_argument(ctx, expr) {
        return Some((inner, 1));
    }

    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            let power = small_positive_integer_value(ctx, *exponent)?;
            if let Some(inner) = abs_argument(ctx, *base) {
                Some((inner, power))
            } else {
                Some((*base, power))
            }
        }
        _ => Some((expr, 1)),
    }
}

fn extract_odd_half_power_product_form(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<OddHalfPowerProductForm> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().copied().enumerate() {
        let Some(base) = extract_sqrt_argument(ctx, sqrt_factor) else {
            continue;
        };
        let outer_factor = factors[1 - sqrt_index];
        let Some((outer_base, outside_power)) =
            extract_odd_half_power_outer_factor(ctx, outer_factor)
        else {
            continue;
        };
        if compare_expr(ctx, outer_base, base) == Ordering::Equal {
            return Some(OddHalfPowerProductForm {
                base,
                outside_power,
            });
        }
    }

    None
}

fn odd_half_power_domain_equivalent_target_match(
    ctx: &cas_ast::Context,
    rewritten: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let rewritten_form = extract_odd_half_power_product_form(ctx, rewritten)?;
    let target_form = extract_odd_half_power_product_form(ctx, target_expr)?;
    (rewritten_form.outside_power == target_form.outside_power
        && compare_expr(ctx, rewritten_form.base, target_form.base) == Ordering::Equal)
        .then_some(rewritten_form.base)
}

pub(super) fn try_match_odd_half_power_cancellation_side(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<OddHalfPowerCancellationMatch> {
    let rewritten = try_rewrite_odd_half_power_with_optional_simplify(ctx, focus_expr)?;
    if compare_expr(ctx, rewritten, target_expr) == Ordering::Equal {
        return Some(OddHalfPowerCancellationMatch {
            focus_before: focus_expr,
            focus_after: target_expr,
            rewritten_expr: target_expr,
            base: None,
        });
    }

    let base = odd_half_power_domain_equivalent_target_match(ctx, rewritten, target_expr)?;
    Some(OddHalfPowerCancellationMatch {
        focus_before: focus_expr,
        focus_after: target_expr,
        rewritten_expr: target_expr,
        base: Some(base),
    })
}

pub(super) fn extract_square_plus_minus_one_pattern_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, Sign, Sign)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut square_term = None;
    let mut one_term = None;
    for (term_expr, term_sign) in terms {
        let (term_expr, term_sign) = normalize_signed_add_term(ctx, term_expr, term_sign);
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if one_term.replace(term_sign).is_some() {
                return None;
            }
            continue;
        }

        let square_base = extract_square_power_base(ctx, term_expr)?;
        if square_term.replace((square_base, term_sign)).is_some() {
            return None;
        }
    }

    let (square_base, square_sign) = square_term?;
    Some((square_base, square_sign, one_term?))
}

pub(super) fn expr_contains_variable_square(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
    if let Some(hit) = VARIABLE_SQUARE_GATE_MEMO.with(|m| m.borrow().get(&root).copied()) {
        return hit;
    }
    let result = expr_contains_variable_square_uncached(ctx, root);
    VARIABLE_SQUARE_GATE_MEMO.with(|m| m.borrow_mut().insert(root, result));
    result
}

fn expr_contains_variable_square_uncached(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
    let mut stack = vec![root];
    // Per-node rule gate: on recurrence-shaped DAGs (tan(10*arcsin(t)))
    // an unmemoized walk revisits shared subtrees exponentially; the
    // visited set keeps the same answer at DAG-sized cost.
    let mut seen: rustc_hash::FxHashSet<cas_ast::ExprId> = rustc_hash::FxHashSet::default();
    while let Some(expr) = stack.pop() {
        if !seen.insert(expr) {
            continue;
        }
        match ctx.get(expr) {
            Expr::Pow(base, exp)
                if extract_i64_integer(ctx, *exp) == Some(2)
                    && matches!(ctx.get(*base), Expr::Variable(_)) =>
            {
                return true;
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(super) fn collect_squared_variable_names(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> Vec<String> {
    let mut names = std::collections::BTreeSet::new();
    let mut stack = vec![root];
    // Per-node rule gate: on recurrence-shaped DAGs (tan(10*arcsin(t)))
    // an unmemoized walk revisits shared subtrees exponentially; the
    // visited set keeps the same answer at DAG-sized cost.
    let mut seen: rustc_hash::FxHashSet<cas_ast::ExprId> = rustc_hash::FxHashSet::default();
    while let Some(expr) = stack.pop() {
        if !seen.insert(expr) {
            continue;
        }
        match ctx.get(expr) {
            Expr::Pow(base, exp) if extract_i64_integer(ctx, *exp) == Some(2) => {
                if let Expr::Variable(sym_id) = ctx.get(*base) {
                    names.insert(ctx.sym_name(*sym_id).to_string());
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
            Expr::Pow(_, _) => {}
        }
    }

    names.into_iter().collect()
}

fn collect_non_division_square_variable_names_from_term(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    names: &mut std::collections::BTreeSet<String>,
) {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Pow(base, exp)
                if extract_i64_integer(ctx, *exp) == Some(2)
                    && matches!(ctx.get(*base), Expr::Variable(_)) =>
            {
                if let Expr::Variable(sym_id) = ctx.get(*base) {
                    names.insert(ctx.sym_name(*sym_id).to_string());
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Div(_, _)
            | Expr::Number(_)
            | Expr::Constant(_)
            | Expr::Variable(_)
            | Expr::SessionRef(_) => {}
            Expr::Pow(base, exp) => {
                stack.push(*base);
                stack.push(*exp);
            }
        }
    }
}

pub(super) fn collect_direct_additive_square_variable_names(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> Vec<String> {
    let mut names = std::collections::BTreeSet::new();
    let view = AddView::from_expr(ctx, root);
    for (term_expr, _) in view.terms {
        collect_non_division_square_variable_names_from_term(ctx, term_expr, &mut names);
    }
    names.into_iter().collect()
}

pub(super) fn expr_contains_named_var_outside_simple_square(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    var: &str,
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => return true,
            Expr::Pow(base, exp) if extract_i64_integer(ctx, *exp) == Some(2) => {
                if matches!(ctx.get(*base), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
                {
                    continue;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            Expr::Variable(_) => {}
        }
    }

    false
}

pub(super) fn is_direct_complete_square_symbolic_scale_expr(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    matches!(ctx.get(expr), Expr::Variable(_))
}

pub(super) fn build_complete_square_binomial_expr(
    ctx: &mut cas_ast::Context,
    var_expr: cas_ast::ExprId,
    shift: cas_ast::ExprId,
    additive_orientation: Sign,
) -> cas_ast::ExprId {
    match additive_orientation {
        Sign::Pos => {
            if let Some(inner_shift) = strip_unit_negation_for_phase_shift(ctx, shift) {
                ctx.add_raw(Expr::Sub(var_expr, inner_shift))
            } else {
                ctx.add_raw(Expr::Add(var_expr, shift))
            }
        }
        Sign::Neg => {
            if let Some(inner_shift) = strip_unit_negation_for_phase_shift(ctx, shift) {
                ctx.add_raw(Expr::Add(var_expr, inner_shift))
            } else {
                ctx.add_raw(Expr::Sub(var_expr, shift))
            }
        }
    }
}

pub(super) fn build_complete_square_candidate_for_var_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    var: &str,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, SolvePrepBuildRoute)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let candidate_sample = profiling.then(|| {
        format!(
            "{} :: {}",
            render_expr_for_orchestrator_profile(ctx, expr),
            var
        )
    });
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, candidate_sample.clone(), || {
                    Some(())
                });
        }
    };
    let (a, b, c) = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.solve_prep.build_candidate.extract_coeffs",
            candidate_sample.clone(),
            || {
                extract_profiled_solve_prep_nonzero_quadratic_coefficients(
                    ctx,
                    expr,
                    var,
                    profiling,
                    &candidate_sample,
                )
            },
        )?
    } else {
        extract_profiled_solve_prep_nonzero_quadratic_coefficients(
            ctx,
            expr,
            var,
            profiling,
            &candidate_sample,
        )?
    };

    if is_default_simplified_zero_expr(ctx, b) {
        profile_route("rule.solve_prep.build_candidate.reject_zero_linear");
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let var_expr = ctx.var(var);
    let b_squared = ctx.add(Expr::Pow(b, two));

    let (candidate_raw, nonzero_expr, build_route) =
        if let Some(positive_a) = strip_unit_negation_for_phase_shift(ctx, a) {
            if is_direct_complete_square_symbolic_scale_expr(ctx, positive_a)
                && strip_unit_negation_for_phase_shift(ctx, b).is_none()
            {
                profile_route("rule.solve_prep.build_candidate.route.neg_symbolic_scale");
                let two_pos_a = ctx.add(Expr::Mul(two, positive_a));
                let shift = ctx.add(Expr::Div(b, two_pos_a));
                let completed_binomial =
                    build_complete_square_binomial_expr(ctx, var_expr, shift, Sign::Neg);
                let square = ctx.add(Expr::Pow(completed_binomial, two));
                let scaled_square_inner = ctx.add(Expr::Mul(positive_a, square));
                let scaled_square = ctx.add(Expr::Neg(scaled_square_inner));
                let four_pos_a = ctx.add(Expr::Mul(four, positive_a));
                let correction = ctx.add(Expr::Div(b_squared, four_pos_a));
                let tail = ctx.add(Expr::Add(c, correction));
                let candidate_raw = ctx.add(Expr::Add(scaled_square, tail));
                let candidate = run_default_simplify(ctx, candidate_raw);
                return Some((candidate, positive_a, SolvePrepBuildRoute::NegSymbolic));
            }

            profile_route("rule.solve_prep.build_candidate.route.neg_generic_scale");
            let two_pos_a_raw = ctx.add(Expr::Mul(two, positive_a));
            let two_pos_a = run_default_simplify(ctx, two_pos_a_raw);
            let shift_raw = ctx.add(Expr::Div(b, two_pos_a));
            let shift = run_default_simplify(ctx, shift_raw);
            let completed_binomial_raw =
                build_complete_square_binomial_expr(ctx, var_expr, shift, Sign::Neg);
            let completed_binomial = run_default_simplify(ctx, completed_binomial_raw);
            let square = ctx.add(Expr::Pow(completed_binomial, two));
            let scaled_square_inner = ctx.add(Expr::Mul(positive_a, square));
            let scaled_square = ctx.add(Expr::Neg(scaled_square_inner));

            let four_pos_a_raw = ctx.add(Expr::Mul(four, positive_a));
            let four_pos_a = run_default_simplify(ctx, four_pos_a_raw);
            let correction_raw = ctx.add(Expr::Div(b_squared, four_pos_a));
            let correction = run_default_simplify(ctx, correction_raw);
            let tail_raw = ctx.add(Expr::Add(c, correction));
            let tail = run_default_simplify(ctx, tail_raw);
            (
                ctx.add(Expr::Add(scaled_square, tail)),
                positive_a,
                SolvePrepBuildRoute::NegGeneric,
            )
        } else {
            if let Some(double_a) = extract_positive_half_scaled_base_expr(ctx, a) {
                profile_route("rule.solve_prep.build_candidate.route.pos_half_scale");
                let half_a = ctx.add(Expr::Div(double_a, two));
                let shift = ctx.add(Expr::Div(b, double_a));
                let completed_binomial =
                    build_complete_square_binomial_expr(ctx, var_expr, shift, Sign::Pos);
                let square = ctx.add(Expr::Pow(completed_binomial, two));
                let scaled_square = ctx.add(Expr::Mul(half_a, square));
                let two_double_a = ctx.add(Expr::Mul(two, double_a));
                let correction = ctx.add(Expr::Div(b_squared, two_double_a));
                let tail = ctx.add(Expr::Sub(c, correction));
                let candidate_raw = ctx.add(Expr::Add(scaled_square, tail));
                let candidate = run_default_simplify(ctx, candidate_raw);
                return Some((candidate, double_a, SolvePrepBuildRoute::PosHalf));
            }

            if is_direct_complete_square_symbolic_scale_expr(ctx, a)
                && strip_unit_negation_for_phase_shift(ctx, b).is_none()
            {
                profile_route("rule.solve_prep.build_candidate.route.pos_symbolic_scale");
                let two_a = ctx.add(Expr::Mul(two, a));
                let shift = ctx.add(Expr::Div(b, two_a));
                let completed_binomial =
                    build_complete_square_binomial_expr(ctx, var_expr, shift, Sign::Pos);
                let square = ctx.add(Expr::Pow(completed_binomial, two));
                let scaled_square = ctx.add(Expr::Mul(a, square));
                let four_a = ctx.add(Expr::Mul(four, a));
                let correction = ctx.add(Expr::Div(b_squared, four_a));
                let tail = ctx.add(Expr::Sub(c, correction));
                let candidate_raw = ctx.add(Expr::Add(scaled_square, tail));
                let candidate = run_default_simplify(ctx, candidate_raw);
                return Some((candidate, a, SolvePrepBuildRoute::PosSymbolic));
            }

            profile_route("rule.solve_prep.build_candidate.route.pos_generic_scale");
            let two_a = ctx.add(Expr::Mul(two, a));
            let shift = ctx.add(Expr::Div(b, two_a));
            let completed_binomial =
                build_complete_square_binomial_expr(ctx, var_expr, shift, Sign::Pos);
            let square = ctx.add(Expr::Pow(completed_binomial, two));
            let scaled_square = ctx.add(Expr::Mul(a, square));

            let four_a = ctx.add(Expr::Mul(four, a));
            let correction = ctx.add(Expr::Div(b_squared, four_a));
            let tail = ctx.add(Expr::Sub(c, correction));
            let candidate_raw = ctx.add(Expr::Add(scaled_square, tail));
            let candidate = run_default_simplify(ctx, candidate_raw);
            (candidate, a, SolvePrepBuildRoute::PosGeneric)
        };

    Some((candidate_raw, nonzero_expr, build_route))
}

pub(super) fn is_sqrt_two_for_cancellation(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
        {
            extract_i64_integer(ctx, args[0]) == Some(2)
        }
        Expr::Pow(base, exp) => {
            extract_i64_integer(ctx, *base) == Some(2) && is_positive_one_half_expr(ctx, *exp)
        }
        _ => false,
    }
}

pub(super) fn divide_by_sqrt_two_fast_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let two = ctx.num(2);
    let sqrt_two = ctx.call_builtin(BuiltinFn::Sqrt, vec![two]);

    if extract_i64_integer(ctx, expr) == Some(1) {
        return Some(ctx.add(Expr::Div(sqrt_two, two)));
    }

    if let Some(stripped) = split_out_small_integer_factor_for_cancellation(ctx, expr, 2) {
        return Some(smart_mul(ctx, stripped, sqrt_two));
    }

    let scaled = smart_mul(ctx, expr, sqrt_two);
    Some(ctx.add(Expr::Div(scaled, two)))
}

pub(super) fn try_build_structural_difference_squares_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return None;
    }

    for target_index in 0..terms.len() {
        let (product_expr, product_sign) = terms[target_index];
        let Some(product_rewrite) =
            cas_math::factoring_support::try_rewrite_difference_of_squares_product_expr(
                ctx,
                product_expr,
            )
        else {
            continue;
        };
        let other_terms: Vec<_> = terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != target_index).then_some(term))
            .collect();
        let other_expr = build_signed_sum_expr(ctx, &other_terms);
        let expected_other = match product_sign {
            Sign::Neg => product_rewrite.rewritten,
            Sign::Pos => ctx.add(Expr::Neg(product_rewrite.rewritten)),
        };
        if exprs_match_for_cancellation(ctx, expected_other, other_expr) {
            return Some(
                Rewrite::with_local(
                    ctx.num(0),
                    "Difference of Squares",
                    expr,
                    ctx.num(0),
                )
                .substep(
                    "Reconocer diferencia de cuadrados",
                    vec![
                        "El producto conjugado coincide con la diferencia de cuadrados, así que el residuo vale 0.".to_string(),
                    ],
                ),
            );
        }
    }

    None
}

pub(super) fn try_build_small_odd_half_power_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let parent_ctx = ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rewrite = ExpandOddHalfPowerToEnableCancellationRule.apply(ctx, expr, &parent_ctx)?;
    let cancel_rewrite = SubSelfToZeroRule.apply(ctx, rewrite.new_expr, &parent_ctx)?;
    let zero = ctx.num(0);

    let mut exact_rewrite = Rewrite::with_local(zero, rewrite.description.clone(), expr, zero)
        .requires_all(rewrite.required_conditions.clone())
        .assume_all(rewrite.assumption_events.clone());
    if let Some(poly_proof) = rewrite.poly_proof.clone() {
        exact_rewrite = exact_rewrite.poly_proof(poly_proof);
    }
    exact_rewrite.substeps = rewrite.substeps.clone();
    exact_rewrite = exact_rewrite.substep(
        "Cancelar términos iguales",
        vec![
            "Tras reescribir la potencia semientera impar, la resta restante es exacta."
                .to_string(),
        ],
    );
    if let Some(poly_proof) = cancel_rewrite.poly_proof.clone() {
        exact_rewrite = exact_rewrite.poly_proof(poly_proof);
    }

    Some(exact_rewrite)
}

fn extract_square_power_base(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if extract_i64_integer(ctx, *exp) == Some(2) => Some(*base),
        _ => None,
    }
}

fn extract_difference_of_square_bases(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut positive_base = None;
    let mut negative_base = None;
    for (term, sign) in terms {
        let base = extract_square_power_base(ctx, term)?;
        match sign {
            Sign::Pos if positive_base.is_none() => positive_base = Some(base),
            Sign::Neg if negative_base.is_none() => negative_base = Some(base),
            _ => return None,
        }
    }

    Some((positive_base?, negative_base?))
}

pub(super) fn try_build_difference_of_equivalent_square_bases_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_base, rhs_base) = extract_difference_of_square_bases(ctx, expr)?;
    let child_rewrite =
        try_build_direct_sub_fraction_combination_equivalence_rewrite(ctx, lhs_base, rhs_base)
            .or_else(|| try_build_direct_core_equivalence_rewrite(ctx, lhs_base, rhs_base))
            .or_else(|| {
                let base_residual = ctx.add(Expr::Sub(lhs_base, rhs_base));
                try_build_exact_zero_identity_rewrite_direct(ctx, base_residual)
            })?;
    let zero = ctx.num(0);
    if compare_expr(ctx, child_rewrite.final_expr(), zero) != Ordering::Equal {
        return None;
    }

    let mut rewrite = Rewrite::with_local(zero, child_rewrite.description.clone(), expr, zero)
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone())
        .substep(
            "Reducir diferencia de cuadrados equivalentes",
            vec![
                "Los términos son cuadrados de bases equivalentes, así que la diferencia se anula."
                    .to_string(),
            ],
        );

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }
    rewrite.substeps.extend(child_rewrite.substeps.clone());
    Some(rewrite)
}

fn split_numeric_coefficient_for_square_shell(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (BigRational, cas_ast::ExprId) {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut non_numeric = Vec::new();

    for factor in factors {
        match ctx.get(factor).clone() {
            Expr::Number(value) => coefficient *= value,
            _ => non_numeric.push(factor),
        }
    }

    let residual = match non_numeric.as_slice() {
        [] => ctx.num(1),
        [single] => *single,
        _ => build_balanced_mul(ctx, &non_numeric),
    };

    (coefficient, residual)
}

fn build_rational_scaled_expr_for_square_shell(
    ctx: &mut cas_ast::Context,
    scale: BigRational,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if scale == BigRational::one() {
        expr
    } else {
        let scale_expr = ctx.add(Expr::Number(scale));
        smart_mul(ctx, scale_expr, expr)
    }
}

fn extract_scaled_square_base_for_shell(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (coefficient, residual) = split_numeric_coefficient_for_square_shell(ctx, expr);
    if !coefficient.is_positive() {
        return None;
    }
    let root = rational_sqrt(&coefficient)?;
    let base = extract_square_power_base(ctx, residual)?;
    Some(build_rational_scaled_expr_for_square_shell(ctx, root, base))
}

fn extract_expanded_binomial_square_base_for_shell(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let terms: Vec<_> = AddView::from_expr(ctx, expr)
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();
    if terms.len() != 3 {
        return None;
    }

    for first_index in 0..terms.len() {
        for second_index in (first_index + 1)..terms.len() {
            let cross_index =
                (0..terms.len()).find(|index| *index != first_index && *index != second_index)?;

            let (first_square, first_sign) = terms[first_index];
            let (second_square, second_sign) = terms[second_index];
            if first_sign != Sign::Pos || second_sign != Sign::Pos {
                continue;
            }

            let Some(first_base) = extract_scaled_square_base_for_shell(ctx, first_square) else {
                continue;
            };
            let Some(second_base) = extract_scaled_square_base_for_shell(ctx, second_square) else {
                continue;
            };

            let (cross_term, cross_sign) = terms[cross_index];
            let product = smart_mul(ctx, first_base, second_base);
            let two = ctx.num(2);
            let expected_cross = smart_mul(ctx, two, product);
            if !(exprs_match_for_cancellation(ctx, cross_term, expected_cross)
                || exprs_match_after_default_simplify(ctx, cross_term, expected_cross))
            {
                continue;
            }

            return Some(match cross_sign {
                Sign::Pos => ctx.add(Expr::Add(first_base, second_base)),
                Sign::Neg => ctx.add(Expr::Sub(first_base, second_base)),
            });
        }
    }

    None
}

fn extract_square_equivalence_base_for_shell(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    extract_scaled_square_base_for_shell(ctx, expr)
        .or_else(|| extract_expanded_binomial_square_base_for_shell(ctx, expr))
}

pub(super) fn try_match_symbolic_root_denesting_pair(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let outer_inner = extract_square_root_base(ctx, expr)?;
    let terms = AddView::from_expr(ctx, outer_inner).terms;
    if terms.len() != 2 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    for (plain_term, surd_term) in [(terms[0].0, terms[1].0), (terms[1].0, terms[0].0)] {
        let Some(delta) = extract_square_root_base(ctx, surd_term) else {
            continue;
        };
        let Some((square_base, other_base)) = extract_difference_of_square_bases(ctx, delta) else {
            continue;
        };
        if !exprs_match_for_cancellation(ctx, plain_term, square_base) {
            continue;
        }

        let plus_arg = ctx.add(Expr::Add(plain_term, other_base));
        let minus_arg = ctx.add(Expr::Sub(plain_term, other_base));
        return Some((plus_arg, minus_arg));
    }

    None
}

fn build_symbolic_root_denesting_target_expr(
    ctx: &mut cas_ast::Context,
    plus_arg: cas_ast::ExprId,
    minus_arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let sqrt_plus = ctx.call_builtin(BuiltinFn::Sqrt, vec![plus_arg]);
    let sqrt_minus = ctx.call_builtin(BuiltinFn::Sqrt, vec![minus_arg]);
    let numerator = ctx.add(Expr::Add(sqrt_plus, sqrt_minus));
    let two = ctx.num(2);
    let sqrt_two = ctx.call_builtin(BuiltinFn::Sqrt, vec![two]);
    ctx.add(Expr::Div(numerator, sqrt_two))
}

pub(super) fn try_build_small_symbolic_root_denesting_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let zero = ctx.num(0);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (root_side, target_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((plus_arg, minus_arg)) = try_match_symbolic_root_denesting_pair(ctx, root_side)
        else {
            continue;
        };
        let rewritten = build_symbolic_root_denesting_target_expr(ctx, plus_arg, minus_arg);
        if !(exprs_match_for_cancellation(ctx, rewritten, target_side)
            || exprs_match_after_default_simplify(ctx, rewritten, target_side))
        {
            continue;
        }

        return Some(
            Rewrite::with_local(zero, "Root Denesting", root_side, target_side)
                .requires(crate::ImplicitCondition::NonNegative(plus_arg))
                .requires(crate::ImplicitCondition::NonNegative(minus_arg))
                .substep(
                    "Denestar la raíz anidada",
                    vec![
                        "La forma sqrt(a + sqrt(a^2 - b^2)) coincide con (sqrt(a+b) + sqrt(a-b))/sqrt(2) cuando ambos radicandos lineales son no negativos."
                            .to_string(),
                    ],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras denestar la raíz, ambos lados coinciden exactamente y la diferencia vale 0."
                            .to_string(),
                    ],
                ),
        );
    }

    None
}

pub(crate) fn matches_direct_symbolic_root_denesting_zero_identity(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 || !view.terms.iter().any(|(_, sign)| *sign == Sign::Neg) {
        return false;
    }
    if !expr_contains_sqrt_or_half_power(ctx, expr) {
        return false;
    }

    try_build_small_symbolic_root_denesting_zero_core_rewrite(ctx, expr).is_some()
}

fn build_square_preserving_one(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if extract_i64_integer(ctx, expr) == Some(1) {
        expr
    } else {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(expr, two))
    }
}

pub(super) fn try_build_small_difference_square_partial_fraction_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_division_node(ctx, expr) {
        return None;
    }

    let positive_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Pos)
        .collect();
    let negative_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Neg)
        .collect();
    if positive_terms.len() != 1 || negative_terms.len() != 2 {
        return None;
    }

    let positive_denominator = extract_unit_fraction_denominator(ctx, positive_terms[0].0)?;
    let Expr::Sub(base, shift) = ctx.get(positive_denominator).clone() else {
        return None;
    };

    for (negative_unit, target_fraction) in [
        (negative_terms[0].0, negative_terms[1].0),
        (negative_terms[1].0, negative_terms[0].0),
    ] {
        let Some(negative_denominator) = extract_unit_fraction_denominator(ctx, negative_unit)
        else {
            continue;
        };
        if !positive_two_term_sum_matches_terms(ctx, negative_denominator, base, shift) {
            continue;
        }

        let Some((target_numerator, target_denominator)) = as_div(ctx, target_fraction) else {
            continue;
        };
        if !numerator_matches_two_times_shift(ctx, target_numerator, shift) {
            continue;
        }

        let two = ctx.num(2);
        let base_squared = ctx.add(Expr::Pow(base, two));
        let shift_squared = build_square_preserving_one(ctx, shift);
        let expected_denominator = ctx.add(Expr::Sub(base_squared, shift_squared));
        if !exprs_match_for_cancellation(ctx, target_denominator, expected_denominator) {
            continue;
        }

        return Some(
            Rewrite::with_local(ctx.num(0), "Subtract Fractions", expr, ctx.num(0))
                .requires(crate::ImplicitCondition::NonZero(positive_denominator))
                .requires(crate::ImplicitCondition::NonZero(negative_denominator))
                .requires(crate::ImplicitCondition::NonZero(target_denominator))
                .substep(
                    "Reconocer la diferencia de fracciones simétricas",
                    vec![
                        "La identidad 1/(u-a) - 1/(u+a) = 2a/(u^2-a^2) anula exactamente toda la combinación."
                            .to_string(),
                    ],
                ),
        );
    }

    None
}

pub(super) fn try_build_small_rationalized_sum_of_sqrts_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let zero = ctx.num(0);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    let extract_square_root_base_local =
        |ctx: &cas_ast::Context, candidate: cas_ast::ExprId| -> Option<cas_ast::ExprId> {
            match ctx.get(candidate) {
                Expr::Function(name, args)
                    if ctx.is_builtin(*name, BuiltinFn::Sqrt) && args.len() == 1 =>
                {
                    Some(args[0])
                }
                Expr::Pow(base, exp)
                    if matches!(
                        ctx.get(*exp),
                        Expr::Number(value)
                            if !value.is_integer()
                                && value.numer() == &BigInt::from(1)
                                && value.denom() == &BigInt::from(2)
                    ) =>
                {
                    Some(*base)
                }
                _ => None,
            }
        };

    for (fraction_side, target_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_sum_of_sqrts_den_expr(
                ctx,
                fraction_side,
            )
        else {
            continue;
        };

        if !(exprs_match_for_cancellation(ctx, rewrite.rewritten, target_side)
            || exprs_match_after_default_simplify(ctx, rewrite.rewritten, target_side))
        {
            continue;
        }

        let Some((_, source_denominator)) = as_div(ctx, fraction_side) else {
            continue;
        };
        let Some((_, target_denominator)) = as_div(ctx, target_side) else {
            continue;
        };
        let (left_sqrt, right_sqrt) = match ctx.get(source_denominator) {
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (*lhs, *rhs),
            _ => continue,
        };
        let Some(left_base) = extract_square_root_base_local(ctx, left_sqrt) else {
            continue;
        };
        let Some(right_base) = extract_square_root_base_local(ctx, right_sqrt) else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                zero,
                "Rationalize Sum of Sqrts Denominator",
                fraction_side,
                target_side,
            )
            .requires(crate::ImplicitCondition::NonNegative(left_base))
            .requires(crate::ImplicitCondition::NonNegative(right_base))
            .requires(crate::ImplicitCondition::NonZero(source_denominator))
            .requires(crate::ImplicitCondition::NonZero(target_denominator))
            .substep(
                "Racionalizar la fracción con radicales",
                vec![
                    "Al multiplicar por el conjugado, la fracción con radicales coincide exactamente con el otro término."
                        .to_string(),
                ],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Después de racionalizar, ambos lados son idénticos y la diferencia vale 0."
                        .to_string(),
                ],
            ),
        );
    }

    None
}

pub(super) fn try_rewrite_symbolic_difference_squares_telescoping_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (scale_factor, diff_factor) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        let Some((scale_num, scale_den)) = as_div(ctx, scale_factor) else {
            continue;
        };
        if extract_i64_integer(ctx, scale_num) != Some(1) {
            continue;
        }
        let Some(scale_arg) = extract_two_times_factor_arg(ctx, scale_den) else {
            continue;
        };

        let (left, right) = match ctx.get(diff_factor).clone() {
            Expr::Sub(lhs, rhs) => (lhs, rhs),
            Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
                Expr::Neg(inner) => (lhs, inner),
                _ => continue,
            },
            _ => continue,
        };

        let parts = extract_fraction_pair(ctx, left, right);
        if !parts.is_frac1
            || !parts.is_frac2
            || extract_i64_integer(ctx, parts.n1) != Some(1)
            || extract_i64_integer(ctx, parts.n2) != Some(1)
        {
            continue;
        }

        let Expr::Sub(base_arg, shift_arg) = ctx.get(parts.d1).clone() else {
            continue;
        };
        if !positive_two_term_sum_matches_terms(ctx, parts.d2, base_arg, shift_arg) {
            continue;
        }
        if !exprs_match_for_cancellation(ctx, scale_arg, shift_arg) {
            continue;
        }

        let one = ctx.num(1);
        let two = ctx.num(2);
        let base_sq = ctx.add(Expr::Pow(base_arg, two));
        let shift_sq = ctx.add(Expr::Pow(shift_arg, two));
        let denominator = ctx.add(Expr::Sub(base_sq, shift_sq));
        return Some(ctx.add(Expr::Div(one, denominator)));
    }

    None
}

pub(crate) fn try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(num, den) = ctx.get(source).clone() else {
            continue;
        };
        let Some(plan) = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
            ctx, num, den, false,
        ) else {
            continue;
        };

        let cancelled = canonicalize_nested_integer_powers(ctx, plan.cancelled_result);
        let target = canonicalize_nested_integer_powers(ctx, target);
        if !(cas_math::expr_domain::exprs_equivalent(ctx, cancelled, target)
            || exprs_equal_up_to_add_term_order(ctx, cancelled, target))
        {
            continue;
        }

        let identity_title = match ctx.get(num) {
            Expr::Sub(_, _) => "Usar a^3 - b^3 = (a - b)(a^2 + ab + b^2)",
            Expr::Add(_, rhs) if matches!(ctx.get(*rhs), Expr::Neg(_)) => {
                "Usar a^3 - b^3 = (a - b)(a^2 + ab + b^2)"
            }
            _ => "Usar a^3 + b^3 = (a + b)(a^2 - ab + b^2)",
        };

        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            "Subtract Expanded Sum/Difference of Cubes Quotient",
            lhs_core,
            rhs_core,
        )
        .requires(crate::ImplicitCondition::NonZero(den));
        rewrite.substeps = vec![
            crate::step::SubStep::new(identity_title, vec![]),
            crate::step::SubStep::new(
                "Cancelar el factor común del numerador y el denominador",
                vec![],
            ),
        ];
        return Some(rewrite);
    }

    None
}

pub(super) fn reciprocal_half_power_base(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    (exponent == BigRational::new((-1).into(), 2.into())).then_some(*base)
}

pub(super) fn negative_even_root_power_parts(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, BigRational)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    if !exponent.is_negative() || !exponent.denom().is_even() {
        return None;
    }

    Some((*base, -exponent))
}

fn positive_even_root_power_parts(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, BigRational)> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
        if !exponent.is_positive() || !exponent.denom().is_even() {
            return None;
        }

        return Some((*base, exponent));
    }

    let radicand = extract_square_root_base(ctx, expr)?;
    if let Expr::Pow(base, exp) = ctx.get(radicand) {
        let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
        if !exponent.is_positive() || !exponent.is_integer() {
            return None;
        }
        return Some((*base, exponent / BigRational::from_integer(2.into())));
    }

    Some((radicand, BigRational::new(1.into(), 2.into())))
}

fn split_negative_even_root_power_factor_from_product(
    ctx: &mut cas_ast::Context,
    product: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, BigRational)> {
    if let Some((base, exponent)) = negative_even_root_power_parts(ctx, product) {
        return Some((ctx.num(1), base, exponent));
    }

    let view = MulView::from_expr(ctx, product);
    for (index, factor) in view.factors.iter().copied().enumerate() {
        let Some((base, exponent)) = negative_even_root_power_parts(ctx, factor) else {
            continue;
        };

        let remaining_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        return Some((
            cas_math::expr_nary::build_balanced_mul(ctx, &remaining_factors),
            base,
            exponent,
        ));
    }

    None
}

fn positive_even_root_power_reciprocal_parts(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, BigRational)> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let (base, exponent) = positive_even_root_power_parts(ctx, denominator)?;
    Some((numerator, base, exponent))
}

fn try_match_negative_even_root_power_reciprocal_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (source_cofactor, source_base, source_exponent) =
        split_negative_even_root_power_factor_from_product(ctx, source)?;
    let (target_cofactor, target_base, target_exponent) =
        positive_even_root_power_reciprocal_parts(ctx, target)?;

    if source_exponent != target_exponent
        || !exprs_match_for_cancellation(ctx, source_base, target_base)
        || !exprs_match_for_cancellation(ctx, source_cofactor, target_cofactor)
    {
        let shifted_source_exponent = source_exponent.clone() + BigRational::one();
        if shifted_source_exponent != target_exponent
            || !exprs_match_for_cancellation(ctx, source_base, target_base)
        {
            return None;
        }

        let target_cofactor_over_base = ctx.add(Expr::Div(target_cofactor, source_base));
        if !exprs_match_for_cancellation(ctx, source_cofactor, target_cofactor_over_base)
            && try_match_direct_trig_ratio_equivalence(
                ctx,
                target_cofactor_over_base,
                source_cofactor,
            )
            .is_none()
        {
            return None;
        }
    }

    Some(source_base)
}

pub(super) fn try_build_direct_negative_even_root_power_reciprocal_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) =
            try_match_negative_even_root_power_reciprocal_equivalence(ctx, source, target)
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Negative Even-Root Power Reciprocal Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(base)),
        );
    }

    None
}

fn split_sqrt_factor_from_product(
    ctx: &mut cas_ast::Context,
    product: cas_ast::ExprId,
    expected_base: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let view = MulView::from_expr(ctx, product);
    for (index, factor) in view.factors.iter().copied().enumerate() {
        let Some(base) = extract_square_root_base(ctx, factor) else {
            continue;
        };
        if !exprs_match_for_cancellation(ctx, base, expected_base) {
            continue;
        }

        let remaining_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        return Some((
            factor,
            cas_math::expr_nary::build_balanced_mul(ctx, &remaining_factors),
        ));
    }

    None
}

fn sqrt_like_half_power_base(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(base) = extract_square_root_base(ctx, expr) {
        return Some(base);
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    (exponent == BigRational::new(1.into(), 2.into())).then_some(*base)
}

fn scaled_reciprocal_sqrt_quotient_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let base = sqrt_like_half_power_base(ctx, denominator)?;
    Some((base, numerator))
}

fn split_any_sqrt_factor_from_product(
    ctx: &mut cas_ast::Context,
    product: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if let Some(base) = sqrt_like_half_power_base(ctx, product) {
        return Some((base, ctx.num(1)));
    }

    let view = MulView::from_expr(ctx, product);
    for (index, factor) in view.factors.iter().copied().enumerate() {
        let Some(base) = sqrt_like_half_power_base(ctx, factor) else {
            continue;
        };

        let remaining_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        return Some((base, build_scale_from_factors(ctx, &remaining_factors)));
    }

    None
}

fn expand_scaled_additive_product_factors_for_half_power_match(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let mut expanded_factors = Vec::new();
    for factor in flatten_mul_chain(ctx, expr) {
        if let Some((common_factor, residual_expr)) =
            extract_common_multiplicative_residual_sum(ctx, factor)
        {
            expanded_factors.extend(flatten_mul_chain(ctx, common_factor));
            expanded_factors.push(residual_expr);
        } else {
            expanded_factors.push(factor);
        }
    }

    build_mul_expr_from_factors(ctx, &expanded_factors)
}

fn scaled_denominator_quotients_match_for_half_power_cancellation(
    ctx: &mut cas_ast::Context,
    source_scale: cas_ast::ExprId,
    source_denominator: cas_ast::ExprId,
    target_scale: cas_ast::ExprId,
    target_denominator: cas_ast::ExprId,
) -> bool {
    if exprs_match_for_cancellation(ctx, source_scale, target_scale)
        && exprs_match_for_cancellation(ctx, source_denominator, target_denominator)
    {
        return true;
    }

    let source_cross = smart_mul(ctx, source_scale, target_denominator);
    let target_cross = smart_mul(ctx, target_scale, source_denominator);
    if exprs_match_for_cancellation(ctx, source_cross, target_cross) {
        return true;
    }

    let source_expanded =
        expand_scaled_additive_product_factors_for_half_power_match(ctx, source_cross);
    let target_expanded =
        expand_scaled_additive_product_factors_for_half_power_match(ctx, target_cross);
    exprs_match_for_cancellation(ctx, source_expanded, target_expanded)
}

fn try_match_reciprocal_half_power_shared_denominator(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let (source_numerator, source_denominator) = as_div(ctx, source)?;
    let base = reciprocal_half_power_base(ctx, source_numerator)?;
    let (target_numerator, target_denominator) = as_div(ctx, target)?;
    if extract_i64_integer(ctx, target_numerator) != Some(1) {
        return None;
    }

    let (sqrt_factor, remaining_denominator) =
        split_sqrt_factor_from_product(ctx, target_denominator, base)?;
    if !exprs_match_for_cancellation(ctx, source_denominator, remaining_denominator) {
        return None;
    }

    Some((base, sqrt_factor))
}

pub(super) fn try_build_direct_reciprocal_half_power_shared_denominator_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((base, sqrt_factor)) =
            try_match_reciprocal_half_power_shared_denominator(ctx, source, target)
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Reciprocal Half-Power Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(base))
            .requires(crate::ImplicitCondition::NonZero(sqrt_factor)),
        );
    }

    None
}

fn fraction_like_division_parts_for_half_power_match(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if let Some((numerator, denominator)) = as_div(ctx, expr) {
        return Some((numerator, denominator));
    }

    let view = MulView::from_expr(ctx, expr);
    if view.factors.len() < 2 {
        return None;
    }

    let mut division_parts = None;
    let mut numerator_factors = Vec::new();
    for factor in view.factors.iter().copied() {
        if let Some((factor_numerator, factor_denominator)) = as_div(ctx, factor) {
            if division_parts
                .replace((factor_numerator, factor_denominator))
                .is_some()
            {
                return None;
            }
        } else {
            numerator_factors.push(factor);
        }
    }

    let (factor_numerator, denominator) = division_parts?;
    numerator_factors.push(factor_numerator);
    Some((
        build_mul_expr_from_factors(ctx, &numerator_factors),
        denominator,
    ))
}

fn scaled_sqrt_denominator_division_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let (numerator, denominator) = fraction_like_division_parts_for_half_power_match(ctx, expr)?;
    let (base, denominator_without_sqrt) = split_any_sqrt_factor_from_product(ctx, denominator)?;
    Some((base, numerator, denominator_without_sqrt))
}

fn scaled_reciprocal_half_power_division_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let (numerator, denominator) = fraction_like_division_parts_for_half_power_match(ctx, expr)?;
    if let Some(base) = reciprocal_half_power_base(ctx, numerator) {
        return Some((base, ctx.num(1), denominator));
    }
    if let Some((base, scale)) = scaled_reciprocal_sqrt_quotient_parts(ctx, numerator) {
        return Some((base, scale, denominator));
    }

    let (base, scale) = split_reciprocal_half_power_factor_from_product(ctx, numerator)?;
    Some((base, scale, denominator))
}

fn try_match_scaled_reciprocal_half_power_shared_denominator_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (source_base, source_scale, source_denominator) =
        scaled_reciprocal_half_power_division_parts(ctx, source)?;
    let (target_base, target_scale, target_denominator) =
        scaled_sqrt_denominator_division_parts(ctx, target)?;

    if !exprs_match_for_cancellation(ctx, source_base, target_base) {
        return None;
    }
    if !scaled_denominator_quotients_match_for_half_power_cancellation(
        ctx,
        source_scale,
        source_denominator,
        target_scale,
        target_denominator,
    ) {
        return None;
    }

    Some(source_base)
}

pub(super) fn try_build_direct_scaled_reciprocal_half_power_shared_denominator_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) = try_match_scaled_reciprocal_half_power_shared_denominator_equivalence(
            ctx, source, target,
        ) else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Reciprocal Half-Power Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(base)),
        );
    }

    None
}

fn scaled_sqrt_over_base_division_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let (numerator, denominator) = fraction_like_division_parts_for_half_power_match(ctx, expr)?;
    let (base, scale) = split_any_sqrt_factor_from_product(ctx, numerator)?;
    let denominator_without_base = split_matching_factor_from_product(ctx, denominator, base)?;
    Some((base, scale, denominator_without_base))
}

fn try_match_scaled_reciprocal_half_power_over_base_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (source_base, source_scale, source_denominator) =
        scaled_reciprocal_half_power_division_parts(ctx, source)?;
    let (target_base, target_scale, target_denominator) =
        scaled_sqrt_over_base_division_parts(ctx, target)?;

    if !exprs_match_for_cancellation(ctx, source_base, target_base) {
        return None;
    }
    if !scaled_denominator_quotients_match_for_half_power_cancellation(
        ctx,
        source_scale,
        source_denominator,
        target_scale,
        target_denominator,
    ) {
        return None;
    }

    Some(source_base)
}

pub(super) fn try_build_direct_scaled_reciprocal_half_power_over_base_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) =
            try_match_scaled_reciprocal_half_power_over_base_equivalence(ctx, source, target)
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Reciprocal Half-Power Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(base)),
        );
    }

    None
}

fn reciprocal_half_power_product_bases(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<[cas_ast::ExprId; 2]> {
    let view = MulView::from_expr(ctx, expr);
    if view.factors.len() != 2 {
        return None;
    }

    Some([
        reciprocal_half_power_base(ctx, view.factors[0])?,
        reciprocal_half_power_base(ctx, view.factors[1])?,
    ])
}

fn reciprocal_half_power_product_base_pair(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<[cas_ast::ExprId; 2]> {
    let product_base = reciprocal_half_power_base(ctx, expr)?;
    let view = MulView::from_expr(ctx, product_base);
    if view.factors.len() != 2 {
        return None;
    }

    Some([view.factors[0], view.factors[1]])
}

fn reciprocal_sqrt_product_denominator_bases(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<[cas_ast::ExprId; 2]> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    if extract_i64_integer(ctx, numerator) != Some(1) {
        return None;
    }

    let view = MulView::from_expr(ctx, denominator);
    if view.factors.len() != 2 {
        return None;
    }

    Some([
        extract_square_root_base(ctx, view.factors[0])?,
        extract_square_root_base(ctx, view.factors[1])?,
    ])
}

fn scaled_reciprocal_sqrt_product_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, [cas_ast::ExprId; 2])> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);

    if let Some(bases) = reciprocal_half_power_product_base_pair(ctx, expr) {
        let scale = ctx.num(1);
        return Some((scale, bases));
    }

    if let Some((numerator, denominator)) = as_div(ctx, expr) {
        let mut bases = Vec::new();
        let mut scale_denominator_factors = Vec::new();
        for factor in MulView::from_expr(ctx, denominator).factors {
            let Some(base) = extract_square_root_base(ctx, factor) else {
                scale_denominator_factors.push(factor);
                continue;
            };
            let base_factors = MulView::from_expr(ctx, base).factors;
            if base_factors.len() == 2 {
                bases.extend(base_factors);
            } else {
                bases.push(base);
            }
        }
        if bases.len() == 2 {
            let scale = if scale_denominator_factors.is_empty() {
                numerator
            } else {
                let scale_denominator = build_balanced_mul(ctx, &scale_denominator_factors);
                ctx.add(Expr::Div(numerator, scale_denominator))
            };
            return Some((scale, [bases[0], bases[1]]));
        }
    }

    let view = MulView::from_expr(ctx, expr);
    for (index, factor) in view.factors.iter().copied().enumerate() {
        if factor == expr {
            continue;
        }
        let Some((factor_scale, bases)) = scaled_reciprocal_sqrt_product_parts(ctx, factor) else {
            continue;
        };
        let mut scale_factors = Vec::with_capacity(view.factors.len());
        scale_factors.push(factor_scale);
        scale_factors.extend(
            view.factors
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor)),
        );
        let scale = build_scale_from_factors(ctx, &scale_factors);
        return Some((scale, bases));
    }

    for (index, factor) in view.factors.iter().copied().enumerate() {
        let Some(product_base) = reciprocal_half_power_base(ctx, factor) else {
            continue;
        };
        let product_factors = MulView::from_expr(ctx, product_base).factors;
        if product_factors.len() != 2 {
            continue;
        }
        let scale_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        let scale = build_scale_from_factors(ctx, &scale_factors);
        return Some((scale, [product_factors[0], product_factors[1]]));
    }

    let reciprocal_indices_and_bases: Vec<_> = view
        .factors
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, factor)| {
            reciprocal_half_power_base(ctx, factor).map(|base| (index, base))
        })
        .collect();
    if reciprocal_indices_and_bases.len() == 2 {
        let scale_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| {
                (!reciprocal_indices_and_bases
                    .iter()
                    .any(|(index, _)| *index == factor_index))
                .then_some(factor)
            })
            .collect();
        let scale = build_scale_from_factors(ctx, &scale_factors);
        return Some((
            scale,
            [
                reciprocal_indices_and_bases[0].1,
                reciprocal_indices_and_bases[1].1,
            ],
        ));
    }

    None
}

pub(super) fn try_build_direct_scaled_reciprocal_half_power_product_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_scale, lhs_bases) = scaled_reciprocal_sqrt_product_parts(ctx, lhs_core)?;
    let (rhs_scale, rhs_bases) = scaled_reciprocal_sqrt_product_parts(ctx, rhs_core)?;
    if !exprs_match_for_cancellation(ctx, lhs_scale, rhs_scale)
        || !reciprocal_half_power_base_pairs_match(ctx, lhs_bases, rhs_bases)
    {
        return None;
    }

    Some(
        Rewrite::with_local(
            ctx.num(0),
            "Reciprocal Half-Power Product Cancellation",
            lhs_core,
            rhs_core,
        )
        .requires(crate::ImplicitCondition::Positive(lhs_bases[0]))
        .requires(crate::ImplicitCondition::Positive(lhs_bases[1])),
    )
}

fn reciprocal_half_power_quotient_product_one_bases(
    ctx: &mut cas_ast::Context,
    product: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let product = cas_ast::hold::unwrap_internal_hold(ctx, product);
    let mut quotient_bases = None;
    let mut reciprocal_base = None;
    let mut positive_half_base = None;

    for factor in MulView::from_expr(ctx, product).factors {
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some_and(|value| value.is_one()) {
            continue;
        }

        if let Some(base) = reciprocal_half_power_base(ctx, factor) {
            if let Some((numerator, denominator)) = as_div(ctx, base) {
                if quotient_bases.replace((numerator, denominator)).is_some() {
                    return None;
                }
            } else if reciprocal_base.replace(base).is_some() {
                return None;
            }
            continue;
        }

        if let Some(base) = sqrt_like_half_power_base(ctx, factor) {
            if positive_half_base.replace(base).is_some() {
                return None;
            }
            continue;
        }

        return None;
    }

    let (quotient_numerator, quotient_denominator) = quotient_bases?;
    let reciprocal_base = reciprocal_base?;
    let positive_half_base = positive_half_base?;
    if !reciprocal_half_power_base_matches(ctx, reciprocal_base, quotient_denominator)
        || !reciprocal_half_power_base_matches(ctx, positive_half_base, quotient_numerator)
    {
        return None;
    }

    Some((quotient_numerator, quotient_denominator))
}

pub(super) fn try_build_direct_reciprocal_half_power_quotient_product_one_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (product, one) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(one_value) = cas_ast::views::as_rational_const(ctx, one, 8) else {
            continue;
        };
        if !one_value.is_one() {
            continue;
        }

        let Some((quotient_numerator, quotient_denominator)) =
            reciprocal_half_power_quotient_product_one_bases(ctx, product)
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Reciprocal Half-Power Quotient Product Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(quotient_numerator))
            .requires(crate::ImplicitCondition::Positive(quotient_denominator)),
        );
    }

    None
}

fn sqrt_over_base_times_base_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let (numerator, denominator) = signed_division_parts(ctx, expr)?;
    let sqrt_base = sqrt_like_half_power_base(ctx, numerator)?;
    let remaining_denominator = split_matching_factor_from_product(ctx, denominator, sqrt_base)?;
    Some((sqrt_base, remaining_denominator))
}

fn try_match_reciprocal_half_power_quotient_over_base_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let (quotient_numerator, quotient_denominator) = sqrt_over_base_times_base_parts(ctx, source)?;
    let (target_quotient, target_denominator) = reciprocal_three_half_quotient_parts(ctx, target)?;
    if !reciprocal_half_power_base_matches(ctx, quotient_denominator, target_denominator) {
        return None;
    }

    let expected_quotient = ctx.add(Expr::Div(quotient_numerator, quotient_denominator));
    if !reciprocal_half_power_base_matches(ctx, target_quotient, expected_quotient)
        && !shifted_unit_fraction_quotient_matches(
            ctx,
            target_quotient,
            quotient_numerator,
            quotient_denominator,
        )
    {
        return None;
    }

    Some((quotient_numerator, quotient_denominator))
}

pub(super) fn try_build_direct_reciprocal_half_power_quotient_over_base_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((quotient_numerator, quotient_denominator)) =
            try_match_reciprocal_half_power_quotient_over_base_equivalence(ctx, source, target)
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Reciprocal Half-Power Quotient/Base Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(quotient_numerator))
            .requires(crate::ImplicitCondition::Positive(quotient_denominator)),
        );
    }

    None
}

fn add_matching_base_exponent(
    ctx: &mut cas_ast::Context,
    current_base: &mut Option<cas_ast::ExprId>,
    exponent_sum: &mut BigRational,
    candidate_base: cas_ast::ExprId,
    candidate_exponent: BigRational,
) -> Option<()> {
    if let Some(base) = *current_base {
        if !exprs_match_for_cancellation(ctx, base, candidate_base) {
            return None;
        }
    } else {
        *current_base = Some(candidate_base);
    }

    *exponent_sum += candidate_exponent;
    Some(())
}

fn denominator_three_half_power_base_and_scale(
    ctx: &mut cas_ast::Context,
    denominator: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, BigRational)> {
    let mut scale = BigRational::one();
    let mut base = None;
    let mut exponent_sum = BigRational::zero();

    for factor in MulView::from_expr(ctx, denominator).factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            if value.is_zero() {
                return None;
            }
            scale *= value;
            continue;
        }

        if let Some((factor_base, exponent)) = positive_even_root_power_parts(ctx, factor) {
            add_matching_base_exponent(ctx, &mut base, &mut exponent_sum, factor_base, exponent)?;
            continue;
        }

        add_matching_base_exponent(
            ctx,
            &mut base,
            &mut exponent_sum,
            factor,
            BigRational::one(),
        )?;
    }

    (exponent_sum == BigRational::new(3.into(), 2.into())).then_some((base?, scale))
}

fn scale_expr_by_rational_for_half_power_residual(
    ctx: &mut cas_ast::Context,
    scale: BigRational,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if scale.is_zero() {
        return ctx.num(0);
    }
    if scale == BigRational::one() {
        return expr;
    }
    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(expr));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    smart_mul(ctx, scale_expr, expr)
}

fn scale_half_power_residual_numerator(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    scale: BigRational,
) -> cas_ast::ExprId {
    scale_expr_by_rational_for_half_power_residual(ctx, scale, numerator)
}

fn reciprocal_half_power_linear_residual_term_numerator(
    ctx: &mut cas_ast::Context,
    term: cas_ast::ExprId,
    sign: Sign,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    let mut term_sign = BigRational::from_integer(BigInt::from(sign_to_i64(sign)));

    if let Some((numerator, denominator)) = as_div(ctx, term) {
        if let Some((base, denominator_scale)) =
            denominator_three_half_power_base_and_scale(ctx, denominator)
        {
            term_sign /= denominator_scale;
            let numerator = scale_half_power_residual_numerator(ctx, numerator, term_sign.clone());
            return Some((base, numerator));
        }

        if let Some(denominator_scale) = cas_ast::views::as_rational_const(ctx, denominator, 8) {
            if denominator_scale.is_zero() {
                return None;
            }
            let (base, numerator) =
                reciprocal_half_power_linear_residual_term_numerator(ctx, numerator, sign)?;
            let numerator = scale_half_power_residual_numerator(
                ctx,
                numerator,
                BigRational::one() / denominator_scale,
            );
            return Some((base, numerator));
        }
    }

    let mut scale = term_sign;
    let mut root_base = None;
    let mut root_exponent = None;
    let mut numerator_factors = Vec::new();

    for factor in MulView::from_expr(ctx, term).factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }

        if let Some((base, exponent)) = negative_even_root_power_parts(ctx, factor) {
            if root_base.replace(base).is_some() || root_exponent.replace(exponent).is_some() {
                return None;
            }
            continue;
        }

        numerator_factors.push(factor);
    }

    let root_base = root_base?;
    let root_exponent = root_exponent?;
    if root_exponent == BigRational::new(1.into(), 2.into()) {
        numerator_factors.push(root_base);
    } else if root_exponent != BigRational::new(3.into(), 2.into()) {
        return None;
    }

    let numerator = build_scale_from_factors(ctx, &numerator_factors);
    Some((
        root_base,
        scale_half_power_residual_numerator(ctx, numerator, scale),
    ))
}

pub(super) fn try_build_direct_reciprocal_half_power_linear_residual_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 3 || view.terms.len() > 4 {
        return None;
    }

    let mut common_base = None;
    let mut numerator_terms = Vec::with_capacity(view.terms.len());
    for (term, sign) in view.terms {
        let (base, numerator) =
            reciprocal_half_power_linear_residual_term_numerator(ctx, term, sign)?;
        if let Some(existing_base) = common_base {
            if !exprs_match_for_cancellation(ctx, existing_base, base) {
                return None;
            }
        } else {
            common_base = Some(base);
        }
        numerator_terms.push(numerator);
    }

    let common_base = common_base?;
    let numerator_sum = build_balanced_add(ctx, &numerator_terms);
    let zero = ctx.num(0);
    let numerator_is_zero = exprs_match_after_default_simplify(ctx, numerator_sum, zero)
        || try_build_fast_small_polynomial_expansion_zero_scope_rewrite(ctx, numerator_sum)
            .is_some()
        || [2_i64, 3, 4, 6, 8, 12].iter().copied().any(|scale| {
            let scale_expr = ctx.num(scale);
            let scaled_numerator_sum = smart_mul(ctx, scale_expr, numerator_sum);
            exprs_match_after_default_simplify(ctx, scaled_numerator_sum, zero)
                || try_build_fast_small_polynomial_expansion_zero_scope_rewrite(
                    ctx,
                    scaled_numerator_sum,
                )
                .is_some()
        });
    if !numerator_is_zero {
        return None;
    }

    Some(
        Rewrite::with_local(
            zero,
            "Reciprocal Half-Power Linear Residual Cancellation",
            expr,
            zero,
        )
        .requires(crate::ImplicitCondition::Positive(common_base)),
    )
}

pub(super) fn try_build_direct_common_sqrt_denominator_fraction_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_numerator, lhs_denominator) = signed_division_parts(ctx, lhs_core)?;
    let (rhs_numerator, rhs_denominator) = signed_division_parts(ctx, rhs_core)?;
    if !exprs_match_for_cancellation(ctx, lhs_numerator, rhs_numerator) {
        return None;
    }

    let base =
        denominators_match_after_common_sqrt_factor_pull(ctx, lhs_denominator, rhs_denominator)?;
    Some(
        Rewrite::with_local(
            ctx.num(0),
            "Sqrt Denominator Factor Cancellation",
            lhs_core,
            rhs_core,
        )
        .requires(crate::ImplicitCondition::Positive(base)),
    )
}

fn pull_common_sqrt_factor_from_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    split_common_sqrt_factor_from_term(ctx, expr)
        .or_else(|| pull_common_sqrt_factor_from_additive_terms(ctx, expr))
}

fn try_match_sqrt_over_base_fraction_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (source_numerator, source_denominator) = signed_division_parts(ctx, source)?;
    let (target_numerator, target_denominator) = signed_division_parts(ctx, target)?;
    let (base, target_residual_numerator) =
        pull_common_sqrt_factor_from_expr(ctx, target_numerator)?;
    let target_denominator_without_base =
        split_matching_factor_from_product(ctx, target_denominator, base)?;
    let source_denominator_scale = split_matching_factor_from_product(
        ctx,
        target_denominator_without_base,
        source_denominator,
    )?;

    let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    let scaled_source_numerator = build_balanced_mul(
        ctx,
        &[source_numerator, source_denominator_scale, sqrt_base],
    );
    if !exprs_match_for_cancellation(ctx, scaled_source_numerator, target_residual_numerator)
        && !exprs_match_after_default_simplify(
            ctx,
            scaled_source_numerator,
            target_residual_numerator,
        )
    {
        return None;
    }

    Some(base)
}

pub(super) fn try_build_direct_sqrt_over_base_fraction_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) = try_match_sqrt_over_base_fraction_equivalence(ctx, source, target) else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Sqrt/Base Fraction Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(base)),
        );
    }

    None
}

fn try_match_rationalized_common_sqrt_denominator_fraction_equivalence(
    ctx: &mut cas_ast::Context,
    simple_fraction: cas_ast::ExprId,
    rationalized_fraction: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (simple_numerator, simple_denominator) = signed_division_parts(ctx, simple_fraction)?;
    let (rationalized_numerator, rationalized_denominator) =
        signed_division_parts(ctx, rationalized_fraction)?;

    let conjugate_factor =
        split_matching_factor_from_product(ctx, rationalized_numerator, simple_numerator)?;
    let (base, conjugate_residual) =
        pull_common_sqrt_factor_from_additive_terms(ctx, conjugate_factor)?;
    let (_sqrt_factor, simple_tail) =
        split_sqrt_factor_from_product(ctx, simple_denominator, base)?;
    let expected_rationalized_denominator =
        build_balanced_mul(ctx, &[base, conjugate_residual, simple_tail]);

    if exprs_match_for_cancellation(
        ctx,
        expected_rationalized_denominator,
        rationalized_denominator,
    ) || exprs_match_after_default_simplify(
        ctx,
        expected_rationalized_denominator,
        rationalized_denominator,
    ) {
        return Some(base);
    }

    None
}

pub(super) fn try_build_direct_rationalized_common_sqrt_denominator_fraction_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (simple_fraction, rationalized_fraction) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) = try_match_rationalized_common_sqrt_denominator_fraction_equivalence(
            ctx,
            simple_fraction,
            rationalized_fraction,
        ) else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Rationalized Sqrt Denominator Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(base)),
        );
    }

    None
}

fn split_reciprocal_half_power_factor_from_product(
    ctx: &mut cas_ast::Context,
    product: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let view = MulView::from_expr(ctx, product);
    for (index, factor) in view.factors.iter().copied().enumerate() {
        let base_and_scale = if let Some(base) = reciprocal_half_power_base(ctx, factor) {
            Some((base, ctx.num(1)))
        } else {
            scaled_reciprocal_sqrt_quotient_parts(ctx, factor)
        };
        let Some((base, factor_scale)) = base_and_scale else {
            continue;
        };

        let mut remaining_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        let one = ctx.num(1);
        if compare_expr(ctx, factor_scale, one) != Ordering::Equal {
            remaining_factors.push(factor_scale);
        }
        return Some((
            base,
            cas_math::expr_nary::build_balanced_mul(ctx, &remaining_factors),
        ));
    }

    None
}

fn reciprocal_half_power_over_sqrt_bases(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<[cas_ast::ExprId; 2]> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    if let (Some(half_base), Some(sqrt_base)) = (
        reciprocal_half_power_base(ctx, numerator),
        extract_square_root_base(ctx, denominator),
    ) {
        return Some([half_base, sqrt_base]);
    }

    let (inner_numerator, inner_denominator) = as_div(ctx, numerator)?;
    let sqrt_base = extract_square_root_base(ctx, inner_denominator)?;
    let (half_base, scale) = split_reciprocal_half_power_factor_from_product(ctx, inner_numerator)?;
    if !exprs_match_for_cancellation(ctx, scale, denominator) {
        return None;
    }

    Some([half_base, sqrt_base])
}

fn reciprocal_half_power_base_pairs_match(
    ctx: &mut cas_ast::Context,
    source_bases: [cas_ast::ExprId; 2],
    target_bases: [cas_ast::ExprId; 2],
) -> bool {
    (reciprocal_half_power_base_matches(ctx, source_bases[0], target_bases[0])
        && reciprocal_half_power_base_matches(ctx, source_bases[1], target_bases[1]))
        || (reciprocal_half_power_base_matches(ctx, source_bases[0], target_bases[1])
            && reciprocal_half_power_base_matches(ctx, source_bases[1], target_bases[0]))
}

pub(super) fn reciprocal_half_power_base_matches(
    ctx: &mut cas_ast::Context,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
) -> bool {
    exprs_match_for_cancellation(ctx, left, right)
        || exprs_match_after_default_simplify(ctx, left, right)
}

fn try_match_reciprocal_half_power_product_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<[cas_ast::ExprId; 2]> {
    let source_bases = reciprocal_half_power_product_bases(ctx, source)
        .or_else(|| reciprocal_half_power_over_sqrt_bases(ctx, source))
        .or_else(|| reciprocal_sqrt_product_denominator_bases(ctx, source))?;
    let target_bases = reciprocal_half_power_product_base_pair(ctx, target)
        .or_else(|| reciprocal_sqrt_product_denominator_bases(ctx, target))?;
    reciprocal_half_power_base_pairs_match(ctx, source_bases, target_bases).then_some(source_bases)
}

pub(super) fn try_build_direct_reciprocal_half_power_product_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some([first_base, second_base]) =
            try_match_reciprocal_half_power_product_equivalence(ctx, source, target)
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Reciprocal Half-Power Product Cancellation",
                lhs_core,
                rhs_core,
            )
            .requires(crate::ImplicitCondition::Positive(first_base))
            .requires(crate::ImplicitCondition::Positive(second_base)),
        );
    }

    None
}

fn build_exact_zero_squared_shared_passthrough_rewrite(
    ctx: &mut cas_ast::Context,
    whole_expr: cas_ast::ExprId,
    child_rewrite: Rewrite,
) -> Rewrite {
    let mut rewrite = Rewrite::with_local(
        ctx.num(0),
        child_rewrite.description.clone(),
        whole_expr,
        ctx.num(0),
    )
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone())
    .substep(
        "Pelar el wrapper cuadrático compartido",
        vec![
            "Ambos lados tienen la forma (core^2 + p)^2 con el mismo término de paso p."
                .to_string(),
        ],
    );

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }
    rewrite.substeps.extend(child_rewrite.substeps.clone());
    rewrite
}

pub(super) fn try_build_shared_passthrough_square_base_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let lhs_base = extract_square_equivalence_base_for_shell(ctx, lhs_core)?;
    let rhs_base = extract_square_equivalence_base_for_shell(ctx, rhs_core)?;

    let child_rewrite =
        try_build_direct_sub_fraction_combination_equivalence_rewrite(ctx, lhs_base, rhs_base)
            .or_else(|| try_build_direct_core_equivalence_rewrite(ctx, lhs_base, rhs_base))
            .or_else(|| {
                let base_residual = ctx.add(Expr::Sub(lhs_base, rhs_base));
                try_build_exact_zero_identity_rewrite_direct(ctx, base_residual)
            })?;
    let zero = ctx.num(0);
    if compare_expr(ctx, child_rewrite.final_expr(), zero) != Ordering::Equal {
        return None;
    }

    let mut rewrite =
        Rewrite::with_local(zero, child_rewrite.description.clone(), lhs_core, rhs_core)
            .requires_all(child_rewrite.required_conditions.clone())
            .assume_all(child_rewrite.assumption_events.clone())
            .substep(
                "Pelar el wrapper cuadrático compartido",
                vec![
            "Ambos cores tienen la forma base^2, así que basta probar la equivalencia de las bases."
                .to_string(),
        ],
            );

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }
    rewrite.substeps.extend(child_rewrite.substeps.clone());
    Some(rewrite)
}

pub(super) fn try_build_exact_zero_squared_shared_passthrough_difference_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_outer, rhs_outer) = extract_two_term_core_difference(ctx, expr)?;
    let lhs_shell = extract_square_power_base(ctx, lhs_outer)?;
    let rhs_shell = extract_square_power_base(ctx, rhs_outer)?;

    let shell_difference = ctx.add(Expr::Sub(lhs_shell, rhs_shell));
    let (lhs_core, rhs_core) =
        extract_shared_additive_passthrough_difference_cores(ctx, shell_difference)?;

    let child_rewrite =
        try_build_direct_sub_fraction_combination_equivalence_rewrite(ctx, lhs_core, rhs_core)
            .or_else(|| {
                let lhs_base = extract_square_equivalence_base_for_shell(ctx, lhs_core)?;
                let rhs_base = extract_square_equivalence_base_for_shell(ctx, rhs_core)?;
                try_build_direct_sub_fraction_combination_equivalence_rewrite(
                    ctx, lhs_base, rhs_base,
                )
                .or_else(|| try_build_direct_core_equivalence_rewrite(ctx, lhs_base, rhs_base))
                .or_else(|| {
                    let base_residual = ctx.add(Expr::Sub(lhs_base, rhs_base));
                    try_build_exact_zero_identity_rewrite_direct(ctx, base_residual)
                })
            })
            .or_else(|| try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core))?;

    let zero = ctx.num(0);
    (compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal)
        .then(|| build_exact_zero_squared_shared_passthrough_rewrite(ctx, expr, child_rewrite))
}
