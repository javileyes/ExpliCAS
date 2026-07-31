//! `arithmetic`: familia `support`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn render_expr_for_orchestrator_profile(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> String {
    crate::orchestrator_shortcut_profiler::render_expr_shape_for_orchestrator_profile(ctx, expr)
}

pub(super) fn run_profiled_orchestrator_option_section<T>(
    name: &'static str,
    sample: Option<String>,
    body: impl FnOnce() -> Option<T>,
) -> Option<T> {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return body();
    }

    if let Some(sample) = sample {
        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(name, sample);
    }

    let start = Instant::now();
    let result = body();
    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_attempt(
        name,
        result.is_some(),
        start.elapsed(),
    );
    result
}

pub(super) fn expr_contains_any_builtin(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    builtins: &[BuiltinFn],
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if builtins
                    .iter()
                    .any(|builtin| ctx.is_builtin(*fn_id, *builtin))
                {
                    return true;
                }
                stack.extend(args.iter().copied());
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
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

pub(super) fn expr_contains_symbolic_atom_for_cancellation(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Variable(_) | Expr::SessionRef(_) => return true,
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) => {}
        }
    }

    false
}

pub(super) fn expr_contains_any_function_call(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(_, _) => {
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
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(super) fn apply_sign_to_expr(
    ctx: &mut cas_ast::Context,
    sign: i64,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if sign < 0 {
        ctx.add(Expr::Neg(expr))
    } else {
        expr
    }
}

pub(super) fn expr_contains_sqrt_or_half_power(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];
    let half = num_rational::BigRational::new(1.into(), 2.into());

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                return true;
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Pow(base, exp) => {
                if matches!(ctx.get(*exp), Expr::Number(n) if *n == half) {
                    return true;
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
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(super) fn default_simplify_nesting_depth() -> usize {
    DEFAULT_SIMPLIFY_NESTING.with(|depth| depth.get())
}

pub(super) fn run_default_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    struct DefaultSimplifyNestingGuard;

    impl Drop for DefaultSimplifyNestingGuard {
        fn drop(&mut self) {
            DEFAULT_SIMPLIFY_NESTING.with(|depth| {
                depth.set(depth.get().saturating_sub(1));
            });
        }
    }

    let nesting = DEFAULT_SIMPLIFY_NESTING.with(|depth| {
        let current = depth.get();
        depth.set(current + 1);
        current
    });
    let _nesting_guard = DefaultSimplifyNestingGuard;

    // Speculative exact-zero probes may nest at most TWO default
    // simplifies: observed successful matches happen at nesting 0-1
    // (the phase-shift quotient pair needs one nested probe inside its
    // full-pipeline probe); nesting 2-3 only burns CPU. The
    // double-angle/power-reduction probe pair otherwise regenerates
    // cos(4x)+1 one level deeper each round at x20-40 the work,
    // hanging sums like sin(x)^2 cos(x)^2 - sin(x)^4 indefinitely.
    if nesting >= 2 {
        return expr;
    }

    // Breadth cap: the subset-enumeration probes each launch a
    // simplify here; past the per-pipeline budget they fall back to
    // the syntactic fast path. Outside an armed pipeline scope (unit
    // contexts) the budget is inactive.
    // Memo hit: replay the earlier probe result without consuming budget or
    // nesting (each expr is served the strength of its FIRST probe, which the
    // decaying budget makes the strongest one it would ever get).
    let probe_value_domain = ambient_pipeline_value_domain();
    let memo_key = (ctx.instance_tag(), expr, probe_value_domain);
    if let Some(cached) =
        DEFAULT_SIMPLIFY_PROBE_MEMO.with(|memo| memo.borrow().get(&memo_key).copied())
    {
        return cached;
    }

    let mut force_local = false;
    match DEFAULT_SIMPLIFY_PROBES_LEFT.with(|left| left.get()) {
        Some(0) => return expr,
        Some(probes_left) => {
            DEFAULT_SIMPLIFY_PROBES_LEFT.with(|left| left.set(Some(probes_left - 1)));
            // Only the first FULL_PROBE_BUDGET probes may launch a
            // full fresh pipeline: a full pipeline per probe is what
            // turned the subset enumeration into a hang (16 probes x
            // 1-2s pipelines on sin^4 + cos^4 - 1 + 2 sin^2 cos^2).
            force_local =
                probes_left <= DEFAULT_SIMPLIFY_PROBE_BUDGET - DEFAULT_SIMPLIFY_FULL_PROBE_BUDGET;
        }
        None => {}
    }

    if nesting > 0 || force_local {
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.set_collect_steps(false);
        simplifier.set_sticky_value_domain(probe_value_domain);
        std::mem::swap(&mut simplifier.context, ctx);
        let pattern_marks = crate::pattern_marks::PatternMarks::new();
        let rewritten = crate::with_suppressed_depth_overflow_warnings(|| {
            let (core, _) = simplifier.local_simplify_with_phase(
                expr,
                &pattern_marks,
                crate::phase::SimplifyPhase::Core,
            );
            let (transform, _) = simplifier.local_simplify_with_phase(
                core,
                &pattern_marks,
                crate::phase::SimplifyPhase::Transform,
            );
            let (post, _) = simplifier.local_simplify_with_phase(
                transform,
                &pattern_marks,
                crate::phase::SimplifyPhase::PostCleanup,
            );
            post
        });
        std::mem::swap(&mut simplifier.context, ctx);
        DEFAULT_SIMPLIFY_PROBE_MEMO.with(|memo| memo.borrow_mut().insert(memo_key, rewritten));
        return rewritten;
    }

    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);
    simplifier.set_sticky_value_domain(probe_value_domain);
    std::mem::swap(&mut simplifier.context, ctx);
    let mut probe_options = crate::SimplifyOptions {
        suppress_depth_overflow_warnings: true,
        ..crate::SimplifyOptions::default()
    };
    probe_options.shared.semantics.value_domain = probe_value_domain;
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(expr, probe_options);
    std::mem::swap(&mut simplifier.context, ctx);
    DEFAULT_SIMPLIFY_PROBE_MEMO.with(|memo| memo.borrow_mut().insert(memo_key, rewritten));
    rewritten
}

pub(super) fn is_zero_after_default_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let zero = ctx.num(0);
    let simplified = run_default_simplify(ctx, expr);
    compare_expr(ctx, simplified, zero) == Ordering::Equal
}

pub(super) fn small_positive_integer_value(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n)
            if n.is_integer() && *n > num_rational::BigRational::from_integer(0.into()) =>
        {
            n.to_integer().try_into().ok()
        }
        _ => None,
    }
}

pub(super) fn build_scaled_expr(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let one = ctx.num(1);
    if compare_expr(ctx, scale, one) == Ordering::Equal {
        expr
    } else {
        ctx.add(Expr::Mul(scale, expr))
    }
}

pub(super) fn normalize_signed_add_term(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
    term_sign: Sign,
) -> (cas_ast::ExprId, Sign) {
    let unheld = cas_ast::hold::unwrap_internal_hold(ctx, term_expr);
    if unheld != term_expr {
        return normalize_signed_add_term(ctx, unheld, term_sign);
    }

    if let Some(positive_expr) = strip_term_negation(ctx, term_expr) {
        return (positive_expr, term_sign.negate());
    }

    match ctx.get(term_expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some(positive_lhs) = strip_term_negation(ctx, lhs) {
                return (
                    build_scaled_expr(ctx, positive_lhs, rhs),
                    term_sign.negate(),
                );
            }
            if let Some(positive_rhs) = strip_term_negation(ctx, rhs) {
                return (
                    build_scaled_expr(ctx, positive_rhs, lhs),
                    term_sign.negate(),
                );
            }
            (term_expr, term_sign)
        }
        Expr::Div(num, den) => {
            if let Some(positive_num) = strip_term_negation(ctx, num) {
                return (ctx.add(Expr::Div(positive_num, den)), term_sign.negate());
            }
            (term_expr, term_sign)
        }
        _ => (term_expr, term_sign),
    }
}

pub(super) fn normalize_signed_add_term_for_fast_match(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
    term_sign: Sign,
) -> (cas_ast::ExprId, Sign) {
    let (term_expr, term_sign) = normalize_signed_add_term(ctx, term_expr, term_sign);
    let factors = flatten_mul_chain(ctx, term_expr);
    if factors.len() <= 1 {
        return (term_expr, term_sign);
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Some(positive_factor) = strip_term_negation(ctx, factor) else {
            continue;
        };

        let mut rebuilt_factors = factors.clone();
        rebuilt_factors[index] = positive_factor;
        let rebuilt = if rebuilt_factors.len() == 1 {
            rebuilt_factors[0]
        } else {
            build_balanced_mul(ctx, &rebuilt_factors)
        };
        return (rebuilt, term_sign.negate());
    }

    (term_expr, term_sign)
}

pub(super) fn exprs_match_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if let Some(hit) = CANCELLATION_MATCH_MEMO.with(|m| m.borrow().get(&(lhs, rhs)).copied()) {
        return hit;
    }
    let result = exprs_match_for_cancellation_uncached(ctx, lhs, rhs);
    CANCELLATION_MATCH_MEMO.with(|m| {
        m.borrow_mut().insert((lhs, rhs), result);
    });
    result
}

pub(super) fn exprs_match_for_cancellation_leaf(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if term_has_matrix_product_factor(ctx, lhs) || term_has_matrix_product_factor(ctx, rhs) {
        // Non-commutative matrix product present: only order-preserving
        // structural equality is sound (see `term_has_matrix_product_factor`).
        return compare_expr(ctx, lhs, rhs) == Ordering::Equal;
    }
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
        || exprs_equal_up_to_add_term_order(ctx, lhs, rhs)
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs, rhs)
    {
        return true;
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    compare_expr(ctx, lhs_normalized, rhs_normalized) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_add_term_order(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs_normalized, rhs_normalized)
}

pub(super) fn exprs_match_after_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if exprs_match_for_cancellation(ctx, lhs, rhs) {
        return true;
    }

    let lhs_simplified = run_default_simplify(ctx, lhs);
    let rhs_simplified = run_default_simplify(ctx, rhs);
    exprs_match_for_cancellation(ctx, lhs_simplified, rhs_simplified)
}

pub(super) fn expr_contains_hyperbolic_builtin(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_any_builtin(
        ctx,
        expr,
        &[
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
            BuiltinFn::Asinh,
            BuiltinFn::Acosh,
            BuiltinFn::Atanh,
        ],
    )
}

pub(super) fn expr_matches_negation_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> bool {
    let neg_target = ctx.add(Expr::Neg(target));
    exprs_match_for_cancellation(ctx, expr, neg_target)
}

pub(super) fn expr_matches_negation_after_default_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> bool {
    let neg_target = ctx.add(Expr::Neg(target));
    exprs_match_after_default_simplify(ctx, expr, neg_target)
}

pub(super) fn build_signed_sum_expr(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> cas_ast::ExprId {
    let Some((first_expr, first_sign)) = terms.first().copied() else {
        return ctx.num(0);
    };
    let mut acc = signed_term_expr(ctx, first_expr, first_sign);
    for (expr, sign) in terms.iter().copied().skip(1) {
        let term = signed_term_expr(ctx, expr, sign);
        acc = ctx.add(Expr::Add(acc, term));
    }
    acc
}

pub(super) fn normalize_additive_scope_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let terms = AddView::from_expr(ctx, expr).terms;
    build_signed_sum_expr(ctx, &terms)
}

pub(super) fn additive_scopes_match_after_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let lhs_normalized = normalize_additive_scope_expr(ctx, lhs);
    let rhs_normalized = normalize_additive_scope_expr(ctx, rhs);
    let lhs_terms = AddView::from_expr(ctx, lhs_normalized).terms;
    let rhs_terms = AddView::from_expr(ctx, rhs_normalized).terms;
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let lhs_signed_terms: Vec<_> = lhs_terms
        .into_iter()
        .map(|(term_expr, sign)| signed_term_expr(ctx, term_expr, sign))
        .collect();
    let rhs_signed_terms: Vec<_> = rhs_terms
        .into_iter()
        .map(|(term_expr, sign)| signed_term_expr(ctx, term_expr, sign))
        .collect();

    let mut used_rhs = vec![false; rhs_signed_terms.len()];
    for lhs_term in lhs_signed_terms {
        let Some(match_index) =
            rhs_signed_terms
                .iter()
                .enumerate()
                .find_map(|(index, rhs_term)| {
                    (!used_rhs[index]
                        && exprs_match_after_default_simplify(ctx, lhs_term, *rhs_term))
                    .then_some(index)
                })
        else {
            return false;
        };
        used_rhs[match_index] = true;
    }

    true
}

pub(super) fn negate_additive_scope_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let terms = AddView::from_expr(ctx, expr).terms;
    let negated_terms: Vec<_> = terms
        .into_iter()
        .map(|(term_expr, sign)| (term_expr, sign.negate()))
        .collect();
    build_signed_sum_expr(ctx, &negated_terms)
}

pub(super) fn try_build_fast_small_polynomial_residual_child_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    maybe_small_polynomial_expand_zero_candidate(ctx, expr)
        .then(|| try_build_fast_small_polynomial_expansion_zero_scope_rewrite(ctx, expr))
        .flatten()
}

pub(super) fn strip_unit_negation_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => return Some(inner),
        Expr::Number(n) if n.is_negative() => return Some(ctx.add(Expr::Number(-n))),
        Expr::Mul(left, right) if is_minus_one_expr(ctx, left) => return Some(right),
        Expr::Mul(left, right) if is_minus_one_expr(ctx, right) => return Some(left),
        Expr::Div(num, den) => {
            let positive_num = strip_unit_negation_for_phase_shift(ctx, num);
            let positive_den = strip_unit_negation_for_phase_shift(ctx, den);
            return match (positive_num, positive_den) {
                (Some(pos_num), None) => Some(ctx.add(Expr::Div(pos_num, den))),
                (None, Some(pos_den)) => Some(ctx.add(Expr::Div(num, pos_den))),
                _ => None,
            };
        }
        Expr::Mul(_, _) => {}
        _ => return None,
    }

    let factors = flatten_mul_chain(ctx, expr);
    for (index, factor) in factors.iter().copied().enumerate() {
        let replacement = match ctx.get(factor).clone() {
            Expr::Neg(inner) => Some(inner),
            Expr::Number(n) if n.is_negative() => Some(ctx.add(Expr::Number(-n))),
            _ => None,
        };
        let Some(replacement) = replacement else {
            continue;
        };

        let mut rebuilt = None;
        for (other_index, other) in factors.iter().copied().enumerate() {
            let current = if other_index == index {
                replacement
            } else {
                other
            };
            rebuilt = Some(match rebuilt {
                Some(acc) => smart_mul(ctx, acc, current),
                None => current,
            });
        }
        return rebuilt;
    }

    None
}

pub(super) fn extract_sin_or_cos_linear_term_for_phase_shift(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        Some((BuiltinFn::Sin, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        Some((BuiltinFn::Cos, args[0]))
    } else {
        None
    }
}

pub(super) fn try_find_trig_phase_shift_cancellation_match(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    target_is_negated: bool,
) -> Option<TrigPhaseShiftCancellationMatch> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, focus_expr),
            render_expr_for_orchestrator_profile(ctx, target_expr)
        )
    });
    let focus_has_plain_trig =
        expr_contains_any_builtin(ctx, focus_expr, &[BuiltinFn::Sin, BuiltinFn::Cos]);
    let target_has_plain_trig =
        expr_contains_any_builtin(ctx, target_expr, &[BuiltinFn::Sin, BuiltinFn::Cos]);
    if profiling {
        let pair_shape_label = match (focus_has_plain_trig, target_has_plain_trig) {
            (true, true) => "rule.phase_shift.route.entry_pair_shape.both_trig",
            (true, false) => "rule.phase_shift.route.entry_pair_shape.target_non_trig",
            (false, true) => "rule.phase_shift.route.entry_pair_shape.focus_non_trig",
            (false, false) => "rule.phase_shift.route.entry_pair_shape.both_non_trig",
        };
        let _ =
            run_profiled_orchestrator_option_section(pair_shape_label, pair_sample.clone(), || {
                Some(())
            });
        if !focus_has_plain_trig && !target_has_plain_trig {
            let detail_label =
                classify_phase_shift_nontrig_entry_detail_for_profile(ctx, focus_expr, target_expr);
            let _ =
                run_profiled_orchestrator_option_section(detail_label, pair_sample.clone(), || {
                    Some(())
                });
        }
    }
    if !focus_has_plain_trig || !target_has_plain_trig {
        return None;
    }

    let mut try_direct_general_signature_match = || -> Option<TrigPhaseShiftCancellationMatch> {
        let (arg, sin_coeff, cos_coeff, sin_sign, cos_sign) =
            extract_weighted_phase_shift_linear_combination_for_cancellation(ctx, focus_expr)?;
        let target_data = extract_general_phase_shift_term_data_for_cancellation(ctx, target_expr)?;
        if matches_general_phase_shift_shifted_term_candidate_for_cancellation(
            ctx,
            target_data,
            (arg, sin_coeff, cos_coeff, sin_sign, cos_sign),
            target_is_negated,
        ) {
            return Some(TrigPhaseShiftCancellationMatch {
                local_before: focus_expr,
                local_after: target_expr,
                mode: TrigPhaseShiftCancellationMode::LinearToShifted,
            });
        }
        let (target_arg, target_sin_coeff, target_cos_coeff, target_sin_sign, target_cos_sign) =
            extract_general_phase_shift_linear_signature_for_cancellation(ctx, target_data)?;
        let expected_sin_sign = if target_is_negated {
            -sin_sign
        } else {
            sin_sign
        };
        let expected_cos_sign = if target_is_negated {
            -cos_sign
        } else {
            cos_sign
        };

        (compare_expr(ctx, target_arg, arg) == Ordering::Equal
            && target_sin_sign == expected_sin_sign
            && target_cos_sign == expected_cos_sign
            && exprs_match_for_cancellation(ctx, target_sin_coeff, sin_coeff)
            && exprs_match_for_cancellation(ctx, target_cos_coeff, cos_coeff))
        .then_some(TrigPhaseShiftCancellationMatch {
            local_before: focus_expr,
            local_after: target_expr,
            mode: TrigPhaseShiftCancellationMode::LinearToShifted,
        })
    };
    if let Some(rewrite_match) = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.route.general_direct_try",
            pair_sample.clone(),
            &mut try_direct_general_signature_match,
        )
    } else {
        try_direct_general_signature_match()
    } {
        return Some(rewrite_match);
    }

    let focus_has_shift_signal = expr_has_phase_shift_signal_for_cancellation(ctx, focus_expr);
    let target_has_shift_signal = expr_has_phase_shift_signal_for_cancellation(ctx, target_expr);
    let focus_is_additive = matches!(ctx.get(focus_expr), Expr::Add(_, _) | Expr::Sub(_, _));
    let target_is_additive = matches!(ctx.get(target_expr), Expr::Add(_, _) | Expr::Sub(_, _));
    if !focus_has_shift_signal
        && !target_has_shift_signal
        && !focus_is_additive
        && !target_is_additive
    {
        if profiling {
            let _ = run_profiled_orchestrator_option_section(
                "rule.phase_shift.route.entry_pair_shape.no_shift_signal_single_term_reject",
                pair_sample.clone(),
                || Some(()),
            );
        }
        return None;
    }
    let matches_target = |ctx: &mut cas_ast::Context, candidate: cas_ast::ExprId| {
        if target_is_negated {
            expr_matches_negation_after_default_simplify(ctx, candidate, target_expr)
        } else {
            exprs_match_after_default_simplify(ctx, candidate, target_expr)
        }
    };
    let linear_focus_outcome = if profiling {
        run_profiled_orchestrator_section(
            "rule.phase_shift.route.linear_focus_try",
            pair_sample.clone(),
            || {
                find_linear_focus_phase_shift_cancellation_match(
                    ctx,
                    focus_expr,
                    target_expr,
                    target_is_negated,
                )
            },
            |outcome| matches!(outcome, LinearFocusPhaseShiftMatchOutcome::Matched(_)),
        )
    } else {
        find_linear_focus_phase_shift_cancellation_match(
            ctx,
            focus_expr,
            target_expr,
            target_is_negated,
        )
    };

    match linear_focus_outcome {
        LinearFocusPhaseShiftMatchOutcome::Matched(rewrite_match) => {
            if profiling {
                let _ = run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.linear_focus_matched",
                    pair_sample.clone(),
                    || Some(()),
                );
            }
            return Some(rewrite_match);
        }
        LinearFocusPhaseShiftMatchOutcome::LinearNoMatch => {
            if profiling {
                let _ = run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.linear_focus_linear_no_match",
                    pair_sample.clone(),
                    || Some(()),
                );
            }
            return None;
        }
        LinearFocusPhaseShiftMatchOutcome::NeedsGeneralRoute => {
            if profiling {
                let _ = run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.linear_focus_needs_general",
                    pair_sample.clone(),
                    || Some(()),
                );
            }
        }
        LinearFocusPhaseShiftMatchOutcome::NotLinear => {
            if profiling {
                let _ = run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.linear_focus_not_linear",
                    pair_sample.clone(),
                    || Some(()),
                );
            }
        }
    }

    let mut try_general_route = || -> Option<TrigPhaseShiftCancellationMatch> {
        if let Some((arg, sin_coeff, cos_coeff, sin_sign, cos_sign)) =
            extract_weighted_phase_shift_linear_combination_for_cancellation(ctx, focus_expr)
        {
            if let Some(target_data) =
                extract_general_phase_shift_term_data_for_cancellation(ctx, target_expr)
            {
                if let Some((
                    target_arg,
                    target_sin_coeff,
                    target_cos_coeff,
                    target_sin_sign,
                    target_cos_sign,
                )) =
                    extract_general_phase_shift_linear_signature_for_cancellation(ctx, target_data)
                {
                    let expected_sin_sign = if target_is_negated {
                        -sin_sign
                    } else {
                        sin_sign
                    };
                    let expected_cos_sign = if target_is_negated {
                        -cos_sign
                    } else {
                        cos_sign
                    };
                    if compare_expr(ctx, target_arg, arg) == Ordering::Equal
                        && target_sin_sign == expected_sin_sign
                        && target_cos_sign == expected_cos_sign
                        && exprs_match_for_cancellation(ctx, target_sin_coeff, sin_coeff)
                        && exprs_match_for_cancellation(ctx, target_cos_coeff, cos_coeff)
                    {
                        if profiling {
                            let _ = run_profiled_orchestrator_option_section(
                                "rule.phase_shift.route.general_signature_match",
                                pair_sample.clone(),
                                || Some(()),
                            );
                        }
                        return Some(TrigPhaseShiftCancellationMatch {
                            local_before: focus_expr,
                            local_after: target_expr,
                            mode: TrigPhaseShiftCancellationMode::LinearToShifted,
                        });
                    }
                }
            }

            let sine_candidate = build_general_phase_shift_sine_term_candidate_for_cancellation(
                ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
            );
            if matches_target(ctx, sine_candidate) {
                if profiling {
                    let _ = run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.general_sine_candidate",
                        pair_sample.clone(),
                        || Some(()),
                    );
                }
                return Some(TrigPhaseShiftCancellationMatch {
                    local_before: focus_expr,
                    local_after: sine_candidate,
                    mode: TrigPhaseShiftCancellationMode::LinearToShifted,
                });
            }

            let cosine_candidate = build_general_phase_shift_cosine_term_candidate_for_cancellation(
                ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
            );
            if matches_target(ctx, cosine_candidate) {
                if profiling {
                    let _ = run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.general_cosine_candidate",
                        pair_sample.clone(),
                        || Some(()),
                    );
                }
                return Some(TrigPhaseShiftCancellationMatch {
                    local_before: focus_expr,
                    local_after: cosine_candidate,
                    mode: TrigPhaseShiftCancellationMode::LinearToShifted,
                });
            }
        }

        None
    };
    if let Some(rewrite_match) = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.route.general_try",
            pair_sample.clone(),
            &mut try_general_route,
        )
    } else {
        try_general_route()
    } {
        return Some(rewrite_match);
    }

    let mut try_exact_route = || -> Option<TrigPhaseShiftCancellationMatch> {
        let focus_exact = if profiling {
            run_profiled_orchestrator_option_section(
                "rule.phase_shift.route.exact_try.focus_exact_extract",
                pair_sample.clone(),
                || extract_exact_phase_shift_term_data_for_cancellation(ctx, focus_expr),
            )
        } else {
            extract_exact_phase_shift_term_data_for_cancellation(ctx, focus_expr)
        };
        if let Some((arg, coeff, kind, sin_sign, cos_sign)) = focus_exact {
            let target_exact = if profiling {
                run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.exact_try.target_exact_extract",
                    pair_sample.clone(),
                    || extract_exact_phase_shift_term_data_for_cancellation(ctx, target_expr),
                )
            } else {
                extract_exact_phase_shift_term_data_for_cancellation(ctx, target_expr)
            };
            if let Some((target_arg, target_coeff, target_kind, target_sin_sign, target_cos_sign)) =
                target_exact
            {
                let (focus_sin_coeff, focus_cos_coeff) =
                    exact_phase_shift_linear_signature_for_cancellation(ctx, coeff, kind);
                let (target_sin_coeff, target_cos_coeff) =
                    exact_phase_shift_linear_signature_for_cancellation(
                        ctx,
                        target_coeff,
                        target_kind,
                    );
                let expected_target_sin_sign = if target_is_negated {
                    -sin_sign
                } else {
                    sin_sign
                };
                let expected_target_cos_sign = if target_is_negated {
                    -cos_sign
                } else {
                    cos_sign
                };

                if compare_expr(ctx, target_arg, arg) == Ordering::Equal
                    && target_sin_sign == expected_target_sin_sign
                    && target_cos_sign == expected_target_cos_sign
                    && exprs_match_for_cancellation(ctx, target_sin_coeff, focus_sin_coeff)
                    && exprs_match_for_cancellation(ctx, target_cos_coeff, focus_cos_coeff)
                {
                    if profiling {
                        let _ = run_profiled_orchestrator_option_section(
                            "rule.phase_shift.route.exact_signature_match",
                            pair_sample.clone(),
                            || Some(()),
                        );
                    }
                    return Some(TrigPhaseShiftCancellationMatch {
                        local_before: focus_expr,
                        local_after: target_expr,
                        mode: TrigPhaseShiftCancellationMode::ShiftedToShifted,
                    });
                }
            }

            let target_exact_arg_compatible = if let Some((target_arg, _, _, _, _)) = target_exact {
                if profiling {
                    let relation_label =
                        exact_phase_shift_arg_relation_label_for_profile(ctx, arg, target_arg);
                    let relation_profile = match relation_label {
                        "exact_match" => {
                            "rule.phase_shift.route.exact_try.target_exact_arg_relation.exact_match"
                        }
                        "symbolic_leaf_mismatch" => {
                            "rule.phase_shift.route.exact_try.target_exact_arg_relation.symbolic_leaf_mismatch"
                        }
                        "leaf_equivalent_match" => {
                            "rule.phase_shift.route.exact_try.target_exact_arg_relation.leaf_equivalent_match"
                        }
                        "other_mismatch" => {
                            "rule.phase_shift.route.exact_try.target_exact_arg_relation.other_mismatch"
                        }
                        _ => unreachable!(),
                    };
                    let _ = run_profiled_orchestrator_option_section(
                        relation_profile,
                        pair_sample.clone(),
                        || Some(()),
                    );
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.exact_try.target_exact_arg_gate",
                        pair_sample.clone(),
                        || {
                            exact_phase_shift_args_match_for_cancellation(ctx, arg, target_arg)
                                .then_some(())
                        },
                    )
                    .is_some()
                } else {
                    exact_phase_shift_args_match_for_cancellation(ctx, arg, target_arg)
                }
            } else {
                true
            };
            if !target_exact_arg_compatible {
                return None;
            }

            let target_contains_trig = if target_exact.is_some() {
                true
            } else {
                expr_contains_any_builtin(ctx, target_expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
            };
            if !target_contains_trig && expr_contains_symbolic_atom_for_cancellation(ctx, arg) {
                return None;
            }

            let expanded = build_phase_shift_linear_combination_for_cancellation(
                ctx, coeff, arg, kind, sin_sign, cos_sign,
            );
            let expanded_matches_target = if profiling {
                run_profiled_orchestrator_section(
                    "rule.phase_shift.route.exact_try.expanded_target_compare",
                    pair_sample.clone(),
                    || {
                        if target_is_negated {
                            expr_matches_negation_after_default_simplify(ctx, expanded, target_expr)
                        } else {
                            exprs_match_after_default_simplify(ctx, expanded, target_expr)
                        }
                    },
                    |matched| *matched,
                )
            } else if target_is_negated {
                expr_matches_negation_after_default_simplify(ctx, expanded, target_expr)
            } else {
                exprs_match_after_default_simplify(ctx, expanded, target_expr)
            };
            if expanded_matches_target {
                if profiling {
                    let _ = run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.exact_expanded_linear",
                        pair_sample.clone(),
                        || Some(()),
                    );
                }
                return Some(TrigPhaseShiftCancellationMatch {
                    local_before: focus_expr,
                    local_after: expanded,
                    mode: TrigPhaseShiftCancellationMode::ShiftedToLinear,
                });
            }

            let expanded_linear = if profiling {
                run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.exact_try.expanded_linear_extract",
                    pair_sample.clone(),
                    || extract_phase_shift_linear_combination_for_cancellation(ctx, expanded),
                )
            } else {
                extract_phase_shift_linear_combination_for_cancellation(ctx, expanded)
            };
            if let Some((linear_arg, linear_coeff, linear_kind, linear_sin_sign, linear_cos_sign)) =
                expanded_linear
            {
                if profiling {
                    profile_generated_candidate_target_shape_for_phase_shift(
                        ctx,
                        target_expr,
                        target_is_negated,
                        pair_sample.clone(),
                    );
                }
                let generated_target_arg_compatible = if let Some((target_arg, _, _, _, _)) =
                    target_exact
                {
                    if profiling {
                        run_profiled_orchestrator_option_section(
                            "rule.phase_shift.route.exact_try.generated_candidate.exact_arg_gate",
                            pair_sample.clone(),
                            || {
                                exact_phase_shift_args_match_for_cancellation(
                                    ctx, linear_arg, target_arg,
                                )
                                .then_some(())
                            },
                        )
                        .is_some()
                    } else {
                        exact_phase_shift_args_match_for_cancellation(ctx, linear_arg, target_arg)
                    }
                } else {
                    true
                };
                let generated_target_contains_trig = if !generated_target_arg_compatible {
                    false
                } else if target_contains_trig {
                    true
                } else if profiling {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.exact_try.generated_candidate.target_trig_gate",
                        pair_sample.clone(),
                        || {
                            expr_contains_any_builtin(
                                ctx,
                                target_expr,
                                &[BuiltinFn::Sin, BuiltinFn::Cos],
                            )
                            .then_some(())
                        },
                    )
                    .is_some()
                } else {
                    expr_contains_any_builtin(ctx, target_expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
                };
                let generated_candidate = if !generated_target_contains_trig {
                    None
                } else if profiling {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.exact_try.generated_candidate_match",
                        pair_sample.clone(),
                        || {
                            for candidate in generate_phase_shift_term_candidates_for_cancellation(
                                ctx,
                                linear_coeff,
                                linear_arg,
                                linear_kind,
                            ) {
                                if let (Some(candidate_exact), Some(target_exact)) = (
                                    extract_exact_phase_shift_term_data_for_cancellation(
                                        ctx, candidate,
                                    ),
                                    target_exact,
                                ) {
                                    profile_exact_phase_shift_pair_relation_for_phase_shift(
                                        ctx,
                                        candidate_exact,
                                        target_exact,
                                        target_is_negated,
                                        "rule.phase_shift.route.exact_try.generated_candidate.exact_pair",
                                        pair_sample.clone(),
                                    );
                                }
                                let matches = if target_is_negated {
                                    expr_matches_negation_after_default_simplify(
                                        ctx,
                                        candidate,
                                        target_expr,
                                    )
                                } else {
                                    exprs_match_after_default_simplify(ctx, candidate, target_expr)
                                };
                                if matches {
                                    return Some(candidate);
                                }
                            }
                            None
                        },
                    )
                } else {
                    let mut matched_candidate = None;
                    for candidate in generate_phase_shift_term_candidates_for_cancellation(
                        ctx,
                        linear_coeff,
                        linear_arg,
                        linear_kind,
                    ) {
                        let matches = if target_is_negated {
                            expr_matches_negation_after_default_simplify(
                                ctx,
                                candidate,
                                target_expr,
                            )
                        } else {
                            exprs_match_after_default_simplify(ctx, candidate, target_expr)
                        };
                        if matches {
                            matched_candidate = Some(candidate);
                            break;
                        }
                    }
                    matched_candidate
                };
                if let Some(candidate) = generated_candidate {
                    if profiling {
                        let _ = run_profiled_orchestrator_option_section(
                            "rule.phase_shift.route.exact_generated_candidate",
                            pair_sample.clone(),
                            || Some(()),
                        );
                    }
                    return Some(TrigPhaseShiftCancellationMatch {
                        local_before: focus_expr,
                        local_after: candidate,
                        mode: TrigPhaseShiftCancellationMode::ShiftedToShifted,
                    });
                }

                if generated_target_contains_trig {
                    let linear_reexpanded = build_phase_shift_linear_combination_for_cancellation(
                        ctx,
                        linear_coeff,
                        linear_arg,
                        linear_kind,
                        linear_sin_sign,
                        linear_cos_sign,
                    );
                    let matches = if profiling {
                        run_profiled_orchestrator_section(
                            "rule.phase_shift.route.exact_try.reexpanded_target_compare",
                            pair_sample.clone(),
                            || {
                                if target_is_negated {
                                    expr_matches_negation_after_default_simplify(
                                        ctx,
                                        linear_reexpanded,
                                        target_expr,
                                    )
                                } else {
                                    exprs_match_after_default_simplify(
                                        ctx,
                                        linear_reexpanded,
                                        target_expr,
                                    )
                                }
                            },
                            |matched| *matched,
                        )
                    } else if target_is_negated {
                        expr_matches_negation_after_default_simplify(
                            ctx,
                            linear_reexpanded,
                            target_expr,
                        )
                    } else {
                        exprs_match_after_default_simplify(ctx, linear_reexpanded, target_expr)
                    };
                    if matches {
                        if profiling {
                            let _ = run_profiled_orchestrator_option_section(
                                "rule.phase_shift.route.exact_reexpanded_linear",
                                pair_sample.clone(),
                                || Some(()),
                            );
                        }
                        return Some(TrigPhaseShiftCancellationMatch {
                            local_before: focus_expr,
                            local_after: target_expr,
                            mode: TrigPhaseShiftCancellationMode::ShiftedToShifted,
                        });
                    }
                }
            }
        }

        None
    };
    if let Some(rewrite_match) = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.route.exact_try",
            pair_sample.clone(),
            &mut try_exact_route,
        )
    } else {
        try_exact_route()
    } {
        return Some(rewrite_match);
    }

    let mut try_shifted_route = || -> Option<TrigPhaseShiftCancellationMatch> {
        let focus_general = if profiling {
            run_profiled_orchestrator_option_section(
                "rule.phase_shift.route.shifted_try.focus_general_extract",
                pair_sample.clone(),
                || extract_general_phase_shift_term_data_for_cancellation(ctx, focus_expr),
            )
        } else {
            extract_general_phase_shift_term_data_for_cancellation(ctx, focus_expr)
        };
        if let Some(data) = focus_general {
            let linear_signature = if profiling {
                run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.shifted_try.focus_linear_signature",
                    pair_sample.clone(),
                    || extract_general_phase_shift_linear_signature_for_cancellation(ctx, data),
                )
            } else {
                extract_general_phase_shift_linear_signature_for_cancellation(ctx, data)
            };
            if let Some((linear_arg, sin_coeff, cos_coeff, sin_sign, cos_sign)) = linear_signature {
                let mut try_target_shifted_general_match =
                    || -> Option<TrigPhaseShiftCancellationMatch> {
                        let target_general = if profiling {
                            run_profiled_orchestrator_option_section(
                                "rule.phase_shift.route.shifted_try.target_general_extract",
                                pair_sample.clone(),
                                || {
                                    extract_general_phase_shift_term_data_for_cancellation(
                                        ctx,
                                        target_expr,
                                    )
                                },
                            )
                        } else {
                            extract_general_phase_shift_term_data_for_cancellation(ctx, target_expr)
                        }?;

                        let target_general_match = if profiling {
                            run_profiled_orchestrator_option_section(
                                "rule.phase_shift.route.shifted_general_target_probe",
                                pair_sample.clone(),
                                || {
                                    matches_general_phase_shift_shifted_term_candidate_for_cancellation(
                                        ctx,
                                        target_general,
                                        (
                                            linear_arg,
                                            sin_coeff,
                                            cos_coeff,
                                            sin_sign,
                                            cos_sign,
                                        ),
                                        target_is_negated,
                                    )
                                    .then_some(())
                                },
                            )
                            .is_some()
                        } else {
                            matches_general_phase_shift_shifted_term_candidate_for_cancellation(
                                ctx,
                                target_general,
                                (linear_arg, sin_coeff, cos_coeff, sin_sign, cos_sign),
                                target_is_negated,
                            )
                        };
                        if !target_general_match {
                            return None;
                        }

                        Some(TrigPhaseShiftCancellationMatch {
                            local_before: focus_expr,
                            local_after: target_expr,
                            mode: TrigPhaseShiftCancellationMode::ShiftedToShifted,
                        })
                    };
                if let Some(rewrite_match) = if profiling {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.shifted_direct_general_target_match",
                        pair_sample.clone(),
                        &mut try_target_shifted_general_match,
                    )
                } else {
                    try_target_shifted_general_match()
                } {
                    return Some(rewrite_match);
                }

                let target_linear_match = if profiling {
                    run_profiled_orchestrator_section(
                        "rule.phase_shift.route.shifted_try.target_linear_match",
                        pair_sample.clone(),
                        || {
                            matches_weighted_phase_shift_linear_combination_target(
                                ctx,
                                target_expr,
                                target_is_negated,
                                linear_arg,
                                sin_coeff,
                                cos_coeff,
                                (sin_sign, cos_sign),
                            )
                        },
                        |matched| *matched,
                    )
                } else {
                    matches_weighted_phase_shift_linear_combination_target(
                        ctx,
                        target_expr,
                        target_is_negated,
                        linear_arg,
                        sin_coeff,
                        cos_coeff,
                        (sin_sign, cos_sign),
                    )
                };
                if target_linear_match {
                    if profiling {
                        let _ = run_profiled_orchestrator_option_section(
                            "rule.phase_shift.route.shifted_linear_signature_match",
                            pair_sample.clone(),
                            || Some(()),
                        );
                    }
                    return Some(TrigPhaseShiftCancellationMatch {
                        local_before: focus_expr,
                        local_after: target_expr,
                        mode: TrigPhaseShiftCancellationMode::ShiftedToLinear,
                    });
                }

                let expanded = build_weighted_phase_shift_linear_combination_for_cancellation(
                    ctx, linear_arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
                );
                let expanded = run_default_simplify(ctx, expanded);

                let expanded_matches_target = if profiling {
                    run_profiled_orchestrator_section(
                        "rule.phase_shift.route.shifted_try.expanded_target_compare",
                        pair_sample.clone(),
                        || matches_target(ctx, expanded),
                        |matched| *matched,
                    )
                } else {
                    matches_target(ctx, expanded)
                };
                if expanded_matches_target {
                    if profiling {
                        let _ = run_profiled_orchestrator_option_section(
                            "rule.phase_shift.route.shifted_expanded_linear",
                            pair_sample.clone(),
                            || Some(()),
                        );
                    }
                    return Some(TrigPhaseShiftCancellationMatch {
                        local_before: focus_expr,
                        local_after: expanded,
                        mode: TrigPhaseShiftCancellationMode::ShiftedToLinear,
                    });
                }

                let generated_candidate = if profiling {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.shifted_try.generated_candidate_match",
                        pair_sample.clone(),
                        || {
                            profile_shifted_generated_candidate_target_family_for_phase_shift(
                                ctx,
                                target_expr,
                                pair_sample.clone(),
                            );
                            for (candidate_index, candidate) in
                                generate_general_phase_shift_term_candidates_for_cancellation(
                                    ctx, linear_arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
                                )
                                .into_iter()
                                .enumerate()
                            {
                                let candidate_label = match candidate_index {
                                    0 => {
                                        "rule.phase_shift.route.shifted_try.generated_candidate.sine_candidate"
                                    }
                                    1 => {
                                        "rule.phase_shift.route.shifted_try.generated_candidate.cosine_candidate"
                                    }
                                    _ => {
                                        "rule.phase_shift.route.shifted_try.generated_candidate.other_candidate"
                                    }
                                };
                                if let Some(matched_candidate) =
                                    run_profiled_orchestrator_option_section(
                                        candidate_label,
                                        pair_sample.clone(),
                                        || matches_target(ctx, candidate).then_some(candidate),
                                    )
                                {
                                    return Some(matched_candidate);
                                }
                            }
                            None
                        },
                    )
                } else {
                    generate_general_phase_shift_term_candidates_for_cancellation(
                        ctx, linear_arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
                    )
                    .into_iter()
                    .find(|candidate| matches_target(ctx, *candidate))
                };
                if let Some(candidate) = generated_candidate {
                    if profiling {
                        let _ = run_profiled_orchestrator_option_section(
                            "rule.phase_shift.route.shifted_generated_candidate",
                            pair_sample.clone(),
                            || Some(()),
                        );
                    }
                    return Some(TrigPhaseShiftCancellationMatch {
                        local_before: focus_expr,
                        local_after: candidate,
                        mode: TrigPhaseShiftCancellationMode::ShiftedToShifted,
                    });
                }
            }
        }

        None
    };
    if let Some(rewrite_match) = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.route.shifted_try",
            pair_sample.clone(),
            &mut try_shifted_route,
        )
    } else {
        try_shifted_route()
    } {
        return Some(rewrite_match);
    }

    let mut try_final_linear_compare = || -> Option<TrigPhaseShiftCancellationMatch> {
        let focus_linear = if profiling {
            let exact_linear = run_profiled_orchestrator_option_section(
                "rule.phase_shift.route.final_linear_compare.focus_exact_linear",
                pair_sample.clone(),
                || {
                    extract_exact_phase_shift_term_data_for_cancellation(ctx, focus_expr).map(
                        |exact_data @ (arg, coeff, kind, sin_sign, cos_sign)| {
                            (
                                build_phase_shift_linear_combination_for_cancellation(
                                    ctx, coeff, arg, kind, sin_sign, cos_sign,
                                ),
                                exact_data,
                            )
                        },
                    )
                },
            );
            exact_linear
                .map(|(linear, exact_data)| (linear, Some(exact_data)))
                .or_else(|| {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.final_linear_compare.focus_general_linear",
                        pair_sample.clone(),
                        || {
                            extract_general_phase_shift_term_data_for_cancellation(ctx, focus_expr)
                                .and_then(|data| {
                                    build_general_phase_shift_linear_combination_for_cancellation(
                                        ctx, data,
                                    )
                                })
                        },
                    )
                    .map(|linear| (linear, None))
                })
        } else {
            extract_exact_phase_shift_term_data_for_cancellation(ctx, focus_expr)
                .map(|exact_data @ (arg, coeff, kind, sin_sign, cos_sign)| {
                    (
                        build_phase_shift_linear_combination_for_cancellation(
                            ctx, coeff, arg, kind, sin_sign, cos_sign,
                        ),
                        exact_data,
                    )
                })
                .map(|(linear, exact_data)| (linear, Some(exact_data)))
                .or_else(|| {
                    extract_general_phase_shift_term_data_for_cancellation(ctx, focus_expr)
                        .and_then(|data| {
                            build_general_phase_shift_linear_combination_for_cancellation(ctx, data)
                        })
                        .map(|linear| (linear, None))
                })
        };
        let (focus_linear, focus_exact) = focus_linear?;

        let target_linear = if profiling {
            let exact_linear = run_profiled_orchestrator_option_section(
                "rule.phase_shift.route.final_linear_compare.target_exact_linear",
                pair_sample.clone(),
                || {
                    extract_exact_phase_shift_term_data_for_cancellation(ctx, target_expr).map(
                        |exact_data @ (arg, coeff, kind, sin_sign, cos_sign)| {
                            (
                                build_phase_shift_linear_combination_for_cancellation(
                                    ctx, coeff, arg, kind, sin_sign, cos_sign,
                                ),
                                exact_data,
                            )
                        },
                    )
                },
            );
            exact_linear
                .map(|(linear, exact_data)| (linear, Some(exact_data)))
                .or_else(|| {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.route.final_linear_compare.target_general_linear",
                        pair_sample.clone(),
                        || {
                            extract_general_phase_shift_term_data_for_cancellation(ctx, target_expr)
                                .and_then(|data| {
                                    build_general_phase_shift_linear_combination_for_cancellation(
                                        ctx, data,
                                    )
                                })
                        },
                    )
                    .map(|linear| (linear, None))
                })
        } else {
            extract_exact_phase_shift_term_data_for_cancellation(ctx, target_expr)
                .map(|exact_data @ (arg, coeff, kind, sin_sign, cos_sign)| {
                    (
                        build_phase_shift_linear_combination_for_cancellation(
                            ctx, coeff, arg, kind, sin_sign, cos_sign,
                        ),
                        exact_data,
                    )
                })
                .map(|(linear, exact_data)| (linear, Some(exact_data)))
                .or_else(|| {
                    extract_general_phase_shift_term_data_for_cancellation(ctx, target_expr)
                        .and_then(|data| {
                            build_general_phase_shift_linear_combination_for_cancellation(ctx, data)
                        })
                        .map(|linear| (linear, None))
                })
        };
        let (target_linear, target_exact) = target_linear?;

        if profiling {
            let origin_label = match (focus_exact.is_some(), target_exact.is_some()) {
                (true, true) => "rule.phase_shift.route.final_linear_compare.origin.exact_exact",
                (true, false) => "rule.phase_shift.route.final_linear_compare.origin.exact_general",
                (false, true) => "rule.phase_shift.route.final_linear_compare.origin.general_exact",
                (false, false) => {
                    "rule.phase_shift.route.final_linear_compare.origin.general_general"
                }
            };
            let _ =
                run_profiled_orchestrator_option_section(origin_label, pair_sample.clone(), || {
                    Some(())
                });
            if let (Some(focus_exact), Some(target_exact)) = (focus_exact, target_exact) {
                profile_exact_phase_shift_pair_relation_for_phase_shift(
                    ctx,
                    focus_exact,
                    target_exact,
                    target_is_negated,
                    "rule.phase_shift.route.final_linear_compare.origin.exact_exact_pair",
                    pair_sample.clone(),
                );
            }
        }

        let exact_arg_compatible = if let (
            Some((focus_arg, _, _, _, _)),
            Some((target_arg, _, _, _, _)),
        ) = (focus_exact, target_exact)
        {
            if profiling {
                let relation_label =
                    exact_phase_shift_arg_relation_label_for_profile(ctx, focus_arg, target_arg);
                let relation_profile = match relation_label {
                        "exact_match" => {
                            "rule.phase_shift.route.final_linear_compare.origin.exact_exact_arg_relation.exact_match"
                        }
                        "symbolic_leaf_mismatch" => {
                            "rule.phase_shift.route.final_linear_compare.origin.exact_exact_arg_relation.symbolic_leaf_mismatch"
                        }
                        "leaf_equivalent_match" => {
                            "rule.phase_shift.route.final_linear_compare.origin.exact_exact_arg_relation.leaf_equivalent_match"
                        }
                        "other_mismatch" => {
                            "rule.phase_shift.route.final_linear_compare.origin.exact_exact_arg_relation.other_mismatch"
                        }
                        _ => unreachable!(),
                    };
                let _ = run_profiled_orchestrator_option_section(
                    relation_profile,
                    pair_sample.clone(),
                    || Some(()),
                );
                run_profiled_orchestrator_option_section(
                    "rule.phase_shift.route.final_linear_compare.origin.exact_exact_arg_gate",
                    pair_sample.clone(),
                    || {
                        exact_phase_shift_args_match_for_cancellation(ctx, focus_arg, target_arg)
                            .then_some(())
                    },
                )
                .is_some()
            } else {
                exact_phase_shift_args_match_for_cancellation(ctx, focus_arg, target_arg)
            }
        } else {
            true
        };
        if !exact_arg_compatible {
            return None;
        }

        let matches = if profiling {
            run_profiled_orchestrator_section(
                "rule.phase_shift.route.final_linear_compare.compare",
                pair_sample.clone(),
                || {
                    if target_is_negated {
                        expr_matches_negation_after_default_simplify(
                            ctx,
                            focus_linear,
                            target_linear,
                        )
                    } else {
                        exprs_match_after_default_simplify(ctx, focus_linear, target_linear)
                    }
                },
                |matched| *matched,
            )
        } else if target_is_negated {
            expr_matches_negation_after_default_simplify(ctx, focus_linear, target_linear)
        } else {
            exprs_match_after_default_simplify(ctx, focus_linear, target_linear)
        };
        if matches {
            return Some(TrigPhaseShiftCancellationMatch {
                local_before: focus_expr,
                local_after: target_expr,
                mode: TrigPhaseShiftCancellationMode::ShiftedToShifted,
            });
        }

        None
    };
    if let Some(rewrite_match) = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.route.final_linear_compare_try",
            pair_sample.clone(),
            &mut try_final_linear_compare,
        )
    } else {
        try_final_linear_compare()
    } {
        return Some(rewrite_match);
    }

    None
}

pub(super) fn try_build_direct_trig_square_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((arg, is_sum)) = extract_trig_binomial_square_identity_data(ctx, source) else {
            continue;
        };
        let target_candidate = build_trig_binomial_square_target(ctx, arg, is_sum);
        if exprs_match_for_cancellation(ctx, target_candidate, target)
            || exprs_match_after_default_simplify(ctx, target_candidate, target)
        {
            return Some(build_direct_trig_square_equivalence_rewrite(
                ctx, lhs_core, rhs_core, target, arg, is_sum,
            ));
        }
    }

    None
}

pub(super) fn expr_contains_division_node(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Div(_, _) => return true,
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(crate) fn try_build_exact_zero_identity_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, expr) {
        if is_atanh_common_log_definition_mismatch_pair(ctx, lhs_core, rhs_core) {
            return None;
        }
    }

    if let Some(rewrite) = try_build_exact_zero_identity_rewrite_direct(ctx, expr) {
        return Some(rewrite);
    }

    let view = AddView::from_expr(ctx, expr);
    if !(2..=4).contains(&view.terms.len()) {
        return None;
    }

    let flipped_terms: Vec<_> = view
        .terms
        .iter()
        .map(|(term_expr, term_sign)| (*term_expr, term_sign.negate()))
        .collect();
    let flipped_expr = build_signed_sum_expr(ctx, &flipped_terms);
    let child_rewrite = try_build_exact_zero_identity_rewrite_direct(ctx, flipped_expr)?;

    let mut rewrite = Rewrite::with_local(
        ctx.num(0),
        child_rewrite.description.clone(),
        expr,
        ctx.num(0),
    )
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone());

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    rewrite.substeps = child_rewrite.substeps.clone();
    Some(rewrite)
}

pub(super) fn build_mul_expr_from_factors(
    ctx: &mut cas_ast::Context,
    factors: &[cas_ast::ExprId],
) -> cas_ast::ExprId {
    match factors {
        [] => ctx.num(1),
        [single] => *single,
        _ => build_balanced_mul(ctx, factors),
    }
}

pub(super) fn extract_common_multiplicative_residual_sum(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();

    let first_factors = flatten_mul_chain(ctx, normalized_terms.first()?.0);
    if first_factors.is_empty() {
        return None;
    }

    let mut used_by_term: Vec<Vec<bool>> = normalized_terms
        .iter()
        .map(|(term_expr, _)| vec![false; flatten_mul_chain(ctx, *term_expr).len()])
        .collect();
    let factor_lists: Vec<Vec<_>> = normalized_terms
        .iter()
        .map(|(term_expr, _)| flatten_mul_chain(ctx, *term_expr))
        .collect();

    let mut common = Vec::new();
    for first_factor in first_factors {
        let mut matched_indexes = Vec::new();
        let mut all_match = true;

        for (term_index, factors) in factor_lists.iter().enumerate().skip(1) {
            let Some(factor_index) =
                factors
                    .iter()
                    .enumerate()
                    .find_map(|(factor_index, factor)| {
                        (!used_by_term[term_index][factor_index]
                            && compare_expr(ctx, *factor, first_factor) == Ordering::Equal)
                            .then_some(factor_index)
                    })
            else {
                all_match = false;
                break;
            };
            matched_indexes.push((term_index, factor_index));
        }

        if !all_match {
            continue;
        }

        common.push(first_factor);
        for (term_index, factor_index) in matched_indexes {
            used_by_term[term_index][factor_index] = true;
        }

        if let Some(first_index) =
            factor_lists[0]
                .iter()
                .enumerate()
                .find_map(|(factor_index, factor)| {
                    (!used_by_term[0][factor_index]
                        && compare_expr(ctx, *factor, first_factor) == Ordering::Equal)
                        .then_some(factor_index)
                })
        {
            used_by_term[0][first_index] = true;
        }
    }

    if common.is_empty() {
        return None;
    }

    let residual_terms: Vec<_> = normalized_terms
        .iter()
        .enumerate()
        .map(|(term_index, (_term_expr, term_sign))| {
            let residual_factors: Vec<_> = factor_lists[term_index]
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(factor_index, factor)| {
                    (!used_by_term[term_index][factor_index]).then_some(factor)
                })
                .collect();
            (
                build_mul_expr_from_factors(ctx, &residual_factors),
                *term_sign,
            )
        })
        .collect();

    let common_factor = build_mul_expr_from_factors(ctx, &common);
    let residual_expr = build_signed_sum_expr(ctx, &residual_terms);
    let one = ctx.num(1);
    if compare_expr(ctx, common_factor, one) == Ordering::Equal
        || compare_expr(ctx, residual_expr, expr) == Ordering::Equal
    {
        return None;
    }
    Some((common_factor, residual_expr))
}

pub(super) fn try_build_stripped_zero_log_identity_child_rewrite(
    ctx: &mut cas_ast::Context,
    residual_expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let stripped_residual = strip_single_additive_zero_term(ctx, residual_expr)?;
    if !expr_contains_any_builtin(
        ctx,
        stripped_residual,
        &[
            BuiltinFn::Ln,
            BuiltinFn::Log,
            BuiltinFn::Log2,
            BuiltinFn::Log10,
            BuiltinFn::Abs,
        ],
    ) {
        return None;
    }

    let child_rewrite = try_build_exact_zero_identity_rewrite(ctx, stripped_residual)?;
    let zero = ctx.num(0);
    (compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal)
        .then_some(child_rewrite)
}

pub(super) fn try_build_exact_zero_identity_rewrite_direct(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    try_build_exact_zero_identity_rewrite_direct_impl(ctx, expr, true)
}

pub(super) fn sign_to_i64(sign: Sign) -> i64 {
    match sign {
        Sign::Pos => 1,
        Sign::Neg => -1,
    }
}

pub(crate) fn extract_two_term_core_difference(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => {
            let (lhs_expr, lhs_sign) = normalize_core_difference_term(ctx, lhs, Sign::Pos);
            let (rhs_expr, rhs_sign) = normalize_core_difference_term(ctx, rhs, Sign::Pos);
            Some((
                apply_sign_to_expr(ctx, sign_to_i64(lhs_sign), lhs_expr),
                apply_sign_to_expr(ctx, sign_to_i64(rhs_sign), rhs_expr),
            ))
        }
        Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
            Expr::Neg(inner) => {
                let (lhs_expr, lhs_sign) = normalize_core_difference_term(ctx, lhs, Sign::Pos);
                let (rhs_expr, rhs_sign) = normalize_core_difference_term(ctx, inner, Sign::Pos);
                Some((
                    apply_sign_to_expr(ctx, sign_to_i64(lhs_sign), lhs_expr),
                    apply_sign_to_expr(ctx, sign_to_i64(rhs_sign), rhs_expr),
                ))
            }
            _ => {
                let terms = AddView::from_expr(ctx, expr).terms;
                if terms.len() != 2 {
                    return None;
                }
                let (first_expr, first_sign) =
                    normalize_core_difference_term(ctx, terms[0].0, terms[0].1);
                let (second_expr, second_sign) =
                    normalize_core_difference_term(ctx, terms[1].0, terms[1].1);
                Some((
                    apply_sign_to_expr(ctx, sign_to_i64(first_sign), first_expr),
                    apply_sign_to_expr(ctx, sign_to_i64(second_sign).checked_neg()?, second_expr),
                ))
            }
        },
        _ => {
            let terms = AddView::from_expr(ctx, expr).terms;
            if terms.len() != 2 {
                return None;
            }
            let (first_expr, first_sign) =
                normalize_core_difference_term(ctx, terms[0].0, terms[0].1);
            let (second_expr, second_sign) =
                normalize_core_difference_term(ctx, terms[1].0, terms[1].1);
            Some((
                apply_sign_to_expr(ctx, sign_to_i64(first_sign), first_expr),
                apply_sign_to_expr(ctx, sign_to_i64(second_sign).checked_neg()?, second_expr),
            ))
        }
    }
}

pub(super) fn try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((rewritten, den)) =
            try_rewrite_trig_cos_diff_sin_diff_quotient_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(
                Rewrite::with_local(ctx.num(0), "Cos-Diff / Sin-Diff Quotient", source, target)
                    .requires(crate::ImplicitCondition::NonZero(den)),
            );
        }
    }

    None
}

pub(super) fn try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (linear_side, shifted_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(linear_groups) =
            extract_structural_unit_linear_phase_shift_pair_side(ctx, linear_side)
        else {
            continue;
        };
        let Some(shifted_groups) =
            extract_structural_unit_exact_quarter_shifted_phase_shift_pair_side(ctx, shifted_side)
        else {
            continue;
        };

        if !structural_unit_phase_shift_pair_groups_match(ctx, &linear_groups, &shifted_groups) {
            continue;
        }

        return Some(
            Rewrite::with_local(ctx.num(0), "Phase Shift Identity", lhs_core, rhs_core)
                .substep(
                    "Aplicar identidad de desfase",
                    vec![
                        "Reescribir cada par sin(u) + cos(u) como sqrt(2)·sin(u + pi/4)."
                            .to_string(),
                    ],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras reescribir ambos pares, los dos lados quedan idénticos y la diferencia se anula."
                            .to_string(),
                    ],
                ),
        );
    }

    None
}

pub(super) fn try_build_direct_trig_double_angle_contraction_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            try_rewrite_signed_double_angle_contraction_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Contraction",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Hyperbolic Product-to-Sum and Triple-Angle Identity",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_tanh_exp_definition_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) = try_rewrite_tanh_exp_definition_for_cancellation(ctx, source) else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Recognize Hyperbolic from Exponential",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(crate) fn try_build_direct_sub_fraction_combination_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target), Expr::Div(_, _)) {
            continue;
        }
        if let Some(rewritten) =
            try_rewrite_symbolic_difference_squares_telescoping_for_cancellation(ctx, source)
        {
            if exprs_match_for_cancellation(ctx, rewritten, target) {
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    "Subtract Fractions",
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        let Some(rewritten) =
            try_rewrite_scaled_sub_fraction_combination_for_cancellation(ctx, source)
        else {
            continue;
        };
        let residual = ctx.add(Expr::Sub(rewritten, target));

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
            || is_zero_after_default_simplify(ctx, residual)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Subtract Fractions",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_safe_hyperbolic_core_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let nested_default_simplify = default_simplify_nesting_depth() > 0;
    if !expr_contains_hyperbolic_builtin(ctx, lhs_core)
        && !expr_contains_hyperbolic_builtin(ctx, rhs_core)
    {
        return None;
    }

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let source_has_direct_hyperbolic = expr_contains_direct_hyperbolic_builtin(ctx, source);
        let source_has_atanh = expr_contains_any_builtin(ctx, source, &[BuiltinFn::Atanh]);
        if source_has_atanh
            && !source_has_direct_hyperbolic
            && !expr_contains_any_builtin(
                ctx,
                target,
                &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
            )
        {
            continue;
        }

        let Some((rewritten, description)) =
            try_rewrite_safe_direct_hyperbolic_equivalence_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || (!nested_default_simplify
                && exprs_match_after_default_simplify(ctx, rewritten, target))
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                description,
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_core_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, lhs_core),
            render_expr_for_orchestrator_profile(ctx, rhs_core)
        )
    });
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, pair_sample.clone(), || Some(()));
        }
    };

    if exprs_match_shallow_noncall_for_cancellation(ctx, lhs_core, rhs_core) {
        profile_route("rule.direct_core_equivalence.route.direct_match");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewrite) =
        try_build_direct_reciprocal_half_power_shared_denominator_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.reciprocal_half_power_shared_denominator",
        );
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_scaled_reciprocal_half_power_shared_denominator_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.scaled_reciprocal_half_power_shared_denominator",
        );
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_scaled_reciprocal_half_power_over_base_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.scaled_reciprocal_half_power_over_base");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_negative_even_root_power_reciprocal_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.negative_even_root_power_reciprocal");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_reciprocal_half_power_product_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.reciprocal_half_power_product");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_scaled_reciprocal_half_power_product_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.scaled_reciprocal_half_power_product");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_common_sqrt_denominator_fraction_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.common_sqrt_denominator_fraction");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_sqrt_over_base_fraction_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.sqrt_over_base_fraction");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_rationalized_common_sqrt_denominator_fraction_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.rationalized_common_sqrt_denominator_fraction",
        );
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_tanh_exp_definition_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.tanh_exp");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_square");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_product_to_sum_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_product_to_sum");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_sum_to_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_sum_to_product");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.phase_shift_pair");
        return Some(rewrite);
    }

    let maybe_phase_shift_pair_residual =
        if expr_contains_any_builtin(ctx, lhs_core, &[BuiltinFn::Sin, BuiltinFn::Cos])
            || expr_contains_any_builtin(ctx, rhs_core, &[BuiltinFn::Sin, BuiltinFn::Cos])
        {
            let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
            (AddView::from_expr(ctx, residual_expr).terms.len() == 6).then_some(residual_expr)
        } else {
            None
        };

    if let Some(residual_expr) = maybe_phase_shift_pair_residual {
        if let Some(rewrite) =
            try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr)
        {
            profile_route("rule.direct_core_equivalence.route.repeated_phase_shift_pair");
            return Some(rewrite);
        }
    }

    let has_hyperbolic_core = expr_contains_hyperbolic_builtin(ctx, lhs_core)
        || expr_contains_hyperbolic_builtin(ctx, rhs_core);
    if has_hyperbolic_core {
        if let Some(rewrite) =
            try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            profile_route("rule.direct_core_equivalence.route.safe_hyperbolic");
            return Some(rewrite);
        }
    }

    if exprs_match_for_cancellation(ctx, lhs_core, rhs_core) {
        profile_route("rule.direct_core_equivalence.route.direct_match");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewrite) = try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.cos_diff_sin_diff_quotient");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.sum_diff_cubes_quotient");
        return Some(rewrite);
    }

    if try_find_trig_phase_shift_cancellation_match(ctx, lhs_core, rhs_core, false)
        .or_else(|| try_find_trig_phase_shift_cancellation_match(ctx, rhs_core, lhs_core, false))
        .is_some()
    {
        profile_route("rule.direct_core_equivalence.route.phase_shift_identity");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Phase Shift Identity",
            lhs_core,
            rhs_core,
        ));
    }

    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some(rewrite) = try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr)
    {
        profile_route("rule.direct_core_equivalence.route.repeated_phase_shift_residual");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_cos_product_telescoping_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.cos_product_telescoping");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_finite_sum_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.finite_sum");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_dirichlet_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.dirichlet");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_finite_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.finite_product");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_power_reduction_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_power_reduction");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_double_angle_contraction_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.double_angle_contraction");
        return Some(rewrite);
    }
    if let Some(rewrite) = try_build_direct_trig_cos_double_angle_polynomial_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.cos_double_angle_poly");
        return Some(rewrite);
    }
    if let Some(rewrite) = try_build_direct_trig_mixed_double_angle_polynomial_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.mixed_double_angle_poly");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.double_angle_cos_variant");
        return Some(rewrite);
    }
    if maybe_two_term_embedded_double_angle_expansion_candidate(ctx, lhs_core, rhs_core) {
        if let Some(rewrite) =
            try_build_direct_trig_embedded_double_angle_expansion_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        {
            profile_route("rule.direct_core_equivalence.route.embedded_double_angle");
            return Some(rewrite);
        }
    }
    if let Some(rewrite) = try_build_direct_multi_angle_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.multi_angle");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_recursive_trig_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.recursive_trig");
        return Some(rewrite);
    }

    if let Some((rewritten, description)) =
        try_rewrite_exact_trig_equivalence_for_cancellation(ctx, lhs_core)
    {
        if exprs_match_for_cancellation(ctx, rewritten, rhs_core)
            || exprs_match_after_default_simplify(ctx, rewritten, rhs_core)
        {
            profile_route("rule.direct_core_equivalence.route.exact_trig_lhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                description,
                lhs_core,
                rhs_core,
            ));
        }
    }
    if let Some((rewritten, description)) =
        try_rewrite_exact_trig_equivalence_for_cancellation(ctx, rhs_core)
    {
        if exprs_match_for_cancellation(ctx, rewritten, lhs_core)
            || exprs_match_after_default_simplify(ctx, rewritten, lhs_core)
        {
            profile_route("rule.direct_core_equivalence.route.exact_trig_rhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                description,
                lhs_core,
                rhs_core,
            ));
        }
    }

    let has_tanh_core = expr_contains_any_builtin(ctx, lhs_core, &[BuiltinFn::Tanh])
        || expr_contains_any_builtin(ctx, rhs_core, &[BuiltinFn::Tanh]);
    if has_tanh_core {
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, lhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, rhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, rhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_lhs_tanh");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, rhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, lhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, lhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_rhs_tanh");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    if classify_symbolic_scale_sum_profile_detail(ctx, lhs_core) == "grouped_multi_scale"
        && grouped_symbolic_scale_sum_matches_target_for_cancellation(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_lhs");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewritten) = try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, lhs_core) {
        if exprs_match_for_cancellation(ctx, rewritten, rhs_core) {
            profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_lhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Equivalent Residual Cancellation",
                lhs_core,
                rhs_core,
            ));
        }
    }

    if classify_symbolic_scale_sum_profile_detail(ctx, rhs_core) == "grouped_multi_scale"
        && grouped_symbolic_scale_sum_matches_target_for_cancellation(ctx, rhs_core, lhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_rhs");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewritten) = try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, rhs_core) {
        if exprs_match_for_cancellation(ctx, rewritten, lhs_core) {
            profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_rhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Equivalent Residual Cancellation",
                lhs_core,
                rhs_core,
            ));
        }
    }

    if let Some(rewrite) =
        try_build_direct_trig_reciprocal_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_reciprocal");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_trig_ratio_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_ratio");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_log_expansion_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.log_expansion");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_log_chain_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.log_chain_product");
        return Some(rewrite);
    }

    if let Some(false) =
        reject_noncall_vs_surface_symbolic_trig_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_noncall_surface_trig_reject",
        );
        return None;
    }

    if let Some(false) = reject_atomic_noncall_pair_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_atomic_noncall_pair_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_scaled_symbolic_atom_mismatch_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_scaled_symbolic_atom_mismatch_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_noncall_product_vs_division_shared_numerator_scale_before_default_simplify(
            ctx, lhs_core, rhs_core,
        )
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_product_division_shared_scale_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_surface_plain_cross_trig_pair_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_plain_cross_trig_reject",
        );
        return None;
    }

    if let Some(false) = reject_shifted_surface_trig_symbolic_base_mismatch_before_default_simplify(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_shifted_surface_trig_symbolic_base_mismatch_reject",
        );
        return None;
    }

    if let Some(false) = reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_surface_trig_power_numeric_atom_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_plain_surface_trig_power_gap_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_surface_trig_power_gap_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_hyperbolic_additive_mismatch_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_hyperbolic_additive_mismatch_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_obvious_hyperbolic_pair_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.default_simplify_hyperbolic_pair_reject");
        return None;
    }

    if let Some(false) = reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_negated_log_nonreciprocal_reject",
        );
        return None;
    }

    if has_hyperbolic_core {
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, lhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, rhs_core) {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_exact_pre_default");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, rhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, lhs_core) {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_exact_pre_default");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    let default_simplify_match = if profiling {
        let label = direct_core_default_simplify_profile_label(ctx, lhs_core, rhs_core);
        run_profiled_orchestrator_option_section(label, pair_sample.clone(), || {
            exprs_match_after_default_simplify(ctx, lhs_core, rhs_core).then_some(())
        })
        .is_some()
    } else {
        exprs_match_after_default_simplify(ctx, lhs_core, rhs_core)
    };
    if default_simplify_match {
        profile_route("rule.direct_core_equivalence.route.default_simplify_match");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if !has_tanh_core {
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, lhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, rhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, rhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_lhs");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, rhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, lhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, lhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_rhs");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    None
}

pub(super) fn expr_is_atomic_noncall(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) | Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Neg(inner) => matches!(
            ctx.get(*inner),
            Expr::Variable(_) | Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_)
        ),
        _ => false,
    }
}

pub(super) fn try_build_fast_trig_residual_identity_child_rewrite(
    ctx: &mut cas_ast::Context,
    residual_expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let residual_expr = if let Some((lhs_core, rhs_core)) =
        extract_two_term_core_difference(ctx, residual_expr)
    {
        if let Some(rewrite) =
            try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_direct_trig_product_to_sum_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_direct_trig_sum_to_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_direct_trig_double_angle_contraction_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        ) {
            return Some(rewrite);
        }
        ctx.add(Expr::Sub(lhs_core, rhs_core))
    } else {
        residual_expr
    };

    let term_count = AddView::from_expr(ctx, residual_expr).terms.len();
    if !(2..=4).contains(&term_count) {
        return None;
    }
    if !expr_contains_any_builtin(
        ctx,
        residual_expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    ) {
        return None;
    }
    if expr_contains_plain_trig_angle_identity_term(ctx, residual_expr) {
        return None;
    }

    let rewrite = try_build_exact_zero_identity_rewrite_direct(ctx, residual_expr)?;
    let zero = ctx.num(0);
    (compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal).then_some(rewrite)
}

pub(super) fn try_build_fast_multiterm_hyperbolic_residual_child_rewrite(
    ctx: &mut cas_ast::Context,
    residual_expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let term_count = AddView::from_expr(ctx, residual_expr).terms.len();
    if !(3..=4).contains(&term_count) {
        return None;
    }
    if !expr_contains_hyperbolic_builtin(ctx, residual_expr) {
        return None;
    }

    let rewrite = try_build_exact_zero_identity_rewrite(ctx, residual_expr)?;
    let zero = ctx.num(0);
    (compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal).then_some(rewrite)
}

pub(super) fn try_build_repeated_trig_phase_shift_pair_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let normalized_expr = normalize_additive_scope_expr(ctx, expr);
    let view = AddView::from_expr(ctx, normalized_expr);
    if view.terms.len() != 6 {
        return None;
    }

    if let Some(rewrite) =
        try_build_fast_repeated_trig_phase_shift_pair_zero_rewrite(ctx, &view.terms)
    {
        return Some(rewrite);
    }

    let zero = ctx.num(0);

    for first_index in 0..view.terms.len().saturating_sub(2) {
        for second_index in (first_index + 1)..view.terms.len().saturating_sub(1) {
            for third_index in (second_index + 1)..view.terms.len() {
                let first_terms = [
                    view.terms[first_index],
                    view.terms[second_index],
                    view.terms[third_index],
                ];
                let first_expr = build_signed_sum_expr(ctx, &first_terms);
                let Some(first_rewrite) =
                    try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, first_expr)
                else {
                    continue;
                };
                if compare_expr(ctx, first_rewrite.final_expr(), zero) != Ordering::Equal {
                    continue;
                }

                let remaining_terms: Vec<_> = view
                    .terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index != first_index && index != second_index && index != third_index)
                            .then_some(term)
                    })
                    .collect();
                if remaining_terms.len() != 3 {
                    continue;
                }

                let second_expr = build_signed_sum_expr(ctx, &remaining_terms);
                let Some(second_rewrite) =
                    try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, second_expr)
                else {
                    continue;
                };
                if compare_expr(ctx, second_rewrite.final_expr(), zero) != Ordering::Equal {
                    continue;
                }

                let mut rewrite = Rewrite::with_local(
                    zero,
                    first_rewrite.description.clone(),
                    normalized_expr,
                    zero,
                )
                .requires_all(first_rewrite.required_conditions.clone())
                .requires_all(second_rewrite.required_conditions.clone())
                .assume_all(first_rewrite.assumption_events.clone())
                .assume_all(second_rewrite.assumption_events.clone());

                if let Some(poly_proof) = first_rewrite.poly_proof.clone() {
                    rewrite = rewrite.poly_proof(poly_proof);
                }
                let mut substeps = if first_rewrite.substeps.is_empty() {
                    vec![build_phase_shift_zero_substep(ctx, first_expr)]
                } else {
                    first_rewrite.substeps.clone()
                };
                if second_rewrite.substeps.is_empty() {
                    substeps.push(build_phase_shift_zero_substep(ctx, second_expr));
                } else {
                    substeps.extend(second_rewrite.substeps.clone());
                }
                rewrite.substeps = substeps;

                return Some(rewrite);
            }
        }
    }

    None
}

pub(super) fn try_build_exact_zero_shared_passthrough_difference_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    if !has_plausible_shared_additive_passthrough_difference_shape(ctx, expr) {
        return None;
    }

    let (lhs_core, rhs_core) = extract_shared_additive_passthrough_difference_cores(ctx, expr)?;
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, lhs_core),
            render_expr_for_orchestrator_profile(ctx, rhs_core)
        )
    });
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, pair_sample.clone(), || Some(()));
        }
    };
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.sub_fraction",
        &pair_sample,
        || try_build_direct_sub_fraction_combination_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.sub_fraction");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.tanh_exp",
        &pair_sample,
        || try_build_direct_tanh_exp_definition_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.tanh_exp");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.trig_product_to_sum",
        &pair_sample,
        || try_build_direct_trig_product_to_sum_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.trig_product_to_sum");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.trig_sum_to_product",
        &pair_sample,
        || try_build_direct_trig_sum_to_product_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.trig_sum_to_product");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.trig_square",
        &pair_sample,
        || try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.trig_square");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.sinh_cubic",
        &pair_sample,
        || {
            try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        },
    ) {
        profile_route("rule.shared_passthrough.route.sinh_cubic");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.repeated_phase_shift_pair",
        &pair_sample,
        || try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr),
    ) {
        profile_route("rule.shared_passthrough.route.repeated_phase_shift_pair");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.phase_shift_quarter_pair",
        &pair_sample,
        || {
            try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        },
    ) {
        profile_route("rule.shared_passthrough.route.phase_shift_quarter_pair");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.safe_hyperbolic",
        &pair_sample,
        || try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.safe_hyperbolic");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.repeated_phase_shift_pair_late",
        &pair_sample,
        || try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr),
    ) {
        profile_route("rule.shared_passthrough.route.repeated_phase_shift_pair_late");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.direct_identity",
        &pair_sample,
        || try_build_exact_zero_identity_rewrite_direct(ctx, residual_expr),
    ) {
        let zero = ctx.num(0);
        if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
            profile_route("rule.shared_passthrough.route.direct_identity");
            return Some(rewrite);
        }
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.square_base_equivalence",
        &pair_sample,
        || try_build_shared_passthrough_square_base_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.square_base_equivalence");
        return Some(rewrite);
    }
    let normalized_residual = normalize_additive_scope_expr(ctx, residual_expr);

    let child_rewrite = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.tail_fast_multiterm_hyperbolic",
        &pair_sample,
        || try_build_fast_multiterm_hyperbolic_residual_child_rewrite(ctx, residual_expr),
    )
    .inspect(|_| {
        profile_route("rule.shared_passthrough.route.tail_fast_multiterm_hyperbolic");
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_safe_hyperbolic",
            &pair_sample,
            || try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_safe_hyperbolic");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_stripped_zero_log",
            &pair_sample,
            || try_build_stripped_zero_log_identity_child_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_stripped_zero_log");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_stripped_zero_log_normalized",
            &pair_sample,
            || try_build_stripped_zero_log_identity_child_rewrite(ctx, normalized_residual),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_stripped_zero_log_normalized");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_trig_residual",
            &pair_sample,
            || try_build_fast_trig_residual_identity_child_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_trig_residual");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_trig_residual_normalized",
            &pair_sample,
            || try_build_fast_trig_residual_identity_child_rewrite(ctx, normalized_residual),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_trig_residual_normalized");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_small_polynomial",
            &pair_sample,
            || try_build_fast_small_polynomial_residual_child_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_small_polynomial");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_small_polynomial_normalized",
            &pair_sample,
            || try_build_fast_small_polynomial_residual_child_rewrite(ctx, normalized_residual),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_small_polynomial_normalized");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_direct_core_equivalence",
            &pair_sample,
            || try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_direct_core_equivalence");
            profile_shared_passthrough_tail_direct_core_family(
                ctx,
                lhs_core,
                rhs_core,
                pair_sample.clone(),
            );
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_exact_zero_identity",
            &pair_sample,
            || try_build_exact_zero_identity_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_exact_zero_identity");
        })
    })?;
    let zero = ctx.num(0);

    (compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal)
        .then_some(child_rewrite)
}
