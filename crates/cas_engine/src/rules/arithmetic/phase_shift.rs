//! `arithmetic`: familia `phase_shift`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn try_rewrite_shifted_hyperbolic_pythagorean_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if is_positive_one_expr(ctx, rhs) => {
            let arg = extract_squared_hyperbolic_arg(ctx, lhs, BuiltinFn::Cosh)?;
            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
            let two = ctx.num(2);
            Some(ctx.add(Expr::Pow(sinh, two)))
        }
        Expr::Sub(lhs, rhs) if is_positive_one_expr(ctx, lhs) => {
            let arg = extract_squared_hyperbolic_arg(ctx, rhs, BuiltinFn::Cosh)?;
            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
            let two = ctx.num(2);
            let sinh_sq = ctx.add(Expr::Pow(sinh, two));
            Some(ctx.add(Expr::Neg(sinh_sq)))
        }
        Expr::Add(lhs, rhs) => {
            let sinh_sq = if is_positive_one_expr(ctx, lhs) {
                rhs
            } else if is_positive_one_expr(ctx, rhs) {
                lhs
            } else {
                return None;
            };
            let arg = extract_squared_hyperbolic_arg(ctx, sinh_sq, BuiltinFn::Sinh)?;
            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
            let two = ctx.num(2);
            Some(ctx.add(Expr::Pow(cosh, two)))
        }
        _ => None,
    }
}

pub(super) fn expr_contains_shifted_square(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
    if let Some(hit) = SHIFTED_SQUARE_GATE_MEMO.with(|m| m.borrow().get(&root).copied()) {
        return hit;
    }
    let result = expr_contains_shifted_square_uncached(ctx, root);
    SHIFTED_SQUARE_GATE_MEMO.with(|m| m.borrow_mut().insert(root, result));
    result
}

fn expr_contains_shifted_square_uncached(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
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
                    && matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _)) =>
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

fn try_extract_shifted_square_primary_variable_name(
    ctx: &cas_ast::Context,
    base: cas_ast::ExprId,
) -> Option<String> {
    let (lhs, rhs) = match ctx.get(base) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        _ => return None,
    };

    if let Some(var_name) = bare_variable_name(ctx, lhs) {
        if bare_variable_name(ctx, rhs).is_none() && !contains_named_var(ctx, rhs, &var_name) {
            return Some(var_name);
        }
    }

    if let Some(var_name) = bare_variable_name(ctx, rhs) {
        if bare_variable_name(ctx, lhs).is_none() && !contains_named_var(ctx, lhs, &var_name) {
            return Some(var_name);
        }
    }

    None
}

pub(super) fn collect_shifted_square_primary_variable_names(
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
            Expr::Pow(base, exp)
                if extract_i64_integer(ctx, *exp) == Some(2)
                    && matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _)) =>
            {
                if let Some(var_name) = try_extract_shifted_square_primary_variable_name(ctx, *base)
                {
                    names.insert(var_name);
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
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    names.into_iter().collect()
}

pub(super) fn exact_phase_shift_args_match_for_cancellation(
    ctx: &mut cas_ast::Context,
    left_arg: cas_ast::ExprId,
    right_arg: cas_ast::ExprId,
) -> bool {
    if compare_expr(ctx, left_arg, right_arg) == Ordering::Equal {
        return true;
    }

    if matches!(
        (ctx.get(left_arg), ctx.get(right_arg)),
        (Expr::Variable(_), Expr::Variable(_))
            | (Expr::Variable(_), Expr::SessionRef(_))
            | (Expr::SessionRef(_), Expr::Variable(_))
            | (Expr::SessionRef(_), Expr::SessionRef(_))
    ) {
        return false;
    }

    exprs_match_for_cancellation_leaf(ctx, left_arg, right_arg)
}

pub(super) fn maybe_trig_phase_shift_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
        && (expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Atan, BuiltinFn::Arctan])
            || expr_contains_pi_constant(ctx, expr))
}

pub(super) fn expr_has_phase_shift_signal_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_pi_constant(ctx, expr)
        || expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Atan, BuiltinFn::Arctan])
}

pub(super) fn is_surface_plain_algebraic_term_for_phase_shift(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => {}
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            _ => return false,
        }
    }
    true
}

pub(super) fn binary_add_pair_has_trig_phase_shift_shape_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    sample: Option<String>,
) -> bool {
    let lhs_has_trig = expr_contains_any_builtin(ctx, lhs, &[BuiltinFn::Sin, BuiltinFn::Cos]);
    let rhs_has_trig = expr_contains_any_builtin(ctx, rhs, &[BuiltinFn::Sin, BuiltinFn::Cos]);

    if crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        let label = match (lhs_has_trig, rhs_has_trig) {
            (true, true) => "rule.phase_shift.binary_add_match.pair_shape.both_trig",
            (true, false) => "rule.phase_shift.binary_add_match.pair_shape.rhs_non_trig",
            (false, true) => "rule.phase_shift.binary_add_match.pair_shape.lhs_non_trig",
            (false, false) => "rule.phase_shift.binary_add_match.pair_shape.both_non_trig",
        };
        let _ = run_profiled_orchestrator_option_section(label, sample, || Some(()));
    }

    lhs_has_trig && rhs_has_trig
}

pub(super) fn classify_binary_add_term_family_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> &'static str {
    if let Some((trig_fn, arg, _, _)) = extract_surface_scaled_trig_term_for_phase_shift(ctx, expr)
    {
        if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled()
            && extract_surface_supported_phase_shift_argument_subset_for_cancellation(
                ctx,
                trig_fn,
                arg,
                &[4_i64, 3_i64, 6_i64],
            )
            .is_some()
        {
            return "exact_shifted_surface";
        }
        if extract_supported_phase_shift_argument_for_cancellation(ctx, trig_fn, arg).is_some() {
            return "exact_shifted_surface";
        }
        if extract_general_phase_shift_argument_for_cancellation(ctx, arg).is_some() {
            return "general_shifted_surface";
        }
        return "plain_surface_trig";
    }

    if expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos]) {
        "other_trig"
    } else {
        "non_trig"
    }
}

pub(super) fn phase_shift_term_families_are_single_plain_against_shifted(
    lhs_family: &str,
    rhs_family: &str,
) -> bool {
    matches!(
        (lhs_family, rhs_family),
        ("plain_surface_trig", "exact_shifted_surface")
            | ("exact_shifted_surface", "plain_surface_trig")
            | ("plain_surface_trig", "general_shifted_surface")
            | ("general_shifted_surface", "plain_surface_trig")
    )
}

pub(super) fn binary_add_pair_is_surface_plain_trig_against_shift_signal_for_phase_shift(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let lhs_surface = extract_surface_scaled_trig_term_for_phase_shift(ctx, lhs);
    let rhs_surface = extract_surface_scaled_trig_term_for_phase_shift(ctx, rhs);

    match (lhs_surface, rhs_surface) {
        (Some((_, lhs_arg, _, _)), Some((_, rhs_arg, _, _))) => {
            let lhs_shifted = expr_has_phase_shift_signal_for_cancellation(ctx, lhs_arg);
            let rhs_shifted = expr_has_phase_shift_signal_for_cancellation(ctx, rhs_arg);
            lhs_shifted != rhs_shifted
        }
        _ => false,
    }
}

pub(super) fn exact_phase_shift_pair_relation_label_for_cancellation(
    ctx: &mut cas_ast::Context,
    left: ExactPhaseShiftTermData,
    right: ExactPhaseShiftTermData,
    right_is_negated_relative_to_left: bool,
) -> &'static str {
    let (left_arg, left_coeff, left_kind, left_sin_sign, left_cos_sign) = left;
    let (right_arg, right_coeff, right_kind, right_sin_sign, right_cos_sign) = right;

    let arg_matches = exprs_match_for_cancellation(ctx, left_arg, right_arg);
    let expected_right_sin_sign = if right_is_negated_relative_to_left {
        -left_sin_sign
    } else {
        left_sin_sign
    };
    let expected_right_cos_sign = if right_is_negated_relative_to_left {
        -left_cos_sign
    } else {
        left_cos_sign
    };

    if !arg_matches {
        "arg_mismatch"
    } else if right_sin_sign != expected_right_sin_sign || right_cos_sign != expected_right_cos_sign
    {
        "sign_mismatch"
    } else {
        let (left_sin_coeff, left_cos_coeff) =
            exact_phase_shift_linear_signature_for_cancellation(ctx, left_coeff, left_kind);
        let (right_sin_coeff, right_cos_coeff) =
            exact_phase_shift_linear_signature_for_cancellation(ctx, right_coeff, right_kind);

        if exprs_match_for_cancellation(ctx, left_sin_coeff, right_sin_coeff)
            && exprs_match_for_cancellation(ctx, left_cos_coeff, right_cos_coeff)
        {
            "signature_match"
        } else {
            "coeff_mismatch"
        }
    }
}

fn exact_phase_shift_pair_relation_matches_for_cancellation(
    ctx: &mut cas_ast::Context,
    left: ExactPhaseShiftTermData,
    right: ExactPhaseShiftTermData,
    right_is_negated_relative_to_left: bool,
) -> bool {
    exact_phase_shift_pair_relation_label_for_cancellation(
        ctx,
        left,
        right,
        right_is_negated_relative_to_left,
    ) == "signature_match"
}

pub(super) fn binary_add_pair_has_productive_phase_shift_term_family_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let lhs_family = classify_binary_add_term_family_for_phase_shift(ctx, lhs);
    let rhs_family = classify_binary_add_term_family_for_phase_shift(ctx, rhs);
    if phase_shift_term_families_are_single_plain_against_shifted(lhs_family, rhs_family) {
        return false;
    }

    match (lhs_family, rhs_family) {
        ("other_trig", "other_trig") => false,
        ("exact_shifted_surface", "exact_shifted_surface") => {
            if !profiling {
                let raw_subset_base_arg = |ctx: &mut cas_ast::Context, expr: cas_ast::ExprId| {
                    let (trig_fn, raw_arg, _, _) =
                        extract_surface_scaled_trig_term_for_phase_shift(ctx, expr)?;
                    let (base_arg, _kind, _subtract_shift) =
                        extract_surface_supported_phase_shift_argument_subset_for_cancellation(
                            ctx,
                            trig_fn,
                            raw_arg,
                            &[4_i64, 3_i64, 6_i64],
                        )?;
                    Some(base_arg)
                };

                if let Some((left_base_arg, right_base_arg)) =
                    raw_subset_base_arg(ctx, lhs).zip(raw_subset_base_arg(ctx, rhs))
                {
                    return compare_expr(ctx, left_base_arg, right_base_arg) == Ordering::Equal;
                }

                return true;
            }

            let Some(left_exact) = extract_exact_phase_shift_term_data_for_cancellation(ctx, lhs)
            else {
                return true;
            };
            let Some(right_exact) = extract_exact_phase_shift_term_data_for_cancellation(ctx, rhs)
            else {
                return true;
            };

            exact_phase_shift_pair_relation_matches_for_cancellation(
                ctx,
                left_exact,
                right_exact,
                true,
            )
        }
        _ => true,
    }
}

fn extract_scaled_sin_or_cos_linear_term_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, cas_ast::ExprId)> {
    if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, expr) {
        return Some((trig_fn, arg, ctx.num(1)));
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_term = None;
    let mut coeff_factors = Vec::new();
    for factor in factors {
        if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor) {
            if trig_term.is_some() {
                return None;
            }
            trig_term = Some((trig_fn, arg));
        } else {
            coeff_factors.push(factor);
        }
    }

    let (trig_fn, arg) = trig_term?;
    if coeff_factors.is_empty() {
        return None;
    }

    let coeff = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));
    Some((trig_fn, arg, coeff))
}

fn extract_signed_scaled_sin_or_cos_linear_term_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    sign: Sign,
) -> Option<(BuiltinFn, cas_ast::ExprId, cas_ast::ExprId, i8)> {
    let (trig_fn, arg, coeff) = extract_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, expr)?;
    let sign = match sign {
        Sign::Pos => 1_i8,
        Sign::Neg => -1_i8,
    };
    let (coeff, sign) = normalize_phase_shift_coefficient_sign_for_cancellation(ctx, coeff, sign);
    Some((trig_fn, arg, coeff, sign))
}

fn collapse_surface_numeric_product_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return expr;
    }

    let mut product: Option<BigRational> = None;
    for factor in factors {
        let Expr::Number(value) = ctx.get(factor) else {
            return expr;
        };
        product = Some(match product {
            Some(acc) => acc * value.clone(),
            None => value.clone(),
        });
    }

    let Some(product) = product else {
        return expr;
    };

    ctx.add(Expr::Number(product))
}

pub(super) fn extract_surface_scaled_trig_term_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, cas_ast::ExprId, i8)> {
    let (sign, positive_expr) = if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, expr)
    {
        (-1_i8, inner)
    } else {
        (1_i8, expr)
    };

    let (trig_fn, arg, coeff) =
        extract_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr)?;
    let coeff = collapse_surface_numeric_product_for_phase_shift(ctx, coeff);
    Some((trig_fn, arg, coeff, sign))
}

fn strip_surface_sqrt_three_factor_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let positive_expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    let three = ctx.num(3);
    let sqrt_three = ctx.call_builtin(BuiltinFn::Sqrt, vec![three]);
    strip_common_factor_from_term(ctx, positive_expr, sqrt_three)
}

fn matches_surface_sqrt_three_factor_ratio_for_phase_shift(
    ctx: &mut cas_ast::Context,
    scaled_coeff: cas_ast::ExprId,
    base_coeff: cas_ast::ExprId,
) -> bool {
    let Some(residual) = strip_surface_sqrt_three_factor_for_phase_shift(ctx, scaled_coeff) else {
        return false;
    };
    exprs_match_for_cancellation(ctx, residual, base_coeff)
}

fn kind_detect_numeric_ratio_fast_reject_for_phase_shift(
    ctx: &cas_ast::Context,
    lhs_coeff: cas_ast::ExprId,
    rhs_coeff: cas_ast::ExprId,
) -> bool {
    matches!(ctx.get(lhs_coeff), Expr::Number(_)) && matches!(ctx.get(rhs_coeff), Expr::Number(_))
}

fn normalize_phase_shift_coefficient_sign_for_cancellation(
    ctx: &mut cas_ast::Context,
    coeff: cas_ast::ExprId,
    sign: i8,
) -> (cas_ast::ExprId, i8) {
    if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, coeff) {
        (inner, -sign)
    } else {
        (coeff, sign)
    }
}

fn is_atan_call_for_phase_shift(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
}

pub(super) fn extract_general_phase_shift_argument_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, bool)> {
    let normalized = rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, expr);
    match ctx.get(normalized).clone() {
        Expr::Add(left, right) => {
            if is_atan_call_for_phase_shift(ctx, left) {
                Some((right, left, false))
            } else if is_atan_call_for_phase_shift(ctx, right) {
                Some((left, right, false))
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            is_atan_call_for_phase_shift(ctx, right).then_some((left, right, true))
        }
        _ => None,
    }
}

pub(super) fn split_linear_angle_term_for_phase_shift_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (BigRational, cas_ast::ExprId) {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (coeff, base) = split_linear_angle_term_for_phase_shift_cancellation(ctx, *inner);
            (-coeff, base)
        }
        Expr::Mul(left, right) => {
            if let Expr::Number(n) = ctx.get(*left) {
                (n.clone(), *right)
            } else if let Expr::Number(n) = ctx.get(*right) {
                (n.clone(), *left)
            } else {
                (BigRational::from_integer(1.into()), expr)
            }
        }
        Expr::Number(n) => (n.clone(), expr),
        _ => (BigRational::from_integer(1.into()), expr),
    }
}

fn combine_linear_angle_terms_for_phase_shift_cancellation(
    ctx: &mut cas_ast::Context,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
    subtract: bool,
) -> cas_ast::ExprId {
    let (left_coeff, left_base) = split_linear_angle_term_for_phase_shift_cancellation(ctx, left);
    let (right_coeff, right_base) =
        split_linear_angle_term_for_phase_shift_cancellation(ctx, right);
    if compare_expr(ctx, left_base, right_base) != Ordering::Equal {
        return if subtract {
            ctx.add(Expr::Sub(left, right))
        } else {
            ctx.add(Expr::Add(left, right))
        };
    }

    let coeff = if subtract {
        left_coeff - right_coeff
    } else {
        left_coeff + right_coeff
    };
    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    let neg_one = -one.clone();

    if coeff == zero {
        return ctx.num(0);
    }
    if coeff == one {
        return left_base;
    }
    if coeff == neg_one {
        return ctx.add(Expr::Neg(left_base));
    }

    let coeff_id = ctx.add(Expr::Number(coeff));
    smart_mul(ctx, coeff_id, left_base)
}

fn rewrite_linear_angle_expr_for_phase_shift_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let rewritten_left = rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, left);
            let rewritten_right =
                rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, right);
            combine_linear_angle_terms_for_phase_shift_cancellation(
                ctx,
                rewritten_left,
                rewritten_right,
                false,
            )
        }
        Expr::Sub(left, right) => {
            let rewritten_left = rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, left);
            let rewritten_right =
                rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, right);
            combine_linear_angle_terms_for_phase_shift_cancellation(
                ctx,
                rewritten_left,
                rewritten_right,
                true,
            )
        }
        Expr::Neg(inner) => {
            let rewritten_inner =
                rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, inner);
            if rewritten_inner == inner {
                expr
            } else {
                ctx.add(Expr::Neg(rewritten_inner))
            }
        }
        _ => expr,
    }
}

fn extract_surface_pi_shift_denominator_for_cancellation(
    ctx: &cas_ast::Context,
    shift: cas_ast::ExprId,
) -> Option<i64> {
    match ctx.get(shift) {
        Expr::Div(left, right) => {
            if matches!(ctx.get(*left), Expr::Constant(cas_ast::Constant::Pi)) {
                extract_i64_integer(ctx, *right)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn matches_surface_phase_shift_constant_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    shift: cas_ast::ExprId,
) -> bool {
    if compare_expr(ctx, expr, shift) == Ordering::Equal {
        return true;
    }

    let Some(denominator) = extract_surface_pi_shift_denominator_for_cancellation(ctx, shift)
    else {
        return false;
    };
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let unit_fraction = ctx.add(Expr::Number(BigRational::new(1.into(), denominator.into())));
    let numerator_is_surface_pi =
        |ctx: &cas_ast::Context, expr: cas_ast::ExprId| match ctx.get(expr) {
            Expr::Constant(cas_ast::Constant::Pi) => true,
            Expr::Mul(left, right) => {
                (compare_expr(ctx, *left, pi) == Ordering::Equal
                    && extract_i64_integer(ctx, *right) == Some(1))
                    || (compare_expr(ctx, *right, pi) == Ordering::Equal
                        && extract_i64_integer(ctx, *left) == Some(1))
            }
            _ => false,
        };

    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            (compare_expr(ctx, *left, pi) == Ordering::Equal
                && compare_expr(ctx, *right, unit_fraction) == Ordering::Equal)
                || (compare_expr(ctx, *right, pi) == Ordering::Equal
                    && compare_expr(ctx, *left, unit_fraction) == Ordering::Equal)
        }
        Expr::Div(numerator, denominator_expr) => {
            extract_i64_integer(ctx, *denominator_expr) == Some(denominator)
                && numerator_is_surface_pi(ctx, *numerator)
        }
        _ => false,
    }
}

fn extract_surface_supported_phase_shift_argument_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    shift: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, bool)> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if matches_surface_phase_shift_constant_for_cancellation(ctx, left, shift) {
                Some((right, false))
            } else if matches_surface_phase_shift_constant_for_cancellation(ctx, right, shift) {
                Some((left, false))
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            matches_surface_phase_shift_constant_for_cancellation(ctx, right, shift)
                .then_some((left, true))
        }
        _ => None,
    }
}

fn extract_surface_supported_phase_shift_argument_subset_for_cancellation(
    ctx: &mut cas_ast::Context,
    trig_fn: BuiltinFn,
    expr: cas_ast::ExprId,
    denominators: &[i64],
) -> Option<(cas_ast::ExprId, PhaseShiftKindForCancellation, bool)> {
    for &denominator in denominators {
        let shift = build_pi_over_for_cancellation(ctx, denominator);
        if let Some((base_arg, subtract_shift)) =
            extract_surface_supported_phase_shift_argument_for_cancellation(ctx, expr, shift)
        {
            let kind = phase_shift_kind_for_cancellation(trig_fn, denominator)?;
            return Some((base_arg, kind, subtract_shift));
        }
    }
    None
}

fn phase_shift_kind_for_cancellation(
    trig_fn: BuiltinFn,
    denominator: i64,
) -> Option<PhaseShiftKindForCancellation> {
    match (trig_fn, denominator) {
        (BuiltinFn::Sin, 4) | (BuiltinFn::Cos, 4) => Some(PhaseShiftKindForCancellation::Quarter),
        (BuiltinFn::Sin, 3) | (BuiltinFn::Cos, 6) => Some(PhaseShiftKindForCancellation::Third),
        (BuiltinFn::Sin, 6) | (BuiltinFn::Cos, 3) => Some(PhaseShiftKindForCancellation::Sixth),
        _ => None,
    }
}

pub(super) fn extract_supported_phase_shift_argument_for_cancellation(
    ctx: &mut cas_ast::Context,
    trig_fn: BuiltinFn,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, PhaseShiftKindForCancellation, bool)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let sample = profiling.then(|| render_expr_for_orchestrator_profile(ctx, expr));

    if profiling {
        for denominator in [4_i64, 3_i64, 6_i64] {
            let shift = build_pi_over_for_cancellation(ctx, denominator);
            let _ = run_profiled_orchestrator_option_section(
                "rule.phase_shift.supported_arg.raw_structural_match",
                sample.clone(),
                || {
                    extract_surface_supported_phase_shift_argument_for_cancellation(
                        ctx, expr, shift,
                    )
                },
            );
        }
    }

    let normalized = if profiling {
        run_profiled_orchestrator_section(
            "rule.phase_shift.supported_arg.rewrite_linear_angle",
            sample.clone(),
            || rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, expr),
            |_| true,
        )
    } else {
        rewrite_linear_angle_expr_for_phase_shift_cancellation(ctx, expr)
    };

    for denominator in [4_i64, 3_i64, 6_i64] {
        let shift = build_pi_over_for_cancellation(ctx, denominator);
        let structural_match = if profiling {
            run_profiled_orchestrator_option_section(
                "rule.phase_shift.supported_arg.normalized_structural_match",
                sample.clone(),
                || {
                    extract_surface_supported_phase_shift_argument_for_cancellation(
                        ctx, normalized, shift,
                    )
                },
            )
        } else {
            extract_surface_supported_phase_shift_argument_for_cancellation(ctx, normalized, shift)
        };
        let matched = structural_match.or_else(|| {
            if profiling {
                run_profiled_orchestrator_option_section(
                    phase_shift_supported_arg_fallback_profile_label(denominator),
                    sample.clone(),
                    || match ctx.get(normalized).clone() {
                        Expr::Add(left, right) => {
                            if exprs_match_after_default_simplify(ctx, left, shift) {
                                Some((right, false))
                            } else if exprs_match_after_default_simplify(ctx, right, shift) {
                                Some((left, false))
                            } else {
                                None
                            }
                        }
                        Expr::Sub(left, right) => {
                            exprs_match_after_default_simplify(ctx, right, shift)
                                .then_some((left, true))
                        }
                        _ => None,
                    },
                )
            } else {
                match ctx.get(normalized).clone() {
                    Expr::Add(left, right) => {
                        if exprs_match_after_default_simplify(ctx, left, shift) {
                            Some((right, false))
                        } else if exprs_match_after_default_simplify(ctx, right, shift) {
                            Some((left, false))
                        } else {
                            None
                        }
                    }
                    Expr::Sub(left, right) => exprs_match_after_default_simplify(ctx, right, shift)
                        .then_some((left, true)),
                    _ => None,
                }
            }
        });

        let Some((base_arg, subtract_shift)) = matched else {
            continue;
        };

        let kind = phase_shift_kind_for_cancellation(trig_fn, denominator)?;

        return Some((base_arg, kind, subtract_shift));
    }

    None
}

pub(super) fn extract_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(
    cas_ast::ExprId,
    cas_ast::ExprId,
    PhaseShiftKindForCancellation,
    i8,
    i8,
)> {
    match extract_exact_phase_shift_linear_combination_for_cancellation(ctx, expr) {
        ExactPhaseShiftLinearCombinationExtraction::Exact(data) => Some((
            data.arg,
            data.coeff,
            data.kind,
            data.sin_sign,
            data.cos_sign,
        )),
        ExactPhaseShiftLinearCombinationExtraction::NotLinear
        | ExactPhaseShiftLinearCombinationExtraction::LinearButNotExact => None,
    }
}

fn extract_exact_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> ExactPhaseShiftLinearCombinationExtraction {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let sample = profiling.then(|| render_expr_for_orchestrator_profile(ctx, expr));
    let (sin_arg, sin_coeff, cos_coeff, sin_sign, cos_sign) = if profiling {
        let Some(data) = run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.weighted_terms",
            sample.clone(),
            || extract_weighted_phase_shift_linear_combination_for_cancellation(ctx, expr),
        ) else {
            return ExactPhaseShiftLinearCombinationExtraction::NotLinear;
        };
        data
    } else {
        let Some(data) =
            extract_weighted_phase_shift_linear_combination_for_cancellation(ctx, expr)
        else {
            return ExactPhaseShiftLinearCombinationExtraction::NotLinear;
        };
        data
    };

    let Some((coeff, kind)) =
        detect_exact_phase_shift_kind_for_cancellation(ctx, sin_coeff, cos_coeff, sample)
    else {
        return ExactPhaseShiftLinearCombinationExtraction::LinearButNotExact;
    };

    ExactPhaseShiftLinearCombinationExtraction::Exact(ExactPhaseShiftLinearCombinationData {
        arg: sin_arg,
        coeff,
        kind,
        sin_sign,
        cos_sign,
    })
}

fn detect_exact_phase_shift_kind_for_cancellation(
    ctx: &mut cas_ast::Context,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sample: Option<String>,
) -> Option<(cas_ast::ExprId, PhaseShiftKindForCancellation)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let three = ctx.num(3);
    let sqrt_three = ctx.call_builtin(BuiltinFn::Sqrt, vec![three]);
    let quarter_match = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.quarter",
            sample.clone(),
            || (compare_expr(ctx, sin_coeff, cos_coeff) == Ordering::Equal).then_some(()),
        )
        .is_some()
    } else {
        compare_expr(ctx, sin_coeff, cos_coeff) == Ordering::Equal
    };
    if quarter_match {
        return Some((sin_coeff, PhaseShiftKindForCancellation::Quarter));
    }

    let numeric_ratio_reject = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.numeric_ratio_fast_reject",
            sample.clone(),
            || {
                kind_detect_numeric_ratio_fast_reject_for_phase_shift(ctx, sin_coeff, cos_coeff)
                    .then_some(())
            },
        )
        .is_some()
    } else {
        kind_detect_numeric_ratio_fast_reject_for_phase_shift(ctx, sin_coeff, cos_coeff)
    };
    if numeric_ratio_reject {
        return None;
    }

    let sin_times_sqrt_three = smart_mul(ctx, sin_coeff, sqrt_three);
    let cos_times_sqrt_three = smart_mul(ctx, cos_coeff, sqrt_three);
    let third_surface_match = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.third.surface_match",
            sample.clone(),
            || exprs_match_for_cancellation(ctx, sin_times_sqrt_three, cos_coeff).then_some(()),
        )
        .is_some()
    } else {
        false
    };
    let sixth_surface_match = if third_surface_match {
        false
    } else if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.sixth.surface_match",
            sample.clone(),
            || exprs_match_for_cancellation(ctx, cos_times_sqrt_three, sin_coeff).then_some(()),
        )
        .is_some()
    } else {
        exprs_match_for_cancellation(ctx, cos_times_sqrt_three, sin_coeff)
    };
    if profiling {
        let _ = run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.third.sqrt3_factor_ratio",
            sample.clone(),
            || {
                matches_surface_sqrt_three_factor_ratio_for_phase_shift(ctx, cos_coeff, sin_coeff)
                    .then_some(())
            },
        );
    }
    let third_match = if third_surface_match {
        true
    } else if sixth_surface_match {
        false
    } else if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.third.default_simplify_fallback",
            sample.clone(),
            || {
                exprs_match_after_default_simplify(ctx, sin_times_sqrt_three, cos_coeff)
                    .then_some(())
            },
        )
        .is_some()
    } else {
        exprs_match_after_default_simplify(ctx, sin_times_sqrt_three, cos_coeff)
    };
    if third_match {
        return Some((sin_coeff, PhaseShiftKindForCancellation::Third));
    }

    if profiling {
        let _ = run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.sixth.sqrt3_factor_ratio",
            sample.clone(),
            || {
                matches_surface_sqrt_three_factor_ratio_for_phase_shift(ctx, sin_coeff, cos_coeff)
                    .then_some(())
            },
        );
    }
    let sixth_match = if sixth_surface_match {
        true
    } else if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_extract.kind_detect.sixth.default_simplify_fallback",
            sample.clone(),
            || {
                exprs_match_after_default_simplify(ctx, cos_times_sqrt_three, sin_coeff)
                    .then_some(())
            },
        )
        .is_some()
    } else {
        exprs_match_after_default_simplify(ctx, cos_times_sqrt_three, sin_coeff)
    };
    sixth_match.then_some((cos_coeff, PhaseShiftKindForCancellation::Sixth))
}

pub(super) fn build_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    coeff: cas_ast::ExprId,
    arg: cas_ast::ExprId,
    kind: PhaseShiftKindForCancellation,
    sin_sign: i8,
    cos_sign: i8,
) -> cas_ast::ExprId {
    let three = ctx.num(3);
    let sqrt_three = ctx.call_builtin(BuiltinFn::Sqrt, vec![three]);
    let (sin_coeff, cos_coeff) = match kind {
        PhaseShiftKindForCancellation::Quarter => (coeff, coeff),
        PhaseShiftKindForCancellation::Third => (coeff, smart_mul(ctx, coeff, sqrt_three)),
        PhaseShiftKindForCancellation::Sixth => (smart_mul(ctx, coeff, sqrt_three), coeff),
    };

    build_weighted_phase_shift_linear_combination_for_cancellation(
        ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
    )
}

fn build_scaled_shifted_phase_term_for_cancellation(
    ctx: &mut cas_ast::Context,
    coeff: cas_ast::ExprId,
    arg: cas_ast::ExprId,
    trig_fn: BuiltinFn,
    denominator: i64,
    subtract_shift: bool,
    negate: bool,
) -> cas_ast::ExprId {
    let shift = build_pi_over_for_cancellation(ctx, denominator);
    let shifted_arg = if subtract_shift {
        ctx.add(Expr::Sub(arg, shift))
    } else {
        ctx.add(Expr::Add(arg, shift))
    };
    let trig_call = ctx.call_builtin(trig_fn, vec![shifted_arg]);
    let scaled = smart_mul(ctx, coeff, trig_call);
    if negate {
        ctx.add(Expr::Neg(scaled))
    } else {
        scaled
    }
}

pub(super) fn generate_phase_shift_term_candidates_for_cancellation(
    ctx: &mut cas_ast::Context,
    coeff: cas_ast::ExprId,
    arg: cas_ast::ExprId,
    kind: PhaseShiftKindForCancellation,
) -> Vec<cas_ast::ExprId> {
    let amplitude = match kind {
        PhaseShiftKindForCancellation::Quarter => {
            let two = ctx.num(2);
            ctx.call_builtin(BuiltinFn::Sqrt, vec![two])
        }
        PhaseShiftKindForCancellation::Sixth | PhaseShiftKindForCancellation::Third => ctx.num(2),
    };
    let scaled_coeff = smart_mul(ctx, coeff, amplitude);
    let specs: &[(BuiltinFn, i64)] = match kind {
        PhaseShiftKindForCancellation::Quarter => &[(BuiltinFn::Sin, 4), (BuiltinFn::Cos, 4)],
        PhaseShiftKindForCancellation::Third => &[(BuiltinFn::Sin, 3), (BuiltinFn::Cos, 6)],
        PhaseShiftKindForCancellation::Sixth => &[(BuiltinFn::Sin, 6), (BuiltinFn::Cos, 3)],
    };

    let mut candidates = Vec::new();
    for (trig_fn, denominator) in specs {
        for (subtract_shift, negate) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let candidate = build_scaled_shifted_phase_term_for_cancellation(
                ctx,
                scaled_coeff,
                arg,
                *trig_fn,
                *denominator,
                subtract_shift,
                negate,
            );
            if candidates
                .iter()
                .all(|existing| compare_expr(ctx, *existing, candidate) != Ordering::Equal)
            {
                candidates.push(candidate);
            }
        }
    }

    candidates
}

pub(super) fn extract_exact_phase_shift_term_data_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(
    cas_ast::ExprId,
    cas_ast::ExprId,
    PhaseShiftKindForCancellation,
    i8,
    i8,
)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let sample = profiling.then(|| render_expr_for_orchestrator_profile(ctx, expr));
    let (global_sign, positive_expr) =
        if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, expr) {
            (-1_i8, inner)
        } else {
            (1_i8, expr)
        };

    let mut trig_factor = extract_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr);
    let mut has_sqrt_two = false;
    let mut coeff_factors = Vec::new();

    if trig_factor.is_none() {
        let scanned = if profiling {
            run_profiled_orchestrator_option_section(
                "rule.phase_shift.exact_target_extract.factor_partition",
                sample.clone(),
                || {
                    let factors = flatten_mul_chain(ctx, positive_expr);
                    if factors.len() < 2 {
                        return None;
                    }

                    let mut local_trig_factor = None;
                    let mut local_has_sqrt_two = false;
                    let mut local_coeff_factors = Vec::new();

                    for factor in factors {
                        if local_trig_factor.is_none() {
                            if let Some((trig_fn, arg)) =
                                extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor)
                            {
                                local_trig_factor = Some((trig_fn, arg));
                                continue;
                            }
                        }

                        if !local_has_sqrt_two && is_sqrt_two_for_cancellation(ctx, factor) {
                            local_has_sqrt_two = true;
                            continue;
                        }

                        local_coeff_factors.push(factor);
                    }

                    Some((local_trig_factor, local_has_sqrt_two, local_coeff_factors))
                },
            )?
        } else {
            let factors = flatten_mul_chain(ctx, positive_expr);
            if factors.len() < 2 {
                return None;
            }

            let mut local_trig_factor = None;
            let mut local_has_sqrt_two = false;
            let mut local_coeff_factors = Vec::new();

            for factor in factors {
                if local_trig_factor.is_none() {
                    if let Some((trig_fn, arg)) =
                        extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor)
                    {
                        local_trig_factor = Some((trig_fn, arg));
                        continue;
                    }
                }

                if !local_has_sqrt_two && is_sqrt_two_for_cancellation(ctx, factor) {
                    local_has_sqrt_two = true;
                    continue;
                }

                local_coeff_factors.push(factor);
            }

            (local_trig_factor, local_has_sqrt_two, local_coeff_factors)
        };

        trig_factor = scanned.0;
        has_sqrt_two = scanned.1;
        coeff_factors = scanned.2;
    }

    let (trig_fn, raw_arg) = trig_factor?;
    let has_pi_shift = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.exact_target_extract.pi_arg_gate",
            sample.clone(),
            || expr_contains_pi_constant(ctx, raw_arg).then_some(()),
        )
        .is_some()
    } else {
        expr_contains_pi_constant(ctx, raw_arg)
    };
    if !has_pi_shift {
        return None;
    }
    let raw_supported_arg = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.exact_target_extract.raw_supported_arg_match",
            sample.clone(),
            || {
                extract_surface_supported_phase_shift_argument_subset_for_cancellation(
                    ctx,
                    trig_fn,
                    raw_arg,
                    &[4_i64, 3_i64, 6_i64],
                )
            },
        )
    } else {
        extract_surface_supported_phase_shift_argument_subset_for_cancellation(
            ctx,
            trig_fn,
            raw_arg,
            &[4_i64, 3_i64, 6_i64],
        )
    };
    let (base_arg, kind, subtract_shift) = if let Some(raw_supported_arg) = raw_supported_arg {
        raw_supported_arg
    } else if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.exact_target_extract.shifted_arg_match",
            sample.clone(),
            || extract_supported_phase_shift_argument_for_cancellation(ctx, trig_fn, raw_arg),
        )?
    } else {
        extract_supported_phase_shift_argument_for_cancellation(ctx, trig_fn, raw_arg)?
    };
    let coeff_expr = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));

    let coeff = match kind {
        PhaseShiftKindForCancellation::Quarter => {
            if profiling {
                run_profiled_orchestrator_section(
                    "rule.phase_shift.exact_target_extract.coeff_normalize.quarter",
                    sample.clone(),
                    || {
                        if has_sqrt_two {
                            run_profiled_orchestrator_section(
                                "rule.phase_shift.exact_target_extract.coeff_normalize.quarter.sqrt_two_passthrough",
                                sample.clone(),
                                || coeff_expr,
                                |_| true,
                            )
                        } else if let Some(fast_divided) =
                            divide_by_sqrt_two_fast_for_cancellation(ctx, coeff_expr)
                        {
                            run_profiled_orchestrator_section(
                                "rule.phase_shift.exact_target_extract.coeff_normalize.quarter.fast_divide_by_sqrt_two",
                                sample.clone(),
                                || fast_divided,
                                |_| true,
                            )
                        } else {
                            run_profiled_orchestrator_section(
                                "rule.phase_shift.exact_target_extract.coeff_normalize.quarter.divide_sqrt_two_fallback",
                                sample.clone(),
                                || {
                                    let two = ctx.num(2);
                                    let sqrt_two = ctx.call_builtin(BuiltinFn::Sqrt, vec![two]);
                                    let divided = ctx.add(Expr::Div(coeff_expr, sqrt_two));
                                    run_default_simplify(ctx, divided)
                                },
                                |_| true,
                            )
                        }
                    },
                    |_| true,
                )
            } else if has_sqrt_two {
                coeff_expr
            } else if let Some(fast_divided) =
                divide_by_sqrt_two_fast_for_cancellation(ctx, coeff_expr)
            {
                fast_divided
            } else {
                let two = ctx.num(2);
                let sqrt_two = ctx.call_builtin(BuiltinFn::Sqrt, vec![two]);
                let divided = ctx.add(Expr::Div(coeff_expr, sqrt_two));
                run_default_simplify(ctx, divided)
            }
        }
        PhaseShiftKindForCancellation::Sixth | PhaseShiftKindForCancellation::Third => {
            if profiling {
                run_profiled_orchestrator_section(
                    "rule.phase_shift.exact_target_extract.coeff_normalize.third_or_sixth",
                    sample.clone(),
                    || {
                        if let Some(stripped) =
                            split_out_small_integer_factor_for_cancellation(ctx, coeff_expr, 2)
                        {
                            run_profiled_orchestrator_section(
                                "rule.phase_shift.exact_target_extract.coeff_normalize.third_or_sixth.integer_factor_strip",
                                sample.clone(),
                                || stripped,
                                |_| true,
                            )
                        } else {
                            run_profiled_orchestrator_section(
                                "rule.phase_shift.exact_target_extract.coeff_normalize.third_or_sixth.divide_by_two_fallback",
                                sample.clone(),
                                || {
                                    let two = ctx.num(2);
                                    let divided = ctx.add(Expr::Div(coeff_expr, two));
                                    run_default_simplify(ctx, divided)
                                },
                                |_| true,
                            )
                        }
                    },
                    |_| true,
                )
            } else if let Some(stripped) =
                split_out_small_integer_factor_for_cancellation(ctx, coeff_expr, 2)
            {
                stripped
            } else {
                let two = ctx.num(2);
                let divided = ctx.add(Expr::Div(coeff_expr, two));
                run_default_simplify(ctx, divided)
            }
        }
    };

    let (sin_sign, cos_sign) = match (trig_fn, subtract_shift) {
        (BuiltinFn::Sin, false) => (global_sign, global_sign),
        (BuiltinFn::Sin, true) => (global_sign, -global_sign),
        (BuiltinFn::Cos, false) => (-global_sign, global_sign),
        (BuiltinFn::Cos, true) => (global_sign, global_sign),
        _ => return None,
    };

    Some((base_arg, coeff, kind, sin_sign, cos_sign))
}

fn extract_structural_exact_quarter_phase_shift_term_data_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, i8, i8)> {
    let (global_sign, positive_expr) =
        if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, expr) {
            (-1_i8, inner)
        } else {
            (1_i8, expr)
        };

    let mut trig_factor = extract_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr);
    let mut has_sqrt_two = false;
    let mut coeff_factors = Vec::new();

    if trig_factor.is_none() {
        let factors = flatten_mul_chain(ctx, positive_expr);
        if factors.len() < 2 {
            return None;
        }

        for factor in factors {
            if trig_factor.is_none() {
                if let Some((trig_fn, arg)) =
                    extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor)
                {
                    trig_factor = Some((trig_fn, arg));
                    continue;
                }
            }

            if !has_sqrt_two && is_sqrt_two_for_cancellation(ctx, factor) {
                has_sqrt_two = true;
                continue;
            }

            coeff_factors.push(factor);
        }
    }

    if !has_sqrt_two {
        return None;
    }

    let (trig_fn, raw_arg) = trig_factor?;
    let quarter_shift = build_pi_over_for_cancellation(ctx, 4);
    let (base_arg, subtract_shift) = match ctx.get(raw_arg).clone() {
        Expr::Add(left, right) => {
            if compare_expr(ctx, left, quarter_shift) == Ordering::Equal {
                (right, false)
            } else if compare_expr(ctx, right, quarter_shift) == Ordering::Equal {
                (left, false)
            } else {
                return None;
            }
        }
        Expr::Sub(left, right) => {
            if compare_expr(ctx, right, quarter_shift) == Ordering::Equal {
                (left, true)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let coeff = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));

    let (sin_sign, cos_sign) = match (trig_fn, subtract_shift) {
        (BuiltinFn::Sin, false) => (global_sign, global_sign),
        (BuiltinFn::Sin, true) => (global_sign, -global_sign),
        (BuiltinFn::Cos, false) => (-global_sign, global_sign),
        (BuiltinFn::Cos, true) => (global_sign, global_sign),
        _ => return None,
    };

    Some((base_arg, coeff, sin_sign, cos_sign))
}

pub(super) fn extract_structural_exact_phase_shift_term_data_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(
    cas_ast::ExprId,
    cas_ast::ExprId,
    PhaseShiftKindForCancellation,
    i8,
    i8,
)> {
    let (global_sign, positive_expr) =
        if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, expr) {
            (-1_i8, inner)
        } else {
            (1_i8, expr)
        };

    let mut trig_factor = extract_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr);
    let mut has_sqrt_two = false;
    let mut coeff_factors = Vec::new();

    if trig_factor.is_none() {
        let factors = flatten_mul_chain(ctx, positive_expr);
        if factors.len() < 2 {
            return None;
        }

        for factor in factors {
            if trig_factor.is_none() {
                if let Some((trig_fn, arg)) =
                    extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor)
                {
                    trig_factor = Some((trig_fn, arg));
                    continue;
                }
            }

            if !has_sqrt_two && is_sqrt_two_for_cancellation(ctx, factor) {
                has_sqrt_two = true;
                continue;
            }

            coeff_factors.push(factor);
        }
    }

    let (trig_fn, raw_arg) = trig_factor?;
    let mut matched = None;
    for denominator in [4_i64, 3_i64, 6_i64] {
        let shift = build_pi_over_for_cancellation(ctx, denominator);
        if let Some((base_arg, subtract_shift)) =
            extract_surface_supported_phase_shift_argument_for_cancellation(ctx, raw_arg, shift)
        {
            let kind = phase_shift_kind_for_cancellation(trig_fn, denominator)?;
            matched = Some((base_arg, kind, subtract_shift));
            break;
        }
    }

    let (base_arg, kind, subtract_shift) = matched?;
    let coeff_expr = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));

    let coeff = match kind {
        PhaseShiftKindForCancellation::Quarter => {
            if has_sqrt_two {
                coeff_expr
            } else {
                let two = ctx.num(2);
                let sqrt_two = ctx.call_builtin(BuiltinFn::Sqrt, vec![two]);
                let divided = ctx.add(Expr::Div(coeff_expr, sqrt_two));
                run_default_simplify(ctx, divided)
            }
        }
        PhaseShiftKindForCancellation::Sixth | PhaseShiftKindForCancellation::Third => {
            if let Some(stripped) =
                split_out_small_integer_factor_for_cancellation(ctx, coeff_expr, 2)
            {
                stripped
            } else {
                let two = ctx.num(2);
                let divided = ctx.add(Expr::Div(coeff_expr, two));
                run_default_simplify(ctx, divided)
            }
        }
    };

    let (sin_sign, cos_sign) = match (trig_fn, subtract_shift) {
        (BuiltinFn::Sin, false) => (global_sign, global_sign),
        (BuiltinFn::Sin, true) => (global_sign, -global_sign),
        (BuiltinFn::Cos, false) => (-global_sign, global_sign),
        (BuiltinFn::Cos, true) => (global_sign, global_sign),
        _ => return None,
    };

    Some((base_arg, coeff, kind, sin_sign, cos_sign))
}

fn extract_structural_unit_linear_trig_term_for_phase_shift_pair(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, i8)> {
    let (sign, positive_expr) = if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, expr)
    {
        (-1_i8, inner)
    } else {
        (1_i8, expr)
    };

    let (trig_fn, arg) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr)?;
    Some((trig_fn, arg, sign))
}

fn extract_structural_unit_exact_quarter_shifted_sine_term_for_phase_shift_pair(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, i8, i8)> {
    let (base_arg, coeff, sin_sign, cos_sign) =
        extract_structural_exact_quarter_phase_shift_term_data_for_cancellation(ctx, expr)?;
    let one = ctx.num(1);
    (compare_expr(ctx, coeff, one) == Ordering::Equal).then_some((base_arg, sin_sign, cos_sign))
}

pub(super) fn extract_structural_unit_linear_phase_shift_pair_side(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Vec<(cas_ast::ExprId, i8, i8)>> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 4 {
        return None;
    }

    let mut grouped_terms: Vec<(cas_ast::ExprId, Option<i8>, Option<i8>)> = Vec::with_capacity(2);
    for (term_expr, term_sign) in view.terms {
        let signed_expr = apply_sign_to_expr(ctx, sign_to_i64(term_sign), term_expr);
        let (trig_fn, raw_arg, sign) =
            extract_structural_unit_linear_trig_term_for_phase_shift_pair(ctx, signed_expr)?;

        let group_index = grouped_terms
            .iter()
            .position(|(arg, _, _)| compare_expr(ctx, *arg, raw_arg) == Ordering::Equal);

        let slot = if let Some(index) = group_index {
            &mut grouped_terms[index]
        } else {
            grouped_terms.push((raw_arg, None, None));
            grouped_terms.last_mut()?
        };

        match trig_fn {
            BuiltinFn::Sin => {
                if slot.1.is_some() {
                    return None;
                }
                slot.1 = Some(sign);
            }
            BuiltinFn::Cos => {
                if slot.2.is_some() {
                    return None;
                }
                slot.2 = Some(sign);
            }
            _ => return None,
        }
    }

    if grouped_terms.len() != 2 {
        return None;
    }

    grouped_terms
        .into_iter()
        .map(|(arg, sin_sign, cos_sign)| Some((arg, sin_sign?, cos_sign?)))
        .collect()
}

pub(super) fn extract_structural_unit_exact_quarter_shifted_phase_shift_pair_side(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Vec<(cas_ast::ExprId, i8, i8)>> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut shifted_terms = Vec::with_capacity(2);
    for (term_expr, term_sign) in view.terms {
        let signed_expr = apply_sign_to_expr(ctx, sign_to_i64(term_sign), term_expr);
        let shifted = extract_structural_unit_exact_quarter_shifted_sine_term_for_phase_shift_pair(
            ctx,
            signed_expr,
        )?;
        if shifted_terms
            .iter()
            .any(|(arg, _, _)| compare_expr(ctx, *arg, shifted.0) == Ordering::Equal)
        {
            return None;
        }
        shifted_terms.push(shifted);
    }

    (shifted_terms.len() == 2).then_some(shifted_terms)
}

pub(super) fn structural_unit_phase_shift_pair_groups_match(
    ctx: &mut cas_ast::Context,
    linear_groups: &[(cas_ast::ExprId, i8, i8)],
    shifted_groups: &[(cas_ast::ExprId, i8, i8)],
) -> bool {
    if linear_groups.len() != 2 || shifted_groups.len() != 2 {
        return false;
    }

    let mut used = vec![false; shifted_groups.len()];
    for &(linear_arg, linear_sin_sign, linear_cos_sign) in linear_groups {
        let mut matched_index = None;
        for (index, &(shifted_arg, shifted_sin_sign, shifted_cos_sign)) in
            shifted_groups.iter().enumerate()
        {
            if used[index] {
                continue;
            }
            if compare_expr(ctx, linear_arg, shifted_arg) == Ordering::Equal
                && linear_sin_sign == shifted_sin_sign
                && linear_cos_sign == shifted_cos_sign
            {
                matched_index = Some(index);
                break;
            }
        }

        let Some(index) = matched_index else {
            return false;
        };
        used[index] = true;
    }

    true
}

pub(super) fn extract_weighted_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId, i8, i8)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let sample = profiling.then(|| render_expr_for_orchestrator_profile(ctx, expr));
    let terms = if profiling {
        run_profiled_orchestrator_section(
            "rule.phase_shift.weighted_terms.add_terms_signed",
            sample.clone(),
            || cas_math::expr_nary::add_terms_signed(ctx, expr),
            |_| true,
        )
    } else {
        cas_math::expr_nary::add_terms_signed(ctx, expr)
    };

    if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.weighted_terms.from_terms",
            sample,
            || extract_weighted_phase_shift_linear_combination_terms_for_cancellation(ctx, &terms),
        )
    } else {
        extract_weighted_phase_shift_linear_combination_terms_for_cancellation(ctx, &terms)
    }
}

fn extract_weighted_phase_shift_linear_combination_terms_for_cancellation(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId, i8, i8)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let arity_two = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.weighted_terms.term_count_gate",
            None,
            || (terms.len() == 2).then_some(()),
        )
        .is_some()
    } else {
        terms.len() == 2
    };
    if !arity_two {
        return None;
    }

    let mut sin_term = None;
    let mut cos_term = None;

    for &(term, sign) in terms {
        let signed = match sign {
            Sign::Pos => 1,
            Sign::Neg => -1,
        };
        let term_sample = profiling.then(|| render_expr_for_orchestrator_profile(ctx, term));
        let (trig_fn, arg, coeff) = if profiling {
            run_profiled_orchestrator_option_section(
                "rule.phase_shift.weighted_terms.term_extract",
                term_sample,
                || extract_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, term),
            )?
        } else {
            extract_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, term)?
        };
        let (coeff, signed) =
            normalize_phase_shift_coefficient_sign_for_cancellation(ctx, coeff, signed);

        match trig_fn {
            BuiltinFn::Sin => {
                if sin_term.is_some() {
                    return None;
                }
                sin_term = Some((arg, coeff, signed));
            }
            BuiltinFn::Cos => {
                if cos_term.is_some() {
                    return None;
                }
                cos_term = Some((arg, coeff, signed));
            }
            _ => return None,
        }
    }

    let (sin_arg, sin_coeff, sin_sign) = sin_term?;
    let (cos_arg, cos_coeff, cos_sign) = cos_term?;
    let shared_arg = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.weighted_terms.shared_arg_compare",
            None,
            || (compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal).then_some(()),
        )
        .is_some()
    } else {
        compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal
    };
    if !shared_arg {
        return None;
    }

    Some((sin_arg, sin_coeff, cos_coeff, sin_sign, cos_sign))
}

pub(super) fn exact_phase_shift_linear_signature_for_cancellation(
    ctx: &mut cas_ast::Context,
    coeff: cas_ast::ExprId,
    kind: PhaseShiftKindForCancellation,
) -> (cas_ast::ExprId, cas_ast::ExprId) {
    let three = ctx.num(3);
    let sqrt_three = ctx.call_builtin(BuiltinFn::Sqrt, vec![three]);
    match kind {
        PhaseShiftKindForCancellation::Quarter => (coeff, coeff),
        PhaseShiftKindForCancellation::Third => (coeff, smart_mul(ctx, coeff, sqrt_three)),
        PhaseShiftKindForCancellation::Sixth => (smart_mul(ctx, coeff, sqrt_three), coeff),
    }
}

fn matches_exact_phase_shift_linear_combination_target_from_extracted(
    ctx: &mut cas_ast::Context,
    sin_term: (cas_ast::ExprId, cas_ast::ExprId, i8),
    cos_term: (cas_ast::ExprId, cas_ast::ExprId, i8),
    arg: cas_ast::ExprId,
    coeff: cas_ast::ExprId,
    kind: PhaseShiftKindForCancellation,
    signs: (i8, i8),
) -> bool {
    let (target_sin_arg, target_sin_coeff, target_sin_sign) = sin_term;
    let (target_cos_arg, target_cos_coeff, target_cos_sign) = cos_term;
    let (sin_sign, cos_sign) = signs;
    let (sin_coeff, cos_coeff) =
        exact_phase_shift_linear_signature_for_cancellation(ctx, coeff, kind);

    compare_expr(ctx, target_sin_arg, arg) == Ordering::Equal
        && compare_expr(ctx, target_cos_arg, arg) == Ordering::Equal
        && target_sin_sign == sin_sign
        && target_cos_sign == cos_sign
        && exprs_match_for_cancellation(ctx, target_sin_coeff, sin_coeff)
        && exprs_match_for_cancellation(ctx, target_cos_coeff, cos_coeff)
}

pub(super) fn matches_weighted_phase_shift_linear_combination_target(
    ctx: &mut cas_ast::Context,
    target_expr: cas_ast::ExprId,
    target_is_negated: bool,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    signs: (i8, i8),
) -> bool {
    let Some((target_arg, target_sin_coeff, target_cos_coeff, target_sin_sign, target_cos_sign)) =
        extract_weighted_phase_shift_linear_combination_for_cancellation(ctx, target_expr)
    else {
        return false;
    };
    let (sin_sign, cos_sign) = signs;

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

    compare_expr(ctx, target_arg, arg) == Ordering::Equal
        && target_sin_sign == expected_sin_sign
        && target_cos_sign == expected_cos_sign
        && exprs_match_for_cancellation(ctx, target_sin_coeff, sin_coeff)
        && exprs_match_for_cancellation(ctx, target_cos_coeff, cos_coeff)
}

pub(super) fn extract_general_phase_shift_term_data_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<GeneralPhaseShiftTermData> {
    let (global_sign, positive_expr) =
        if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, expr) {
            (-1_i8, inner)
        } else {
            (1_i8, expr)
        };

    if let Some((trig_fn, raw_arg)) =
        extract_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr)
    {
        let (base_arg, ratio, subtract_shift) =
            extract_general_phase_shift_argument_for_cancellation(ctx, raw_arg)?;
        return Some(GeneralPhaseShiftTermData {
            coeff: ctx.num(1),
            trig_fn,
            base_arg,
            ratio,
            subtract_shift,
            global_sign,
        });
    }

    let factors = flatten_mul_chain(ctx, positive_expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_term = None;
    let mut coeff_factors = Vec::new();
    for factor in factors {
        if trig_term.is_none() {
            if let Some((trig_fn, raw_arg)) =
                extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor)
            {
                trig_term = Some((trig_fn, raw_arg));
                continue;
            }
        }
        coeff_factors.push(factor);
    }

    let (trig_fn, raw_arg) = trig_term?;
    let (base_arg, ratio, subtract_shift) =
        extract_general_phase_shift_argument_for_cancellation(ctx, raw_arg)?;
    let coeff = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));
    let (coeff, global_sign) =
        normalize_phase_shift_coefficient_sign_for_cancellation(ctx, coeff, global_sign);

    Some(GeneralPhaseShiftTermData {
        coeff,
        trig_fn,
        base_arg,
        ratio,
        subtract_shift,
        global_sign,
    })
}

pub(super) fn build_weighted_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> cas_ast::ExprId {
    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_term = smart_mul(ctx, sin_coeff, sin_call);
    let cos_term = smart_mul(ctx, cos_coeff, cos_call);
    let signed_sin = if sin_sign < 0 {
        ctx.add(Expr::Neg(sin_term))
    } else {
        sin_term
    };
    let signed_cos = if cos_sign < 0 {
        ctx.add(Expr::Neg(cos_term))
    } else {
        cos_term
    };

    ctx.add(Expr::Add(signed_sin, signed_cos))
}

pub(super) fn extract_general_phase_shift_linear_signature_for_cancellation(
    ctx: &mut cas_ast::Context,
    data: GeneralPhaseShiftTermData,
) -> Option<GeneralPhaseShiftLinearSignature> {
    let ratio_arg = extract_atan_ratio_arg_for_phase_shift(ctx, data.ratio)?;
    let numeric_ratio = match ctx.get(ratio_arg) {
        Expr::Number(ratio) => Some(ratio.clone()),
        _ => None,
    };
    let numeric_coeff = match ctx.get(data.coeff) {
        Expr::Number(coeff) => Some(coeff.clone()),
        _ => None,
    };
    let (sin_sign, cos_sign) = general_phase_shift_linear_signs_for_cancellation(data)?;
    if let (Some(ratio), Some(coeff)) = (numeric_ratio, numeric_coeff) {
        let hyp_sq = BigRational::one() + ratio.clone() * ratio.clone();
        if let Some(hyp) = rational_sqrt(&hyp_sq) {
            let cos_shift = BigRational::one() / hyp.clone();
            let sin_shift = ratio.clone() / hyp;
            let sin_coeff = ctx.add(Expr::Number(coeff.clone() * cos_shift));
            let cos_coeff = ctx.add(Expr::Number(coeff * sin_shift));
            return Some((data.base_arg, sin_coeff, cos_coeff, sin_sign, cos_sign));
        }
    }

    let (sin_shift, _) = cas_math::trig_inverse_expansion_support::expand_trig_inverse_composition(
        ctx, "sin", "arctan", ratio_arg,
    )?;
    let (cos_shift, _) = cas_math::trig_inverse_expansion_support::expand_trig_inverse_composition(
        ctx, "cos", "arctan", ratio_arg,
    )?;

    let sin_coeff_raw = smart_mul(ctx, data.coeff, cos_shift);
    let cos_coeff_raw = smart_mul(ctx, data.coeff, sin_shift);
    let sin_coeff = run_default_simplify(ctx, sin_coeff_raw);
    let cos_coeff = run_default_simplify(ctx, cos_coeff_raw);

    Some((data.base_arg, sin_coeff, cos_coeff, sin_sign, cos_sign))
}

fn extract_atan_ratio_arg_for_phase_shift(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        ))
    .then_some(args[0])
}

fn general_phase_shift_linear_signs_for_cancellation(
    data: GeneralPhaseShiftTermData,
) -> Option<(i8, i8)> {
    match (data.trig_fn, data.subtract_shift) {
        (BuiltinFn::Sin, false) => Some((data.global_sign, data.global_sign)),
        (BuiltinFn::Sin, true) => Some((data.global_sign, -data.global_sign)),
        (BuiltinFn::Cos, false) => Some((-data.global_sign, data.global_sign)),
        (BuiltinFn::Cos, true) => Some((data.global_sign, data.global_sign)),
        _ => None,
    }
}

pub(super) fn build_general_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    data: GeneralPhaseShiftTermData,
) -> Option<cas_ast::ExprId> {
    let (arg, sin_coeff, cos_coeff, sin_sign, cos_sign) =
        extract_general_phase_shift_linear_signature_for_cancellation(ctx, data)?;
    let rewritten = build_weighted_phase_shift_linear_combination_for_cancellation(
        ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
    );

    Some(run_default_simplify(ctx, rewritten))
}

pub(super) fn build_general_phase_shift_sine_term_candidate_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> cas_ast::ExprId {
    let ratio_raw = ctx.add(Expr::Div(cos_coeff, sin_coeff));
    let ratio = run_default_simplify(ctx, ratio_raw);
    let shift = ctx.call_builtin(BuiltinFn::Arctan, vec![ratio]);
    let shifted_arg = if sin_sign == cos_sign {
        ctx.add(Expr::Add(arg, shift))
    } else {
        ctx.add(Expr::Sub(arg, shift))
    };

    let sin_sq = smart_mul(ctx, sin_coeff, sin_coeff);
    let cos_sq = smart_mul(ctx, cos_coeff, cos_coeff);
    let amplitude_sq = ctx.add(Expr::Add(sin_sq, cos_sq));
    let amplitude_raw = ctx.call_builtin(BuiltinFn::Sqrt, vec![amplitude_sq]);
    let amplitude = run_default_simplify(ctx, amplitude_raw);
    let shifted_sine = ctx.call_builtin(BuiltinFn::Sin, vec![shifted_arg]);
    let rewritten = smart_mul(ctx, amplitude, shifted_sine);
    let rewritten = if sin_sign < 0 {
        ctx.add(Expr::Neg(rewritten))
    } else {
        rewritten
    };

    run_default_simplify(ctx, rewritten)
}

pub(super) fn build_general_phase_shift_cosine_term_candidate_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> cas_ast::ExprId {
    let ratio_raw = ctx.add(Expr::Div(sin_coeff, cos_coeff));
    let ratio = run_default_simplify(ctx, ratio_raw);
    let shift = ctx.call_builtin(BuiltinFn::Arctan, vec![ratio]);
    let shifted_arg = if sin_sign == cos_sign {
        ctx.add(Expr::Sub(arg, shift))
    } else {
        ctx.add(Expr::Add(arg, shift))
    };

    let sin_sq = smart_mul(ctx, sin_coeff, sin_coeff);
    let cos_sq = smart_mul(ctx, cos_coeff, cos_coeff);
    let amplitude_sq = ctx.add(Expr::Add(sin_sq, cos_sq));
    let amplitude_raw = ctx.call_builtin(BuiltinFn::Sqrt, vec![amplitude_sq]);
    let amplitude = run_default_simplify(ctx, amplitude_raw);
    let shifted_cosine = ctx.call_builtin(BuiltinFn::Cos, vec![shifted_arg]);
    let rewritten = smart_mul(ctx, amplitude, shifted_cosine);
    let rewritten = if cos_sign < 0 {
        ctx.add(Expr::Neg(rewritten))
    } else {
        rewritten
    };

    run_default_simplify(ctx, rewritten)
}

pub(super) fn generate_general_phase_shift_term_candidates_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> Vec<cas_ast::ExprId> {
    let mut candidates = Vec::new();

    let sine_candidate = build_general_phase_shift_sine_term_candidate_for_cancellation(
        ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
    );
    candidates.push(sine_candidate);

    let cosine_candidate = build_general_phase_shift_cosine_term_candidate_for_cancellation(
        ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
    );
    if candidates
        .iter()
        .all(|existing| compare_expr(ctx, *existing, cosine_candidate) != Ordering::Equal)
    {
        candidates.push(cosine_candidate);
    }

    candidates
}

pub(super) fn matches_general_phase_shift_shifted_term_candidate_for_cancellation(
    ctx: &mut cas_ast::Context,
    target_data: GeneralPhaseShiftTermData,
    linear_signature: GeneralPhaseShiftLinearSignature,
    target_is_negated: bool,
) -> bool {
    let (arg, sin_coeff, cos_coeff, sin_sign, cos_sign) = linear_signature;
    if compare_expr(ctx, target_data.base_arg, arg) != Ordering::Equal {
        return false;
    }

    let (expected_ratio_raw, expected_subtract_shift, expected_global_sign) =
        match target_data.trig_fn {
            BuiltinFn::Sin => (
                ctx.add(Expr::Div(cos_coeff, sin_coeff)),
                sin_sign != cos_sign,
                sin_sign,
            ),
            BuiltinFn::Cos => (
                ctx.add(Expr::Div(sin_coeff, cos_coeff)),
                sin_sign == cos_sign,
                cos_sign,
            ),
            _ => return false,
        };
    let expected_global_sign = if target_is_negated {
        -expected_global_sign
    } else {
        expected_global_sign
    };
    if target_data.subtract_shift != expected_subtract_shift
        || target_data.global_sign != expected_global_sign
    {
        return false;
    }

    let expected_ratio = run_default_simplify(ctx, expected_ratio_raw);
    let Some(target_ratio_arg) = extract_atan_ratio_arg_for_phase_shift(ctx, target_data.ratio)
    else {
        return false;
    };
    let numeric_fast_match = extract_literal_rational_for_cancellation(ctx, target_ratio_arg)
        .zip(extract_literal_rational_for_cancellation(ctx, sin_coeff))
        .zip(extract_literal_rational_for_cancellation(ctx, cos_coeff))
        .zip(extract_literal_rational_for_cancellation(
            ctx,
            target_data.coeff,
        ))
        .and_then(
            |(((target_ratio, sin_numeric), cos_numeric), target_coeff_numeric)| {
                let expected_ratio_numeric = match target_data.trig_fn {
                    BuiltinFn::Sin if !sin_numeric.is_zero() => {
                        Some(cos_numeric.clone() / sin_numeric.clone())
                    }
                    BuiltinFn::Cos if !cos_numeric.is_zero() => {
                        Some(sin_numeric.clone() / cos_numeric.clone())
                    }
                    _ => None,
                }?;
                let amplitude_sq =
                    sin_numeric.clone() * sin_numeric + cos_numeric.clone() * cos_numeric;
                let amplitude_numeric = rational_sqrt(&amplitude_sq)?;
                Some(
                    target_ratio == expected_ratio_numeric
                        && target_coeff_numeric == amplitude_numeric,
                )
            },
        );
    if let Some(matched) = numeric_fast_match {
        return matched;
    }
    if !exprs_match_for_cancellation(ctx, target_ratio_arg, expected_ratio) {
        return false;
    }

    let sin_sq = smart_mul(ctx, sin_coeff, sin_coeff);
    let cos_sq = smart_mul(ctx, cos_coeff, cos_coeff);
    let amplitude_sq = ctx.add(Expr::Add(sin_sq, cos_sq));
    let amplitude_raw = ctx.call_builtin(BuiltinFn::Sqrt, vec![amplitude_sq]);
    let amplitude = run_default_simplify(ctx, amplitude_raw);

    exprs_match_for_cancellation(ctx, target_data.coeff, amplitude)
}

fn resolve_surface_scaled_trig_candidate_vs_negated_target_for_phase_shift(
    ctx: &mut cas_ast::Context,
    candidate: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<bool> {
    let inner_target = strip_unit_negation_for_phase_shift(ctx, target_expr)?;
    let (candidate_fn, candidate_arg, candidate_coeff, candidate_sign) =
        extract_surface_scaled_trig_term_for_phase_shift(ctx, candidate)?;
    let (target_fn, target_arg, target_coeff, target_sign) =
        extract_surface_scaled_trig_term_for_phase_shift(ctx, inner_target)?;

    if candidate_fn == target_fn
        && candidate_sign == target_sign
        && exprs_match_for_cancellation(ctx, candidate_arg, target_arg)
        && exprs_match_for_cancellation(ctx, candidate_coeff, target_coeff)
    {
        return Some(true);
    }

    if candidate_fn != target_fn
        || candidate_sign != target_sign
        || !exprs_match_for_cancellation(ctx, candidate_coeff, target_coeff)
    {
        return None;
    }

    let (candidate_base, candidate_kind, candidate_subtract_shift) =
        extract_supported_phase_shift_argument_for_cancellation(ctx, candidate_fn, candidate_arg)?;
    let (target_base, target_kind, target_subtract_shift) =
        extract_supported_phase_shift_argument_for_cancellation(ctx, target_fn, target_arg)?;

    (candidate_kind == target_kind
        && candidate_subtract_shift != target_subtract_shift
        && exprs_match_for_cancellation(ctx, candidate_base, target_base))
    .then_some(false)
}

pub(super) fn resolve_surface_shifted_candidate_vs_plain_trig_target_for_phase_shift(
    ctx: &mut cas_ast::Context,
    candidate: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<bool> {
    let (candidate_fn, candidate_arg, _, _) =
        extract_surface_scaled_trig_term_for_phase_shift(ctx, candidate)?;
    let (target_fn, target_arg, _, _) =
        extract_surface_scaled_trig_term_for_phase_shift(ctx, target_expr)?;
    let (candidate_base, candidate_kind, _) =
        extract_supported_phase_shift_argument_for_cancellation(ctx, candidate_fn, candidate_arg)?;
    if extract_supported_phase_shift_argument_for_cancellation(ctx, target_fn, target_arg).is_some()
    {
        return None;
    }

    if candidate_fn != target_fn {
        if candidate_kind == PhaseShiftKindForCancellation::Quarter
            && !expr_contains_pi_constant(ctx, target_arg)
            && matches!(
                (ctx.get(candidate_base), ctx.get(target_arg)),
                (Expr::Variable(_), Expr::Variable(_))
                    | (Expr::Variable(_), Expr::SessionRef(_))
                    | (Expr::SessionRef(_), Expr::Variable(_))
                    | (Expr::SessionRef(_), Expr::SessionRef(_))
            )
            && !exact_phase_shift_args_match_for_cancellation(ctx, candidate_base, target_arg)
        {
            return Some(false);
        }
        return None;
    }

    if expr_contains_pi_constant(ctx, target_arg)
        || !expr_contains_symbolic_atom_for_cancellation(ctx, candidate_base)
        || !expr_contains_symbolic_atom_for_cancellation(ctx, target_arg)
    {
        return None;
    }

    Some(false)
}

fn linear_focus_phase_shift_candidate_matches_target(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    candidate: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    target_is_negated: bool,
    cached_target_simplified: &mut Option<cas_ast::ExprId>,
) -> bool {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    if !profiling {
        if target_is_negated {
            if expr_matches_negation_for_cancellation(ctx, candidate, target_expr) {
                return true;
            }
            if let Some(false) =
                resolve_surface_shifted_candidate_vs_plain_trig_target_for_phase_shift(
                    ctx,
                    candidate,
                    target_expr,
                )
            {
                return false;
            }
            if let Some(resolution) =
                resolve_surface_scaled_trig_candidate_vs_negated_target_for_phase_shift(
                    ctx,
                    candidate,
                    target_expr,
                )
            {
                return resolution;
            }
        } else if exprs_match_for_cancellation(ctx, candidate, target_expr) {
            return true;
        }

        let candidate_simplified = run_default_simplify(ctx, candidate);
        let target_simplified = if let Some(existing) = *cached_target_simplified {
            existing
        } else {
            let simplified = if target_is_negated {
                let neg_target = ctx.add(Expr::Neg(target_expr));
                run_default_simplify(ctx, neg_target)
            } else {
                run_default_simplify(ctx, target_expr)
            };
            *cached_target_simplified = Some(simplified);
            simplified
        };
        return exprs_match_for_cancellation(ctx, candidate_simplified, target_simplified);
    }

    let compare_sample = Some(format!(
        "{}  =>  {}  ||  {}",
        render_expr_for_orchestrator_profile(ctx, focus_expr),
        render_expr_for_orchestrator_profile(ctx, candidate),
        render_expr_for_orchestrator_profile(ctx, target_expr)
    ));

    if target_is_negated {
        if run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_focus.compare_candidate.negated.fast_negation_check",
            compare_sample.clone(),
            || expr_matches_negation_for_cancellation(ctx, candidate, target_expr).then_some(()),
        )
        .is_some()
        {
            return true;
        }

        if let Some(resolution) = run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_focus.compare_candidate.negated.surface_plain_trig_reject",
            compare_sample.clone(),
            || {
                resolve_surface_shifted_candidate_vs_plain_trig_target_for_phase_shift(
                    ctx,
                    candidate,
                    target_expr,
                )
            },
        ) {
            return resolution;
        }

        if let Some(resolution) = run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_focus.compare_candidate.negated.surface_scaled_trig_resolution",
            compare_sample.clone(),
            || {
                resolve_surface_scaled_trig_candidate_vs_negated_target_for_phase_shift(
                    ctx,
                    candidate,
                    target_expr,
                )
            },
        ) {
            return resolution;
        }

        profile_linear_focus_negated_fallback_target_relation_for_phase_shift(
            ctx,
            candidate,
            target_expr,
            compare_sample.clone(),
        );

        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_focus.compare_candidate.negated.default_simplify_fallback",
            compare_sample,
            || {
                profiled_negated_default_simplify_fallback_for_phase_shift_compare(
                    ctx,
                    focus_expr,
                    candidate,
                    target_expr,
                    cached_target_simplified,
                )
                .then_some(())
            },
        )
        .is_some()
    } else {
        if run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_focus.compare_candidate.direct.fast_match",
            compare_sample.clone(),
            || exprs_match_for_cancellation(ctx, candidate, target_expr).then_some(()),
        )
        .is_some()
        {
            return true;
        }

        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_focus.compare_candidate.direct.default_simplify_fallback",
            compare_sample,
            || {
                profiled_direct_default_simplify_fallback_for_phase_shift_compare(
                    ctx,
                    focus_expr,
                    candidate,
                    target_expr,
                    cached_target_simplified,
                )
                .then_some(())
            },
        )
        .is_some()
    }
}

pub(super) fn find_linear_focus_phase_shift_cancellation_match(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    target_is_negated: bool,
) -> LinearFocusPhaseShiftMatchOutcome {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, focus_expr),
            render_expr_for_orchestrator_profile(ctx, target_expr)
        )
    });

    let extracted = if profiling {
        run_profiled_orchestrator_section(
            "rule.phase_shift.linear_focus.extract_linear_combination",
            pair_sample.clone(),
            || extract_exact_phase_shift_linear_combination_for_cancellation(ctx, focus_expr),
            |_| true,
        )
    } else {
        extract_exact_phase_shift_linear_combination_for_cancellation(ctx, focus_expr)
    };
    let exact_data = match extracted {
        ExactPhaseShiftLinearCombinationExtraction::NotLinear => {
            return LinearFocusPhaseShiftMatchOutcome::NotLinear;
        }
        ExactPhaseShiftLinearCombinationExtraction::LinearButNotExact => {
            return LinearFocusPhaseShiftMatchOutcome::NeedsGeneralRoute;
        }
        ExactPhaseShiftLinearCombinationExtraction::Exact(data) => data,
    };
    let (arg, coeff, kind) = (exact_data.arg, exact_data.coeff, exact_data.kind);

    let exact_target_has_pi = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.phase_shift.linear_focus.exact_target_pi_gate",
            pair_sample.clone(),
            || expr_contains_pi_constant(ctx, target_expr).then_some(()),
        )
        .is_some()
    } else {
        expr_contains_pi_constant(ctx, target_expr)
    };
    let exact_target_match = if exact_target_has_pi {
        if profiling {
            let _ = run_profiled_orchestrator_option_section(
                "rule.phase_shift.linear_focus.exact_target_structural_match",
                pair_sample.clone(),
                || {
                    let (target_arg, target_coeff, target_kind, target_sin_sign, target_cos_sign) =
                        extract_structural_exact_phase_shift_term_data_for_cancellation(
                            ctx,
                            target_expr,
                        )?;
                    let expected_target_sin_sign = if target_is_negated {
                        -exact_data.sin_sign
                    } else {
                        exact_data.sin_sign
                    };
                    let expected_target_cos_sign = if target_is_negated {
                        -exact_data.cos_sign
                    } else {
                        exact_data.cos_sign
                    };

                    (compare_expr(ctx, target_arg, exact_data.arg) == Ordering::Equal
                        && target_kind == exact_data.kind
                        && target_sin_sign == expected_target_sin_sign
                        && target_cos_sign == expected_target_cos_sign
                        && exprs_match_for_cancellation(ctx, target_coeff, exact_data.coeff))
                    .then_some(TrigPhaseShiftCancellationMatch {
                        local_before: focus_expr,
                        local_after: target_expr,
                        mode: TrigPhaseShiftCancellationMode::LinearToShifted,
                    })
                },
            );
        }
        if profiling {
            run_profiled_orchestrator_option_section(
                "rule.phase_shift.linear_focus.exact_target_match",
                pair_sample.clone(),
                || {
                    let (target_arg, target_coeff, target_kind, target_sin_sign, target_cos_sign) =
                        extract_exact_phase_shift_term_data_for_cancellation(ctx, target_expr)?;
                    let expected_target_sin_sign = if target_is_negated {
                        -exact_data.sin_sign
                    } else {
                        exact_data.sin_sign
                    };
                    let expected_target_cos_sign = if target_is_negated {
                        -exact_data.cos_sign
                    } else {
                        exact_data.cos_sign
                    };

                    (compare_expr(ctx, target_arg, exact_data.arg) == Ordering::Equal
                        && target_kind == exact_data.kind
                        && target_sin_sign == expected_target_sin_sign
                        && target_cos_sign == expected_target_cos_sign
                        && exprs_match_for_cancellation(ctx, target_coeff, exact_data.coeff))
                    .then_some(TrigPhaseShiftCancellationMatch {
                        local_before: focus_expr,
                        local_after: target_expr,
                        mode: TrigPhaseShiftCancellationMode::LinearToShifted,
                    })
                },
            )
        } else if let Some((
            target_arg,
            target_coeff,
            target_kind,
            target_sin_sign,
            target_cos_sign,
        )) = extract_exact_phase_shift_term_data_for_cancellation(ctx, target_expr)
        {
            let expected_target_sin_sign = if target_is_negated {
                -exact_data.sin_sign
            } else {
                exact_data.sin_sign
            };
            let expected_target_cos_sign = if target_is_negated {
                -exact_data.cos_sign
            } else {
                exact_data.cos_sign
            };

            (compare_expr(ctx, target_arg, exact_data.arg) == Ordering::Equal
                && target_kind == exact_data.kind
                && target_sin_sign == expected_target_sin_sign
                && target_cos_sign == expected_target_cos_sign
                && exprs_match_for_cancellation(ctx, target_coeff, exact_data.coeff))
            .then_some(TrigPhaseShiftCancellationMatch {
                local_before: focus_expr,
                local_after: target_expr,
                mode: TrigPhaseShiftCancellationMode::LinearToShifted,
            })
        } else {
            None
        }
    } else {
        None
    };
    if let Some(rewrite_match) = exact_target_match {
        return LinearFocusPhaseShiftMatchOutcome::Matched(rewrite_match);
    }

    let candidates = if profiling {
        run_profiled_orchestrator_section(
            "rule.phase_shift.linear_focus.generate_candidates",
            pair_sample,
            || generate_phase_shift_term_candidates_for_cancellation(ctx, coeff, arg, kind),
            |_| true,
        )
    } else {
        generate_phase_shift_term_candidates_for_cancellation(ctx, coeff, arg, kind)
    };
    let mut cached_target_simplified = None;

    for candidate in candidates {
        let matches = if profiling {
            let compare_label =
                linear_focus_phase_shift_compare_profile_label(ctx, candidate, target_is_negated);
            run_profiled_orchestrator_option_section(
                compare_label,
                Some(format!(
                    "{}  =>  {}  ||  {}",
                    render_expr_for_orchestrator_profile(ctx, focus_expr),
                    render_expr_for_orchestrator_profile(ctx, candidate),
                    render_expr_for_orchestrator_profile(ctx, target_expr)
                )),
                || {
                    linear_focus_phase_shift_candidate_matches_target(
                        ctx,
                        focus_expr,
                        candidate,
                        target_expr,
                        target_is_negated,
                        &mut cached_target_simplified,
                    )
                    .then_some(())
                },
            )
            .is_some()
        } else {
            linear_focus_phase_shift_candidate_matches_target(
                ctx,
                focus_expr,
                candidate,
                target_expr,
                target_is_negated,
                &mut cached_target_simplified,
            )
        };

        if matches {
            return LinearFocusPhaseShiftMatchOutcome::Matched(TrigPhaseShiftCancellationMatch {
                local_before: focus_expr,
                local_after: candidate,
                mode: TrigPhaseShiftCancellationMode::LinearToShifted,
            });
        }
    }

    LinearFocusPhaseShiftMatchOutcome::LinearNoMatch
}

pub(crate) fn matches_trig_phase_shift_cancellation_pair(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    target_is_negated: bool,
) -> bool {
    try_find_trig_phase_shift_cancellation_match(ctx, focus_expr, target_expr, target_is_negated)
        .is_some()
}

pub(super) fn build_trig_phase_shift_zero_rewrite(
    ctx: &mut cas_ast::Context,
    rewrite_match: TrigPhaseShiftCancellationMatch,
) -> Rewrite {
    let focus_after_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: rewrite_match.local_after
        }
    );

    let rewrite = Rewrite::with_local(
        ctx.num(0),
        "Phase Shift Identity",
        rewrite_match.local_before,
        rewrite_match.local_after,
    );

    match rewrite_match.mode {
        TrigPhaseShiftCancellationMode::LinearToShifted => rewrite
            .substep(
                "Reescribir la combinación lineal",
                vec![format!("Se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras aplicar la identidad de desfase, el término restante es exactamente el opuesto y toda la expresión se anula."
                        .to_string(),
                ],
            ),
        TrigPhaseShiftCancellationMode::ShiftedToLinear => rewrite
            .substep(
                "Expandir el término desplazado",
                vec![format!("Se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras expandir el término desplazado, el resto de la expresión es exactamente el opuesto y el resultado es 0."
                        .to_string(),
                ],
            ),
        TrigPhaseShiftCancellationMode::ShiftedToShifted => rewrite
            .substep(
                "Reescribir el término desplazado",
                vec![format!("Se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras usar la identidad de desfase equivalente, el resto de la expresión es exactamente el opuesto y el resultado es 0."
                        .to_string(),
                ],
            ),
    }
}

fn try_build_direct_half_angle_binomial_square_shifted_quotient_rewrite(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    denominator: cas_ast::ExprId,
    numerator_core: cas_ast::ExprId,
    denominator_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let matches_pair = (matches_direct_half_angle_square_zero_identity(ctx, numerator_core)
        && matches_direct_trig_binomial_square_zero_identity(ctx, denominator_core))
        || (matches_direct_half_angle_square_zero_identity(ctx, denominator_core)
            && matches_direct_trig_binomial_square_zero_identity(ctx, numerator_core));

    if !matches_pair {
        return None;
    }

    let child_rewrite = Rewrite::with_local(
        ctx.num(0),
        "Trig Square Identity",
        numerator_core,
        denominator_core,
    )
    .substep(
        "Anular el residuo del numerador",
        vec![
            "El término adicional del numerador es una identidad trigonométrica exacta, así que vale 0."
                .to_string(),
        ],
    )
    .substep(
        "Anular el residuo del denominador",
        vec![
            "El término adicional del denominador también es una identidad trigonométrica exacta, así que vale 0."
                .to_string(),
        ],
    )
    .substep(
        "Evaluar 1/1",
        vec!["Tras sumar 1 a ambos lados, la fracción queda 1/1 y por tanto vale 1.".to_string()],
    );

    Some(build_shifted_quotient_exact_one_rewrite(
        ctx,
        numerator,
        denominator,
        child_rewrite,
    ))
}

pub(super) fn try_build_direct_small_zero_core_shifted_quotient_rewrite(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    denominator: cas_ast::ExprId,
    numerator_core: cas_ast::ExprId,
    denominator_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let numerator_zero_rewrite = try_build_small_direct_zero_core_rewrite(ctx, numerator_core)
        .or_else(|| try_build_exact_zero_identity_rewrite_direct(ctx, numerator_core))?;
    let denominator_zero_rewrite = try_build_small_direct_zero_core_rewrite(ctx, denominator_core)
        .or_else(|| try_build_exact_zero_identity_rewrite_direct(ctx, denominator_core))?;

    let description = if numerator_zero_rewrite.description == denominator_zero_rewrite.description
    {
        numerator_zero_rewrite.description.clone()
    } else {
        "Exact Zero Core Quotient Identity".into()
    };

    let mut rewrite = Rewrite::with_local(ctx.num(1), description, numerator, denominator)
        .requires(crate::ImplicitCondition::NonZero(denominator))
        .requires_all(numerator_zero_rewrite.required_conditions.clone())
        .requires_all(denominator_zero_rewrite.required_conditions.clone())
        .assume_all(numerator_zero_rewrite.assumption_events.clone())
        .assume_all(denominator_zero_rewrite.assumption_events.clone())
        .substep(
            "Anular el core del numerador",
            vec![
                "El término adicional del numerador es una identidad exacta pequeña, así que vale 0."
                    .to_string(),
            ],
        )
        .substep(
            "Anular el core del denominador",
            vec![
                "El término adicional del denominador también es una identidad exacta pequeña, así que vale 0."
                    .to_string(),
            ],
        )
        .substep(
            "Evaluar 1/1",
            vec!["Tras sumar 1 a ambos lados, la fracción queda 1/1 y por tanto vale 1.".to_string()],
        );

    if let Some(poly_proof) = numerator_zero_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    } else if let Some(poly_proof) = denominator_zero_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    Some(rewrite)
}

pub(crate) fn try_build_exact_trig_phase_shift_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }
    if let Some(rewrite) =
        try_build_fast_structural_exact_phase_shift_triple_zero_rewrite(ctx, &view.terms)
    {
        return Some(rewrite);
    }
    if let Some(rewrite) = try_build_fast_general_phase_shift_triple_zero_rewrite(ctx, &view.terms)
    {
        return Some(rewrite);
    }
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();

    for first_index in 0..view.terms.len() {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms = [view.terms[first_index], view.terms[second_index]];
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);
            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if remaining_terms.is_empty() {
                continue;
            }
            let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
            let focus_has_plain_trig =
                expr_contains_any_builtin(ctx, focus_expr, &[BuiltinFn::Sin, BuiltinFn::Cos]);
            let remaining_has_plain_trig =
                expr_contains_any_builtin(ctx, remaining_expr, &[BuiltinFn::Sin, BuiltinFn::Cos]);
            let rewrite_match = if profiling {
                let pair_sample = Some(format!(
                    "{}  ||  {}",
                    render_expr_for_orchestrator_profile(ctx, focus_expr),
                    render_expr_for_orchestrator_profile(ctx, remaining_expr)
                ));
                let pair_shape_label = match (focus_has_plain_trig, remaining_has_plain_trig) {
                    (true, true) => "rule.phase_shift.exact_scope_pair_shape.both_trig",
                    (true, false) => "rule.phase_shift.exact_scope_pair_shape.remaining_non_trig",
                    (false, true) => "rule.phase_shift.exact_scope_pair_shape.focus_non_trig",
                    (false, false) => "rule.phase_shift.exact_scope_pair_shape.both_non_trig",
                };
                let _ = run_profiled_orchestrator_option_section(
                    pair_shape_label,
                    pair_sample.clone(),
                    || Some(()),
                );
                if !focus_has_plain_trig && !remaining_has_plain_trig {
                    let detail_label = classify_phase_shift_exact_scope_nontrig_detail_for_profile(
                        ctx,
                        focus_expr,
                        remaining_expr,
                    );
                    let _ = run_profiled_orchestrator_option_section(
                        detail_label,
                        pair_sample.clone(),
                        || Some(()),
                    );
                }
                if !focus_has_plain_trig || !remaining_has_plain_trig {
                    None
                } else {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.exact_scope_pair_match",
                        pair_sample,
                        || {
                            try_find_trig_phase_shift_cancellation_match(
                                ctx,
                                focus_expr,
                                remaining_expr,
                                true,
                            )
                        },
                    )
                }
            } else if !focus_has_plain_trig || !remaining_has_plain_trig {
                None
            } else {
                try_find_trig_phase_shift_cancellation_match(ctx, focus_expr, remaining_expr, true)
            };

            if let Some(rewrite_match) = rewrite_match {
                let mode_label = match rewrite_match.mode {
                    TrigPhaseShiftCancellationMode::LinearToShifted => {
                        "rule.phase_shift.exact_scope.linear_to_shifted"
                    }
                    TrigPhaseShiftCancellationMode::ShiftedToLinear => {
                        "rule.phase_shift.exact_scope.shifted_to_linear"
                    }
                    TrigPhaseShiftCancellationMode::ShiftedToShifted => {
                        "rule.phase_shift.exact_scope.shifted_to_shifted"
                    }
                };
                let rewrite = if profiling {
                    run_profiled_orchestrator_section(
                        mode_label,
                        Some(render_expr_for_orchestrator_profile(ctx, expr)),
                        || build_trig_phase_shift_zero_rewrite(ctx, rewrite_match),
                        |_| true,
                    )
                } else {
                    build_trig_phase_shift_zero_rewrite(ctx, rewrite_match)
                };
                return Some(rewrite);
            }
        }
    }

    None
}

fn build_shifted_quotient_exact_one_rewrite(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    denominator: cas_ast::ExprId,
    child_rewrite: Rewrite,
) -> Rewrite {
    let mut rewrite = Rewrite::with_local(
        ctx.num(1),
        child_rewrite.description.clone(),
        numerator,
        denominator,
    )
    .requires(crate::ImplicitCondition::NonZero(denominator))
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone());

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    rewrite.substeps = child_rewrite.substeps.clone();
    rewrite
}

pub(super) fn maybe_shifted_quotient_exact_zero_direct_residual_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let term_count = AddView::from_expr(ctx, expr).terms.len();
    if !(2..=4).contains(&term_count) {
        return false;
    }

    expr_contains_division_node(ctx, expr)
        || expr_contains_sqrt_or_half_power(ctx, expr)
        || expr_contains_factorial_call(ctx, expr)
        || expr_contains_any_builtin(
            ctx,
            expr,
            &[
                BuiltinFn::Sin,
                BuiltinFn::Cos,
                BuiltinFn::Tan,
                BuiltinFn::Sec,
                BuiltinFn::Csc,
                BuiltinFn::Cot,
                BuiltinFn::Asin,
                BuiltinFn::Acos,
                BuiltinFn::Atan,
                BuiltinFn::Asec,
                BuiltinFn::Acsc,
                BuiltinFn::Acot,
                BuiltinFn::Arcsin,
                BuiltinFn::Arccos,
                BuiltinFn::Arctan,
                BuiltinFn::Arcsec,
                BuiltinFn::Arccsc,
                BuiltinFn::Arccot,
                BuiltinFn::Sinh,
                BuiltinFn::Cosh,
                BuiltinFn::Tanh,
                BuiltinFn::Asinh,
                BuiltinFn::Acosh,
                BuiltinFn::Atanh,
                BuiltinFn::Ln,
                BuiltinFn::Log,
                BuiltinFn::Log2,
                BuiltinFn::Log10,
                BuiltinFn::Exp,
                BuiltinFn::Sqrt,
                BuiltinFn::Cbrt,
                BuiltinFn::Root,
                BuiltinFn::Abs,
            ],
        )
}

fn matches_shifted_quotient_exact_zero_direct_residual_trig_power_reduction_target(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return false;
    };

    matches!(ctx.get(*numerator), Expr::Add(_, _) | Expr::Sub(_, _))
        && matches!(ctx.get(*denominator), Expr::Number(_))
}

fn matches_shifted_quotient_exact_zero_direct_residual_trig_power_reduction_pair(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    [(lhs_core, rhs_core), (rhs_core, lhs_core)]
        .into_iter()
        .any(|(power_side, target_side)| {
            (extract_scaled_trig_even_power_for_cancellation(ctx, power_side).is_some()
                || extract_trig_square_product_same_arg_for_cancellation(ctx, power_side).is_some())
                && matches_shifted_quotient_exact_zero_direct_residual_trig_power_reduction_target(
                    ctx,
                    target_side,
                )
        })
}

pub(super) fn maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    residual_expr: cas_ast::ExprId,
) -> bool {
    maybe_shifted_quotient_exact_zero_direct_residual_candidate(ctx, residual_expr)
        && (expr_contains_hyperbolic_builtin(ctx, residual_expr)
            || matches_shifted_quotient_exact_zero_direct_residual_trig_power_reduction_pair(
                ctx, lhs_core, rhs_core,
            ))
}

pub(super) fn numerator_matches_two_times_shift(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    shift: cas_ast::ExprId,
) -> bool {
    if extract_i64_integer(ctx, numerator) == Some(2) && extract_i64_integer(ctx, shift) == Some(1)
    {
        return true;
    }

    extract_two_times_factor_arg(ctx, numerator)
        .map(|factor| exprs_match_for_cancellation(ctx, factor, shift))
        .unwrap_or(false)
}

pub(super) fn shifted_unit_fraction_quotient_matches(
    ctx: &mut cas_ast::Context,
    quotient: cas_ast::ExprId,
    numerator: cas_ast::ExprId,
    denominator: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, quotient);
    if view.terms.len() != 2 {
        return false;
    }

    let mut has_unit = false;
    let mut shift = None;
    for (term, sign) in view.terms {
        if cas_ast::views::as_rational_const(ctx, term, 8).is_some_and(|value| value.is_one()) {
            if sign != Sign::Pos || has_unit {
                return false;
            }
            has_unit = true;
            continue;
        }

        let Some((term_numerator, term_denominator)) = as_div(ctx, term) else {
            return false;
        };
        if !reciprocal_half_power_base_matches(ctx, term_denominator, denominator) {
            return false;
        }
        let Some(mut value) = cas_ast::views::as_rational_const(ctx, term_numerator, 8) else {
            return false;
        };
        if sign == Sign::Neg {
            value = -value;
        }
        if shift.replace(value).is_some() {
            return false;
        }
    }

    let Some(shift) = shift else {
        return false;
    };
    if !has_unit {
        return false;
    }

    let shift_expr = ctx.add(Expr::Number(shift));
    let candidate_numerator = ctx.add(Expr::Add(denominator, shift_expr));
    reciprocal_half_power_base_matches(ctx, candidate_numerator, numerator)
}

pub(super) fn reject_shifted_surface_trig_symbolic_base_mismatch_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    let (lhs_fn, lhs_base_arg, lhs_coeff) =
        extract_shifted_surface_trig_base_arg_for_profile(ctx, lhs_core)?;
    let (rhs_fn, rhs_base_arg, rhs_coeff) =
        extract_shifted_surface_trig_base_arg_for_profile(ctx, rhs_core)?;

    if lhs_fn != rhs_fn || compare_expr(ctx, lhs_coeff, rhs_coeff) != Ordering::Equal {
        return None;
    }

    if !expr_is_symbolic_leaf_for_hyperbolic_reject(ctx, lhs_base_arg)
        || !expr_is_symbolic_leaf_for_hyperbolic_reject(ctx, rhs_base_arg)
        || compare_expr(ctx, lhs_base_arg, rhs_base_arg) == Ordering::Equal
    {
        return None;
    }

    Some(false)
}

fn try_build_shifted_quotient_sinh_cosh_exp_definition_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (direct_side, exp_definition_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !expr_contains_any_builtin(ctx, direct_side, &[BuiltinFn::Sinh, BuiltinFn::Cosh])
            || !contains_division_like_term(ctx, exp_definition_side)
        {
            continue;
        }

        let Some(rewritten) =
            try_rewrite_sinh_cosh_exp_definition_for_cancellation(ctx, exp_definition_side)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, direct_side, rewritten)
            || exprs_match_after_default_simplify(ctx, direct_side, rewritten)
        {
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

pub(super) fn try_build_shifted_quotient_exact_one_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let gate_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, numerator),
            render_expr_for_orchestrator_profile(ctx, denominator)
        )
    });
    if profiling {
        let numerator_kind = classify_positive_one_passthrough_profile_kind(ctx, numerator);
        profile_shifted_quotient_exact_one_gate_side(
            "numerator",
            numerator_kind,
            gate_sample.clone(),
        );
        if matches!(
            numerator_kind,
            PositiveOnePassthroughProfileKind::AddNoPositiveOne
        ) {
            profile_shifted_quotient_exact_one_gate_add_no_positive_one_detail(
                ctx,
                "numerator",
                numerator,
                gate_sample.clone(),
            );
        }
    }
    let numerator_core = strip_positive_one_passthrough(ctx, numerator)?;

    if profiling {
        let denominator_kind = classify_positive_one_passthrough_profile_kind(ctx, denominator);
        profile_shifted_quotient_exact_one_gate_side(
            "denominator",
            denominator_kind,
            gate_sample.clone(),
        );
        if matches!(
            denominator_kind,
            PositiveOnePassthroughProfileKind::AddNoPositiveOne
        ) {
            profile_shifted_quotient_exact_one_gate_add_no_positive_one_detail(
                ctx,
                "denominator",
                denominator,
                gate_sample.clone(),
            );
        }
    }
    let denominator_core = strip_positive_one_passthrough(ctx, denominator)?;

    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, numerator_core),
            render_expr_for_orchestrator_profile(ctx, denominator_core)
        )
    });
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, pair_sample.clone(), || Some(()));
        }
    };
    if profiling {
        profile_shifted_quotient_exact_one_rule_apply_pair_family(
            ctx,
            numerator_core,
            denominator_core,
            pair_sample.clone(),
        );
    }
    let maybe_tanh_exp_pair =
        maybe_two_term_tanh_exp_equivalence_candidate(ctx, numerator_core, denominator_core);
    let maybe_hyperbolic_direct_pair = maybe_two_term_hyperbolic_direct_core_equivalence_candidate(
        ctx,
        numerator_core,
        denominator_core,
    );
    let maybe_sinh_cosh_exp_definition_pair =
        (expr_contains_any_builtin(ctx, numerator_core, &[BuiltinFn::Sinh, BuiltinFn::Cosh])
            && contains_division_like_term(ctx, denominator_core))
            || (expr_contains_any_builtin(
                ctx,
                denominator_core,
                &[BuiltinFn::Sinh, BuiltinFn::Cosh],
            ) && contains_division_like_term(ctx, numerator_core));

    if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.sub_fraction",
        &pair_sample,
        || {
            try_build_direct_sub_fraction_combination_equivalence_rewrite(
                ctx,
                numerator_core,
                denominator_core,
            )
        },
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.sub_fraction");
        return Some(build_shifted_quotient_exact_one_rewrite(
            ctx,
            numerator,
            denominator,
            child_rewrite,
        ));
    }
    if maybe_tanh_exp_pair {
        if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
            profiling,
            "rule.shifted_quotient.exact_one.try.tanh_exp",
            &pair_sample,
            || {
                try_build_direct_tanh_exp_definition_equivalence_rewrite(
                    ctx,
                    numerator_core,
                    denominator_core,
                )
            },
        ) {
            profile_route("rule.shifted_quotient.exact_one.route.tanh_exp");
            return Some(build_shifted_quotient_exact_one_rewrite(
                ctx,
                numerator,
                denominator,
                child_rewrite,
            ));
        }
    }
    if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.trig_square",
        &pair_sample,
        || try_build_direct_trig_square_equivalence_rewrite(ctx, numerator_core, denominator_core),
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.trig_square");
        return Some(build_shifted_quotient_exact_one_rewrite(
            ctx,
            numerator,
            denominator,
            child_rewrite,
        ));
    }
    if let Some(rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.half_angle_binomial",
        &pair_sample,
        || {
            try_build_direct_half_angle_binomial_square_shifted_quotient_rewrite(
                ctx,
                numerator,
                denominator,
                numerator_core,
                denominator_core,
            )
        },
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.half_angle_binomial");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.small_zero_core",
        &pair_sample,
        || {
            try_build_direct_small_zero_core_shifted_quotient_rewrite(
                ctx,
                numerator,
                denominator,
                numerator_core,
                denominator_core,
            )
        },
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.small_zero_core");
        return Some(rewrite);
    }
    if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.sinh_cubic",
        &pair_sample,
        || {
            try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
                ctx,
                numerator_core,
                denominator_core,
            )
        },
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.sinh_cubic");
        return Some(build_shifted_quotient_exact_one_rewrite(
            ctx,
            numerator,
            denominator,
            child_rewrite,
        ));
    }
    if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.phase_shift_pair",
        &pair_sample,
        || {
            try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                ctx,
                numerator_core,
                denominator_core,
            )
        },
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.phase_shift_pair");
        return Some(build_shifted_quotient_exact_one_rewrite(
            ctx,
            numerator,
            denominator,
            child_rewrite,
        ));
    }
    if maybe_hyperbolic_direct_pair {
        if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
            profiling,
            "rule.shifted_quotient.exact_one.try.safe_hyperbolic",
            &pair_sample,
            || {
                try_build_direct_safe_hyperbolic_core_equivalence_rewrite(
                    ctx,
                    numerator_core,
                    denominator_core,
                )
            },
        ) {
            profile_route("rule.shifted_quotient.exact_one.route.safe_hyperbolic");
            return Some(build_shifted_quotient_exact_one_rewrite(
                ctx,
                numerator,
                denominator,
                child_rewrite,
            ));
        }
    }
    if maybe_sinh_cosh_exp_definition_pair {
        if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
            profiling,
            "rule.shifted_quotient.exact_one.try.sinh_cosh_exp_definition",
            &pair_sample,
            || {
                try_build_shifted_quotient_sinh_cosh_exp_definition_rewrite(
                    ctx,
                    numerator_core,
                    denominator_core,
                )
            },
        ) {
            profile_route("rule.shifted_quotient.exact_one.route.sinh_cosh_exp_definition");
            profile_shifted_quotient_exact_one_route_pair_family(
                "sinh_cosh_exp_definition",
                ctx,
                numerator_core,
                denominator_core,
                pair_sample.clone(),
            );
            return Some(build_shifted_quotient_exact_one_rewrite(
                ctx,
                numerator,
                denominator,
                child_rewrite,
            ));
        }
    }
    let residual_difference = ctx.add(Expr::Sub(numerator_core, denominator_core));
    let residual_contains_log_or_abs = expr_contains_any_builtin(
        ctx,
        residual_difference,
        &[
            BuiltinFn::Ln,
            BuiltinFn::Log,
            BuiltinFn::Log10,
            BuiltinFn::Abs,
        ],
    );
    if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.repeated_phase_shift_residual",
        &pair_sample,
        || try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_difference),
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.repeated_phase_shift_residual");
        return Some(build_shifted_quotient_exact_one_rewrite(
            ctx,
            numerator,
            denominator,
            child_rewrite,
        ));
    }
    if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.shared_passthrough_residual",
        &pair_sample,
        || try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_difference),
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.shared_passthrough_residual");
        profile_shifted_quotient_exact_one_route_pair_family(
            "shared_passthrough_residual",
            ctx,
            numerator_core,
            denominator_core,
            pair_sample.clone(),
        );
        return Some(build_shifted_quotient_exact_one_rewrite(
            ctx,
            numerator,
            denominator,
            child_rewrite,
        ));
    }
    if maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
        ctx,
        numerator_core,
        denominator_core,
        residual_difference,
    ) {
        if let Some(child_rewrite) = run_profiled_shifted_quotient_exact_one_route(
            profiling,
            "rule.shifted_quotient.exact_one.try.exact_zero_direct_residual",
            &pair_sample,
            || try_build_exact_zero_identity_rewrite_direct(ctx, residual_difference),
        ) {
            let zero = ctx.num(0);
            if compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal {
                profile_route("rule.shifted_quotient.exact_one.route.exact_zero_direct_residual");
                profile_shifted_quotient_exact_one_route_pair_family(
                    "exact_zero_direct_residual",
                    ctx,
                    numerator_core,
                    denominator_core,
                    pair_sample.clone(),
                );
                return Some(build_shifted_quotient_exact_one_rewrite(
                    ctx,
                    numerator,
                    denominator,
                    child_rewrite,
                ));
            }
        }
    }
    let residual_contains_strippable_zero_term =
        residual_contains_log_or_abs && additive_scope_contains_zero_term(ctx, residual_difference);
    let residual_child_rewrite = residual_contains_strippable_zero_term
        .then(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.stripped_zero_log_residual",
                &pair_sample,
                || try_build_stripped_zero_log_identity_child_rewrite(ctx, residual_difference),
            )
            .inspect(|_| {
                profile_route("rule.shifted_quotient.exact_one.route.stripped_zero_log_residual");
            })
        })
        .flatten()
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.fast_multiterm_hyperbolic_residual",
                &pair_sample,
                || {
                    try_build_fast_multiterm_hyperbolic_residual_child_rewrite(
                        ctx,
                        residual_difference,
                    )
                },
            )
            .inspect(|_| {
                profile_route(
                    "rule.shifted_quotient.exact_one.route.fast_multiterm_hyperbolic_residual",
                );
            })
        })
        .or_else(|| {
            maybe_hyperbolic_direct_pair
                .then(|| {
                    run_profiled_shifted_quotient_exact_one_route(
                        profiling,
                        "rule.shifted_quotient.exact_one.try.safe_hyperbolic_residual_pair",
                        &pair_sample,
                        || {
                            try_build_direct_safe_hyperbolic_core_equivalence_rewrite(
                                ctx,
                                numerator_core,
                                denominator_core,
                            )
                        },
                    )
                    .inspect(|_| {
                        profile_route(
                            "rule.shifted_quotient.exact_one.route.safe_hyperbolic_residual_pair",
                        );
                    })
                })
                .flatten()
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.fast_trig_residual",
                &pair_sample,
                || try_build_fast_trig_residual_identity_child_rewrite(ctx, residual_difference),
            )
            .inspect(|_| {
                profile_route("rule.shifted_quotient.exact_one.route.fast_trig_residual");
            })
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.fast_small_polynomial_residual",
                &pair_sample,
                || try_build_fast_small_polynomial_residual_child_rewrite(ctx, residual_difference),
            )
            .inspect(|_| {
                profile_route(
                    "rule.shifted_quotient.exact_one.route.fast_small_polynomial_residual",
                );
            })
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.power_merge_residual_narrow",
                &pair_sample,
                || {
                    try_build_shifted_quotient_power_merge_residual_rewrite(
                        ctx,
                        numerator_core,
                        denominator_core,
                    )
                },
            )
            .inspect(|_| {
                profile_route("rule.shifted_quotient.exact_one.route.power_merge_residual_narrow");
            })
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.cancel_common_factors_residual_narrow",
                &pair_sample,
                || {
                    try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
                        ctx,
                        numerator_core,
                        denominator_core,
                    )
                },
            )
            .inspect(|_| {
                profile_route(
                    "rule.shifted_quotient.exact_one.route.cancel_common_factors_residual_narrow",
                );
            })
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.fraction_combine_residual_narrow",
                &pair_sample,
                || {
                    try_build_shifted_quotient_fraction_combine_residual_rewrite(
                        ctx,
                        numerator_core,
                        denominator_core,
                    )
                },
            )
            .inspect(|_| {
                profile_route(
                    "rule.shifted_quotient.exact_one.route.fraction_combine_residual_narrow",
                );
            })
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.nested_fraction_residual_narrow",
                &pair_sample,
                || {
                    try_build_shifted_quotient_nested_fraction_residual_rewrite(
                        ctx,
                        numerator_core,
                        denominator_core,
                    )
                },
            )
            .inspect(|_| {
                profile_route(
                    "rule.shifted_quotient.exact_one.route.nested_fraction_residual_narrow",
                );
            })
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.fraction_decompose_residual_narrow",
                &pair_sample,
                || {
                    try_build_shifted_quotient_fraction_decompose_residual_rewrite(
                        ctx,
                        numerator_core,
                        denominator_core,
                    )
                },
            )
            .inspect(|_| {
                profile_route(
                    "rule.shifted_quotient.exact_one.route.fraction_decompose_residual_narrow",
                );
            })
        })
        .or_else(|| {
            run_profiled_shifted_quotient_exact_one_route(
                profiling,
                "rule.shifted_quotient.exact_one.try.direct_core_equivalence",
                &pair_sample,
                || try_build_direct_core_equivalence_rewrite(ctx, numerator_core, denominator_core),
            )
            .inspect(|_| {
                profile_route("rule.shifted_quotient.exact_one.route.direct_core_equivalence");
                profile_shifted_quotient_exact_one_direct_core_equivalence_family(
                    ctx,
                    numerator_core,
                    denominator_core,
                    pair_sample.clone(),
                );
                profile_shifted_quotient_exact_one_route_pair_family(
                    "direct_core_equivalence",
                    ctx,
                    numerator_core,
                    denominator_core,
                    pair_sample.clone(),
                );
            })
        });
    if let Some(child_rewrite) = residual_child_rewrite {
        return Some(build_shifted_quotient_exact_one_rewrite(
            ctx,
            numerator,
            denominator,
            child_rewrite,
        ));
    }
    let child_rewrite = if let Some(rewrite) = run_profiled_shifted_quotient_exact_one_route(
        profiling,
        "rule.shifted_quotient.exact_one.try.exact_zero_identity",
        &pair_sample,
        || try_build_exact_zero_identity_rewrite(ctx, residual_difference),
    ) {
        profile_route("rule.shifted_quotient.exact_one.route.exact_zero_identity");
        rewrite
    } else {
        let rewrite = run_profiled_shifted_quotient_exact_one_route(
            profiling,
            "rule.shifted_quotient.exact_one.try.default_simplify_residual_zero",
            &pair_sample,
            || {
                let zero = ctx.num(0);
                let simplified_difference = run_default_simplify(ctx, residual_difference);
                (compare_expr(ctx, simplified_difference, zero) == Ordering::Equal).then(|| {
                    Rewrite::with_local(
                        ctx.num(0),
                        "Equivalent Residual Cancellation",
                        numerator_core,
                        denominator_core,
                    )
                })
            },
        )?;
        profile_route("rule.shifted_quotient.exact_one.route.default_simplify_residual_zero");
        rewrite
    };

    Some(build_shifted_quotient_exact_one_rewrite(
        ctx,
        numerator,
        denominator,
        child_rewrite,
    ))
}

pub(super) fn try_build_shifted_quotient_power_merge_residual_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    if matches_shifted_quotient_power_merge_residual_side(ctx, lhs_core, rhs_core)
        || matches_shifted_quotient_power_merge_residual_side(ctx, rhs_core, lhs_core)
    {
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    None
}

pub(super) fn try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    try_rewrite_shifted_quotient_cancel_common_factors_source(ctx, lhs_core, rhs_core).or_else(
        || try_rewrite_shifted_quotient_cancel_common_factors_source(ctx, rhs_core, lhs_core),
    )
}

pub(super) fn try_build_shifted_quotient_fraction_combine_residual_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    if matches_shifted_quotient_fraction_combine_residual_side(ctx, lhs_core, rhs_core)
        || matches_shifted_quotient_fraction_combine_residual_side(ctx, rhs_core, lhs_core)
    {
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Combine Fractions",
            lhs_core,
            rhs_core,
        ));
    }

    None
}

pub(crate) fn matches_shifted_quotient_fraction_residual_narrow_pair_candidate(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    matches_shifted_quotient_fraction_combine_residual_side(ctx, lhs_core, rhs_core)
        || matches_shifted_quotient_fraction_combine_residual_side(ctx, rhs_core, lhs_core)
        || try_rewrite_shifted_quotient_fraction_decompose_source(ctx, lhs_core, rhs_core).is_some()
        || try_rewrite_shifted_quotient_fraction_decompose_source(ctx, rhs_core, lhs_core).is_some()
}

fn try_rewrite_shifted_quotient_cancel_common_factors_source(
    ctx: &mut cas_ast::Context,
    source_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let Expr::Div(source_num, source_den) = ctx.get(source_expr).clone() else {
        return None;
    };

    if expr_contains_any_function_call(ctx, source_expr)
        || expr_contains_any_function_call(ctx, target_expr)
        || matches!(ctx.get(source_num), Expr::Add(_, _) | Expr::Sub(_, _))
        || matches!(ctx.get(source_den), Expr::Add(_, _) | Expr::Sub(_, _))
        || matches!(ctx.get(target_expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || contains_division_like_term(ctx, source_num)
        || contains_division_like_term(ctx, source_den)
    {
        return None;
    }

    let rewrite_match = cas_math::fraction_factors::try_rewrite_cancel_common_factors_expr_with(
        ctx,
        source_expr,
        |ctx, nonzero_base, emit_assumption| {
            if is_zero_expr(ctx, nonzero_base) {
                return cas_math::fraction_factors::CancelCommonFactorsGate {
                    allow: false,
                    assumed: false,
                };
            }

            cas_math::fraction_factors::CancelCommonFactorsGate {
                allow: true,
                assumed: emit_assumption,
            }
        },
    )?;

    let rewritten = rewrite_match.rewritten;
    let residual = ctx.add(Expr::Sub(rewritten, target_expr));
    if !(exprs_match_for_cancellation(ctx, rewritten, target_expr)
        || exprs_match_after_default_simplify(ctx, rewritten, target_expr)
        || is_zero_after_default_simplify(ctx, residual))
    {
        return None;
    }

    let mut rewrite = Rewrite::with_local(
        ctx.num(0),
        "Equivalent Residual Cancellation",
        source_expr,
        target_expr,
    );
    let mut seen_nonzero_targets: Vec<cas_ast::ExprId> = Vec::new();
    for nonzero_target in rewrite_match.assumed_nonzero_targets {
        if seen_nonzero_targets
            .iter()
            .any(|existing| compare_expr(ctx, *existing, nonzero_target) == Ordering::Equal)
        {
            continue;
        }
        seen_nonzero_targets.push(nonzero_target);
        rewrite = rewrite.requires(crate::ImplicitCondition::NonZero(nonzero_target));
    }

    Some(rewrite)
}

pub(super) fn try_build_shifted_quotient_nested_fraction_residual_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    if matches_shifted_quotient_nested_fraction_residual_side(ctx, lhs_core, rhs_core)
        || matches_shifted_quotient_nested_fraction_residual_side(ctx, rhs_core, lhs_core)
    {
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Simplify Nested Fraction",
            lhs_core,
            rhs_core,
        ));
    }

    None
}

fn try_rewrite_shifted_quotient_fraction_decompose_source(
    ctx: &mut cas_ast::Context,
    source_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if expr_contains_any_function_call(ctx, source_expr)
        || expr_contains_any_function_call(ctx, target_expr)
    {
        return None;
    }

    let (_source_num, source_den) = as_div(ctx, source_expr)?;
    let source_den_factors = extract_atomic_noncall_factor_set(ctx, source_den)?;
    let target_terms = AddView::from_expr(ctx, target_expr).terms;
    if !(2..=4).contains(&target_terms.len()) {
        return None;
    }

    let mut numerator_terms = Vec::with_capacity(target_terms.len());
    for (term_expr, term_sign) in target_terms {
        let contribution =
            if let Some((term_num, term_den)) = extract_fraction_like_add_term(ctx, term_expr) {
                let term_den_factors = extract_atomic_noncall_factor_set(ctx, term_den)?;
                if term_den_factors.iter().any(|term_factor| {
                    !source_den_factors.iter().any(|source_factor| {
                        compare_expr(ctx, *source_factor, *term_factor) == Ordering::Equal
                    })
                }) {
                    return None;
                }

                let missing_factors: Vec<_> = source_den_factors
                    .iter()
                    .copied()
                    .filter(|source_factor| {
                        !term_den_factors.iter().any(|term_factor| {
                            compare_expr(ctx, *source_factor, *term_factor) == Ordering::Equal
                        })
                    })
                    .collect();
                let missing_factor_expr = build_mul_expr_from_factors(ctx, &missing_factors);
                smart_mul(ctx, term_num, missing_factor_expr)
            } else {
                if contains_division_like_term(ctx, term_expr)
                    || expr_contains_any_function_call(ctx, term_expr)
                    || matches!(ctx.get(term_expr), Expr::Add(_, _) | Expr::Sub(_, _))
                {
                    return None;
                }
                smart_mul(ctx, term_expr, source_den)
            };

        numerator_terms.push((contribution, term_sign));
    }

    let rebuilt_numerator = build_signed_sum_expr(ctx, &numerator_terms);
    let rebuilt = ctx.add(Expr::Div(rebuilt_numerator, source_den));
    let residual = ctx.add(Expr::Sub(rebuilt, source_expr));
    if exprs_match_for_cancellation(ctx, rebuilt, source_expr)
        || exprs_match_after_default_simplify(ctx, rebuilt, source_expr)
        || is_zero_after_default_simplify(ctx, residual)
    {
        Some(rebuilt)
    } else {
        None
    }
}

fn try_build_shifted_quotient_fraction_decompose_residual_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    if try_rewrite_shifted_quotient_fraction_decompose_source(ctx, lhs_core, rhs_core).is_some()
        || try_rewrite_shifted_quotient_fraction_decompose_source(ctx, rhs_core, lhs_core).is_some()
    {
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    None
}

fn matches_shifted_quotient_fraction_combine_residual_side(
    ctx: &mut cas_ast::Context,
    source_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> bool {
    if expr_contains_any_function_call(ctx, source_expr)
        || expr_contains_any_function_call(ctx, target_expr)
        || !matches!(ctx.get(target_expr), Expr::Div(_, _))
    {
        return false;
    }

    let Some(rewritten) = try_rewrite_shifted_quotient_fraction_combine_source(ctx, source_expr)
    else {
        return false;
    };

    let residual = ctx.add(Expr::Sub(rewritten, target_expr));
    exprs_match_for_cancellation(ctx, rewritten, target_expr)
        || exprs_match_after_default_simplify(ctx, rewritten, target_expr)
        || is_zero_after_default_simplify(ctx, residual)
}

fn matches_shifted_quotient_nested_fraction_residual_side(
    ctx: &mut cas_ast::Context,
    source_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> bool {
    if expr_contains_any_function_call(ctx, source_expr)
        || expr_contains_any_function_call(ctx, target_expr)
        || !matches!(ctx.get(target_expr), Expr::Div(_, _))
    {
        return false;
    }

    let Some(rewrite) = try_rewrite_simplify_nested_fraction_expr(ctx, source_expr) else {
        return false;
    };
    let rewritten = rewrite.rewritten;
    let residual = ctx.add(Expr::Sub(rewritten, target_expr));
    exprs_match_for_cancellation(ctx, rewritten, target_expr)
        || exprs_match_after_default_simplify(ctx, rewritten, target_expr)
        || is_zero_after_default_simplify(ctx, residual)
}

fn try_rewrite_shifted_quotient_fraction_combine_source(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    let left = terms[0].0;
    let right = terms[1].0;

    if let Some(ops) = extract_fold_add_operands(ctx, left, right) {
        if !contains_division_like_term(ctx, ops.term) {
            if let Some(rewritten) = try_build_fold_add_fraction_rewrite(
                ctx,
                expr,
                ops.term,
                ops.numerator,
                ops.denominator,
            ) {
                return Some(rewritten);
            }
        }
    }

    if let Some(rewritten) =
        try_build_shifted_quotient_combined_fraction_from_scaled_fold_add(ctx, left, right)
    {
        return Some(rewritten);
    }

    if let Some(plan) =
        try_plan_same_denominator_combination_with(ctx, expr, false, false, |_ctx, _den| false)
    {
        return Some(plan.build.result);
    }

    try_plan_shifted_quotient_add_fraction_pair(ctx, expr)
}

fn try_build_shifted_quotient_combined_fraction_from_scaled_fold_add(
    ctx: &mut cas_ast::Context,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let candidates = [(left, right), (right, left)];

    for (whole_term, remainder_term) in candidates {
        let Expr::Div(whole_num, whole_den) = ctx.get(whole_term) else {
            continue;
        };
        let Expr::Div(rem_num, rem_den) = ctx.get(remainder_term) else {
            continue;
        };
        let whole_num = *whole_num;
        let whole_den = *whole_den;
        let rem_num = *rem_num;
        let rem_den = *rem_den;

        let candidate_vars = cas_ast::collect_variables(ctx, rem_den);
        for var_name in candidate_vars {
            if contains_named_var(ctx, whole_num, &var_name)
                || contains_named_var(ctx, whole_den, &var_name)
                || contains_named_var(ctx, rem_num, &var_name)
            {
                continue;
            }

            let Some((linear_coeff, offset)) = get_linear_coeffs(ctx, rem_den, &var_name) else {
                continue;
            };
            let zero = ctx.num(0);
            if compare_expr(ctx, linear_coeff, zero) == Ordering::Equal {
                continue;
            }
            let coeff_matches_whole_den = compare_expr(ctx, linear_coeff, whole_den)
                == Ordering::Equal
                || poly_eq(ctx, linear_coeff, whole_den);
            let neg_whole_den = ctx.add(Expr::Neg(whole_den));
            let coeff_matches_neg_whole_den = compare_expr(ctx, linear_coeff, neg_whole_den)
                == Ordering::Equal
                || poly_eq(ctx, linear_coeff, neg_whole_den);
            if !coeff_matches_whole_den && !coeff_matches_neg_whole_den {
                continue;
            }

            let var_expr = ctx.var(&var_name);
            let signed_whole_num = if coeff_matches_neg_whole_den {
                ctx.add(Expr::Neg(whole_num))
            } else {
                whole_num
            };
            let whole_times_var = ctx.add(Expr::Mul(signed_whole_num, var_expr));
            let whole_times_offset = ctx.add(Expr::Mul(whole_num, offset));
            let lifted_offset = ctx.add(Expr::Div(whole_times_offset, whole_den));
            let numerator_tail = ctx.add(Expr::Add(lifted_offset, rem_num));
            let numerator = ctx.add(Expr::Add(whole_times_var, numerator_tail));
            return Some(ctx.add(Expr::Div(numerator, rem_den)));
        }
    }

    None
}

fn try_plan_shifted_quotient_add_fraction_pair(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    let left = terms[0].0;
    let right = terms[1].0;
    let pair = extract_fraction_pair(ctx, left, right);
    if !pair.is_frac1 || !pair.is_frac2 {
        return None;
    }

    let plan = plan_add_fraction_rewrite_with(
        ctx,
        AddFractionRewriteInput {
            expr,
            l: left,
            r: right,
            n1: pair.n1,
            d1: pair.d1,
            n2: pair.n2,
            d2: pair.d2,
            same_sign: pair.sign1 == pair.sign2,
            inside_trig: false,
        },
        cas_math::expand_ops::expand,
    )?;

    Some(plan.rewritten)
}

fn matches_shifted_quotient_power_merge_residual_side(
    ctx: &mut cas_ast::Context,
    source_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> bool {
    let Some((target_base, target_exp)) =
        extract_shifted_quotient_power_merge_target(ctx, target_expr)
    else {
        return false;
    };
    let Some((source_base, source_exp)) =
        extract_shifted_quotient_power_merge_source(ctx, source_expr)
    else {
        return false;
    };

    if compare_expr(ctx, source_base, target_base) != Ordering::Equal {
        return false;
    }

    let normalized_source_exp = cas_math::canonical_forms::normalize_core(ctx, source_exp);
    let normalized_target_exp = cas_math::canonical_forms::normalize_core(ctx, target_exp);
    compare_expr(ctx, normalized_source_exp, normalized_target_exp) == Ordering::Equal
        || exprs_match_after_default_simplify(ctx, normalized_source_exp, target_exp)
        || exprs_match_after_default_simplify(ctx, source_exp, target_exp)
}

fn extract_shifted_quotient_power_merge_target(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if expr_contains_any_function_call(ctx, expr) {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };

    if !power_merge_base_supported_for_cancellation(ctx, *base) {
        return None;
    }

    Some((*base, *exp))
}

fn extract_shifted_quotient_power_merge_source(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if let Expr::Div(numerator, denominator) = ctx.get(expr).clone() {
        let numerator_factors: smallvec::SmallVec<[cas_ast::ExprId; 4]> =
            if cas_math::expr_predicates::is_one_expr(ctx, numerator) {
                smallvec::SmallVec::new()
            } else {
                flatten_mul_chain(ctx, numerator).into_iter().collect()
            };
        let denominator_factors: smallvec::SmallVec<[cas_ast::ExprId; 4]> =
            if cas_math::expr_predicates::is_one_expr(ctx, denominator) {
                smallvec::SmallVec::new()
            } else {
                flatten_mul_chain(ctx, denominator).into_iter().collect()
            };
        let total_factor_count = numerator_factors.len() + denominator_factors.len();
        if !(1..=2).contains(&total_factor_count) {
            return None;
        }

        let mut combined_base = None;
        let mut combined_exp = None;
        for factor in numerator_factors {
            let (factor_base, factor_exp) =
                extract_shifted_quotient_power_merge_factor(ctx, factor)?;
            if let Some(existing_base) = combined_base {
                if compare_expr(ctx, existing_base, factor_base) != Ordering::Equal {
                    return None;
                }
            } else {
                combined_base = Some(factor_base);
            }

            combined_exp = Some(match combined_exp {
                Some(current_exp) => {
                    cas_math::exponents_support::add_exp(ctx, current_exp, factor_exp)
                }
                None => factor_exp,
            });
        }
        for factor in denominator_factors {
            let (factor_base, factor_exp) =
                extract_shifted_quotient_power_merge_factor(ctx, factor)?;
            if let Some(existing_base) = combined_base {
                if compare_expr(ctx, existing_base, factor_base) != Ordering::Equal {
                    return None;
                }
            } else {
                combined_base = Some(factor_base);
            }

            let neg_factor_exp = ctx.add(Expr::Neg(factor_exp));
            combined_exp = Some(match combined_exp {
                Some(current_exp) => {
                    cas_math::exponents_support::add_exp(ctx, current_exp, neg_factor_exp)
                }
                None => neg_factor_exp,
            });
        }

        return Some((combined_base?, combined_exp?));
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let (lhs_base, lhs_exp) = extract_shifted_quotient_power_merge_factor(ctx, factors[0])?;
    let (rhs_base, rhs_exp) = extract_shifted_quotient_power_merge_factor(ctx, factors[1])?;

    if compare_expr(ctx, lhs_base, rhs_base) != Ordering::Equal
        || !power_merge_base_supported_for_cancellation(ctx, lhs_base)
    {
        return None;
    }

    let combined_exp = cas_math::exponents_support::add_exp(ctx, lhs_exp, rhs_exp);
    Some((lhs_base, combined_exp))
}

fn extract_shifted_quotient_power_merge_factor(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if let Some(base) = extract_sqrt_argument(ctx, expr) {
        return Some((
            base,
            ctx.add(Expr::Number(num_rational::BigRational::new(
                1.into(),
                2.into(),
            ))),
        ));
    }

    if expr_contains_any_function_call(ctx, expr) {
        return None;
    }

    match ctx.get(expr) {
        Expr::Pow(base, exp) if power_merge_base_supported_for_cancellation(ctx, *base) => {
            Some((*base, *exp))
        }
        _ if expr_is_atomic_noncall(ctx, expr)
            && expr_contains_symbolic_atom_for_cancellation(ctx, expr) =>
        {
            Some((expr, ctx.num(1)))
        }
        _ => None,
    }
}

pub(crate) fn extract_repeated_trig_phase_shift_pair_zero_chunks(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let normalized_expr = normalize_additive_scope_expr(ctx, expr);
    let view = AddView::from_expr(ctx, normalized_expr);
    if view.terms.len() != 6 {
        return None;
    }

    let shifted_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, (term_expr, term_sign))| {
            let signed_expr = apply_sign_to_expr(ctx, sign_to_i64(term_sign), term_expr);
            let (base_arg, coeff, kind, sin_sign, cos_sign) =
                extract_exact_phase_shift_term_data_for_cancellation(ctx, signed_expr)?;
            Some((index, base_arg, coeff, kind, sin_sign, cos_sign))
        })
        .collect();

    if shifted_terms.len() != 2 {
        return None;
    }

    let linear_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| {
            extract_signed_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, term_expr, term_sign)
        })
        .collect();

    let mut used = vec![false; view.terms.len()];
    let mut matched_chunks = Vec::new();

    for (shifted_index, base_arg, coeff, kind, sin_sign, cos_sign) in shifted_terms {
        if used[shifted_index] {
            continue;
        }

        let mut sin_index = None;
        let mut cos_index = None;
        for index in 0..view.terms.len() {
            if used[index] || index == shifted_index {
                continue;
            }
            let Some((trig_fn, raw_arg, _coeff, _signed_coeff_sign)) = linear_terms[index] else {
                continue;
            };
            if compare_expr(ctx, raw_arg, base_arg) != Ordering::Equal {
                continue;
            }

            match trig_fn {
                BuiltinFn::Sin if sin_index.is_none() => sin_index = Some(index),
                BuiltinFn::Cos if cos_index.is_none() => cos_index = Some(index),
                _ => {}
            }
        }

        let (sin_index, cos_index) = (sin_index?, cos_index?);
        let Some((BuiltinFn::Sin, sin_arg, sin_coeff, signed_sin_sign)) = linear_terms[sin_index]
        else {
            return None;
        };
        let Some((BuiltinFn::Cos, cos_arg, cos_coeff, signed_cos_sign)) = linear_terms[cos_index]
        else {
            return None;
        };
        if !matches_exact_phase_shift_linear_combination_target_from_extracted(
            ctx,
            (sin_arg, sin_coeff, signed_sin_sign),
            (cos_arg, cos_coeff, signed_cos_sign),
            base_arg,
            coeff,
            kind,
            (sin_sign, cos_sign),
        ) {
            return None;
        }

        let chunk_terms = [
            view.terms[sin_index],
            view.terms[cos_index],
            view.terms[shifted_index],
        ];
        let chunk_expr = build_signed_sum_expr(ctx, &chunk_terms);

        used[sin_index] = true;
        used[cos_index] = true;
        used[shifted_index] = true;
        matched_chunks.push(chunk_expr);
    }

    if matched_chunks.len() != 2 || used.iter().any(|used_term| !*used_term) {
        return None;
    }

    let first_expr = matched_chunks.remove(0);
    let second_expr = matched_chunks.remove(0);
    Some((first_expr, second_expr))
}

pub(super) fn try_build_fast_repeated_trig_phase_shift_pair_zero_rewrite(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> Option<Rewrite> {
    if terms.len() != 6 {
        return None;
    }

    if let Some(rewrite) =
        try_build_fast_unit_exact_quarter_shifted_sine_pair_zero_rewrite(ctx, terms)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_fast_structural_exact_quarter_phase_shift_pair_zero_rewrite(ctx, terms)
    {
        return Some(rewrite);
    }

    let zero = ctx.num(0);
    let full_expr = build_signed_sum_expr(ctx, terms);
    let (first_expr, second_expr) =
        extract_repeated_trig_phase_shift_pair_zero_chunks(ctx, full_expr)?;

    let mut rewrite = Rewrite::with_local(zero, "Phase Shift Identity", full_expr, zero);
    rewrite.substeps = vec![
        build_phase_shift_zero_substep(ctx, first_expr),
        build_phase_shift_zero_substep(ctx, second_expr),
    ];
    Some(rewrite)
}

fn try_build_fast_unit_exact_quarter_shifted_sine_pair_zero_rewrite(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> Option<Rewrite> {
    let shifted_terms: Vec<_> = terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, (term_expr, term_sign))| {
            let signed_expr = apply_sign_to_expr(ctx, sign_to_i64(term_sign), term_expr);
            let (base_arg, sin_sign, cos_sign) =
                extract_structural_unit_exact_quarter_shifted_sine_term_for_phase_shift_pair(
                    ctx,
                    signed_expr,
                )?;
            Some((index, base_arg, sin_sign, cos_sign))
        })
        .collect();

    if shifted_terms.len() != 2 {
        return None;
    }

    let linear_terms: Vec<_> = terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| {
            let signed_expr = apply_sign_to_expr(ctx, sign_to_i64(term_sign), term_expr);
            extract_structural_unit_linear_trig_term_for_phase_shift_pair(ctx, signed_expr)
        })
        .collect();

    let mut used = vec![false; terms.len()];
    let zero = ctx.num(0);
    let mut matched_rewrites = Vec::new();

    for (shifted_index, base_arg, sin_sign, cos_sign) in shifted_terms {
        if used[shifted_index] {
            continue;
        }

        let mut sin_index = None;
        let mut cos_index = None;
        for index in 0..terms.len() {
            if used[index] || index == shifted_index {
                continue;
            }
            let Some((trig_fn, raw_arg, sign)) = linear_terms[index] else {
                continue;
            };
            if compare_expr(ctx, raw_arg, base_arg) != Ordering::Equal {
                continue;
            }

            match trig_fn {
                BuiltinFn::Sin if sign == sin_sign && sin_index.is_none() => {
                    sin_index = Some(index)
                }
                BuiltinFn::Cos if sign == cos_sign && cos_index.is_none() => {
                    cos_index = Some(index)
                }
                _ => {}
            }
        }

        let (sin_index, cos_index) = (sin_index?, cos_index?);
        let triple_terms = [terms[sin_index], terms[cos_index], terms[shifted_index]];
        let triple_expr = build_signed_sum_expr(ctx, &triple_terms);

        used[sin_index] = true;
        used[cos_index] = true;
        used[shifted_index] = true;
        matched_rewrites.push(triple_expr);
    }

    if matched_rewrites.len() != 2 || used.iter().any(|used_term| !*used_term) {
        return None;
    }

    let first_expr = matched_rewrites.remove(0);
    let second_expr = matched_rewrites.remove(0);
    let full_expr = build_signed_sum_expr(ctx, terms);

    let mut rewrite = Rewrite::with_local(zero, "Phase Shift Identity", full_expr, zero);
    rewrite.substeps = vec![
        build_phase_shift_zero_substep(ctx, first_expr),
        build_phase_shift_zero_substep(ctx, second_expr),
    ];
    Some(rewrite)
}

fn try_build_fast_structural_exact_quarter_phase_shift_pair_zero_rewrite(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> Option<Rewrite> {
    let shifted_terms: Vec<_> = terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, (term_expr, term_sign))| {
            let signed_expr = apply_sign_to_expr(ctx, sign_to_i64(term_sign), term_expr);
            let (base_arg, coeff, sin_sign, cos_sign) =
                extract_structural_exact_quarter_phase_shift_term_data_for_cancellation(
                    ctx,
                    signed_expr,
                )?;
            Some((index, base_arg, coeff, sin_sign, cos_sign))
        })
        .collect();

    if shifted_terms.len() != 2 {
        return None;
    }

    let linear_terms: Vec<_> = terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| {
            extract_signed_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, term_expr, term_sign)
        })
        .collect();

    let mut used = vec![false; terms.len()];
    let zero = ctx.num(0);
    let mut matched_rewrites = Vec::new();

    for (shifted_index, base_arg, coeff, sin_sign, cos_sign) in shifted_terms {
        if used[shifted_index] {
            continue;
        }

        let mut sin_index = None;
        let mut cos_index = None;
        for index in 0..terms.len() {
            if used[index] || index == shifted_index {
                continue;
            }
            let Some((trig_fn, raw_arg, _coeff, _signed_coeff_sign)) = linear_terms[index] else {
                continue;
            };
            if compare_expr(ctx, raw_arg, base_arg) != Ordering::Equal {
                continue;
            }

            match trig_fn {
                BuiltinFn::Sin if sin_index.is_none() => sin_index = Some(index),
                BuiltinFn::Cos if cos_index.is_none() => cos_index = Some(index),
                _ => {}
            }
        }

        let (sin_index, cos_index) = (sin_index?, cos_index?);
        let Some((BuiltinFn::Sin, sin_arg, sin_coeff, signed_sin_sign)) = linear_terms[sin_index]
        else {
            return None;
        };
        let Some((BuiltinFn::Cos, cos_arg, cos_coeff, signed_cos_sign)) = linear_terms[cos_index]
        else {
            return None;
        };
        if !matches_exact_phase_shift_linear_combination_target_from_extracted(
            ctx,
            (sin_arg, sin_coeff, signed_sin_sign),
            (cos_arg, cos_coeff, signed_cos_sign),
            base_arg,
            coeff,
            PhaseShiftKindForCancellation::Quarter,
            (sin_sign, cos_sign),
        ) {
            return None;
        }

        let triple_terms = [terms[sin_index], terms[cos_index], terms[shifted_index]];
        let triple_expr = build_signed_sum_expr(ctx, &triple_terms);

        used[sin_index] = true;
        used[cos_index] = true;
        used[shifted_index] = true;
        matched_rewrites.push(triple_expr);
    }

    if matched_rewrites.len() != 2 || used.iter().any(|used_term| !*used_term) {
        return None;
    }

    let first_expr = matched_rewrites.remove(0);
    let second_expr = matched_rewrites.remove(0);
    let full_expr = build_signed_sum_expr(ctx, terms);

    let mut rewrite = Rewrite::with_local(zero, "Phase Shift Identity", full_expr, zero);
    rewrite.substeps = vec![
        build_phase_shift_zero_substep(ctx, first_expr),
        build_phase_shift_zero_substep(ctx, second_expr),
    ];
    Some(rewrite)
}

pub(super) fn build_phase_shift_zero_substep(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> crate::step::SubStep {
    crate::step::SubStep::new(
        "Phase Shift Identity",
        vec![format!(
            "{} -> 0",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: expr
            }
        )],
    )
}

fn try_build_fast_structural_exact_phase_shift_triple_zero_rewrite(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> Option<Rewrite> {
    if terms.len() != 3 {
        return None;
    }

    let full_expr = build_signed_sum_expr(ctx, terms);
    let zero = ctx.num(0);
    let linear_terms: Vec<_> = terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| {
            extract_signed_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, term_expr, term_sign)
        })
        .collect();

    for shifted_index in 0..terms.len() {
        let signed_shifted = apply_sign_to_expr(
            ctx,
            sign_to_i64(terms[shifted_index].1),
            terms[shifted_index].0,
        );
        let Some((base_arg, coeff, kind, sin_sign, cos_sign)) =
            extract_structural_exact_phase_shift_term_data_for_cancellation(ctx, signed_shifted)
        else {
            continue;
        };

        let other_indices: Vec<_> = (0..terms.len())
            .filter(|index| *index != shifted_index)
            .collect();
        if other_indices.len() != 2 {
            continue;
        }

        let mut sin_term = None;
        let mut cos_term = None;
        for index in other_indices {
            let Some((trig_fn, raw_arg, coeff, signed_coeff_sign)) = linear_terms[index] else {
                sin_term = None;
                cos_term = None;
                break;
            };
            match trig_fn {
                BuiltinFn::Sin if sin_term.is_none() => {
                    sin_term = Some((raw_arg, coeff, signed_coeff_sign))
                }
                BuiltinFn::Cos if cos_term.is_none() => {
                    cos_term = Some((raw_arg, coeff, signed_coeff_sign))
                }
                _ => {
                    sin_term = None;
                    cos_term = None;
                    break;
                }
            }
        }

        let (sin_term, cos_term) = (sin_term?, cos_term?);
        if !matches_exact_phase_shift_linear_combination_target_from_extracted(
            ctx,
            sin_term,
            cos_term,
            base_arg,
            coeff,
            kind,
            (sin_sign, cos_sign),
        ) {
            continue;
        }

        let mut rewrite = Rewrite::with_local(zero, "Phase Shift Identity", full_expr, zero);
        rewrite.substeps = vec![build_phase_shift_zero_substep(ctx, full_expr)];
        return Some(rewrite);
    }

    None
}

fn try_build_fast_general_phase_shift_triple_zero_rewrite(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> Option<Rewrite> {
    if terms.len() != 3 {
        return None;
    }

    for shifted_index in 0..terms.len() {
        let signed_shifted = apply_sign_to_expr(
            ctx,
            sign_to_i64(terms[shifted_index].1),
            terms[shifted_index].0,
        );
        let Some(target_data) =
            extract_general_phase_shift_term_data_for_cancellation(ctx, signed_shifted)
        else {
            continue;
        };

        let linear_terms: Vec<_> = terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != shifted_index).then_some(term))
            .collect();
        if linear_terms.len() != 2 {
            continue;
        }

        let Some(linear_signature) =
            extract_weighted_phase_shift_linear_combination_terms_for_cancellation(
                ctx,
                &linear_terms,
            )
        else {
            continue;
        };

        if !matches_general_phase_shift_shifted_term_candidate_for_cancellation(
            ctx,
            target_data,
            linear_signature,
            true,
        ) {
            continue;
        }

        let linear_expr = build_signed_sum_expr(ctx, &linear_terms);
        let shifted_after = apply_sign_to_expr(
            ctx,
            -sign_to_i64(terms[shifted_index].1),
            terms[shifted_index].0,
        );

        return Some(build_trig_phase_shift_zero_rewrite(
            ctx,
            TrigPhaseShiftCancellationMatch {
                local_before: linear_expr,
                local_after: shifted_after,
                mode: TrigPhaseShiftCancellationMode::LinearToShifted,
            },
        ));
    }

    None
}

pub(super) fn try_build_repeated_trig_phase_shift_pair_with_canceling_passthrough_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let normalized_expr = normalize_additive_scope_expr(ctx, expr);
    let view = AddView::from_expr(ctx, normalized_expr);
    if view.terms.len() != 8 {
        return None;
    }

    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();

    for first_index in 0..normalized_terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..normalized_terms.len() {
            let (first_expr, first_sign) = normalized_terms[first_index];
            let (second_expr, second_sign) = normalized_terms[second_index];
            if first_sign == second_sign
                || compare_expr(ctx, first_expr, second_expr) != Ordering::Equal
            {
                continue;
            }

            let remaining_terms: Vec<_> = normalized_terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if remaining_terms.len() != 6 {
                continue;
            }

            let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
            if let Some(rewrite) =
                try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, remaining_expr)
            {
                return Some(rewrite);
            }
        }
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
