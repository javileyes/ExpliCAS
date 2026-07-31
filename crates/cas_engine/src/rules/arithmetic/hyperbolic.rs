//! `arithmetic`: familia `hyperbolic`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

fn is_direct_trig_or_hyperbolic_call(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    matches!(
        ctx.get(expr),
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Sin
                            | BuiltinFn::Cos
                            | BuiltinFn::Sinh
                            | BuiltinFn::Cosh
                    )
                )
    )
}

fn term_has_variable_scaled_direct_trig_or_hyperbolic_factor(
    ctx: &cas_ast::Context,
    term: cas_ast::ExprId,
) -> bool {
    let factors = MulView::from_expr(ctx, term).factors;
    if factors.len() < 2
        || !factors
            .iter()
            .any(|factor| is_direct_trig_or_hyperbolic_call(ctx, *factor))
    {
        return false;
    }

    factors.iter().any(|factor| {
        !is_direct_trig_or_hyperbolic_call(ctx, *factor)
            && expr_contains_symbolic_atom_for_cancellation(ctx, *factor)
    })
}

pub(super) fn additive_has_variable_scaled_direct_trig_or_hyperbolic_term(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    min_terms: usize,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    (min_terms..=6).contains(&view.terms.len())
        && view
            .terms
            .iter()
            .any(|(term, _)| term_has_variable_scaled_direct_trig_or_hyperbolic_factor(ctx, *term))
}

pub(super) fn product_has_variable_scaled_direct_trig_or_hyperbolic_additive_factor(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    MulView::from_expr(ctx, expr).factors.iter().any(|factor| {
        let factor = cas_ast::hold::unwrap_hold(ctx, *factor);
        if same_arg_sin_cos_additive_pair(ctx, factor) {
            return true;
        }
        matches!(
            ctx.get(factor),
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
        ) && additive_has_variable_scaled_direct_trig_or_hyperbolic_term(ctx, factor, 2)
    })
}

pub(super) fn maybe_hyperbolic_angle_sum_diff_zero_candidate(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if (ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cosh))
                    && args.len() == 1 =>
            {
                if matches!(ctx.get(args[0]), Expr::Add(_, _) | Expr::Sub(_, _)) {
                    return true;
                }
                stack.push(args[0]);
            }
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
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn extract_hyperbolic_power_shape(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    builtin: BuiltinFn,
) -> Option<(cas_ast::ExprId, i64)> {
    let positive_expr = match ctx.get(expr) {
        Expr::Neg(inner) => (-1, *inner),
        _ => (1, expr),
    }
    .1;

    let mut arg = None;
    let mut power = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, builtin) && args.len() == 1 => {
                if arg.is_some() {
                    return None;
                }
                arg = Some(args[0]);
                power = Some(1);
            }
            Expr::Pow(base, exponent) => {
                let Expr::Function(fn_id, args) = ctx.get(*base) else {
                    continue;
                };
                if !ctx.is_builtin(*fn_id, builtin) || args.len() != 1 {
                    continue;
                }
                if arg.is_some() {
                    return None;
                }
                arg = Some(args[0]);
                power = Some(extract_i64_integer(ctx, *exponent)?);
            }
            _ => {}
        }
    }

    Some((arg?, power?))
}

fn extract_cosh_power_shape(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, i64)> {
    extract_hyperbolic_power_shape(ctx, expr, BuiltinFn::Cosh)
}

fn extract_sinh_power_shape(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, i64)> {
    extract_hyperbolic_power_shape(ctx, expr, BuiltinFn::Sinh)
}

fn extract_signed_hyperbolic_power(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    builtin: BuiltinFn,
) -> Option<(i64, cas_ast::ExprId, cas_ast::ExprId, i64)> {
    let (sign, positive_expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (-1, *inner),
        _ => (1, expr),
    };

    let mut coefficient_factors = smallvec::SmallVec::<[cas_ast::ExprId; 8]>::new();
    let mut arg = None;
    let mut power = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, builtin) && args.len() == 1 => {
                if arg.is_some() {
                    return None;
                }
                arg = Some(args[0]);
                power = Some(1);
            }
            Expr::Pow(base, exponent) => {
                let Expr::Function(fn_id, args) = ctx.get(*base) else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if !ctx.is_builtin(*fn_id, builtin) || args.len() != 1 {
                    coefficient_factors.push(factor);
                    continue;
                }
                if arg.is_some() {
                    return None;
                }
                arg = Some(args[0]);
                power = Some(extract_i64_integer(ctx, *exponent)?);
            }
            _ => coefficient_factors.push(factor),
        }
    }

    let coefficient = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        let coefficient_vec = coefficient_factors.into_vec();
        build_balanced_mul(ctx, &coefficient_vec)
    };

    Some((sign, coefficient, arg?, power?))
}

fn extract_signed_cosh_power(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(i64, cas_ast::ExprId, cas_ast::ExprId, i64)> {
    extract_signed_hyperbolic_power(ctx, expr, BuiltinFn::Cosh)
}

pub(super) fn maybe_hyperbolic_pythagorean_factor_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    let mut seen_linear = Vec::new();
    let mut seen_cubic = Vec::new();

    for (term_expr, _sign) in view.terms {
        if let Some((arg, power)) = extract_cosh_power_shape(ctx, term_expr)
            .or_else(|| extract_sinh_power_shape(ctx, term_expr))
        {
            match power {
                1 => seen_linear.push(arg),
                3 => seen_cubic.push(arg),
                _ => {}
            }
        }
    }

    seen_linear.iter().any(|linear_arg| {
        seen_cubic
            .iter()
            .any(|cubic_arg| compare_expr(ctx, *linear_arg, *cubic_arg) == Ordering::Equal)
    })
}

pub(super) fn maybe_two_term_hyperbolic_direct_core_equivalence_candidate(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    let lhs_direct_hyperbolic = expr_contains_direct_hyperbolic_builtin(ctx, lhs_core);
    let rhs_direct_hyperbolic = expr_contains_direct_hyperbolic_builtin(ctx, rhs_core);

    (lhs_direct_hyperbolic && (rhs_direct_hyperbolic || contains_division_like_term(ctx, rhs_core)))
        || (rhs_direct_hyperbolic
            && (lhs_direct_hyperbolic || contains_division_like_term(ctx, lhs_core)))
}

fn is_plain_two_factor_direct_hyperbolic_product(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    let factors = flatten_mul_chain(ctx, expr);
    factors.len() == 2
        && factors
            .iter()
            .copied()
            .all(|factor| extract_hyperbolic_linear_term_for_profile(ctx, factor).is_some())
}

pub(super) fn maybe_two_term_hyperbolic_direct_identity_candidate(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    maybe_two_term_hyperbolic_direct_core_equivalence_candidate(ctx, lhs_core, rhs_core)
        && (!is_plain_two_factor_direct_hyperbolic_product(ctx, lhs_core)
            || !is_plain_two_factor_direct_hyperbolic_product(ctx, rhs_core)
            || contains_division_like_term(ctx, lhs_core)
            || contains_division_like_term(ctx, rhs_core))
}

pub(super) fn maybe_two_term_tanh_exp_equivalence_candidate(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    let lhs_has_tanh = expr_contains_any_builtin(ctx, lhs_core, &[BuiltinFn::Tanh]);
    let rhs_has_tanh = expr_contains_any_builtin(ctx, rhs_core, &[BuiltinFn::Tanh]);
    let lhs_has_exp = expr_contains_any_builtin(ctx, lhs_core, &[BuiltinFn::Exp]);
    let rhs_has_exp = expr_contains_any_builtin(ctx, rhs_core, &[BuiltinFn::Exp]);
    let lhs_division_like = contains_division_like_term(ctx, lhs_core);
    let rhs_division_like = contains_division_like_term(ctx, rhs_core);

    (lhs_has_tanh && (rhs_has_exp || rhs_division_like))
        || (rhs_has_tanh && (lhs_has_exp || lhs_division_like))
}

pub(super) fn expr_contains_direct_hyperbolic_builtin(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_any_builtin(
        ctx,
        expr,
        &[BuiltinFn::Sinh, BuiltinFn::Cosh, BuiltinFn::Tanh],
    )
}

pub(super) fn try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some((hyperbolic_fn, multiple, base)) =
        extract_hyperbolic_linear_multiple_term_for_fast_recursive_identity(ctx, expr)
    {
        let left_multiple = multiple.checked_sub(1)?;
        let left_arg = if left_multiple == 1 {
            base
        } else {
            let coeff = ctx.num(left_multiple);
            smart_mul(ctx, coeff, base)
        };
        let right_arg = base;

        let rewritten = match hyperbolic_fn {
            BuiltinFn::Sinh => {
                let left_sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![left_arg]);
                let left_cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![left_arg]);
                let right_sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![right_arg]);
                let right_cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![right_arg]);
                let first = smart_mul(ctx, left_sinh, right_cosh);
                let second = smart_mul(ctx, left_cosh, right_sinh);
                let combined = ctx.add(Expr::Add(first, second));
                run_default_simplify(ctx, combined)
            }
            BuiltinFn::Cosh => {
                let left_cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![left_arg]);
                let left_sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![left_arg]);
                let right_cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![right_arg]);
                let right_sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![right_arg]);
                let first = smart_mul(ctx, left_cosh, right_cosh);
                let second = smart_mul(ctx, left_sinh, right_sinh);
                let combined = ctx.add(Expr::Add(first, second));
                run_default_simplify(ctx, combined)
            }
            _ => unreachable!("only sinh/cosh should reach recursive hyperbolic expansion"),
        };

        if rewritten != expr {
            return Some(rewritten);
        }
    }

    try_expand_direct_hyperbolic_angle_sum_diff_for_cancellation(ctx, expr)
}

fn is_direct_hyperbolic_angle_sum_diff_call(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    if args.len() != 1
        || !(ctx.is_builtin(*fn_id, BuiltinFn::Sinh) || ctx.is_builtin(*fn_id, BuiltinFn::Cosh))
    {
        return false;
    }

    matches!(ctx.get(args[0]), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn try_expand_direct_hyperbolic_angle_sum_diff_call(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if !is_direct_hyperbolic_angle_sum_diff_call(ctx, expr) {
        return None;
    }

    let rewritten = cas_math::expand_ops::expand(ctx, expr);
    (rewritten != expr).then_some(rewritten)
}

fn try_expand_direct_hyperbolic_angle_sum_diff_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewritten) = try_expand_direct_hyperbolic_angle_sum_diff_call(ctx, expr) {
        return Some(rewritten);
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let rewritten = try_expand_direct_hyperbolic_angle_sum_diff_call(ctx, inner)?;
            Some(ctx.add(Expr::Neg(rewritten)))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(rewritten_lhs) = try_expand_direct_hyperbolic_angle_sum_diff_call(ctx, lhs)
            {
                return Some(smart_mul(ctx, rewritten_lhs, rhs));
            }
            if let Some(rewritten_rhs) = try_expand_direct_hyperbolic_angle_sum_diff_call(ctx, rhs)
            {
                return Some(smart_mul(ctx, lhs, rewritten_rhs));
            }
            None
        }
        Expr::Div(num, den) => {
            let rewritten_num = try_expand_direct_hyperbolic_angle_sum_diff_call(ctx, num)?;
            Some(ctx.add(Expr::Div(rewritten_num, den)))
        }
        _ => None,
    }
}

fn extract_hyperbolic_linear_multiple_term_for_fast_recursive_identity(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, i64, cas_ast::ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let hyperbolic_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
        BuiltinFn::Sinh
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
        BuiltinFn::Cosh
    } else {
        return None;
    };

    let (coeff, base) = split_linear_angle_term_for_phase_shift_cancellation(ctx, args[0]);
    if !coeff.is_integer() {
        return None;
    }

    let multiple = coeff.to_integer().to_i64()?;
    (multiple >= 2).then_some((hyperbolic_fn, multiple, base))
}

fn hyperbolic_call_arg_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
        Some((BuiltinFn::Sinh, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
        Some((BuiltinFn::Cosh, args[0]))
    } else {
        None
    }
}

fn build_canonical_hyperbolic_call_for_cancellation(
    ctx: &mut cas_ast::Context,
    builtin: BuiltinFn,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    match builtin {
        BuiltinFn::Cosh => {
            let arg = strip_unit_negation_for_phase_shift(ctx, arg).unwrap_or(arg);
            ctx.call_builtin(BuiltinFn::Cosh, vec![arg])
        }
        BuiltinFn::Sinh => {
            if let Some(arg) = strip_unit_negation_for_phase_shift(ctx, arg) {
                let positive = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                ctx.add(Expr::Neg(positive))
            } else {
                ctx.call_builtin(BuiltinFn::Sinh, vec![arg])
            }
        }
        _ => unreachable!("only sinh/cosh are supported here"),
    }
}

fn half_sum_for_hyperbolic_cancellation(
    ctx: &mut cas_ast::Context,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let two = ctx.num(2);
    let sum = ctx.add(Expr::Add(left, right));
    let average = ctx.add(Expr::Div(sum, two));
    run_default_simplify(ctx, average)
}

fn half_diff_for_hyperbolic_cancellation(
    ctx: &mut cas_ast::Context,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let two = ctx.num(2);
    let diff = ctx.add(Expr::Sub(left, right));
    let half_difference = ctx.add(Expr::Div(diff, two));
    run_default_simplify(ctx, half_difference)
}

fn extract_scaled_hyperbolic_two_factor_product_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    scale: i64,
) -> Option<ScaledHyperbolicProductPatternForCancellation> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut saw_scale = false;
    let mut hyperbolic_factors = Vec::new();

    for factor in factors {
        if !saw_scale && extract_i64_integer(ctx, factor) == Some(scale) {
            saw_scale = true;
            continue;
        }

        let (builtin, arg) = hyperbolic_call_arg_for_cancellation(ctx, factor)?;
        hyperbolic_factors.push((builtin, arg));
    }

    if !saw_scale || hyperbolic_factors.len() != 2 {
        return None;
    }

    match (hyperbolic_factors[0].0, hyperbolic_factors[1].0) {
        (BuiltinFn::Sinh, BuiltinFn::Cosh) => {
            Some(ScaledHyperbolicProductPatternForCancellation::SinhCosh(
                hyperbolic_factors[0].1,
                hyperbolic_factors[1].1,
            ))
        }
        (BuiltinFn::Cosh, BuiltinFn::Sinh) => {
            Some(ScaledHyperbolicProductPatternForCancellation::SinhCosh(
                hyperbolic_factors[1].1,
                hyperbolic_factors[0].1,
            ))
        }
        (BuiltinFn::Cosh, BuiltinFn::Cosh) => {
            Some(ScaledHyperbolicProductPatternForCancellation::CoshCosh(
                hyperbolic_factors[0].1,
                hyperbolic_factors[1].1,
            ))
        }
        (BuiltinFn::Sinh, BuiltinFn::Sinh) => {
            Some(ScaledHyperbolicProductPatternForCancellation::SinhSinh(
                hyperbolic_factors[0].1,
                hyperbolic_factors[1].1,
            ))
        }
        _ => None,
    }
}

fn try_rewrite_hyperbolic_sum_to_product_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let (first_fn, first_arg) = hyperbolic_call_arg_for_cancellation(ctx, terms[0].0)?;
    let (second_fn, second_arg) = hyperbolic_call_arg_for_cancellation(ctx, terms[1].0)?;
    if first_fn != second_fn {
        return None;
    }

    let two = ctx.num(2);
    let rewritten = match first_fn {
        BuiltinFn::Sinh => {
            if terms[0].1 == terms[1].1 {
                let avg = half_sum_for_hyperbolic_cancellation(ctx, first_arg, second_arg);
                let half_diff = half_diff_for_hyperbolic_cancellation(ctx, first_arg, second_arg);
                let avg_sinh =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Sinh, avg);
                let diff_cosh = build_canonical_hyperbolic_call_for_cancellation(
                    ctx,
                    BuiltinFn::Cosh,
                    half_diff,
                );
                let scaled_avg = smart_mul(ctx, two, avg_sinh);
                smart_mul(ctx, scaled_avg, diff_cosh)
            } else {
                let (positive_arg, negative_arg) = if terms[0].1 == Sign::Pos {
                    (first_arg, second_arg)
                } else {
                    (second_arg, first_arg)
                };
                let avg = half_sum_for_hyperbolic_cancellation(ctx, positive_arg, negative_arg);
                let half_diff =
                    half_diff_for_hyperbolic_cancellation(ctx, positive_arg, negative_arg);
                let avg_cosh =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Cosh, avg);
                let diff_sinh = build_canonical_hyperbolic_call_for_cancellation(
                    ctx,
                    BuiltinFn::Sinh,
                    half_diff,
                );
                let scaled_avg = smart_mul(ctx, two, avg_cosh);
                smart_mul(ctx, scaled_avg, diff_sinh)
            }
        }
        BuiltinFn::Cosh => {
            if terms[0].1 == terms[1].1 {
                let avg = half_sum_for_hyperbolic_cancellation(ctx, first_arg, second_arg);
                let half_diff = half_diff_for_hyperbolic_cancellation(ctx, first_arg, second_arg);
                let avg_cosh =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Cosh, avg);
                let diff_cosh = build_canonical_hyperbolic_call_for_cancellation(
                    ctx,
                    BuiltinFn::Cosh,
                    half_diff,
                );
                let scaled_avg = smart_mul(ctx, two, avg_cosh);
                smart_mul(ctx, scaled_avg, diff_cosh)
            } else {
                let (positive_arg, negative_arg) = if terms[0].1 == Sign::Pos {
                    (first_arg, second_arg)
                } else {
                    (second_arg, first_arg)
                };
                let avg = half_sum_for_hyperbolic_cancellation(ctx, positive_arg, negative_arg);
                let half_diff =
                    half_diff_for_hyperbolic_cancellation(ctx, positive_arg, negative_arg);
                let avg_sinh =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Sinh, avg);
                let diff_sinh = build_canonical_hyperbolic_call_for_cancellation(
                    ctx,
                    BuiltinFn::Sinh,
                    half_diff,
                );
                let scaled_avg = smart_mul(ctx, two, avg_sinh);
                smart_mul(ctx, scaled_avg, diff_sinh)
            }
        }
        _ => return None,
    };

    Some(run_default_simplify(ctx, rewritten))
}

fn try_rewrite_hyperbolic_product_to_sum_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let rewritten =
        match extract_scaled_hyperbolic_two_factor_product_for_cancellation(ctx, expr, 2)? {
            ScaledHyperbolicProductPatternForCancellation::SinhCosh(left, right) => {
                let sum_expr = ctx.add(Expr::Add(left, right));
                let sum = run_default_simplify(ctx, sum_expr);
                let diff_expr = ctx.add(Expr::Sub(left, right));
                let diff = run_default_simplify(ctx, diff_expr);
                let sum_term =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Sinh, sum);
                let diff_term =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Sinh, diff);
                ctx.add(Expr::Add(sum_term, diff_term))
            }
            ScaledHyperbolicProductPatternForCancellation::CoshCosh(left, right) => {
                let sum_expr = ctx.add(Expr::Add(left, right));
                let sum = run_default_simplify(ctx, sum_expr);
                let diff_expr = ctx.add(Expr::Sub(left, right));
                let diff = run_default_simplify(ctx, diff_expr);
                let sum_term =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Cosh, sum);
                let diff_term =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Cosh, diff);
                ctx.add(Expr::Add(sum_term, diff_term))
            }
            ScaledHyperbolicProductPatternForCancellation::SinhSinh(left, right) => {
                let sum_expr = ctx.add(Expr::Add(left, right));
                let sum = run_default_simplify(ctx, sum_expr);
                let diff_expr = ctx.add(Expr::Sub(left, right));
                let diff = run_default_simplify(ctx, diff_expr);
                let sum_term =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Cosh, sum);
                let diff_term =
                    build_canonical_hyperbolic_call_for_cancellation(ctx, BuiltinFn::Cosh, diff);
                ctx.add(Expr::Sub(sum_term, diff_term))
            }
        };

    Some(run_default_simplify(ctx, rewritten))
}

fn try_rewrite_hyperbolic_product_sum_triple_angle_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let ScaledHyperbolicProductPatternForCancellation::SinhCosh(sinh_arg, cosh_arg) =
        extract_scaled_hyperbolic_two_factor_product_for_cancellation(ctx, expr, 2)?
    else {
        return None;
    };

    let base_arg = if let Some(base) = extract_double_angle_arg_relaxed(ctx, sinh_arg) {
        (compare_expr(ctx, base, cosh_arg) == Ordering::Equal).then_some(base)
    } else if let Some(base) = extract_double_angle_arg_relaxed(ctx, cosh_arg) {
        (compare_expr(ctx, base, sinh_arg) == Ordering::Equal).then_some(base)
    } else {
        None
    }?;

    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, base_arg);
    let triple_term = ctx.call_builtin(BuiltinFn::Sinh, vec![triple_arg]);
    let triple_expanded = try_rewrite_hyperbolic_triple_angle(ctx, triple_term)?.rewritten;
    let linear_term = ctx.call_builtin(BuiltinFn::Sinh, vec![base_arg]);
    let combined = ctx.add(Expr::Add(triple_expanded, linear_term));
    let rewritten = run_default_simplify(ctx, combined);
    Some(rewritten)
}

fn try_rewrite_hyperbolic_product_sum_cosh_cubic_polynomial_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let ScaledHyperbolicProductPatternForCancellation::SinhSinh(left_arg, right_arg) =
        extract_scaled_hyperbolic_two_factor_product_for_cancellation(ctx, expr, 2)?
    else {
        return None;
    };

    let base_arg = if let Some(base) = extract_double_angle_arg_relaxed(ctx, left_arg) {
        (compare_expr(ctx, base, right_arg) == Ordering::Equal).then_some(base)
    } else if let Some(base) = extract_double_angle_arg_relaxed(ctx, right_arg) {
        (compare_expr(ctx, base, left_arg) == Ordering::Equal).then_some(base)
    } else {
        None
    }?;

    let cosh_base = ctx.call_builtin(BuiltinFn::Cosh, vec![base_arg]);
    let three = ctx.num(3);
    let cosh_cube = ctx.add(Expr::Pow(cosh_base, three));
    let four = ctx.num(4);
    let four_cosh_cube = smart_mul(ctx, four, cosh_cube);
    let four_cosh = smart_mul(ctx, four, cosh_base);
    Some(ctx.add(Expr::Sub(four_cosh_cube, four_cosh)))
}

pub(super) fn try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let ScaledHyperbolicProductPatternForCancellation::SinhCosh(left_arg, right_arg) =
        extract_scaled_hyperbolic_two_factor_product_for_cancellation(ctx, expr, 2)?
    else {
        return None;
    };

    let base_arg = if let Some(base) = extract_double_angle_arg_relaxed(ctx, left_arg) {
        (compare_expr(ctx, base, right_arg) == Ordering::Equal).then_some(base)
    } else if let Some(base) = extract_double_angle_arg_relaxed(ctx, right_arg) {
        (compare_expr(ctx, base, left_arg) == Ordering::Equal).then_some(base)
    } else {
        None
    }?;

    let sinh_base = ctx.call_builtin(BuiltinFn::Sinh, vec![base_arg]);
    let three = ctx.num(3);
    let sinh_cube = ctx.add(Expr::Pow(sinh_base, three));
    let four = ctx.num(4);
    let four_sinh_cube = smart_mul(ctx, four, sinh_cube);
    let four_sinh = smart_mul(ctx, four, sinh_base);
    let rewritten = ctx.add(Expr::Add(four_sinh, four_sinh_cube));
    Some(run_default_simplify(ctx, rewritten))
}

pub(super) fn extract_squared_hyperbolic_arg(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    builtin: BuiltinFn,
) -> Option<cas_ast::ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exp) != Some(2) {
        return None;
    }
    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    (ctx.is_builtin(*fn_id, builtin) && args.len() == 1).then_some(args[0])
}

fn build_hyperbolic_double_angle_expansion(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let two = ctx.num(2);
    let one = ctx.num(1);
    let cosh_sq = ctx.add(Expr::Pow(cosh, two));
    let scaled = smart_mul(ctx, two, cosh_sq);
    ctx.add(Expr::Sub(scaled, one))
}

fn try_rewrite_hyperbolic_pythagorean_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewrite) = try_rewrite_hyperbolic_pythagorean_sub_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cosh_pos_arg = None;
    let mut cosh_neg_arg = None;
    let mut sinh_pos_arg = None;
    let mut sinh_neg_arg = None;
    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_squared_hyperbolic_arg(ctx, term_expr, BuiltinFn::Cosh) {
            match term_sign {
                Sign::Pos => cosh_pos_arg = Some(arg),
                Sign::Neg => cosh_neg_arg = Some(arg),
            }
            continue;
        }
        if let Some(arg) = extract_squared_hyperbolic_arg(ctx, term_expr, BuiltinFn::Sinh) {
            match term_sign {
                Sign::Pos => sinh_pos_arg = Some(arg),
                Sign::Neg => sinh_neg_arg = Some(arg),
            }
            continue;
        }
        return None;
    }

    if let (Some(cosh_arg), Some(sinh_arg)) = (cosh_pos_arg, sinh_neg_arg) {
        if compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal {
            return Some(ctx.num(1));
        }
    }
    if let (Some(sinh_arg), Some(cosh_arg)) = (sinh_pos_arg, cosh_neg_arg) {
        if compare_expr(ctx, sinh_arg, cosh_arg) == Ordering::Equal {
            return Some(ctx.num(-1));
        }
    }

    None
}

fn try_rewrite_hyperbolic_double_angle_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 {
            let arg = extract_relaxed_double_angle_arg(ctx, args[0])?;
            return Some(build_hyperbolic_double_angle_expansion(ctx, arg));
        }
    }

    if let Expr::Sub(lhs, rhs) = ctx.get(expr).clone() {
        if is_positive_one_expr(ctx, rhs) {
            if let Expr::Mul(left, right) = ctx.get(lhs).clone() {
                let (scale, candidate) = if extract_i64_integer(ctx, left) == Some(2) {
                    (left, right)
                } else if extract_i64_integer(ctx, right) == Some(2) {
                    (right, left)
                } else {
                    (ctx.num(0), ctx.num(0))
                };
                if scale != ctx.num(0) {
                    let arg = extract_squared_hyperbolic_arg(ctx, candidate, BuiltinFn::Cosh)?;
                    let two = ctx.num(2);
                    let two_arg = smart_mul(ctx, two, arg);
                    return Some(ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]));
                }
            }
        }
    }

    if let Expr::Add(lhs, rhs) = ctx.get(expr).clone() {
        let (candidate, _one_term) = if is_positive_one_expr(ctx, lhs) {
            (rhs, lhs)
        } else if is_positive_one_expr(ctx, rhs) {
            (lhs, rhs)
        } else {
            return None;
        };

        if let Expr::Mul(left, right) = ctx.get(candidate).clone() {
            let scaled_term = if extract_i64_integer(ctx, left) == Some(2) {
                right
            } else if extract_i64_integer(ctx, right) == Some(2) {
                left
            } else {
                return None;
            };
            let arg = extract_squared_hyperbolic_arg(ctx, scaled_term, BuiltinFn::Sinh)?;
            let two = ctx.num(2);
            let two_arg = smart_mul(ctx, two, arg);
            return Some(ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]));
        }
    }

    None
}

fn try_rewrite_hyperbolic_exp_equivalence_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_sinh_cosh_to_exp(ctx, expr)
    {
        return Some(rewrite.rewritten);
    }

    let arg = extract_exp_argument(ctx, expr)?;
    let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    Some(ctx.add(Expr::Add(sinh, cosh)))
}

fn extract_scaled_atanh_arg_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BigRational, cas_ast::ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Atanh) && args.len() == 1 =>
        {
            Some((BigRational::from_integer(1.into()), args[0]))
        }
        Expr::Neg(inner) => {
            let (scale, arg) = extract_scaled_atanh_arg_for_cancellation(ctx, *inner)?;
            Some((-scale, arg))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(n) = extract_literal_rational_for_cancellation(ctx, *lhs) {
                let arg = extract_unary_builtin_arg(ctx, *rhs, BuiltinFn::Atanh)?;
                return Some((n, arg));
            }
            if let Some(n) = extract_literal_rational_for_cancellation(ctx, *rhs) {
                let arg = extract_unary_builtin_arg(ctx, *lhs, BuiltinFn::Atanh)?;
                return Some((n, arg));
            }
            None
        }
        _ => None,
    }
}

fn extract_common_log_atanh_definition_arg(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };

    let numerator_arg = match ctx.get(*numerator) {
        Expr::Add(lhs, rhs) => {
            if matches!(ctx.get(*lhs), Expr::Number(n) if n.is_one()) {
                Some(*rhs)
            } else if matches!(ctx.get(*rhs), Expr::Number(n) if n.is_one()) {
                Some(*lhs)
            } else {
                None
            }
        }
        _ => None,
    }?;

    let denominator_arg = match ctx.get(*denominator) {
        Expr::Sub(lhs, rhs) if matches!(ctx.get(*lhs), Expr::Number(n) if n.is_one()) => Some(*rhs),
        Expr::Add(lhs, rhs) if matches!(ctx.get(*lhs), Expr::Number(n) if n.is_one()) => {
            match ctx.get(*rhs) {
                Expr::Neg(inner) => Some(*inner),
                _ => None,
            }
        }
        Expr::Add(lhs, rhs) if matches!(ctx.get(*rhs), Expr::Number(n) if n.is_one()) => {
            match ctx.get(*lhs) {
                Expr::Neg(inner) => Some(*inner),
                _ => None,
            }
        }
        _ => None,
    }?;

    (compare_expr(ctx, numerator_arg, denominator_arg) == Ordering::Equal).then_some(numerator_arg)
}

fn extract_atanh_square_ratio_log_arg_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, bool)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };

    let (den_base, den_square_sign, den_one_sign) =
        extract_square_plus_minus_one_pattern_for_cancellation(ctx, denominator)?;
    if den_square_sign != Sign::Pos || den_one_sign != Sign::Pos {
        return None;
    }

    let (num_base, num_square_sign, num_one_sign) =
        extract_square_plus_minus_one_pattern_for_cancellation(ctx, numerator)?;
    if !exprs_match_for_cancellation(ctx, den_base, num_base) {
        return None;
    }

    match (num_square_sign, num_one_sign) {
        (Sign::Pos, Sign::Neg) => Some((den_base, true)),
        (Sign::Neg, Sign::Pos) => Some((den_base, false)),
        _ => None,
    }
}

fn extract_scaled_common_log_atanh_definition_arg_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BigRational, cas_ast::ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Log10) && args.len() == 1 =>
        {
            extract_common_log_atanh_definition_arg(ctx, args[0])
                .map(|arg| (BigRational::from_integer(1.into()), arg))
        }
        Expr::Neg(inner) => {
            let (scale, arg) =
                extract_scaled_common_log_atanh_definition_arg_for_cancellation(ctx, *inner)?;
            Some((-scale, arg))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(n) = extract_literal_rational_for_cancellation(ctx, *lhs) {
                let arg =
                    extract_scaled_common_log_atanh_definition_arg_for_cancellation(ctx, *rhs)?;
                return Some((n * arg.0, arg.1));
            }
            if let Some(n) = extract_literal_rational_for_cancellation(ctx, *rhs) {
                let arg =
                    extract_scaled_common_log_atanh_definition_arg_for_cancellation(ctx, *lhs)?;
                return Some((n * arg.0, arg.1));
            }
            None
        }
        _ => None,
    }
}

pub(super) fn is_atanh_common_log_definition_mismatch_pair(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    for (atanh_side, log_side) in [(lhs, rhs), (rhs, lhs)] {
        let Some((atanh_scale, atanh_arg)) =
            extract_scaled_atanh_arg_for_cancellation(ctx, atanh_side)
        else {
            continue;
        };
        let Some((log_scale, log_arg)) =
            extract_scaled_common_log_atanh_definition_arg_for_cancellation(ctx, log_side)
        else {
            continue;
        };

        if compare_expr(ctx, atanh_arg, log_arg) == Ordering::Equal
            && log_scale * BigRational::from_integer(2.into()) == atanh_scale
        {
            return true;
        }
    }

    false
}

fn has_atanh_common_log_mismatch_with_plain_passthrough_terms(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> bool {
    if terms.len() != 3 {
        return false;
    }

    let mut log_term = None;
    let mut atanh_term = None;

    for (term_expr, term_sign) in terms.iter().copied() {
        let has_log = expr_contains_any_builtin(ctx, term_expr, &[BuiltinFn::Log10]);
        let has_atanh = expr_contains_any_builtin(ctx, term_expr, &[BuiltinFn::Atanh]);
        let has_ln_or_abs = expr_contains_any_builtin(
            ctx,
            term_expr,
            &[
                BuiltinFn::Ln,
                BuiltinFn::Log,
                BuiltinFn::Log2,
                BuiltinFn::Log10,
                BuiltinFn::Abs,
            ],
        );

        if has_log && !has_atanh && !has_ln_or_abs {
            let Some((scale, arg)) =
                extract_scaled_common_log_atanh_definition_arg_for_cancellation(ctx, term_expr)
            else {
                return false;
            };
            if log_term
                .replace((scale_with_add_sign(scale, term_sign), arg))
                .is_some()
            {
                return false;
            }
            continue;
        }

        if has_atanh && !has_log && !has_ln_or_abs {
            let Some((scale, arg)) = extract_scaled_atanh_arg_for_cancellation(ctx, term_expr)
            else {
                return false;
            };
            if atanh_term
                .replace((scale_with_add_sign(scale, term_sign), arg))
                .is_some()
            {
                return false;
            }
            continue;
        }

        if has_log || has_atanh || has_ln_or_abs {
            return false;
        }
    }

    match (log_term, atanh_term) {
        (Some((log_scale, log_arg)), Some((atanh_scale, atanh_arg))) => {
            compare_expr(ctx, log_arg, atanh_arg) == Ordering::Equal
                && log_scale.abs() * BigRational::from_integer(2.into()) == atanh_scale.abs()
        }
        _ => false,
    }
}

pub(super) fn has_atanh_common_log_mismatch_with_plain_passthrough(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();

    has_atanh_common_log_mismatch_with_plain_passthrough_terms(ctx, &normalized_terms)
}

fn try_rewrite_atanh_ln_definition_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (scale, arg) = extract_scaled_atanh_arg_for_cancellation(ctx, expr)?;

    let one = ctx.num(1);
    let numerator = ctx.add(Expr::Add(one, arg));
    let denominator = ctx.add(Expr::Sub(one, arg));
    let quotient = ctx.add(Expr::Div(numerator, denominator));
    let ln_expr = ctx.call_builtin(BuiltinFn::Ln, vec![quotient]);

    let scaled_coeff = scale / BigRational::from_integer(2.into());
    if scaled_coeff == BigRational::from_integer(1.into()) {
        return Some(ln_expr);
    }
    if scaled_coeff == BigRational::from_integer((-1).into()) {
        return Some(ctx.add(Expr::Neg(ln_expr)));
    }

    let coeff_expr = ctx.add(Expr::Number(scaled_coeff));
    Some(smart_mul(ctx, coeff_expr, ln_expr))
}

fn try_rewrite_atanh_square_ratio_log_equivalence_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (scale, arg) = extract_scaled_atanh_arg_for_cancellation(ctx, expr)?;
    let (log_arg, positive_orientation) =
        extract_atanh_square_ratio_log_arg_for_cancellation(ctx, arg)?;
    let ln_expr = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);
    let oriented = if positive_orientation {
        ln_expr
    } else {
        ctx.add(Expr::Neg(ln_expr))
    };

    if scale == BigRational::from_integer(1.into()) {
        return Some(oriented);
    }
    if scale == BigRational::from_integer((-1).into()) {
        return Some(ctx.add(Expr::Neg(oriented)));
    }

    let coeff_expr = ctx.add(Expr::Number(scale));
    Some(smart_mul(ctx, coeff_expr, oriented))
}

pub(super) fn try_rewrite_exact_hyperbolic_equivalence_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, &'static str)> {
    if let Some(rewrite) = try_rewrite_hyperbolic_half_angle_squares_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Hyperbolic Half-Angle Squares"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_pythagorean_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Pythagorean Identity"));
    }

    if let Some(rewritten) = try_rewrite_sinh_cosh_exp_definition_for_cancellation(ctx, expr) {
        return Some((rewritten, "Recognize Hyperbolic from Exponential"));
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_triple_angle(ctx, expr)
    {
        return Some((rewrite.rewritten, "Hyperbolic Triple-Angle Identity"));
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_double_angle_sum(ctx, expr)
    {
        return Some((rewrite.rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_sinh_double_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_tanh_double_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_tanh_exp_definition_for_cancellation(ctx, expr) {
        return Some((rewritten, "Recognize Hyperbolic from Exponential"));
    }

    if let Some(rewritten) = try_rewrite_tanh_angle_sum_diff_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Angle Sum/Difference Identity"));
    }

    if let Some(rewritten) = try_rewrite_tanh_triple_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Triple-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_sum_to_product_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Product-to-Sum Identity"));
    }

    if let Some(rewritten) =
        try_rewrite_hyperbolic_product_sum_cosh_cubic_polynomial_for_cancellation(ctx, expr)
    {
        return Some((
            rewritten,
            "Hyperbolic Product-to-Sum and Triple-Angle Identity",
        ));
    }

    if let Some(rewritten) =
        try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(ctx, expr)
    {
        return Some((
            rewritten,
            "Hyperbolic Product-to-Sum and Triple-Angle Identity",
        ));
    }

    if let Some(rewritten) =
        try_rewrite_hyperbolic_product_sum_triple_angle_for_cancellation(ctx, expr)
    {
        return Some((
            rewritten,
            "Hyperbolic Product-to-Sum and Triple-Angle Identity",
        ));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_product_to_sum_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Product-to-Sum Identity"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Angle Sum/Difference Identity"));
    }

    if let Some(rewritten) = try_rewrite_shifted_hyperbolic_pythagorean_for_cancellation(ctx, expr)
    {
        return Some((rewritten, "Hyperbolic Pythagorean Identity"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_double_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_exp_equivalence_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Sum to Exponential"));
    }

    None
}

fn tanh_call_arg_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }
    Some(args[0])
}

fn tanh_product_matches_pair_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    left_arg: cas_ast::ExprId,
    right_arg: cas_ast::ExprId,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return false;
    };
    let Some(left_tanh_arg) = tanh_call_arg_for_cancellation(ctx, *left) else {
        return false;
    };
    let Some(right_tanh_arg) = tanh_call_arg_for_cancellation(ctx, *right) else {
        return false;
    };

    (compare_expr(ctx, left_tanh_arg, left_arg) == Ordering::Equal
        && compare_expr(ctx, right_tanh_arg, right_arg) == Ordering::Equal)
        || (compare_expr(ctx, left_tanh_arg, right_arg) == Ordering::Equal
            && compare_expr(ctx, right_tanh_arg, left_arg) == Ordering::Equal)
}

fn try_rewrite_tanh_angle_sum_diff_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Tanh) && args.len() == 1 {
            let (is_sum, left, right) = match ctx.get(args[0]) {
                Expr::Add(left, right) => (true, *left, *right),
                Expr::Sub(left, right) => (false, *left, *right),
                _ => return None,
            };

            let tanh_left = ctx.call_builtin(BuiltinFn::Tanh, vec![left]);
            let tanh_right = ctx.call_builtin(BuiltinFn::Tanh, vec![right]);
            let numerator = if is_sum {
                ctx.add(Expr::Add(tanh_left, tanh_right))
            } else {
                ctx.add(Expr::Sub(tanh_left, tanh_right))
            };
            let product = ctx.add(Expr::Mul(tanh_left, tanh_right));
            let one = ctx.num(1);
            let denominator = if is_sum {
                ctx.add(Expr::Add(one, product))
            } else {
                ctx.add(Expr::Sub(one, product))
            };
            return Some(ctx.add(Expr::Div(numerator, denominator)));
        }
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };

    let (is_sum, left_num, right_num) = match ctx.get(*numerator) {
        Expr::Add(left, right) => (true, *left, *right),
        Expr::Sub(left, right) => (false, *left, *right),
        _ => return None,
    };

    let left_arg = tanh_call_arg_for_cancellation(ctx, left_num)?;
    let right_arg = tanh_call_arg_for_cancellation(ctx, right_num)?;

    let denominator_matches = match ctx.get(*denominator) {
        Expr::Add(left, right) if is_sum => {
            (cas_math::expr_predicates::is_one_expr(ctx, *left)
                && tanh_product_matches_pair_for_cancellation(ctx, *right, left_arg, right_arg))
                || (cas_math::expr_predicates::is_one_expr(ctx, *right)
                    && tanh_product_matches_pair_for_cancellation(ctx, *left, left_arg, right_arg))
        }
        Expr::Sub(left, right) if !is_sum => {
            cas_math::expr_predicates::is_one_expr(ctx, *left)
                && tanh_product_matches_pair_for_cancellation(ctx, *right, left_arg, right_arg)
        }
        _ => false,
    };

    if !denominator_matches {
        return None;
    }

    let angle = if is_sum {
        ctx.add(Expr::Add(left_arg, right_arg))
    } else {
        ctx.add(Expr::Sub(left_arg, right_arg))
    };
    Some(ctx.call_builtin(BuiltinFn::Tanh, vec![angle]))
}

fn extract_tanh_power_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    power: i64,
) -> Option<cas_ast::ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent) != Some(power) {
        return None;
    }
    tanh_call_arg_for_cancellation(ctx, *base)
}

fn extract_scaled_tanh_power_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    scale: i64,
    power: i64,
) -> Option<cas_ast::ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if extract_i64_integer(ctx, *left) == Some(scale) {
        return if power == 1 {
            tanh_call_arg_for_cancellation(ctx, *right)
        } else {
            extract_tanh_power_for_cancellation(ctx, *right, power)
        };
    }

    if extract_i64_integer(ctx, *right) == Some(scale) {
        return if power == 1 {
            tanh_call_arg_for_cancellation(ctx, *left)
        } else {
            extract_tanh_power_for_cancellation(ctx, *left, power)
        };
    }

    None
}

fn try_rewrite_tanh_triple_angle_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Tanh) && args.len() == 1 {
            let inner = extract_triple_angle_arg_relaxed(ctx, args[0])?;
            let tanh_inner = ctx.call_builtin(BuiltinFn::Tanh, vec![inner]);
            let two = ctx.num(2);
            let three = ctx.num(3);
            let tanh_sq = ctx.add(Expr::Pow(tanh_inner, two));
            let tanh_cube = ctx.add(Expr::Pow(tanh_inner, three));
            let three_tanh = ctx.add(Expr::Mul(three, tanh_inner));
            let numerator = ctx.add(Expr::Add(tanh_cube, three_tanh));
            let three_tanh_sq = ctx.add(Expr::Mul(three, tanh_sq));
            let one = ctx.num(1);
            let denominator = ctx.add(Expr::Add(three_tanh_sq, one));
            return Some(ctx.add(Expr::Div(numerator, denominator)));
        }
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };

    let Expr::Add(num_left, num_right) = ctx.get(*numerator) else {
        return None;
    };
    let inner = match (
        extract_tanh_power_for_cancellation(ctx, *num_left, 3),
        extract_scaled_tanh_power_for_cancellation(ctx, *num_right, 3, 1),
    ) {
        (Some(left_inner), Some(right_inner))
            if compare_expr(ctx, left_inner, right_inner) == Ordering::Equal =>
        {
            left_inner
        }
        _ => match (
            extract_tanh_power_for_cancellation(ctx, *num_right, 3),
            extract_scaled_tanh_power_for_cancellation(ctx, *num_left, 3, 1),
        ) {
            (Some(left_inner), Some(right_inner))
                if compare_expr(ctx, left_inner, right_inner) == Ordering::Equal =>
            {
                left_inner
            }
            _ => return None,
        },
    };

    let Expr::Add(den_left, den_right) = ctx.get(*denominator) else {
        return None;
    };
    let denominator_matches = (cas_math::expr_predicates::is_one_expr(ctx, *den_left)
        && extract_scaled_tanh_power_for_cancellation(ctx, *den_right, 3, 2)
            .is_some_and(|den_inner| compare_expr(ctx, den_inner, inner) == Ordering::Equal))
        || (cas_math::expr_predicates::is_one_expr(ctx, *den_right)
            && extract_scaled_tanh_power_for_cancellation(ctx, *den_left, 3, 2)
                .is_some_and(|den_inner| compare_expr(ctx, den_inner, inner) == Ordering::Equal));
    if !denominator_matches {
        return None;
    }

    let three = ctx.num(3);
    let triple_inner = smart_mul(ctx, three, inner);
    Some(ctx.call_builtin(BuiltinFn::Tanh, vec![triple_inner]))
}

fn try_rewrite_sinh_double_angle_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewritten) =
        cas_math::hyperbolic_identity_support::try_rewrite_sinh_double_angle_expansion(ctx, expr)
    {
        return Some(rewritten);
    }

    let ScaledHyperbolicProductPatternForCancellation::SinhCosh(sinh_arg, cosh_arg) =
        extract_scaled_hyperbolic_two_factor_product_for_cancellation(ctx, expr, 2)?
    else {
        return None;
    };
    if compare_expr(ctx, sinh_arg, cosh_arg) != Ordering::Equal {
        return None;
    }

    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, sinh_arg);
    Some(ctx.call_builtin(BuiltinFn::Sinh, vec![double_arg]))
}

fn try_rewrite_tanh_double_angle_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewritten) =
        cas_math::hyperbolic_identity_support::try_rewrite_tanh_double_angle_expansion(ctx, expr)
    {
        return Some(rewritten);
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };

    let inner = extract_scaled_tanh_power_for_cancellation(ctx, *numerator, 2, 1)?;
    let Expr::Add(den_left, den_right) = ctx.get(*denominator) else {
        return None;
    };

    let denominator_matches = (cas_math::expr_predicates::is_one_expr(ctx, *den_left)
        && extract_tanh_power_for_cancellation(ctx, *den_right, 2)
            .is_some_and(|den_inner| compare_expr(ctx, den_inner, inner) == Ordering::Equal))
        || (cas_math::expr_predicates::is_one_expr(ctx, *den_right)
            && extract_tanh_power_for_cancellation(ctx, *den_left, 2)
                .is_some_and(|den_inner| compare_expr(ctx, den_inner, inner) == Ordering::Equal));
    if !denominator_matches {
        return None;
    }

    let two = ctx.num(2);
    let double_inner = smart_mul(ctx, two, inner);
    Some(ctx.call_builtin(BuiltinFn::Tanh, vec![double_inner]))
}

pub(super) fn try_rewrite_tanh_exp_definition_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let rewrite = cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(
        ctx, expr,
    )?;
    match rewrite.kind {
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::TanhRatio
        | cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegTanhRatio => {
            Some(rewrite.rewritten)
        }
        _ => None,
    }
}

pub(super) fn try_rewrite_sinh_cosh_exp_definition_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let rewrite = cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(
        ctx, expr,
    )?;
    match rewrite.kind {
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshHalf
        | cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::SinhHalf
        | cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegSinhHalf
        | cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshDirect
        | cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::SinhDirect
        | cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegSinhDirect => {
            Some(rewrite.rewritten)
        }
        _ => None,
    }
}

pub(super) fn try_rewrite_hyperbolic_pythagorean_factor_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<HyperbolicPythagoreanFactorCancellationMatch> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => (lhs, rhs),
        Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
            Expr::Neg(inner) => (lhs, inner),
            _ => match ctx.get(lhs).clone() {
                Expr::Neg(inner) => (rhs, inner),
                _ => return None,
            },
        },
        _ => return None,
    };

    if let (
        Some((lhs_sign, lhs_coeff, lhs_arg, lhs_power)),
        Some((rhs_sign, rhs_coeff, rhs_arg, rhs_power)),
    ) = (
        extract_signed_cosh_power(ctx, lhs),
        extract_signed_cosh_power(ctx, rhs),
    ) {
        if compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            && lhs_sign == rhs_sign
            && exprs_match_for_cancellation(ctx, lhs_coeff, rhs_coeff)
        {
            let coeff_with_sign = apply_sign_to_expr(ctx, lhs_sign, lhs_coeff);
            let oriented_coeff = match (lhs_power, rhs_power) {
                (3, 1) => coeff_with_sign,
                (1, 3) => ctx.add(Expr::Neg(coeff_with_sign)),
                _ => return None,
            };

            let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![lhs_arg]);
            let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![lhs_arg]);
            let two = ctx.num(2);
            let one = ctx.num(1);
            let cosh_sq = ctx.add(Expr::Pow(cosh_arg, two));
            let sinh_sq = ctx.add(Expr::Pow(sinh_arg, two));
            let factorized_inner = ctx.add(Expr::Sub(cosh_sq, one));
            let factorized_product = smart_mul(ctx, cosh_arg, factorized_inner);
            let rewritten_product = smart_mul(ctx, cosh_arg, sinh_sq);
            let factorized = smart_mul(ctx, oriented_coeff, factorized_product);
            let rewritten = smart_mul(ctx, oriented_coeff, rewritten_product);

            return Some(HyperbolicPythagoreanFactorCancellationMatch {
                local_before: factorized,
                local_after: rewritten,
                mode: HyperbolicPythagoreanFactorCancellationMode::FactorThenRewrite {
                    factorized,
                    rewritten,
                },
            });
        }
    }

    try_rewrite_factored_hyperbolic_pythagorean_for_cancellation(ctx, lhs, rhs)
        .or_else(|| try_rewrite_factored_hyperbolic_pythagorean_for_cancellation(ctx, rhs, lhs))
}

fn extract_sinh_sq_multiplier(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let mut coefficient_factors = smallvec::SmallVec::<[cas_ast::ExprId; 8]>::new();
    let mut seen_sinh_sq = false;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        match ctx.get(factor) {
            Expr::Pow(base, exponent) => {
                let Expr::Function(fn_id, args) = ctx.get(*base) else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if !ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                    || args.len() != 1
                    || compare_expr(ctx, args[0], arg) != Ordering::Equal
                    || extract_i64_integer(ctx, *exponent)? != 2
                {
                    coefficient_factors.push(factor);
                    continue;
                }
                if seen_sinh_sq {
                    return None;
                }
                seen_sinh_sq = true;
            }
            _ => coefficient_factors.push(factor),
        }
    }

    if !seen_sinh_sq {
        return None;
    }

    Some(if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &coefficient_factors.into_vec())
    })
}

fn extract_cosh_sq_multiplier(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let mut coefficient_factors = smallvec::SmallVec::<[cas_ast::ExprId; 8]>::new();
    let mut seen_cosh_sq = false;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        match ctx.get(factor) {
            Expr::Pow(base, exponent) => {
                let Expr::Function(fn_id, args) = ctx.get(*base) else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if !ctx.is_builtin(*fn_id, BuiltinFn::Cosh)
                    || args.len() != 1
                    || compare_expr(ctx, args[0], arg) != Ordering::Equal
                    || extract_i64_integer(ctx, *exponent)? != 2
                {
                    coefficient_factors.push(factor);
                    continue;
                }
                if seen_cosh_sq {
                    return None;
                }
                seen_cosh_sq = true;
            }
            _ => coefficient_factors.push(factor),
        }
    }

    if !seen_cosh_sq {
        return None;
    }

    Some(if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &coefficient_factors.into_vec())
    })
}

fn match_sinh_sq_plus_one_multiple(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    for candidate in [expr, cas_math::canonical_forms::normalize_core(ctx, expr)] {
        let view = AddView::from_expr(ctx, candidate);
        if view.terms.len() != 2 || view.terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
            continue;
        }

        for squared_index in 0..2 {
            let const_index = 1 - squared_index;
            let squared_term = view.terms[squared_index].0;
            let const_term = view.terms[const_index].0;
            let Some(coeff) = extract_sinh_sq_multiplier(ctx, squared_term, arg) else {
                continue;
            };
            if exprs_match_for_cancellation(ctx, coeff, const_term) {
                return Some(coeff);
            }
        }
    }

    None
}

fn match_cosh_sq_minus_one_multiple(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    for candidate in [expr, cas_math::canonical_forms::normalize_core(ctx, expr)] {
        let view = AddView::from_expr(ctx, candidate);
        if view.terms.len() != 2 {
            continue;
        }

        for squared_index in 0..2 {
            let const_index = 1 - squared_index;
            let squared_term = view.terms[squared_index];
            let const_term = view.terms[const_index];
            if squared_term.1 != Sign::Pos || const_term.1 != Sign::Neg {
                continue;
            }
            let Some(coeff) = extract_cosh_sq_multiplier(ctx, squared_term.0, arg) else {
                continue;
            };
            if exprs_match_for_cancellation(ctx, coeff, const_term.0) {
                return Some(coeff);
            }
        }
    }

    None
}

fn extract_factored_hyperbolic_linear_term(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(
    i64,
    cas_ast::ExprId,
    cas_ast::ExprId,
    cas_ast::ExprId,
    BuiltinFn,
    &'static str,
)> {
    let (sign, positive_expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (-1, *inner),
        _ => (1, expr),
    };

    let mut outer_coeff_factors = smallvec::SmallVec::<[cas_ast::ExprId; 8]>::new();
    let mut hyperbolic_arg = None;
    let mut outer_builtin = None;
    let mut additive_factor = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 =>
            {
                if hyperbolic_arg.is_some() {
                    return None;
                }
                hyperbolic_arg = Some(args[0]);
                outer_builtin = Some(BuiltinFn::Cosh);
            }
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) && args.len() == 1 =>
            {
                if hyperbolic_arg.is_some() {
                    return None;
                }
                hyperbolic_arg = Some(args[0]);
                outer_builtin = Some(BuiltinFn::Sinh);
            }
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                if additive_factor.is_some() {
                    return None;
                }
                additive_factor = Some(factor);
            }
            _ => outer_coeff_factors.push(factor),
        }
    }

    let arg = hyperbolic_arg?;
    let outer_builtin = outer_builtin?;
    let additive = additive_factor?;
    let (inner_coeff, identity_desc) = match outer_builtin {
        BuiltinFn::Cosh => (
            match_sinh_sq_plus_one_multiple(ctx, additive, arg)?,
            "Usar sinh(u)^2 + 1 = cosh(u)^2",
        ),
        BuiltinFn::Sinh => (
            match_cosh_sq_minus_one_multiple(ctx, additive, arg)?,
            "Usar cosh(u)^2 - 1 = sinh(u)^2",
        ),
        _ => return None,
    };
    let outer_coeff = if outer_coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &outer_coeff_factors.into_vec())
    };

    Some((
        sign,
        smart_mul(ctx, outer_coeff, inner_coeff),
        arg,
        positive_expr,
        outer_builtin,
        identity_desc,
    ))
}

fn try_rewrite_factored_hyperbolic_pythagorean_for_cancellation(
    ctx: &mut cas_ast::Context,
    linear_expr: cas_ast::ExprId,
    cubic_expr: cas_ast::ExprId,
) -> Option<HyperbolicPythagoreanFactorCancellationMatch> {
    let (
        linear_sign,
        linear_coeff,
        linear_arg,
        linear_before_positive,
        outer_builtin,
        identity_desc,
    ) = extract_factored_hyperbolic_linear_term(ctx, linear_expr)?;
    let (cubic_sign, cubic_coeff, cubic_arg, cubic_power) =
        extract_signed_hyperbolic_power(ctx, cubic_expr, outer_builtin)?;

    if cubic_power != 3
        || linear_sign != cubic_sign
        || compare_expr(ctx, linear_arg, cubic_arg) != Ordering::Equal
        || !exprs_match_for_cancellation(ctx, linear_coeff, cubic_coeff)
    {
        return None;
    }

    let hyperbolic_arg = ctx.call_builtin(outer_builtin, vec![linear_arg]);
    let three = ctx.num(3);
    let hyperbolic_cubed = ctx.add(Expr::Pow(hyperbolic_arg, three));
    let rewritten_positive = smart_mul(ctx, cubic_coeff, hyperbolic_cubed);

    Some(HyperbolicPythagoreanFactorCancellationMatch {
        local_before: apply_sign_to_expr(ctx, linear_sign, linear_before_positive),
        local_after: apply_sign_to_expr(ctx, cubic_sign, rewritten_positive),
        mode: HyperbolicPythagoreanFactorCancellationMode::AlreadyFactored { identity_desc },
    })
}

pub(super) fn try_build_exact_hyperbolic_equivalence_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let nested_default_simplify = default_simplify_nesting_depth() > 0;
    let view = AddView::from_expr(ctx, expr);
    if !(2..=3).contains(&view.terms.len()) {
        return None;
    }
    if reject_linear_hyperbolic_combination_before_zero_scope(ctx, expr) {
        return None;
    }

    for subset_len in 1..=2 {
        for first_index in 0..view.terms.len() {
            let second_index_options: Vec<Option<usize>> = if subset_len == 1 {
                vec![None]
            } else {
                ((first_index + 1)..view.terms.len()).map(Some).collect()
            };

            for second_index in second_index_options {
                let focus_terms: Vec<_> = view
                    .terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index == first_index || second_index == Some(index)).then_some(term)
                    })
                    .collect();
                if focus_terms.len() != subset_len {
                    continue;
                }
                let focus_expr = build_signed_sum_expr(ctx, &focus_terms);

                let remaining_terms: Vec<_> = view
                    .terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index != first_index && second_index != Some(index)).then_some(term)
                    })
                    .collect();
                if remaining_terms.is_empty() {
                    continue;
                }
                let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

                let focus_variants = [
                    (focus_expr, 1_i64),
                    (ctx.add(Expr::Neg(focus_expr)), -1_i64),
                ];

                for (candidate_focus, focus_sign) in focus_variants {
                    let Some((rewritten, description)) =
                        try_rewrite_exact_hyperbolic_equivalence_for_cancellation(
                            ctx,
                            candidate_focus,
                        )
                    else {
                        continue;
                    };

                    let adjusted_rewritten = apply_sign_to_expr(ctx, focus_sign, rewritten);
                    let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
                    let distributed_neg_adjusted_rewritten =
                        negate_additive_scope_expr(ctx, adjusted_rewritten);
                    let matches_remaining = expr_matches_negation_for_cancellation(
                        ctx,
                        adjusted_rewritten,
                        remaining_expr,
                    ) || (!nested_default_simplify
                        && (expr_matches_negation_after_default_simplify(
                            ctx,
                            adjusted_rewritten,
                            remaining_expr,
                        ) || exprs_match_after_default_simplify(
                            ctx,
                            neg_adjusted_rewritten,
                            remaining_expr,
                        ) || exprs_match_after_default_simplify(
                            ctx,
                            distributed_neg_adjusted_rewritten,
                            remaining_expr,
                        ) || additive_scopes_match_after_default_simplify(
                            ctx,
                            distributed_neg_adjusted_rewritten,
                            remaining_expr,
                        )));
                    if matches_remaining {
                        return Some(
                            Rewrite::with_local(
                                ctx.num(0),
                                description,
                                focus_expr,
                                adjusted_rewritten,
                            )
                            .substep(
                                "Cancelar términos iguales",
                                vec![
                                    "Tras aplicar la identidad hiperbólica, el resto de la expresión es el opuesto y toda la expresión se anula."
                                        .to_string(),
                                ],
                            ),
                        );
                    }
                }
            }
        }
    }

    None
}

pub(super) fn try_build_exact_hyperbolic_angle_sum_diff_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    for focus_index in 0..view.terms.len() {
        let (focus_expr, focus_sign) = view.terms[focus_index];
        if focus_sign != Sign::Pos {
            continue;
        }

        let Some(rewritten) =
            try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, focus_expr)
        else {
            continue;
        };

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

        let neg_rewritten = ctx.add(Expr::Neg(rewritten));
        if expr_matches_negation_for_cancellation(ctx, rewritten, remaining_expr)
            || exprs_match_after_default_simplify(ctx, neg_rewritten, remaining_expr)
        {
            return Some(
                Rewrite::with_local(
                    ctx.num(0),
                    "Expand hyperbolic angle sum/difference",
                    focus_expr,
                    rewritten,
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras aplicar la identidad, el resto de la expresión es exactamente el opuesto y el resultado es 0."
                            .to_string(),
                    ],
                ),
            );
        }
    }

    None
}

pub(super) fn try_build_exact_hyperbolic_pythagorean_factor_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for first_index in 0..view.terms.len() {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms = [view.terms[first_index], view.terms[second_index]];
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);
            let Some(rewrite_match) =
                try_rewrite_hyperbolic_pythagorean_factor_for_cancellation(ctx, focus_expr)
            else {
                continue;
            };

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
            if !expr_matches_negation_after_default_simplify(
                ctx,
                rewrite_match.local_after,
                remaining_expr,
            ) {
                continue;
            }

            return Some(build_hyperbolic_pythagorean_factor_zero_rewrite(
                ctx,
                rewrite_match,
            ));
        }
    }

    None
}

fn build_hyperbolic_pythagorean_factor_zero_rewrite(
    ctx: &mut cas_ast::Context,
    rewrite_match: HyperbolicPythagoreanFactorCancellationMatch,
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
        "Apply hyperbolic Pythagorean identity",
        rewrite_match.local_before,
        rewrite_match.local_after,
    );

    match rewrite_match.mode {
        HyperbolicPythagoreanFactorCancellationMode::FactorThenRewrite {
            factorized,
            rewritten: _,
        } => {
            let factorized_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: factorized
                }
            );
            rewrite
                .substep(
                    "Sacar factor común",
                    vec![format!("Sacar factor común para obtener {factorized_display}.")],
                )
                .substep(
                    "Usar cosh(u)^2 - 1 = sinh(u)^2",
                    vec![format!("Así se obtiene {focus_after_display}.")],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                            .to_string(),
                    ],
                )
        }
        HyperbolicPythagoreanFactorCancellationMode::AlreadyFactored { identity_desc } => rewrite
            .substep(
                identity_desc,
                vec![format!("Así se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                        .to_string(),
                ],
            ),
    }
}

pub(super) fn build_hyperbolic_pythagorean_factor_root_zero_rewrite(
    ctx: &mut cas_ast::Context,
    whole_expr: cas_ast::ExprId,
    rewrite_match: HyperbolicPythagoreanFactorCancellationMatch,
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
        "Apply hyperbolic Pythagorean identity",
        whole_expr,
        ctx.num(0),
    );

    match rewrite_match.mode {
        HyperbolicPythagoreanFactorCancellationMode::FactorThenRewrite {
            factorized,
            rewritten: _,
        } => {
            let factorized_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: factorized
                }
            );
            rewrite
                .substep(
                    "Sacar factor común",
                    vec![format!("Sacar factor común para obtener {factorized_display}.")],
                )
                .substep(
                    "Usar cosh(u)^2 - 1 = sinh(u)^2",
                    vec![format!("Así se obtiene {focus_after_display}.")],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                            .to_string(),
                    ],
                )
        }
        HyperbolicPythagoreanFactorCancellationMode::AlreadyFactored { identity_desc } => rewrite
            .substep(
                identity_desc,
                vec![format!("Así se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                        .to_string(),
                ],
            ),
    }
}

pub(crate) fn try_build_hyperbolic_pythagorean_factor_root_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    if !maybe_hyperbolic_pythagorean_factor_zero_candidate(ctx, expr) {
        return None;
    }

    let parent_ctx = ParentContext::root();
    ExpandHyperbolicPythagoreanFactorToEnableCancellationRule.apply(ctx, expr, &parent_ctx)
}

pub(super) fn try_build_exact_zero_hyperbolic_sinh_cubic_polynomial_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if !(2..=3).contains(&view.terms.len()) {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let Some(rewritten) =
            try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(
                ctx,
                focus_term_expr,
            )
        else {
            continue;
        };

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
        let adjusted_rewritten = apply_sign_to_expr(ctx, sign_to_i64(focus_term_sign), rewritten);
        let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
        let distributed_neg_adjusted_rewritten =
            negate_additive_scope_expr(ctx, adjusted_rewritten);

        if expr_matches_negation_for_cancellation(ctx, adjusted_rewritten, remaining_expr)
            || expr_matches_negation_after_default_simplify(ctx, adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(ctx, neg_adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
            || additive_scopes_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Hyperbolic Product-to-Sum and Triple-Angle Identity",
                focus_term_expr,
                rewritten,
            ));
        }
    }

    None
}

pub(super) fn try_rewrite_safe_direct_hyperbolic_equivalence_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, &'static str)> {
    let has_direct_hyperbolic = expr_contains_direct_hyperbolic_builtin(ctx, expr);

    if let Some(rewritten) =
        try_rewrite_atanh_square_ratio_log_equivalence_for_cancellation(ctx, expr)
    {
        return Some((rewritten, "Inverse Hyperbolic Log Definition"));
    }

    if let Some(rewritten) = try_rewrite_atanh_ln_definition_for_cancellation(ctx, expr) {
        return Some((rewritten, "Inverse Hyperbolic Log Definition"));
    }

    // If the expression only contains inverse-hyperbolic structure, the reusable
    // cancellation routes in this helper stop at the atanh/log definitions above.
    if !has_direct_hyperbolic {
        return None;
    }

    if let Some(rewrite) = try_rewrite_hyperbolic_half_angle_squares_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Hyperbolic Half-Angle Squares"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_exp_equivalence_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Sum to Exponential"));
    }

    if let Some(rewritten) = try_rewrite_sinh_cosh_exp_definition_for_cancellation(ctx, expr) {
        return Some((rewritten, "Recognize Hyperbolic from Exponential"));
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_triple_angle(ctx, expr)
    {
        return Some((rewrite.rewritten, "Hyperbolic Triple-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Angle Sum/Difference Identity"));
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_double_angle_sum(ctx, expr)
    {
        return Some((rewrite.rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_double_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_sinh_double_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_tanh_double_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Double-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_tanh_exp_definition_for_cancellation(ctx, expr) {
        return Some((rewritten, "Recognize Hyperbolic from Exponential"));
    }

    if let Some(rewritten) = try_rewrite_tanh_angle_sum_diff_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Angle Sum/Difference Identity"));
    }

    if let Some(rewritten) = try_rewrite_tanh_triple_angle_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Triple-Angle Identity"));
    }

    if let Some(rewritten) = try_rewrite_hyperbolic_sum_to_product_for_cancellation(ctx, expr) {
        return Some((rewritten, "Hyperbolic Product-to-Sum Identity"));
    }

    if let Some(rewritten) =
        try_rewrite_hyperbolic_product_sum_cosh_cubic_polynomial_for_cancellation(ctx, expr)
    {
        return Some((
            rewritten,
            "Hyperbolic Product-to-Sum and Triple-Angle Identity",
        ));
    }

    if let Some(rewritten) =
        try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(ctx, expr)
    {
        return Some((
            rewritten,
            "Hyperbolic Product-to-Sum and Triple-Angle Identity",
        ));
    }

    None
}

pub(super) fn try_build_fast_hyperbolic_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if !(2..=4).contains(&view.terms.len()) {
        return None;
    }

    if !expr_contains_any_builtin(
        ctx,
        expr,
        &[BuiltinFn::Sinh, BuiltinFn::Cosh, BuiltinFn::Tanh],
    ) {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let Some((rewritten, description)) =
            try_rewrite_safe_direct_hyperbolic_equivalence_for_cancellation(ctx, focus_term_expr)
        else {
            continue;
        };

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
        let adjusted_rewritten = apply_sign_to_expr(ctx, sign_to_i64(focus_term_sign), rewritten);

        if expr_matches_negation_for_cancellation(ctx, adjusted_rewritten, remaining_expr) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                description,
                focus_term_expr,
                rewritten,
            ));
        }
    }

    None
}

pub(super) fn reject_hyperbolic_additive_mismatch_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    let one = ctx.num(1);

    for (additive_expr, other_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((add_builtin, add_arg, tail_expr, tail_sign)) =
            extract_hyperbolic_square_atomic_tail_profile_pair(ctx, additive_expr)
        else {
            continue;
        };
        let Some((other_builtin, other_arg, other_power)) =
            extract_single_hyperbolic_linear_or_small_power_term_for_reject(ctx, other_expr)
        else {
            continue;
        };

        if other_power != 2
            || add_builtin == other_builtin
            || compare_expr(ctx, add_arg, other_arg) != Ordering::Equal
        {
            continue;
        }

        // Preserve the exact hyperbolic Pythagorean pairs:
        // cosh(u)^2 - 1 = sinh(u)^2 and sinh(u)^2 + 1 = cosh(u)^2.
        if compare_expr(ctx, tail_expr, one) == Ordering::Equal
            && matches!(
                (add_builtin, other_builtin, tail_sign),
                (BuiltinFn::Cosh, BuiltinFn::Sinh, Sign::Neg)
                    | (BuiltinFn::Sinh, BuiltinFn::Cosh, Sign::Pos)
            )
        {
            return None;
        }

        return Some(false);
    }

    None
}

pub(super) fn expr_is_symbolic_leaf_for_hyperbolic_reject(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    matches!(ctx.get(expr), Expr::Variable(_) | Expr::SessionRef(_))
}

pub(super) fn extract_single_hyperbolic_linear_or_small_power_term_for_reject(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, i64)> {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    if let Some((builtin, arg)) = extract_hyperbolic_linear_term_for_profile(ctx, expr) {
        return Some((builtin, arg, 1));
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let (builtin, arg) = extract_hyperbolic_linear_term_for_profile(ctx, base)?;
    let power = small_positive_integer_value(ctx, exp)?;
    ((2..=3).contains(&power)).then_some((builtin, arg, power))
}

pub(super) fn reject_obvious_hyperbolic_pair_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    let (lhs_builtin, lhs_arg, lhs_power) =
        extract_single_hyperbolic_linear_or_small_power_term_for_reject(ctx, lhs_core)?;
    let (rhs_builtin, rhs_arg, rhs_power) =
        extract_single_hyperbolic_linear_or_small_power_term_for_reject(ctx, rhs_core)?;

    if lhs_power == 1
        && rhs_power == 1
        && lhs_builtin == rhs_builtin
        && expr_is_symbolic_leaf_for_hyperbolic_reject(ctx, lhs_arg)
        && expr_is_symbolic_leaf_for_hyperbolic_reject(ctx, rhs_arg)
        && compare_expr(ctx, lhs_arg, rhs_arg) != Ordering::Equal
    {
        return Some(false);
    }

    if lhs_power == 2
        && rhs_power == 2
        && lhs_builtin != rhs_builtin
        && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
    {
        return Some(false);
    }

    if lhs_builtin == rhs_builtin
        && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
        && ((lhs_power == 1 && rhs_power == 3) || (lhs_power == 3 && rhs_power == 1))
    {
        return Some(false);
    }

    None
}

fn extract_scaled_single_hyperbolic_term_for_reject(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<SingleHyperbolicTermRejectProfile> {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    if let Some((builtin, arg)) = extract_hyperbolic_linear_term_for_profile(ctx, expr) {
        return Some(SingleHyperbolicTermRejectProfile { builtin, arg });
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut hyperbolic_arg = None;
    let mut hyperbolic_builtin = None;
    for factor in factors {
        if let Some((builtin, arg)) = extract_hyperbolic_linear_term_for_profile(ctx, factor) {
            if hyperbolic_arg.replace(arg).is_some() {
                return None;
            }
            hyperbolic_builtin = Some(builtin);
            continue;
        }

        if expr_contains_any_function_call(ctx, factor) {
            return None;
        }
    }

    Some(SingleHyperbolicTermRejectProfile {
        builtin: hyperbolic_builtin?,
        arg: hyperbolic_arg?,
    })
}

pub(super) fn reject_linear_hyperbolic_combination_before_zero_scope(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if !(2..=3).contains(&view.terms.len()) {
        return false;
    }

    let mut profiles = Vec::with_capacity(view.terms.len());
    for (term_expr, term_sign) in view.terms {
        let (term_expr, _) = normalize_signed_add_term(ctx, term_expr, term_sign);
        let Some(profile) = extract_scaled_single_hyperbolic_term_for_reject(ctx, term_expr) else {
            return false;
        };
        profiles.push(profile);
    }

    let first_arg = profiles[0].arg;
    if profiles
        .iter()
        .any(|profile| compare_expr(ctx, profile.arg, first_arg) != Ordering::Equal)
    {
        return false;
    }

    let has_sinh = profiles
        .iter()
        .any(|profile| profile.builtin == BuiltinFn::Sinh);
    let has_cosh = profiles
        .iter()
        .any(|profile| profile.builtin == BuiltinFn::Cosh);

    has_sinh && has_cosh
}
