//! `arithmetic`: familia `trig` — parte de la **familia ANGULAR del
//! núcleo de cancelación** (D1c-11, 2026-08-02); el ancla y la doctrina
//! viven en la cabecera de `phase_shift.rs`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto del troceo.

use super::*;

fn direct_sin_cos_arg(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId)> {
    let expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    match ctx.builtin_of(*fn_id) {
        Some(builtin @ (BuiltinFn::Sin | BuiltinFn::Cos)) => Some((builtin, args[0])),
        _ => None,
    }
}

pub(super) fn same_arg_sin_cos_core_pair(
    ctx: &cas_ast::Context,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
) -> bool {
    let Some((left_builtin, left_arg)) = direct_sin_cos_arg(ctx, left) else {
        return false;
    };
    let Some((right_builtin, right_arg)) = direct_sin_cos_arg(ctx, right) else {
        return false;
    };

    left_builtin != right_builtin && compare_expr(ctx, left_arg, right_arg) == Ordering::Equal
}

pub(super) fn same_arg_sin_cos_additive_pair(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    same_arg_sin_cos_core_pair(ctx, view.terms[0].0, view.terms[1].0)
}

pub(super) fn top_level_terms_match_trig_family_or_number(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    trig_fns: &[BuiltinFn],
    min_trig_terms: usize,
) -> bool {
    let mut trig_term_count = 0usize;

    for (term_expr, _) in AddView::from_expr(ctx, expr).terms {
        if expr_contains_any_builtin(ctx, term_expr, trig_fns) {
            trig_term_count += 1;
            continue;
        }
        if matches!(ctx.get(term_expr), Expr::Number(_)) {
            continue;
        }
        return false;
    }

    trig_term_count >= min_trig_terms
}

fn is_direct_trig_function_term_for_exact_scope(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && (ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Tan)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cot)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Sec)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Csc))
        }
        _ => false,
    }
}

fn extract_scaled_direct_trig_function_term_for_exact_scope(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let positive_expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };

    if is_direct_trig_function_term_for_exact_scope(ctx, positive_expr) {
        return true;
    }

    if !matches!(ctx.get(positive_expr), Expr::Mul(_, _)) {
        return false;
    }

    let mut saw_trig = false;
    for factor in flatten_mul_chain(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Number(_) => {}
            _ if is_direct_trig_function_term_for_exact_scope(ctx, factor) && !saw_trig => {
                saw_trig = true;
            }
            _ => return false,
        }
    }

    saw_trig
}

fn is_surface_trig_product_term_for_exact_scope(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let positive_expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };

    if !matches!(ctx.get(positive_expr), Expr::Mul(_, _)) {
        return false;
    }

    let mut trig_factor_count = 0usize;
    for factor in flatten_mul_chain(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Number(_) => {}
            _ if is_direct_trig_function_term_for_exact_scope(ctx, factor) => {
                trig_factor_count += 1;
            }
            _ => return false,
        }
    }

    trig_factor_count >= 2
}

fn matches_two_term_direct_trig_against_trig_product_zero_scope_family(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let lhs = view.terms[0].0;
    let rhs = view.terms[1].0;
    (extract_scaled_direct_trig_function_term_for_exact_scope(ctx, lhs)
        && is_surface_trig_product_term_for_exact_scope(ctx, rhs))
        || (extract_scaled_direct_trig_function_term_for_exact_scope(ctx, rhs)
            && is_surface_trig_product_term_for_exact_scope(ctx, lhs))
}

pub(super) fn maybe_exact_trig_equivalence_zero_scope_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let term_count = AddView::from_expr(ctx, expr).terms.len();
    (2..=6).contains(&term_count)
        && top_level_terms_match_trig_family_or_number(
            ctx,
            expr,
            &[BuiltinFn::Sin, BuiltinFn::Cos],
            2,
        )
        && !matches_two_term_direct_trig_against_trig_product_zero_scope_family(ctx, expr)
        && !matches_structural_trig_sum_to_product_zero_scope_family(ctx, expr)
}

pub(super) fn has_direct_trig_builtin_on_either_side(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    expr_contains_direct_trig_builtin(ctx, lhs_core)
        || expr_contains_direct_trig_builtin(ctx, rhs_core)
}

pub(super) fn expr_contains_direct_trig_builtin(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_any_builtin(
        ctx,
        expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    )
}

pub(super) fn expr_is_sin_ratio_term(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return false;
    };
    let Expr::Function(num_fn_id, num_args) = ctx.get(*numerator) else {
        return false;
    };
    let Expr::Function(den_fn_id, den_args) = ctx.get(*denominator) else {
        return false;
    };
    ctx.is_builtin(*num_fn_id, BuiltinFn::Sin)
        && ctx.is_builtin(*den_fn_id, BuiltinFn::Sin)
        && num_args.len() == 1
        && den_args.len() == 1
}

pub(super) fn extract_plain_sin_or_cos_arg_for_product_sum_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && (ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cos)) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

pub(super) fn is_plain_two_term_sin_cos_sum_or_diff_product_sum_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    terms.len() == 2
        && terms.iter().all(|(term_expr, _)| {
            let (_coeff, base) = extract_coef_and_base(ctx, *term_expr);
            extract_plain_sin_or_cos_arg_for_product_sum_candidate(ctx, base).is_some()
        })
}

pub(super) fn expr_contains_direct_sin_or_cos_with_arg(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    target_arg: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if args.len() == 1
                    && (ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                        || ctx.is_builtin(*fn_id, BuiltinFn::Cos))
                    && compare_expr(ctx, args[0], target_arg) == Ordering::Equal
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

pub(super) fn try_rewrite_exact_trig_equivalence_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, &'static str)> {
    if let Some(rewrite) = try_rewrite_pythagorean_identity_add_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Pythagorean Identity"));
    }

    if let Some(rewrite) = try_rewrite_pythagorean_factor_form_add_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Pythagorean Identity"));
    }

    if let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Pythagorean Identity"));
    }

    if let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Pythagorean Identity"));
    }

    if let Some((rewritten, description)) =
        try_rewrite_trig_sum_to_product_for_cancellation(ctx, expr)
    {
        return Some((rewritten, description));
    }

    if let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, expr) {
        let description = match rewrite.kind {
            TrigProductToSumRewriteKind::SinCos
            | TrigProductToSumRewriteKind::CosSin
            | TrigProductToSumRewriteKind::CosCos
            | TrigProductToSumRewriteKind::SinSin => "Product-to-Sum Identity",
        };
        return Some((rewrite.rewritten, description));
    }

    if let Some(rewrite) = try_rewrite_angle_sum_diff_identity_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Angle Sum/Diff Identity"));
    }

    if let Some(rewritten) = try_rewrite_trig_double_angle_cos_one_minus_two_sin_sq_expr(ctx, expr)
    {
        return Some((rewritten, "Double Angle Expansion"));
    }

    if let Some(rewrite) = try_rewrite_double_angle_function_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Double Angle Expansion"));
    }

    if let Some(rewrite) = try_rewrite_triple_angle_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Triple Angle Identity"));
    }

    if let Some(rewrite) = try_rewrite_quintuple_angle_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Quintuple Angle Identity"));
    }

    if let Some(rewrite) = try_rewrite_recursive_trig_expansion_expr(ctx, expr) {
        let description = if rewrite.desc.starts_with("sin(") || rewrite.desc.starts_with("cos(") {
            "Angle Sum/Diff Identity"
        } else {
            "Recursive Trig Expansion"
        };
        return Some((rewrite.rewritten, description));
    }

    if let Some(rewritten) = try_rewrite_trig_square_double_angle_half_expr(ctx, expr) {
        return Some((rewritten, "Half-Angle Square Identity"));
    }

    if let Some(rewrite) = try_rewrite_trig_half_angle_squares_expr(ctx, expr) {
        let description = match rewrite.kind {
            cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::TrigSin
            | cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::TrigCos => {
                "Half-Angle Square Identity"
            }
            cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicCosh
            | cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicSinh => {
                return None;
            }
        };
        return Some((rewrite.rewritten, description));
    }

    if let Some(rewritten) = try_rewrite_trig_power_reduction_expr_for_cancellation(ctx, expr) {
        return Some((rewritten, "Power Reduction Identity"));
    }

    if let Some(rewrite) = try_rewrite_recognize_sec_squared_add_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Recognize Secant Squared"));
    }

    if let Some(rewrite) = try_rewrite_recognize_csc_squared_add_expr(ctx, expr) {
        return Some((rewrite.rewritten, "Recognize Cosecant Squared"));
    }

    None
}

fn extract_trig_even_power_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, u32)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let power = extract_i64_integer(ctx, *exp)?;
    if !(4..=62).contains(&power) || !power.is_multiple_of(&2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let trig_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        BuiltinFn::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        BuiltinFn::Cos
    } else {
        return None;
    };

    Some((trig_fn, args[0], u32::try_from(power).ok()?))
}

pub(super) fn extract_scaled_trig_even_power_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BigRational, BuiltinFn, cas_ast::ExprId, u32)> {
    let (base_expr, mut coeff) = if let Expr::Neg(inner) = ctx.get(expr) {
        (*inner, BigRational::from_integer(BigInt::from(-1_i32)))
    } else {
        (expr, BigRational::from_integer(BigInt::from(1_i32)))
    };

    let mut factors = Vec::new();
    let mut stack = vec![base_expr];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(lhs, rhs) = ctx.get(curr) {
            stack.push(*lhs);
            stack.push(*rhs);
        } else {
            factors.push(curr);
        }
    }

    let mut trig_match = None;
    for factor in &factors {
        if trig_match.is_none() {
            trig_match = extract_trig_even_power_for_cancellation(ctx, *factor);
            if trig_match.is_some() {
                continue;
            }
        }

        let Expr::Number(n) = ctx.get(*factor) else {
            return None;
        };
        coeff *= n.clone();
    }

    let (trig_fn, arg, power) = trig_match?;
    Some((coeff, trig_fn, arg, power))
}

pub(super) fn extract_trig_square_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exp) != Some(2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let trig_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        BuiltinFn::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        BuiltinFn::Cos
    } else {
        return None;
    };

    Some((trig_fn, args[0]))
}

pub(super) fn extract_trig_square_product_same_arg_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    let factors = [*left, *right];

    let mut sin_arg = None;
    let mut cos_arg = None;
    for factor in factors {
        let (trig_fn, arg) = extract_trig_square_for_cancellation(ctx, factor)?;
        match trig_fn {
            BuiltinFn::Sin => sin_arg = Some(arg),
            BuiltinFn::Cos => cos_arg = Some(arg),
            _ => return None,
        }
    }

    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    (compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal).then_some(sin_arg)
}

fn build_cos_multiple_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    factor: i64,
) -> cas_ast::ExprId {
    let factor_expr = ctx.num(factor);
    let scaled_arg = smart_mul(ctx, factor_expr, arg);
    ctx.call_builtin(BuiltinFn::Cos, vec![scaled_arg])
}

fn build_scaled_cos_multiple_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    angle_factor: i64,
    coeff: i64,
) -> cas_ast::ExprId {
    let cos_term = build_cos_multiple_for_cancellation(ctx, arg, angle_factor);
    let coeff_expr = ctx.num(coeff);
    smart_mul(ctx, coeff_expr, cos_term)
}

fn build_trig_power_reduction_rewritten_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    power: u32,
    trig_fn: BuiltinFn,
) -> Option<cas_ast::ExprId> {
    if !(4..=62).contains(&power) || !power.is_multiple_of(2) {
        return None;
    }
    if trig_fn != BuiltinFn::Sin && trig_fn != BuiltinFn::Cos {
        return None;
    }

    let half = power / 2;
    let denominator = 1_i64.checked_shl(power - 1)?;
    let constant_coeff = binomial_i64_for_cancellation(power, half)?.checked_div(2)?;

    let mut numerator = ctx.num(constant_coeff);
    for j in 1..=half {
        let coeff = binomial_i64_for_cancellation(power, half - j)?;
        let term = if coeff == 1 {
            build_cos_multiple_for_cancellation(ctx, arg, i64::from(2 * j))
        } else {
            build_scaled_cos_multiple_for_cancellation(ctx, arg, i64::from(2 * j), coeff)
        };

        numerator = match trig_fn {
            BuiltinFn::Sin if j % 2 == 1 => ctx.add(Expr::Sub(numerator, term)),
            BuiltinFn::Sin | BuiltinFn::Cos => ctx.add(Expr::Add(numerator, term)),
            _ => unreachable!("only sin/cos are valid here"),
        };
    }

    let denominator = ctx.num(denominator);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn build_sin_cos_square_product_reduction_rewritten_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let one = ctx.num(1);
    let cos_quadruple = build_cos_multiple_for_cancellation(ctx, arg, 4);
    let numerator = ctx.add(Expr::Sub(one, cos_quadruple));
    let eight = ctx.num(8);
    ctx.add(Expr::Div(numerator, eight))
}

fn try_rewrite_trig_power_reduction_expr_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(arg) = extract_trig_square_product_same_arg_for_cancellation(ctx, expr) {
        return Some(build_sin_cos_square_product_reduction_rewritten_for_cancellation(ctx, arg));
    }

    let (coeff, trig_fn, arg, power) = extract_scaled_trig_even_power_for_cancellation(ctx, expr)?;
    let rewritten =
        build_trig_power_reduction_rewritten_for_cancellation(ctx, arg, power, trig_fn)?;
    Some(apply_numeric_scale_for_cancellation(ctx, &coeff, rewritten))
}

pub(super) fn maybe_trig_power_reduction_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    AddView::from_expr(ctx, expr)
        .terms
        .iter()
        .copied()
        .any(|(term_expr, _term_sign)| {
            extract_trig_even_power_for_cancellation(ctx, term_expr).is_some()
                || extract_trig_square_product_same_arg_for_cancellation(ctx, term_expr).is_some()
        })
}

pub(super) fn extract_trig_linear_multiple_term_for_fast_recursive_identity(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, i64, cas_ast::ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let trig_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        BuiltinFn::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        BuiltinFn::Cos
    } else {
        return None;
    };

    let (coeff, base) = split_linear_angle_term_for_phase_shift_cancellation(ctx, args[0]);
    if !coeff.is_integer() {
        return None;
    }
    let multiple = coeff.to_integer().to_i64()?;
    (multiple >= 1).then_some((trig_fn, multiple, base))
}

pub(super) fn build_trig_sine_product_cubic_polynomial(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let cos_cubed = ctx.add(Expr::Pow(cos_arg, three));
    let linear = smart_mul(ctx, four, cos_arg);
    let cubic = smart_mul(ctx, four, cos_cubed);
    let base = ctx.add(Expr::Sub(linear, cubic));
    build_scaled_expr(ctx, scale, base)
}

pub(super) fn build_trig_sine_product_cosine_cubic_target(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let scaled = build_trig_sine_product_cubic_polynomial(ctx, scale, arg);
    run_default_simplify(ctx, scaled)
}

pub(super) fn maybe_trig_square_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
}

pub(super) fn extract_trig_binomial_square_identity_data(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, bool)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }

    let (left, right, is_sum) = match ctx.get(*base) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };

    let lhs = extract_sin_or_cos_linear_term_for_phase_shift(ctx, left)?;
    let rhs = extract_sin_or_cos_linear_term_for_phase_shift(ctx, right)?;
    if compare_expr(ctx, lhs.1, rhs.1) != Ordering::Equal {
        return None;
    }

    let trig_kinds = [lhs.0, rhs.0];
    if !trig_kinds.contains(&BuiltinFn::Sin) || !trig_kinds.contains(&BuiltinFn::Cos) {
        return None;
    }

    Some((lhs.1, is_sum))
}

pub(super) fn build_trig_binomial_square_target(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> cas_ast::ExprId {
    let double_angle = build_trig_square_double_angle_term(ctx, arg);
    let one = ctx.num(1);
    if is_sum {
        ctx.add(Expr::Add(one, double_angle))
    } else {
        ctx.add(Expr::Sub(one, double_angle))
    }
}

fn build_trig_binomial_square_expansion(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> cas_ast::ExprId {
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let two = ctx.num(2);
    let sin_sq = ctx.add(Expr::Pow(sin_arg, two));
    let cos_sq = ctx.add(Expr::Pow(cos_arg, two));
    let sin_cos = smart_mul(ctx, sin_arg, cos_arg);
    let cross = smart_mul(ctx, two, sin_cos);
    let partial = if is_sum {
        ctx.add(Expr::Add(sin_sq, cross))
    } else {
        ctx.add(Expr::Sub(sin_sq, cross))
    };
    ctx.add(Expr::Add(partial, cos_sq))
}

fn build_trig_binomial_square_reduced(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> cas_ast::ExprId {
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let two = ctx.num(2);
    let sin_cos = smart_mul(ctx, sin_arg, cos_arg);
    let cross = smart_mul(ctx, two, sin_cos);
    let one = ctx.num(1);
    if is_sum {
        ctx.add(Expr::Add(one, cross))
    } else {
        ctx.add(Expr::Sub(one, cross))
    }
}

fn build_trig_square_zero_rewrite(
    ctx: &mut cas_ast::Context,
    square_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> Rewrite {
    let expanded = build_trig_binomial_square_expansion(ctx, arg, is_sum);
    let reduced = build_trig_binomial_square_reduced(ctx, arg, is_sum);
    let expanded_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expanded
        }
    );
    let reduced_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: reduced
        }
    );
    let target_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: target_expr
        }
    );

    Rewrite::with_local(ctx.num(0), "Trig Square Identity", square_expr, target_expr)
        .substep(
            "Expandir el binomio",
            vec![format!("Se obtiene {expanded_display}.")],
        )
        .substep(
            "Usar sin(u)^2 + cos(u)^2 = 1",
            vec![format!("Así queda {reduced_display}.")],
        )
        .substep(
            "Usar 2 · sin(u) · cos(u) = sin(2u)",
            vec![format!("La expresión se convierte en {target_display}.")],
        )
        .substep(
            "Cancelar términos iguales",
            vec![
                "Tras reconocer la misma identidad en el otro lado, toda la expresión se anula."
                    .to_string(),
            ],
        )
}

pub(super) fn build_direct_trig_square_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> Rewrite {
    let expanded = build_trig_binomial_square_expansion(ctx, arg, is_sum);
    let reduced = build_trig_binomial_square_reduced(ctx, arg, is_sum);
    let expanded_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expanded
        }
    );
    let reduced_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: reduced
        }
    );
    let target_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: target_expr
        }
    );

    Rewrite::with_local(ctx.num(0), "Trig Square Identity", lhs_core, rhs_core)
        .substep(
            "Expandir el binomio",
            vec![format!("Se obtiene {expanded_display}.")],
        )
        .substep(
            "Usar sin(u)^2 + cos(u)^2 = 1",
            vec![format!("Así queda {reduced_display}.")],
        )
        .substep(
            "Usar 2 · sin(u) · cos(u) = sin(2u)",
            vec![format!("La expresión se convierte en {target_display}.")],
        )
        .substep(
            "Cancelar términos iguales",
            vec![
                "Tras reconocer la misma identidad en el otro lado, toda la expresión se anula."
                    .to_string(),
            ],
        )
}

pub(super) fn try_build_exact_trig_square_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) = view.terms[index];
        let Some((arg, is_sum)) = extract_trig_binomial_square_identity_data(ctx, term_expr) else {
            continue;
        };
        let target_expr = build_trig_binomial_square_target(ctx, arg, is_sum);
        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

        let nested_default_simplify = default_simplify_nesting_depth() > 0;
        let matches = match term_sign {
            Sign::Pos => {
                expr_matches_negation_for_cancellation(ctx, remaining_expr, target_expr)
                    || (!nested_default_simplify
                        && expr_matches_negation_after_default_simplify(
                            ctx,
                            remaining_expr,
                            target_expr,
                        ))
            }
            Sign::Neg => {
                exprs_match_for_cancellation(ctx, remaining_expr, target_expr)
                    || (!nested_default_simplify
                        && exprs_match_after_default_simplify(ctx, remaining_expr, target_expr))
            }
        };
        if matches {
            return Some(build_trig_square_zero_rewrite(
                ctx,
                term_expr,
                target_expr,
                arg,
                is_sum,
            ));
        }
    }

    None
}

pub(super) fn matches_direct_trig_binomial_square_zero_identity(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    try_build_exact_trig_square_zero_scope_rewrite(ctx, expr).is_some()
}

pub(super) fn try_build_exact_trig_equivalence_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let nested_default_simplify = default_simplify_nesting_depth() > 0;
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    if let Some(rewrite) = try_build_exact_trig_pythagorean_zero_scope_rewrite(ctx, expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(ctx, expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_exact_trig_product_to_sum_sin_sin_zero_scope_rewrite(ctx, expr)
    {
        return Some(rewrite);
    }

    if view.terms.len() == 2 {
        return try_build_exact_trig_equivalence_two_term_zero_scope_rewrite(ctx, expr);
    }

    let allow_default_simplify_fallback = !nested_default_simplify
        && !matches_structural_trig_product_to_sum_sin_sin_three_term_family(ctx, expr);

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
                        try_rewrite_exact_trig_equivalence_for_cancellation(ctx, candidate_focus)
                    else {
                        continue;
                    };

                    let adjusted_rewritten = apply_sign_to_expr(ctx, focus_sign, rewritten);
                    let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
                    let distributed_neg_adjusted_rewritten =
                        negate_additive_scope_expr(ctx, adjusted_rewritten);
                    let allow_candidate_default_simplify = allow_default_simplify_fallback
                        && !(remaining_terms.len() == 1
                            && extract_exact_double_sine_product_args_from_signed_expr(
                                ctx,
                                candidate_focus,
                            )
                            .is_some());
                    if expr_matches_negation_for_cancellation(
                        ctx,
                        adjusted_rewritten,
                        remaining_expr,
                    ) || (allow_candidate_default_simplify
                        && expr_matches_negation_after_default_simplify(
                            ctx,
                            adjusted_rewritten,
                            remaining_expr,
                        ))
                        || (allow_candidate_default_simplify
                            && exprs_match_after_default_simplify(
                                ctx,
                                neg_adjusted_rewritten,
                                remaining_expr,
                            ))
                        || (allow_candidate_default_simplify
                            && exprs_match_after_default_simplify(
                                ctx,
                                distributed_neg_adjusted_rewritten,
                                remaining_expr,
                            ))
                        || (allow_candidate_default_simplify
                            && additive_scopes_match_after_default_simplify(
                                ctx,
                                distributed_neg_adjusted_rewritten,
                                remaining_expr,
                            ))
                    {
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
                                    "Tras aplicar la identidad trigonométrica, el resto de la expresión es el opuesto y toda la expresión se anula."
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

fn try_build_exact_trig_equivalence_two_term_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let nested_default_simplify = default_simplify_nesting_depth() > 0;
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (focus_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let neg_focus = ctx.add(Expr::Neg(focus_expr));
        for (candidate_focus, focus_sign) in [(focus_expr, 1_i64), (neg_focus, -1_i64)] {
            let Some((rewritten, description)) =
                try_rewrite_exact_trig_equivalence_for_cancellation(ctx, candidate_focus)
            else {
                continue;
            };

            let adjusted_rewritten = apply_sign_to_expr(ctx, focus_sign, rewritten);
            if exprs_match_for_cancellation(ctx, adjusted_rewritten, target_expr)
                || (!nested_default_simplify
                    && exprs_match_after_default_simplify(ctx, adjusted_rewritten, target_expr))
            {
                return Some(
                    Rewrite::with_local(ctx.num(0), description, focus_expr, adjusted_rewritten)
                        .substep(
                            "Cancelar términos iguales",
                            vec![
                                "Tras aplicar la identidad trigonométrica, ambos lados coinciden y la diferencia se anula."
                                    .to_string(),
                            ],
                        ),
                );
            }
        }
    }

    None
}

pub(super) fn classify_exact_cos_sum_or_diff_term_for_sin_sin_zero_scope(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    lhs_arg: cas_ast::ExprId,
    rhs_arg: cas_ast::ExprId,
) -> Option<ExactSinSinProductToSumCosKind> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        return None;
    }

    match ctx.get(args[0]) {
        Expr::Add(left, right) => args_match_as_multiset(ctx, *left, *right, lhs_arg, rhs_arg)
            .then_some(ExactSinSinProductToSumCosKind::Sum),
        Expr::Sub(left, right) => {
            let direct = compare_expr(ctx, *left, lhs_arg) == Ordering::Equal
                && compare_expr(ctx, *right, rhs_arg) == Ordering::Equal;
            let reversed = compare_expr(ctx, *left, rhs_arg) == Ordering::Equal
                && compare_expr(ctx, *right, lhs_arg) == Ordering::Equal;
            (direct || reversed).then_some(ExactSinSinProductToSumCosKind::Diff)
        }
        _ => None,
    }
}

pub(super) fn try_build_direct_cos_product_telescoping_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite_match) = crate::try_rewrite_cos_product_telescoping_expr(ctx, source)
        else {
            continue;
        };

        let rewritten_normalized =
            cas_math::canonical_forms::normalize_core(ctx, rewrite_match.rewritten);
        let target_normalized = cas_math::canonical_forms::normalize_core(ctx, target);
        if exprs_match_for_cancellation(ctx, rewrite_match.rewritten, target)
            || exprs_equal_up_to_fraction_parts_for_cancellation(
                ctx,
                rewrite_match.rewritten,
                target,
            )
            || compare_expr(ctx, rewritten_normalized, target_normalized) == Ordering::Equal
            || exprs_equal_up_to_fraction_parts_for_cancellation(
                ctx,
                rewritten_normalized,
                target_normalized,
            )
        {
            return Some(
                Rewrite::with_local(
                    ctx.num(0),
                    "Apply Morrie's law",
                    source,
                    rewrite_match.rewritten,
                )
                .requires(crate::ImplicitCondition::NonZero(
                    rewrite_match.assume_nonzero_expr,
                )),
            );
        }
    }

    None
}

pub(crate) fn try_build_direct_trig_power_reduction_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let matches_trig_power_reduction_form =
        |ctx: &mut cas_ast::Context, lhs: cas_ast::ExprId, rhs: cas_ast::ExprId| {
            if compare_expr(ctx, lhs, rhs) == Ordering::Equal
                || exprs_equal_up_to_add_term_order(ctx, lhs, rhs)
                || exprs_equal_up_to_add_term_multiset_for_cancellation(ctx, lhs, rhs)
                || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs, rhs)
                || exprs_equal_up_to_same_denominator(ctx, lhs, rhs)
            {
                return true;
            }

            let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
            let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
            compare_expr(ctx, lhs_normalized, rhs_normalized) == Ordering::Equal
                || exprs_equal_up_to_add_term_order(ctx, lhs_normalized, rhs_normalized)
                || exprs_equal_up_to_add_term_multiset_for_cancellation(
                    ctx,
                    lhs_normalized,
                    rhs_normalized,
                )
                || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs_normalized, rhs_normalized)
                || exprs_equal_up_to_same_denominator(ctx, lhs_normalized, rhs_normalized)
        };

    if let Some(rewritten) = try_rewrite_trig_power_reduction_expr_for_cancellation(ctx, lhs_core) {
        if matches_trig_power_reduction_form(ctx, rewritten, rhs_core) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Power Reduction Identity",
                lhs_core,
                rhs_core,
            ));
        }
    }
    if let Some(rewritten) = try_rewrite_trig_power_reduction_expr_for_cancellation(ctx, rhs_core) {
        if matches_trig_power_reduction_form(ctx, rewritten, lhs_core) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Power Reduction Identity",
                lhs_core,
                rhs_core,
            ));
        }
    }
    None
}

pub(super) fn try_rewrite_trig_cos_diff_sin_diff_quotient_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let (num, den) = as_div(ctx, expr)?;
    let (cos_l, cos_r) = extract_trig_two_term_diff(ctx, num, "cos")?;
    let (sin_l, sin_r) = extract_trig_two_term_diff(ctx, den, "sin")?;

    let same_order = compare_expr(ctx, cos_l, sin_l) == Ordering::Equal
        && compare_expr(ctx, cos_r, sin_r) == Ordering::Equal;
    let reversed_order = compare_expr(ctx, cos_l, sin_r) == Ordering::Equal
        && compare_expr(ctx, cos_r, sin_l) == Ordering::Equal;
    if same_order == reversed_order {
        return None;
    }

    let avg = build_avg_with_simplifier(ctx, cos_l, cos_r, crate::collect::collect);
    let tan_avg = ctx.call_builtin(BuiltinFn::Tan, vec![avg]);
    let rewritten = if reversed_order {
        tan_avg
    } else {
        ctx.add(Expr::Neg(tan_avg))
    };
    Some((rewritten, den))
}

pub(super) fn try_build_direct_trig_reciprocal_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(numerator, denominator) = ctx.get(source).clone() else {
            continue;
        };
        if extract_i64_integer(ctx, numerator) != Some(1) {
            continue;
        }

        for (builtin, reciprocal_name) in [
            (BuiltinFn::Cos, BuiltinFn::Sec),
            (BuiltinFn::Sin, BuiltinFn::Csc),
        ] {
            let Some(base_arg) = extract_unary_builtin_arg(ctx, denominator, builtin) else {
                continue;
            };
            let reciprocal = ctx.call_builtin(reciprocal_name, vec![base_arg]);
            if exprs_match_for_cancellation(ctx, reciprocal, target)
                || exprs_match_after_default_simplify(ctx, reciprocal, target)
            {
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    "Reciprocal Quotient Identity",
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    None
}

pub(crate) fn try_build_direct_trig_sine_product_cubic_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((scale, arg)) = extract_scaled_double_sine_product_for_cancellation(ctx, source)
        else {
            continue;
        };

        let rewritten = build_trig_sine_product_cosine_cubic_target(ctx, scale, arg);
        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Product-to-Sum and Triple-Angle Identity",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

fn try_match_tan_cot_sum_arg(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    for (first, second) in [(terms[0].0, terms[1].0), (terms[1].0, terms[0].0)] {
        let Some(tan_arg) = extract_unary_builtin_arg(ctx, first, BuiltinFn::Tan) else {
            continue;
        };
        let Some(cot_arg) = extract_unary_builtin_arg(ctx, second, BuiltinFn::Cot) else {
            continue;
        };
        if exprs_match_for_cancellation(ctx, tan_arg, cot_arg) {
            return Some(tan_arg);
        }
    }

    None
}

fn try_match_sec_csc_product_arg(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (first, second) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        let Some(sec_arg) = extract_unary_builtin_arg(ctx, first, BuiltinFn::Sec) else {
            continue;
        };
        let Some(csc_arg) = extract_unary_builtin_arg(ctx, second, BuiltinFn::Csc) else {
            continue;
        };
        if exprs_match_for_cancellation(ctx, sec_arg, csc_arg) {
            return Some(sec_arg);
        }
    }

    None
}

pub(super) fn try_build_small_tan_cot_product_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (product_side, one_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if extract_i64_integer(ctx, one_side) != Some(1) {
            continue;
        }

        let factors = flatten_mul_chain(ctx, product_side);
        if factors.len() != 2 {
            continue;
        }

        for (first, second) in [(factors[0], factors[1]), (factors[1], factors[0])] {
            let Some(tan_arg) = extract_unary_builtin_arg(ctx, first, BuiltinFn::Tan) else {
                continue;
            };
            let Some(cot_arg) = extract_unary_builtin_arg(ctx, second, BuiltinFn::Cot) else {
                continue;
            };
            if !exprs_match_for_cancellation(ctx, tan_arg, cot_arg) {
                continue;
            }

            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![tan_arg]);
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![tan_arg]);
            return Some(
                Rewrite::with_local(zero, "Reciprocal Quotient Identity", product_side, one)
                    .requires(crate::ImplicitCondition::NonZero(sin_arg))
                    .requires(crate::ImplicitCondition::NonZero(cos_arg))
                    .substep(
                        "Usar tan(u)·cot(u) = 1",
                        vec![
                            "Tangente y cotangente recíprocas se cancelan exactamente y dejan 1."
                                .to_string(),
                        ],
                    ),
            );
        }
    }

    None
}

pub(super) fn try_build_small_tan_cot_sec_csc_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let zero = ctx.num(0);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (sum_side, product_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(sum_arg) = try_match_tan_cot_sum_arg(ctx, sum_side) else {
            continue;
        };
        let Some(product_arg) = try_match_sec_csc_product_arg(ctx, product_side) else {
            continue;
        };
        if !exprs_match_for_cancellation(ctx, sum_arg, product_arg) {
            continue;
        }

        let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![sum_arg]);
        let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![sum_arg]);
        return Some(
            Rewrite::with_local(zero, "Reciprocal Quotient Identity", sum_side, product_side)
                .requires(crate::ImplicitCondition::NonZero(sin_arg))
                .requires(crate::ImplicitCondition::NonZero(cos_arg))
                .substep(
                    "Usar tan(u) + cot(u) = sec(u)csc(u)",
                    vec![
                        "La suma de tangente y cotangente coincide exactamente con el producto secante por cosecante."
                            .to_string(),
                    ],
                ),
        );
    }

    None
}

pub(super) fn maybe_unit_fraction_trig_denominator_equivalence_zero_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, expr) else {
        return false;
    };
    let Some(lhs_denominator) = extract_unit_fraction_denominator(ctx, lhs_core) else {
        return false;
    };
    let Some(rhs_denominator) = extract_unit_fraction_denominator(ctx, rhs_core) else {
        return false;
    };

    unit_fraction_trig_denominator_supported(ctx, lhs_denominator)
        && unit_fraction_trig_denominator_supported(ctx, rhs_denominator)
}

fn unit_fraction_trig_denominator_supported(
    ctx: &cas_ast::Context,
    denominator: cas_ast::ExprId,
) -> bool {
    matches!(ctx.get(denominator), Expr::Add(_, _) | Expr::Sub(_, _))
        && !expr_contains_division_node(ctx, denominator)
        && AddView::from_expr(ctx, denominator).terms.len() <= 5
        && additive_has_variable_scaled_direct_trig_or_hyperbolic_term(ctx, denominator, 2)
}

pub(super) fn try_build_unit_fraction_trig_denominator_equivalence_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;
    let lhs_denominator = extract_unit_fraction_denominator(ctx, lhs_core)?;
    let rhs_denominator = extract_unit_fraction_denominator(ctx, rhs_core)?;

    if !unit_fraction_trig_denominator_supported(ctx, lhs_denominator)
        || !unit_fraction_trig_denominator_supported(ctx, rhs_denominator)
    {
        return None;
    }

    let denominator_difference = ctx.add(Expr::Sub(lhs_denominator, rhs_denominator));
    if !is_zero_after_default_simplify(ctx, denominator_difference) {
        return None;
    }

    Some(
        Rewrite::with_local(ctx.num(0), "Subtract Fractions", expr, ctx.num(0))
            .requires(crate::ImplicitCondition::NonZero(lhs_denominator))
            .requires(crate::ImplicitCondition::NonZero(rhs_denominator))
            .substep(
                "Comparar denominadores equivalentes",
                vec![
                    "Los denominadores de las fracciones recíprocas se simplifican a la misma expresión, por lo que las fracciones se cancelan exactamente."
                        .to_string(),
                ],
            ),
    )
}

pub(super) fn find_first_tan_arg(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if ctx.is_builtin(fn_id, BuiltinFn::Tan) && args.len() == 1 => {
            Some(args[0])
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
            find_first_tan_arg(ctx, lhs).or_else(|| find_first_tan_arg(ctx, rhs))
        }
        Expr::Pow(base, exp) => {
            find_first_tan_arg(ctx, base).or_else(|| find_first_tan_arg(ctx, exp))
        }
        Expr::Neg(inner) | Expr::Hold(inner) => find_first_tan_arg(ctx, inner),
        Expr::Matrix { data, .. } => data
            .into_iter()
            .find_map(|entry| find_first_tan_arg(ctx, entry)),
        _ => None,
    }
}

pub(super) fn try_build_direct_recursive_trig_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_recursive_trig_expansion_expr(ctx, source) else {
            continue;
        };
        let description = if rewrite.desc.starts_with("sin(") || rewrite.desc.starts_with("cos(") {
            "Angle Sum/Diff Identity"
        } else {
            "Recursive Trig Expansion"
        };
        if exprs_match_for_cancellation(ctx, rewrite.rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewrite.rewritten, target)
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

pub(super) fn try_match_direct_trig_ratio_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<(&'static str, cas_ast::ExprId, &'static str, &'static str)> {
    let (numerator, denominator) = as_div(ctx, source)?;

    for (numerator_builtin, denominator_builtin, target_builtin, step_text) in [
        (
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            "Usar sin(u)/cos(u) = tan(u)",
        ),
        (
            BuiltinFn::Cos,
            BuiltinFn::Sin,
            BuiltinFn::Cot,
            "Usar cos(u)/sin(u) = cot(u)",
        ),
    ] {
        let Some(numerator_arg) = extract_unary_builtin_arg(ctx, numerator, numerator_builtin)
        else {
            continue;
        };
        let Some(denominator_arg) =
            extract_unary_builtin_arg(ctx, denominator, denominator_builtin)
        else {
            continue;
        };
        if !exprs_match_with_local_default_simplify(ctx, numerator_arg, denominator_arg) {
            continue;
        }

        let Some(target_arg) = extract_unary_builtin_arg(ctx, target, target_builtin) else {
            continue;
        };
        if !exprs_match_with_local_default_simplify(ctx, numerator_arg, target_arg) {
            continue;
        }

        return Some((
            "Trigonometric Quotient Identity",
            denominator,
            step_text,
            "Al coincidir los argumentos, el cociente trigonométrico se reconoce sin caer al compare caro.",
        ));
    }

    None
}

pub(super) fn try_build_direct_trig_ratio_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((description, denominator, step_text, detail_text)) =
            try_match_direct_trig_ratio_equivalence(ctx, source, target)
                .or_else(|| try_match_half_angle_tan_equivalence(ctx, source, target))
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(ctx.num(0), description, lhs_core, rhs_core)
                .requires(crate::ImplicitCondition::NonZero(denominator))
                .substep(step_text, vec![detail_text.to_string()]),
        );
    }

    None
}

pub(super) fn try_build_same_denominator_tail_trig_ratio_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((description, denominator, step_text, detail_text)) =
            try_match_direct_trig_ratio_equivalence(ctx, source, target)
                .or_else(|| try_match_half_angle_tan_equivalence(ctx, source, target))
        else {
            continue;
        };

        return Some(
            Rewrite::with_local(ctx.num(0), description, lhs_core, rhs_core)
                .requires(crate::ImplicitCondition::NonZero(denominator))
                .substep(step_text, vec![detail_text.to_string()]),
        );
    }

    None
}

pub(super) fn reject_noncall_vs_surface_symbolic_trig_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    let lhs_surface = extract_surface_scaled_trig_term_for_phase_shift(ctx, lhs_core);
    let rhs_surface = extract_surface_scaled_trig_term_for_phase_shift(ctx, rhs_core);

    let ((_, trig_arg, _, _), noncall_expr) = match (lhs_surface, rhs_surface) {
        (Some(surface), None) => (surface, rhs_core),
        (None, Some(surface)) => (surface, lhs_core),
        _ => return None,
    };

    if expr_contains_any_function_call(ctx, noncall_expr)
        || expr_contains_any_function_call(ctx, trig_arg)
        || !expr_contains_symbolic_atom_for_cancellation(ctx, trig_arg)
    {
        return None;
    }

    Some(false)
}

pub(super) fn reject_surface_plain_cross_trig_pair_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    let (lhs_fn, lhs_arg, _, _) = extract_surface_scaled_trig_term_for_phase_shift(ctx, lhs_core)?;
    let (rhs_fn, rhs_arg, _, _) = extract_surface_scaled_trig_term_for_phase_shift(ctx, rhs_core)?;

    if lhs_fn == rhs_fn {
        return None;
    }

    let lhs_shifted = extract_supported_phase_shift_argument_for_cancellation(ctx, lhs_fn, lhs_arg)
        .is_some()
        || extract_general_phase_shift_argument_for_cancellation(ctx, lhs_arg).is_some();
    let rhs_shifted = extract_supported_phase_shift_argument_for_cancellation(ctx, rhs_fn, rhs_arg)
        .is_some()
        || extract_general_phase_shift_argument_for_cancellation(ctx, rhs_arg).is_some();
    if lhs_shifted || rhs_shifted {
        return None;
    }

    if expr_contains_pi_constant(ctx, lhs_arg)
        || expr_contains_pi_constant(ctx, rhs_arg)
        || !expr_contains_symbolic_atom_for_cancellation(ctx, lhs_arg)
        || !expr_contains_symbolic_atom_for_cancellation(ctx, rhs_arg)
    {
        return None;
    }

    Some(false)
}

fn extract_sin_or_cos_power_factor_for_default_simplify_reject(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, i64)> {
    if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, expr) {
        return Some((trig_fn, arg, 1));
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let (trig_fn, arg) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, base)?;
    let power = small_positive_integer_value(ctx, exp)?;
    (power > 1).then_some((trig_fn, arg, power))
}

fn extract_scaled_surface_trig_power_for_default_simplify_reject(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, i64)> {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    let factors = flatten_mul_chain(ctx, expr);
    let mut trig_power = None;

    for factor in factors {
        if let Expr::Number(value) = ctx.get(factor) {
            if value.is_zero() {
                return None;
            }
            continue;
        }

        let extracted = extract_sin_or_cos_power_factor_for_default_simplify_reject(ctx, factor)?;
        if trig_power.replace(extracted).is_some() {
            return None;
        }
    }

    let (trig_fn, arg, power) = trig_power?;
    if expr_contains_pi_constant(ctx, arg)
        || expr_contains_any_function_call(ctx, arg)
        || !expr_contains_symbolic_atom_for_cancellation(ctx, arg)
    {
        return None;
    }

    Some((trig_fn, arg, power))
}

pub(super) fn reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    for (trig_expr, atom_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if extract_scaled_surface_trig_power_for_default_simplify_reject(ctx, trig_expr).is_none() {
            continue;
        }
        let atom_expr = strip_unit_negation_for_phase_shift(ctx, atom_expr).unwrap_or(atom_expr);
        if matches!(ctx.get(atom_expr), Expr::Number(_)) {
            return Some(false);
        }
    }

    None
}

pub(super) fn reject_plain_surface_trig_power_gap_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    if let (Some((lhs_fn, lhs_arg, lhs_power)), Some((rhs_fn, rhs_arg, rhs_power))) = (
        extract_scaled_surface_trig_power_for_default_simplify_reject(ctx, lhs_core),
        extract_scaled_surface_trig_power_for_default_simplify_reject(ctx, rhs_core),
    ) {
        if lhs_power != rhs_power
            && lhs_fn == rhs_fn
            && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
        {
            return Some(false);
        }
    }

    for (plain_expr, power_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((plain_fn, plain_arg)) =
            extract_sin_or_cos_linear_term_for_phase_shift(ctx, plain_expr)
        else {
            continue;
        };
        let Expr::Pow(power_base, power_exp) = ctx.get(power_expr).clone() else {
            continue;
        };
        let Some((power_fn, power_arg)) =
            extract_sin_or_cos_linear_term_for_phase_shift(ctx, power_base)
        else {
            continue;
        };
        let Some(power) = small_positive_integer_value(ctx, power_exp) else {
            continue;
        };

        if power < 2
            || plain_fn != power_fn
            || compare_expr(ctx, plain_arg, power_arg) != Ordering::Equal
        {
            continue;
        }

        if expr_contains_pi_constant(ctx, plain_arg)
            || expr_contains_any_function_call(ctx, plain_arg)
            || !expr_contains_symbolic_atom_for_cancellation(ctx, plain_arg)
        {
            continue;
        }

        return Some(false);
    }

    None
}
