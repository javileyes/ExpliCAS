//! Orquestador: familia `hyperbolic` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

pub(super) fn extract_plain_sinh_or_cosh_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
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

pub(super) fn extract_plain_sinh_or_cosh_pow2_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }
    extract_plain_sinh_or_cosh_arg_root(ctx, *base)
}

pub(super) fn build_tanh_pythagorean_target_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let two = ctx.num(2);
    let denominator = ctx.add(Expr::Pow(cosh_arg, two));
    ctx.add(Expr::Div(one, denominator))
}

pub(super) fn extract_direct_tanh_pythagorean_identity_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut tanh_sq_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        if term_sign != Sign::Neg {
            return None;
        }
        let Expr::Pow(base, exponent) = ctx.get(term_expr) else {
            return None;
        };
        if extract_i64_integer(ctx, *exponent)? != 2 {
            return None;
        }
        let Expr::Function(fn_id, args) = ctx.get(*base) else {
            return None;
        };
        if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 || tanh_sq_arg.is_some() {
            return None;
        }
        tanh_sq_arg = Some(args[0]);
    }

    saw_positive_one.then_some(tanh_sq_arg?).or(None)
}

fn extract_direct_tanh_pythagorean_target_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *numerator)? != 1 {
        return None;
    }
    let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_pow2_arg_root(ctx, *denominator)
    else {
        return None;
    };
    Some(arg)
}

pub(super) fn extract_direct_hyperbolic_half_angle_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let base = extract_half_scaled_base_root(ctx, expr)?;
    let view = AddView::from_expr(ctx, base);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cosh_arg = None;
    let mut one_sign = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if one_sign.is_some() {
                return None;
            }
            one_sign = Some(term_sign);
            continue;
        }

        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
        else {
            return None;
        };
        if term_sign != Sign::Pos || cosh_arg.is_some() {
            return None;
        }
        cosh_arg = Some(arg);
    }

    let hyperbolic_fn = match one_sign? {
        Sign::Pos => BuiltinFn::Cosh,
        Sign::Neg => BuiltinFn::Sinh,
    };
    Some((hyperbolic_fn, cosh_arg?))
}

pub(super) fn matches_direct_hyperbolic_half_angle_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_hyperbolic_half_angle_squares_expr(ctx, source) else {
            continue;
        };
        let Some((rewritten_fn, rewritten_arg)) =
            extract_direct_hyperbolic_half_angle_square_target_root(ctx, rewrite.rewritten)
        else {
            continue;
        };
        let Some((target_fn, target_arg)) =
            extract_direct_hyperbolic_half_angle_square_target_root(ctx, target)
        else {
            continue;
        };
        if rewritten_fn == target_fn
            && compare_expr(ctx, rewritten_arg, target_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

pub(super) fn build_plain_hyperbolic_half_angle_pow2_root(
    ctx: &mut Context,
    hyperbolic_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let half_arg = ctx.add(Expr::Div(arg, two));
    let hyperbolic_expr = ctx.call_builtin(hyperbolic_fn, vec![half_arg]);
    ctx.add(Expr::Pow(hyperbolic_expr, two))
}

fn build_direct_hyperbolic_half_angle_square_target_root(
    ctx: &mut Context,
    hyperbolic_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let numerator = match hyperbolic_fn {
        BuiltinFn::Sinh => ctx.add(Expr::Sub(cosh_arg, one)),
        BuiltinFn::Cosh => ctx.add(Expr::Add(cosh_arg, one)),
        _ => unreachable!("only sinh/cosh half-angle squares are supported"),
    };
    ctx.add(Expr::Div(numerator, two))
}

fn extract_scaled_hyperbolic_sinh_cosh_product_half_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sinh_arg = None;
    let mut cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        match extract_plain_sinh_or_cosh_arg_root(ctx, factor) {
            Some((BuiltinFn::Sinh, arg)) => {
                if sinh_arg.is_some() {
                    return None;
                }
                sinh_arg = Some(arg);
            }
            Some((BuiltinFn::Cosh, arg)) => {
                if cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(arg);
            }
            _ => return None,
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((sinh_arg?, cosh_arg?))
}

fn extract_scaled_hyperbolic_cosh_cosh_product_half_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_cosh_arg = None;
    let mut second_cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        if first_cosh_arg.is_none() {
            first_cosh_arg = Some(arg);
        } else if second_cosh_arg.is_none() {
            second_cosh_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_cosh_arg?, second_cosh_arg?))
}

fn extract_scaled_hyperbolic_sinh_sinh_product_half_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_sinh_arg = None;
    let mut second_sinh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Sinh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        if first_sinh_arg.is_none() {
            first_sinh_arg = Some(arg);
        } else if second_sinh_arg.is_none() {
            second_sinh_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_sinh_arg?, second_sinh_arg?))
}

pub(super) fn extract_plain_hyperbolic_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<((BuiltinFn, ExprId), (BuiltinFn, ExprId))> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let lhs = extract_plain_sinh_or_cosh_arg_root(ctx, factors[0])?;
    let rhs = extract_plain_sinh_or_cosh_arg_root(ctx, factors[1])?;
    Some((lhs, rhs))
}

pub(super) fn matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let sum_view = AddView::from_expr(ctx, sum_expr);
        if sum_view.terms.len() != 2 || !sum_view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let mut sum_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();
        let mut bad_sum = false;
        for (term_expr, _term_sign) in sum_view.terms {
            let Some((BuiltinFn::Sinh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
            else {
                bad_sum = true;
                break;
            };
            sum_args.push(arg);
        }
        if bad_sum || sum_args.len() != 2 {
            continue;
        }

        let Some((sinh_half_sum_arg, cosh_half_diff_arg)) =
            extract_scaled_hyperbolic_sinh_cosh_product_half_args_root(ctx, product_expr)
        else {
            continue;
        };

        let sum_expr = ctx.add(Expr::Add(sum_args[0], sum_args[1]));
        let half_sum = build_half_expr_root(ctx, sum_expr);
        if compare_expr(ctx, sinh_half_sum_arg, half_sum) != Ordering::Equal {
            continue;
        }

        let diff_ab_expr = ctx.add(Expr::Sub(sum_args[0], sum_args[1]));
        let half_diff_ab = build_half_expr_root(ctx, diff_ab_expr);
        let diff_ba_expr = ctx.add(Expr::Sub(sum_args[1], sum_args[0]));
        let half_diff_ba = build_half_expr_root(ctx, diff_ba_expr);
        if compare_expr(ctx, cosh_half_diff_arg, half_diff_ab) == Ordering::Equal
            || compare_expr(ctx, cosh_half_diff_arg, half_diff_ba) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let sum_view = AddView::from_expr(ctx, sum_expr);
        if sum_view.terms.len() != 2 || !sum_view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let mut sum_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();
        let mut bad_sum = false;
        for (term_expr, _term_sign) in sum_view.terms {
            let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
            else {
                bad_sum = true;
                break;
            };
            sum_args.push(arg);
        }
        if bad_sum || sum_args.len() != 2 {
            continue;
        }

        let Some((product_arg_a, product_arg_b)) =
            extract_scaled_hyperbolic_cosh_cosh_product_half_args_root(ctx, product_expr)
        else {
            continue;
        };

        let sum_expr = ctx.add(Expr::Add(sum_args[0], sum_args[1]));
        let half_sum = build_half_expr_root(ctx, sum_expr);
        let diff_ab_expr = ctx.add(Expr::Sub(sum_args[0], sum_args[1]));
        let half_diff_ab = build_half_expr_root(ctx, diff_ab_expr);
        let diff_ba_expr = ctx.add(Expr::Sub(sum_args[1], sum_args[0]));
        let half_diff_ba = build_half_expr_root(ctx, diff_ba_expr);

        let direct_order = compare_expr(ctx, product_arg_a, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_b, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_b, half_diff_ba) == Ordering::Equal);
        let swapped_order = compare_expr(ctx, product_arg_b, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_a, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_a, half_diff_ba) == Ordering::Equal);
        if direct_order || swapped_order {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (difference_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let diff_view = AddView::from_expr(ctx, difference_expr);
        if diff_view.terms.len() != 2 {
            continue;
        }

        let mut positive_cosh_arg = None;
        let mut negative_cosh_arg = None;
        let mut bad_diff = false;
        for (term_expr, term_sign) in diff_view.terms {
            let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
            else {
                bad_diff = true;
                break;
            };
            match term_sign {
                Sign::Pos if positive_cosh_arg.is_none() => positive_cosh_arg = Some(arg),
                Sign::Neg if negative_cosh_arg.is_none() => negative_cosh_arg = Some(arg),
                _ => {
                    bad_diff = true;
                    break;
                }
            }
        }
        if bad_diff {
            continue;
        }

        let (Some(lhs_arg), Some(rhs_arg)) = (positive_cosh_arg, negative_cosh_arg) else {
            continue;
        };
        let Some((product_arg_a, product_arg_b)) =
            extract_scaled_hyperbolic_sinh_sinh_product_half_args_root(ctx, product_expr)
        else {
            continue;
        };

        let sum_expr = ctx.add(Expr::Add(lhs_arg, rhs_arg));
        let half_sum = build_half_expr_root(ctx, sum_expr);
        let diff_ab_expr = ctx.add(Expr::Sub(lhs_arg, rhs_arg));
        let half_diff_ab = build_half_expr_root(ctx, diff_ab_expr);
        let diff_ba_expr = ctx.add(Expr::Sub(rhs_arg, lhs_arg));
        let half_diff_ba = build_half_expr_root(ctx, diff_ba_expr);

        let direct_order = compare_expr(ctx, product_arg_a, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_b, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_b, half_diff_ba) == Ordering::Equal);
        let swapped_order = compare_expr(ctx, product_arg_b, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_a, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_a, half_diff_ba) == Ordering::Equal);
        if direct_order || swapped_order {
            return true;
        }
    }

    false
}

pub(super) fn extract_direct_hyperbolic_exp_sum_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    use cas_math::expr_nary::{AddView, Sign};

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cosh_arg = None;
    let mut sinh_arg = None;
    let mut sinh_positive = None;

    for (term_expr, term_sign) in view.terms {
        match extract_plain_sinh_or_cosh_arg_root(ctx, term_expr) {
            Some((BuiltinFn::Cosh, arg)) => {
                if term_sign != Sign::Pos || cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(arg);
            }
            Some((BuiltinFn::Sinh, arg)) => {
                if sinh_arg.is_some() {
                    return None;
                }
                sinh_arg = Some(arg);
                sinh_positive = Some(term_sign == Sign::Pos);
            }
            _ => return None,
        }
    }

    let cosh_arg = cosh_arg?;
    let sinh_arg = sinh_arg?;
    if compare_expr(ctx, cosh_arg, sinh_arg) != Ordering::Equal {
        return None;
    }

    Some((cosh_arg, sinh_positive?))
}

pub(super) fn build_direct_hyperbolic_exp_sum_target_root(
    ctx: &mut Context,
    arg: ExprId,
    is_sum: bool,
) -> ExprId {
    let exp_arg = if is_sum { arg } else { ctx.add(Expr::Neg(arg)) };
    ctx.call_builtin(BuiltinFn::Exp, vec![exp_arg])
}

pub(super) fn matches_direct_hyperbolic_exp_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (hyper_expr, exp_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((arg, is_sum)) = extract_direct_hyperbolic_exp_sum_target_root(ctx, hyper_expr)
        else {
            continue;
        };
        let Some(exp_arg) = extract_exp_argument(ctx, exp_expr) else {
            continue;
        };
        let expected_arg = if is_sum { arg } else { ctx.add(Expr::Neg(arg)) };
        if compare_expr(ctx, exp_arg, expected_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (identity_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target_expr), Expr::Number(n) if n.is_one()) {
            continue;
        }

        let view = AddView::from_expr(ctx, identity_expr);
        if view.terms.len() != 2 {
            continue;
        }

        let mut cosh_arg = None;
        let mut sinh_arg = None;
        let mut valid = true;
        for (term_expr, sign) in view.terms {
            match (
                extract_plain_sinh_or_cosh_pow2_arg_root(ctx, term_expr),
                sign,
            ) {
                (Some((BuiltinFn::Cosh, arg)), Sign::Pos) => {
                    if cosh_arg.replace(arg).is_some() {
                        valid = false;
                        break;
                    }
                }
                (Some((BuiltinFn::Sinh, arg)), Sign::Neg) => {
                    if sinh_arg.replace(arg).is_some() {
                        valid = false;
                        break;
                    }
                }
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        let (Some(cosh_arg), Some(sinh_arg)) = (cosh_arg, sinh_arg) else {
            continue;
        };
        if valid && compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_triple_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((source_builtin, inner_arg)) =
            extract_rewritable_hyperbolic_triple_angle_source_root(ctx, source_expr)
        else {
            continue;
        };
        if matches_direct_hyperbolic_triple_angle_expansion_target_root(
            ctx,
            target_expr,
            source_builtin,
            inner_arg,
        ) {
            return true;
        }
        let Some(rewrite) = try_rewrite_hyperbolic_triple_angle(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 32
            && cas_ast::count_nodes(ctx, target_expr) <= 32
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn is_simple_hyperbolic_triple_angle_inner_root(ctx: &Context, expr: ExprId) -> bool {
    let is_atomic = |id: ExprId| {
        matches!(
            ctx.get(id),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
        )
    };

    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => true,
        Expr::Mul(lhs, rhs) => is_atomic(*lhs) && is_atomic(*rhs),
        Expr::Neg(inner) => is_atomic(*inner),
        _ => false,
    }
}

fn extract_rewritable_hyperbolic_triple_angle_source_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let (source_builtin, source_arg) = extract_plain_sinh_or_cosh_arg_root(ctx, expr)?;
    let inner_arg = extract_triple_angle_arg(ctx, source_arg)?;
    is_simple_hyperbolic_triple_angle_inner_root(ctx, inner_arg)
        .then_some((source_builtin, inner_arg))
}

fn matches_direct_hyperbolic_triple_angle_expansion_target_root(
    ctx: &Context,
    expr: ExprId,
    source_builtin: BuiltinFn,
    inner_arg: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut linear_coeff = None;
    let mut cubic_coeff = None;

    for (term_expr, term_sign) in view.terms {
        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }

        if let Some((term_builtin, term_arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, base) {
            if term_builtin != source_builtin
                || compare_expr(ctx, term_arg, inner_arg) != Ordering::Equal
                || linear_coeff.replace(coeff).is_some()
            {
                return false;
            }
            continue;
        }

        let Expr::Pow(pow_base, pow_exp) = ctx.get(base) else {
            return false;
        };
        if extract_i64_integer(ctx, *pow_exp) != Some(3) {
            return false;
        }
        let Some((term_builtin, term_arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, *pow_base)
        else {
            return false;
        };
        if term_builtin != source_builtin
            || compare_expr(ctx, term_arg, inner_arg) != Ordering::Equal
            || cubic_coeff.replace(coeff).is_some()
        {
            return false;
        }
    }

    let three = BigRational::from_integer(3.into());
    let four = BigRational::from_integer(4.into());
    match source_builtin {
        BuiltinFn::Sinh => linear_coeff == Some(three) && cubic_coeff == Some(four),
        BuiltinFn::Cosh => linear_coeff == Some(-three) && cubic_coeff == Some(four),
        _ => false,
    }
}

pub(super) fn matches_direct_hyperbolic_double_angle_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_hyperbolic_double_angle_sum(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_from_exp_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_recognize_hyperbolic_from_exp(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_tanh_to_sinh_cosh_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_tanh_to_sinh_cosh(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_tanh_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (identity_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(identity_arg) =
            extract_direct_tanh_pythagorean_identity_arg_root(ctx, identity_expr)
        else {
            continue;
        };
        let Some(target_arg) = extract_direct_tanh_pythagorean_target_root(ctx, target_expr) else {
            continue;
        };
        if compare_expr(ctx, identity_arg, target_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_tanh_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) = try_rewrite_tanh_double_angle_expansion(ctx, source) else {
            continue;
        };
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_recursive_hyperbolic_sinh_sum_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for single_index in 0..view.terms.len() {
        let (single_expr, single_sign) = view.terms[single_index];
        if single_sign != Sign::Pos {
            continue;
        }
        let Some((BuiltinFn::Sinh, _)) = extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let expanded_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, (term_expr, term_sign))| {
                (index != single_index).then_some((term_expr, term_sign))
            })
            .collect();
        if expanded_terms.len() != 2 || expanded_terms.iter().any(|(_, sign)| *sign != Sign::Neg) {
            continue;
        }

        let expanded_expr = build_signed_sum_expr_root(
            ctx,
            &[
                (expanded_terms[0].0, Sign::Pos),
                (expanded_terms[1].0, Sign::Pos),
            ],
        );
        if matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, single_expr, expanded_expr) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_recursive_hyperbolic_cosh_sum_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for single_index in 0..view.terms.len() {
        let (single_expr, single_sign) = view.terms[single_index];
        if single_sign != Sign::Pos {
            continue;
        }
        let Some((BuiltinFn::Cosh, _)) = extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let expanded_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, (term_expr, term_sign))| {
                (index != single_index).then_some((term_expr, term_sign))
            })
            .collect();
        if expanded_terms.len() != 2 || expanded_terms.iter().any(|(_, sign)| *sign != Sign::Neg) {
            continue;
        }

        let expanded_expr = build_signed_sum_expr_root(
            ctx,
            &[
                (expanded_terms[0].0, Sign::Pos),
                (expanded_terms[1].0, Sign::Pos),
            ],
        );
        if matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, single_expr, expanded_expr) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_exp_sum_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut cosh_arg = None;
    let mut sinh_arg = None;
    let mut exp_arg = None;

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            match extract_plain_sinh_or_cosh_arg_root(ctx, term_expr) {
                Some((BuiltinFn::Cosh, arg)) => cosh_arg = Some(arg),
                Some((BuiltinFn::Sinh, arg)) => sinh_arg = Some(arg),
                _ => return false,
            }
            continue;
        }

        if term_sign == Sign::Neg {
            if let Some(arg) = extract_exp_argument(ctx, term_expr) {
                exp_arg = Some(arg);
                continue;
            }
        }

        return false;
    }

    match (cosh_arg, sinh_arg, exp_arg) {
        (Some(cosh_arg), Some(sinh_arg), Some(exp_arg)) => {
            compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal
                && compare_expr(ctx, cosh_arg, exp_arg) == Ordering::Equal
        }
        _ => false,
    }
}

pub(super) fn matches_direct_hyperbolic_pythagorean_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut cosh_pos = None;
    let mut cosh_neg = None;
    let mut sinh_pos = None;
    let mut sinh_neg = None;
    let mut saw_pos_one = false;
    let mut saw_neg_one = false;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr)
            .is_some_and(|value| matches!((value, term_sign), (1, Sign::Pos) | (-1, Sign::Neg)))
        {
            saw_pos_one = true;
            continue;
        }
        if extract_i64_integer(ctx, term_expr)
            .is_some_and(|value| matches!((value, term_sign), (1, Sign::Neg) | (-1, Sign::Pos)))
        {
            saw_neg_one = true;
            continue;
        }

        match extract_plain_sinh_or_cosh_pow2_arg_root(ctx, term_expr) {
            Some((BuiltinFn::Cosh, arg)) if term_sign == Sign::Pos => cosh_pos = Some(arg),
            Some((BuiltinFn::Cosh, arg)) if term_sign == Sign::Neg => cosh_neg = Some(arg),
            Some((BuiltinFn::Sinh, arg)) if term_sign == Sign::Pos => sinh_pos = Some(arg),
            Some((BuiltinFn::Sinh, arg)) if term_sign == Sign::Neg => sinh_neg = Some(arg),
            _ => return false,
        }
    }

    match (
        cosh_pos,
        sinh_neg,
        saw_neg_one,
        cosh_neg,
        sinh_pos,
        saw_pos_one,
    ) {
        (Some(cosh_arg), Some(sinh_arg), true, _, _, _) => {
            compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal
        }
        (_, _, _, Some(cosh_arg), Some(sinh_arg), true) => {
            compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal
        }
        _ => false,
    }
}

pub(super) fn extract_scaled_plain_cosh_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        if cosh_arg.is_some() {
            return None;
        }
        cosh_arg = Some(arg);
    }

    (numeric_coeff == BigRational::from_integer(4.into()))
        .then_some(cosh_arg)
        .flatten()
}

pub(super) fn extract_scaled_cosh_cubic_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Expr::Pow(base, exponent) = ctx.get(factor) else {
            return None;
        };
        if extract_i64_integer(ctx, *exponent)? != 3 {
            return None;
        }
        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, *base) else {
            return None;
        };
        if cosh_arg.is_some() {
            return None;
        }
        cosh_arg = Some(arg);
    }

    (numeric_coeff == BigRational::from_integer(4.into()))
        .then_some(cosh_arg)
        .flatten()
}

pub(super) fn extract_scaled_sinh_double_angle_sinh_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sinh_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let Some((BuiltinFn::Sinh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        sinh_args.push(arg);
    }

    if numeric_coeff != BigRational::from_integer(2.into()) || sinh_args.len() != 2 {
        return None;
    }

    for &candidate_u in &sinh_args {
        let two = ctx.num(2);
        let doubled_candidate_u = smart_mul(ctx, two, candidate_u);
        if sinh_args
            .iter()
            .any(|arg| compare_expr(ctx, *arg, doubled_candidate_u) == Ordering::Equal)
        {
            return Some(candidate_u);
        }
    }

    None
}

pub(super) fn matches_direct_exp_hyperbolic_double_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut exp_terms: smallvec::SmallVec<[(BigRational, ExprId); 2]> = smallvec::SmallVec::new();
    let mut hyper_term: Option<(BigRational, BuiltinFn, ExprId)> = None;

    for (term_expr, term_sign) in view.terms {
        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }

        if let Some(arg) = extract_exp_argument(ctx, base) {
            exp_terms.push((coeff, arg));
            continue;
        }

        match extract_plain_sinh_or_cosh_arg_root(ctx, base) {
            Some((BuiltinFn::Cosh, arg)) => hyper_term = Some((coeff, BuiltinFn::Cosh, arg)),
            Some((BuiltinFn::Sinh, arg)) => hyper_term = Some((coeff, BuiltinFn::Sinh, arg)),
            _ => return false,
        }
    }

    let Some((hyper_coeff, hyper_fn, hyper_arg)) = hyper_term else {
        return false;
    };
    if exp_terms.len() != 2 {
        return false;
    }

    let two = BigRational::from_integer(2.into());
    for first_index in 0..exp_terms.len() {
        let (first_coeff, first_arg) = &exp_terms[first_index];
        if compare_expr(ctx, *first_arg, hyper_arg) != Ordering::Equal {
            continue;
        }
        let second_index = 1 - first_index;
        let (second_coeff, second_arg) = &exp_terms[second_index];
        let neg_hyper_arg = ctx.add(Expr::Neg(hyper_arg));
        if compare_expr(ctx, *second_arg, neg_hyper_arg) != Ordering::Equal {
            continue;
        }

        let matches_cosh = hyper_fn == BuiltinFn::Cosh
            && *first_coeff == *second_coeff
            && hyper_coeff == -(&two * first_coeff);
        let matches_sinh = hyper_fn == BuiltinFn::Sinh
            && *second_coeff == -first_coeff.clone()
            && hyper_coeff == -(&two * first_coeff);
        if matches_cosh || matches_sinh {
            return true;
        }
    }

    false
}

fn extract_plain_hyperbolic_double_angle_arg_root(
    ctx: &mut Context,
    expr: ExprId,
    expected_fn: BuiltinFn,
) -> Option<ExprId> {
    let (actual_fn, arg) = extract_plain_sinh_or_cosh_arg_root(ctx, expr)?;
    if actual_fn != expected_fn {
        return None;
    }
    extract_double_angle_arg_relaxed(ctx, arg)
}

fn extract_positive_two_cosh_square_minus_one_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cosh_sq_arg = None;
    let mut saw_negative_one = false;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr)
            .is_some_and(|value| matches!((value, term_sign), (1, Sign::Neg) | (-1, Sign::Pos)))
        {
            saw_negative_one = true;
            continue;
        }

        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }
        if coeff != BigRational::from_integer(2.into()) || cosh_sq_arg.is_some() {
            return None;
        }

        let Expr::Pow(pow_base, exponent) = ctx.get(base) else {
            return None;
        };
        if extract_i64_integer(ctx, *exponent) != Some(2) {
            return None;
        }
        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, *pow_base)
        else {
            return None;
        };
        cosh_sq_arg = Some(arg);
    }

    saw_negative_one.then_some(cosh_sq_arg?).or(None)
}

pub(super) fn matches_direct_hyperbolic_sinh_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (double_angle_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base_arg) =
            extract_plain_hyperbolic_double_angle_arg_root(ctx, double_angle_expr, BuiltinFn::Sinh)
        else {
            continue;
        };
        let Some((sinh_arg, cosh_arg)) =
            extract_scaled_hyperbolic_sinh_cosh_product_half_args_root(ctx, product_expr)
        else {
            continue;
        };
        if compare_expr(ctx, sinh_arg, base_arg) == Ordering::Equal
            && compare_expr(ctx, cosh_arg, base_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_cosh_double_angle_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (double_angle_expr, square_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base_arg) =
            extract_plain_hyperbolic_double_angle_arg_root(ctx, double_angle_expr, BuiltinFn::Cosh)
        else {
            continue;
        };
        let Some(cosh_arg) = extract_positive_two_cosh_square_minus_one_arg_root(ctx, square_expr)
        else {
            continue;
        };
        if compare_expr(ctx, base_arg, cosh_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_nested_zero_hyperbolic_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_direct_hyperbolic_sinh_double_angle_pair_root(ctx, lhs, rhs)
            || matches_direct_hyperbolic_cosh_double_angle_square_pair_root(ctx, lhs, rhs)
    };

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if matches_pair(ctx, lhs, rhs) => return true,
        Expr::Add(lhs, rhs) => {
            if let Expr::Neg(inner) = ctx.get(rhs) {
                if matches_pair(ctx, lhs, *inner) {
                    return true;
                }
            }
        }
        _ => {}
    }

    extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
        .is_some_and(|(lhs_core, rhs_core)| matches_pair(ctx, lhs_core, rhs_core))
}

fn matches_direct_hyperbolic_angle_difference_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (single_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((single_fn, angle_arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };
        let view = AddView::from_expr(ctx, expanded_expr);
        if view.terms.len() != 2 {
            continue;
        }

        match single_fn {
            BuiltinFn::Sinh => {
                let mut positive_pair = None;
                let mut negative_pair = None;

                for (term_expr, term_sign) in view.terms {
                    let Some(((fn_a, arg_a), (fn_b, arg_b))) =
                        extract_plain_hyperbolic_product_pair_args_root(ctx, term_expr)
                    else {
                        positive_pair = None;
                        negative_pair = None;
                        break;
                    };
                    let is_sinh_cosh = matches!(
                        (fn_a, fn_b),
                        (BuiltinFn::Sinh, BuiltinFn::Cosh) | (BuiltinFn::Cosh, BuiltinFn::Sinh)
                    );
                    if !is_sinh_cosh {
                        positive_pair = None;
                        negative_pair = None;
                        break;
                    }

                    let sinh_arg = if fn_a == BuiltinFn::Sinh {
                        arg_a
                    } else {
                        arg_b
                    };
                    let cosh_arg = if fn_a == BuiltinFn::Cosh {
                        arg_a
                    } else {
                        arg_b
                    };

                    match term_sign {
                        Sign::Pos if positive_pair.is_none() => {
                            positive_pair = Some((sinh_arg, cosh_arg));
                        }
                        Sign::Neg if negative_pair.is_none() => {
                            negative_pair = Some((sinh_arg, cosh_arg));
                        }
                        _ => {
                            positive_pair = None;
                            negative_pair = None;
                            break;
                        }
                    }
                }

                let (Some((positive_sinh, positive_cosh)), Some((negative_sinh, negative_cosh))) =
                    (positive_pair, negative_pair)
                else {
                    continue;
                };

                if compare_expr(ctx, positive_sinh, negative_cosh) == Ordering::Equal
                    && compare_expr(ctx, positive_cosh, negative_sinh) == Ordering::Equal
                    && matches_angle_sum_or_diff_arg_root(
                        ctx,
                        angle_arg,
                        positive_sinh,
                        positive_cosh,
                        false,
                    )
                {
                    return true;
                }
            }
            BuiltinFn::Cosh => {
                let mut positive_pair = None;
                let mut negative_pair = None;

                for (term_expr, term_sign) in view.terms {
                    let Some(((fn_a, arg_a), (fn_b, arg_b))) =
                        extract_plain_hyperbolic_product_pair_args_root(ctx, term_expr)
                    else {
                        positive_pair = None;
                        negative_pair = None;
                        break;
                    };

                    match term_sign {
                        Sign::Pos
                            if positive_pair.is_none()
                                && fn_a == BuiltinFn::Cosh
                                && fn_b == BuiltinFn::Cosh =>
                        {
                            positive_pair = Some((arg_a, arg_b));
                        }
                        Sign::Neg
                            if negative_pair.is_none()
                                && fn_a == BuiltinFn::Sinh
                                && fn_b == BuiltinFn::Sinh =>
                        {
                            negative_pair = Some((arg_a, arg_b));
                        }
                        _ => {
                            positive_pair = None;
                            negative_pair = None;
                            break;
                        }
                    }
                }

                let (Some((positive_u, positive_v)), Some((negative_u, negative_v))) =
                    (positive_pair, negative_pair)
                else {
                    continue;
                };

                if matches_unordered_expr_pair_root(
                    ctx, positive_u, positive_v, negative_u, negative_v,
                ) && matches_angle_sum_or_diff_arg_root(
                    ctx, angle_arg, positive_u, positive_v, false,
                ) {
                    return true;
                }
            }
            _ => {}
        }
    }

    false
}

pub(super) fn matches_direct_nested_zero_hyperbolic_angle_difference_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_direct_hyperbolic_angle_difference_pair_root(ctx, lhs, rhs)
    };

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if matches_pair(ctx, lhs, rhs) => return true,
        Expr::Add(lhs, rhs) => {
            if let Expr::Neg(inner) = ctx.get(rhs) {
                if matches_pair(ctx, lhs, *inner) {
                    return true;
                }
            }
        }
        _ => {}
    }

    extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
        .is_some_and(|(lhs_core, rhs_core)| matches_pair(ctx, lhs_core, rhs_core))
}

pub(super) fn matches_direct_nested_zero_hyperbolic_triple_angle_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Sinh, BuiltinFn::Cosh]) {
        return false;
    }

    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_direct_hyperbolic_triple_angle_pair_root(ctx, lhs, rhs)
    };

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if matches_pair(ctx, lhs, rhs) => return true,
        Expr::Add(lhs, rhs) => {
            if let Expr::Neg(inner) = ctx.get(rhs) {
                if matches_pair(ctx, lhs, *inner) {
                    return true;
                }
            }
        }
        _ => {}
    }

    extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
        .is_some_and(|(lhs_core, rhs_core)| matches_pair(ctx, lhs_core, rhs_core))
}

pub(super) fn matches_direct_atanh_square_ratio_log_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    if !expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Atanh]) {
        return false;
    }
    if !expr_contains_any_builtin_local(
        ctx,
        expr,
        &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
    ) {
        return false;
    }

    crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(ctx, expr).is_some()
}

pub(super) fn is_small_trig_or_hyperbolic_zero_child(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
    {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    let has_positive = terms.iter().any(|(_, sign)| *sign == Sign::Pos);
    let has_negative = terms.iter().any(|(_, sign)| *sign == Sign::Neg);
    if terms.len() > 4
        || !has_positive
        || !has_negative
        || !terms.iter().all(|(term, _)| {
            expr_contains_trig_or_hyperbolic_builtin_local(ctx, *term)
                || matches!(ctx.get(*term), Expr::Number(_))
        })
    {
        return false;
    }

    let mut isolated_ctx = Context::new();
    let isolated_child = transplant_expr_subtree(ctx, expr, &mut isolated_ctx);
    if isolated_simplify_rewrites_to_zero(options, &mut isolated_ctx, isolated_child) {
        return true;
    }

    if let Some(rewritten) =
        isolated_simplify_expr_if_changed(options, &mut isolated_ctx, isolated_child)
    {
        return isolated_simplify_rewrites_to_zero(options, &mut isolated_ctx, rewritten);
    }

    false
}

fn is_small_trig_or_hyperbolic_numeric_subset_term_root(ctx: &Context, term: ExprId) -> bool {
    matches!(
        classify_small_trig_or_hyperbolic_numeric_subset_term_root(ctx, term),
        SmallTrigOrHyperbolicNumericSubsetTermRootKind::Number
            | SmallTrigOrHyperbolicNumericSubsetTermRootKind::TrigOrHyperbolic
    )
}

fn expr_function_calls_are_only_trig_or_hyperbolic_local(ctx: &Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                let is_supported = ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Tan)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cot)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Sec)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Csc)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cosh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Tanh);
                if !is_supported {
                    return false;
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
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => {}
            Expr::Matrix { .. } | Expr::SessionRef(_) | Expr::Hold(_) => return false,
        }
    }

    true
}

fn classify_small_trig_or_hyperbolic_numeric_subset_term_root(
    ctx: &Context,
    term: ExprId,
) -> SmallTrigOrHyperbolicNumericSubsetTermRootKind {
    if matches!(ctx.get(term), Expr::Number(_)) {
        return SmallTrigOrHyperbolicNumericSubsetTermRootKind::Number;
    }

    let contains_trig_or_hyperbolic = expr_contains_trig_or_hyperbolic_builtin_local(ctx, term);
    if !contains_trig_or_hyperbolic
        || expr_contains_log_builtin_local(ctx, term)
        || !expr_function_calls_are_only_trig_or_hyperbolic_local(ctx, term)
    {
        return SmallTrigOrHyperbolicNumericSubsetTermRootKind::Other;
    }

    SmallTrigOrHyperbolicNumericSubsetTermRootKind::TrigOrHyperbolic
}

fn maybe_small_trig_or_hyperbolic_numeric_subset_root_candidate(
    ctx: &Context,
    expr: ExprId,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(5..=17).contains(&terms.len()) {
        return false;
    }

    let mut subset_like_terms = 0usize;
    let mut actual_trig_or_hyperbolic_terms = 0usize;
    let mut partner_like_terms = 0usize;
    let mut mixed_log_trig_terms = 0usize;
    let mut nonlog_partner_terms = 0usize;
    for (term, _) in terms.iter().copied() {
        let contains_log_like = expr_contains_any_builtin_local(
            ctx,
            term,
            &[
                BuiltinFn::Ln,
                BuiltinFn::Log,
                BuiltinFn::Log10,
                BuiltinFn::Atanh,
                BuiltinFn::Abs,
            ],
        );
        let contains_trig_or_hyperbolic = expr_contains_trig_or_hyperbolic_builtin_local(ctx, term);
        if contains_log_like && contains_trig_or_hyperbolic {
            mixed_log_trig_terms += 1;
        }

        match classify_small_trig_or_hyperbolic_numeric_subset_term_root(ctx, term) {
            SmallTrigOrHyperbolicNumericSubsetTermRootKind::Number => {
                subset_like_terms += 1;
            }
            SmallTrigOrHyperbolicNumericSubsetTermRootKind::TrigOrHyperbolic => {
                subset_like_terms += 1;
                actual_trig_or_hyperbolic_terms += 1;
            }
            SmallTrigOrHyperbolicNumericSubsetTermRootKind::Other => {
                partner_like_terms += 1;
                if !contains_log_like {
                    nonlog_partner_terms += 1;
                }
            }
        }
    }

    if mixed_log_trig_terms > 0 && nonlog_partner_terms > 0 {
        return false;
    }

    subset_like_terms >= 3 && actual_trig_or_hyperbolic_terms >= 2 && partner_like_terms >= 2
}

fn classify_small_trig_or_hyperbolic_numeric_additive_chunk_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<bool> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if is_supported_nonlog_additive_nested_zero_child_partner(ctx, expr) {
        return Some(false);
    }
    if expr_contains_log_builtin_local(ctx, expr) && terms.len() <= 4 {
        return Some(false);
    }

    let mut saw_subset = false;
    let mut saw_partner = false;

    for (term, _) in terms.iter().copied() {
        if is_small_trig_or_hyperbolic_numeric_subset_term_root(ctx, term) {
            saw_subset = true;
        } else {
            saw_partner = true;
        }

        if saw_subset && saw_partner {
            return None;
        }
    }

    Some(saw_subset)
}

fn collect_small_trig_or_hyperbolic_numeric_chunks_root(
    ctx: &Context,
    expr: ExprId,
    sign: Sign,
    subset_chunks: &mut smallvec::SmallVec<[(ExprId, Sign); 8]>,
    partner_chunks: &mut smallvec::SmallVec<[(ExprId, Sign); 8]>,
) {
    if let Some(is_subset_chunk) =
        classify_small_trig_or_hyperbolic_numeric_additive_chunk_root(ctx, expr)
    {
        if is_subset_chunk {
            subset_chunks.push((expr, sign));
        } else {
            partner_chunks.push((expr, sign));
        }
        return;
    }

    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            collect_small_trig_or_hyperbolic_numeric_chunks_root(
                ctx,
                *lhs,
                sign,
                subset_chunks,
                partner_chunks,
            );
            collect_small_trig_or_hyperbolic_numeric_chunks_root(
                ctx,
                *rhs,
                sign,
                subset_chunks,
                partner_chunks,
            );
        }
        Expr::Sub(lhs, rhs) => {
            collect_small_trig_or_hyperbolic_numeric_chunks_root(
                ctx,
                *lhs,
                sign,
                subset_chunks,
                partner_chunks,
            );
            collect_small_trig_or_hyperbolic_numeric_chunks_root(
                ctx,
                *rhs,
                sign.negate(),
                subset_chunks,
                partner_chunks,
            );
        }
        Expr::Neg(inner) => {
            collect_small_trig_or_hyperbolic_numeric_chunks_root(
                ctx,
                *inner,
                sign.negate(),
                subset_chunks,
                partner_chunks,
            );
        }
        _ => {
            if is_small_trig_or_hyperbolic_numeric_subset_term_root(ctx, expr) {
                subset_chunks.push((expr, sign));
            } else {
                partner_chunks.push((expr, sign));
            }
        }
    }
}

pub(super) fn extract_small_trig_or_hyperbolic_numeric_subset_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if !maybe_small_trig_or_hyperbolic_numeric_subset_root_candidate(ctx, expr) {
        return None;
    }

    let mut subset_chunks = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
    let mut partner_chunks = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
    collect_small_trig_or_hyperbolic_numeric_chunks_root(
        ctx,
        expr,
        Sign::Pos,
        &mut subset_chunks,
        &mut partner_chunks,
    );

    if subset_chunks.is_empty() || partner_chunks.is_empty() {
        return None;
    }

    let subset_expr = build_signed_sum_expr_root(ctx, &subset_chunks);
    let partner_expr = build_signed_sum_expr_root(ctx, &partner_chunks);
    Some((subset_expr, partner_expr))
}

pub(super) fn try_standard_collapsed_fraction_hyperbolic_half_angle_factor_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (partner_index, factor_index) in [(0usize, 1usize), (1usize, 0usize)] {
        let partner = factors[partner_index];
        let partner_collapsed = if let Some(base) =
            extract_addition_of_successive_unit_fractions_arg_root(ctx, partner)
        {
            build_collapsed_successive_unit_fractions_expr_root(ctx, base)
        } else if let Some(base) =
            extract_collapsed_successive_unit_fractions_arg_root(ctx, partner)
        {
            build_collapsed_successive_unit_fractions_expr_root(ctx, base)
        } else {
            continue;
        };

        let Some((hyperbolic_fn, half_arg)) =
            extract_plain_sinh_or_cosh_pow2_arg_root(ctx, factors[factor_index])
        else {
            continue;
        };
        let Some(arg) = extract_half_scaled_base_root(ctx, half_arg) else {
            continue;
        };
        let target_factor =
            build_direct_hyperbolic_half_angle_square_target_root(ctx, hyperbolic_fn, arg);

        let rewritten = if partner_index == 0 {
            build_mul_expr_from_factors_root(ctx, &[partner_collapsed, target_factor])
        } else {
            build_mul_expr_from_factors_root(ctx, &[target_factor, partner_collapsed])
        };
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Canonizar producto de fracción consecutiva con media-ángulo hiperbólico");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Canonical Hyperbolic Half-Angle Product",
            collect_steps,
        ));
    }

    None
}

pub(super) fn is_potential_tanh_ratio_anchor_source_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return false;
    };
    let Some((num_arg, false, _)) =
        cas_math::hyperbolic_exp_support::extract_exp_pair(ctx, *numerator)
    else {
        return false;
    };
    let Some((den_arg, true, _)) =
        cas_math::hyperbolic_exp_support::extract_exp_pair(ctx, *denominator)
    else {
        return false;
    };
    compare_expr(ctx, num_arg, den_arg) == Ordering::Equal
}

pub(super) fn try_standard_hyperbolic_cosh_cubic_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(5..=6).contains(&view.terms.len()) {
        return None;
    }

    let matches_small_hyperbolic_identity_subset = |ctx: &mut Context, subset_expr: ExprId| {
        let subset_view = AddView::from_expr(ctx, subset_expr);
        let has_positive = subset_view.terms.iter().any(|(_, sign)| *sign == Sign::Pos);
        let has_negative = subset_view.terms.iter().any(|(_, sign)| *sign == Sign::Neg);

        subset_view.terms.len() <= 3
            && has_positive
            && has_negative
            && expr_contains_hyperbolic_builtin_local(ctx, subset_expr)
            && !expr_contains_trig_builtin_local(ctx, subset_expr)
            && subset_view.terms.iter().all(|(term, _)| {
                expr_contains_hyperbolic_builtin_local(ctx, *term)
                    || matches!(ctx.get(*term), Expr::Number(_))
            })
            && (matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, subset_expr)
                || isolated_simplify_rewrites_to_zero(options, ctx, subset_expr))
    };

    for subset_size in [2usize, 3usize] {
        for first_index in 0..view.terms.len() {
            for second_index in (first_index + 1)..view.terms.len() {
                if subset_size == 2 {
                    let subset_terms = [view.terms[first_index], view.terms[second_index]];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_small_hyperbolic_identity_subset(ctx, subset_expr) {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index).then_some(term)
                        })
                        .collect();
                    if !(2..=4).contains(&remaining_terms.len()) {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                    continue;
                }

                for third_index in (second_index + 1)..view.terms.len() {
                    let subset_terms = [
                        view.terms[first_index],
                        view.terms[second_index],
                        view.terms[third_index],
                    ];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_small_hyperbolic_identity_subset(ctx, subset_expr) {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index && index != third_index)
                                .then_some(term)
                        })
                        .collect();
                    if !(2..=4).contains(&remaining_terms.len()) {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                }
            }
        }
    }

    None
}

pub(super) fn try_extract_atanh_square_ratio_log_subset_zero_chunks_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(5..=17).contains(&view.terms.len()) {
        return None;
    }
    if expr_contains_trig_builtin_local(ctx, expr) {
        return None;
    }
    if !expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Atanh]) {
        return None;
    }
    if !expr_contains_any_builtin_local(
        ctx,
        expr,
        &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
    ) {
        return None;
    }

    for first_index in 0..view.terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..view.terms.len() {
            let subset_terms = [view.terms[first_index], view.terms[second_index]];
            let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
            if crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(
                ctx,
                subset_expr,
            )
            .is_none()
            {
                continue;
            }

            let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if !(3..=15).contains(&remaining_terms.len()) {
                continue;
            }

            let remaining_expr = AddView {
                root: expr,
                terms: remaining_terms,
            }
            .rebuild(ctx);

            if expr_contains_hyperbolic_builtin_local(ctx, remaining_expr)
                || expr_contains_trig_builtin_local(ctx, remaining_expr)
            {
                continue;
            }
            if !expr_contains_any_builtin_local(
                ctx,
                remaining_expr,
                &[
                    BuiltinFn::Ln,
                    BuiltinFn::Log,
                    BuiltinFn::Log10,
                    BuiltinFn::Exp,
                ],
            ) {
                continue;
            }

            let remaining_rewrites_to_zero =
                supported_nested_zero_partner_rewrites_to_zero(options, ctx, remaining_expr)
                    || try_standard_multiterm_trig_numeric_subset_zero_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                    || try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                    || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr);
            if !remaining_rewrites_to_zero {
                continue;
            }

            return Some((subset_expr, remaining_expr));
        }
    }

    None
}

pub(super) fn try_standard_atanh_square_ratio_log_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (subset_expr, remaining_expr) =
        try_extract_atanh_square_ratio_log_subset_zero_chunks_root(options, ctx, expr)?;
    let zero = ctx.num(0);
    if collect_steps {
        if let Some(steps) = try_build_chunk_pair_zero_shortcut_steps_root(
            options,
            ctx,
            expr,
            subset_expr,
            remaining_expr,
        ) {
            return Some((zero, steps));
        }
    }
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        zero,
        collect_steps,
    ))
}

pub(super) fn try_standard_atanh_square_ratio_log_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    if AddView::from_expr(ctx, expr).terms.len() != 2 {
        return None;
    }

    let rewrite =
        crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(ctx, expr)?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

fn matches_direct_nested_fraction_zero_hyperbolic_identity_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    for (fraction_side, hyperbolic_side) in [(lhs, rhs), (rhs, lhs)] {
        if !expr_contains_division_node_local(ctx, fraction_side)
            || expr_contains_hyperbolic_builtin_local(ctx, fraction_side)
            || expr_contains_division_node_local(ctx, hyperbolic_side)
            || !expr_contains_hyperbolic_builtin_local(ctx, hyperbolic_side)
        {
            continue;
        }

        if !matches_direct_nested_fraction_simplified_zero_identity_root(ctx, fraction_side) {
            continue;
        }

        if matches_direct_recursive_hyperbolic_sinh_sum_zero_identity_root(ctx, hyperbolic_side)
            || matches_direct_recursive_hyperbolic_cosh_sum_zero_identity_root(ctx, hyperbolic_side)
            || matches_direct_hyperbolic_exp_sum_zero_identity_root(ctx, hyperbolic_side)
            || matches_direct_exp_hyperbolic_double_identity_root(ctx, hyperbolic_side)
            || matches_direct_hyperbolic_pythagorean_zero_identity_root(ctx, hyperbolic_side)
            || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, hyperbolic_side)
        {
            return true;
        }
    }

    false
}

pub(super) fn try_standard_nested_fraction_zero_hyperbolic_identity_pair_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return None,
    };

    if !matches_direct_nested_fraction_zero_hyperbolic_identity_pair_root(ctx, lhs, rhs) {
        return None;
    }

    let zero = ctx.num(0);
    let rewrite = crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

fn matches_direct_log_zero_hyperbolic_cosh_cubic_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    for (log_side, hyperbolic_side) in [(lhs, rhs), (rhs, lhs)] {
        if !expr_contains_log_builtin_local(ctx, log_side)
            || expr_contains_hyperbolic_builtin_local(ctx, log_side)
            || !expr_contains_hyperbolic_builtin_local(ctx, hyperbolic_side)
            || expr_contains_log_builtin_local(ctx, hyperbolic_side)
        {
            continue;
        }

        if is_hot_log_split_zero_side_root(ctx, log_side)
            && matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, hyperbolic_side)
        {
            return true;
        }
    }

    false
}

pub(super) fn try_standard_log_zero_hyperbolic_cosh_cubic_pair_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return None,
    };

    if !matches_direct_log_zero_hyperbolic_cosh_cubic_pair_root(ctx, lhs, rhs) {
        return None;
    }

    let zero = ctx.num(0);
    let rewrite = crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}
