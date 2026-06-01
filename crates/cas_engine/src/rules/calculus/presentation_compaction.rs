use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::presentation_utils::small_rational_const_for_calculus_presentation;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    split_numeric_scale_single_core,
};

pub(super) fn scale_ordered_product_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    left: ExprId,
    right: ExprId,
) -> ExprId {
    let product = ctx.add_raw(Expr::Mul(left, right));
    if coeff.is_one() {
        return product;
    }
    if coeff == -BigRational::one() {
        return ctx.add_raw(Expr::Neg(product));
    }
    let coeff = rational_const_for_calculus_presentation(ctx, coeff);
    ctx.add_raw(Expr::Mul(coeff, product))
}

pub(super) fn unary_variable_builtin_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(builtin) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

pub(super) fn split_signed_numeric_scale_single_core_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr) {
        let (scale, core) = split_numeric_scale_single_core(ctx, *inner)?;
        return Some((-scale, core));
    }
    split_numeric_scale_single_core(ctx, expr)
}

pub(super) fn distribute_half_over_additive_numerator_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let half = BigRational::new(1.into(), 2.into());
    let mut improves_integer_scale = false;
    for (term, _) in terms.iter().copied() {
        let (term_scale, _) =
            split_numeric_scale_single_core(ctx, term).unwrap_or((BigRational::one(), term));
        let scaled = term_scale * half.clone();
        if scaled.is_integer() {
            improves_integer_scale = true;
            break;
        }
    }
    if !improves_integer_scale {
        return None;
    }

    let mut scaled_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (term_scale, term_core) =
            split_numeric_scale_single_core(ctx, term).unwrap_or((BigRational::one(), term));
        let coeff = match sign {
            cas_math::expr_nary::Sign::Pos => half.clone(),
            cas_math::expr_nary::Sign::Neg => -half.clone(),
        } * term_scale;
        scaled_terms.push(scale_expr_for_calculus_presentation(ctx, coeff, term_core));
    }

    Some(cas_math::expr_nary::build_balanced_add(ctx, &scaled_terms))
}

pub(super) fn compact_small_power_exponents_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(left, right) => {
            let left = compact_small_power_exponents_for_calculus_presentation(ctx, left);
            let right = compact_small_power_exponents_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Div(left, right))
        }
        Expr::Neg(inner) => {
            let inner = compact_small_power_exponents_for_calculus_presentation(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Pow(base, exp) => {
            let base = compact_small_power_exponents_for_calculus_presentation(ctx, base);
            if let Some(exponent) = small_rational_const_for_calculus_presentation(ctx, exp) {
                if exponent.is_zero() {
                    return ctx.num(1);
                }
                if exponent.is_one() {
                    return base;
                }
                let exp = rational_const_for_calculus_presentation(ctx, exponent);
                return ctx.add(Expr::Pow(base, exp));
            }
            let exp = compact_small_power_exponents_for_calculus_presentation(ctx, exp);
            ctx.add(Expr::Pow(base, exp))
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| compact_small_power_exponents_for_calculus_presentation(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    }
}

pub(super) fn compact_numeric_mul_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            let product = ctx.add(Expr::Mul(left, right));
            compact_numeric_product_for_calculus_presentation(ctx, product)
        }
        Expr::Div(left, right) => {
            let left = compact_numeric_mul_factors_for_calculus_presentation(ctx, left);
            let right = compact_numeric_mul_factors_for_calculus_presentation(ctx, right);
            ctx.add(Expr::Div(left, right))
        }
        Expr::Neg(inner) => {
            let inner = compact_numeric_mul_factors_for_calculus_presentation(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Pow(base, exp) => {
            let base = compact_numeric_mul_factors_for_calculus_presentation(ctx, base);
            let exp = compact_numeric_mul_factors_for_calculus_presentation(ctx, exp);
            ctx.add(Expr::Pow(base, exp))
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| compact_numeric_mul_factors_for_calculus_presentation(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    }
}

fn compact_numeric_product_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return expr;
    }

    let mut scale = BigRational::one();
    let mut non_numeric_factors = Vec::with_capacity(factors.len());
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if let Expr::Neg(inner) = ctx.get(factor) {
            scale = -scale;
            non_numeric_factors.push(*inner);
        } else {
            non_numeric_factors.push(factor);
        }
    }

    if scale.is_zero() {
        return ctx.num(0);
    }
    if non_numeric_factors.is_empty() {
        return rational_const_for_calculus_presentation(ctx, scale);
    }

    let core = if non_numeric_factors.len() == 1 {
        non_numeric_factors[0]
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric_factors)
    };
    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(core));
    }
    scale_expr_for_calculus_presentation(ctx, scale, core)
}

pub(crate) fn compact_double_angle_sine_products_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut changed = false;
    let mut rebuilt = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let compact = double_angle_sine_product_for_calculus_presentation(ctx, term);
        changed |= compact.is_some();
        let mut rebuilt_term = compact.unwrap_or(term);
        if sign == cas_math::expr_nary::Sign::Neg {
            rebuilt_term = ctx.add(Expr::Neg(rebuilt_term));
        }
        rebuilt.push(rebuilt_term);
    }

    changed.then(|| cas_math::expr_nary::build_balanced_add(ctx, &rebuilt))
}

pub(super) fn signed_add_terms_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut saw_negative = false;
    let rebuilt = terms
        .into_iter()
        .map(|(term, sign)| {
            if sign == cas_math::expr_nary::Sign::Neg {
                saw_negative = true;
                ctx.add_raw(Expr::Neg(term))
            } else {
                term
            }
        })
        .collect::<Vec<_>>();

    saw_negative.then(|| build_balanced_add_raw_for_calculus_presentation(ctx, &rebuilt))
}

fn build_balanced_add_raw_for_calculus_presentation(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    match terms.len() {
        0 => ctx.num(0),
        1 => terms[0],
        2 => ctx.add_raw(Expr::Add(terms[0], terms[1])),
        n => {
            let mid = n / 2;
            let left = build_balanced_add_raw_for_calculus_presentation(ctx, &terms[..mid]);
            let right = build_balanced_add_raw_for_calculus_presentation(ctx, &terms[mid..]);
            ctx.add_raw(Expr::Add(left, right))
        }
    }
}

fn double_angle_sine_product_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let mut scale = BigRational::one();
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        match ctx.builtin_of(*fn_id) {
            Some(BuiltinFn::Sin) if sin_arg.replace(args[0]).is_none() => {}
            Some(BuiltinFn::Cos) if cos_arg.replace(args[0]).is_none() => {}
            _ => return None,
        }
    }

    if scale != BigRational::from_integer(2.into()) {
        return None;
    }
    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    if compare_expr(ctx, sin_arg, cos_arg) != std::cmp::Ordering::Equal {
        return None;
    }

    let two = rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
    let doubled_arg = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sin_arg]);
    Some(ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg]))
}

pub(super) fn bounded_sin_cos_shift_margin_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    let mut constant_shift = BigRational::zero();
    let mut trig_bound = BigRational::zero();
    let mut has_bounded_trig = false;

    for term in cas_math::expr_nary::add_terms_no_sign(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, term, 8) {
            constant_shift += value;
            continue;
        }

        let bound = bounded_sin_cos_term_bound_for_calculus_presentation(ctx, term)?;
        trig_bound += bound;
        has_bounded_trig = true;
    }

    if has_bounded_trig && constant_shift > trig_bound {
        Some(constant_shift - trig_bound)
    } else {
        None
    }
}

pub(super) fn bounded_sin_cos_term_bound_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if bounded_sin_cos_unit_factor_for_calculus_presentation(ctx, expr) {
        return Some(BigRational::one());
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return bounded_sin_cos_term_bound_for_calculus_presentation(ctx, *inner);
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let mut scale = BigRational::one();
    let mut has_bounded_factor = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if bounded_sin_cos_unit_factor_for_calculus_presentation(ctx, factor) {
            has_bounded_factor = true;
        } else {
            return None;
        }
    }

    has_bounded_factor.then(|| scale.abs())
}

fn bounded_sin_cos_unit_factor_for_calculus_presentation(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Sin | BuiltinFn::Cos)
                )
        }
        Expr::Pow(base, exp)
            if bounded_sin_cos_unit_factor_for_calculus_presentation(ctx, *base) =>
        {
            cas_ast::views::as_rational_const(ctx, *exp, 8)
                .is_some_and(|value| value.is_integer() && value > BigRational::zero())
        }
        _ => false,
    }
}
