//! Simplify action handler for `Engine::eval`.

use super::*;
use cas_ast::{BuiltinFn, Expr};
use num_traits::{ToPrimitive, Zero};

fn expr_contains_any_builtin_local(
    ctx: &cas_ast::Context,
    root: ExprId,
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

fn expr_contains_hyperbolic_builtin_local(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
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

fn exprs_equivalent_ignoring_internal_holds_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left = cas_ast::hold::strip_all_holds(ctx, left);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, left, right)
        || format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: left
            }
        ) == format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: right
            }
        )
}

fn fractions_cross_products_equivalent_ignoring_internal_holds_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left = cas_ast::hold::strip_all_holds(ctx, left);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    let (left_num, left_den) = match ctx.get(left).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    let (right_num, right_den) = match ctx.get(right).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };

    let left_cross = cas_math::expr_nary::build_balanced_mul(ctx, &[left_num, right_den]);
    let right_cross = cas_math::expr_nary::build_balanced_mul(ctx, &[right_num, left_den]);
    exprs_equivalent_ignoring_internal_holds_local(ctx, left_cross, right_cross)
}

fn commutative_products_equivalent_ignoring_internal_holds_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left = cas_ast::hold::strip_all_holds(ctx, left);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    let left_factors = cas_math::expr_nary::mul_leaves(ctx, left);
    let right_factors = cas_math::expr_nary::mul_leaves(ctx, right);
    if left_factors.len() != right_factors.len() || left_factors.len() <= 1 {
        return false;
    }

    let mut matched = vec![false; right_factors.len()];
    for left_factor in left_factors {
        let Some((index, _)) = right_factors
            .iter()
            .enumerate()
            .find(|(index, right_factor)| {
                !matched[*index]
                    && residual_factor_equivalent_ignoring_internal_holds_local(
                        ctx,
                        left_factor,
                        **right_factor,
                    )
            })
        else {
            return false;
        };
        matched[index] = true;
    }
    true
}

fn commutative_products_equivalent_expanding_small_integer_powers_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let Some(left_factors) = product_factors_with_small_integer_powers_expanded_local(ctx, left)
    else {
        return false;
    };
    let Some(right_factors) = product_factors_with_small_integer_powers_expanded_local(ctx, right)
    else {
        return false;
    };
    if left_factors.len() != right_factors.len() || left_factors.len() <= 1 {
        return false;
    }

    let mut matched = vec![false; right_factors.len()];
    for left_factor in left_factors {
        let Some((index, _)) = right_factors
            .iter()
            .enumerate()
            .find(|(index, right_factor)| {
                !matched[*index]
                    && residual_factor_equivalent_ignoring_internal_holds_local(
                        ctx,
                        left_factor,
                        **right_factor,
                    )
            })
        else {
            return false;
        };
        matched[index] = true;
    }
    true
}

fn product_factors_with_small_integer_powers_expanded_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<Vec<ExprId>> {
    let mut expanded = Vec::new();
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
            expanded.push(factor);
            continue;
        };
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if !exponent.is_integer() {
            expanded.push(factor);
            continue;
        }
        let power = exponent.to_integer().to_usize()?;
        if !(2..=4).contains(&power) {
            expanded.push(factor);
            continue;
        }
        expanded.extend(std::iter::repeat_n(base, power));
    }
    Some(expanded)
}

fn residual_factor_equivalent_ignoring_internal_holds_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    exprs_equivalent_ignoring_internal_holds_local(ctx, left, right)
        || sqrt_factors_have_signed_additive_radicands_equivalent_local(ctx, left, right)
}

fn sqrt_factors_have_signed_additive_radicands_equivalent_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let Some(left_radicand) = cas_math::root_forms::extract_square_root_base(ctx, left) else {
        return false;
    };
    let Some(right_radicand) = cas_math::root_forms::extract_square_root_base(ctx, right) else {
        return false;
    };
    signed_additive_terms_equivalent_ignoring_internal_holds_local(
        ctx,
        left_radicand,
        right_radicand,
    )
}

fn signed_additive_terms_equivalent_ignoring_internal_holds_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left_terms = cas_math::expr_nary::add_terms_signed(ctx, left);
    let right_terms = cas_math::expr_nary::add_terms_signed(ctx, right);
    if left_terms.len() != right_terms.len() || !(2..=6).contains(&left_terms.len()) {
        return false;
    }

    let mut matched = vec![false; right_terms.len()];
    for (left_term, left_sign) in left_terms {
        let Some((index, _)) =
            right_terms
                .iter()
                .enumerate()
                .find(|(index, (right_term, right_sign))| {
                    !matched[*index]
                        && left_sign == *right_sign
                        && exprs_equivalent_ignoring_internal_holds_local(
                            ctx,
                            left_term,
                            *right_term,
                        )
                })
        else {
            return false;
        };
        matched[index] = true;
    }
    true
}

fn fractions_equivalent_up_to_commutative_products_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left = cas_ast::hold::strip_all_holds(ctx, left);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    let (left_num, left_den) = match ctx.get(left).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    let (right_num, right_den) = match ctx.get(right).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };

    (exprs_equivalent_ignoring_internal_holds_local(ctx, left_num, right_num)
        || commutative_products_equivalent_ignoring_internal_holds_local(ctx, left_num, right_num))
        && (exprs_equivalent_ignoring_internal_holds_local(ctx, left_den, right_den)
            || commutative_products_equivalent_ignoring_internal_holds_local(
                ctx, left_den, right_den,
            ))
}

fn split_numeric_scale_product_for_residual_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> (num_rational::BigRational, ExprId) {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let (num_scale, num_core) = split_numeric_scale_product_for_residual_local(ctx, num);
        let (den_scale, den_core) = split_numeric_scale_product_for_residual_local(ctx, den);
        if !den_scale.is_zero() {
            let scale = num_scale / den_scale;
            let one = num_rational::BigRational::from_integer(1.into());
            let num_core_is_one = cas_ast::views::as_rational_const(ctx, num_core, 8)
                .is_some_and(|value| value == one);
            let den_core_is_one = cas_ast::views::as_rational_const(ctx, den_core, 8)
                .is_some_and(|value| value == one);
            let core = match (num_core_is_one, den_core_is_one) {
                (_, true) => num_core,
                (true, false) => {
                    let one_expr = ctx.num(1);
                    ctx.add(Expr::Div(one_expr, den_core))
                }
                (false, false) => ctx.add(Expr::Div(num_core, den_core)),
            };
            return (scale, core);
        }
    }

    let mut scale = num_rational::BigRational::from_integer(1.into());
    let mut non_numeric = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    let core = match non_numeric.as_slice() {
        [] => ctx.num(1),
        [single] => *single,
        _ => cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric),
    };
    (scale, core)
}

fn scale_expr_for_residual_match_local(
    ctx: &mut cas_ast::Context,
    scale: num_rational::BigRational,
    expr: ExprId,
) -> ExprId {
    if scale.is_zero() {
        return ctx.num(0);
    }
    if scale == num_rational::BigRational::from_integer(1.into()) {
        return expr;
    }
    let scale = ctx.add(Expr::Number(scale));
    cas_math::expr_nary::build_balanced_mul(ctx, &[scale, expr])
}

fn scaled_additive_terms_for_residual_match_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    outer_scale: num_rational::BigRational,
) -> Option<Vec<(num_rational::BigRational, ExprId)>> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() > 8 {
        return None;
    }

    let mut scaled_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (term_scale, core) = split_numeric_scale_product_for_residual_local(ctx, term);
        let signed_scale = if sign == cas_math::expr_nary::Sign::Neg {
            -term_scale
        } else {
            term_scale
        };
        scaled_terms.push((outer_scale.clone() * signed_scale, core));
    }
    Some(scaled_terms)
}

fn scaled_additive_terms_equivalent_for_residual_match_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    left_scale: num_rational::BigRational,
    right: ExprId,
    right_scale: num_rational::BigRational,
) -> bool {
    let Some(left_terms) = scaled_additive_terms_for_residual_match_local(ctx, left, left_scale)
    else {
        return false;
    };
    let Some(right_terms) = scaled_additive_terms_for_residual_match_local(ctx, right, right_scale)
    else {
        return false;
    };
    if left_terms.len() != right_terms.len() {
        return false;
    }

    let mut matched = vec![false; right_terms.len()];
    for (left_scale, left_core) in left_terms {
        let Some((index, _)) =
            right_terms
                .iter()
                .enumerate()
                .find(|(index, (right_scale, right_core))| {
                    !matched[*index]
                        && left_scale == *right_scale
                        && additive_residual_term_cores_equivalent_local(
                            ctx,
                            left_core,
                            *right_core,
                        )
                })
        else {
            return false;
        };
        matched[index] = true;
    }
    true
}

fn additive_residual_term_cores_equivalent_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    exprs_equivalent_ignoring_internal_holds_local(ctx, left, right)
        || commutative_products_equivalent_ignoring_internal_holds_local(ctx, left, right)
        || commutative_products_equivalent_expanding_small_integer_powers_local(ctx, left, right)
        || same_power_with_equal_rational_exponent_local(ctx, left, right)
        || reciprocal_trig_square_base_square_product_matches_one_local(ctx, left, right)
        || reciprocal_trig_square_base_square_product_matches_one_local(ctx, right, left)
        || reciprocal_sqrt_matches_negative_half_power_local(ctx, left, right)
        || reciprocal_sqrt_matches_negative_half_power_local(ctx, right, left)
}

fn reciprocal_trig_square_base_square_product_matches_one_local(
    ctx: &mut cas_ast::Context,
    maybe_one: ExprId,
    product: ExprId,
) -> bool {
    if cas_ast::views::as_rational_const(ctx, maybe_one, 8)
        .is_none_or(|value| value != num_rational::BigRational::from_integer(1.into()))
    {
        return false;
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, product);
    if factors.len() != 2 {
        return false;
    }

    let mut reciprocal_trig = None;
    let mut base_trig = None;
    for factor in factors {
        if let Some(arg) = squared_builtin_arg_local(ctx, factor, BuiltinFn::Sec) {
            if reciprocal_trig.replace((BuiltinFn::Sec, arg)).is_some() {
                return false;
            }
            continue;
        }
        if let Some(arg) = squared_builtin_arg_local(ctx, factor, BuiltinFn::Csc) {
            if reciprocal_trig.replace((BuiltinFn::Csc, arg)).is_some() {
                return false;
            }
            continue;
        }
        if let Some(arg) = squared_builtin_arg_local(ctx, factor, BuiltinFn::Cos) {
            if base_trig.replace((BuiltinFn::Cos, arg)).is_some() {
                return false;
            }
            continue;
        }
        if let Some(arg) = squared_builtin_arg_local(ctx, factor, BuiltinFn::Sin) {
            if base_trig.replace((BuiltinFn::Sin, arg)).is_some() {
                return false;
            }
            continue;
        }
        return false;
    }

    match (reciprocal_trig, base_trig) {
        (Some((BuiltinFn::Sec, reciprocal_arg)), Some((BuiltinFn::Cos, base_arg)))
        | (Some((BuiltinFn::Csc, reciprocal_arg)), Some((BuiltinFn::Sin, base_arg))) => {
            exprs_equivalent_ignoring_internal_holds_local(ctx, reciprocal_arg, base_arg)
        }
        _ => false,
    }
}

fn squared_builtin_arg_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let expected = num_rational::BigRational::from_integer(2.into());
    if cas_ast::views::as_rational_const(ctx, exp, 8).is_none_or(|value| value != expected) {
        return None;
    }

    let base = cas_ast::hold::strip_all_holds(ctx, base);
    let Expr::Function(fn_id, args) = ctx.get(base) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) {
        Some(args[0])
    } else {
        None
    }
}

fn same_power_with_equal_rational_exponent_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left = cas_ast::hold::strip_all_holds(ctx, left);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    let Expr::Pow(left_base, left_exp) = ctx.get(left).clone() else {
        return false;
    };
    let Expr::Pow(right_base, right_exp) = ctx.get(right).clone() else {
        return false;
    };
    let Some(left_exp) = cas_ast::views::as_rational_const(ctx, left_exp, 8) else {
        return false;
    };
    let Some(right_exp) = cas_ast::views::as_rational_const(ctx, right_exp, 8) else {
        return false;
    };
    left_exp == right_exp
        && exprs_equivalent_ignoring_internal_holds_local(ctx, left_base, right_base)
}

fn reciprocal_sqrt_matches_negative_half_power_local(
    ctx: &mut cas_ast::Context,
    reciprocal: ExprId,
    power: ExprId,
) -> bool {
    let reciprocal = cas_ast::hold::strip_all_holds(ctx, reciprocal);
    let power = cas_ast::hold::strip_all_holds(ctx, power);
    let Expr::Div(num, den) = ctx.get(reciprocal).clone() else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, num, 8)
        .is_none_or(|value| value != num_rational::BigRational::from_integer(1.into()))
    {
        return false;
    }
    let Some(radicand) = cas_math::root_forms::extract_square_root_base(ctx, den) else {
        return false;
    };
    let Expr::Pow(base, exp) = ctx.get(power).clone() else {
        return false;
    };
    let expected_exp = num_rational::BigRational::new((-1).into(), 2.into());
    cas_ast::views::as_rational_const(ctx, exp, 8).is_some_and(|value| value == expected_exp)
        && exprs_equivalent_ignoring_internal_holds_local(ctx, radicand, base)
}

fn fractions_equivalent_after_denominator_numeric_rescale_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left = cas_ast::hold::strip_all_holds(ctx, left);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    let (left_num, left_den) = match ctx.get(left).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    let (right_num, right_den) = match ctx.get(right).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };

    let (left_den_scale, left_den_core) =
        split_numeric_scale_product_for_residual_local(ctx, left_den);
    let (right_den_scale, right_den_core) =
        split_numeric_scale_product_for_residual_local(ctx, right_den);
    if !exprs_equivalent_ignoring_internal_holds_local(ctx, left_den_core, right_den_core)
        && !commutative_products_equivalent_ignoring_internal_holds_local(
            ctx,
            left_den_core,
            right_den_core,
        )
    {
        return false;
    }

    let scaled_left_num =
        scale_expr_for_residual_match_local(ctx, right_den_scale.clone(), left_num);
    let scaled_right_num =
        scale_expr_for_residual_match_local(ctx, left_den_scale.clone(), right_num);
    exprs_equivalent_ignoring_internal_holds_local(ctx, scaled_left_num, scaled_right_num)
        || commutative_products_equivalent_ignoring_internal_holds_local(
            ctx,
            scaled_left_num,
            scaled_right_num,
        )
        || scaled_additive_terms_equivalent_for_residual_match_local(
            ctx,
            left_num,
            right_den_scale,
            right_num,
            left_den_scale,
        )
}

fn exprs_equivalent_for_post_calculus_residual_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    exprs_equivalent_ignoring_internal_holds_local(ctx, left, right)
        || commutative_products_equivalent_ignoring_internal_holds_local(ctx, left, right)
        || fractions_equivalent_up_to_commutative_products_local(ctx, left, right)
        || fractions_equivalent_after_denominator_numeric_rescale_local(ctx, left, right)
        || fractions_cross_products_equivalent_ignoring_internal_holds_local(ctx, left, right)
}

fn sqrt_denominator_reciprocal_variant_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let mut sqrt_factor = None;
    let mut denominator_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
        if let Some(radicand) = cas_math::root_forms::extract_square_root_base(ctx, factor) {
            if sqrt_factor.replace(factor).is_some() {
                return None;
            }
            denominator_factors.push(radicand);
        } else {
            denominator_factors.push(factor);
        }
    }

    let sqrt_factor = sqrt_factor?;
    let numerator = if cas_ast::views::as_rational_const(ctx, num, 8)
        .is_some_and(|value| value == num_rational::BigRational::from_integer(1.into()))
    {
        sqrt_factor
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[num, sqrt_factor])
    };
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn residual_difference_terms_local(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right)),
        Expr::Add(_, _) => {
            let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
            if terms.len() != 2 {
                return None;
            }
            match (terms[0], terms[1]) {
                (
                    (left, cas_math::expr_nary::Sign::Pos),
                    (right, cas_math::expr_nary::Sign::Neg),
                ) => Some((left, right)),
                (
                    (left, cas_math::expr_nary::Sign::Neg),
                    (right, cas_math::expr_nary::Sign::Pos),
                ) => Some((right, left)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn negated_expr_for_local_match(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inner,
        Expr::Div(num, den) => {
            if let Expr::Neg(inner_num) = ctx.get(num) {
                return ctx.add(Expr::Div(*inner_num, den));
            }
            if let Some(value) = cas_ast::views::as_rational_const(ctx, num, 8) {
                if value < num_rational::BigRational::from_integer(0.into()) {
                    let positive = ctx.add(Expr::Number(-value));
                    return ctx.add(Expr::Div(positive, den));
                }
            }
            ctx.add(Expr::Neg(expr))
        }
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn try_diff_ln_sum_equal_derivative_roots_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;

    try_diff_ln_sum_equal_derivative_roots_residual_zero_ordered_local(ctx, left, right).or_else(
        || try_diff_ln_sum_equal_derivative_roots_residual_zero_ordered_local(ctx, right, left),
    )
}

fn try_diff_ln_sum_equal_derivative_roots_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !exprs_equivalent_ignoring_internal_holds_local(ctx, result, target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_diff_sqrt_over_positive_shifted_sqrt_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Some((left, right)) = residual_difference_terms_local(ctx, expr) {
        if let Some(zero) =
            try_diff_sqrt_over_positive_shifted_sqrt_residual_zero_ordered_local(ctx, left, right)
                .or_else(|| {
                    try_diff_sqrt_over_positive_shifted_sqrt_residual_zero_ordered_local(
                        ctx, right, left,
                    )
                })
        {
            return Some(zero);
        }
    }

    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    try_diff_sqrt_over_positive_shifted_sqrt_additive_inverse_residual_zero_ordered_local(
        ctx, left, right,
    )
    .or_else(|| {
        try_diff_sqrt_over_positive_shifted_sqrt_additive_inverse_residual_zero_ordered_local(
            ctx, right, left,
        )
    })
}

fn try_diff_sqrt_over_positive_shifted_sqrt_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::sqrt_over_positive_shifted_sqrt_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_shifted_diff_result_matches_target_local(ctx, result, target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_diff_sqrt_over_positive_shifted_sqrt_additive_inverse_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::sqrt_over_positive_shifted_sqrt_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    let negated_target = negated_expr_for_local_match(ctx, target);
    if !sqrt_shifted_diff_result_matches_target_local(ctx, result, negated_target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_diff_sqrt_polynomial_quotient_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Some((left, right)) = residual_difference_terms_local(ctx, expr) {
        return try_diff_sqrt_polynomial_quotient_residual_zero_ordered_local(ctx, left, right)
            .or_else(|| {
                try_diff_sqrt_polynomial_quotient_residual_zero_ordered_local(ctx, right, left)
            });
    }

    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    try_diff_sqrt_polynomial_quotient_residual_zero_additive_ordered_local(ctx, left, right)
        .or_else(|| {
            try_diff_sqrt_polynomial_quotient_residual_zero_additive_ordered_local(ctx, right, left)
        })
}

fn try_diff_arctan_sqrt_plus_sqrt_over_x_plus_one_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_arctan_sqrt_plus_sqrt_over_x_plus_one_residual_zero_ordered_local(ctx, left, right)
        .or_else(|| {
            try_diff_arctan_sqrt_plus_sqrt_over_x_plus_one_residual_zero_ordered_local(
                ctx, right, left,
            )
        })
        .or_else(|| {
            try_reciprocal_sqrt_x_plus_one_square_residual_zero_ordered_local(ctx, left, right)
        })
        .or_else(|| {
            try_reciprocal_sqrt_x_plus_one_square_residual_zero_ordered_local(ctx, right, left)
        })
}

fn try_reciprocal_sqrt_x_plus_one_square_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let vars = {
        let mut vars = cas_ast::collect_variables(ctx, left);
        vars.extend(cas_ast::collect_variables(ctx, right));
        vars
    };
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?.as_str();
    let left_scale = reciprocal_sqrt_x_plus_one_square_scale_local(ctx, left, var_name)?;
    let right_scale = reciprocal_sqrt_x_plus_one_square_scale_local(ctx, right, var_name)?;
    if left_scale != right_scale {
        return None;
    }
    let var = ctx.var(var_name);
    Some((ctx.num(0), vec![crate::ImplicitCondition::Positive(var)]))
}

fn try_diff_arctan_sqrt_plus_sqrt_over_x_plus_one_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let scale =
        arctan_sqrt_plus_sqrt_over_x_plus_one_scale_local(ctx, call.target, &call.var_name)?;
    if reciprocal_sqrt_x_plus_one_square_scale_local(ctx, target, &call.var_name)
        .is_some_and(|target_scale| target_scale == scale)
    {
        let var = ctx.var(&call.var_name);
        return Some((ctx.num(0), vec![crate::ImplicitCondition::Positive(var)]));
    }
    let expected = build_scaled_reciprocal_sqrt_x_plus_one_square_local(ctx, scale, &call.var_name);
    if !sqrt_shifted_diff_result_matches_target_local(ctx, expected, target) {
        return None;
    }

    let var = ctx.var(&call.var_name);
    Some((ctx.num(0), vec![crate::ImplicitCondition::Positive(var)]))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    if let Some((outer_scale, core)) = scaled_nontrivial_core_local(ctx, expr) {
        if let Some(inner_scale) =
            arctan_sqrt_plus_sqrt_over_x_plus_one_scale_local(ctx, core, var_name)
        {
            return Some(outer_scale * inner_scale);
        }
    }

    if let Some(scale) =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale_local(ctx, expr, var_name)
    {
        return Some(scale);
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let mut arctan_scale = None;
    let mut rational_scale = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_arctan_sqrt_var_scale_local(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_over_x_plus_one_scale_local(ctx, term, var_name)
        {
            if rational_scale.replace(scale).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let arctan_scale = arctan_scale?;
    let rational_scale = rational_scale?;
    (arctan_scale == rational_scale).then_some(arctan_scale)
}

fn scaled_nontrivial_core_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(num_rational::BigRational, ExprId)> {
    let (scale, core) = split_numeric_scale_one_core_local(ctx, expr)?;
    let one = num_rational::BigRational::from_integer(1.into());
    (scale != one && core != expr).then_some((scale, core))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let denominator_scale = x_plus_one_linear_scale_local(ctx, den, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    let numerator_scale =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale_local(ctx, num, var_name)?;
    Some(numerator_scale / denominator_scale)
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale_local(
    ctx: &mut cas_ast::Context,
    numerator: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, numerator);
    if terms.len() != 3 {
        return None;
    }

    let mut arctan_scale = None;
    let mut sqrt_scale = None;
    let mut x_arctan_scale = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_arctan_sqrt_var_scale_local(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_scale_local(ctx, term, var_name) {
            if sqrt_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) =
            scaled_var_times_arctan_sqrt_var_scale_local(ctx, term, var_name)
        {
            if x_arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let arctan_scale = arctan_scale?;
    (sqrt_scale? == arctan_scale && x_arctan_scale? == arctan_scale).then_some(arctan_scale)
}

fn scaled_arctan_sqrt_var_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let (scale, core) = split_numeric_scale_one_core_local(ctx, expr)?;
    is_arctan_sqrt_var_local(ctx, core, var_name).then_some(scale)
}

fn scaled_sqrt_var_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let (scale, core) = split_numeric_scale_one_core_local(ctx, expr)?;
    is_sqrt_var_local(ctx, core, var_name).then_some(scale)
}

fn scaled_var_times_arctan_sqrt_var_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let mut scale = num_rational::BigRational::from_integer(1.into());
    let mut saw_var = false;
    let mut saw_arctan = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if is_var_local(ctx, factor, var_name) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if is_arctan_sqrt_var_local(ctx, factor, var_name) {
            if saw_arctan {
                return None;
            }
            saw_arctan = true;
        } else {
            return None;
        }
    }
    (saw_var && saw_arctan).then_some(scale)
}

fn scaled_sqrt_var_over_x_plus_one_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let (outer_scale, core) = split_numeric_scale_one_core_local(ctx, expr)?;
    let core = cas_ast::hold::strip_all_holds(ctx, core);
    let (num, den) = match ctx.get(core).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    if !matches_x_plus_one_local(ctx, den, var_name) {
        return None;
    }
    let (num_scale, num_core) = split_numeric_scale_one_core_local(ctx, num)?;
    if !is_sqrt_var_local(ctx, num_core, var_name) {
        return None;
    }
    Some(outer_scale * num_scale)
}

fn split_numeric_scale_one_core_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(num_rational::BigRational, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let mut scale = num_rational::BigRational::from_integer(1.into());
    let mut core = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if core.replace(factor).is_some() {
            return None;
        }
    }
    Some((scale, core.unwrap_or(expr)))
}

fn x_plus_one_linear_scale_local(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let poly = cas_math::polynomial::Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let offset = poly.coeffs.first()?;
    let slope = poly.coeffs.get(1)?;
    (offset == slope).then_some(offset.clone())
}

fn is_arctan_sqrt_var_local(ctx: &mut cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    args.len() == 1
        && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Arctan)
        && is_sqrt_var_local(ctx, args[0], var_name)
}

fn is_sqrt_var_local(ctx: &mut cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    cas_math::root_forms::extract_square_root_base(ctx, expr).is_some_and(|radicand| {
        let radicand = cas_ast::hold::strip_all_holds(ctx, radicand);
        matches!(ctx.get(radicand), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
    })
}

fn matches_x_plus_one_local(ctx: &mut cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let var = ctx.var(var_name);
    let one = ctx.num(1);
    let expected = ctx.add(Expr::Add(var, one));
    exprs_equivalent_ignoring_internal_holds_local(ctx, expr, expected)
}

fn matches_x_plus_one_square_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return false;
    };
    cas_ast::views::as_rational_const(ctx, exp, 8)
        .is_some_and(|value| value == num_rational::BigRational::from_integer(2.into()))
        && matches_x_plus_one_local(ctx, base, var_name)
}

fn is_var_local(ctx: &mut cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

fn sqrt_x_plus_one_square_denominator_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let mut scale = num_rational::BigRational::from_integer(1.into());
    let mut saw_sqrt = false;
    let mut saw_square = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if is_sqrt_var_local(ctx, factor, var_name) {
            if saw_sqrt {
                return None;
            }
            saw_sqrt = true;
        } else if matches_x_plus_one_square_local(ctx, factor, var_name) {
            if saw_square {
                return None;
            }
            saw_square = true;
        } else {
            scale *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
        }
    }
    (saw_sqrt && saw_square).then_some(scale)
}

fn var_times_x_plus_one_square_denominator_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let mut scale = num_rational::BigRational::from_integer(1.into());
    let mut saw_var = false;
    let mut saw_square = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if is_var_local(ctx, factor, var_name) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if matches_x_plus_one_square_local(ctx, factor, var_name) {
            if saw_square {
                return None;
            }
            saw_square = true;
        } else {
            scale *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
        }
    }
    (saw_var && saw_square).then_some(scale)
}

fn reciprocal_sqrt_x_plus_one_square_scale_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    if let Some(den_scale) = sqrt_x_plus_one_square_denominator_scale_local(ctx, den, var_name) {
        let num_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
        return Some(num_scale / den_scale);
    }

    let den_scale = var_times_x_plus_one_square_denominator_scale_local(ctx, den, var_name)?;
    let (num_scale, num_core) = split_numeric_scale_one_core_local(ctx, num)?;
    is_sqrt_var_local(ctx, num_core, var_name).then_some(num_scale / den_scale)
}

fn build_scaled_reciprocal_sqrt_x_plus_one_square_local(
    ctx: &mut cas_ast::Context,
    scale: num_rational::BigRational,
    var_name: &str,
) -> ExprId {
    let var = ctx.var(var_name);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var]);
    let var = ctx.var(var_name);
    let one = ctx.num(1);
    let linear = ctx.add(Expr::Add(var, one));
    let two = ctx.num(2);
    let linear_square = ctx.add(Expr::Pow(linear, two));
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_var, linear_square]);
    let numerator = ctx.add(Expr::Number(scale));
    ctx.add(Expr::Div(numerator, denominator))
}

fn try_diff_sqrt_polynomial_quotient_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::sqrt_polynomial_quotient_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_shifted_diff_result_matches_target_local(ctx, result, target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_diff_sqrt_polynomial_quotient_residual_zero_additive_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::sqrt_polynomial_quotient_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    let negated_target = negated_expr_for_local_match(ctx, target);
    if !sqrt_shifted_diff_result_matches_target_local(ctx, result, negated_target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_diff_sqrt_additive_tan_polynomial_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_sqrt_additive_tan_polynomial_residual_zero_ordered_local(ctx, left, right).or_else(
        || try_diff_sqrt_additive_tan_polynomial_residual_zero_ordered_local(ctx, right, left),
    )
}

fn try_diff_sqrt_additive_tan_polynomial_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, radicand, mut required_conditions) =
        crate::rules::calculus::sqrt_additive_tan_polynomial_derivative_presentation(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_denominator_common_factor_target_variant_exact_local(ctx, result, target)
        && !sqrt_denominator_common_factor_target_variant_exact_local(ctx, target, result)
        && !sqrt_shifted_diff_result_matches_target_local(ctx, result, target)
        && !sqrt_additive_tan_sqrt_variable_sec_target_variant_matches_local(
            ctx,
            call.target,
            &call.var_name,
            target,
        )
    {
        return None;
    }

    required_conditions.insert(0, crate::ImplicitCondition::Positive(radicand));
    Some((ctx.num(0), required_conditions))
}

fn try_diff_arctan_sqrt_additive_tan_polynomial_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_arctan_sqrt_additive_tan_polynomial_residual_zero_ordered_local(ctx, left, right)
        .or_else(|| {
            try_diff_arctan_sqrt_additive_tan_polynomial_residual_zero_ordered_local(
                ctx, right, left,
            )
        })
}

fn try_diff_arctan_sqrt_additive_tan_polynomial_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::arctan_sqrt_additive_tan_polynomial_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_shifted_diff_result_matches_target_local(ctx, result, target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_diff_arctan_sqrt_additive_trig_polynomial_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_arctan_sqrt_additive_trig_polynomial_residual_zero_ordered_local(ctx, left, right)
        .or_else(|| {
            try_diff_arctan_sqrt_additive_trig_polynomial_residual_zero_ordered_local(
                ctx, right, left,
            )
        })
}

fn try_diff_arctan_sqrt_additive_trig_polynomial_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::arctan_sqrt_additive_trig_polynomial_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_shifted_diff_result_matches_target_local(ctx, result, target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn sqrt_additive_tan_sqrt_variable_sec_target_variant_matches_local(
    ctx: &mut cas_ast::Context,
    diff_target: ExprId,
    var_name: &str,
    target: ExprId,
) -> bool {
    let Some(radicand) = cas_math::root_forms::extract_square_root_base(ctx, diff_target) else {
        return false;
    };
    let Some(linear_scale) =
        tan_plus_sqrt_var_plus_affine_linear_scale_local(ctx, radicand, var_name)
    else {
        return false;
    };

    let var = ctx.var(var_name);
    let one_for_half = ctx.num(1);
    let two_for_half = ctx.num(2);
    let half = ctx.add(Expr::Div(one_for_half, two_for_half));
    let sqrt_var = ctx.add(Expr::Pow(var, half));
    let sec = ctx.call_builtin(BuiltinFn::Sec, vec![var]);
    let two = ctx.num(2);
    let sec_square = ctx.add(Expr::Pow(sec, two));
    let two = ctx.num(2);
    let tan_term = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_var, sec_square]);
    let one = ctx.num(1);
    let linear_scale = linear_scale * num_rational::BigRational::from_integer(2.into());
    let linear_term = scale_expr_for_residual_match_local(ctx, linear_scale, sqrt_var);
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &[tan_term, one, linear_term]);

    let four = ctx.num(4);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, sqrt_var, sqrt_radicand]);
    let variant = ctx.add(Expr::Div(numerator, denominator));
    exprs_equivalent_for_post_calculus_residual_local(ctx, variant, target)
}

fn tan_plus_sqrt_var_plus_affine_linear_scale_local(
    ctx: &mut cas_ast::Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<num_rational::BigRational> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if !(3..=4).contains(&terms.len()) {
        return None;
    }

    let mut saw_tan = false;
    let mut saw_sqrt = false;
    let mut linear_scale = num_rational::BigRational::from_integer(0.into());
    for (term, sign) in terms {
        if sign == cas_math::expr_nary::Sign::Neg {
            return None;
        }
        if matches_tan_var_local(ctx, term, var_name) {
            if saw_tan {
                return None;
            }
            saw_tan = true;
        } else if is_sqrt_var_local(ctx, term, var_name) {
            if saw_sqrt {
                return None;
            }
            saw_sqrt = true;
        } else {
            let (scale, core) = split_numeric_scale_product_for_residual_local(ctx, term);
            if is_var_local(ctx, core, var_name) {
                linear_scale += scale;
            } else if cas_ast::views::as_rational_const(ctx, term, 8).is_none() {
                return None;
            }
        }
    }

    if saw_tan && saw_sqrt && !linear_scale.is_zero() {
        Some(linear_scale)
    } else {
        None
    }
}

fn matches_tan_var_local(ctx: &mut cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    args.len() == 1
        && ctx.is_builtin(*fn_id, BuiltinFn::Tan)
        && is_var_local(ctx, args[0], var_name)
}

fn try_diff_sqrt_additive_trig_polynomial_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_sqrt_additive_trig_polynomial_residual_zero_ordered_local(ctx, left, right).or_else(
        || try_diff_sqrt_additive_trig_polynomial_residual_zero_ordered_local(ctx, right, left),
    )
}

fn try_diff_sqrt_additive_trig_polynomial_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, radicand, mut required_conditions) =
        crate::rules::calculus::sqrt_additive_trig_polynomial_derivative_presentation(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_denominator_common_factor_target_variant_exact_local(ctx, result, target)
        && !sqrt_denominator_common_factor_target_variant_exact_local(ctx, target, result)
        && !sqrt_shifted_diff_result_matches_target_local(ctx, result, target)
    {
        return None;
    }

    required_conditions.insert(0, crate::ImplicitCondition::Positive(radicand));
    Some((ctx.num(0), required_conditions))
}

fn sqrt_shifted_diff_result_matches_target_local(
    ctx: &mut cas_ast::Context,
    result: ExprId,
    target: ExprId,
) -> bool {
    exprs_equivalent_for_post_calculus_residual_local(ctx, result, target)
        || sqrt_denominator_reciprocal_variant_local(ctx, result).is_some_and(|variant| {
            exprs_equivalent_for_post_calculus_residual_local(ctx, variant, target)
        })
        || sqrt_denominator_common_factor_target_variant_local(ctx, result, target)
}

fn sqrt_denominator_common_factor_target_variant_local(
    ctx: &mut cas_ast::Context,
    result: ExprId,
    target: ExprId,
) -> bool {
    let result = cas_ast::hold::strip_all_holds(ctx, result);
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let (result_num, result_den) = match ctx.get(result).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    let (target_num, target_den) = match ctx.get(target).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    if !denominator_has_sqrt_factor_local(ctx, target_den) {
        return false;
    }

    let (result_den_scale, result_den_core) =
        split_numeric_scale_product_for_residual_local(ctx, result_den);
    let (target_den_scale, target_den_core) =
        split_numeric_scale_product_for_residual_local(ctx, target_den);
    if target_den_scale.is_zero() {
        return false;
    }
    let extra_scale = result_den_scale / target_den_scale;
    let Some(extra_factors) =
        remove_factor_multiset_for_residual_match_local(ctx, result_den_core, target_den_core)
    else {
        return false;
    };
    if extra_factors.is_empty() || extra_factors.len() > 4 {
        return false;
    }

    let Some(scaled_target_num) = multiply_additive_numerator_by_extra_factors_local(
        ctx,
        target_num,
        extra_scale,
        &extra_factors,
    ) else {
        return false;
    };

    exprs_equivalent_for_post_calculus_residual_local(ctx, result_num, scaled_target_num)
        || scaled_additive_terms_equivalent_for_residual_match_local(
            ctx,
            result_num,
            num_rational::BigRational::from_integer(1.into()),
            scaled_target_num,
            num_rational::BigRational::from_integer(1.into()),
        )
}

fn sqrt_denominator_common_factor_target_variant_exact_local(
    ctx: &mut cas_ast::Context,
    result: ExprId,
    target: ExprId,
) -> bool {
    let result = cas_ast::hold::strip_all_holds(ctx, result);
    let target = cas_ast::hold::strip_all_holds(ctx, target);
    let (result_num, result_den) = match ctx.get(result).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    let (target_num, target_den) = match ctx.get(target).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    if !denominator_has_sqrt_factor_local(ctx, result_den)
        || !denominator_has_sqrt_factor_local(ctx, target_den)
    {
        return false;
    }

    let (result_den_scale, result_den_core) =
        split_numeric_scale_product_for_residual_local(ctx, result_den);
    let (target_den_scale, target_den_core) =
        split_numeric_scale_product_for_residual_local(ctx, target_den);
    if target_den_scale.is_zero() {
        return false;
    }
    let extra_scale = result_den_scale / target_den_scale;
    let Some(extra_factors) = remove_factor_multiset_exact_for_residual_match_local(
        ctx,
        result_den_core,
        target_den_core,
    ) else {
        return false;
    };
    if extra_factors.is_empty() || extra_factors.len() > 4 {
        return false;
    }

    let Some(scaled_target_num) = multiply_additive_numerator_by_extra_factors_exact_local(
        ctx,
        target_num,
        extra_scale,
        &extra_factors,
    ) else {
        return false;
    };
    exprs_equivalent_for_post_calculus_residual_local(ctx, result_num, scaled_target_num)
        || scaled_additive_terms_equivalent_for_residual_match_local(
            ctx,
            result_num,
            num_rational::BigRational::from_integer(1.into()),
            scaled_target_num,
            num_rational::BigRational::from_integer(1.into()),
        )
}

fn denominator_has_sqrt_factor_local(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    cas_math::expr_nary::mul_leaves(ctx, expr)
        .into_iter()
        .any(|factor| cas_math::root_forms::extract_square_root_base(ctx, factor).is_some())
}

fn non_unit_mul_factors_for_residual_match_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<ExprId> {
    let one = num_rational::BigRational::from_integer(1.into());
    if cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value == one) {
        return Vec::new();
    }
    cas_math::expr_nary::mul_leaves(ctx, expr)
        .into_iter()
        .filter(|factor| {
            cas_ast::views::as_rational_const(ctx, *factor, 8).is_none_or(|value| value != one)
        })
        .collect()
}

fn remove_factor_multiset_for_residual_match_local(
    ctx: &mut cas_ast::Context,
    superset: ExprId,
    subset: ExprId,
) -> Option<Vec<ExprId>> {
    let mut remaining = non_unit_mul_factors_for_residual_match_local(ctx, superset);
    for subset_factor in non_unit_mul_factors_for_residual_match_local(ctx, subset) {
        let index = remaining.iter().position(|factor| {
            exprs_equivalent_ignoring_internal_holds_local(ctx, *factor, subset_factor)
                || commutative_products_equivalent_ignoring_internal_holds_local(
                    ctx,
                    *factor,
                    subset_factor,
                )
        })?;
        remaining.remove(index);
    }
    Some(remaining)
}

fn remove_factor_multiset_exact_for_residual_match_local(
    ctx: &mut cas_ast::Context,
    superset: ExprId,
    subset: ExprId,
) -> Option<Vec<ExprId>> {
    let mut remaining = non_unit_mul_factors_for_residual_match_local(ctx, superset);
    for subset_factor in non_unit_mul_factors_for_residual_match_local(ctx, subset) {
        let index = remaining.iter().position(|factor| {
            exprs_exact_ignoring_internal_holds_local(ctx, *factor, subset_factor)
                || sqrt_factors_have_signed_additive_radicands_equivalent_local(
                    ctx,
                    *factor,
                    subset_factor,
                )
        })?;
        remaining.remove(index);
    }
    Some(remaining)
}

fn quotient_factor_parts_for_residual_match_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> (Vec<ExprId>, Vec<ExprId>) {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let one = num_rational::BigRational::from_integer(1.into());
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let numerator_factors = if cas_ast::views::as_rational_const(ctx, num, 8)
                .is_some_and(|value| value == one)
            {
                Vec::new()
            } else {
                non_unit_mul_factors_for_residual_match_local(ctx, num)
            };
            (
                numerator_factors,
                non_unit_mul_factors_for_residual_match_local(ctx, den),
            )
        }
        _ => (
            non_unit_mul_factors_for_residual_match_local(ctx, expr),
            Vec::new(),
        ),
    }
}

fn multiply_additive_numerator_by_extra_factors_local(
    ctx: &mut cas_ast::Context,
    numerator: ExprId,
    extra_scale: num_rational::BigRational,
    extra_factors: &[ExprId],
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, numerator);
    if terms.is_empty() || terms.len() > 8 {
        return None;
    }

    let mut scaled_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (term_scale, term_core) = split_numeric_scale_product_for_residual_local(ctx, term);
        let mut scale = extra_scale.clone() * term_scale;
        if sign == cas_math::expr_nary::Sign::Neg {
            scale = -scale;
        }
        let (mut numerator_factors, denominator_factors) =
            quotient_factor_parts_for_residual_match_local(ctx, term_core);
        let mut remaining_extra = extra_factors.to_vec();
        for denominator_factor in denominator_factors {
            let index = remaining_extra.iter().position(|factor| {
                exprs_equivalent_ignoring_internal_holds_local(ctx, *factor, denominator_factor)
                    || commutative_products_equivalent_ignoring_internal_holds_local(
                        ctx,
                        *factor,
                        denominator_factor,
                    )
            })?;
            remaining_extra.remove(index);
        }
        numerator_factors.extend(remaining_extra);
        let core = if numerator_factors.is_empty() {
            ctx.num(1)
        } else {
            cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors)
        };
        scaled_terms.push(scale_expr_for_residual_match_local(ctx, scale, core));
    }

    Some(cas_math::expr_nary::build_balanced_add(ctx, &scaled_terms))
}

fn multiply_additive_numerator_by_extra_factors_exact_local(
    ctx: &mut cas_ast::Context,
    numerator: ExprId,
    extra_scale: num_rational::BigRational,
    extra_factors: &[ExprId],
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, numerator);
    if terms.is_empty() || terms.len() > 8 {
        return None;
    }

    let mut scaled_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (term_scale, term_core) = split_numeric_scale_product_for_residual_local(ctx, term);
        let mut scale = extra_scale.clone() * term_scale;
        if sign == cas_math::expr_nary::Sign::Neg {
            scale = -scale;
        }
        let (mut numerator_factors, denominator_factors) =
            quotient_factor_parts_for_residual_match_local(ctx, term_core);
        let mut remaining_extra = extra_factors.to_vec();
        for denominator_factor in denominator_factors {
            let index = remaining_extra.iter().position(|factor| {
                exprs_exact_ignoring_internal_holds_local(ctx, *factor, denominator_factor)
            })?;
            remaining_extra.remove(index);
        }
        let mut kept_numerator_factors = Vec::with_capacity(numerator_factors.len());
        for numerator_factor in numerator_factors {
            let Some(index) = remaining_extra.iter().position(|extra_factor| {
                sqrt_factor_matches_negative_half_power_local(ctx, *extra_factor, numerator_factor)
                    || sqrt_factor_matches_negative_half_power_local(
                        ctx,
                        numerator_factor,
                        *extra_factor,
                    )
            }) else {
                kept_numerator_factors.push(numerator_factor);
                continue;
            };
            remaining_extra.remove(index);
        }
        numerator_factors = kept_numerator_factors;
        numerator_factors.extend(remaining_extra);
        let core = if numerator_factors.is_empty() {
            ctx.num(1)
        } else {
            cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors)
        };
        scaled_terms.push(scale_expr_for_residual_match_local(ctx, scale, core));
    }

    Some(cas_math::expr_nary::build_balanced_add(ctx, &scaled_terms))
}

fn sqrt_factor_matches_negative_half_power_local(
    ctx: &mut cas_ast::Context,
    sqrt_factor: ExprId,
    power_factor: ExprId,
) -> bool {
    let Some(sqrt_base) = cas_math::root_forms::extract_square_root_base(ctx, sqrt_factor) else {
        return false;
    };
    let power_factor = cas_ast::hold::strip_all_holds(ctx, power_factor);
    let Expr::Pow(power_base, power_exp) = ctx.get(power_factor).clone() else {
        return false;
    };
    let expected_exp = num_rational::BigRational::new((-1).into(), 2.into());
    cas_ast::views::as_rational_const(ctx, power_exp, 8).is_some_and(|value| value == expected_exp)
        && exprs_equivalent_ignoring_internal_holds_local(ctx, sqrt_base, power_base)
}

fn exprs_exact_ignoring_internal_holds_local(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left = cas_ast::hold::strip_all_holds(ctx, left);
    let right = cas_ast::hold::strip_all_holds(ctx, right);
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn expr_is_named_function_call_local(ctx: &cas_ast::Context, expr: ExprId, names: &[&str]) -> bool {
    let Expr::Function(fn_id, _) = ctx.get(expr) else {
        return false;
    };
    names.iter().any(|name| ctx.sym_name(*fn_id) == *name)
}

fn expr_contains_named_function_local(
    ctx: &cas_ast::Context,
    root: ExprId,
    names: &[&str],
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if names.iter().any(|name| ctx.sym_name(*fn_id) == *name) {
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

fn expr_is_symbolic_calculus_call_local(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr).is_some()
        || crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, expr).is_some()
}

fn expr_contains_symbolic_calculus_call_local(ctx: &cas_ast::Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        if expr_is_symbolic_calculus_call_local(ctx, expr) {
            return true;
        }
        match ctx.get(expr) {
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

fn expr_is_post_calculus_residual_candidate_local(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_) => true,
        Expr::Div(num, den) => {
            cas_math::numeric_eval::as_rational_const(ctx, *den)
                .is_none_or(|value| value != num_rational::BigRational::from_integer(0.into()))
                && expr_is_post_calculus_residual_candidate_local(ctx, *num)
        }
        Expr::Mul(left, right) => {
            (expr_contains_named_function_local(ctx, *left, &["diff", "integrate", "int"])
                || expr_contains_symbolic_calculus_call_local(ctx, *left))
                && expr_is_post_calculus_residual_candidate_local(ctx, *left)
                || (expr_contains_named_function_local(ctx, *right, &["diff", "integrate", "int"])
                    || expr_contains_symbolic_calculus_call_local(ctx, *right))
                    && expr_is_post_calculus_residual_candidate_local(ctx, *right)
        }
        _ => false,
    }
}

fn expr_is_zero_const_local(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value.is_zero())
}

fn expr_is_exact_zero_identity_local(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    expr_is_zero_const_local(ctx, expr)
        || crate::rules::arithmetic::try_build_exact_zero_identity_rewrite(ctx, expr)
            .is_some_and(|rewrite| expr_is_zero_const_local(ctx, rewrite.final_expr()))
}

fn try_resolve_post_calculus_residual_before_general_simplify_core_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    crate::calculus_residual_support::try_integrate_resolved_reciprocal_half_power_residual_root_zero(
        ctx,
        expr,
    )
    .or_else(|| {
        crate::calculus_residual_support::try_diff_integral_exp_trig_residual_compact_mismatch(
            ctx, expr,
        )
    })
    .or_else(|| {
        crate::calculus_residual_support::try_diff_integral_exp_trig_residual_root_zero(ctx, expr)
    })
    .or_else(|| try_diff_sqrt_small_additive_elementary_residual_zero_local(ctx, expr))
    .or_else(|| try_diff_arctan_sqrt_small_additive_elementary_residual_zero_local(ctx, expr))
    .or_else(|| try_diff_arctan_sqrt_additive_trig_polynomial_residual_zero_local(ctx, expr))
    .or_else(|| try_diff_arctan_sqrt_additive_tan_polynomial_residual_zero_local(ctx, expr))
    .or_else(|| try_diff_sqrt_additive_tan_polynomial_residual_zero_local(ctx, expr))
    .or_else(|| try_diff_sqrt_additive_trig_polynomial_residual_zero_local(ctx, expr))
    .or_else(|| try_diff_integrate_arctan_sqrt_unit_shift_square_residual_zero_local(ctx, expr))
    .or_else(|| {
        try_diff_arctan_sqrt_plus_sqrt_over_x_plus_one_residual_zero_local(ctx, expr)
    })
}

fn try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    depth: usize,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Some(result) =
        try_diff_arctan_sqrt_small_additive_elementary_residual_zero_local(ctx, expr)
    {
        return Some(result);
    }
    if let Some(result) =
        try_diff_arctan_sqrt_additive_trig_polynomial_residual_zero_local(ctx, expr)
    {
        return Some(result);
    }
    try_rewrite_post_calculus_residual_child_context_before_general_simplify_local(ctx, expr, depth)
}

fn build_additive_remainder_for_post_calculus_residual_local(
    ctx: &mut cas_ast::Context,
    terms: &[(ExprId, cas_math::expr_nary::Sign)],
    removed_left: usize,
    removed_right: usize,
) -> ExprId {
    let mut remaining = Vec::new();
    for (index, (term, sign)) in terms.iter().enumerate() {
        if index == removed_left || index == removed_right {
            continue;
        }
        remaining.push(if *sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(*term))
        } else {
            *term
        });
    }

    match remaining.as_slice() {
        [] => ctx.num(0),
        [only] => *only,
        _ => cas_math::expr_nary::build_balanced_add(ctx, &remaining),
    }
}

fn try_resolve_post_calculus_residual_pair_inside_additive_context_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 3 || terms.len() > 6 {
        return None;
    }

    for left_index in 0..terms.len() {
        for right_index in 0..terms.len() {
            if left_index == right_index || terms[left_index].1 == terms[right_index].1 {
                continue;
            }

            let (positive_term, negative_term) =
                if terms[left_index].1 == cas_math::expr_nary::Sign::Pos {
                    (terms[left_index].0, terms[right_index].0)
                } else {
                    (terms[right_index].0, terms[left_index].0)
                };
            let pair_contains_calculus =
                expr_contains_named_function_local(
                    ctx,
                    positive_term,
                    &["diff", "integrate", "int"],
                ) || expr_contains_symbolic_calculus_call_local(ctx, positive_term)
                    || expr_contains_named_function_local(
                        ctx,
                        negative_term,
                        &["diff", "integrate", "int"],
                    )
                    || expr_contains_symbolic_calculus_call_local(ctx, negative_term);
            if !pair_contains_calculus {
                continue;
            }

            let residual_pair = ctx.add(Expr::Sub(positive_term, negative_term));
            let Some((rewritten_pair, required_conditions)) =
                try_resolve_post_calculus_residual_before_general_simplify_core_local(
                    ctx,
                    residual_pair,
                )
            else {
                continue;
            };
            if !expr_is_zero_const_local(ctx, rewritten_pair) {
                continue;
            }

            let rebuilt = build_additive_remainder_for_post_calculus_residual_local(
                ctx,
                &terms,
                left_index,
                right_index,
            );
            return Some((rebuilt, required_conditions));
        }
    }

    None
}

fn try_rewrite_post_calculus_residual_child_context_before_general_simplify_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    depth: usize,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if depth > 8 {
        return None;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if let Some((rewritten, required_conditions)) =
                try_resolve_post_calculus_residual_pair_inside_additive_context_local(ctx, expr)
            {
                return Some((rewritten, required_conditions));
            }

            if let Some((rewritten_left, required_conditions)) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    left,
                    depth + 1,
                )
            {
                if expr_is_zero_const_local(ctx, rewritten_left)
                    || expr_is_exact_zero_identity_local(ctx, right)
                {
                    return Some((right, required_conditions));
                }
                let rebuilt = ctx.add(Expr::Add(rewritten_left, right));
                return Some((rebuilt, required_conditions));
            }

            if let Some((rewritten_right, required_conditions)) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    right,
                    depth + 1,
                )
            {
                if expr_is_zero_const_local(ctx, rewritten_right)
                    || expr_is_exact_zero_identity_local(ctx, left)
                {
                    return Some((left, required_conditions));
                }
                let rebuilt = ctx.add(Expr::Add(left, rewritten_right));
                return Some((rebuilt, required_conditions));
            }
            None
        }
        Expr::Sub(left, right) => {
            if let Some((rewritten_left, required_conditions)) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    left,
                    depth + 1,
                )
            {
                if expr_is_zero_const_local(ctx, rewritten_left) {
                    let negated = ctx.add(Expr::Neg(right));
                    return Some((negated, required_conditions));
                }
                if expr_is_exact_zero_identity_local(ctx, right) {
                    return Some((rewritten_left, required_conditions));
                }
                let rebuilt = ctx.add(Expr::Sub(rewritten_left, right));
                return Some((rebuilt, required_conditions));
            }

            if let Some((rewritten_right, required_conditions)) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    right,
                    depth + 1,
                )
            {
                if expr_is_zero_const_local(ctx, rewritten_right) {
                    return Some((left, required_conditions));
                }
                if expr_is_exact_zero_identity_local(ctx, left) {
                    let negated = ctx.add(Expr::Neg(rewritten_right));
                    return Some((negated, required_conditions));
                }
                let rebuilt = ctx.add(Expr::Sub(left, rewritten_right));
                return Some((rebuilt, required_conditions));
            }
            None
        }
        Expr::Mul(left, right) => {
            if let Some((rewritten_left, required_conditions)) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    left,
                    depth + 1,
                )
            {
                let rebuilt = ctx.add(Expr::Mul(rewritten_left, right));
                return Some((rebuilt, required_conditions));
            }
            if let Some((rewritten_right, required_conditions)) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    right,
                    depth + 1,
                )
            {
                let rebuilt = ctx.add(Expr::Mul(left, rewritten_right));
                return Some((rebuilt, required_conditions));
            }
            None
        }
        Expr::Div(num, den) => {
            let (rewritten_num, mut required_conditions) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    num,
                    depth + 1,
                )?;
            required_conditions.push(crate::ImplicitCondition::NonZero(den));
            let rebuilt = ctx.add(Expr::Div(rewritten_num, den));
            Some((rebuilt, required_conditions))
        }
        Expr::Neg(inner) => {
            let (rewritten_inner, required_conditions) =
                try_resolve_or_rewrite_post_calculus_residual_child_before_general_simplify_local(
                    ctx,
                    inner,
                    depth + 1,
                )?;
            let rebuilt = ctx.add(Expr::Neg(rewritten_inner));
            Some((rebuilt, required_conditions))
        }
        _ => None,
    }
}

fn try_resolve_post_calculus_residual_before_general_simplify_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_resolve_post_calculus_residual_before_general_simplify_core_local(ctx, expr)
}

fn try_diff_sqrt_small_additive_elementary_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_sqrt_small_additive_elementary_residual_zero_ordered_local(ctx, left, right).or_else(
        || try_diff_sqrt_small_additive_elementary_residual_zero_ordered_local(ctx, right, left),
    )
}

fn try_diff_sqrt_small_additive_elementary_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::sqrt_small_additive_elementary_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_denominator_common_factor_target_variant_exact_local(ctx, result, target)
        && !sqrt_denominator_common_factor_target_variant_exact_local(ctx, target, result)
        && !sqrt_shifted_diff_result_matches_target_local(ctx, result, target)
    {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_diff_arctan_sqrt_small_additive_elementary_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_arctan_sqrt_small_additive_elementary_residual_zero_ordered_local(ctx, left, right)
        .or_else(|| {
            try_diff_arctan_sqrt_small_additive_elementary_residual_zero_ordered_local(
                ctx, right, left,
            )
        })
}

fn try_diff_arctan_sqrt_small_additive_elementary_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (result, required_conditions) =
        crate::rules::calculus::arctan_sqrt_small_additive_elementary_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if !sqrt_shifted_diff_result_matches_target_local(ctx, result, target) {
        return None;
    }

    Some((ctx.num(0), required_conditions))
}

fn try_resolve_direct_post_calculus_before_general_simplify_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let (result, radicand, mut required_conditions) =
        crate::rules::calculus::sqrt_additive_tan_polynomial_derivative_presentation(
            ctx,
            call.target,
            &call.var_name,
        )?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(radicand));
    Some((result, required_conditions))
}

fn try_diff_integrate_arctan_sqrt_unit_shift_square_residual_zero_local(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = residual_difference_terms_local(ctx, expr)?;
    try_diff_integrate_arctan_sqrt_unit_shift_square_residual_zero_ordered_local(ctx, left, right)
        .or_else(|| {
            try_diff_integrate_arctan_sqrt_unit_shift_square_residual_zero_ordered_local(
                ctx, right, left,
            )
        })
}

fn try_diff_integrate_arctan_sqrt_unit_shift_square_residual_zero_ordered_local(
    ctx: &mut cas_ast::Context,
    diff_expr: ExprId,
    target: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }
    let source_key = cas_math::symbolic_integration_support::integrate_symbolic_arctan_sqrt_var_unit_shift_square_key(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )?;
    let target_key = cas_math::symbolic_integration_support::integrate_symbolic_arctan_sqrt_var_unit_shift_square_key(
        ctx,
        target,
        &integrate_call.var_name,
    )?;
    if source_key != target_key {
        return None;
    }
    let var = ctx.var(&integrate_call.var_name);
    Some((ctx.num(0), vec![crate::ImplicitCondition::Positive(var)]))
}

fn collapse_redundant_post_calculus_trace_if_direct_step_is_compact(
    ctx: &mut cas_ast::Context,
    resolved: ExprId,
    presented: ExprId,
    steps: &mut Vec<crate::Step>,
) -> bool {
    let Some(first) = steps.first() else {
        return false;
    };
    if !matches!(
        first.rule_name.as_str(),
        "Symbolic Differentiation" | "Symbolic Integration"
    ) {
        return false;
    }

    let first_presented =
        crate::rules::calculus::try_post_calculus_presentation(ctx, resolved, first.after)
            .unwrap_or(first.after);

    let first_presented_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: first_presented,
        }
    );
    let final_presented_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: presented,
        }
    );
    if first_presented_display != final_presented_display {
        return false;
    }

    steps.truncate(1);
    if let Some(first) = steps.first_mut() {
        first.after = presented;
        first.global_after = Some(presented);
    }
    true
}

impl Engine {
    /// Handle `EvalAction::Expand`.
    ///
    /// Actualmente usa el mismo simplificador principal; se mantiene separado
    /// para que el dispatch quede desacoplado del detalle de implementación.
    pub(super) fn eval_expand(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        let (res, steps) = self.simplifier.simplify(resolved);
        let warnings = collect_domain_warnings(
            &self.simplifier.context,
            options.shared.semantics.value_domain,
            res,
            &steps,
        );
        let rewrite_required = self.simplifier.take_required_conditions();
        Ok((
            EvalResult::Expr(res),
            warnings,
            steps,
            vec![],
            vec![],
            vec![],
            vec![],
            rewrite_required,
        ))
    }

    /// Handle `EvalAction::Simplify`: tool dispatch, simplification, const fold, domain classification.
    pub(super) fn eval_simplify(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        let effective_opts = self.effective_options(options, resolved);

        if let Some(call) = crate::symbolic_calculus_call_support::try_extract_integrate_call(
            &self.simplifier.context,
            resolved,
        ) {
            if let Some((result, required_nonzero)) =
                cas_math::symbolic_integration_support::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
                    &mut self.simplifier.context,
                    call.target,
                    &call.var_name,
                )
            {
                let desc = crate::symbolic_calculus_call_support::render_integrate_desc_with(
                    &call,
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &self.simplifier.context,
                                id
                            }
                        )
                    },
                );
                let required_condition = crate::ImplicitCondition::NonZero(required_nonzero);
                let mut steps = Vec::new();
                if effective_opts.steps_mode != crate::options::StepsMode::Off {
                    let mut step = crate::Step::new(
                        "Symbolic Integration",
                        &desc,
                        resolved,
                        result,
                        Vec::new(),
                        Some(&self.simplifier.context),
                    );
                    step.meta_mut()
                        .required_conditions
                        .push(required_condition.clone());
                    steps.push(step);
                }

                return Ok((
                    crate::EvalResult::Expr(result),
                    Vec::new(),
                    steps,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    vec![required_condition],
                ));
            }
        }

        if let Some(call) = crate::symbolic_calculus_call_support::try_extract_diff_call(
            &self.simplifier.context,
            resolved,
        ) {
            if let Some((result, required_conditions)) =
                crate::rules::calculus::arctan_sqrt_additive_tan_polynomial_derivative_presentation_with_domain(
                    &mut self.simplifier.context,
                    call.target,
                    &call.var_name,
                )
            {
                let desc = crate::symbolic_calculus_call_support::render_diff_desc_with(
                    &call,
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &self.simplifier.context,
                                id
                            }
                        )
                    },
                );
                let mut steps = Vec::new();
                if effective_opts.steps_mode != crate::options::StepsMode::Off {
                    let mut step = crate::Step::new(
                        &desc,
                        "Symbolic Differentiation",
                        resolved,
                        result,
                        Vec::new(),
                        Some(&self.simplifier.context),
                    );
                    step.meta_mut()
                        .required_conditions
                        .extend(required_conditions.iter().cloned());
                    steps.push(step);
                }

                return Ok((
                    crate::EvalResult::Expr(result),
                    Vec::new(),
                    steps,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    required_conditions,
                ));
            }
            if effective_opts.steps_mode != crate::options::StepsMode::Off
                && crate::rules::calculus::sqrt_polynomial_quotient_has_powered_expanded_affine_square_denominator(
                    &mut self.simplifier.context,
                    call.target,
                    &call.var_name,
                )
            {
                if let Some((result, required_conditions)) =
                    crate::rules::calculus::sqrt_polynomial_quotient_derivative_presentation_with_domain(
                        &mut self.simplifier.context,
                        call.target,
                        &call.var_name,
                    )
                {
                    let desc = crate::symbolic_calculus_call_support::render_diff_desc_with(
                        &call,
                        |id| {
                            format!(
                                "{}",
                                cas_formatter::DisplayExpr {
                                    context: &self.simplifier.context,
                                    id
                                }
                            )
                        },
                    );
                    let raw_derivative =
                        cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
                            &mut self.simplifier.context,
                            call.target,
                            &call.var_name,
                        )
                        .unwrap_or(result);
                    let mut steps = Vec::new();
                    let mut diff_step = crate::Step::new(
                        &desc,
                        "Symbolic Differentiation",
                        resolved,
                        raw_derivative,
                        Vec::new(),
                        Some(&self.simplifier.context),
                    );
                    diff_step
                        .meta_mut()
                        .required_conditions
                        .extend(required_conditions.iter().cloned());
                    diff_step.importance = crate::ImportanceLevel::Medium;
                    steps.push(diff_step);
                    if raw_derivative != result {
                        let mut presentation_step = crate::Step::new(
                            "Post-calculus presentation",
                            "Present calculus result in compact form",
                            raw_derivative,
                            result,
                            Vec::new(),
                            Some(&self.simplifier.context),
                        );
                        presentation_step.importance = crate::ImportanceLevel::Medium;
                        steps.push(presentation_step);
                    }

                    return Ok((
                        crate::EvalResult::Expr(result),
                        Vec::new(),
                        steps,
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        required_conditions,
                    ));
                }
            }
            if effective_opts.steps_mode == crate::options::StepsMode::Off {
                if let Some((result, required_conditions)) =
                    crate::rules::calculus::sqrt_polynomial_quotient_derivative_presentation_with_domain(
                        &mut self.simplifier.context,
                        call.target,
                        &call.var_name,
                    )
                {
                    return Ok((
                        crate::EvalResult::Expr(result),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        required_conditions,
                    ));
                }
                if let Some((result, required_conditions)) =
                    crate::rules::calculus::sqrt_of_polynomial_quotient_derivative_presentation_with_domain(
                        &mut self.simplifier.context,
                        call.target,
                        &call.var_name,
                    )
                {
                    return Ok((
                        crate::EvalResult::Expr(result),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        required_conditions,
                    ));
                }
                if let Some((result, required_conditions)) =
                    crate::rules::calculus::reciprocal_sqrt_polynomial_product_derivative_presentation_with_domain(
                        &mut self.simplifier.context,
                        call.target,
                        &call.var_name,
                    )
                {
                    return Ok((
                        crate::EvalResult::Expr(result),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        required_conditions,
                    ));
                }
                if let Some((result, required_conditions)) =
                    crate::rules::calculus::polynomial_over_sqrt_polynomial_derivative_presentation_with_domain(
                        &mut self.simplifier.context,
                        call.target,
                        &call.var_name,
                    )
                {
                    return Ok((
                        crate::EvalResult::Expr(result),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        required_conditions,
                    ));
                }
            }
            if let Some((result, required_conditions)) =
                crate::rules::calculus::sqrt_over_positive_shifted_sqrt_derivative_presentation_with_domain(
                    &mut self.simplifier.context,
                    call.target,
                    &call.var_name,
                )
            {
                let desc = crate::symbolic_calculus_call_support::render_diff_desc_with(
                    &call,
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &self.simplifier.context,
                                id
                            }
                        )
                    },
                );
                let mut steps = Vec::new();
                if effective_opts.steps_mode != crate::options::StepsMode::Off {
                    let mut step = crate::Step::new(
                        &desc,
                        "Symbolic Differentiation",
                        resolved,
                        result,
                        Vec::new(),
                        Some(&self.simplifier.context),
                    );
                    step.meta_mut()
                        .required_conditions
                        .extend(required_conditions.iter().cloned());
                    steps.push(step);
                }

                return Ok((
                    crate::EvalResult::Expr(result),
                    Vec::new(),
                    steps,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    required_conditions,
                ));
            }
            if let Some((result, required_conditions)) =
                crate::rules::calculus::ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
                    &mut self.simplifier.context,
                    call.target,
                    &call.var_name,
                )
            {
                let desc = crate::symbolic_calculus_call_support::render_diff_desc_with(
                    &call,
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &self.simplifier.context,
                                id
                            }
                        )
                    },
                );
                let mut steps = Vec::new();
                if effective_opts.steps_mode != crate::options::StepsMode::Off {
                    let mut step = crate::Step::new(
                        &desc,
                        "Symbolic Differentiation",
                        resolved,
                        result,
                        Vec::new(),
                        Some(&self.simplifier.context),
                    );
                    step.meta_mut()
                        .required_conditions
                        .extend(required_conditions.iter().cloned());
                    steps.push(step);
                }

                return Ok((
                    crate::EvalResult::Expr(result),
                    Vec::new(),
                    steps,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    required_conditions,
                ));
            }
        }

        if let Some((zero, required_conditions)) =
            try_diff_ln_sum_equal_derivative_roots_residual_zero_local(
                &mut self.simplifier.context,
                resolved,
            )
        {
            let mut steps = Vec::new();
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                let mut step = crate::Step::new(
                    "Resolve compact derivative residual",
                    "Post-calculus residual simplification",
                    resolved,
                    zero,
                    Vec::new(),
                    Some(&self.simplifier.context),
                );
                step.meta_mut()
                    .required_conditions
                    .extend(required_conditions.iter().cloned());
                steps.push(step);
            }

            return Ok((
                crate::EvalResult::Expr(zero),
                Vec::new(),
                steps,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                required_conditions,
            ));
        }

        if let Some((zero, required_conditions)) =
            try_diff_sqrt_over_positive_shifted_sqrt_residual_zero_local(
                &mut self.simplifier.context,
                resolved,
            )
        {
            let mut steps = Vec::new();
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                let mut step = crate::Step::new(
                    "Resolve compact derivative residual",
                    "Post-calculus residual simplification",
                    resolved,
                    zero,
                    Vec::new(),
                    Some(&self.simplifier.context),
                );
                step.meta_mut()
                    .required_conditions
                    .extend(required_conditions.iter().cloned());
                steps.push(step);
            }

            return Ok((
                crate::EvalResult::Expr(zero),
                Vec::new(),
                steps,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                required_conditions,
            ));
        }

        if let Some((zero, required_conditions)) =
            try_diff_sqrt_polynomial_quotient_residual_zero_local(
                &mut self.simplifier.context,
                resolved,
            )
        {
            let mut steps = Vec::new();
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                let mut step = crate::Step::new(
                    "Resolve compact derivative residual",
                    "Post-calculus residual simplification",
                    resolved,
                    zero,
                    Vec::new(),
                    Some(&self.simplifier.context),
                );
                step.meta_mut()
                    .required_conditions
                    .extend(required_conditions.iter().cloned());
                steps.push(step);
            }

            return Ok((
                crate::EvalResult::Expr(zero),
                Vec::new(),
                steps,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                required_conditions,
            ));
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_integral_reciprocal_trig_residual_root_zero(
                &mut self.simplifier.context,
                resolved,
            )
        {
            let mut steps = Vec::new();
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                let mut step = crate::Step::new(
                    "Post-calculus residual simplification",
                    "Verify supported antiderivative by differentiating it",
                    resolved,
                    zero,
                    Vec::new(),
                    Some(&self.simplifier.context),
                );
                step.meta_mut()
                    .required_conditions
                    .extend(required_conditions.iter().cloned());
                steps.push(step);
            }

            return Ok((
                crate::EvalResult::Expr(zero),
                Vec::new(),
                steps,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                required_conditions,
            ));
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_inverse_reciprocal_trig_residual_root_zero(
                &mut self.simplifier.context,
                resolved,
            )
        {
            let mut steps = Vec::new();
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                let mut step = crate::Step::new(
                    "Post-calculus residual simplification",
                    "Resolve inverse-trig derivative residual",
                    resolved,
                    zero,
                    Vec::new(),
                    Some(&self.simplifier.context),
                );
                step.meta_mut()
                    .required_conditions
                    .extend(required_conditions.iter().cloned());
                steps.push(step);
            }

            return Ok((
                crate::EvalResult::Expr(zero),
                Vec::new(),
                steps,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                required_conditions,
            ));
        }

        let profile = self.profile_cache.get_or_build(&effective_opts);
        let inherited_allow_numerical_verification = self.simplifier.allow_numerical_verification;
        let inherited_debug_mode = self.simplifier.debug_mode;
        let inherited_step_listener = self.simplifier.replace_step_listener(None);
        let mut ctx_simplifier = Simplifier::from_profile_with_context(
            profile,
            std::mem::take(&mut self.simplifier.context),
        );
        let preserve_hidden_solve_fast_paths = effective_opts.shared.context_mode
            == crate::options::ContextMode::Solve
            && effective_opts.shared.semantics.domain_mode == crate::DomainMode::Strict;
        let runtime_steps_mode = match effective_opts.steps_mode {
            crate::options::StepsMode::Off if preserve_hidden_solve_fast_paths => {
                crate::options::StepsMode::Off
            }
            crate::options::StepsMode::Off
                if !expr_contains_hyperbolic_builtin_local(&ctx_simplifier.context, resolved) =>
            {
                crate::options::StepsMode::Compact
            }
            mode => mode,
        };
        ctx_simplifier.set_steps_mode(runtime_steps_mode);
        ctx_simplifier.allow_numerical_verification = inherited_allow_numerical_verification;
        ctx_simplifier.debug_mode = inherited_debug_mode;
        ctx_simplifier.set_step_listener(inherited_step_listener);

        let mut simplify_opts = effective_opts.to_simplify_options();
        let (mut expr_to_simplify, expand_log_events) =
            if let Expr::Function(fn_id, args) = ctx_simplifier.context.get(resolved).clone() {
                match ctx_simplifier.context.sym_name(fn_id) {
                    "collect" => {
                        simplify_opts.goal = crate::semantics::NormalFormGoal::Collected;
                        (resolved, Vec::new())
                    }
                    "expand_log" if args.len() == 1 => {
                        simplify_opts.goal = crate::semantics::NormalFormGoal::ExpandedLog;
                        crate::rules::logarithms::expand_logs_with_assumptions(
                            &mut ctx_simplifier.context,
                            args[0],
                            effective_opts.shared.semantics.domain_mode,
                            effective_opts.shared.semantics.value_domain,
                        )
                        .unwrap_or((args[0], Vec::new()))
                    }
                    _ => (resolved, Vec::new()),
                }
            } else {
                (resolved, Vec::new())
            };

        let input_embedded_calculus_residual = expr_contains_named_function_local(
            &ctx_simplifier.context,
            expr_to_simplify,
            &["diff", "integrate", "int"],
        ) || expr_contains_symbolic_calculus_call_local(
            &ctx_simplifier.context,
            expr_to_simplify,
        );
        let input_embedded_calculus_residual = input_embedded_calculus_residual
            && !expr_is_named_function_call_local(
                &ctx_simplifier.context,
                expr_to_simplify,
                &["diff", "integrate", "int"],
            )
            && !expr_is_symbolic_calculus_call_local(&ctx_simplifier.context, expr_to_simplify)
            && expr_is_post_calculus_residual_candidate_local(
                &ctx_simplifier.context,
                expr_to_simplify,
            );
        if input_embedded_calculus_residual {
            simplify_opts.suppress_depth_overflow_warnings = true;
        }

        let pre_resolved_calculus_context = if input_embedded_calculus_residual {
            let original_expr = expr_to_simplify;
            try_rewrite_post_calculus_residual_child_context_before_general_simplify_local(
                &mut ctx_simplifier.context,
                expr_to_simplify,
                0,
            )
            .filter(|(rewritten, _)| *rewritten != original_expr)
            .map(|(rewritten, required_conditions)| {
                expr_to_simplify = rewritten;
                (original_expr, rewritten, required_conditions)
            })
        } else {
            None
        };
        let pre_simplify_calculus = if input_embedded_calculus_residual {
            try_resolve_post_calculus_residual_before_general_simplify_local(
                &mut ctx_simplifier.context,
                expr_to_simplify,
            )
            .map(|(result, required_conditions)| {
                (
                    result,
                    required_conditions,
                    "Post-calculus residual simplification",
                    "Resolve calculus calls and simplify matching residual before general simplification",
                )
            })
        } else if expr_is_symbolic_calculus_call_local(&ctx_simplifier.context, expr_to_simplify) {
            try_resolve_direct_post_calculus_before_general_simplify_local(
                &mut ctx_simplifier.context,
                expr_to_simplify,
            )
            .map(|(result, required_conditions)| {
                (
                    result,
                    required_conditions,
                    "Calcular la derivada",
                    "Calcular la derivada",
                )
            })
        } else {
            None
        };

        let (mut res, mut steps, stats) =
            if let Some((result, required_conditions, step_title, step_explanation)) =
                pre_simplify_calculus
            {
                ctx_simplifier.extend_required_conditions(required_conditions.clone());
                let steps = if effective_opts.steps_mode != crate::options::StepsMode::Off {
                    let mut step = crate::Step::new(
                        step_title,
                        step_explanation,
                        expr_to_simplify,
                        result,
                        Vec::new(),
                        Some(&ctx_simplifier.context),
                    );
                    step.importance = crate::ImportanceLevel::Medium;
                    step.meta_mut().required_conditions = required_conditions;
                    vec![step]
                } else {
                    Vec::new()
                };
                (result, steps, crate::phase::PipelineStats::default())
            } else {
                ctx_simplifier.simplify_with_stats(expr_to_simplify, simplify_opts.clone())
            };

        if let Some((original_expr, rewritten, required_conditions)) = pre_resolved_calculus_context
        {
            ctx_simplifier.extend_required_conditions(required_conditions.clone());
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                let mut step = crate::Step::new(
                    "Post-calculus residual simplification",
                    "Resolve a matching calculus residual inside its wrapper before general simplification",
                    original_expr,
                    rewritten,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                step.importance = crate::ImportanceLevel::Medium;
                step.meta_mut().required_conditions = required_conditions;
                steps.insert(0, step);
            }
        }

        let hyperbolic_zero_rewrite =
            crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(
                &mut ctx_simplifier.context,
                expr_to_simplify,
            )
            .map(|rewrite| (expr_to_simplify, rewrite))
            .or_else(|| {
                crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(
                    &mut ctx_simplifier.context,
                    res,
                )
                .map(|rewrite| (res, rewrite))
            });
        if let Some((rewrite_source, rewrite)) = hyperbolic_zero_rewrite {
            let final_expr = rewrite.final_expr();
            res = final_expr;
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                steps.clear();
                let mut step = crate::Step::new(
                    &rewrite.description,
                    &rewrite.description,
                    rewrite_source,
                    final_expr,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                step.meta_mut().required_conditions = rewrite.required_conditions.clone();
                steps.push(step);
            }
        }

        if !expand_log_events.is_empty() {
            if steps.is_empty() {
                let mut step = crate::Step::new(
                    "Log expansion",
                    "expand_log",
                    resolved,
                    res,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                step.meta_mut().assumption_events.extend(expand_log_events);
                steps.push(step);
            } else {
                steps[0]
                    .meta_mut()
                    .assumption_events
                    .extend(expand_log_events);
            }
        }

        let embedded_calculus_residual = expr_contains_named_function_local(
            &ctx_simplifier.context,
            expr_to_simplify,
            &["diff", "integrate", "int"],
        ) || expr_contains_symbolic_calculus_call_local(
            &ctx_simplifier.context,
            expr_to_simplify,
        );
        let embedded_calculus_residual = embedded_calculus_residual
            && !expr_is_named_function_call_local(
                &ctx_simplifier.context,
                expr_to_simplify,
                &["diff", "integrate", "int"],
            )
            && !expr_is_symbolic_calculus_call_local(&ctx_simplifier.context, expr_to_simplify)
            && !expr_contains_named_function_local(
                &ctx_simplifier.context,
                res,
                &["diff", "integrate", "int"],
            )
            && !expr_contains_symbolic_calculus_call_local(&ctx_simplifier.context, res)
            && expr_is_post_calculus_residual_candidate_local(&ctx_simplifier.context, res);

        if embedded_calculus_residual {
            let post_residual_res = if let Some(rewrite) =
                crate::rules::arithmetic::try_build_exact_zero_identity_rewrite(
                    &mut ctx_simplifier.context,
                    res,
                ) {
                let final_expr = rewrite.final_expr();
                ctx_simplifier.extend_required_conditions(rewrite.required_conditions);
                final_expr
            } else {
                let first_pass_required = ctx_simplifier.take_required_conditions();
                let (post_residual_res, _post_residual_steps, _post_residual_stats) =
                    ctx_simplifier.simplify_with_stats(res, simplify_opts);
                let second_pass_required = ctx_simplifier.take_required_conditions();
                ctx_simplifier.extend_required_conditions(
                    first_pass_required.into_iter().chain(second_pass_required),
                );
                let post_residual_res =
                        crate::fraction_residual_support::try_polynomial_denominator_fraction_residual_zero(
                            &mut ctx_simplifier.context,
                            post_residual_res,
                        )
                        .unwrap_or(post_residual_res);
                let post_residual_res =
                    crate::calculus_residual_support::try_reciprocal_half_power_shared_denominator_residual_root_zero(
                        &mut ctx_simplifier.context,
                        post_residual_res,
                    )
                    .map(|(result, required_conditions)| {
                        ctx_simplifier.extend_required_conditions(required_conditions);
                        result
                    })
                    .unwrap_or(post_residual_res);
                crate::rules::arithmetic::try_build_exact_zero_identity_rewrite(
                    &mut ctx_simplifier.context,
                    post_residual_res,
                )
                .map(|rewrite| {
                    let final_expr = rewrite.final_expr();
                    ctx_simplifier.extend_required_conditions(rewrite.required_conditions);
                    final_expr
                })
                .unwrap_or(post_residual_res)
            };

            if post_residual_res != res {
                if effective_opts.steps_mode != crate::options::StepsMode::Off {
                    let mut post_residual_step = crate::Step::new(
                        "Post-calculus residual simplification",
                        "Re-simplify residual after resolving calculus calls",
                        res,
                        post_residual_res,
                        Vec::new(),
                        Some(&ctx_simplifier.context),
                    );
                    post_residual_step.importance = crate::ImportanceLevel::Medium;
                    steps.push(post_residual_step);
                }
                res = post_residual_res;
            }
        }

        let mut final_presented = crate::rules::calculus::try_post_calculus_presentation(
            &mut ctx_simplifier.context,
            resolved,
            res,
        )
        .filter(|presented| *presented != res);
        if final_presented.is_none() {
            final_presented = crate::rules::calculus::try_calculus_result_presentation(
                &mut ctx_simplifier.context,
                res,
            )
            .filter(|presented| *presented != res);
        }

        if let Some(presented) = final_presented {
            if effective_opts.steps_mode != crate::options::StepsMode::Off
                && !collapse_redundant_post_calculus_trace_if_direct_step_is_compact(
                    &mut ctx_simplifier.context,
                    resolved,
                    presented,
                    &mut steps,
                )
            {
                let mut presentation_step = crate::Step::new(
                    "Post-calculus presentation",
                    "Present calculus result in compact form",
                    res,
                    presented,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                presentation_step.importance = crate::ImportanceLevel::Medium;
                steps.push(presentation_step);
            }
            res = presented;
        }

        if effective_opts.const_fold == crate::const_fold::ConstFoldMode::Safe {
            let mut budget = crate::budget::Budget::preset_cli();
            let cfg = crate::semantics::EvalConfig {
                value_domain: effective_opts.shared.semantics.value_domain,
                branch: effective_opts.shared.semantics.branch,
                ..Default::default()
            };
            if let Ok(fold_result) = crate::const_fold::fold_constants(
                &mut ctx_simplifier.context,
                res,
                &cfg,
                effective_opts.const_fold,
                &mut budget,
            ) {
                res = fold_result.expr;
            }
        }

        if let Some(presented) = crate::rules::calculus::try_post_calculus_presentation(
            &mut ctx_simplifier.context,
            resolved,
            res,
        ) {
            if effective_opts.steps_mode != crate::options::StepsMode::Off
                && presented != res
                && !collapse_redundant_post_calculus_trace_if_direct_step_is_compact(
                    &mut ctx_simplifier.context,
                    resolved,
                    presented,
                    &mut steps,
                )
            {
                let mut presentation_step = crate::Step::new(
                    "Post-calculus presentation",
                    "Present calculus result in compact form",
                    res,
                    presented,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                presentation_step.importance = crate::ImportanceLevel::Medium;
                steps.push(presentation_step);
            }
            res = presented;
        }
        self.simplifier
            .extend_blocked_hints(ctx_simplifier.take_blocked_hints());
        let rewrite_required = ctx_simplifier.take_required_conditions();
        let restored_step_listener = ctx_simplifier.replace_step_listener(None);
        self.simplifier.context = ctx_simplifier.context;
        self.simplifier.set_step_listener(restored_step_listener);

        {
            use crate::{classify_assumptions_in_place, infer_implicit_domain, DomainContext};

            let input_domain = infer_implicit_domain(
                &self.simplifier.context,
                resolved,
                effective_opts.shared.semantics.value_domain,
            );
            let global_requires: Vec<_> = input_domain.conditions().iter().cloned().collect();
            let mut dc = DomainContext::new(global_requires);

            for step in &mut steps {
                classify_assumptions_in_place(
                    &self.simplifier.context,
                    &mut dc,
                    &mut step.meta_mut().assumption_events,
                );
            }

            if effective_opts.shared.semantics.domain_mode == crate::DomainMode::Assume {
                for step in &mut steps {
                    if step.rule_name == "Log-Exp Inverse" {
                        for event in &mut step.meta_mut().assumption_events {
                            if matches!(event.kind, crate::AssumptionKind::DerivedFromRequires)
                                && matches!(event.key, crate::AssumptionKey::Positive { .. })
                            {
                                // `log(b, b^x) -> x` in Assume mode should still surface the
                                // user-visible positivity assumption on the symbolic base, even
                                // when that positivity is already implicit in the source log.
                                event.kind = crate::AssumptionKind::HeuristicAssumption;
                            }
                        }
                    }
                }
            }
        }

        let mut warnings = collect_domain_warnings(
            &self.simplifier.context,
            effective_opts.shared.semantics.value_domain,
            res,
            &steps,
        );

        if stats.timed_out {
            let message = match effective_opts.time_budget_ms {
                Some(ms) => format!(
                    "Partial result: simplification stopped after reaching the {} ms time budget.",
                    ms
                ),
                None => "Partial result: simplification stopped after reaching the time budget."
                    .to_string(),
            };
            let timeout_warning = DomainWarning {
                message,
                rule_name: "Simplification Time Budget".to_string(),
            };
            if !warnings.iter().any(|warning| warning == &timeout_warning) {
                warnings.push(timeout_warning);
            }
        }

        if effective_opts.shared.semantics.value_domain == crate::semantics::ValueDomain::RealOnly
            && cas_math::numeric_eval::contains_i(&self.simplifier.context, resolved)
        {
            let i_warning = DomainWarning {
                message: "To use complex arithmetic (i² = -1), run: semantics set value complex"
                    .to_string(),
                rule_name: "Imaginary Usage Warning".to_string(),
            };
            if !warnings.iter().any(|w| w.message == i_warning.message) {
                warnings.push(i_warning);
            }
        }

        Ok((
            EvalResult::Expr(res),
            warnings,
            steps,
            vec![],
            vec![],
            vec![],
            vec![],
            rewrite_required,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn post_calculus_residual_match_accepts_denominator_numeric_rescale() {
        let mut ctx = cas_ast::Context::new();
        let result = parse("(cos(2*x)-1/2*sin(x))/sqrt(sin(2*x)+cos(x)+4)", &mut ctx).unwrap();
        let target = parse("(2*cos(2*x)-sin(x))/(2*sqrt(sin(2*x)+cos(x)+4))", &mut ctx).unwrap();

        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx, result, target
        ));

        let result_with_sqrt_term = parse(
            "(cos(2*x)-1/2*sin(x)+1/(4*sqrt(x)))/sqrt(sin(2*x)+cos(x)+sqrt(x))",
            &mut ctx,
        )
        .unwrap();
        let target = parse(
            "(4*sqrt(x)*cos(2*x)+1-2*sqrt(x)*sin(x))/(4*sqrt(x)*sqrt(sin(2*x)+cos(x)+sqrt(x)))",
            &mut ctx,
        )
        .unwrap();
        assert!(sqrt_denominator_common_factor_target_variant_exact_local(
            &mut ctx,
            target,
            result_with_sqrt_term
        ));
    }

    #[test]
    fn post_calculus_residual_match_accepts_product_terms_in_scaled_additive_numerator() {
        let mut ctx = cas_ast::Context::new();
        let result = parse(
            "(cos(x)^2+e^x*cos(x)^2+1)/(2*cos(x)^2*sqrt(tan(x)+e^x+x))",
            &mut ctx,
        )
        .unwrap();
        let target = parse(
            "(2*cos(x)^2+2*e^x*cos(x)^2+2)/(4*cos(x)^2*sqrt(tan(x)+e^x+x))",
            &mut ctx,
        )
        .unwrap();

        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx, result, target
        ));
    }

    #[test]
    fn post_calculus_residual_match_extracts_scale_from_reciprocal_sqrt_terms() {
        let mut ctx = cas_ast::Context::new();
        let result = parse(
            "(cos(2*x)+1/(4*sqrt(x))-1/2*sin(x))/sqrt(sin(2*x)+cos(x)+sqrt(x))",
            &mut ctx,
        )
        .unwrap();
        let target = parse(
            "(4*cos(2*x)+x^(-1/2)-2*sin(x))/(4*sqrt(sin(2*x)+cos(x)+sqrt(x)))",
            &mut ctx,
        )
        .unwrap();

        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx, result, target
        ));
    }

    #[test]
    fn post_calculus_residual_route_accepts_reciprocal_sqrt_scaled_public_input() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x) - (4*cos(2*x)+x^(-1/2)-2*sin(x))/(4*sqrt(sin(2*x)+cos(x)+sqrt(x)))",
            &mut ctx,
        )
        .unwrap();

        let (left, right) = match residual_difference_terms_local(&ctx, expr) {
            Some(terms) => terms,
            None => panic!("expected residual difference terms"),
        };
        let call = crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, left)
            .or_else(|| crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, right));
        let call = match call {
            Some(call) => call,
            None => panic!("expected diff call"),
        };
        let (presentation, _, _) =
            crate::rules::calculus::sqrt_additive_trig_polynomial_derivative_presentation(
                &mut ctx,
                call.target,
                &call.var_name,
            )
            .unwrap_or_else(|| panic!("expected presentation"));
        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx,
            presentation,
            right
        ));
        let (result, required_conditions) =
            match try_diff_sqrt_additive_trig_polynomial_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected direct trig-root residual route"),
            };

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 2);

        let expr = parse(
            "diff(sqrt(sin(2*x)+cos(x)+sqrt(x)), x) - (4*sqrt(x)*cos(2*x)+1-2*sqrt(x)*sin(x))/(4*sqrt(x)*sqrt(sin(2*x)+cos(x)+sqrt(x)))",
            &mut ctx,
        )
        .unwrap();
        let (left, right) = match residual_difference_terms_local(&ctx, expr) {
            Some(terms) => terms,
            None => panic!("expected common-denominator residual difference terms"),
        };
        let call = crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, left)
            .or_else(|| crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, right));
        let call = match call {
            Some(call) => call,
            None => panic!("expected common-denominator diff call"),
        };
        let (presentation, _, _) =
            crate::rules::calculus::sqrt_additive_trig_polynomial_derivative_presentation(
                &mut ctx,
                call.target,
                &call.var_name,
            )
            .unwrap_or_else(|| panic!("expected common-denominator presentation"));
        assert!(
            sqrt_denominator_common_factor_target_variant_exact_local(
                &mut ctx,
                right,
                presentation
            ),
            "presentation={} target={}",
            DisplayExpr {
                context: &ctx,
                id: presentation
            },
            DisplayExpr {
                context: &ctx,
                id: right
            }
        );
        let (result, required_conditions) =
            match try_diff_sqrt_additive_trig_polynomial_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected common-denominator trig-root residual route"),
            };
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 2);

        let expr = parse(
            "diff(sqrt(sin(2*x)+cos(x)-sqrt(x)), x) - (4*sqrt(x)*cos(2*x)-1-2*sqrt(x)*sin(x))/(4*sqrt(x)*sqrt(sin(2*x)+cos(x)-sqrt(x)))",
            &mut ctx,
        )
        .unwrap();
        let (left, right) = match residual_difference_terms_local(&ctx, expr) {
            Some(terms) => terms,
            None => panic!("expected negative common-denominator residual difference terms"),
        };
        let call = crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, left)
            .or_else(|| crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, right));
        let call = match call {
            Some(call) => call,
            None => panic!("expected negative common-denominator diff call"),
        };
        let (presentation, _, _) =
            crate::rules::calculus::sqrt_additive_trig_polynomial_derivative_presentation(
                &mut ctx,
                call.target,
                &call.var_name,
            )
            .unwrap_or_else(|| panic!("expected negative common-denominator presentation"));
        assert!(
            sqrt_denominator_common_factor_target_variant_exact_local(
                &mut ctx,
                right,
                presentation
            ),
            "presentation={} target={}",
            DisplayExpr {
                context: &ctx,
                id: presentation
            },
            DisplayExpr {
                context: &ctx,
                id: right
            }
        );
        let (result, required_conditions) =
            match try_diff_sqrt_additive_trig_polynomial_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected negative common-denominator trig-root residual route"),
            };
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn post_calculus_residual_route_accepts_tan_sqrt_sec_target_variant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "diff(sqrt(tan(x)+sqrt(x)+x), x) - (2*x^(1/2)*sec(x)^2+1+2*x^(1/2))/(4*x^(1/2)*sqrt(tan(x)+sqrt(x)+x))",
            &mut ctx,
        )
        .unwrap();

        let (result, required_conditions) =
            match try_diff_sqrt_additive_tan_polynomial_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected direct tan-root residual route"),
            };

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn post_calculus_residual_route_accepts_tan_sqrt_affine_sec_target_variant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "diff(sqrt(tan(x)+sqrt(x)+2*x+1), x) - (2*x^(1/2)*sec(x)^2+1+4*x^(1/2))/(4*x^(1/2)*sqrt(tan(x)+sqrt(x)+2*x+1))",
            &mut ctx,
        )
        .unwrap();

        let (result, required_conditions) =
            match try_diff_sqrt_additive_tan_polynomial_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected direct tan-root affine residual route"),
            };

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn post_calculus_residual_route_accepts_tan_exp_sqrt_inline_target_variant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "diff(sqrt(tan(x)+exp(x)+sqrt(x)+x), x) - (sec(x)^2+e^x+1+1/(2*sqrt(x)))/(2*sqrt(tan(x)+exp(x)+sqrt(x)+x))",
            &mut ctx,
        )
        .unwrap();

        let (left, right) = match residual_difference_terms_local(&ctx, expr) {
            Some(terms) => terms,
            None => panic!("expected residual difference terms"),
        };
        let call = crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, left)
            .or_else(|| crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, right));
        let call = match call {
            Some(call) => call,
            None => panic!("expected diff call"),
        };
        let (presentation, _, _) =
            crate::rules::calculus::sqrt_additive_tan_polynomial_derivative_presentation(
                &mut ctx,
                call.target,
                &call.var_name,
            )
            .unwrap_or_else(|| panic!("expected tan/exp/sqrt presentation"));
        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx,
            presentation,
            right
        ));

        let (result, required_conditions) =
            match try_diff_sqrt_additive_tan_polynomial_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected direct tan/exp/sqrt inline residual route"),
            };

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn post_calculus_residual_route_accepts_exp_trig_log_sqrt_common_denominator_variant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)+x)/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(x)))",
            &mut ctx,
        )
        .unwrap();

        let (left, right) = match residual_difference_terms_local(&ctx, expr) {
            Some(terms) => terms,
            None => panic!("expected residual difference terms"),
        };
        let call = crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, left)
            .or_else(|| crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, right));
        let call = match call {
            Some(call) => call,
            None => panic!("expected diff call"),
        };
        let (presentation, _) =
            crate::rules::calculus::sqrt_small_additive_elementary_derivative_presentation_with_domain(
                &mut ctx,
                call.target,
                &call.var_name,
            )
            .unwrap_or_else(|| panic!("expected exp/trig/log/sqrt presentation"));
        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx,
            presentation,
            right
        ));

        let (result, required_conditions) =
            match try_diff_sqrt_small_additive_elementary_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected direct exp/trig/log/sqrt residual route"),
            };

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn post_calculus_residual_route_accepts_polynomial_power_term_variant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "diff(sqrt(exp(sin(x))+ln(x)+sqrt(x)+x^2), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)+x+4*x^2*sqrt(x))/(4*x*sqrt(x)*sqrt(ln(x)+sqrt(x)+e^sin(x)+x^2))",
            &mut ctx,
        )
        .unwrap();

        let (left, right) = match residual_difference_terms_local(&ctx, expr) {
            Some(terms) => terms,
            None => panic!("expected residual difference terms"),
        };
        let call = crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, left)
            .or_else(|| crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, right));
        let call = match call {
            Some(call) => call,
            None => panic!("expected diff call"),
        };
        let (presentation, _) =
            crate::rules::calculus::sqrt_small_additive_elementary_derivative_presentation_with_domain(
                &mut ctx,
                call.target,
                &call.var_name,
            )
            .unwrap_or_else(|| panic!("expected exp/trig/log/sqrt polynomial presentation"));
        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx,
            presentation,
            right
        ));

        let (result, required_conditions) =
            match try_diff_sqrt_small_additive_elementary_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected direct polynomial-term residual route"),
            };

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn post_calculus_residual_route_accepts_signed_sqrt_term_radicand_orientation() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "diff(sqrt(exp(sin(x))+ln(x)-sqrt(x)), x) - (2*sqrt(x)+2*x*sqrt(x)*cos(x)*e^sin(x)-x)/(4*x*sqrt(x)*sqrt(ln(x)-sqrt(x)+e^sin(x)))",
            &mut ctx,
        )
        .unwrap();

        let (left, right) = match residual_difference_terms_local(&ctx, expr) {
            Some(terms) => terms,
            None => panic!("expected residual difference terms"),
        };
        let call = crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, left)
            .or_else(|| crate::symbolic_calculus_call_support::try_extract_diff_call(&ctx, right));
        let call = match call {
            Some(call) => call,
            None => panic!("expected diff call"),
        };
        let (presentation, _) =
            crate::rules::calculus::sqrt_small_additive_elementary_derivative_presentation_with_domain(
                &mut ctx,
                call.target,
                &call.var_name,
            )
            .unwrap_or_else(|| panic!("expected signed sqrt-term presentation"));
        assert!(sqrt_shifted_diff_result_matches_target_local(
            &mut ctx,
            presentation,
            right
        ));

        let (result, required_conditions) =
            match try_diff_sqrt_small_additive_elementary_residual_zero_local(&mut ctx, expr) {
                Some(result) => result,
                None => panic!("expected direct signed sqrt-term residual route"),
            };

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            "0"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn post_calculus_residual_child_rewrite_preserves_wrapper_denominator_condition() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse(
            "((diff(arctan(sqrt(sec(x)+ln(x)+1/sqrt(x)+x)),x)-(2*sqrt(x)+2*x*sqrt(x)+2*x*tan(x)*sec(x)*sqrt(x)-1)/(4*x*sqrt(x)*sqrt(sec(x)+ln(x)+1/sqrt(x)+x)*(sec(x)+ln(x)+1/sqrt(x)+x+1)))+1)/(x+2)",
            &mut ctx,
        )
        .unwrap();

        let (rewritten, required_conditions) =
            try_rewrite_post_calculus_residual_child_context_before_general_simplify_local(
                &mut ctx, expr, 0,
            )
            .unwrap_or_else(|| panic!("expected wrapped residual rewrite"));
        let required_display: Vec<_> = required_conditions
            .iter()
            .map(|condition| match condition {
                crate::ImplicitCondition::NonZero(id)
                | crate::ImplicitCondition::Positive(id)
                | crate::ImplicitCondition::NonNegative(id) => format!(
                    "{}",
                    DisplayExpr {
                        context: &ctx,
                        id: *id
                    }
                ),
                crate::ImplicitCondition::LowerBound(id, bound) => format!(
                    "{} >= {}",
                    DisplayExpr {
                        context: &ctx,
                        id: *id
                    },
                    bound
                ),
            })
            .collect();

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewritten
                }
            ),
            "1 / (x + 2)"
        );
        assert!(
            required_display
                .iter()
                .any(|condition| condition == "x + 2"),
            "missing wrapper denominator condition: {required_display:?}"
        );
    }

    #[test]
    fn eval_stateless_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let mut engine = Engine::new();
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Auto;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_stateless_preserves_reciprocal_trig_intrinsic_domain_requires() {
        for (expr_text, expected_result, expected_requires) in [
            (
                "tan(x)+cot(x)-sec(x)*csc(x)",
                "0",
                vec!["cos(x) ≠ 0", "sin(x) ≠ 0"],
            ),
            ("tan(x)", "tan(x)", vec!["cos(x) ≠ 0"]),
            ("sec(x)", "sec(x)", vec!["cos(x) ≠ 0"]),
            ("cot(x)", "cot(x)", vec!["sin(x) ≠ 0"]),
            ("csc(x)", "csc(x)", vec!["sin(x) ≠ 0"]),
        ] {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{expr_text}: {e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{expr_text}: {e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("{expr_text}: expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected_result,
                "{expr_text}"
            );

            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            for expected in expected_requires {
                assert!(
                    required_display.iter().any(|actual| actual == expected),
                    "{expr_text}: expected {expected}, got {required_display:?}"
                );
            }
        }
    }

    #[test]
    fn eval_stateless_preserves_inverse_trig_bounded_domain_requires() {
        for (expr_text, expected_result, expected_requires) in [
            ("arcsin(x)", "arcsin(x)", vec!["-1 ≤ x ≤ 1"]),
            ("arccos(x)", "arccos(x)", vec!["-1 ≤ x ≤ 1"]),
            (
                "ln(arcsin(x)^2)",
                "2 * ln(|arcsin(x)|)",
                vec!["-1 ≤ x ≤ 1", "arcsin(x) ≠ 0"],
            ),
            (
                "arcsec(x)",
                "arccos(1 / x)",
                vec!["x ≤ -1 or x ≥ 1", "x ≠ 0"],
            ),
            (
                "arccsc(x)",
                "arcsin(1 / x)",
                vec!["x ≤ -1 or x ≥ 1", "x ≠ 0"],
            ),
        ] {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{expr_text}: {e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{expr_text}: {e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("{expr_text}: expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected_result,
                "{expr_text}"
            );

            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            for expected in expected_requires {
                assert!(
                    required_display.iter().any(|actual| actual == expected),
                    "{expr_text}: expected {expected}, got {required_display:?}"
                );
            }
        }
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let mut engine = Engine::new();
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_preserves_post_calculus_sqrt_tan_presentation() {
        let mut engine = Engine::new();
        let expr_text = "3*tan(sqrt(3*x+1))/(2*sqrt(3*x+1))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let options = crate::options::EvalOptions::default();

        let (result, ..) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };

        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_tan_affine_avoids_pre_diff_trig_expansion_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(tan(2*x+1), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "2 / cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_scaled_atanh_surd_polynomial_uses_direct_route() {
        let cases = [
            "diff(1/3 * atanh(1/3 * sqrt(3) * (x^2 + 2*x + 1)) * sqrt(3), x)",
            "diff(atanh((x^2 + 2*x + 1)/sqrt(3))/sqrt(3), x)",
        ];

        for expr_text in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(200);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            let displayed = DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string();
            assert!(
                displayed.ends_with("/ (3 - (x + 1)^4)"),
                "input {expr_text}: unexpected derivative presentation: {displayed}"
            );
            assert!(
                !displayed.contains("(x^2 + 2 * x + 1)^2"),
                "input {expr_text}: denominator should stay compact, got: {displayed}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "unexpected timeout warning for {expr_text}: {:?}",
                output.domain_warnings
            );
        }
    }

    #[test]
    fn eval_simplify_integrate_scaled_denominator_square_preserves_required_domain() {
        let mut engine = Engine::new();
        let expr_text = "integrate((2*x+1)/(3*(x^2+x-1)^2), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let required_display = crate::render_conditions_normalized(
            &mut engine.simplifier.context,
            &output.required_conditions,
        );
        assert_eq!(
            required_display,
            vec!["x^2 + x - 1 ≠ 0".to_string()],
            "unexpected required_conditions: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_tan_square_avoids_pre_diff_trig_expansion_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(tan(2*x+1)^2, x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "(tan(2 * x + 1) * 4)/cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_variable_times_tan_avoids_pre_diff_trig_expansion_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(x*tan(2*x+1), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "tan(2 * x + 1) + (x * 2)/cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_scaled_variable_times_tan_avoids_post_diff_trig_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(2*x*tan(2*x+1), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "2 * tan(2 * x + 1) + (x * 4)/cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_tan_keeps_product_rule_shape() {
        let cases = [
            (
                "diff((x+1)*tan(2*x+1), x)",
                "tan(2 * x + 1) + (2 * x + 2) / cos(2 * x + 1)^2",
            ),
            (
                "diff((3*x+2)*tan(2*x+1), x)",
                "3 * tan(2 * x + 1) + (6 * x + 4) / cos(2 * x + 1)^2",
            ),
        ];

        for (expr_text, expected) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(50);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            assert_eq!(
                required_display
                    .iter()
                    .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                    .count(),
                1,
                "input {expr_text}: expected exactly one displayed cos-domain condition, got: {required_display:?}"
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_tanh_avoids_timeout() {
        let cases = [
            (
                "diff((x+1)*tanh(2*x+1), x)",
                "tanh(2 * x + 1) + (2 * x + 2) / cosh(2 * x + 1)^2",
            ),
            (
                "diff((3*x+2)*tanh(2*x+1), x)",
                "3 * tanh(2 * x + 1) + (6 * x + 4) / cosh(2 * x + 1)^2",
            ),
        ];

        for (expr_text, expected) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(50);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            assert!(
                output.required_conditions.is_empty(),
                "input {expr_text}: cosh is nonzero on the real domain, got redundant conditions: {:?}",
                output.required_conditions
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_cot_keeps_product_rule_shape() {
        let cases = [
            (
                "diff((x+1)*cot(2*x+1), x)",
                "cot(2 * x + 1) - (2 * x + 2) / sin(2 * x + 1)^2",
            ),
            (
                "diff((3*x+2)*cot(2*x+1), x)",
                "3 * cot(2 * x + 1) - (6 * x + 4) / sin(2 * x + 1)^2",
            ),
        ];

        for (expr_text, expected) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(50);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            assert_eq!(
                required_display
                    .iter()
                    .filter(|condition| condition.as_str() == "sin(2 * x + 1) ≠ 0")
                    .count(),
                1,
                "input {expr_text}: expected exactly one displayed sin-domain condition, got: {required_display:?}"
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_sec_csc_avoids_timeout() {
        let cases = [
            (
                "diff((x+1)*sec(2*x+1), x)",
                "(cos(2 * x + 1) + 2 * sin(2 * x + 1) + 2 * x * sin(2 * x + 1)) / cos(2 * x + 1)^2",
                "cos(2 * x + 1) ≠ 0",
            ),
            (
                "diff((3*x+2)*sec(2*x+1), x)",
                "(3 * cos(2 * x + 1) + 4 * sin(2 * x + 1) + 6 * x * sin(2 * x + 1)) / cos(2 * x + 1)^2",
                "cos(2 * x + 1) ≠ 0",
            ),
            (
                "diff((x+1)*csc(2*x+1), x)",
                "csc(2 * x + 1) - cos(2 * x + 1) * (2 * x + 2) / sin(2 * x + 1)^2",
                "sin(2 * x + 1) ≠ 0",
            ),
            (
                "diff((3*x+2)*csc(2*x+1), x)",
                "3 * csc(2 * x + 1) - cos(2 * x + 1) * (6 * x + 4) / sin(2 * x + 1)^2",
                "sin(2 * x + 1) ≠ 0",
            ),
        ];

        for (expr_text, expected, expected_condition) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(500);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            assert_eq!(
                required_display
                    .iter()
                    .filter(|condition| condition.as_str() == expected_condition)
                    .count(),
                1,
                "input {expr_text}: expected exactly one displayed domain condition, got: {required_display:?}"
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_against_polynomial_partner_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_against_polynomial_partner_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_against_polynomial_plus_rational_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_against_polynomial_plus_rational_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_against_polynomial_plus_hyperbolic_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_against_polynomial_plus_hyperbolic_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_hyperbolic_sum_against_telescoping_sum_regression() {
        let expr_text =
            "(sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + (1/(u*(u+1)) - 1/u + 1/(u+1))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_hyperbolic_sum_against_telescoping_sum_regression() {
        let expr_text =
            "(sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + (1/(u*(u+1)) - 1/u + 1/(u+1))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_trig_square_cube_substitution_pair_regression() {
        let expr_text = "(((sin(u)^2)^3 - 1) / ((sin(u)^2) - 1)) - ((sin(u)^4) + (sin(u)^2) + 1)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_negative_double_cos_square_diff_passthrough_forward_regression(
    ) {
        let expr_text = "((sin(x)^2 - cos(x)^2) + m) - ((-cos(2*x)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_negative_double_cos_square_diff_passthrough_forward_regression(
    ) {
        let expr_text = "((sin(x)^2 - cos(x)^2) + m) - ((-cos(2*x)) + m)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_squared_pythagorean_passthrough_forward_regression() {
        let expr_text = "(((sin(x)^2 + cos(x)^2)^2) + m) - (((1)^2) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_negative_double_sine_passthrough_forward_regression() {
        let expr_text = "((-2*sin(x)*cos(x)) + m) - ((-sin(2*x)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_keeps_strict_solve_difference_of_cubes_fast_path() {
        let expr_text = "(x^3 - y^3) / (x - y)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Solve;
        options.shared.semantics.domain_mode = crate::DomainMode::Strict;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        let result_str = DisplayExpr {
            context: &engine.simplifier.context,
            id: result,
        }
        .to_string();
        assert!(
            result_str == "x^2 + y^2 + x * y" || result_str == "x^2 + y^2 + y * x",
            "expected strict solve steps-off cubes fast path, got: {result_str}"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_negative_double_sine_passthrough_forward_regression()
    {
        let expr_text = "((-2*sin(x)*cos(x)) + m) - ((-sin(2*x)) + m)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_sophie_germain_passthrough_forward_regression() {
        let expr_text = "((x^4 + 4*y^4) + m) - (((x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_trig_ratio_alias_passthrough_forward_regression() {
        let expr_text = "((sin(2*x)/cos(x+x)) + m) - ((tan(2*x)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_half_angle_tan_zero_difference_regression() {
        let expr_text = "(1 - cos(2*x))/sin(2*x) - tan(x)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_morrie_scaled_difference_regression() {
        let expr_text = "k*(cos(x)*cos(2*x)*cos(4*x)) - k*(sin(8*x)/(8*sin(x)))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_full_mixed_identity_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_full_mixed_identity_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_polynomial_against_hyperbolic_pythagorean_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_default_rules_simplify_steps_off_handles_triple_sine_against_polynomial_partner_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)";
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn from_profile_simplify_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut cache = crate::profile_cache::ProfileCache::new();
        let profile = cache.get_or_build(&effective_options);
        let mut simplifier =
            crate::Simplifier::from_profile_with_context(profile, cas_ast::Context::new());
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_surfaces_partial_result_warning_when_time_budget_is_hit() {
        let mut engine = Engine::new();
        let expr_text = "a + b";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.time_budget_ms = Some(0);

        let (
            result,
            warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "a + b"
        );
        assert!(
            warnings.iter().any(|warning| {
                warning.rule_name == "Simplification Time Budget"
                    && warning.message.contains("Partial result")
            }),
            "expected partial-result timeout warning, got: {warnings:?}"
        );
    }

    #[test]
    fn eval_simplify_surfaces_partial_result_warning_when_root_shortcut_nested_timeout_is_hit() {
        let mut engine = Engine::new();
        let expr_text = "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (log(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(1);

        let (
            result,
            warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        let rendered = DisplayExpr {
            context: &engine.simplifier.context,
            id: result,
        }
        .to_string();
        assert!(
            !rendered.is_empty(),
            "expected a partial result to be returned before timeout"
        );
        assert!(
            warnings.iter().any(|warning| {
                warning.rule_name == "Simplification Time Budget"
                    && warning.message.contains("Partial result")
            }),
            "expected nested root-shortcut timeout warning, got: {warnings:?}"
        );
    }
}
