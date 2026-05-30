use super::gap_presentation::primitive_positive_gap;
use super::polynomial_support::{
    nonzero_affine_variable_derivative, polynomial_is_strictly_positive_everywhere,
    polynomial_radicand_for_calculus_presentation,
};
use super::presentation_utils::{
    calculus_sqrt_like_radicand, scaled_sqrt_argument_for_calculus_presentation, squared_expr,
};
use super::scalar_presentation::{
    add_rational_for_calculus_presentation,
    positive_constant_over_inverse_sqrt_arg_for_calculus_presentation,
    rational_scaled_single_factor_allow_unit, scale_expr_for_calculus_presentation,
    subtract_expr_for_calculus_presentation, subtract_from_one_for_calculus_presentation,
};
use super::surd_quotient_args::{
    arctan_self_normalized_surd_quotient_parts, atanh_arg_over_sqrt_parts,
};
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

const CALCULUS_DOMAIN_PROOF_DEPTH: usize = 12;

fn nonfinite_or_undefined_calculus_target(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Neg(inner) | Expr::Hold(inner) => nonfinite_or_undefined_calculus_target(ctx, *inner),
        _ => false,
    }
}

fn logarithm_known_empty_positive_domain(ctx: &mut Context, target: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return false;
    };
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) if args.len() == 1 => {
            cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
                ctx,
                args[0],
                CALCULUS_DOMAIN_PROOF_DEPTH,
            )
        }
        Some(BuiltinFn::Log) if args.len() == 2 => {
            cas_math::calculus_domain_support::log_base_is_invalid_over_reals(
                ctx,
                args[0],
                CALCULUS_DOMAIN_PROOF_DEPTH,
            ) || cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
                ctx,
                args[1],
                CALCULUS_DOMAIN_PROOF_DEPTH,
            )
        }
        _ => false,
    }
}

fn sqrt_known_empty_positive_domain(ctx: &mut Context, target: ExprId, var_name: &str) -> bool {
    let Some(radicand) = extract_square_root_base(ctx, target) else {
        return false;
    };
    if !contains_named_var(ctx, radicand, var_name) {
        return cas_ast::views::as_rational_const(ctx, radicand, 8)
            .is_some_and(|value| value.is_negative());
    }

    cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
        ctx,
        radicand,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    )
}

pub(crate) fn diff_target_known_undefined_over_reals(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    nonfinite_or_undefined_calculus_target(ctx, target)
        || logarithm_known_empty_positive_domain(ctx, target)
        || sqrt_known_empty_positive_domain(ctx, target, var_name)
}

pub(super) fn shifted_sqrt_product_required_conditions(
    radicand: ExprId,
    shift: &BigRational,
    shifted_sqrt: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let mut required_conditions = vec![crate::ImplicitCondition::Positive(radicand)];
    if !shift.is_positive() {
        required_conditions.push(crate::ImplicitCondition::NonZero(shifted_sqrt));
    }
    required_conditions
}

pub(super) fn positive_polynomial_radicand_required_conditions(
    radicand: ExprId,
    radicand_poly: &Polynomial,
) -> Vec<crate::ImplicitCondition> {
    if polynomial_is_strictly_positive_everywhere(radicand_poly) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(radicand)]
    }
}

pub(super) fn positive_polynomial_radicand_and_nonzero_required_conditions(
    radicand: ExprId,
    radicand_poly: &Polynomial,
    nonzero_witness: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let mut required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, radicand_poly);
    required_conditions.push(crate::ImplicitCondition::NonZero(nonzero_witness));
    required_conditions
}

pub(super) fn atanh_open_interval_condition(ctx: &mut Context, arg: ExprId) -> ExprId {
    if let Some((num, radicand)) = atanh_arg_over_sqrt_parts(ctx, arg) {
        let num_square = squared_expr(ctx, num);
        return ctx.add(Expr::Sub(radicand, num_square));
    }

    if let Some(radicand) = extract_square_root_base(ctx, arg) {
        return subtract_from_one_for_calculus_presentation(ctx, radicand);
    }

    if let Some(open_interval) = denominator_scaled_sqrt_open_interval_condition(ctx, arg) {
        return open_interval;
    }

    if let Some(scaled_radicand) = scaled_sqrt_square_for_open_interval_condition(ctx, arg) {
        return subtract_from_one_for_calculus_presentation(ctx, scaled_radicand);
    }

    let arg_sq = squared_expr(ctx, arg);
    subtract_from_one_for_calculus_presentation(ctx, arg_sq)
}

fn scaled_sqrt_square_for_open_interval_condition(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<ExprId> {
    let factors = match ctx.get(arg).clone() {
        Expr::Neg(inner) | Expr::Hold(inner) => {
            return scaled_sqrt_square_for_open_interval_condition(ctx, inner);
        }
        Expr::Mul(_, _) => cas_math::expr_nary::mul_leaves(ctx, arg),
        _ => return None,
    };
    let mut radicand = None;
    let mut squared_scale_factors = Vec::new();
    let mut has_symbolic_scale_factor = false;

    for factor in factors {
        if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_none() {
            has_symbolic_scale_factor = true;
        }
        squared_scale_factors.push(squared_expr(ctx, factor));
    }

    let radicand = radicand?;
    if squared_scale_factors.is_empty() || !has_symbolic_scale_factor {
        return None;
    }
    squared_scale_factors.push(radicand);

    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &squared_scale_factors,
    ))
}

fn denominator_scaled_sqrt_open_interval_condition(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<ExprId> {
    let (radicand, denominator, sqrt_scale) =
        denominator_scaled_sqrt_open_interval_parts(ctx, arg)?;
    let denominator_square = squared_expr(ctx, denominator);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    Some(subtract_expr_for_calculus_presentation(
        ctx,
        denominator_square,
        scaled_radicand,
    ))
}

fn denominator_scaled_sqrt_open_interval_parts(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId, BigRational)> {
    match ctx.get(arg).clone() {
        Expr::Neg(inner) => {
            let (radicand, denominator, sqrt_scale) =
                denominator_scaled_sqrt_open_interval_parts(ctx, inner)?;
            Some((radicand, denominator, -sqrt_scale))
        }
        Expr::Hold(inner) => {
            let (radicand, denominator, sqrt_scale) =
                denominator_scaled_sqrt_open_interval_parts(ctx, inner)?;
            Some((radicand, denominator, sqrt_scale))
        }
        Expr::Mul(_, _) => {
            let (outer_scale, inner) = rational_scaled_single_factor_allow_unit(ctx, arg)?;
            if outer_scale.is_one() {
                return None;
            }
            let (radicand, denominator, sqrt_scale) =
                denominator_scaled_sqrt_open_interval_parts(ctx, inner)?;
            Some((radicand, denominator, sqrt_scale * outer_scale))
        }
        Expr::Div(num, den) => {
            if cas_ast::views::as_rational_const(ctx, den, 8).is_some() {
                return None;
            }
            let (radicand, sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, num)?;
            Some((radicand, den, sqrt_scale))
        }
        _ => None,
    }
}

pub(super) fn collect_atanh_open_interval_conditions(
    ctx: &mut Context,
    root: ExprId,
) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr).clone() {
            Expr::Function(fn_id, args) => {
                if ctx.builtin_of(fn_id) == Some(BuiltinFn::Atanh) && args.len() == 1 {
                    let arg = args[0];
                    out.push(atanh_open_interval_condition(ctx, arg));
                }
                stack.extend(args);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                stack.push(l);
                stack.push(r);
            }
            Expr::Pow(base, exp) => {
                stack.push(base);
                stack.push(exp);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(inner),
            Expr::Matrix { data, .. } => stack.extend(data),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    out
}

pub(super) fn atanh_diff_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    match ctx.get(target).clone() {
        Expr::Neg(inner) => return atanh_diff_required_conditions(ctx, inner, var_name),
        Expr::Div(inner, denominator) if !contains_named_var(ctx, denominator, var_name) => {
            return atanh_diff_required_conditions(ctx, inner, var_name);
        }
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, target);
            let mut conditions = Vec::new();
            for (idx, factor) in factors.iter().enumerate() {
                if !factors.iter().enumerate().all(|(other_idx, other)| {
                    other_idx == idx || !contains_named_var(ctx, *other, var_name)
                }) {
                    continue;
                }
                conditions.extend(atanh_diff_required_conditions(ctx, *factor, var_name));
            }
            if !conditions.is_empty() {
                return conditions;
            }
        }
        _ => {}
    }

    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => {
            return collect_atanh_open_interval_conditions(ctx, target)
                .into_iter()
                .map(crate::ImplicitCondition::Positive)
                .collect()
        }
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return collect_atanh_open_interval_conditions(ctx, target)
            .into_iter()
            .map(crate::ImplicitCondition::Positive)
            .collect();
    }

    let arg = args[0];
    if atanh_self_normalized_surd_quotient_positive_gap(ctx, arg, var_name).is_some() {
        return Vec::new();
    }
    let open_interval = if let Some((numerator_value, denominator)) =
        positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(ctx, arg)
    {
        add_rational_for_calculus_presentation(ctx, denominator, -numerator_value)
    } else {
        atanh_open_interval_condition(ctx, arg)
    };
    vec![crate::ImplicitCondition::Positive(open_interval)]
}

pub(super) fn atanh_self_normalized_surd_quotient_positive_gap(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let arg = match ctx.get(arg).clone() {
        Expr::Neg(inner) => inner,
        _ => arg,
    };
    let (num, radicand) = arctan_self_normalized_surd_quotient_parts(ctx, arg)?;
    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let gap_poly = radicand_poly.sub(&num_poly.mul(&num_poly));
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_constant = gap_poly.coeffs.first().cloned()?;
    gap_constant.is_positive().then_some(gap_constant)
}

pub(super) fn acosh_sqrt_diff_required_conditions(
    ctx: &mut Context,
    target: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return vec![],
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Acosh) || args.len() != 1 {
        return vec![];
    }

    let Some(radicand) = extract_square_root_base(ctx, args[0]) else {
        return vec![];
    };

    vec![crate::ImplicitCondition::Positive(radicand)]
}

pub(super) fn reciprocal_trig_diff_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    let target = match ctx.get(target).clone() {
        Expr::Neg(inner) => inner,
        _ => target,
    };
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        Expr::Div(numerator, denominator) => {
            if cas_ast::views::as_rational_const(ctx, numerator, 8)
                .is_none_or(|value| !value.is_one())
            {
                return vec![];
            }
            let Expr::Function(den_fn_id, den_args) = ctx.get(denominator).clone() else {
                return vec![];
            };
            let den_builtin = match ctx.builtin_of(den_fn_id) {
                Some(builtin @ (BuiltinFn::Cos | BuiltinFn::Sin)) => builtin,
                _ => return vec![],
            };
            if den_args.len() == 1
                && reciprocal_trig_diff_pole_arg_needs_explicit_condition(
                    ctx,
                    den_args[0],
                    var_name,
                )
            {
                let denominator_arg = reciprocal_trig_diff_pole_display_arg(ctx, den_args[0]);
                let denominator = ctx.call_builtin(den_builtin, vec![denominator_arg]);
                return vec![crate::ImplicitCondition::NonZero(denominator)];
            }
            return vec![];
        }
        _ => return vec![],
    };
    if args.len() != 1 {
        return vec![];
    }
    if !reciprocal_trig_diff_pole_arg_needs_explicit_condition(ctx, args[0], var_name) {
        return vec![];
    }

    let denominator_builtin = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sec) => BuiltinFn::Cos,
        Some(BuiltinFn::Csc) => BuiltinFn::Sin,
        _ => return vec![],
    };
    let denominator_arg = reciprocal_trig_diff_pole_display_arg(ctx, args[0]);
    let denominator = ctx.call_builtin(denominator_builtin, vec![denominator_arg]);
    vec![crate::ImplicitCondition::NonZero(denominator)]
}

pub(super) fn inverse_reciprocal_trig_affine_abs_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return vec![];
    };
    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec | BuiltinFn::Arccsc | BuiltinFn::Acsc) => {}
        _ => return vec![],
    }
    if args.len() != 1 {
        return vec![];
    }

    let arg = args[0];
    if nonzero_affine_variable_derivative(ctx, arg, var_name).is_none() {
        return vec![];
    }
    let arg_sq = squared_expr(ctx, arg);
    let one = ctx.num(1);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let (gap, _) = primitive_positive_gap(ctx, raw_gap);
    vec![crate::ImplicitCondition::Positive(gap)]
}

fn reciprocal_trig_diff_pole_display_arg(ctx: &mut Context, arg: ExprId) -> ExprId {
    calculus_sqrt_like_radicand(ctx, arg)
        .map(|radicand| ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]))
        .unwrap_or(arg)
}

fn reciprocal_trig_diff_pole_arg_needs_explicit_condition(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> bool {
    if nonzero_affine_variable_derivative(ctx, arg, var_name).is_some() {
        return true;
    }

    let Some(radicand) = calculus_sqrt_like_radicand(ctx, arg) else {
        return false;
    };
    polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)
        .map(|poly| !poly.derivative().is_zero())
        .unwrap_or(false)
}

pub(super) fn log_reciprocal_abs_or_sqrt_negative_even_power_diff_required_conditions(
    ctx: &Context,
    target: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return vec![];
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return vec![];
    }

    if let Expr::Div(num, den) = ctx.get(args[0]) {
        if cas_ast::views::as_rational_const(ctx, *num, 8).is_some_and(|value| value.is_one()) {
            if let Some(base) = cas_math::expr_extract::extract_abs_argument_view(ctx, *den) {
                return vec![crate::ImplicitCondition::NonZero(base)];
            }
            if let Some(radicand) = calculus_sqrt_like_radicand(ctx, *den) {
                if let Some(base) = nonzero_even_integer_power_base(ctx, radicand) {
                    return vec![crate::ImplicitCondition::NonZero(base)];
                }
            }
        }
    }

    let Some(radicand) = calculus_sqrt_like_radicand(ctx, args[0]) else {
        return vec![];
    };
    negative_even_integer_power_base(ctx, radicand)
        .map(|base| vec![crate::ImplicitCondition::NonZero(base)])
        .unwrap_or_default()
}

fn negative_even_integer_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exp_value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    if !exp_value.is_integer() {
        return None;
    }

    let exp_int = exp_value.to_integer();
    let zero: BigInt = 0.into();
    let two: BigInt = 2.into();
    (exp_int < zero && exp_int.mod_floor(&two).is_zero()).then_some(*base)
}

fn nonzero_even_integer_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exp_value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    if !exp_value.is_integer() || exp_value.is_zero() {
        return None;
    }

    let exp_int = exp_value.to_integer();
    let two: BigInt = 2.into();
    exp_int.mod_floor(&two).is_zero().then_some(*base)
}

pub(super) fn zero_base_variable_exponent_diff_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    let (base, exponent) = match ctx.get(target) {
        Expr::Pow(base, exponent) => (*base, *exponent),
        _ => return Vec::new(),
    };
    if contains_named_var(ctx, base, var_name) || !contains_named_var(ctx, exponent, var_name) {
        return Vec::new();
    };
    if cas_ast::views::as_rational_const(ctx, base, 8).is_some_and(|value| value.is_zero()) {
        if zero_base_variable_exponent_positive_domain_is_empty(ctx, exponent) {
            Vec::new()
        } else {
            vec![crate::ImplicitCondition::Positive(exponent)]
        }
    } else {
        Vec::new()
    }
}

fn zero_base_variable_exponent_positive_domain_is_empty(
    ctx: &mut Context,
    exponent: ExprId,
) -> bool {
    cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
        ctx,
        exponent,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    )
}

pub(super) fn atanh_known_empty_open_interval_gap(
    ctx: &mut Context,
    target: ExprId,
) -> Option<ExprId> {
    let arg = atanh_open_interval_arg(ctx, target)?;
    let gap = atanh_open_interval_condition(ctx, arg);
    if cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
        ctx,
        gap,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    ) || arg_is_proven_outside_open_unit_interval(ctx, arg)
        || known_constant_abs_exceeds_one(ctx, arg)
    {
        return Some(gap);
    }

    None
}

fn arg_is_proven_outside_open_unit_interval(ctx: &mut Context, arg: ExprId) -> bool {
    let one = ctx.num(1);
    let at_or_above_one = ctx.add(Expr::Sub(arg, one));
    if cas_math::calculus_domain_support::nonnegative_condition_is_proven_over_reals(
        ctx,
        at_or_above_one,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    ) {
        return true;
    }

    let neg_arg = ctx.add(Expr::Neg(arg));
    let one = ctx.num(1);
    let at_or_below_minus_one = ctx.add(Expr::Sub(neg_arg, one));
    cas_math::calculus_domain_support::nonnegative_condition_is_proven_over_reals(
        ctx,
        at_or_below_minus_one,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    )
}

fn atanh_open_interval_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    match ctx.get(target) {
        Expr::Neg(inner) | Expr::Hold(inner) => return atanh_open_interval_arg(ctx, *inner),
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, *den, 8).is_some() => {
            return atanh_open_interval_arg(ctx, *num);
        }
        Expr::Mul(left, right) if cas_ast::views::as_rational_const(ctx, *left, 8).is_some() => {
            return atanh_open_interval_arg(ctx, *right);
        }
        Expr::Mul(left, right) if cas_ast::views::as_rational_const(ctx, *right, 8).is_some() => {
            return atanh_open_interval_arg(ctx, *left);
        }
        _ => {}
    }

    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    (ctx.builtin_of(*fn_id) == Some(BuiltinFn::Atanh) && args.len() == 1).then_some(args[0])
}

pub(super) fn bounded_inverse_trig_known_empty_open_interval_gap(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    match ctx.get(target).clone() {
        Expr::Neg(inner) | Expr::Hold(inner) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, inner, var_name);
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, num, var_name);
        }
        Expr::Mul(left, right) if cas_ast::views::as_rational_const(ctx, left, 8).is_some() => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, right, var_name);
        }
        Expr::Mul(left, right) if cas_ast::views::as_rational_const(ctx, right, 8).is_some() => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, left, var_name);
        }
        Expr::Add(left, right) if domain_policy_constant_offset(ctx, left) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, right, var_name);
        }
        Expr::Add(left, right) if domain_policy_constant_offset(ctx, right) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, left, var_name);
        }
        Expr::Sub(left, right) if domain_policy_constant_offset(ctx, left) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, right, var_name);
        }
        Expr::Sub(left, right) if domain_policy_constant_offset(ctx, right) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, left, var_name);
        }
        _ => {}
    }

    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos)
        )
    {
        return None;
    }

    let arg = args[0];
    if bounded_inverse_trig_finite_constant_domain_arg(ctx, arg) {
        return None;
    }

    let gap = bounded_inverse_trig_open_interval_gap(ctx, arg);
    if cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
        ctx,
        gap,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    ) || known_constant_abs_exceeds_one(ctx, arg)
        || (contains_named_var(ctx, arg, var_name)
            && arg_is_proven_outside_open_unit_interval(ctx, arg))
    {
        return Some(gap);
    }

    None
}

fn bounded_inverse_trig_finite_constant_domain_arg(ctx: &Context, arg: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, arg, 8)
        .is_some_and(|value| value.abs() <= BigRational::one())
}

fn domain_policy_constant_offset(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8).is_some()
        || matches!(ctx.get(expr), Expr::Constant(Constant::Pi | Constant::E))
}

fn known_constant_abs_exceeds_one(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Pi | Constant::E) => true,
        Expr::Neg(inner) | Expr::Hold(inner) => known_constant_abs_exceeds_one(ctx, *inner),
        _ => false,
    }
}

fn bounded_inverse_trig_open_interval_gap(ctx: &mut Context, arg: ExprId) -> ExprId {
    if let Expr::Neg(inner) = ctx.get(arg) {
        return bounded_inverse_trig_open_interval_gap(ctx, *inner);
    }
    if let Some(radicand) = extract_square_root_base(ctx, arg) {
        return subtract_from_one_for_calculus_presentation(ctx, radicand);
    }

    let arg_square = squared_expr(ctx, arg);
    subtract_from_one_for_calculus_presentation(ctx, arg_square)
}

pub(super) fn inverse_reciprocal_trig_bounded_trig_empty_open_interval_gap(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, bool)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let (trig_arg, should_emit_hint) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec | BuiltinFn::Arccsc | BuiltinFn::Acsc) => {
            variable_dependent_direct_bounded_trig_arg(ctx, args[0], var_name)?;
            (args[0], true)
        }
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos) => {
            let canonical = variable_dependent_reciprocal_bounded_trig_arg(ctx, args[0], var_name)?;
            (args[0], canonical)
        }
        _ => return None,
    };

    let two = ctx.num(2);
    let trig_square = ctx.add(Expr::Pow(trig_arg, two));
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec | BuiltinFn::Arccsc | BuiltinFn::Acsc) => {
            let one = ctx.num(1);
            Some((ctx.add(Expr::Sub(trig_square, one)), should_emit_hint))
        }
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos) => {
            let one = ctx.num(1);
            Some((ctx.add(Expr::Sub(one, trig_square)), should_emit_hint))
        }
        _ => None,
    }
}

fn variable_dependent_direct_bounded_trig_arg(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<()> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Sin | BuiltinFn::Cos)
        )
        && contains_named_var(ctx, args[0], var_name))
    .then_some(())
}

fn variable_dependent_reciprocal_bounded_trig_arg(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<bool> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Sec | BuiltinFn::Csc)
                )
                && contains_named_var(ctx, args[0], var_name) =>
        {
            Some(true)
        }
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, *num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            variable_dependent_direct_bounded_trig_arg(ctx, *den, var_name).map(|()| false)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                .is_some_and(|value| value == -BigRational::one()) =>
        {
            variable_dependent_direct_bounded_trig_arg(ctx, *base, var_name).map(|()| false)
        }
        _ => None,
    }
}
