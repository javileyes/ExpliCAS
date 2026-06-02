use super::atanh_open_interval_domain::atanh_diff_required_conditions;
use super::gap_presentation::primitive_positive_gap;
use super::polynomial_support::{
    nonzero_affine_variable_derivative, polynomial_radicand_for_calculus_presentation,
};
use super::presentation_utils::{calculus_sqrt_like_radicand, squared_expr};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::root_forms::extract_square_root_base;
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};

const CALCULUS_DOMAIN_PROOF_DEPTH: usize = 12;

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

pub(super) fn reciprocal_trig_and_log_diff_required_conditions(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    reciprocal_trig_diff_required_conditions(ctx, target, var_name)
        .into_iter()
        .chain(log_reciprocal_abs_or_sqrt_negative_even_power_diff_required_conditions(ctx, target))
        .collect()
}

pub(super) fn diff_required_conditions_for_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Vec<crate::ImplicitCondition> {
    atanh_diff_required_conditions(ctx, target, var_name)
        .into_iter()
        .chain(reciprocal_trig_and_log_diff_required_conditions(
            ctx, target, var_name,
        ))
        .chain(zero_base_variable_exponent_diff_required_conditions(
            ctx, target, var_name,
        ))
        .collect()
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
