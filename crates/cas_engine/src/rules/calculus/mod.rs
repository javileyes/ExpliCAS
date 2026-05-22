//! Calculus rules: differentiation, integration, summation, and products.
//!
//! This module is split into submodules:
//! - `differentiation`: symbolic derivative computation
//! - `integration`: symbolic integral computation + helpers
//! - `summation`: finite sum/product evaluation (SumRule, ProductRule)

mod differentiation;
mod integration;
mod summation;

use crate::define_rule;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{
    render_diff_desc_with, render_integrate_desc_with, try_extract_diff_call,
    try_extract_integrate_call,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::{extract_square_root_base, try_rewrite_simplify_square_root_expr};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

use differentiation::differentiate;
use integration::{
    integrate, integrate_required_nonzero_conditions, integrate_required_positive_conditions,
};

const CALCULUS_DOMAIN_PROOF_DEPTH: usize = 12;

fn atanh_sqrt_known_empty_open_interval_gap(ctx: &mut Context, target: ExprId) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;

    let gap = subtract_from_one_for_calculus_presentation(ctx, radicand);
    let raw_nonpositive_gap = ctx.add(Expr::Neg(gap));
    let nonpositive_gap =
        cas_math::expr_normalization::normalize_condition_expr(ctx, raw_nonpositive_gap);
    cas_math::prove_sign::prove_nonnegative_depth_with(
        ctx,
        nonpositive_gap,
        CALCULUS_DOMAIN_PROOF_DEPTH,
        true,
        |_ctx, _expr, _depth| cas_math::tri_proof::TriProof::Unknown,
    )
    .is_proven()
    .then_some(gap)
}

fn inverse_reciprocal_trig_bounded_trig_empty_open_interval_gap(
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

fn atanh_diff_required_conditions(
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

fn acosh_sqrt_diff_required_conditions(
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

fn reciprocal_trig_diff_required_conditions(
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

fn reciprocal_trig_diff_pole_display_arg(ctx: &mut Context, arg: ExprId) -> ExprId {
    calculus_sqrt_like_radicand(ctx, arg)
        .map(|radicand| ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]))
        .unwrap_or(arg)
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

fn log_reciprocal_abs_or_sqrt_negative_even_power_diff_required_conditions(
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

fn unwrap_internal_hold_for_calculus(ctx: &mut Context, target: ExprId) -> ExprId {
    cas_ast::hold::strip_all_holds(ctx, target)
}

fn variable_named(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

fn is_half_power_exponent(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8)
        .is_some_and(|value| value == BigRational::new(1.into(), 2.into()))
}

fn positive_scaled_variable_factor(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if variable_named(ctx, target, var_name) {
        return Some(BigRational::one());
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut saw_variable = false;
    for factor in factors {
        if variable_named(ctx, factor, var_name) {
            if saw_variable {
                return None;
            }
            saw_variable = true;
            continue;
        }

        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        if !value.is_positive() {
            return None;
        }
        scale *= value;
    }

    saw_variable.then_some(scale)
}

fn nonzero_affine_variable_derivative(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 1,
        max_pow_exp: 1,
    };
    let poly = multipoly_from_expr(ctx, target, &budget).ok()?;
    if poly.vars.len() != 1 || poly.vars[0] != var_name || poly.total_degree() > 1 {
        return None;
    }

    let mut linear_coeff = BigRational::zero();
    for (coeff, mono) in &poly.terms {
        match mono.as_slice() {
            [0] => {}
            [1] => linear_coeff += coeff.clone(),
            _ => return None,
        }
    }

    (!linear_coeff.is_zero()).then_some(linear_coeff)
}

fn arctan_sqrt_scaled_variable_arg(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let scale = positive_scaled_variable_factor(ctx, radicand, var_name)
        .or_else(|| nonzero_affine_variable_derivative(ctx, radicand, var_name))?;
    Some((radicand, scale))
}

fn arctan_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }

    let radicand = match ctx.get(args[0]) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            sqrt_args[0]
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => *base,
        _ => return None,
    };

    Some(radicand)
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, offset) =
        if let Some(parts) = arctan_sqrt_positive_shift_primitive_parts(ctx, target, var_name) {
            (parts.primitive_scale, parts.offset)
        } else {
            (
                arctan_sqrt_plus_sqrt_over_x_plus_one_scale(ctx, target, var_name)?,
                BigRational::one(),
            )
        };
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&scale)?;

    let var = ctx.var(var_name);
    let neg_half = ctx.rational(-1, 2);
    let reciprocal_sqrt_var = ctx.add(Expr::Pow(var, neg_half));
    let offset_expr = rational_const_for_calculus_presentation(ctx, offset);
    let unit_shift = ctx.add(Expr::Add(var, offset_expr));
    let two = ctx.num(2);
    let unit_shift_square = ctx.add(Expr::Pow(unit_shift, two));
    let numerator = if numerator_coeff == BigRational::one() {
        reciprocal_sqrt_var
    } else {
        let numerator_coeff = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        ctx.add_raw(Expr::Mul(numerator_coeff, reciprocal_sqrt_var))
    };
    let denominator = if denominator_coeff == BigRational::one() {
        unit_shift_square
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add_raw(Expr::Mul(denominator_coeff, unit_shift_square))
    };
    let result = ctx.add_raw(Expr::Div(numerator, denominator));
    let var = ctx.var(var_name);
    Some((result, vec![crate::ImplicitCondition::Positive(var)]))
}

fn arctan_sqrt_variable_over_positive_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Arctan | BuiltinFn::Atan)
        )
    {
        return None;
    }

    let (argument_scale, numerator, denominator) = match ctx.get(args[0]).clone() {
        Expr::Div(numerator, denominator) => (BigRational::one(), numerator, denominator),
        _ => {
            let (argument_scale, quotient_core) = split_numeric_scale_single_core(ctx, args[0])?;
            if !argument_scale.is_positive() {
                return None;
            }
            let Expr::Div(numerator, denominator) = ctx.get(quotient_core).clone() else {
                return None;
            };
            (argument_scale, numerator, denominator)
        }
    };
    let (numerator_scale, numerator_core) = split_numeric_scale_single_core(ctx, numerator)?;
    let numerator_scale = argument_scale * numerator_scale;
    if !numerator_scale.is_positive() {
        return None;
    }
    let radicand = extract_square_root_base(ctx, numerator_core)?;
    let radicand_scale = positive_scaled_variable_factor(ctx, radicand, var_name)?;
    let numerator_derivative_scale = numerator_scale.clone() * radicand_scale.clone();
    let denominator_variable_scale =
        numerator_scale.clone() * numerator_scale * radicand_scale.clone();

    let denominator_poly = Polynomial::from_expr(ctx, denominator, var_name).ok()?;
    if denominator_poly.degree() != 1 {
        return None;
    }
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_slope = denominator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !denominator_slope.is_positive() || denominator_constant.is_negative() {
        return None;
    }

    let var_poly = Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        var_name.to_string(),
    );
    let scaled_var_poly = Polynomial::new(
        vec![BigRational::zero(), denominator_variable_scale],
        var_name.to_string(),
    );
    let scale_poly = Polynomial::new(vec![numerator_derivative_scale], var_name.to_string());
    let two = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let mut numerator_poly = denominator_poly
        .sub(&var_poly.mul(&denominator_poly.derivative()).mul(&two))
        .mul(&scale_poly);
    if numerator_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let mut denominator_coeff = BigRational::from_integer(2.into());
    let numerator_content = numerator_poly.content();
    if numerator_content.is_positive() && numerator_content.denom().is_one() {
        let common_integer = numerator_content
            .numer()
            .gcd(denominator_coeff.numer())
            .abs();
        if common_integer > BigInt::from(1) {
            let common = BigRational::from_integer(common_integer);
            numerator_poly = numerator_poly.div_scalar(&common);
            denominator_coeff /= common;
        }
    }

    let denominator_sum_poly = denominator_poly
        .mul(&denominator_poly)
        .add(&scaled_var_poly);
    let denominator_sum = denominator_sum_poly.to_expr(ctx);
    let numerator_expr = numerator_poly.to_expr(ctx);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(sqrt_var);
    denominator_factors.push(denominator_sum);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let result = ctx.add(Expr::Div(numerator_expr, denominator));
    let var = ctx.var(var_name);

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        vec![crate::ImplicitCondition::Positive(var)],
    ))
}

fn constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        let denominator_scale = cas_ast::views::as_rational_const(ctx, outer_den, 8)?;
        if denominator_scale.is_zero() {
            return None;
        }
        let (derivative, required_conditions) =
            arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                ctx, inner, var_name,
            )?;
        let scaled = scale_compact_derivative_by_rational(
            ctx,
            derivative,
            BigRational::one() / denominator_scale,
        );
        return Some((cas_ast::hold::wrap_hold(ctx, scaled), required_conditions));
    }

    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let (derivative, required_conditions) =
        arctan_sqrt_variable_over_positive_affine_derivative_presentation(ctx, inner, var_name)?;
    let scaled = scale_compact_derivative_by_rational(ctx, derivative, scale);
    Some((cas_ast::hold::wrap_hold(ctx, scaled), required_conditions))
}

fn compact_sqrt_var_over_var_times_positive_shift_square_diff_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (num_scale, shift_square, denominator_scale) =
        sqrt_var_over_var_times_positive_shift_square_parts(ctx, expr, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    let coefficient = num_scale / denominator_scale;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let var = ctx.var(var_name);
    let neg_half = ctx.rational(-1, 2);
    let reciprocal_sqrt_var = ctx.add(Expr::Pow(var, neg_half));
    let numerator = if numerator_coeff == BigRational::one() {
        reciprocal_sqrt_var
    } else {
        let numerator_coeff = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        ctx.add_raw(Expr::Mul(numerator_coeff, reciprocal_sqrt_var))
    };
    let denominator = if denominator_coeff == BigRational::one() {
        shift_square
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add_raw(Expr::Mul(denominator_coeff, shift_square))
    };
    let result = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(result)
}

fn sqrt_var_over_var_times_positive_shift_square_parts(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BigRational)> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
        let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
        if !is_calculus_var(ctx, radicand, var_name) {
            return None;
        }
        let (shift_square, denominator_scale) =
            var_times_positive_shift_square_denominator_parts(ctx, den, var_name)?;
        return Some((num_scale, shift_square, denominator_scale));
    }

    let mut num_scale = BigRational::one();
    let mut saw_sqrt_var = false;
    let mut denominator_factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            num_scale *= value;
        } else if calculus_sqrt_like_radicand(ctx, factor)
            .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        {
            if saw_sqrt_var {
                return None;
            }
            saw_sqrt_var = true;
        } else if let Expr::Pow(base, exp) = ctx.get(factor) {
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                != Some(BigRational::new((-1).into(), 1.into()))
            {
                return None;
            }
            denominator_factors.push(*base);
        } else {
            return None;
        }
    }

    if !saw_sqrt_var || denominator_factors.is_empty() {
        return None;
    }
    let denominator_factors = denominator_factors
        .into_iter()
        .flat_map(|factor| cas_math::expr_nary::mul_leaves(ctx, factor))
        .collect::<Vec<_>>();
    let (shift_square, denominator_scale) =
        var_times_positive_shift_square_denominator_factor_parts(
            ctx,
            denominator_factors,
            var_name,
        )?;
    Some((num_scale, shift_square, denominator_scale))
}

fn var_times_positive_shift_square_denominator_parts(
    ctx: &Context,
    den: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    var_times_positive_shift_square_denominator_factor_parts(
        ctx,
        cas_math::expr_nary::mul_leaves(ctx, den).to_vec(),
        var_name,
    )
}

fn var_times_positive_shift_square_denominator_factor_parts(
    ctx: &Context,
    factors: Vec<ExprId>,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    let mut denominator_scale = BigRational::one();
    let mut saw_var_factor = false;
    let mut shift_square = None;
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            denominator_scale *= value;
        } else if is_calculus_var(ctx, factor, var_name) {
            if saw_var_factor {
                return None;
            }
            saw_var_factor = true;
        } else if positive_shift_square_factor_for_calculus_presentation(ctx, factor, var_name)
            .is_some()
        {
            if shift_square.replace(factor).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    saw_var_factor.then_some((shift_square?, denominator_scale))
}

fn positive_shift_square_factor_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if cas_ast::views::as_rational_const(ctx, *exp, 8) != Some(BigRational::from_integer(2.into()))
    {
        return None;
    }
    positive_shift_denominator_scale(ctx, *base, var_name).map(|(_, offset)| offset)
}

struct ArctanSqrtPositiveShiftPrimitiveParts {
    primitive_scale: BigRational,
    offset: BigRational,
}

fn arctan_sqrt_positive_shift_primitive_parts(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtPositiveShiftPrimitiveParts> {
    if let Some((outer_scale, core)) = scaled_nontrivial_core_for_calculus_presentation(ctx, target)
    {
        let inner = arctan_sqrt_positive_shift_primitive_parts(ctx, core, var_name)?;
        return Some(ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: outer_scale * inner.primitive_scale,
            offset: inner.offset,
        });
    }

    if let Some(parts) = arctan_sqrt_positive_shift_combined_quotient_parts(ctx, target, var_name) {
        return Some(parts);
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() != 2 {
        return None;
    }

    let mut arctan_scale_and_offset = None;
    let mut quotient_scale_and_offset = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(parts) = scaled_arctan_sqrt_positive_shift_term(ctx, term, var_name) {
            if arctan_scale_and_offset.replace(parts).is_some() {
                return None;
            }
        } else if let Some(parts) = scaled_sqrt_var_over_positive_shift_term(ctx, term, var_name) {
            if quotient_scale_and_offset.replace(parts).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let (arctan_scale, offset, offset_root) = arctan_scale_and_offset?;
    let (quotient_scale, quotient_offset) = quotient_scale_and_offset?;
    if offset != quotient_offset {
        return None;
    }

    let primitive_scale_from_arctan = arctan_scale * offset.clone() * offset_root;
    let primitive_scale_from_quotient = quotient_scale * offset.clone();
    (primitive_scale_from_arctan == primitive_scale_from_quotient).then_some(
        ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: primitive_scale_from_arctan,
            offset,
        },
    )
}

fn arctan_sqrt_positive_shift_combined_quotient_parts(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtPositiveShiftPrimitiveParts> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let (den_scale, denominator_offset) = positive_shift_denominator_scale(ctx, den, var_name)?;
    if den_scale.is_zero() {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, num);
    if terms.len() != 2 {
        return None;
    }

    let mut sqrt_scale = None;
    let mut arctan_linear = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_sqrt_var_term(ctx, term, var_name) {
            if sqrt_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(parts) =
            scaled_positive_shift_times_arctan_sqrt_term(ctx, term, var_name)
        {
            if arctan_linear.replace(parts).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let quotient_offset = denominator_offset;
    let quotient_scale = sqrt_scale? / den_scale.clone();
    let (arctan_linear_scale, arctan_offset, arctan_offset_root) = arctan_linear?;
    if quotient_offset != arctan_offset {
        return None;
    }
    let arctan_scale = arctan_linear_scale / den_scale;
    let primitive_scale_from_quotient = quotient_scale * quotient_offset.clone();
    let primitive_scale_from_arctan = arctan_scale * quotient_offset.clone() * arctan_offset_root;
    (primitive_scale_from_quotient == primitive_scale_from_arctan).then_some(
        ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: primitive_scale_from_quotient,
            offset: quotient_offset,
        },
    )
}

fn scaled_positive_shift_times_arctan_sqrt_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational, BigRational)> {
    let mut scale = BigRational::one();
    let mut offset = None;
    let mut offset_root = None;
    let mut saw_arctan = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if let Some((factor_offset, factor_root)) =
            arctan_sqrt_positive_shift_arg(ctx, factor, var_name)
        {
            if saw_arctan {
                return None;
            }
            saw_arctan = true;
            offset_root = Some(factor_root);
            match &offset {
                Some(existing) if existing != &factor_offset => return None,
                Some(_) => {}
                None => offset = Some(factor_offset),
            }
        } else if let Some((linear_scale, linear_offset)) =
            positive_shift_denominator_scale(ctx, factor, var_name)
        {
            match &offset {
                Some(existing) if existing != &linear_offset => return None,
                Some(_) => {}
                None => offset = Some(linear_offset),
            }
            scale *= linear_scale;
        } else {
            return None;
        }
    }
    let offset = offset?;
    let offset_root = offset_root?;
    saw_arctan.then_some((scale, offset, offset_root))
}

fn scaled_arctan_sqrt_positive_shift_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational, BigRational)> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let (offset, offset_root) = arctan_sqrt_positive_shift_arg(ctx, core, var_name)?;
    Some((scale, offset, offset_root))
}

fn arctan_sqrt_positive_shift_arg(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }
    let (sqrt_scale, sqrt_core) = split_numeric_scale_single_core(ctx, args[0])?;
    if !sqrt_scale.is_positive() {
        return None;
    }
    let radicand = calculus_sqrt_like_radicand(ctx, sqrt_core)?;
    if !is_calculus_var(ctx, radicand, var_name) {
        return None;
    }
    let offset_root = BigRational::one() / sqrt_scale;
    let offset = offset_root.clone() * offset_root.clone();
    Some((offset, offset_root))
}

fn scaled_sqrt_var_over_positive_shift_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let (outer_scale, core) = split_outer_numeric_mul_for_calculus_presentation(ctx, expr)?;
    let core = unwrap_internal_hold_for_calculus(ctx, core);
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
    let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
    if !is_calculus_var(ctx, radicand, var_name) {
        return None;
    }
    let (den_scale, offset) = positive_shift_denominator_scale(ctx, den, var_name)?;
    Some((outer_scale * num_scale / den_scale, offset))
}

fn split_outer_numeric_mul_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let mut scale = BigRational::one();
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

fn positive_shift_denominator_scale(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut scale = BigRational::one();
    let mut offset = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let constant = poly.coeffs.first()?.clone();
        let slope = poly.coeffs.get(1)?;
        if !slope.is_positive() {
            return None;
        }
        let candidate_offset = constant / slope.clone();
        if !candidate_offset.is_positive() {
            return None;
        }
        scale *= slope.clone();
        if offset.replace(candidate_offset).is_some() {
            return None;
        }
    }
    Some((scale, offset?))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_scale(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if let Some((outer_scale, core)) = scaled_nontrivial_core_for_calculus_presentation(ctx, target)
    {
        if let Some(inner_scale) = arctan_sqrt_plus_sqrt_over_x_plus_one_scale(ctx, core, var_name)
        {
            return Some(outer_scale * inner_scale);
        }
    }

    if let Some(scale) =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale(ctx, target, var_name)
    {
        return Some(scale);
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() != 2 {
        return None;
    }

    let mut arctan_scale = None;
    let mut rational_scale = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_arctan_sqrt_var_term(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_over_x_plus_one_term(ctx, term, var_name) {
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

fn scaled_nontrivial_core_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    (!scale.is_one() && core != expr).then_some((scale, core))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let den_scale = x_plus_one_linear_scale_for_calculus_presentation(ctx, den, var_name)?;
    if den_scale.is_zero() {
        return None;
    }
    let num_scale =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale(ctx, num, var_name)?;
    Some(num_scale / den_scale)
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale(
    ctx: &mut Context,
    numerator: ExprId,
    var_name: &str,
) -> Option<BigRational> {
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
        if let Some(scale) = scaled_arctan_sqrt_var_term(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_term(ctx, term, var_name) {
            if sqrt_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_var_times_arctan_sqrt_var_term(ctx, term, var_name) {
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

fn scaled_arctan_sqrt_var_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    arctan_sqrt_radicand_arg(ctx, core)
        .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        .then_some(scale)
}

fn scaled_sqrt_var_term(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<BigRational> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let radicand = calculus_sqrt_like_radicand(ctx, core)?;
    is_calculus_var(ctx, radicand, var_name).then_some(scale)
}

fn scaled_var_times_arctan_sqrt_var_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let mut scale = BigRational::one();
    let mut saw_var = false;
    let mut saw_arctan = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if is_calculus_var(ctx, factor, var_name) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if arctan_sqrt_radicand_arg(ctx, factor)
            .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        {
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

fn scaled_sqrt_var_over_x_plus_one_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let (outer_scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let core = unwrap_internal_hold_for_calculus(ctx, core);
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    if !is_x_plus_one_for_calculus_presentation(ctx, den, var_name) {
        return None;
    }

    let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
    let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
    is_calculus_var(ctx, radicand, var_name).then_some(outer_scale * num_scale)
}

fn split_numeric_scale_single_core(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if den_scale.is_zero() {
            return None;
        }
        let (num_scale, core) = split_numeric_scale_single_core(ctx, num)?;
        return Some((num_scale / den_scale, core));
    }
    let mut scale = BigRational::one();
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

fn x_plus_one_linear_scale_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let offset = poly.coeffs.first()?;
    let slope = poly.coeffs.get(1)?;
    (offset == slope).then_some(offset.clone())
}

fn is_calculus_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

fn is_x_plus_one_for_calculus_presentation(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok();
    poly.is_some_and(|poly| {
        poly.degree() == 1
            && poly
                .coeffs
                .first()
                .is_some_and(|offset| offset == &BigRational::one())
            && poly
                .coeffs
                .get(1)
                .is_some_and(|slope| slope == &BigRational::one())
    })
}

fn scaled_sqrt_argument_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(radicand) = calculus_sqrt_like_radicand(ctx, expr) {
        return Some((radicand, BigRational::one()));
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (radicand, scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, *inner)?;
            Some((radicand, -scale))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, *den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            let (radicand, num_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, *num)?;
            Some((radicand, num_scale / den_scale))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut radicand = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }

                if radicand.is_none() {
                    if let Some(base) = calculus_sqrt_like_radicand(ctx, factor) {
                        radicand = Some(base);
                        continue;
                    }
                }

                return None;
            }

            Some((radicand?, scale))
        }
        _ => None,
    }
}

fn inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if args.len() != 1 {
        return None;
    }
    let expected_sign = match builtin {
        BuiltinFn::Atan | BuiltinFn::Arctan => BigRational::one(),
        BuiltinFn::Acot | BuiltinFn::Arccot => -BigRational::one(),
        _ => return None,
    };
    if derivative_sign != expected_sign {
        return None;
    }

    let (radicand, sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, args[0])?;
    if sqrt_scale.is_zero() || sqrt_scale.is_one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * sqrt_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    let radicand_gap = add_one_for_calculus_presentation(ctx, scaled_radicand);
    let (numerator_coeff, denominator_coeff, radicand_gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_gap,
        );
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_coeff / denominator_coeff))?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn constant_scaled_inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        inner,
        var_name,
        BigRational::one(),
    )
    .or_else(|| {
        inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            inner,
            var_name,
            -BigRational::one(),
        )
    })?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (radicand, sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(
        ctx,
        match ctx.get(target).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(
                            BuiltinFn::Atan
                                | BuiltinFn::Arctan
                                | BuiltinFn::Acot
                                | BuiltinFn::Arccot
                        )
                    ) =>
            {
                args[0]
            }
            _ => return None,
        },
    )?;
    if sqrt_scale.is_zero() || sqrt_scale.abs() == BigRational::one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    )
    .or_else(|| {
        inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            target,
            var_name,
            -BigRational::one(),
        )
    })?;
    let required_conditions = if polynomial_is_strictly_positive_everywhere(&radicand_poly) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(radicand)]
    };
    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

fn arctan_sqrt_affine_partition_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() > 1 || denominator_poly.degree() > 1 {
        return None;
    }

    let partition_sum = numerator_poly.add(&denominator_poly);
    if partition_sum.degree() != 0 {
        return None;
    }
    let partition_total = partition_sum
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !partition_total.is_positive() {
        return None;
    }

    let wronskian = numerator_poly
        .derivative()
        .mul(&denominator_poly)
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if wronskian.degree() != 0 {
        return None;
    }
    let wronskian_value = wronskian
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if wronskian_value.is_zero() {
        return Some(ctx.num(0));
    }

    let coefficient =
        derivative_sign * wronskian_value / (BigRational::from_integer(2.into()) * partition_total);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let sqrt_num = ctx.call_builtin(BuiltinFn::Sqrt, vec![num]);
    let sqrt_den = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_num, sqrt_den]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn arctan_sqrt_polynomial_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() == 0
        || denominator_poly.degree() == 0
        || numerator_poly.degree() > 2
        || denominator_poly.degree() > 2
    {
        return None;
    }

    let sum_poly = numerator_poly.add(&denominator_poly);
    if sum_poly.degree() == 0 || sum_poly.degree() > 2 {
        return None;
    }

    let wronskian = numerator_poly
        .derivative()
        .mul(&denominator_poly)
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if wronskian.degree() > 1 {
        return None;
    }
    if wronskian.is_zero() {
        return Some(ctx.num(0));
    }

    let sum_expr = sum_poly.to_expr(ctx);
    let (sum_core, sum_content) = split_polynomial_content_for_calculus_presentation(ctx, sum_expr);
    if sum_content.is_zero() {
        return None;
    }

    let wronskian_expr = wronskian.to_expr(ctx);
    let (wronskian_core, wronskian_content) =
        split_polynomial_content_for_calculus_presentation(ctx, wronskian_expr);
    if wronskian_content.is_zero() {
        return None;
    }

    let coefficient =
        derivative_sign * wronskian_content / (BigRational::from_integer(2.into()) * sum_content);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, wronskian_core);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sum_core, den, sqrt_radicand])
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[denominator_scale, sum_core, den, sqrt_radicand],
        )
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let sum_poly = numerator_poly.add(&denominator_poly);

    if !polynomial_is_strictly_positive_everywhere(&numerator_poly)
        || !polynomial_is_strictly_positive_everywhere(&denominator_poly)
        || !polynomial_is_strictly_positive_everywhere(&sum_poly)
    {
        return None;
    }

    let compact = arctan_sqrt_polynomial_quotient_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    )?;
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(crate) fn arctan_sqrt_positive_polynomial_quotient_derivative_for_diff_call(
    ctx: &mut Context,
    source: ExprId,
) -> Option<ExprId> {
    let call = try_extract_diff_call(ctx, source)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(ctx, target, &call.var_name)
}

fn arctan_reciprocal_abs_inverse_sqrt_radicand_arg(
    ctx: &Context,
    target: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }

    let Expr::Div(num, den) = ctx.get(args[0]) else {
        return None;
    };
    let numerator = cas_ast::views::as_rational_const(ctx, *num, 8)?;
    if !numerator.is_one() {
        return None;
    }
    let Expr::Function(abs_fn, abs_args) = ctx.get(*den) else {
        return None;
    };
    if abs_args.len() != 1 || !ctx.is_builtin(*abs_fn, BuiltinFn::Abs) {
        return None;
    }
    let Expr::Pow(base, exp) = ctx.get(abs_args[0]) else {
        return None;
    };
    if cas_ast::views::as_rational_const(ctx, *exp, 8)
        != Some(BigRational::new((-1).into(), 2.into()))
    {
        return None;
    }

    Some(*base)
}

fn arccot_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }

    let radicand = match ctx.get(args[0]) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            sqrt_args[0]
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => *base,
        _ => return None,
    };

    Some(radicand)
}

fn positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(
    ctx: &Context,
    arg: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(arg) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            positive_constant_over_expr_for_calculus_presentation(ctx, sqrt_args[0])
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => {
            positive_constant_over_expr_for_calculus_presentation(ctx, *base)
        }
        Expr::Function(abs_fn, abs_args)
            if abs_args.len() == 1 && ctx.is_builtin(*abs_fn, BuiltinFn::Abs) =>
        {
            positive_constant_over_reciprocal_sqrt_expr_for_calculus_presentation(ctx, abs_args[0])
        }
        _ => None,
    }
}

fn positive_constant_over_reciprocal_sqrt_expr_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            Some((BigRational::one(), *base))
        }
        Expr::Div(num, den) => {
            let numerator = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            if !numerator.is_positive() {
                return None;
            }
            let denominator = calculus_sqrt_like_radicand(ctx, *den)?;
            Some((numerator, denominator))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut denominator = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 2.into()))
                    || denominator.replace(*base).is_some()
                {
                    return None;
                }
            }
            if !scale.is_positive() {
                return None;
            }
            let scale_squared = &scale * &scale;
            Some((scale_squared, denominator?))
        }
        _ => None,
    }
}

fn reciprocal_sqrt_radicand_arg_for_inverse_tangent(
    ctx: &Context,
    target: ExprId,
) -> Option<(ExprId, BigRational, BigRational)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => -BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => BigRational::one(),
        _ => return None,
    };

    let mut abs_wrapped = false;
    let arg = match ctx.get(args[0]) {
        Expr::Function(abs_fn, abs_args)
            if abs_args.len() == 1 && ctx.is_builtin(*abs_fn, BuiltinFn::Abs) =>
        {
            abs_wrapped = true;
            abs_args[0]
        }
        _ => args[0],
    };

    let (radicand, mut argument_scale) = match ctx.get(arg) {
        Expr::Div(num, den) => {
            let scale = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            if scale.is_zero() {
                return None;
            }
            (calculus_sqrt_like_radicand(ctx, *den)?, scale)
        }
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::MulView::from_expr(ctx, args[0]).factors;
            let mut scale = BigRational::one();
            let mut radicand = None;

            for factor in factors {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }

                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 2.into()))
                    || radicand.replace(*base).is_some()
                {
                    return None;
                }
            }

            if scale.is_zero() {
                return None;
            }
            (radicand?, scale)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            (*base, BigRational::one())
        }
        _ => return None,
    };
    if abs_wrapped {
        argument_scale = argument_scale.abs();
    }

    Some((radicand, sign, argument_scale))
}

fn rational_const_for_calculus_presentation(ctx: &mut Context, value: BigRational) -> ExprId {
    if value == BigRational::one() {
        ctx.num(1)
    } else {
        ctx.add(Expr::Number(value))
    }
}

fn nonzero_rational_parts(value: &BigRational) -> Option<(BigRational, BigRational)> {
    if value.is_zero() {
        return None;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    Some((numerator, denominator))
}

fn add_one_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    add_rational_for_calculus_presentation(ctx, expr, BigRational::one())
}

fn subtract_from_one_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let one = ctx.num(1);
    let raw = ctx.add(Expr::Sub(one, expr));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

fn subtract_from_rational_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
    expr: ExprId,
) -> ExprId {
    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Sub(constant, expr));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

fn add_rational_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    if value.is_zero() {
        return expr;
    }

    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Add(expr, constant));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

fn add_rational_combining_additive_constant_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return add_rational_for_calculus_presentation(ctx, expr, value);
    }

    let mut constant = value;
    let mut saw_constant = false;
    let mut rebuilt_terms = Vec::new();
    for (term, sign) in terms {
        if let Some(term_value) = cas_ast::views::as_rational_const(ctx, term, 8) {
            saw_constant = true;
            if sign == cas_math::expr_nary::Sign::Neg {
                constant -= term_value;
            } else {
                constant += term_value;
            }
            continue;
        }

        if sign == cas_math::expr_nary::Sign::Neg {
            rebuilt_terms.push(ctx.add(Expr::Neg(term)));
        } else {
            rebuilt_terms.push(term);
        }
    }

    if !saw_constant {
        return add_rational_for_calculus_presentation(ctx, expr, constant);
    }
    if !constant.is_zero() {
        rebuilt_terms.push(rational_const_for_calculus_presentation(ctx, constant));
    }

    match rebuilt_terms.len() {
        0 => ctx.num(0),
        1 => rebuilt_terms[0],
        _ => cas_math::expr_nary::build_balanced_add(ctx, &rebuilt_terms),
    }
}

fn reciprocal_integer_radicand_content_for_calculus_presentation(
    value: &BigRational,
) -> Option<BigRational> {
    if value.is_positive() && value.numer().is_one() && value.denom() > &BigInt::one() {
        Some(BigRational::from_integer(value.denom().clone()))
    } else {
        None
    }
}

fn positive_constant_over_expr_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            let value = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            value.is_positive().then_some((value, *den))
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            Some((BigRational::one(), *base))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut denominator = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 1.into()))
                    || denominator.replace(*base).is_some()
                {
                    return None;
                }
            }
            scale.is_positive().then_some((scale, denominator?))
        }
        _ => None,
    }
}

fn arctan_sqrt_reciprocal_content_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    radicand_poly: &Polynomial,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand_content = rational_polynomial_content_for_calculus_presentation(radicand_poly);
    let reciprocal_content =
        reciprocal_integer_radicand_content_for_calculus_presentation(&radicand_content)?;
    let primitive_radicand_poly = radicand_poly.div_scalar(&radicand_content);
    let derivative_poly = primitive_radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let primitive_radicand = primitive_radicand_poly.to_expr(ctx);
    let compact_gap =
        add_rational_for_calculus_presentation(ctx, primitive_radicand, reciprocal_content);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, compact_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn arctan_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    sqrt_constant_over_polynomial_presentation(ctx, radicand, var_name, derivative_sign)
}

fn arccot_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arccot_sqrt_radicand_arg(ctx, target)?;
    sqrt_constant_over_polynomial_presentation(ctx, radicand, var_name, -BigRational::one())
}

fn sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };
    let numerator_value = cas_ast::views::as_rational_const(ctx, num, 8)?;
    if !numerator_value.is_positive() {
        return None;
    }

    let denominator_poly = polynomial_radicand_for_calculus_presentation(ctx, den, var_name)?;
    let derivative_poly = denominator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let sqrt_numerator_value =
        exact_positive_rational_sqrt_for_calculus_presentation(&numerator_value);
    let displayed_numerator_value = sqrt_numerator_value
        .clone()
        .unwrap_or_else(|| numerator_value.clone());
    let coefficient = -derivative_sign
        * displayed_numerator_value
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let denominator_plus_numerator =
        add_rational_for_calculus_presentation(ctx, den, numerator_value);
    let core_denominator = if sqrt_numerator_value.is_some() {
        let sqrt_denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[sqrt_denominator, denominator_plus_numerator],
        )
    } else {
        let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[sqrt_radicand, den, denominator_plus_numerator],
        )
    };
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn asinh_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if builtin != BuiltinFn::Asinh {
        return None;
    }
    let (numerator_value, den) =
        positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(ctx, args[0])?;
    let (sqrt_numerator_outside, sqrt_numerator_inside) =
        split_square_factor_positive_rational_for_calculus_presentation(&numerator_value);

    let denominator_poly = polynomial_radicand_for_calculus_presentation(ctx, den, var_name)?;
    let derivative_poly = denominator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), crate::ImplicitCondition::Positive(den)));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient =
        -sqrt_numerator_outside * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let denominator_plus_numerator =
        add_rational_for_calculus_presentation(ctx, den, numerator_value);
    let (numerator_coeff, denominator_coeff, den) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            den,
        );
    let scaled_derivative =
        scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let numerator = if sqrt_numerator_inside.is_one() {
        scaled_derivative
    } else {
        scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            sqrt_numerator_inside,
            scaled_derivative,
        )
    };
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![denominator_plus_numerator]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[den, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        crate::ImplicitCondition::Positive(den),
    ))
}

fn scaled_asinh_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale_factors = Vec::new();
    let mut asinh_target = None;
    for factor in factors {
        if asinh_target.is_none()
            && matches!(
                ctx.get(factor),
                Expr::Function(fn_id, args)
                    if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Asinh)
            )
        {
            asinh_target = Some(factor);
        } else if !cas_math::expr_predicates::contains_named_var(ctx, factor, var_name) {
            scale_factors.push(factor);
        } else {
            return None;
        }
    }

    let asinh_target = asinh_target?;
    if scale_factors.is_empty() {
        return None;
    }

    let (compact, required_condition) =
        asinh_sqrt_constant_over_polynomial_presentation(ctx, asinh_target, var_name)?;
    let compact = unwrap_internal_hold_for_calculus(ctx, compact);
    scale_factors.push(compact);
    let scaled = cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors);

    Some((cas_ast::hold::wrap_hold(ctx, scaled), required_condition))
}

fn atanh_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if builtin != BuiltinFn::Atanh {
        return None;
    }
    let (numerator_value, den) =
        positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(ctx, args[0])?;
    let (displayed_numerator_value, sqrt_numerator_denominator) =
        positive_rational_sqrt_denominator_factor_for_calculus_presentation(ctx, &numerator_value)?;

    let denominator_poly = polynomial_radicand_for_calculus_presentation(ctx, den, var_name)?;
    let derivative_poly = denominator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), crate::ImplicitCondition::Positive(den)));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let orient_gap_to_numerator = numerator_value > denominator_constant;
    let coefficient_sign = if orient_gap_to_numerator {
        BigRational::one()
    } else {
        -BigRational::one()
    };
    let coefficient = coefficient_sign
        * displayed_numerator_value
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let combine_constant_with_sqrt_denominator =
        sqrt_numerator_denominator.is_some() && numerator_value.is_integer();
    let sqrt_denominator = if combine_constant_with_sqrt_denominator {
        let scaled_denominator_poly =
            scale_polynomial_for_calculus_presentation(&denominator_poly, &numerator_value);
        let scaled_denominator = scaled_denominator_poly.to_expr(ctx);
        ctx.call_builtin(BuiltinFn::Sqrt, vec![scaled_denominator])
    } else {
        ctx.call_builtin(BuiltinFn::Sqrt, vec![den])
    };
    let denominator_gap = if orient_gap_to_numerator {
        subtract_from_rational_for_calculus_presentation(ctx, numerator_value, den)
    } else {
        add_rational_for_calculus_presentation(ctx, den, -numerator_value)
    };
    let (numerator_coeff, denominator_coeff, denominator_gap) = if orient_gap_to_numerator {
        cancel_positive_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        )
    } else {
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        )
    };
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator = if combine_constant_with_sqrt_denominator {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_denominator, denominator_gap])
    } else if let Some(sqrt_numerator_denominator) = sqrt_numerator_denominator {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[
                sqrt_numerator_denominator,
                sqrt_denominator,
                denominator_gap,
            ],
        )
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_denominator, denominator_gap])
    };
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        crate::ImplicitCondition::Positive(den),
    ))
}

fn split_polynomial_content_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> (ExprId, BigRational) {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    let Ok(poly) = multipoly_from_expr(ctx, expr, &budget) else {
        return (expr, BigRational::one());
    };
    let (content, primitive) = poly.primitive_part();
    if content.is_zero() || content.is_one() {
        return (expr, BigRational::one());
    }

    (multipoly_to_expr(&primitive, ctx), content)
}

fn polynomial_radicand_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<Polynomial> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() > 4 || poly.coeffs.len() > 8 {
        return None;
    }
    Some(poly)
}

fn polynomial_is_strictly_positive_everywhere(poly: &Polynomial) -> bool {
    if polynomial_has_positive_constant_and_nonnegative_even_terms(poly) {
        return true;
    }
    match poly.degree() {
        0 => poly
            .coeffs
            .first()
            .is_some_and(|constant| constant.is_positive()),
        2 => {
            let a = poly
                .coeffs
                .get(2)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            if !a.is_positive() {
                return false;
            }
            let b = poly
                .coeffs
                .get(1)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            let c = poly
                .coeffs
                .first()
                .cloned()
                .unwrap_or_else(BigRational::zero);
            let four = BigRational::from_integer(4.into());
            b.clone() * b - four * a * c < BigRational::zero()
        }
        _ => false,
    }
}

fn polynomial_has_positive_constant_and_nonnegative_even_terms(poly: &Polynomial) -> bool {
    if !poly
        .coeffs
        .first()
        .is_some_and(|constant| constant.is_positive())
    {
        return false;
    }
    poly.coeffs
        .iter()
        .enumerate()
        .skip(1)
        .all(|(power, coeff)| coeff.is_zero() || (power % 2 == 0 && !coeff.is_negative()))
}

fn polynomial_derivative_expr_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let poly = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name)?;
    Some(poly.derivative().to_expr(ctx))
}

fn affine_square_root_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 2 {
        return None;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let linear_coeff = exact_positive_rational_sqrt_for_calculus_presentation(&a)?;
    let constant_abs = if c.is_zero() {
        BigRational::zero()
    } else {
        exact_positive_rational_sqrt_for_calculus_presentation(&c)?
    };
    let expected_cross =
        BigRational::from_integer(2.into()) * linear_coeff.clone() * constant_abs.clone();
    let constant = if b == expected_cross {
        constant_abs
    } else if b == -expected_cross {
        -constant_abs
    } else {
        return None;
    };

    let affine = Polynomial::new(vec![constant, linear_coeff], var_name.to_string());
    Some(affine.to_expr(ctx))
}

fn compact_squared_affine_gap_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let Expr::Sub(left, right) = ctx.get(expr).clone() else {
        return expr;
    };
    let Expr::Pow(base, exp) = ctx.get(right).clone() else {
        return expr;
    };
    if cas_ast::views::as_rational_const(ctx, exp, 8) != Some(BigRational::from_integer(2.into())) {
        return expr;
    }
    if let Expr::Pow(_, inner_exp) = ctx.get(base).clone() {
        if cas_ast::views::as_rational_const(ctx, inner_exp, 8)
            == Some(BigRational::from_integer(2.into()))
        {
            return expr;
        }
    }

    let Some(affine) = affine_square_root_for_calculus_presentation(ctx, base, var_name) else {
        return expr;
    };
    let four = ctx.num(4);
    let compact_power = ctx.add(Expr::Pow(affine, four));
    ctx.add(Expr::Sub(left, compact_power))
}

fn rational_polynomial_content_for_calculus_presentation(poly: &Polynomial) -> BigRational {
    let mut numer_gcd: Option<BigInt> = None;
    let mut denom_lcm = BigInt::one();

    for coeff in &poly.coeffs {
        if coeff.is_zero() {
            continue;
        }
        let numer = coeff.numer().abs();
        let denom = coeff.denom().clone();
        numer_gcd = Some(match numer_gcd {
            Some(gcd) => gcd.gcd(&numer),
            None => numer,
        });
        denom_lcm = denom_lcm.lcm(&denom);
    }

    match numer_gcd {
        Some(numer_gcd) if !numer_gcd.is_zero() => BigRational::new(numer_gcd, denom_lcm),
        _ => BigRational::zero(),
    }
}

fn scale_polynomial_for_calculus_presentation(
    poly: &Polynomial,
    coeff: &BigRational,
) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|term| term * coeff).collect(),
        poly.var.clone(),
    )
}

fn scale_expr_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff.is_one() {
        return expr;
    }
    let coeff = rational_const_for_calculus_presentation(ctx, coeff);
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if value == BigRational::one() {
            return coeff;
        }
        if let Some(coeff_value) = cas_ast::views::as_rational_const(ctx, coeff, 8) {
            return rational_const_for_calculus_presentation(ctx, coeff_value * value);
        }
    }
    cas_math::expr_nary::build_balanced_mul(ctx, &[coeff, expr])
}

fn compact_integer_affine_inverse_args_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(
                        BuiltinFn::Arcsin
                            | BuiltinFn::Asin
                            | BuiltinFn::Arccos
                            | BuiltinFn::Acos
                            | BuiltinFn::Acosh
                    )
                ) =>
        {
            let arg = args[0];
            let Some(builtin) = ctx.builtin_of(fn_id) else {
                return expr;
            };
            let compact_arg = Polynomial::from_expr(ctx, arg, var_name)
                .ok()
                .filter(|poly| {
                    poly.degree() == 1
                        && (poly.coeffs.iter().all(|c| c.is_integer())
                            || (builtin == BuiltinFn::Acosh
                                && poly
                                    .coeffs
                                    .first()
                                    .is_some_and(|constant| constant.is_one())))
                })
                .map(|poly| poly.to_expr(ctx))
                .unwrap_or(arg);
            ctx.add(Expr::Function(fn_id, vec![compact_arg]))
        }
        Expr::Neg(inner) => {
            let inner = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, inner, var_name,
            );
            ctx.add(Expr::Neg(inner))
        }
        Expr::Add(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Div(left, right))
        }
        _ => expr,
    }
}

fn cancel_denominator_content_with_numerator_for_calculus_presentation(
    ctx: &mut Context,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
    denominator_factor: ExprId,
) -> (BigRational, BigRational, ExprId) {
    let (primitive_factor, factor_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_factor);
    if factor_content.is_zero() || factor_content.is_one() {
        return (numerator_coeff, denominator_coeff, denominator_factor);
    }

    let adjusted_numerator = numerator_coeff.clone() / factor_content.clone();
    if !adjusted_numerator.is_integer() {
        let adjusted_denominator = denominator_coeff * factor_content;
        let (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / adjusted_denominator))
                .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
        return (numerator_coeff, denominator_coeff, primitive_factor);
    }

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(adjusted_numerator / denominator_coeff))
            .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    (numerator_coeff, denominator_coeff, primitive_factor)
}

fn cancel_positive_denominator_content_with_numerator_for_calculus_presentation(
    ctx: &mut Context,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
    denominator_factor: ExprId,
) -> (BigRational, BigRational, ExprId) {
    let (primitive_factor, factor_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_factor);
    if factor_content.is_zero() || !factor_content.is_positive() || factor_content.is_one() {
        return (numerator_coeff, denominator_coeff, denominator_factor);
    }

    let adjusted_numerator = numerator_coeff.clone() / factor_content.clone();
    if !adjusted_numerator.is_integer() {
        let adjusted_denominator = denominator_coeff * factor_content;
        let (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / adjusted_denominator))
                .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
        return (numerator_coeff, denominator_coeff, primitive_factor);
    }

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(adjusted_numerator / denominator_coeff))
            .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    (numerator_coeff, denominator_coeff, primitive_factor)
}

fn positive_rational_sqrt_denominator_factor_for_calculus_presentation(
    ctx: &mut Context,
    value: &BigRational,
) -> Option<(BigRational, Option<ExprId>)> {
    if let Some(root) = exact_positive_rational_sqrt_for_calculus_presentation(value) {
        return Some((root, None));
    }

    let radicand = rational_const_for_calculus_presentation(ctx, value.clone());
    let sqrt_factor = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some((value.clone(), Some(sqrt_factor)))
}

fn signed_numerator_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff == -BigRational::one()
        && !cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value.is_one())
    {
        return ctx.add(Expr::Neg(expr));
    }
    scale_expr_for_calculus_presentation(ctx, coeff, expr)
}

fn signed_rational_const_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return Some(value);
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return cas_ast::views::as_rational_const(ctx, *inner, 8).map(|value| -value);
    }
    None
}

fn exact_positive_rational_sqrt_for_calculus_presentation(
    value: &BigRational,
) -> Option<BigRational> {
    if !value.is_positive() {
        return None;
    }

    let numer_sqrt = value.numer().sqrt();
    let denom_sqrt = value.denom().sqrt();
    if &numer_sqrt * &numer_sqrt == *value.numer() && &denom_sqrt * &denom_sqrt == *value.denom() {
        Some(BigRational::new(numer_sqrt, denom_sqrt))
    } else {
        None
    }
}

fn split_square_factor_positive_bigint_for_calculus_presentation(
    value: &BigInt,
) -> (BigInt, BigInt) {
    let mut outside = BigInt::one();
    let mut inside = value.clone();

    for factor in 2_i64..=97 {
        let factor = BigInt::from(factor);
        let square = &factor * &factor;
        while (&inside % &square).is_zero() {
            inside /= &square;
            outside *= &factor;
        }
    }

    let root = inside.sqrt();
    if root > BigInt::one() && &root * &root == inside {
        outside *= root;
        inside = BigInt::one();
    }

    (outside, inside)
}

fn split_square_factor_positive_rational_for_calculus_presentation(
    value: &BigRational,
) -> (BigRational, BigRational) {
    debug_assert!(value.is_positive());

    let (numer_outside, numer_inside) =
        split_square_factor_positive_bigint_for_calculus_presentation(value.numer());
    let (denom_outside, denom_inside) =
        split_square_factor_positive_bigint_for_calculus_presentation(value.denom());

    (
        BigRational::new(numer_outside, denom_outside),
        BigRational::new(numer_inside, denom_inside),
    )
}

fn sqrt_positive_rational_expr_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
) -> ExprId {
    if let Some(sqrt_value) = exact_positive_rational_sqrt_for_calculus_presentation(&value) {
        return rational_const_for_calculus_presentation(ctx, sqrt_value);
    }

    let (outside, inside) = split_square_factor_positive_rational_for_calculus_presentation(&value);
    if inside.is_one() {
        return rational_const_for_calculus_presentation(ctx, outside);
    }

    let radicand = rational_const_for_calculus_presentation(ctx, inside);
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    scale_expr_for_calculus_presentation(ctx, outside, sqrt)
}

fn scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
    expr: ExprId,
) -> ExprId {
    if value.is_one() {
        return expr;
    }

    let sqrt_scale = sqrt_positive_rational_expr_for_calculus_presentation(ctx, value);
    if let Some(expr_value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if expr_value.is_one() {
            return sqrt_scale;
        }
        if expr_value == -BigRational::one() {
            return ctx.add(Expr::Neg(sqrt_scale));
        }
        return scale_expr_for_calculus_presentation(ctx, expr_value, sqrt_scale);
    }
    cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_scale, expr])
}

fn arctan_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if let Some(compact) = arctan_sqrt_reciprocal_content_presentation(
        ctx,
        radicand,
        &radicand_poly,
        derivative_sign.clone(),
    ) {
        return Some(compact);
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn arctan_sqrt_additive_tan_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_tan_polynomial_derivative_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

fn arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

fn arctan_sqrt_additive_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let (sqrt_derivative, required_positive, required_conditions) =
        sqrt_additive_trig_polynomial_derivative_presentation(ctx, sqrt_radicand, var_name)?;
    if required_positive != radicand {
        return None;
    }

    let sqrt_derivative = unwrap_internal_hold_for_calculus(ctx, sqrt_derivative);
    let radicand_plus_one = add_rational_combining_additive_constant_for_calculus_presentation(
        ctx,
        radicand,
        BigRational::one(),
    );
    let result = match ctx.get(sqrt_derivative).clone() {
        Expr::Div(numerator, denominator) => {
            let denominator =
                cas_math::expr_nary::build_balanced_mul(ctx, &[denominator, radicand_plus_one]);
            ctx.add_raw(Expr::Div(numerator, denominator))
        }
        _ => ctx.add_raw(Expr::Div(sqrt_derivative, radicand_plus_one)),
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, result),
        radicand,
        required_conditions,
    ))
}

fn arctan_sqrt_small_additive_elementary_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let (radicand_derivative, derivative_denominator, required_conditions) =
        small_additive_elementary_radicand_derivative_for_calculus_presentation(
            ctx, radicand, var_name,
        )?;

    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let mut denominator_factors = vec![two];
    if let Some(derivative_denominator) = derivative_denominator {
        denominator_factors.push(derivative_denominator);
    }
    denominator_factors.push(sqrt_radicand);
    denominator_factors.push(radicand_plus_one);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let denominator = compact_numeric_mul_factors_for_calculus_presentation(ctx, denominator);
    let compact = ctx.add_raw(Expr::Div(radicand_derivative, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

fn sqrt_small_additive_elementary_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (radicand_derivative, derivative_denominator, required_conditions) =
        small_additive_elementary_radicand_derivative_for_calculus_presentation(
            ctx, radicand, var_name,
        )?;

    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let mut denominator_factors = vec![two];
    if let Some(derivative_denominator) = derivative_denominator {
        denominator_factors.push(derivative_denominator);
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let denominator = compact_numeric_mul_factors_for_calculus_presentation(ctx, denominator);
    let compact = ctx.add_raw(Expr::Div(radicand_derivative, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

fn small_additive_elementary_radicand_derivative_for_calculus_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<(ExprId, Option<ExprId>, Vec<crate::ImplicitCondition>)> {
    small_additive_elementary_common_derivative_for_calculus_presentation(ctx, radicand, var_name)
}

fn small_additive_elementary_common_derivative_for_calculus_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<(ExprId, Option<ExprId>, Vec<crate::ImplicitCondition>)> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 4 {
        return None;
    }

    let var = ctx.var(var_name);
    let two = ctx.num(2);
    let sqrt_var = sqrt_raw_for_calculus_presentation(ctx, var);
    let common_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, var, sqrt_var]);
    let common_denominator =
        compact_numeric_mul_factors_for_calculus_presentation(ctx, common_denominator);
    let common_scale = cas_math::expr_nary::build_balanced_mul(ctx, &[two, var, sqrt_var]);

    let mut saw_ln_var = false;
    let mut saw_sqrt_var = false;
    let mut numerator_terms = Vec::new();
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };

        if let Some((scale, arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if arg != var {
                return None;
            }
            saw_ln_var = true;
            required_conditions.push(crate::ImplicitCondition::Positive(arg));
            let two_sqrt = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_var]);
            numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, two_sqrt));
            continue;
        }

        if let Some((scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if sqrt_arg != var {
                return None;
            }
            saw_sqrt_var = true;
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, var));
            continue;
        }

        if let Some(derivative) = scaled_exp_trig_variable_term_derivative_for_calculus_presentation(
            ctx,
            signed_term,
            var_name,
        ) {
            let term = ctx.add_raw(Expr::Mul(common_scale, derivative));
            numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
                ctx, term,
            ));
            continue;
        }

        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }

        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            let derivative = derivative.to_expr(ctx);
            let term = ctx.add_raw(Expr::Mul(common_scale, derivative));
            numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
                ctx, term,
            ));
        }
    }

    if !(saw_ln_var && saw_sqrt_var) || numerator_terms.is_empty() {
        return None;
    }
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    Some((numerator, Some(common_denominator), required_conditions))
}

fn scaled_exp_trig_variable_term_derivative_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, term)
        .unwrap_or((BigRational::one(), term));
    let exp_arg = match ctx.get(core).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Exp) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => exp,
        _ => return None,
    };

    let (trig_derivative, sign) = match ctx.get(exp_arg).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sin) =>
        {
            let var = unary_variable_builtin_arg_for_calculus_presentation(
                ctx,
                exp_arg,
                var_name,
                BuiltinFn::Sin,
            )?;
            (
                ctx.call_builtin(BuiltinFn::Cos, vec![var]),
                BigRational::one(),
            )
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Cos) =>
        {
            let var = unary_variable_builtin_arg_for_calculus_presentation(
                ctx,
                exp_arg,
                var_name,
                BuiltinFn::Cos,
            )?;
            (
                ctx.call_builtin(BuiltinFn::Sin, vec![var]),
                -BigRational::one(),
            )
        }
        _ => return None,
    };

    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[trig_derivative, core]);
    Some(scale_expr_for_calculus_presentation(
        ctx,
        scale * sign,
        product,
    ))
}

pub(crate) fn arctan_sqrt_additive_tan_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_additive_trig_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn arctan_sqrt_small_additive_elementary_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

pub(crate) fn sqrt_small_additive_elementary_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive, mut required_conditions) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    required_conditions.insert(0, crate::ImplicitCondition::Positive(required_positive));
    Some((
        unwrap_internal_hold_for_calculus(ctx, result),
        required_conditions,
    ))
}

fn constant_scaled_arctan_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative =
        arctan_sqrt_polynomial_derivative_presentation(ctx, inner, var_name, BigRational::one())
            .or_else(|| arccot_sqrt_polynomial_derivative_presentation(ctx, inner, var_name))?;
    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_reciprocal_abs_inverse_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let synthetic_arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![sqrt_radicand]);
    let result = arctan_sqrt_polynomial_derivative_presentation(
        ctx,
        synthetic_arctan,
        var_name,
        BigRational::one(),
    )?;
    let required_conditions = if polynomial_is_strictly_positive_everywhere(&radicand_poly) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(radicand)]
    };
    Some((result, required_conditions))
}

fn arccot_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arccot_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if let Some(compact) = arctan_sqrt_reciprocal_content_presentation(
        ctx,
        radicand,
        &radicand_poly,
        -BigRational::one(),
    ) {
        return Some(compact);
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = -derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn negative_arccot_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }

    let (radicand, argument_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, args[0])?;
    if argument_scale != -BigRational::one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let (numerator_coeff, denominator_coeff, radicand_plus_one) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_plus_one,
        );
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let required_conditions = if polynomial_is_strictly_positive_everywhere(&radicand_poly) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(radicand)]
    };
    Some((
        ctx.add(Expr::Div(numerator, denominator)),
        required_conditions,
    ))
}

fn inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (radicand, derivative_sign, argument_scale) =
        reciprocal_sqrt_radicand_arg_for_inverse_tangent(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let argument_scale_square = argument_scale.clone() * argument_scale.clone();
    let radicand_content = rational_polynomial_content_for_calculus_presentation(&radicand_poly);
    if radicand_content.is_positive() && !radicand_content.is_one() {
        let primitive_radicand_poly = radicand_poly.div_scalar(&radicand_content);
        let derivative_poly = primitive_radicand_poly.derivative();
        if derivative_poly.is_zero() {
            return Some(ctx.num(0));
        }
        let derivative = derivative_poly.to_expr(ctx);

        let (derivative_core, derivative_content) =
            split_polynomial_content_for_calculus_presentation(ctx, derivative);
        let coefficient = derivative_sign
            * argument_scale.clone()
            * derivative_content
            * BigRational::new(1.into(), 2.into());
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        let numerator =
            signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

        let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        let primitive_radicand = primitive_radicand_poly.to_expr(ctx);
        let radicand_plus_one = add_rational_for_calculus_presentation(
            ctx,
            primitive_radicand,
            argument_scale_square / radicand_content,
        );
        let core_denominator =
            cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
        let denominator = if denominator_coeff == BigRational::one() {
            core_denominator
        } else {
            let denominator_scale =
                rational_const_for_calculus_presentation(ctx, denominator_coeff);
            cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
        };

        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * argument_scale
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one =
        add_rational_for_calculus_presentation(ctx, radicand, argument_scale_square);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (radicand, _, _) = reciprocal_sqrt_radicand_arg_for_inverse_tangent(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result =
        inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation(ctx, target, var_name)?;
    let required_conditions = if polynomial_is_strictly_positive_everywhere(&radicand_poly) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(radicand)]
    };
    Some((result, required_conditions))
}

fn sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn sqrt_bounded_trig_positive_shift_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, radicand)?;
    let presentation_radicand =
        compact_double_angle_sine_products_for_calculus_presentation(ctx, radicand)
            .filter(|candidate| {
                bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, *candidate).is_some()
            })
            .unwrap_or(radicand);

    let derivative = differentiate(ctx, presentation_radicand, var_name)?;
    let derivative = compact_small_power_exponents_for_calculus_presentation(ctx, derivative);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let distributed_numerator = if coefficient == BigRational::new(1.into(), 2.into()) {
        distribute_half_over_additive_numerator_for_calculus_presentation(ctx, derivative_core)
    } else {
        None
    };
    let (numerator, denominator_coeff) = if let Some(numerator) = distributed_numerator {
        (numerator, BigRational::one())
    } else {
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        (
            scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core),
            denominator_coeff,
        )
    };
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![presentation_radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(crate) fn sqrt_additive_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let presentation_radicand =
        compact_double_angle_sine_products_for_calculus_presentation(ctx, radicand)
            .or_else(|| signed_add_terms_for_calculus_presentation(ctx, radicand))
            .unwrap_or(radicand);
    let derivative_parts = additive_trig_polynomial_sqrt_radicand_derivative_for_presentation(
        ctx,
        presentation_radicand,
        var_name,
    )?;
    let required_conditions = derivative_parts.required_conditions;
    if let Some(derivative_denominator) = derivative_parts.denominator {
        let numerator =
            compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative_parts.numerator);
        let two =
            rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
        let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, presentation_radicand);
        let denominator = cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[two, derivative_denominator, sqrt_radicand],
        );
        let compact = ctx.add_raw(Expr::Div(numerator, denominator));
        return Some((
            cas_ast::hold::wrap_hold(ctx, compact),
            radicand,
            required_conditions,
        ));
    }

    let derivative = derivative_parts.numerator;
    let derivative = compact_small_power_exponents_for_calculus_presentation(ctx, derivative);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), radicand, required_conditions));
    }

    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let distributed_numerator = if coefficient == BigRational::new(1.into(), 2.into()) {
        distribute_half_over_additive_numerator_for_calculus_presentation(ctx, derivative_core)
    } else {
        None
    };
    let (numerator, denominator_coeff) = if let Some(numerator) = distributed_numerator {
        (numerator, BigRational::one())
    } else {
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        (
            scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core),
            denominator_coeff,
        )
    };
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, presentation_radicand);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

pub(crate) fn sqrt_additive_tan_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut tan_scale = BigRational::zero();
    let mut tan_arg = None;
    let mut common_trig_denominator_builtin = None;
    let mut has_variable_dependency = false;
    let mut common_denominator = None;
    let mut sqrt_variable_derivative = None;
    let mut reciprocal_sqrt_variable_derivative = None;
    let mut reciprocal_derivative_scales = Vec::new();
    let mut other_derivatives = Vec::new();
    let mut has_reciprocal_trig_term = false;
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if let Some((scale, arg, denominator_builtin)) =
            scaled_tan_or_cot_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if tan_arg.is_some_and(|existing| existing != arg) {
                return None;
            }
            if common_trig_denominator_builtin
                .is_some_and(|existing| existing != denominator_builtin)
            {
                return None;
            }
            tan_arg = Some(arg);
            common_trig_denominator_builtin = Some(denominator_builtin);
            tan_scale += scale;
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                other_derivatives.push(derivative);
            }
            continue;
        }
        if let Some((derivative, required_condition)) =
            scaled_sec_or_csc_variable_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            has_reciprocal_trig_term = true;
            other_derivatives.push(derivative);
            required_conditions.push(required_condition);
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if common_denominator.is_some_and(|existing| existing != ln_arg) {
                return None;
            }
            common_denominator = Some(ln_arg);
            reciprocal_derivative_scales.push(ln_scale);
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if let Some((existing_scale, existing_arg)) = &mut sqrt_variable_derivative {
                if *existing_arg != sqrt_arg {
                    return None;
                }
                *existing_scale += sqrt_scale;
            } else {
                sqrt_variable_derivative = Some((sqrt_scale, sqrt_arg));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            match reciprocal_sqrt_variable_derivative.take() {
                Some((mut existing_scale, existing_arg)) if existing_arg == reciprocal_sqrt_arg => {
                    existing_scale += reciprocal_sqrt_scale;
                    reciprocal_sqrt_variable_derivative = Some((existing_scale, existing_arg));
                }
                Some((previous_scale, previous_arg)) => {
                    other_derivatives.push(
                        reciprocal_sqrt_derivative_term_for_calculus_presentation(
                            ctx,
                            previous_scale,
                            previous_arg,
                        ),
                    );
                    other_derivatives.push(
                        reciprocal_sqrt_derivative_term_for_calculus_presentation(
                            ctx,
                            reciprocal_sqrt_scale,
                            reciprocal_sqrt_arg,
                        ),
                    );
                }
                None => {
                    reciprocal_sqrt_variable_derivative =
                        Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg));
                }
            }
            required_conditions.push(crate::ImplicitCondition::Positive(reciprocal_sqrt_arg));
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            other_derivatives.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some(exp_chain_derivative) =
            scaled_exp_bounded_chain_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            other_derivatives.push(exp_chain_derivative);
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            other_derivatives.push(derivative.to_expr(ctx));
        }
    }

    if !has_variable_dependency {
        return None;
    }
    if tan_scale.is_zero() {
        let has_common_denominator_sqrt_and_reciprocal_sqrt_route = common_denominator.is_some()
            && sqrt_variable_derivative.is_some()
            && reciprocal_sqrt_variable_derivative.is_some()
            && !reciprocal_derivative_scales.is_empty()
            && !other_derivatives.is_empty();
        if !has_reciprocal_trig_term && !has_common_denominator_sqrt_and_reciprocal_sqrt_route {
            return None;
        }
        if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
            if let Some(common_denominator) = common_denominator {
                if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
                    reciprocal_sqrt_variable_derivative
                {
                    if reciprocal_sqrt_arg == sqrt_arg
                        && sqrt_arg == common_denominator
                        && !sqrt_scale.is_zero()
                        && !reciprocal_sqrt_scale.is_zero()
                        && !reciprocal_derivative_scales.is_empty()
                        && !other_derivatives.is_empty()
                    {
                        let result =
                            sqrt_additive_generic_common_denominator_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                                ctx,
                                radicand,
                                common_denominator,
                                sqrt_scale,
                                reciprocal_sqrt_scale,
                                reciprocal_derivative_scales,
                                other_derivatives,
                            )?;
                        return Some((result, radicand, required_conditions));
                    }
                    return None;
                } else if !reciprocal_derivative_scales.is_empty() && !other_derivatives.is_empty()
                {
                    let result =
                        sqrt_additive_generic_common_denominator_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            common_denominator,
                            sqrt_arg,
                            sqrt_scale,
                            reciprocal_derivative_scales,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                return None;
            }
            if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
                reciprocal_sqrt_variable_derivative
            {
                if reciprocal_sqrt_arg == sqrt_arg
                    && !sqrt_scale.is_zero()
                    && !reciprocal_sqrt_scale.is_zero()
                {
                    let result =
                        sqrt_additive_generic_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            sqrt_arg,
                            sqrt_scale,
                            reciprocal_sqrt_scale,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
            if has_reciprocal_trig_term {
                let mut derivative_terms = other_derivatives;
                derivative_terms.push(sqrt_variable_derivative_term_for_calculus_presentation(
                    ctx, sqrt_scale, sqrt_arg,
                )?);
                let result =
                    sqrt_additive_generic_derivative_presentation(ctx, radicand, derivative_terms)?;
                return Some((result, radicand, required_conditions));
            }
            let result = sqrt_additive_generic_sqrt_variable_derivative_presentation(
                ctx,
                radicand,
                sqrt_arg,
                sqrt_scale,
                other_derivatives,
            )?;
            return Some((result, radicand, required_conditions));
        }

        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            reciprocal_sqrt_variable_derivative
        {
            if common_denominator.is_none() && !reciprocal_sqrt_scale.is_zero() {
                let result =
                    sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation(
                        ctx,
                        radicand,
                        reciprocal_sqrt_arg,
                        reciprocal_sqrt_scale,
                        other_derivatives,
                    )?;
                return Some((result, radicand, required_conditions));
            }
            if let Some(common_denominator) = common_denominator {
                if reciprocal_sqrt_arg == common_denominator
                    && !reciprocal_sqrt_scale.is_zero()
                    && !reciprocal_derivative_scales.is_empty()
                    && !other_derivatives.is_empty()
                {
                    let result =
                        sqrt_additive_generic_common_denominator_reciprocal_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            common_denominator,
                            reciprocal_sqrt_scale,
                            reciprocal_derivative_scales,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
        }

        if common_denominator.is_none()
            && reciprocal_derivative_scales.is_empty()
            && !other_derivatives.is_empty()
        {
            let result =
                sqrt_additive_generic_derivative_presentation(ctx, radicand, other_derivatives)?;
            return Some((result, radicand, required_conditions));
        }
        if let Some(common_denominator) = common_denominator {
            if !reciprocal_derivative_scales.is_empty() && !other_derivatives.is_empty() {
                let result = sqrt_additive_generic_common_denominator_derivative_presentation(
                    ctx,
                    radicand,
                    common_denominator,
                    reciprocal_derivative_scales,
                    other_derivatives,
                )?;
                return Some((result, radicand, required_conditions));
            }
        }

        return None;
    }
    let tan_arg = tan_arg?;
    let common_trig_denominator_builtin = common_trig_denominator_builtin?;
    let reciprocal_trig_builtin = match common_trig_denominator_builtin {
        BuiltinFn::Cos => BuiltinFn::Sec,
        BuiltinFn::Sin => BuiltinFn::Csc,
        _ => return None,
    };
    let cos_arg = ctx.call_builtin(common_trig_denominator_builtin, vec![tan_arg]);
    let two = ctx.num(2);
    let cos_square = ctx.add_raw(Expr::Pow(cos_arg, two));

    if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
        if common_denominator.is_some() {
            return None;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            reciprocal_sqrt_variable_derivative
        {
            if reciprocal_sqrt_arg == sqrt_arg
                && !sqrt_scale.is_zero()
                && !reciprocal_sqrt_scale.is_zero()
            {
                let result =
                    sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                        ctx,
                        SqrtAdditiveTanDerivativePresentationParts {
                            radicand,
                            tan_arg,
                            reciprocal_trig_builtin,
                            tan_scale: tan_scale.clone(),
                            other_derivatives,
                        },
                        sqrt_arg,
                        sqrt_scale,
                        reciprocal_sqrt_scale,
                    )?;
                required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
                return Some((result, radicand, required_conditions));
            }
            other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                ctx,
                reciprocal_sqrt_scale,
                reciprocal_sqrt_arg,
            ));
        }
        let result = sqrt_additive_tan_sqrt_variable_derivative_presentation(
            ctx,
            SqrtAdditiveTanDerivativePresentationParts {
                radicand,
                tan_arg,
                reciprocal_trig_builtin,
                tan_scale: tan_scale.clone(),
                other_derivatives,
            },
            sqrt_arg,
            sqrt_scale,
        )?;
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        return Some((result, radicand, required_conditions));
    }

    if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) = reciprocal_sqrt_variable_derivative
    {
        if common_denominator.is_none() && !reciprocal_sqrt_scale.is_zero() {
            let result = sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation(
                ctx,
                SqrtAdditiveTanDerivativePresentationParts {
                    radicand,
                    tan_arg,
                    reciprocal_trig_builtin,
                    tan_scale: tan_scale.clone(),
                    other_derivatives,
                },
                reciprocal_sqrt_arg,
                reciprocal_sqrt_scale,
            )?;
            required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
            return Some((result, radicand, required_conditions));
        }
        other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
            ctx,
            reciprocal_sqrt_scale,
            reciprocal_sqrt_arg,
        ));
    }

    if common_denominator.is_none() && reciprocal_derivative_scales.is_empty() {
        let reciprocal_trig_arg = ctx.call_builtin(reciprocal_trig_builtin, vec![tan_arg]);
        let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
        let tan_derivative =
            scale_expr_for_calculus_presentation(ctx, tan_scale.clone(), reciprocal_trig_square);
        other_derivatives.insert(0, tan_derivative);
        let result =
            sqrt_additive_generic_derivative_presentation(ctx, radicand, other_derivatives)?;
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        return Some((result, radicand, required_conditions));
    }

    let mut numerator_terms = Vec::new();
    let common_denominator =
        common_denominator.filter(|_| !reciprocal_derivative_scales.is_empty());
    let tan_numerator = if let Some(denominator) = common_denominator {
        scale_expr_for_calculus_presentation(ctx, tan_scale, denominator)
    } else {
        rational_const_for_calculus_presentation(ctx, tan_scale)
    };
    numerator_terms.push(tan_numerator);
    for scale in reciprocal_derivative_scales {
        numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, cos_square));
    }
    for derivative in other_derivatives {
        let mut term = compact_tan_sqrt_common_denominator_numerator_term(
            ctx, cos_arg, cos_square, derivative,
        );
        if let Some(denominator) = common_denominator {
            term = ctx.add_raw(Expr::Mul(denominator, term));
            term = compact_numeric_mul_factors_for_calculus_presentation(ctx, term);
        }
        numerator_terms.push(term);
    }
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let denominator_scale =
        rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if let Some(common_denominator) = common_denominator {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[
                denominator_scale,
                common_denominator,
                cos_square,
                sqrt_radicand,
            ],
        )
    } else {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[denominator_scale, cos_square, sqrt_radicand],
        )
    };
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some((cas_ast::hold::wrap_hold(ctx, compact), radicand, {
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        required_conditions
    }))
}

pub(crate) fn sqrt_additive_tan_polynomial_derivative_inline_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut tan_scale = BigRational::zero();
    let mut tan_arg = None;
    let mut common_trig_denominator_builtin = None;
    let mut sqrt_variable_derivative = None;
    let mut other_derivatives = Vec::new();
    let mut has_variable_dependency = false;
    let mut required_conditions = Vec::new();

    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if let Some((scale, arg, denominator_builtin)) =
            scaled_tan_or_cot_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if tan_arg.is_some_and(|existing| existing != arg) {
                return None;
            }
            if common_trig_denominator_builtin
                .is_some_and(|existing| existing != denominator_builtin)
            {
                return None;
            }
            tan_arg = Some(arg);
            common_trig_denominator_builtin = Some(denominator_builtin);
            tan_scale += scale;
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if let Some((existing_scale, existing_arg)) = &mut sqrt_variable_derivative {
                if *existing_arg != sqrt_arg {
                    return None;
                }
                *existing_scale += sqrt_scale;
            } else {
                sqrt_variable_derivative = Some((sqrt_scale, sqrt_arg));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                other_derivatives.push(derivative);
            }
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            other_derivatives.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some(exp_chain_derivative) =
            scaled_exp_bounded_chain_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            other_derivatives.push(exp_chain_derivative);
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            other_derivatives.push(derivative.to_expr(ctx));
        }
    }

    if !has_variable_dependency || tan_scale.is_zero() {
        return None;
    }
    let (sqrt_scale, sqrt_arg) = sqrt_variable_derivative?;
    if sqrt_scale.is_zero() {
        return None;
    }
    let tan_arg = tan_arg?;
    let common_trig_denominator_builtin = common_trig_denominator_builtin?;
    let reciprocal_trig_builtin = match common_trig_denominator_builtin {
        BuiltinFn::Cos => BuiltinFn::Sec,
        BuiltinFn::Sin => BuiltinFn::Csc,
        _ => return None,
    };

    let cos_arg = ctx.call_builtin(common_trig_denominator_builtin, vec![tan_arg]);
    let two = ctx.num(2);
    let reciprocal_trig_arg = ctx.call_builtin(reciprocal_trig_builtin, vec![tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let mut derivative_terms = Vec::new();
    derivative_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        tan_scale,
        reciprocal_trig_square,
    ));
    derivative_terms.push(sqrt_variable_derivative_term_for_calculus_presentation(
        ctx, sqrt_scale, sqrt_arg,
    )?);
    derivative_terms.extend(other_derivatives);

    let result = sqrt_additive_generic_derivative_presentation(ctx, radicand, derivative_terms)?;
    required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
    Some((result, radicand, required_conditions))
}

fn sqrt_variable_derivative_term_for_calculus_presentation(
    ctx: &mut Context,
    sqrt_scale: BigRational,
    sqrt_arg: ExprId,
) -> Option<ExprId> {
    let coefficient = sqrt_scale * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_arg_root
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_arg_root])
    };
    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

fn reciprocal_sqrt_derivative_term_for_calculus_presentation(
    ctx: &mut Context,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_sqrt_arg: ExprId,
) -> ExprId {
    let neg_three_half = ctx.rational(-3, 2);
    let reciprocal_sqrt_cubed = ctx.add_raw(Expr::Pow(reciprocal_sqrt_arg, neg_three_half));
    scale_expr_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale * BigRational::new(1.into(), 2.into()),
        reciprocal_sqrt_cubed,
    )
}

fn sqrt_additive_generic_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if derivative_terms.is_empty() {
        return None;
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &derivative_terms);
    let numerator = compact_small_power_exponents_for_calculus_presentation(ctx, numerator);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_derivative_scales.is_empty() || derivative_terms.is_empty() {
        return None;
    }

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(common_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        numerator_terms.push(rational_const_for_calculus_presentation(ctx, scale));
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, common_denominator, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, common_denominator, sqrt_arg_root]);
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_arg);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        sqrt_scale,
        common_denominator,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[four, common_denominator, sqrt_arg_root, sqrt_radicand],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_common_denominator = sqrt_raw_for_calculus_presentation(ctx, common_denominator);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[two, common_denominator, sqrt_common_denominator],
    );
    let two_sqrt_denominator = ctx.add_raw(Expr::Mul(two, sqrt_common_denominator));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_denominator);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[
            four,
            common_denominator,
            sqrt_common_denominator,
            sqrt_radicand,
        ],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_common_denominator_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero()
        || reciprocal_sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_common_denominator = sqrt_raw_for_calculus_presentation(ctx, common_denominator);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[two, common_denominator, sqrt_common_denominator],
    );
    let two_sqrt_denominator = ctx.add_raw(Expr::Mul(two, sqrt_common_denominator));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_denominator);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        sqrt_scale,
        common_denominator,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[
            four,
            common_denominator,
            sqrt_common_denominator,
            sqrt_radicand,
        ],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(ctx, sqrt_scale));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, sqrt_arg_root, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || reciprocal_sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx, sqrt_scale, sqrt_arg,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    reciprocal_sqrt_arg: ExprId,
    reciprocal_sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, reciprocal_sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[reciprocal_sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, reciprocal_sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn compact_tan_sqrt_common_denominator_numerator_term(
    ctx: &mut Context,
    cos_arg: ExprId,
    cos_square: ExprId,
    derivative: ExprId,
) -> ExprId {
    let (scale, core) =
        split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, derivative)
            .unwrap_or((BigRational::one(), derivative));
    if cas_ast::ordering::compare_expr(ctx, core, cos_arg).is_eq() {
        let three = ctx.num(3);
        let cos_cube = ctx.add_raw(Expr::Pow(cos_arg, three));
        return scale_expr_for_calculus_presentation(ctx, scale, cos_cube);
    }

    let term = ctx.add_raw(Expr::Mul(cos_square, derivative));
    let term = compact_small_power_exponents_for_calculus_presentation(ctx, term);
    let term =
        combine_matching_cos_powers_for_calculus_presentation(ctx, cos_arg, term).unwrap_or(term);
    compact_numeric_mul_factors_for_calculus_presentation(ctx, term)
}

struct SqrtAdditiveTanDerivativePresentationParts {
    radicand: ExprId,
    tan_arg: ExprId,
    reciprocal_trig_builtin: BuiltinFn,
    tan_scale: BigRational,
    other_derivatives: Vec<ExprId>,
}

fn sqrt_additive_tan_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    let tan_term =
        scale_expr_for_calculus_presentation(ctx, parts.tan_scale, reciprocal_trig_square);
    let tan_term = ctx.add_raw(Expr::Mul(two_sqrt_arg, tan_term));
    numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
        ctx, tan_term,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(ctx, sqrt_scale));
    for derivative in parts.other_derivatives {
        let term = ctx.add_raw(Expr::Mul(two_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, sqrt_arg_root, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || reciprocal_sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    let tan_term =
        scale_expr_for_calculus_presentation(ctx, parts.tan_scale, reciprocal_trig_square);
    let tan_term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, tan_term));
    numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
        ctx, tan_term,
    ));
    for derivative in parts.other_derivatives {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx, sqrt_scale, sqrt_arg,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    reciprocal_sqrt_arg: ExprId,
    reciprocal_sqrt_scale: BigRational,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, reciprocal_sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[reciprocal_sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, reciprocal_sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    let tan_term =
        scale_expr_for_calculus_presentation(ctx, parts.tan_scale, reciprocal_trig_square);
    let tan_term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, tan_term));
    numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
        ctx, tan_term,
    ));
    for derivative in parts.other_derivatives {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn combine_matching_cos_powers_for_calculus_presentation(
    ctx: &mut Context,
    cos_arg: ExprId,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let combined = combine_matching_cos_powers_for_calculus_presentation(ctx, cos_arg, inner)?;
        return Some(ctx.add_raw(Expr::Neg(combined)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut cos_power_sum: i64 = 0;
    let mut non_cos_factors = Vec::new();
    for factor in factors {
        if let Some(power) =
            matching_cos_integer_power_for_calculus_presentation(ctx, cos_arg, factor)
        {
            cos_power_sum += power;
        } else {
            non_cos_factors.push(factor);
        }
    }

    if cos_power_sum <= 1 {
        return None;
    }

    let cos_power = if cos_power_sum == 1 {
        cos_arg
    } else {
        let exponent = ctx.num(cos_power_sum);
        ctx.add_raw(Expr::Pow(cos_arg, exponent))
    };
    non_cos_factors.push(cos_power);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &non_cos_factors,
    ))
}

fn matching_cos_integer_power_for_calculus_presentation(
    ctx: &Context,
    cos_arg: ExprId,
    expr: ExprId,
) -> Option<i64> {
    if cas_ast::ordering::compare_expr(ctx, expr, cos_arg).is_eq() {
        return Some(1);
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !cas_ast::ordering::compare_expr(ctx, *base, cos_arg).is_eq() {
        return None;
    }
    let value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    value
        .is_integer()
        .then(|| value.to_integer().to_i64())
        .flatten()
}

struct AdditiveTrigPolynomialDerivativeForPresentation {
    numerator: ExprId,
    denominator: Option<ExprId>,
    required_conditions: Vec<crate::ImplicitCondition>,
}

fn additive_trig_polynomial_sqrt_radicand_derivative_for_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<AdditiveTrigPolynomialDerivativeForPresentation> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut has_trig_term = false;
    let mut has_variable_dependency = false;
    let mut derivative_terms = Vec::new();
    let mut denominator = None;
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            has_trig_term = true;
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                derivative_terms.push(derivative);
            }
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            derivative_terms.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if denominator.is_some_and(|existing| existing != ln_arg) {
                return None;
            }
            denominator = Some(ln_arg);
            derivative_terms.push(rational_const_for_calculus_presentation(ctx, ln_scale));
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            let neg_half = ctx.rational(-1, 2);
            let reciprocal_sqrt = ctx.add_raw(Expr::Pow(sqrt_arg, neg_half));
            let derivative = scale_expr_for_calculus_presentation(
                ctx,
                sqrt_scale * BigRational::new(1.into(), 2.into()),
                reciprocal_sqrt,
            );
            derivative_terms.push(derivative);
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((reciprocal_scale, reciprocal_arg)) =
            scaled_reciprocal_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            let two = ctx.num(2);
            let reciprocal_denominator = ctx.add(Expr::Pow(reciprocal_arg, two));
            if denominator.is_some_and(|existing| existing != reciprocal_denominator) {
                return None;
            }
            denominator = Some(reciprocal_denominator);
            derivative_terms.push(rational_const_for_calculus_presentation(
                ctx,
                -reciprocal_scale,
            ));
            required_conditions.push(crate::ImplicitCondition::NonZero(reciprocal_arg));
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            derivative_terms.push(derivative.to_expr(ctx));
        }
    }

    if !has_trig_term || !has_variable_dependency {
        return None;
    }

    let numerator = if let Some(denominator) = denominator {
        let scaled_terms: Vec<_> = derivative_terms
            .into_iter()
            .map(|term| {
                if cas_ast::views::as_rational_const(ctx, term, 8).is_some() {
                    term
                } else {
                    ctx.add(Expr::Mul(denominator, term))
                }
            })
            .collect();
        if scaled_terms.is_empty() {
            ctx.num(0)
        } else {
            cas_math::expr_nary::build_balanced_add(ctx, &scaled_terms)
        }
    } else if derivative_terms.is_empty() {
        ctx.num(0)
    } else {
        cas_math::expr_nary::build_balanced_add(ctx, &derivative_terms)
    };
    Some(AdditiveTrigPolynomialDerivativeForPresentation {
        numerator,
        denominator,
        required_conditions,
    })
}

fn tan_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Tan) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_tan_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = tan_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Cot) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = cot_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn scaled_sin_over_cos_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let sin_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, num, var_name, BuiltinFn::Sin)?;
    let cos_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, den, var_name, BuiltinFn::Cos)?;
    (sin_arg == cos_arg).then_some((scale, sin_arg))
}

fn scaled_cos_over_sin_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let cos_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, num, var_name, BuiltinFn::Cos)?;
    let sin_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, den, var_name, BuiltinFn::Sin)?;
    (cos_arg == sin_arg).then_some((scale, cos_arg))
}

fn scaled_tan_or_cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BuiltinFn)> {
    if let Some((scale, arg)) =
        scaled_tan_variable_arg_for_calculus_presentation(ctx, expr, var_name).or_else(|| {
            scaled_sin_over_cos_variable_arg_for_calculus_presentation(ctx, expr, var_name)
        })
    {
        return Some((scale, arg, BuiltinFn::Cos));
    }

    scaled_cot_variable_arg_for_calculus_presentation(ctx, expr, var_name)
        .or_else(|| scaled_cos_over_sin_variable_arg_for_calculus_presentation(ctx, expr, var_name))
        .map(|(scale, arg)| (-scale, arg, BuiltinFn::Sin))
}

fn scaled_sec_or_csc_variable_derivative_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let core = cas_ast::hold::unwrap_internal_hold(ctx, core);
    let Expr::Function(fn_id, args) = ctx.get(core).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    if ctx.sym_name(*sym_id) != var_name {
        return None;
    }

    match ctx.builtin_of(fn_id)? {
        BuiltinFn::Sec => {
            let sec = ctx.call_builtin(BuiltinFn::Sec, vec![args[0]]);
            let tan = ctx.call_builtin(BuiltinFn::Tan, vec![args[0]]);
            let derivative = cas_math::expr_nary::build_balanced_mul(ctx, &[sec, tan]);
            let derivative = scale_expr_for_calculus_presentation(ctx, scale, derivative);
            let cos = ctx.call_builtin(BuiltinFn::Cos, vec![args[0]]);
            Some((derivative, crate::ImplicitCondition::NonZero(cos)))
        }
        BuiltinFn::Csc => {
            let csc = ctx.call_builtin(BuiltinFn::Csc, vec![args[0]]);
            let cot = ctx.call_builtin(BuiltinFn::Cot, vec![args[0]]);
            let derivative = scale_ordered_product_for_calculus_presentation(ctx, -scale, csc, cot);
            let sin = ctx.call_builtin(BuiltinFn::Sin, vec![args[0]]);
            Some((derivative, crate::ImplicitCondition::NonZero(sin)))
        }
        _ => None,
    }
}

fn scale_ordered_product_for_calculus_presentation(
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

fn unary_variable_builtin_arg_for_calculus_presentation(
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

fn ln_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_ln_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = ln_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn exp_linear_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Exp) {
                return None;
            }
            let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, args[0], var_name)?;
            if arg_poly.degree() != 1 {
                return None;
            }
            let slope = arg_poly.coeffs.get(1)?.clone();
            (!slope.is_zero()).then_some((slope, expr))
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, exp, var_name)?;
            if arg_poly.degree() != 1 {
                return None;
            }
            let slope = arg_poly.coeffs.get(1)?.clone();
            (!slope.is_zero()).then_some((slope, expr))
        }
        _ => None,
    }
}

fn scaled_exp_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let (chain_scale, exp) = exp_linear_term_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale * chain_scale, exp))
}

fn exp_bounded_chain_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let inner = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Exp) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => exp,
        _ => return None,
    };

    bounded_sin_cos_term_bound_for_calculus_presentation(ctx, inner)?;
    let inner_derivative = differentiate(ctx, inner, var_name)?;
    if cas_ast::views::as_rational_const(ctx, inner_derivative, 8)
        .is_some_and(|value| value.is_zero())
    {
        return None;
    }
    Some((inner_derivative, expr))
}

fn scaled_exp_bounded_chain_derivative_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let (inner_derivative, exp_term) =
        exp_bounded_chain_term_for_calculus_presentation(ctx, core, var_name)?;
    let derivative = cas_math::expr_nary::build_balanced_mul(ctx, &[inner_derivative, exp_term]);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    Some(scale_expr_for_calculus_presentation(ctx, scale, derivative))
}

fn sqrt_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let radicand = calculus_sqrt_like_radicand(ctx, expr)?;
    let Expr::Variable(sym_id) = ctx.get(radicand) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(radicand)
}

fn scaled_sqrt_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let radicand = sqrt_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, radicand))
}

fn reciprocal_sqrt_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            sqrt_variable_arg_for_calculus_presentation(ctx, den, var_name)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            let Expr::Variable(sym_id) = ctx.get(base) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(base)
        }
        _ => None,
    }
}

fn scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, radicand) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(ctx, inner, var_name)?;
        return Some((-scale, radicand));
    }

    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    if let Some(radicand) =
        reciprocal_sqrt_variable_arg_for_calculus_presentation(ctx, core, var_name)
    {
        return Some((scale, radicand));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    let radicand = sqrt_variable_arg_for_calculus_presentation(ctx, den, var_name)?;
    Some((numerator_scale, radicand))
}

fn reciprocal_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            let Expr::Variable(sym_id) = ctx.get(den) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(den)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            let Expr::Variable(sym_id) = ctx.get(base) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(base)
        }
        _ => None,
    }
}

fn scaled_reciprocal_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, arg) =
            scaled_reciprocal_variable_term_for_calculus_presentation(ctx, inner, var_name)?;
        return Some((-scale, arg));
    }

    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    if let Some(arg) = reciprocal_variable_arg_for_calculus_presentation(ctx, core, var_name) {
        return Some((scale, arg));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    let Expr::Variable(sym_id) = ctx.get(den) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some((numerator_scale, den))
}

fn split_signed_numeric_scale_single_core_for_calculus_presentation(
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

fn distribute_half_over_additive_numerator_for_calculus_presentation(
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

fn compact_small_power_exponents_for_calculus_presentation(
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

fn compact_numeric_mul_factors_for_calculus_presentation(
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

fn signed_add_terms_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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

fn sqrt_raw_for_calculus_presentation(ctx: &mut Context, radicand: ExprId) -> ExprId {
    let fn_id = ctx.builtin_id(BuiltinFn::Sqrt);
    ctx.add_raw(Expr::Function(fn_id, vec![radicand]))
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

fn bounded_sin_cos_shift_margin_for_calculus_presentation(
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

fn bounded_sin_cos_term_bound_for_calculus_presentation(
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

fn sqrt_elementary_function_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    #[derive(Clone, Copy)]
    enum SqrtElementaryDerivativeShape {
        Function(BuiltinFn),
        DenominatorSquare(BuiltinFn),
        OnePlusArgSquare,
        OneMinusArgSquare,
        SqrtOneMinusArgSquare,
        SqrtOnePlusArgSquare,
        SqrtArgMinusOneTimesArgPlusOne,
        Log,
        LogConstantBase(i64),
    }

    let radicand = extract_square_root_base(ctx, target)?;
    let Expr::Function(fn_id, args) = ctx.get(radicand).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (shape, sign) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Cos),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cos) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Sin),
            -BigRational::one(),
        ),
        Some(BuiltinFn::Exp) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Exp),
            BigRational::one(),
        ),
        Some(BuiltinFn::Tan) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Cos),
            BigRational::one(),
        ),
        Some(BuiltinFn::Tanh) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Cosh),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cot) => (
            SqrtElementaryDerivativeShape::DenominatorSquare(BuiltinFn::Sin),
            -BigRational::one(),
        ),
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => (
            SqrtElementaryDerivativeShape::OnePlusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Atanh) => (
            SqrtElementaryDerivativeShape::OneMinusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Asin | BuiltinFn::Arcsin) => (
            SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Acos | BuiltinFn::Arccos) => (
            SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare,
            -BigRational::one(),
        ),
        Some(BuiltinFn::Ln) => (SqrtElementaryDerivativeShape::Log, BigRational::one()),
        Some(BuiltinFn::Log2) => (
            SqrtElementaryDerivativeShape::LogConstantBase(2),
            BigRational::one(),
        ),
        Some(BuiltinFn::Log10) => (
            SqrtElementaryDerivativeShape::LogConstantBase(10),
            BigRational::one(),
        ),
        Some(BuiltinFn::Asinh) => (
            SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare,
            BigRational::one(),
        ),
        Some(BuiltinFn::Acosh) => (
            SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne,
            BigRational::one(),
        ),
        Some(BuiltinFn::Sinh) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Cosh),
            BigRational::one(),
        ),
        Some(BuiltinFn::Cosh) => (
            SqrtElementaryDerivativeShape::Function(BuiltinFn::Sinh),
            BigRational::one(),
        ),
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (mut derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let mut coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    if let Some(core_value) = signed_rational_const_for_calculus_presentation(ctx, derivative_core)
    {
        coefficient *= core_value;
        derivative_core = ctx.num(1);
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_function = match shape {
        SqrtElementaryDerivativeShape::Function(derivative_fn) => {
            Some(ctx.call_builtin(derivative_fn, vec![args[0]]))
        }
        SqrtElementaryDerivativeShape::DenominatorSquare(_) => None,
        SqrtElementaryDerivativeShape::OnePlusArgSquare => None,
        SqrtElementaryDerivativeShape::OneMinusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare => None,
        SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne => None,
        SqrtElementaryDerivativeShape::Log => None,
        SqrtElementaryDerivativeShape::LogConstantBase(_) => None,
    };
    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = match derivative_function {
        Some(derivative_function) if derivative_core_is_one => derivative_function,
        Some(derivative_function) => {
            cas_math::expr_nary::build_balanced_mul(ctx, &[derivative_core, derivative_function])
        }
        None => derivative_core,
    };
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::Log) {
        denominator_factors.push(args[0]);
    }
    if let SqrtElementaryDerivativeShape::LogConstantBase(base) = shape {
        denominator_factors.push(args[0]);
        let base_expr = ctx.num(base);
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]));
    }
    if let SqrtElementaryDerivativeShape::DenominatorSquare(denominator_fn) = shape {
        let denominator_arg = ctx.call_builtin(denominator_fn, vec![args[0]]);
        let two = ctx.num(2);
        denominator_factors.push(ctx.add(Expr::Pow(denominator_arg, two)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::OnePlusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        denominator_factors.push(ctx.add(Expr::Add(arg_square, one)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::OneMinusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let neg_arg_square = ctx.add(Expr::Neg(arg_square));
        denominator_factors.push(ctx.add(Expr::Add(one, neg_arg_square)));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::SqrtOneMinusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let neg_arg_square = ctx.add(Expr::Neg(arg_square));
        let radicand = ctx.add(Expr::Add(one, neg_arg_square));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]));
    }
    if matches!(shape, SqrtElementaryDerivativeShape::SqrtOnePlusArgSquare) {
        let two = ctx.num(2);
        let one = ctx.num(1);
        let arg_square = ctx.add(Expr::Pow(args[0], two));
        let radicand = ctx.add(Expr::Add(arg_square, one));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]));
    }
    if matches!(
        shape,
        SqrtElementaryDerivativeShape::SqrtArgMinusOneTimesArgPlusOne
    ) {
        let one_poly = Polynomial::one(var_name.to_string());
        let arg_minus_one = arg_poly.sub(&one_poly).to_expr(ctx);
        let arg_plus_one = arg_poly.add(&one_poly).to_expr(ctx);
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_minus_one]));
        denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![arg_plus_one]));
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);

    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

fn sqrt_shifted_exp_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (exp_expr, exp_arg, exp_scale, shift) =
        scaled_exp_plus_positive_rational_shift(ctx, radicand)?;
    if !exp_scale.is_positive() || !shift.is_positive() {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, exp_arg, var_name).ok()?;
    if arg_poly.degree() > 1 {
        return None;
    }
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.degree() != 0 {
        return None;
    }
    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if derivative_coeff.is_zero() {
        return Some(ctx.num(0));
    }

    let coefficient = exp_scale * derivative_coeff * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, exp_expr);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff.is_one() {
        sqrt_radicand
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn scaled_exp_plus_positive_rational_shift(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, BigRational, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    scaled_exp_term_arg(ctx, *left)
        .zip(cas_ast::views::as_rational_const(ctx, *right, 8))
        .map(|((exp_expr, arg, exp_scale), shift)| (exp_expr, arg, exp_scale, shift))
        .or_else(|| {
            scaled_exp_term_arg(ctx, *right)
                .zip(cas_ast::views::as_rational_const(ctx, *left, 8))
                .map(|((exp_expr, arg, exp_scale), shift)| (exp_expr, arg, exp_scale, shift))
        })
}

fn scaled_exp_term_arg(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(arg) = exp_term_arg(ctx, expr) {
        return Some((expr, arg, BigRational::one()));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = BigRational::one();
    let mut exp_term = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        if exp_term.is_none() {
            if let Some(arg) = exp_term_arg(ctx, factor) {
                exp_term = Some((factor, arg));
                continue;
            }
        }
        return None;
    }

    let (exp_expr, arg) = exp_term?;
    Some((exp_expr, arg, scale))
}

fn exp_term_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Exp) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => Some(*exp),
        _ => None,
    }
}

fn same_arg_unary_pair_for_calculus_presentation(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left = cas_ast::hold::unwrap_internal_hold(ctx, left);
    let right = cas_ast::hold::unwrap_internal_hold(ctx, right);
    let Expr::Function(left_fn, left_args) = ctx.get(left) else {
        return None;
    };
    let Expr::Function(right_fn, right_args) = ctx.get(right) else {
        return None;
    };
    if left_args.len() != 1
        || right_args.len() != 1
        || ctx.builtin_of(*left_fn) != Some(left_builtin)
        || ctx.builtin_of(*right_fn) != Some(right_builtin)
        || compare_expr(ctx, left_args[0], right_args[0]) != std::cmp::Ordering::Equal
    {
        return None;
    }

    Some(left_args[0])
}

fn sin_minus_cos_arg_for_calculus_presentation(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Sub(left, right) => same_arg_unary_pair_for_calculus_presentation(
            ctx,
            left,
            BuiltinFn::Sin,
            right,
            BuiltinFn::Cos,
        ),
        Expr::Add(left, right) => {
            if let Expr::Neg(negated_right) = ctx.get(right) {
                if let Some(arg) = same_arg_unary_pair_for_calculus_presentation(
                    ctx,
                    left,
                    BuiltinFn::Sin,
                    *negated_right,
                    BuiltinFn::Cos,
                ) {
                    return Some(arg);
                }
            }
            if let Expr::Neg(negated_left) = ctx.get(left) {
                if let Some(arg) = same_arg_unary_pair_for_calculus_presentation(
                    ctx,
                    right,
                    BuiltinFn::Sin,
                    *negated_left,
                    BuiltinFn::Cos,
                ) {
                    return Some(arg);
                }
            }
            None
        }
        _ => None,
    }
}

fn sin_plus_cos_arg_for_calculus_presentation(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };

    same_arg_unary_pair_for_calculus_presentation(ctx, left, BuiltinFn::Sin, right, BuiltinFn::Cos)
        .or_else(|| {
            same_arg_unary_pair_for_calculus_presentation(
                ctx,
                right,
                BuiltinFn::Sin,
                left,
                BuiltinFn::Cos,
            )
        })
}

fn exp_trig_by_parts_primitive_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut exp_expr = None;
    let mut exp_arg = None;
    let mut trig_result = None;
    for factor in factors {
        if let Some(arg) = exp_term_arg(ctx, factor) {
            if exp_expr.replace(factor).is_some() || exp_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }

        if let Some(arg) = sin_minus_cos_arg_for_calculus_presentation(ctx, factor) {
            if trig_result.replace((arg, BuiltinFn::Sin)).is_some() {
                return None;
            }
            continue;
        }

        if let Some(arg) = sin_plus_cos_arg_for_calculus_presentation(ctx, factor) {
            if trig_result.replace((arg, BuiltinFn::Cos)).is_some() {
                return None;
            }
            continue;
        }

        scale *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
    }

    let exp_expr = exp_expr?;
    let exp_arg = exp_arg?;
    let (trig_arg, result_builtin) = trig_result?;
    if compare_expr(ctx, exp_arg, trig_arg) != std::cmp::Ordering::Equal {
        return None;
    }

    let derivative_coeff = nonzero_affine_variable_derivative(ctx, exp_arg, var_name)?;
    let trig = ctx.call_builtin(result_builtin, vec![exp_arg]);
    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[exp_expr, trig]);
    let scaled = scale_expr_for_calculus_presentation(
        ctx,
        scale * BigRational::from_integer(2.into()) * derivative_coeff,
        product,
    );

    Some(fold_numeric_mul_constants_for_hold(ctx, scaled))
}

fn sqrt_shifted_ln_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (ln_arg, ln_scale, base_ln_factor, shift) =
        scaled_ln_plus_positive_rational_shift(ctx, radicand)?;
    if !ln_scale.is_positive() || !shift.is_positive() {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, ln_arg, var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = ln_scale * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(ln_arg);
    if let Some(base_ln_factor) = base_ln_factor {
        denominator_factors.push(base_ln_factor);
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn scaled_ln_plus_positive_rational_shift(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, Option<ExprId>, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let (left, right) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };

    if let Some((arg, scale, base_ln_factor)) = scaled_ln_term_arg(ctx, left) {
        if let Some(shift) = cas_ast::views::as_rational_const(ctx, right, 8) {
            return Some((arg, scale, base_ln_factor, shift));
        }
    }
    if let Some((arg, scale, base_ln_factor)) = scaled_ln_term_arg(ctx, right) {
        if let Some(shift) = cas_ast::views::as_rational_const(ctx, left, 8) {
            return Some((arg, scale, base_ln_factor, shift));
        }
    }

    None
}

fn scaled_ln_term_arg(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, Option<ExprId>)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some((arg, base_ln_factor)) = shifted_root_log_term_arg(ctx, expr) {
        return Some((arg, BigRational::one(), base_ln_factor));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = BigRational::one();
    let mut ln_arg = None;
    let mut base_ln_factor = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        if let Some((arg, factor_base_ln)) = shifted_root_log_term_arg(ctx, factor) {
            if ln_arg.replace(arg).is_none() && base_ln_factor.replace(factor_base_ln).is_none() {
                continue;
            }
        }
        return None;
    }

    Some((ln_arg?, scale, base_ln_factor?))
}

fn shifted_root_log_term_arg(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, Option<ExprId>)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Ln) => Some((args[0], None)),
        Some(BuiltinFn::Log2) => {
            let two = ctx.num(2);
            Some((args[0], Some(ctx.call_builtin(BuiltinFn::Ln, vec![two]))))
        }
        Some(BuiltinFn::Log10) => {
            let ten = ctx.num(10);
            Some((args[0], Some(ctx.call_builtin(BuiltinFn::Ln, vec![ten]))))
        }
        _ => None,
    }
}

fn sqrt_reciprocal_trig_function_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let Expr::Function(fn_id, args) = ctx.get(radicand).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (derivative_fn, sign) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sec) => (BuiltinFn::Tan, BigRational::one()),
        Some(BuiltinFn::Csc) => (BuiltinFn::Cot, -BigRational::one()),
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (mut derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let mut coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    if let Some(core_value) = signed_rational_const_for_calculus_presentation(ctx, derivative_core)
    {
        coefficient *= core_value;
        derivative_core = ctx.num(1);
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let trig_factor = ctx.call_builtin(derivative_fn, vec![args[0]]);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut numerator_factors = Vec::new();
    if !derivative_core_is_one {
        numerator_factors.push(derivative_core);
    }
    numerator_factors.push(trig_factor);
    numerator_factors.push(sqrt_radicand);
    let numerator_core = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    if denominator_coeff == BigRational::one() {
        return Some(numerator);
    }

    let denominator = rational_const_for_calculus_presentation(ctx, denominator_coeff);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn scaled_square_root_radicand_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    if let Some(radicand) = extract_square_root_base(ctx, expr) {
        return Some((BigRational::one(), radicand));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = BigRational::one();
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }

        let factor_radicand = extract_square_root_base(ctx, factor)?;
        if radicand.replace(factor_radicand).is_some() {
            return None;
        }
    }

    if scale.is_zero() {
        return None;
    }

    Some((scale, radicand?))
}

fn polynomial_times_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(target) else {
        return None;
    };

    let mut polynomial_factors = Vec::new();
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
        } else {
            polynomial_factors.push(factor);
        }
    }

    let radicand = radicand?;
    if polynomial_factors.is_empty() {
        return None;
    }

    let polynomial_expr = if polynomial_factors.len() == 1 {
        polynomial_factors[0]
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &polynomial_factors)
    };
    let multiplier_poly =
        polynomial_radicand_for_calculus_presentation(ctx, polynomial_expr, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let multiplier_derivative = multiplier_poly.derivative();
    let radicand_derivative = radicand_poly.derivative();

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let numerator_poly = multiplier_derivative
        .mul(&radicand_poly)
        .mul(&two_poly)
        .add(&multiplier_poly.mul(&radicand_derivative));
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_content * BigRational::new(1.into(), 2.into())))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn sqrt_polynomial_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let (numerator_scale, radicand) =
        scaled_square_root_radicand_for_calculus_presentation(ctx, numerator_expr)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_expr, var_name)?;
    if denominator_poly.is_zero() {
        return None;
    }

    let radicand_derivative = radicand_poly.derivative();
    let denominator_derivative = denominator_poly.derivative();
    let mut numerator_poly = denominator_poly.mul(&radicand_derivative).sub(
        &radicand_poly
            .mul(&denominator_derivative)
            .mul(&Polynomial::new(
                vec![BigRational::from_integer(2.into())],
                var_name.to_string(),
            )),
    );
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut presentation_denominator_power = {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(denominator_expr, two))
    };
    let denominator_power_factor = positive_integer_polynomial_power_for_calculus_presentation(
        ctx,
        denominator_expr,
        var_name,
    )
    .or_else(|| expanded_affine_square_for_calculus_presentation(ctx, denominator_expr, var_name));
    if let Some((base, exponent, base_poly)) = denominator_power_factor {
        if exponent > 1 {
            let cancellable = polynomial_power_for_calculus_presentation(&base_poly, exponent - 1);
            if let Ok((quotient, remainder)) = numerator_poly.div_rem(&cancellable) {
                if remainder.is_zero() {
                    numerator_poly = quotient;
                    let denominator_exponent = ctx.num((exponent + 1) as i64);
                    presentation_denominator_power = ctx.add(Expr::Pow(base, denominator_exponent));
                }
            }
        }
    } else if let Ok((quotient, remainder)) = numerator_poly.div_rem(&denominator_poly) {
        if remainder.is_zero() {
            numerator_poly = quotient;
            presentation_denominator_power = denominator_expr;
        }
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let coefficient = numerator_scale * numerator_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_radicand, presentation_denominator_power],
    );
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn sqrt_of_polynomial_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let quotient = calculus_sqrt_like_radicand(ctx, target)?;
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(quotient).clone() else {
        return None;
    };
    let numerator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, numerator_expr, var_name)?;
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_expr, var_name)?;
    if denominator_poly.is_zero() {
        return None;
    }

    let mut numerator_result_poly = denominator_poly
        .mul(&numerator_poly.derivative())
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if numerator_result_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut presentation_denominator_power = {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(denominator_expr, two))
    };
    let denominator_power_factor = positive_integer_polynomial_power_for_calculus_presentation(
        ctx,
        denominator_expr,
        var_name,
    )
    .or_else(|| expanded_affine_square_for_calculus_presentation(ctx, denominator_expr, var_name));
    if let Some((base, exponent, base_poly)) = denominator_power_factor {
        let mut remaining_exponent = 2 * exponent;
        while remaining_exponent > 0 {
            let Ok((quotient_poly, remainder)) = numerator_result_poly.div_rem(&base_poly) else {
                break;
            };
            if !remainder.is_zero() {
                break;
            }
            numerator_result_poly = quotient_poly;
            remaining_exponent -= 1;
        }
        presentation_denominator_power = match remaining_exponent {
            0 => ctx.num(1),
            1 => base,
            _ => {
                let exponent = ctx.num(remaining_exponent as i64);
                ctx.add(Expr::Pow(base, exponent))
            }
        };
    } else if let Ok((quotient_poly, remainder)) = numerator_result_poly.div_rem(&denominator_poly)
    {
        if remainder.is_zero() {
            numerator_result_poly = quotient_poly;
            presentation_denominator_power = denominator_expr;
        }
    }

    let raw_numerator = numerator_result_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let coefficient = numerator_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_quotient = ctx.call_builtin(BuiltinFn::Sqrt, vec![quotient]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_quotient, presentation_denominator_power],
    );
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn positive_integer_polynomial_power_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, usize, Polynomial)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
    if !exponent.is_integer() || exponent <= BigRational::one() {
        return None;
    }
    let exponent = exponent.to_integer().to_usize()?;
    if let Some((compact_base, compact_exponent, compact_base_poly)) =
        expanded_affine_square_for_calculus_presentation(ctx, base, var_name)
    {
        return Some((compact_base, exponent * compact_exponent, compact_base_poly));
    }
    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
    Some((base, exponent, base_poly))
}

fn expanded_affine_square_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, usize, Polynomial)> {
    let poly = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name)?;
    if poly.degree() != 2 {
        return None;
    }

    let leading = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = exact_positive_rational_sqrt_for_calculus_presentation(&leading)?;

    let linear = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let shift = linear / (BigRational::from_integer(2.into()) * slope.clone());
    if constant != shift.clone() * shift.clone() {
        return None;
    }

    let base_poly = Polynomial::new(vec![shift, slope], var_name.to_string());
    let base = base_poly.to_expr(ctx);
    Some((base, 2, base_poly))
}

pub(crate) fn sqrt_polynomial_quotient_has_powered_expanded_affine_square_denominator(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return false;
    };
    if calculus_sqrt_like_radicand(ctx, numerator_expr).is_none() {
        return false;
    }

    let denominator_expr = cas_ast::hold::unwrap_internal_hold(ctx, denominator_expr);
    let Expr::Pow(base, exp) = ctx.get(denominator_expr).clone() else {
        return false;
    };
    let Some(exponent) = cas_ast::views::as_rational_const(ctx, exp, 8) else {
        return false;
    };
    if !exponent.is_integer() || exponent < BigRational::from_integer(2.into()) {
        return false;
    }

    expanded_affine_square_for_calculus_presentation(ctx, base, var_name).is_some()
}

fn polynomial_over_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let radicand = calculus_sqrt_like_radicand(ctx, denominator_expr)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    polynomial_over_sqrt_polynomial_derivative_presentation_from_parts(
        ctx,
        numerator_expr,
        radicand,
        &radicand_poly,
        var_name,
    )
}

fn polynomial_over_sqrt_polynomial_derivative_presentation_from_parts(
    ctx: &mut Context,
    numerator_expr: ExprId,
    radicand: ExprId,
    radicand_poly: &Polynomial,
    var_name: &str,
) -> Option<ExprId> {
    let numerator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, numerator_expr, var_name)?;
    if radicand_poly.is_zero() {
        return None;
    }
    let numerator_derivative = numerator_poly.derivative();
    if numerator_derivative.is_zero() {
        return None;
    }
    if numerator_derivative.degree() > 2 {
        return None;
    }

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let mut numerator_result_poly = numerator_derivative
        .mul(radicand_poly)
        .mul(&two_poly)
        .sub(&numerator_poly.mul(&radicand_poly.derivative()));
    if numerator_result_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let mut cancel_radicand_from_denominator = false;
    if let Ok((quotient, remainder)) = numerator_result_poly.div_rem(radicand_poly) {
        if remainder.is_zero() {
            numerator_result_poly = quotient;
            cancel_radicand_from_denominator = true;
        }
    }
    let mut lift_radicand_to_numerator = false;
    if cancel_radicand_from_denominator {
        if let Ok((quotient, remainder)) = numerator_result_poly.div_rem(radicand_poly) {
            if remainder.is_zero() {
                numerator_result_poly = quotient;
                lift_radicand_to_numerator = true;
            }
        }
    }

    let raw_numerator = numerator_result_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let coefficient = numerator_content / BigRational::new(2.into(), 1.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let numerator_core = if lift_radicand_to_numerator {
        if cas_ast::views::as_rational_const(ctx, numerator_core, 8)
            .is_some_and(|value| value.is_one())
        {
            sqrt_radicand
        } else {
            cas_math::expr_nary::build_balanced_mul(ctx, &[numerator_core, sqrt_radicand])
        }
    } else {
        numerator_core
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);
    let core_denominator = if cancel_radicand_from_denominator {
        if lift_radicand_to_numerator {
            ctx.num(1)
        } else {
            sqrt_radicand
        }
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_radicand])
    };
    let denominator_coeff_is_one = denominator_coeff == BigRational::one();
    let denominator = if denominator_coeff_is_one {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    if lift_radicand_to_numerator && denominator_coeff_is_one {
        return Some(numerator);
    }

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(crate) fn polynomial_over_sqrt_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let radicand = calculus_sqrt_like_radicand(ctx, denominator_expr)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result = polynomial_over_sqrt_polynomial_derivative_presentation_from_parts(
        ctx,
        numerator_expr,
        radicand,
        &radicand_poly,
        var_name,
    )?;
    let required_conditions = if polynomial_is_strictly_positive_everywhere(&radicand_poly) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(radicand)]
    };
    Some((result, required_conditions))
}

pub(crate) fn sqrt_polynomial_quotient_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(numerator_expr, _) = ctx.get(target).clone() else {
        return None;
    };
    let (_, radicand) = scaled_square_root_radicand_for_calculus_presentation(ctx, numerator_expr)?;
    let result = sqrt_polynomial_quotient_derivative_presentation(ctx, target, var_name)?;
    Some((result, vec![crate::ImplicitCondition::Positive(radicand)]))
}

pub(crate) fn sqrt_of_polynomial_quotient_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let quotient = calculus_sqrt_like_radicand(ctx, target)?;
    let result = sqrt_of_polynomial_quotient_derivative_presentation(ctx, target, var_name)?;
    Some((result, vec![crate::ImplicitCondition::Positive(quotient)]))
}

fn nonnegative_integer_offset_from_half_exponent(value: BigRational) -> Option<usize> {
    let offset = value - BigRational::new(1.into(), 2.into());
    if offset.is_negative() || !offset.is_integer() {
        return None;
    }
    offset.to_integer().to_usize()
}

fn polynomial_power_for_calculus_presentation(poly: &Polynomial, exponent: usize) -> Polynomial {
    let mut result = Polynomial::one(poly.var.clone());
    for _ in 0..exponent {
        result = result.mul(poly);
    }
    result
}

fn sqrt_times_polynomial_factor_parts(
    ctx: &Context,
    factor: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
    if let Some(radicand) = extract_square_root_base(ctx, factor) {
        polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        return Some((radicand, Polynomial::one(var_name.to_string())));
    }

    let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
    let offset = nonnegative_integer_offset_from_half_exponent(exponent)?;
    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
    Some((
        base,
        polynomial_power_for_calculus_presentation(&base_poly, offset),
    ))
}

fn denominator_product_sqrt_polynomial_parts(
    ctx: &Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    let mut radicand = None;
    let mut polynomial = Polynomial::one(var_name.to_string());
    for factor in cas_math::expr_nary::mul_leaves(ctx, denominator_expr) {
        if let Some((factor_radicand, factor_polynomial)) =
            sqrt_times_polynomial_factor_parts(ctx, factor, var_name)
        {
            if factor_polynomial != Polynomial::one(var_name.to_string()) {
                return None;
            }
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
        } else {
            let factor_poly = polynomial_radicand_for_calculus_presentation(ctx, factor, var_name)?;
            polynomial = polynomial.mul(&factor_poly);
        }
    }

    Some((radicand?, polynomial))
}

fn denominator_sum_sqrt_polynomial_parts(
    ctx: &Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    let mut radicand = None;
    let mut polynomial = Polynomial::zero(var_name.to_string());
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, denominator_expr).terms;
    if terms.len() < 2 {
        return None;
    }
    for (term, sign) in terms {
        let mut term_radicand = None;
        let mut term_polynomial = Polynomial::one(var_name.to_string());
        for factor in cas_math::expr_nary::mul_leaves(ctx, term) {
            if let Some((factor_radicand, factor_polynomial)) =
                sqrt_times_polynomial_factor_parts(ctx, factor, var_name)
            {
                if term_radicand.replace(factor_radicand).is_some() {
                    return None;
                }
                term_polynomial = term_polynomial.mul(&factor_polynomial);
            } else {
                let factor_poly =
                    polynomial_radicand_for_calculus_presentation(ctx, factor, var_name)?;
                term_polynomial = term_polynomial.mul(&factor_poly);
            }
        }
        let term_radicand = term_radicand?;
        if let Some(existing) = radicand {
            if compare_expr(ctx, existing, term_radicand) != std::cmp::Ordering::Equal {
                return None;
            }
        } else {
            radicand = Some(term_radicand);
        }
        polynomial = match sign {
            cas_math::expr_nary::Sign::Pos => polynomial.add(&term_polynomial),
            cas_math::expr_nary::Sign::Neg => polynomial.sub(&term_polynomial),
        };
    }

    Some((radicand?, polynomial))
}

fn denominator_sqrt_polynomial_parts(
    ctx: &Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    denominator_product_sqrt_polynomial_parts(ctx, denominator_expr, var_name)
        .or_else(|| denominator_sum_sqrt_polynomial_parts(ctx, denominator_expr, var_name))
}

fn negative_half_power_target_parts(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, Polynomial)> {
    let target = cas_ast::hold::unwrap_internal_hold(ctx, target);
    let mut scale = BigRational::one();
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
            return None;
        };
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent != BigRational::new((-1).into(), 2.into()) {
            return None;
        }
        polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
        if radicand.replace(base).is_some() {
            return None;
        }
    }

    Some((scale, radicand?, Polynomial::one(var_name.to_string())))
}

fn reciprocal_sqrt_polynomial_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_scale, radicand, denominator_poly, denominator_power_factor) =
        if let Some((scale, radicand, denominator_poly)) =
            negative_half_power_target_parts(ctx, target, var_name)
        {
            (scale, radicand, denominator_poly, None)
        } else {
            let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
                return None;
            };
            let numerator_scale = cas_ast::views::as_rational_const(ctx, numerator_expr, 8)?;
            let (radicand, denominator_poly) =
                denominator_sqrt_polynomial_parts(ctx, denominator_expr, var_name)?;
            let denominator_power_factor =
                sqrt_denominator_positive_integer_power_factor(ctx, denominator_expr, var_name);
            (
                numerator_scale,
                radicand,
                denominator_poly,
                denominator_power_factor,
            )
        };
    if numerator_scale.is_zero() {
        return Some(ctx.num(0));
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if denominator_poly.is_zero() {
        return None;
    }

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let mut numerator_poly = denominator_poly.mul(&radicand_poly.derivative()).add(
        &radicand_poly
            .mul(&denominator_poly.derivative())
            .mul(&two_poly),
    );
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut presentation_denominator_power = None;
    if let Some((base, exponent, base_poly)) = denominator_power_factor {
        if exponent > 1 {
            let cancellable = polynomial_power_for_calculus_presentation(&base_poly, exponent - 1);
            if let Ok((quotient, remainder)) = numerator_poly.div_rem(&cancellable) {
                if remainder.is_zero() {
                    numerator_poly = quotient;
                    let denominator_exponent = ctx.num((exponent + 1) as i64);
                    presentation_denominator_power =
                        Some(ctx.add(Expr::Pow(base, denominator_exponent)));
                }
            }
        }
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let raw_denominator = denominator_poly.to_expr(ctx);
    let (denominator_core, denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_denominator);
    let denominator_content_square = denominator_content.clone() * denominator_content;
    if denominator_content_square.is_zero() {
        return None;
    }

    let coefficient = -numerator_scale * numerator_content * BigRational::new(1.into(), 2.into())
        / denominator_content_square;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_parts = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_parts.push(radicand);
    denominator_parts.push(sqrt_radicand);
    if !cas_ast::views::as_rational_const(ctx, denominator_core, 8)
        .is_some_and(|value| value.is_one())
    {
        denominator_parts.push(
            presentation_denominator_power.unwrap_or_else(|| squared_expr(ctx, denominator_core)),
        );
    }
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(crate) fn reciprocal_sqrt_polynomial_product_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand =
        if let Some((_, radicand, _)) = negative_half_power_target_parts(ctx, target, var_name) {
            radicand
        } else {
            let Expr::Div(_, denominator_expr) = ctx.get(target).clone() else {
                return None;
            };
            let (radicand, _) = denominator_sqrt_polynomial_parts(ctx, denominator_expr, var_name)?;
            radicand
        };
    let result = reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, var_name)?;
    Some((result, vec![crate::ImplicitCondition::Positive(radicand)]))
}

fn sqrt_denominator_positive_integer_power_factor(
    ctx: &mut Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, usize, Polynomial)> {
    let mut power_factor = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, denominator_expr) {
        if extract_square_root_base(ctx, factor).is_some() {
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some() {
            continue;
        }
        let factor_power =
            positive_integer_polynomial_power_for_calculus_presentation(ctx, factor, var_name)?;
        if power_factor.replace(factor_power).is_some() {
            return None;
        }
    }
    power_factor
}

fn inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => -BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => BigRational::one(),
        _ => return None,
    };

    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(args[0]).clone() else {
        return None;
    };
    let argument_scale = cas_ast::views::as_rational_const(ctx, numerator_expr, 8)?;
    if argument_scale.is_zero() {
        return None;
    }
    let (radicand, denominator_poly) =
        denominator_sqrt_polynomial_parts(ctx, denominator_expr, var_name)?;
    if denominator_poly.is_zero() || denominator_poly == Polynomial::one(var_name.to_string()) {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let numerator_poly = denominator_poly.mul(&radicand_poly.derivative()).add(
        &radicand_poly
            .mul(&denominator_poly.derivative())
            .mul(&two_poly),
    );
    if numerator_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let raw_denominator = denominator_poly.to_expr(ctx);
    let (denominator_core, denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_denominator);
    let denominator_content_square = denominator_content.clone() * denominator_content;
    if denominator_content_square.is_zero() {
        return None;
    }

    let coefficient = derivative_sign
        * argument_scale.clone()
        * numerator_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator_square = squared_expr(ctx, denominator_core);
    let root_product =
        cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, denominator_square]);
    let scaled_root_product =
        scale_expr_for_calculus_presentation(ctx, denominator_content_square, root_product);
    let scale_square = argument_scale.clone() * argument_scale;
    let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
    let gap = ctx.add(Expr::Add(scaled_root_product, scale_square_expr));

    let mut denominator_parts = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_parts.push(sqrt_radicand);
    denominator_parts.push(gap);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    let mut required_conditions = Vec::new();
    if !polynomial_is_strictly_positive_everywhere(&radicand_poly) {
        required_conditions.push(crate::ImplicitCondition::Positive(radicand));
    }
    required_conditions.push(crate::ImplicitCondition::NonZero(denominator_core));

    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some((ctx.add(Expr::Hold(compact)), required_conditions))
}

fn inverse_tangent_reciprocal_sqrt_shifted_sqrt_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => -BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => BigRational::one(),
        _ => return None,
    };

    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(args[0]).clone() else {
        return None;
    };
    let argument_scale = cas_ast::views::as_rational_const(ctx, numerator_expr, 8)?;
    if argument_scale.is_zero() {
        return None;
    }

    let (radicand, shift) = sqrt_times_nonzero_shifted_sqrt_parts(ctx, denominator_expr)?;
    let d_radicand = differentiate(ctx, radicand, var_name)?;
    let d_radicand_scale = cas_ast::views::as_rational_const(ctx, d_radicand, 8)?;
    if d_radicand_scale.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let two = ctx.num(2);
    let two_sqrt = ctx.add(Expr::Mul(two, sqrt_radicand));
    let shift_is_positive = shift.is_positive();
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift);
    let numerator_core = ctx.add(Expr::Add(two_sqrt, shift_expr));
    let coefficient = derivative_sign * argument_scale.clone() * d_radicand_scale;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let root_term = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, shifted_sqrt_squared]);
    let scale_square = argument_scale.clone() * argument_scale;
    let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
    let gap = ctx.add(Expr::Add(root_term, scale_square_expr));

    let mut denominator_parts = Vec::new();
    let denominator_scale = denominator_coeff * BigRational::from_integer(2.into());
    if denominator_scale != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_scale,
        ));
    }
    denominator_parts.push(sqrt_radicand);
    denominator_parts.push(gap);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    let compact = ctx.add(Expr::Div(numerator, denominator));
    let mut required_conditions = vec![crate::ImplicitCondition::Positive(radicand)];
    if !shift_is_positive {
        required_conditions.push(crate::ImplicitCondition::NonZero(shifted_sqrt));
    }

    Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions))
}

fn sqrt_over_positive_shifted_sqrt_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let (numerator_scale, numerator_radicand) =
        scaled_square_root_radicand_for_calculus_presentation(ctx, numerator_expr)?;
    let (denominator_radicand, shift) =
        shifted_sqrt_positive_constant_parts(ctx, denominator_expr)?;
    if compare_expr(ctx, numerator_radicand, denominator_radicand) != std::cmp::Ordering::Equal {
        return None;
    }

    let d_radicand = differentiate(ctx, numerator_radicand, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_radicand, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), numerator_radicand));
    }

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_radicand);
    let coefficient =
        numerator_scale * shift.clone() * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![numerator_radicand]);
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift);
    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt_squared]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some((
        ctx.add(Expr::Div(numerator, denominator)),
        numerator_radicand,
    ))
}

pub(crate) fn sqrt_over_positive_shifted_sqrt_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive) =
        sqrt_over_positive_shifted_sqrt_derivative(ctx, target, var_name)?;
    Some((
        result,
        vec![crate::ImplicitCondition::Positive(required_positive)],
    ))
}

fn elementary_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    #[derive(Clone, Copy)]
    enum SqrtChainShape {
        NumeratorFunction(BuiltinFn),
        ExponentialPower,
        DenominatorSquare(BuiltinFn),
        NumeratorOverDenominatorSquare {
            numerator_fn: BuiltinFn,
            denominator_fn: BuiltinFn,
        },
        ReciprocalTrigProduct {
            primary_fn: BuiltinFn,
            companion_fn: BuiltinFn,
        },
    }

    let (arg, shape, sign) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            let (shape, sign) = match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Exp) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Exp),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Sin) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Cos),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Cos) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Sin),
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Tan) => (
                    SqrtChainShape::DenominatorSquare(BuiltinFn::Cos),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Cot) => (
                    SqrtChainShape::DenominatorSquare(BuiltinFn::Sin),
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Sec) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Sec,
                        companion_fn: BuiltinFn::Tan,
                    },
                    BigRational::one(),
                ),
                Some(BuiltinFn::Csc) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Csc,
                        companion_fn: BuiltinFn::Cot,
                    },
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Sinh) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Cosh),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Cosh) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Sinh),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Tanh) => (
                    SqrtChainShape::DenominatorSquare(BuiltinFn::Cosh),
                    BigRational::one(),
                ),
                _ => return None,
            };
            (args[0], shape, sign)
        }
        Expr::Div(numerator, denominator) => {
            let numerator_sign = cas_ast::views::as_rational_const(ctx, numerator, 8)
                .filter(|value| value == &BigRational::one() || value == &-BigRational::one())?;
            let Expr::Function(den_fn_id, den_args) = ctx.get(denominator).clone() else {
                return None;
            };
            if den_args.len() != 1 {
                return None;
            }
            let (shape, sign) = match ctx.builtin_of(den_fn_id) {
                Some(BuiltinFn::Cos) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Sec,
                        companion_fn: BuiltinFn::Tan,
                    },
                    BigRational::one(),
                ),
                Some(BuiltinFn::Sin) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Csc,
                        companion_fn: BuiltinFn::Cot,
                    },
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Cosh) => (
                    SqrtChainShape::NumeratorOverDenominatorSquare {
                        numerator_fn: BuiltinFn::Sinh,
                        denominator_fn: BuiltinFn::Cosh,
                    },
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Sinh) => (
                    SqrtChainShape::NumeratorOverDenominatorSquare {
                        numerator_fn: BuiltinFn::Cosh,
                        denominator_fn: BuiltinFn::Sinh,
                    },
                    -BigRational::one(),
                ),
                _ => return None,
            };
            (den_args[0], shape, sign * numerator_sign)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            (exp, SqrtChainShape::ExponentialPower, BigRational::one())
        }
        _ => return None,
    };

    let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let numerator_function = match shape {
        SqrtChainShape::NumeratorFunction(outer_derivative_fn) => {
            Some(ctx.call_builtin(outer_derivative_fn, vec![sqrt_radicand]))
        }
        SqrtChainShape::ExponentialPower => {
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, sqrt_radicand)))
        }
        SqrtChainShape::DenominatorSquare(_) => None,
        SqrtChainShape::NumeratorOverDenominatorSquare { numerator_fn, .. } => {
            Some(ctx.call_builtin(numerator_fn, vec![sqrt_radicand]))
        }
        SqrtChainShape::ReciprocalTrigProduct {
            primary_fn,
            companion_fn,
        } => {
            let primary = ctx.call_builtin(primary_fn, vec![sqrt_radicand]);
            let companion = ctx.call_builtin(companion_fn, vec![sqrt_radicand]);
            Some(cas_math::expr_nary::build_balanced_mul(
                ctx,
                &[primary, companion],
            ))
        }
    };
    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = match numerator_function {
        Some(numerator_function) if derivative_core_is_one => numerator_function,
        Some(numerator_function) => {
            cas_math::expr_nary::build_balanced_mul(ctx, &[derivative_core, numerator_function])
        }
        None => derivative_core,
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let core_denominator = match shape {
        SqrtChainShape::DenominatorSquare(denominator_fn) => {
            let denominator_arg = ctx.call_builtin(denominator_fn, vec![sqrt_radicand]);
            let two = ctx.num(2);
            let denominator_square = ctx.add(Expr::Pow(denominator_arg, two));
            cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_square])
        }
        SqrtChainShape::NumeratorOverDenominatorSquare { denominator_fn, .. } => {
            let denominator_arg = ctx.call_builtin(denominator_fn, vec![sqrt_radicand]);
            let two = ctx.num(2);
            let denominator_square = ctx.add(Expr::Pow(denominator_arg, two));
            cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_square])
        }
        SqrtChainShape::ReciprocalTrigProduct { .. } => sqrt_radicand,
        _ => sqrt_radicand,
    };
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let raw_gap = Polynomial::one(radicand_poly.var.clone())
        .sub(&radicand_poly)
        .to_expr(ctx);
    let (gap, gap_content) = primitive_positive_gap(ctx, raw_gap);
    let (gap, numerator) = if gap_content.is_one()
        || exact_positive_rational_sqrt_for_calculus_presentation(&gap_content).is_some()
    {
        let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            reciprocal_positive_rational(&gap_content),
            numerator,
        );
        (gap, numerator)
    } else {
        (raw_gap, numerator)
    };
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn bounded_inverse_trig_sqrt_affine_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() != 1 || denominator_poly.degree() != 1 {
        return None;
    }

    let numerator_slope = numerator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_slope = denominator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if numerator_slope.is_zero() || numerator_slope != denominator_slope {
        return None;
    }

    let numerator_constant = numerator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let denominator_constant = denominator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let quotient_gap = denominator_constant - numerator_constant;
    if !quotient_gap.is_positive() {
        return None;
    }

    let coefficient = derivative_sign * numerator_slope * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        quotient_gap,
        numerator,
    );

    let sqrt_numerator = ctx.call_builtin(BuiltinFn::Sqrt, vec![num]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[den, sqrt_numerator]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn bounded_inverse_trig_reciprocal_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => -BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (radicand, argument_scale) =
        reciprocal_sqrt_like_arg_for_calculus_presentation(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * argument_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let gap = add_rational_for_calculus_presentation(
        ctx,
        radicand,
        -(argument_scale.clone() * argument_scale),
    );
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn reciprocal_sqrt_like_arg_for_calculus_presentation(
    ctx: &Context,
    arg: ExprId,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(arg) {
        Expr::Div(num, den) => {
            let scale = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            if scale.is_zero() {
                return None;
            }
            Some((calculus_sqrt_like_radicand(ctx, *den)?, scale))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut radicand = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, arg) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 2.into()))
                    || radicand.replace(*base).is_some()
                {
                    return None;
                }
            }
            if scale.is_zero() {
                return None;
            }
            Some((radicand?, scale))
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            Some((*base, BigRational::one()))
        }
        _ => None,
    }
}

fn constant_scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative =
        bounded_inverse_trig_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn bounded_inverse_trig_self_normalized_projection_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (numerator_arg, denominator_radicand) =
        bounded_inverse_trig_self_normalized_projection_arg(ctx, args[0])?;
    let numerator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, numerator_arg, var_name)?;
    if numerator_poly.degree() > 2 {
        return None;
    }
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_radicand, var_name)?;
    let gap_poly = denominator_poly.sub(&numerator_poly.mul(&numerator_poly));
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_constant = gap_poly.coeffs.first().cloned()?;
    if !gap_constant.is_positive() {
        return None;
    }

    let mut derivative_poly = numerator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let mut derivative_sign = derivative_sign;
    if derivative_poly.leading_coeff().is_negative() {
        derivative_poly = derivative_poly.neg();
        derivative_sign = -derivative_sign;
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let numerator = signed_numerator_for_calculus_presentation(
        ctx,
        derivative_sign * derivative_content,
        derivative_core,
    );
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        gap_constant,
        numerator,
    );
    let denominator = denominator_radicand;
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(compact)
}

fn bounded_inverse_trig_self_normalized_projection_arg(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg).clone() {
        Expr::Neg(inner) => {
            let (numerator, denominator_radicand) =
                bounded_inverse_trig_self_normalized_projection_arg(ctx, inner)?;
            let numerator = ctx.add(Expr::Neg(numerator));
            Some((numerator, denominator_radicand))
        }
        Expr::Div(numerator, denominator) => {
            Some((numerator, extract_square_root_base(ctx, denominator)?))
        }
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
            let mut numerator_factors = Vec::new();
            let mut denominator_radicand = None;

            for factor in factors {
                match ctx.get(factor) {
                    Expr::Pow(base, exp)
                        if cas_ast::views::as_rational_const(ctx, *exp, 8)
                            == Some(BigRational::new((-1).into(), 2.into())) =>
                    {
                        if denominator_radicand.replace(*base).is_some() {
                            return None;
                        }
                    }
                    _ => numerator_factors.push(factor),
                }
            }

            let denominator_radicand = denominator_radicand?;
            let numerator = match numerator_factors.as_slice() {
                [] => ctx.num(1),
                [only] => *only,
                [first, rest @ ..] => {
                    let mut product = *first;
                    for factor in rest {
                        product = ctx.add(Expr::Mul(product, *factor));
                    }
                    product
                }
            };

            Some((numerator, denominator_radicand))
        }
        _ => None,
    }
}

fn scaled_sqrt_radicand_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (sign, expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (-BigRational::one(), *inner),
        _ => (BigRational::one(), expr),
    };

    if let Some(radicand) = extract_square_root_base(ctx, expr) {
        return Some((sign, radicand));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = sign;
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }

        let factor_radicand = extract_square_root_base(ctx, factor)?;
        if radicand.replace(factor_radicand).is_some() {
            return None;
        }
    }

    Some((scale, radicand?))
}

fn shifted_unit_interval_sqrt_arg_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let two = BigRational::from_integer(2.into());
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            if cas_ast::views::as_rational_const(ctx, *right, 8) == Some(BigRational::one()) {
                let (scale, radicand) = scaled_sqrt_radicand_for_calculus_presentation(ctx, *left)?;
                if scale == two {
                    return Some((radicand, BigRational::one()));
                }
            }

            if cas_ast::views::as_rational_const(ctx, *left, 8) == Some(BigRational::one()) {
                let (scale, radicand) =
                    scaled_sqrt_radicand_for_calculus_presentation(ctx, *right)?;
                if scale == two {
                    return Some((radicand, -BigRational::one()));
                }
            }

            None
        }
        _ => None,
    }
}

fn unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (radicand, arg_sign) =
        shifted_unit_interval_sqrt_arg_for_calculus_presentation(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);

    let coefficient =
        derivative_sign * arg_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let gap = ctx.add(Expr::Sub(sqrt_radicand, radicand));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn constant_scaled_unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
        ctx, inner, var_name,
    )?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn bounded_inverse_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() {
        return None;
    }
    let primitive_arg_poly = if arg_content.is_one() {
        arg_poly
    } else {
        arg_poly.div_scalar(&arg_content)
    };
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let numerator = scale_expr_for_calculus_presentation(
        ctx,
        derivative_sign * derivative_content * content_num,
        derivative_core,
    );

    let primitive_arg = primitive_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr(ctx, primitive_arg);
    let raw_gap = if arg_content.is_one() {
        let one = ctx.num(1);
        ctx.add(Expr::Sub(one, primitive_arg_sq))
    } else {
        let content_num_sq =
            BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
        let content_den_sq =
            BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
        let scaled_arg_sq =
            scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
        let den_sq = rational_const_for_calculus_presentation(ctx, content_den_sq);
        ctx.add(Expr::Sub(den_sq, scaled_arg_sq))
    };
    let (gap, gap_content) = primitive_positive_gap(ctx, raw_gap);
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        reciprocal_positive_rational(&gap_content),
        numerator,
    );
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn arctan_rational_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if !matches!(
        ctx.builtin_of(*fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, args[0], var_name)?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() || arg_content.is_integer() {
        return None;
    }

    let primitive_arg_poly = arg_poly.div_scalar(&arg_content);
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    if derivative_poly.degree() != 0 {
        return None;
    }

    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let content_den = BigRational::from_integer(arg_content.denom().clone());
    let numerator_coeff = derivative_coeff * content_num.clone() * content_den.clone();
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);

    let square_arg_poly = if primitive_arg_poly.leading_coeff().is_negative() {
        primitive_arg_poly.neg()
    } else {
        primitive_arg_poly.clone()
    };
    let square_arg = square_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr_for_compact_gap_presentation(ctx, square_arg);
    let content_num_sq =
        BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
    let content_den_sq =
        BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
    let scaled_arg_sq = scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
    let denominator_constant = rational_const_for_calculus_presentation(ctx, content_den_sq);
    let denominator = ctx.add(Expr::Add(scaled_arg_sq, denominator_constant));

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn atanh_rational_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, args[0], var_name)?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() || arg_content.is_integer() {
        return None;
    }

    let primitive_arg_poly = arg_poly.div_scalar(&arg_content);
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() || derivative_poly.degree() != 0 {
        return None;
    }

    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let content_den = BigRational::from_integer(arg_content.denom().clone());
    let numerator_coeff = derivative_coeff * content_num.clone() * content_den.clone();
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);

    let square_arg_poly = if primitive_arg_poly.leading_coeff().is_negative() {
        primitive_arg_poly.neg()
    } else {
        primitive_arg_poly.clone()
    };
    let square_arg = square_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr_for_compact_gap_presentation(ctx, square_arg);
    let content_num_sq =
        BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
    let scaled_arg_sq = scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
    let content_den_sq =
        BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
    let denominator_constant = rational_const_for_calculus_presentation(ctx, content_den_sq);
    let denominator = ctx.add(Expr::Sub(denominator_constant, scaled_arg_sq));

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn asinh_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Asinh) {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() {
        return None;
    }
    let primitive_arg_poly = if arg_content.is_one() {
        arg_poly
    } else {
        arg_poly.div_scalar(&arg_content)
    };
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let numerator = scale_expr_for_calculus_presentation(
        ctx,
        derivative_content * content_num,
        derivative_core,
    );

    let one = ctx.num(1);
    let square_arg_poly =
        if !arg_content.is_one() && primitive_arg_poly.leading_coeff().is_negative() {
            primitive_arg_poly.neg()
        } else {
            primitive_arg_poly.clone()
        };
    let primitive_arg = square_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr_for_compact_gap_presentation(ctx, primitive_arg);
    let radicand = if arg_content.is_one() {
        ctx.add(Expr::Add(primitive_arg_sq, one))
    } else {
        let content_num_sq =
            BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
        let content_den_sq =
            BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
        let scaled_arg_sq =
            scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
        let den_sq = rational_const_for_calculus_presentation(ctx, content_den_sq);
        ctx.add(Expr::Add(scaled_arg_sq, den_sq))
    };
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn acosh_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let arg = args[0];
    if let Some(compact) = acosh_fractional_affine_derivative_presentation(ctx, arg, var_name) {
        return Some(compact);
    }

    let slope = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let one = ctx.num(1);
    let numerator = scale_expr_for_calculus_presentation(ctx, slope, one);
    let lower_branch = add_rational_for_calculus_presentation(ctx, arg, -BigRational::one());
    let upper_branch = add_one_for_calculus_presentation(ctx, arg);
    let sqrt_lower = ctx.call_builtin(BuiltinFn::Sqrt, vec![lower_branch]);
    let sqrt_upper = ctx.call_builtin(BuiltinFn::Sqrt, vec![upper_branch]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_lower, sqrt_upper]);
    let result = ctx.add(Expr::Div(numerator, denominator));

    Some((
        result,
        vec![crate::ImplicitCondition::Positive(lower_branch)],
    ))
}

fn acosh_fractional_affine_derivative_presentation(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    if arg_poly.degree() != 1 {
        return None;
    }

    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() || arg_content.is_integer() {
        return None;
    }

    let primitive_arg_poly = arg_poly.div_scalar(&arg_content);
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() || derivative_poly.degree() != 0 {
        return None;
    }

    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let content_den = BigRational::from_integer(arg_content.denom().clone());
    let scaled_arg_poly =
        scale_polynomial_for_calculus_presentation(&primitive_arg_poly, &content_num);
    let denominator_gap =
        Polynomial::new(vec![content_den.clone()], primitive_arg_poly.var.clone());
    let lower_poly = scaled_arg_poly.sub(&denominator_gap);
    let upper_poly = scaled_arg_poly.add(&denominator_gap);
    let lower_branch = lower_poly.to_expr(ctx);
    let upper_branch = upper_poly.to_expr(ctx);

    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let one = ctx.num(1);
    let mut numerator_coeff = derivative_coeff * content_num;
    let mut denominator_coeff = BigRational::one();
    let denominator = if let Some((primitive_product, shared_content)) =
        shared_positive_content_sqrt_product_for_calculus_presentation(
            ctx,
            lower_branch,
            upper_branch,
        ) {
        (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / shared_content))?;
        primitive_product
    } else {
        let sqrt_lower = ctx.call_builtin(BuiltinFn::Sqrt, vec![lower_branch]);
        let sqrt_upper = ctx.call_builtin(BuiltinFn::Sqrt, vec![upper_branch]);
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_lower, sqrt_upper])
    };
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let denominator = if denominator_coeff == BigRational::one() {
        denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, denominator])
    };
    let result = ctx.add(Expr::Div(numerator, denominator));

    Some((
        result,
        vec![crate::ImplicitCondition::Positive(lower_branch)],
    ))
}

fn acosh_strictly_positive_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    if arg_poly.degree() != 2 {
        return None;
    }

    let one_poly = Polynomial::one(arg_poly.var.clone());
    let lower_poly = arg_poly.sub(&one_poly);
    if !polynomial_is_strictly_positive_everywhere(&lower_poly) {
        return None;
    }

    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&derivative_content)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let lower_branch = lower_poly.to_expr(ctx);
    let upper_branch = arg_poly.add(&one_poly).to_expr(ctx);
    let sqrt_lower = ctx.call_builtin(BuiltinFn::Sqrt, vec![lower_branch]);
    let sqrt_upper = ctx.call_builtin(BuiltinFn::Sqrt, vec![upper_branch]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_lower, sqrt_upper]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some((cas_ast::hold::wrap_hold(ctx, compact), Vec::new()))
}

fn asinh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Asinh) {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if !asinh_sqrt_presentation_safe_radicand(&radicand_poly) {
        return None;
    }
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (mut numerator_coeff, mut denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand_plus_one]);
    let core_denominator = if let Some((primitive_product, shared_content)) =
        shared_positive_content_sqrt_product_for_calculus_presentation(
            ctx,
            radicand,
            radicand_plus_one,
        ) {
        (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / (denominator_coeff * shared_content)))?;
        primitive_product
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap])
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn constant_scaled_asinh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = asinh_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn asinh_sqrt_presentation_safe_radicand(poly: &Polynomial) -> bool {
    match poly.degree() {
        1 => poly.coeffs.get(1).is_some_and(|linear| !linear.is_zero()),
        2 => polynomial_is_strictly_positive_everywhere(poly),
        _ => false,
    }
}

fn shared_positive_content_sqrt_product_for_calculus_presentation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, BigRational)> {
    let (left_primitive, left_content) =
        split_polynomial_content_for_calculus_presentation(ctx, left);
    let (right_primitive, right_content) =
        split_polynomial_content_for_calculus_presentation(ctx, right);

    if !left_content.is_one() && left_content.is_positive() && right_content == left_content {
        let left_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![left_primitive]);
        let right_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![right_primitive]);
        let product = cas_math::expr_nary::build_balanced_mul(ctx, &[left_sqrt, right_sqrt]);
        return Some((product, left_content));
    }

    shared_positive_denominator_sqrt_product_for_calculus_presentation(ctx, left, right)
}

fn shared_positive_denominator_sqrt_product_for_calculus_presentation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, BigRational)> {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };
    let left_poly = multipoly_from_expr(ctx, left, &budget).ok()?;
    let right_poly = multipoly_from_expr(ctx, right, &budget).ok()?;
    if left_poly.vars != right_poly.vars {
        return None;
    }

    let left_denominator = multipoly_denominator_lcm_for_calculus_presentation(&left_poly);
    let right_denominator = multipoly_denominator_lcm_for_calculus_presentation(&right_poly);
    if left_denominator != right_denominator || left_denominator <= BigInt::one() {
        return None;
    }

    let denominator = BigRational::from_integer(left_denominator);
    let left_primitive = left_poly.mul_scalar(&denominator);
    let right_primitive = right_poly.mul_scalar(&denominator);
    if !multipoly_has_integer_coefficients_for_calculus_presentation(&left_primitive)
        || !multipoly_has_integer_coefficients_for_calculus_presentation(&right_primitive)
    {
        return None;
    }

    let left_expr = multipoly_to_expr(&left_primitive, ctx);
    let right_expr = multipoly_to_expr(&right_primitive, ctx);
    let left_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![left_expr]);
    let right_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![right_expr]);
    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[left_sqrt, right_sqrt]);
    Some((product, BigRational::one() / denominator))
}

fn multipoly_denominator_lcm_for_calculus_presentation(
    poly: &cas_math::multipoly::MultiPoly,
) -> BigInt {
    poly.terms
        .iter()
        .fold(BigInt::one(), |acc, (coeff, _)| acc.lcm(coeff.denom()))
}

fn multipoly_has_integer_coefficients_for_calculus_presentation(
    poly: &cas_math::multipoly::MultiPoly,
) -> bool {
    poly.terms.iter().all(|(coeff, _)| coeff.denom().is_one())
}

fn preserve_atanh_sqrt_open_interval_gap_orientation(poly: &Polynomial) -> bool {
    poly.degree() == 1
        && poly
            .coeffs
            .first()
            .is_some_and(|constant| constant.is_positive())
        && poly.leading_coeff().is_negative()
}

fn atanh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() != 1 {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut numerator_sign = BigRational::one();
    let one = Polynomial::one(radicand_poly.var.clone());
    let mut gap_poly = one.sub(&radicand_poly);
    if gap_poly.leading_coeff().is_negative()
        && !preserve_atanh_sqrt_open_interval_gap_orientation(&gap_poly)
    {
        gap_poly = gap_poly.neg();
        numerator_sign = -numerator_sign;
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = numerator_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let gap = gap_poly.to_expr(ctx);
    let (numerator_coeff, denominator_coeff, gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            gap,
        );
    let (sqrt_radicand, numerator_coeff, denominator_coeff) =
        if let Some((compact_sqrt, compact_numerator_coeff, compact_denominator_coeff)) =
            compact_rational_monomial_sqrt_denominator_for_calculus_presentation(
                ctx,
                &radicand_poly,
                numerator_coeff.clone(),
                denominator_coeff.clone(),
            )
        {
            (
                compact_sqrt,
                compact_numerator_coeff,
                compact_denominator_coeff,
            )
        } else {
            (
                ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]),
                numerator_coeff,
                denominator_coeff,
            )
        };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn atanh_sqrt_affine_quotient_positive_gap_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };

    let numerator_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator_poly = Polynomial::from_expr(ctx, den, var_name).ok()?;
    if numerator_poly.degree() != 1 || denominator_poly.degree() != 1 {
        return None;
    }

    let gap_poly = denominator_poly.sub(&numerator_poly);
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_value = gap_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !gap_value.is_positive() {
        return None;
    }

    let wronskian = numerator_poly
        .derivative()
        .mul(&denominator_poly)
        .sub(&numerator_poly.mul(&denominator_poly.derivative()));
    if wronskian.degree() != 0 {
        return None;
    }
    let wronskian_value = wronskian
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if wronskian_value.is_zero() {
        return Some(ctx.num(0));
    }

    let coefficient = wronskian_value / (BigRational::from_integer(2.into()) * gap_value);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let sqrt_num = ctx.call_builtin(BuiltinFn::Sqrt, vec![num]);
    let sqrt_den = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_num, sqrt_den]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn compact_rational_monomial_sqrt_denominator_for_calculus_presentation(
    ctx: &mut Context,
    radicand_poly: &Polynomial,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
) -> Option<(ExprId, BigRational, BigRational)> {
    if !denominator_coeff.is_integer()
        || !denominator_coeff.is_positive()
        || denominator_coeff.is_one()
    {
        return None;
    }
    if radicand_poly.degree() != 1
        || radicand_poly
            .coeffs
            .first()
            .is_some_and(|constant| !constant.is_zero())
    {
        return None;
    }

    let slope = radicand_poly.coeffs.get(1)?;
    let radicand_denominator = slope.denom().clone();
    if radicand_denominator <= BigInt::one() {
        return None;
    }

    let radicand_denominator_rational = BigRational::from_integer(radicand_denominator.clone());
    let compact_slope = slope * &radicand_denominator_rational * &radicand_denominator_rational;
    if !compact_slope.is_integer() {
        return None;
    }

    let compact_poly = Polynomial::new(
        vec![BigRational::zero(), compact_slope],
        radicand_poly.var.clone(),
    );
    let compact_radicand = compact_poly.to_expr(ctx);
    let compact_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![compact_radicand]);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(
        &(numerator_coeff * radicand_denominator_rational / denominator_coeff),
    )?;
    Some((compact_sqrt, numerator_coeff, denominator_coeff))
}

fn constant_scaled_atanh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = atanh_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn acosh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() > 2 {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (mut numerator_coeff, mut denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_minus_one =
        add_rational_for_calculus_presentation(ctx, radicand, -BigRational::one());
    let raw_sqrt_radicand_minus_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand_minus_one]);
    let sqrt_radicand_minus_one =
        try_rewrite_simplify_square_root_expr(ctx, raw_sqrt_radicand_minus_one)
            .map(|rewrite| rewrite.rewritten)
            .unwrap_or(raw_sqrt_radicand_minus_one);
    let core_denominator = if let Some((primitive_product, shared_content)) =
        shared_positive_content_sqrt_product_for_calculus_presentation(
            ctx,
            radicand,
            radicand_minus_one,
        ) {
        (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / (denominator_coeff * shared_content)))?;
        primitive_product
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_radicand_minus_one])
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn constant_scaled_acosh_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = acosh_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn acosh_sqrt_shifted_quadratic_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() != 2
        || radicand_poly
            .coeffs
            .get(1)
            .is_none_or(|linear| linear.is_zero())
    {
        return None;
    }

    acosh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
}

fn acosh_polynomial_over_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    if num_poly.degree() > 2 {
        return None;
    }

    let derivative_poly = num_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);

    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let raw_gap = ctx.add(Expr::Sub(num_square, radicand));
    let expanded_gap = expanded_polynomial_expr_for_calculus_presentation(ctx, raw_gap, 4);
    let (primitive_gap, gap_content) = primitive_positive_gap(ctx, expanded_gap);
    let (gap, numerator) = if gap_content.is_one()
        || exact_positive_rational_sqrt_for_calculus_presentation(&gap_content).is_some()
    {
        let numerator =
            signed_numerator_for_calculus_presentation(ctx, derivative_content, derivative_core);
        (raw_gap, numerator)
    } else {
        let numerator =
            signed_numerator_for_calculus_presentation(ctx, derivative_content, derivative_core);
        let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            reciprocal_positive_rational(&gap_content),
            numerator,
        );
        (primitive_gap, numerator)
    };
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let result = ctx.add(Expr::Div(numerator, sqrt_gap));

    let branch_gap = if let Some(root) =
        exact_positive_rational_sqrt_for_calculus_presentation(&radicand_value)
    {
        add_rational_for_calculus_presentation(ctx, num, -root)
    } else {
        let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        ctx.add(Expr::Sub(num, sqrt_radicand))
    };

    Some((
        result,
        vec![
            crate::ImplicitCondition::Positive(gap),
            crate::ImplicitCondition::Positive(branch_gap),
        ],
    ))
}

fn constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let (derivative, required_conditions) =
        acosh_polynomial_over_sqrt_derivative_presentation(ctx, inner, var_name)?;
    let result = scale_compact_derivative_by_rational(ctx, derivative, scale);

    Some((result, required_conditions))
}

fn inverse_reciprocal_trig_sqrt_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() != 1 {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let derivative_core_factor = cas_ast::views::as_rational_const(ctx, derivative_core, 8)?;
    if derivative_core_factor.is_zero() {
        return None;
    }

    let coefficient =
        sign * derivative_content * derivative_core_factor * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);

    let gap = add_rational_for_calculus_presentation(ctx, radicand, -BigRational::one());
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn inverse_reciprocal_trig_sqrt_quadratic_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if radicand_poly.degree() != 2 {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let gap = add_rational_for_calculus_presentation(ctx, radicand, -BigRational::one());
    let raw_sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let sqrt_gap = try_rewrite_simplify_square_root_expr(ctx, raw_sqrt_gap)
        .map(|rewrite| rewrite.rewritten)
        .unwrap_or(raw_sqrt_gap);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn inverse_reciprocal_trig_affine_abs_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let derivative_scale = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let numerator = rational_const_for_calculus_presentation(ctx, sign * derivative_scale);
    let arg_sq = squared_expr(ctx, arg);
    let one = ctx.num(1);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let (gap, gap_content) = primitive_positive_gap(ctx, raw_gap);
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        reciprocal_positive_rational(&gap_content),
        numerator,
    );
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[abs_arg, sqrt_gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn inverse_reciprocal_trig_affine_abs_required_conditions(
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

fn strictly_positive_quadratic_on_reals(poly: &Polynomial) -> bool {
    if poly.degree() != 2 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !a.is_positive() {
        return false;
    }

    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let discriminant = &b * &b - four * a * c;
    discriminant.is_negative()
}

fn square_of_strictly_positive_quadratic_arg(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(arg) else {
        return None;
    };
    let two = BigRational::from_integer(2.into());
    if cas_ast::views::as_rational_const(ctx, *exp, 8).as_ref() != Some(&two) {
        return None;
    }

    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, *base, var_name)?;
    strictly_positive_quadratic_on_reals(&base_poly).then_some(*base)
}

fn positive_rational_denominator_scaled_base(
    ctx: &mut Context,
    base: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational, ExprId)> {
    if let Expr::Div(numerator, denominator) = ctx.get(base) {
        let scale = cas_ast::views::as_rational_const(ctx, *denominator, 8)?;
        if !scale.is_positive() {
            return None;
        }

        let numerator_poly =
            polynomial_radicand_for_calculus_presentation(ctx, *numerator, var_name)?;
        return strictly_positive_quadratic_on_reals(&numerator_poly)
            .then_some((*numerator, scale, base));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, base);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut non_rational_factor = None;
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            if !value.is_positive() {
                return None;
            }
            scale *= value;
            continue;
        }

        if non_rational_factor.replace(factor).is_some() {
            return None;
        }
    }

    if scale.is_one() {
        return None;
    }
    let core = non_rational_factor?;
    let core_poly = polynomial_radicand_for_calculus_presentation(ctx, core, var_name)?;
    if !strictly_positive_quadratic_on_reals(&core_poly) {
        return None;
    }

    let compact_scaled_base = if scale.numer() == &BigInt::one() {
        let denominator = rational_const_for_calculus_presentation(
            ctx,
            BigRational::from_integer(scale.denom().clone()),
        );
        ctx.add(Expr::Div(core, denominator))
    } else {
        scale_expr_for_calculus_presentation(ctx, scale.clone(), core)
    };

    Some((core, BigRational::one() / scale, compact_scaled_base))
}

fn expanded_polynomial_expr_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    max_total_degree: u32,
) -> ExprId {
    let budget = PolyBudget {
        max_terms: 16,
        max_total_degree,
        max_pow_exp: max_total_degree,
    };

    multipoly_from_expr(ctx, expr, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(expr)
}

fn inverse_reciprocal_trig_positive_quadratic_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    if !strictly_positive_quadratic_on_reals(&arg_poly) {
        return None;
    }

    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, sign * derivative_content, derivative_core);

    let arg_sq = squared_expr(ctx, arg);
    let one = ctx.num(1);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let gap = expanded_polynomial_expr_for_calculus_presentation(ctx, raw_gap, 4);
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[arg, sqrt_gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    if factors.len() < 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().enumerate() {
        let Some(scale_radicand) = extract_square_root_base(ctx, *sqrt_factor) else {
            continue;
        };
        let scale_radicand_value = cas_ast::views::as_rational_const(ctx, scale_radicand, 8)?;
        if !scale_radicand_value.is_positive() {
            return None;
        }

        let mut rational_scale = BigRational::one();
        let mut numerator_factors = Vec::new();
        for (factor_index, factor) in factors.iter().enumerate() {
            if factor_index == sqrt_index {
                continue;
            }

            if let Some(value) = cas_ast::views::as_rational_const(ctx, *factor, 8) {
                rational_scale *= value;
            } else {
                numerator_factors.push(*factor);
            }
        }

        if !rational_scale.is_positive() || numerator_factors.is_empty() {
            continue;
        }

        let scale_square = &rational_scale * &rational_scale;
        let equivalent_denominator_radicand =
            reciprocal_positive_rational(&(scale_square * scale_radicand_value));
        let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
        let radicand =
            rational_const_for_calculus_presentation(ctx, equivalent_denominator_radicand);
        return Some((numerator, radicand));
    }

    None
}

fn inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])
        .or_else(|| sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation(ctx, args[0]))?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    if !strictly_positive_quadratic_on_reals(&num_poly) {
        return None;
    }

    let d_num = num_poly.derivative().to_expr(ctx);
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (d_num_core, d_num_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_num);
    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let derivative_factor =
        signed_numerator_for_calculus_presentation(ctx, sign * d_num_content, d_num_core);
    let radicand_numer = BigRational::from_integer(radicand_value.numer().clone());
    let radicand_denom = BigRational::from_integer(radicand_value.denom().clone());
    let numerator = if radicand_numer.is_one() {
        derivative_factor
    } else {
        let compact_numer = rational_const_for_calculus_presentation(ctx, radicand_numer.clone());
        multiply_by_sqrt_factor_for_calculus_presentation(ctx, derivative_factor, compact_numer)
    };
    let scaled_num_square = if radicand_denom.is_one() {
        num_square
    } else {
        scale_expr_for_calculus_presentation(ctx, radicand_denom, num_square)
    };
    let compact_numer = rational_const_for_calculus_presentation(ctx, radicand_numer);
    let raw_gap = ctx.add(Expr::Sub(scaled_num_square, compact_numer));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![raw_gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[num, sqrt_gap]);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn inverse_reciprocal_trig_positive_quadratic_square_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let base = square_of_strictly_positive_quadratic_arg(ctx, args[0], var_name)?;
    let derivative = polynomial_derivative_expr_for_calculus_presentation(ctx, base, var_name)?;
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let (denominator_base, numerator_scale, gap_base) = positive_rational_denominator_scaled_base(
        ctx, base, var_name,
    )
    .unwrap_or((base, BigRational::one(), base));
    let numerator = signed_numerator_for_calculus_presentation(
        ctx,
        sign * derivative_content * BigRational::from_integer(2.into()) * numerator_scale,
        derivative_core,
    );

    let four = ctx.num(4);
    let base_fourth = ctx.add(Expr::Pow(gap_base, four));
    let one = ctx.num(1);
    let gap = ctx.add(Expr::Sub(base_fourth, one));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_base, sqrt_gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn compact_positive_quadratic_square_derivative_result(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let result = unwrap_internal_hold_for_calculus(ctx, result);
    let Expr::Div(numerator, denominator) = ctx.get(result).clone() else {
        return None;
    };
    let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
    if numerator_value.is_zero() {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(denominator).clone() else {
        return None;
    };
    let two = BigRational::from_integer(2.into());
    if cas_ast::views::as_rational_const(ctx, exp, 8).as_ref() != Some(&two) {
        return None;
    }

    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
    if !strictly_positive_quadratic_on_reals(&base_poly) {
        return None;
    }

    let (base_core, base_content) = split_polynomial_content_for_calculus_presentation(ctx, base);
    let compact_numerator_value = numerator_value / (&base_content * &base_content);
    let numerator = rational_const_for_calculus_presentation(ctx, compact_numerator_value);
    let two_expr = ctx.num(2);
    let denominator = ctx.add(Expr::Pow(base_core, two_expr));
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn positive_quadratic_square_derivative_result_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let result = differentiate(ctx, target, var_name)?;
    compact_positive_quadratic_square_derivative_result(ctx, result, var_name)
}

fn rational_over_matching_denominator_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    denominator_poly: &Polynomial,
    var_name: &str,
) -> Option<BigRational> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
    let (observed_denominator_core, observed_denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator);
    if observed_denominator_content.is_zero() {
        return None;
    }
    let observed_denominator =
        polynomial_radicand_for_calculus_presentation(ctx, observed_denominator_core, var_name)?;
    (observed_denominator == *denominator_poly)
        .then_some(numerator_value / observed_denominator_content)
}

fn positive_quadratic_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(target).clone() else {
        return None;
    };
    let (mut denominator_core, mut denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator);
    if denominator_content.is_zero() {
        return None;
    }
    let mut denominator_core_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_core, var_name)?;
    if !strictly_positive_quadratic_on_reals(&denominator_core_poly) {
        let negated_denominator_core_poly = Polynomial::new(
            denominator_core_poly
                .coeffs
                .iter()
                .map(|coeff| -coeff.clone())
                .collect(),
            denominator_core_poly.var.clone(),
        );
        if !strictly_positive_quadratic_on_reals(&negated_denominator_core_poly) {
            return None;
        }
        denominator_core = negated_denominator_core_poly.to_expr(ctx);
        denominator_content = -denominator_content;
        denominator_core_poly = negated_denominator_core_poly;
    }

    let numerator_derivative = differentiate(ctx, numerator, var_name)?;
    let numerator_derivative_scale = rational_over_matching_denominator_for_calculus_presentation(
        ctx,
        numerator_derivative,
        &denominator_core_poly,
        var_name,
    )?;
    if numerator_derivative_scale.is_zero() {
        return None;
    }

    let denominator_derivative = denominator_core_poly.derivative().to_expr(ctx);
    let (denominator_derivative_core, denominator_derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_derivative);
    let reciprocal_content = BigRational::one() / denominator_content;
    let scaled_numerator_derivative = rational_const_for_calculus_presentation(
        ctx,
        numerator_derivative_scale * reciprocal_content.clone(),
    );
    let scaled_denominator_derivative_coeff = denominator_derivative_content * reciprocal_content;
    let compact_numerator = if scaled_denominator_derivative_coeff.is_negative() {
        let scaled_denominator_derivative = scale_expr_for_calculus_presentation(
            ctx,
            -scaled_denominator_derivative_coeff,
            denominator_derivative_core,
        );
        let quotient_term = cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[scaled_denominator_derivative, numerator],
        );
        ctx.add(Expr::Add(scaled_numerator_derivative, quotient_term))
    } else {
        let scaled_denominator_derivative = scale_expr_for_calculus_presentation(
            ctx,
            scaled_denominator_derivative_coeff,
            denominator_derivative_core,
        );
        let quotient_term = cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[scaled_denominator_derivative, numerator],
        );
        ctx.add(Expr::Sub(scaled_numerator_derivative, quotient_term))
    };
    let two = ctx.num(2);
    let compact_denominator = ctx.add(Expr::Pow(denominator_core, two));
    let compact = ctx.add(Expr::Div(compact_numerator, compact_denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn variable_base_constant_argument_log_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Log) || args.len() != 2 {
        return None;
    }

    let base = args[0];
    let arg = args[1];
    if !contains_named_var(ctx, base, var_name) || contains_named_var(ctx, arg, var_name) {
        return None;
    }
    if variable_named(ctx, base, var_name) {
        return None;
    }

    let d_base = polynomial_derivative_expr_for_calculus_presentation(ctx, base, var_name)
        .or_else(|| differentiate(ctx, base, var_name))?;
    if cas_ast::views::as_rational_const(ctx, d_base, 8).is_some_and(|value| value.is_zero()) {
        return None;
    }

    let ln_arg = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let two = ctx.num(2);
    let ln_base_sq = ctx.add(Expr::Pow(ln_base, two));
    let (d_base_core, d_base_coeff) =
        split_polynomial_content_for_calculus_presentation(ctx, d_base);
    let numerator_core = if cas_ast::views::as_rational_const(ctx, d_base_core, 8)
        .is_some_and(|value| value.is_one())
    {
        ln_arg
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[ln_arg, d_base_core])
    };
    let numerator = if d_base_coeff.is_one() {
        ctx.add(Expr::Neg(numerator_core))
    } else {
        scale_expr_for_calculus_presentation(ctx, -d_base_coeff, numerator_core)
    };
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[base, ln_base_sq]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn inverse_tangent_direct_trig_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let outer_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => -BigRational::one(),
        _ => return None,
    };

    let trig_arg = args[0];
    let Expr::Function(trig_fn_id, trig_args) = ctx.get(trig_arg).clone() else {
        return None;
    };
    if trig_args.len() != 1
        || !matches!(
            ctx.builtin_of(trig_fn_id),
            Some(BuiltinFn::Sin | BuiltinFn::Cos)
        )
    {
        return None;
    }
    let inner_poly = Polynomial::from_expr(ctx, trig_args[0], var_name).ok()?;
    if inner_poly.degree() != 1 {
        return None;
    }

    let inner_derivative = inner_poly.derivative().to_expr(ctx);
    let mut scale = cas_ast::views::as_rational_const(ctx, inner_derivative, 8)?;
    if scale.is_zero() {
        return None;
    }
    scale *= outer_sign;
    let numerator_core = match ctx.builtin_of(trig_fn_id) {
        Some(BuiltinFn::Sin) => ctx.call_builtin(BuiltinFn::Cos, vec![trig_args[0]]),
        Some(BuiltinFn::Cos) => {
            scale = -scale;
            ctx.call_builtin(BuiltinFn::Sin, vec![trig_args[0]])
        }
        _ => return None,
    };
    let numerator = signed_numerator_for_calculus_presentation(ctx, scale, numerator_core);

    let two = ctx.num(2);
    let trig_square = ctx.add(Expr::Pow(trig_arg, two));
    let one = ctx.num(1);
    let denominator = ctx.add(Expr::Add(trig_square, one));
    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn supported_sqrt_shift_constant_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        let value = cas_ast::views::as_rational_const(ctx, *right, 8)?;
        let shift = -value;
        if supported_log_sqrt_shift(&shift) {
            return Some((*left, shift));
        }
    }

    let terms = cas_math::expr_nary::add_leaves(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let first_constant = cas_ast::views::as_rational_const(ctx, terms[0], 8);
    let second_constant = cas_ast::views::as_rational_const(ctx, terms[1], 8);
    match (first_constant, second_constant) {
        (Some(value), None) if supported_log_sqrt_shift(&value) => Some((terms[1], value)),
        (None, Some(value)) if supported_log_sqrt_shift(&value) => Some((terms[0], value)),
        _ => None,
    }
}

fn supported_log_sqrt_shift(value: &BigRational) -> bool {
    !value.is_zero()
}

fn ln_sqrt_shift_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let (sqrt_arg, shift) = supported_sqrt_shift_constant_parts(ctx, args[0])?;
    let radicand = extract_square_root_base(ctx, sqrt_arg)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let shifted_sqrt = add_rational_for_calculus_presentation(ctx, sqrt_radicand, shift);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let (sqrt_arg, shift) = supported_sqrt_shift_constant_parts(ctx, args[0])?;
    if !shift.is_positive() {
        return None;
    }
    let radicand = extract_square_root_base(ctx, sqrt_arg)?;
    if polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name).is_some() {
        return None;
    }

    let derivative = differentiate(ctx, radicand, var_name)?;
    let derivative =
        remove_unit_log_e_factor_for_calculus_presentation(ctx, derivative).unwrap_or(derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), Vec::new()));
    }
    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let shifted_sqrt = add_rational_for_calculus_presentation(ctx, sqrt_radicand, shift);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        vec![crate::ImplicitCondition::Positive(radicand)],
    ))
}

fn remove_unit_log_e_factor_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    if is_ln_e_for_calculus_presentation(ctx, expr) {
        return Some(ctx.num(1));
    }
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = remove_unit_log_e_factor_for_calculus_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    let retained = factors
        .into_iter()
        .filter(|factor| !is_ln_e_for_calculus_presentation(ctx, *factor))
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Some(ctx.num(1));
    }
    if retained.len() == cas_math::expr_nary::mul_leaves(ctx, expr).len() {
        return None;
    }
    Some(cas_math::expr_nary::build_balanced_mul(ctx, &retained))
}

fn is_ln_e_for_calculus_presentation(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Ln) => {
            matches!(ctx.get(args[0]), Expr::Constant(Constant::E))
        }
        _ => false,
    }
}

fn ln_sum_of_equal_derivative_roots_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (compact, _) = ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
        ctx, target, var_name,
    )?;
    Some(compact)
}

pub(crate) fn ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    if terms.len() != 2 {
        return None;
    }

    let mut radicands = Vec::with_capacity(2);
    for (term, sign) in terms {
        if sign == cas_math::expr_nary::Sign::Neg {
            return None;
        }
        radicands.push(extract_square_root_base(ctx, term)?);
    }

    let left_poly = polynomial_radicand_for_calculus_presentation(ctx, radicands[0], var_name)?;
    let right_poly = polynomial_radicand_for_calculus_presentation(ctx, radicands[1], var_name)?;
    let left_positive_everywhere = polynomial_is_strictly_positive_everywhere(&left_poly);
    let right_positive_everywhere = polynomial_is_strictly_positive_everywhere(&right_poly);
    let affine_pair = left_poly.degree() <= 1 && right_poly.degree() <= 1;
    let strictly_positive_quadratic_pair = left_poly.degree() <= 2
        && right_poly.degree() <= 2
        && left_positive_everywhere
        && right_positive_everywhere;
    if !affine_pair && !strictly_positive_quadratic_pair {
        return None;
    }
    let derivative_poly = left_poly.derivative();
    if derivative_poly != right_poly.derivative() {
        return None;
    }
    if derivative_poly.is_zero() {
        let required_conditions =
            positive_radicand_conditions_for_equal_derivative_roots_presentation(
                radicands[0],
                left_positive_everywhere,
                radicands[1],
                right_positive_everywhere,
            );
        return Some((ctx.num(0), required_conditions));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let left_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicands[0]]);
    let right_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicands[1]]);
    let left_sqrt = cas_ast::hold::wrap_hold(ctx, left_sqrt);
    let right_sqrt = cas_ast::hold::wrap_hold(ctx, right_sqrt);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[left_sqrt, right_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));
    let required_conditions = positive_radicand_conditions_for_equal_derivative_roots_presentation(
        radicands[0],
        left_positive_everywhere,
        radicands[1],
        right_positive_everywhere,
    );
    Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions))
}

fn positive_radicand_conditions_for_equal_derivative_roots_presentation(
    left_radicand: ExprId,
    left_positive_everywhere: bool,
    right_radicand: ExprId,
    right_positive_everywhere: bool,
) -> Vec<crate::ImplicitCondition> {
    let mut conditions = Vec::with_capacity(2);
    if !left_positive_everywhere {
        conditions.push(crate::ImplicitCondition::Positive(left_radicand));
    }
    if !right_positive_everywhere {
        conditions.push(crate::ImplicitCondition::Positive(right_radicand));
    }
    conditions
}

fn ln_sqrt_polynomial_gap_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    let mut radicand = None;
    let mut polynomial_term_poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in terms {
        if let Some(term_radicand) = extract_square_root_base(ctx, term) {
            if sign == cas_math::expr_nary::Sign::Neg {
                return None;
            }
            if radicand.is_some() {
                return None;
            }
            radicand = Some(term_radicand);
        } else {
            let mut term_poly = polynomial_radicand_for_calculus_presentation(ctx, term, var_name)?;
            if sign == cas_math::expr_nary::Sign::Neg {
                term_poly = term_poly.neg();
            }
            polynomial_term_poly = polynomial_term_poly.add(&term_poly);
        }
    }

    let radicand = radicand?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let positive_gap = radicand_poly.sub(&polynomial_term_poly.mul(&polynomial_term_poly));
    if positive_gap.degree() != 0
        || positive_gap
            .coeffs
            .first()
            .is_none_or(|value| !value.is_positive())
    {
        return None;
    }

    let derivative_poly = polynomial_term_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let numerator = scale_expr_for_calculus_presentation(ctx, derivative_content, derivative_core);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn ln_sqrt_plus_polynomial_direct_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    let mut radicand = None;
    let mut polynomial_term_poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in terms {
        if let Some(term_radicand) = extract_square_root_base(ctx, term) {
            if sign == cas_math::expr_nary::Sign::Neg || radicand.replace(term_radicand).is_some() {
                return None;
            }
            continue;
        }

        let mut term_poly = polynomial_radicand_for_calculus_presentation(ctx, term, var_name)?;
        if sign == cas_math::expr_nary::Sign::Neg {
            term_poly = term_poly.neg();
        }
        polynomial_term_poly = polynomial_term_poly.add(&term_poly);
    }

    let radicand = radicand?;
    if polynomial_term_poly.is_zero() {
        return None;
    }
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let radicand_derivative_poly = radicand_poly.derivative();
    let polynomial_derivative_poly = polynomial_term_poly.derivative();
    if radicand_derivative_poly.is_zero() && polynomial_derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let square_gap = polynomial_term_poly
        .mul(&polynomial_term_poly)
        .sub(&radicand_poly);
    if square_gap.degree() == 0 {
        if let Some(gap_value) = square_gap.coeffs.first() {
            if gap_value.is_positive() {
                let polynomial_derivative_poly = polynomial_term_poly.derivative();
                if polynomial_derivative_poly.is_zero() {
                    return Some((ctx.num(0), Vec::new()));
                }

                let derivative = polynomial_derivative_poly.to_expr(ctx);
                let (derivative_core, derivative_content) =
                    split_polynomial_content_for_calculus_presentation(ctx, derivative);
                let numerator =
                    scale_expr_for_calculus_presentation(ctx, derivative_content, derivative_core);
                let square_arg_poly = if polynomial_term_poly.leading_coeff().is_negative() {
                    polynomial_term_poly.neg()
                } else {
                    polynomial_term_poly.clone()
                };
                let polynomial_term = polynomial_term_poly.to_expr(ctx);
                let square_arg = square_arg_poly.to_expr(ctx);
                let polynomial_term_sq = squared_expr_for_compact_gap_presentation(ctx, square_arg);
                let gap_expr = rational_const_for_calculus_presentation(ctx, gap_value.clone());
                let compact_radicand = ctx.add(Expr::Sub(polynomial_term_sq, gap_expr));
                let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![compact_radicand]);
                let denominator = cas_ast::hold::wrap_hold(ctx, denominator);
                let compact = ctx.add(Expr::Div(numerator, denominator));

                let branch_boundary =
                    sqrt_positive_rational_expr_for_calculus_presentation(ctx, gap_value.clone());
                let branch_gap = ctx.add(Expr::Sub(polynomial_term, branch_boundary));
                return Some((
                    ctx.add(Expr::Hold(compact)),
                    vec![crate::ImplicitCondition::Positive(branch_gap)],
                ));
            }
        }
    }

    if let Some(scale) = nonzero_polynomial_scale_factor(&polynomial_term_poly, &radicand_poly) {
        let required_conditions = if scale.is_negative() {
            let scale_square = &scale * &scale;
            let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
            let scaled_radicand = ctx.add(Expr::Mul(scale_square_expr, radicand));
            let one = ctx.num(1);
            let upper_boundary = ctx.add(Expr::Sub(one, scaled_radicand));
            vec![
                crate::ImplicitCondition::Positive(radicand),
                crate::ImplicitCondition::Positive(upper_boundary),
            ]
        } else {
            Vec::new()
        };
        let radicand_derivative = radicand_derivative_poly.to_expr(ctx);
        let scaled_sqrt = scale_expr_for_calculus_presentation(ctx, scale.clone(), sqrt_radicand);
        let one = ctx.num(1);
        let denominator_tail = ctx.add(Expr::Add(one, scaled_sqrt));
        let leading = ctx.add(Expr::Div(radicand_derivative, radicand));
        let two = ctx.num(2);
        let correction_denominator =
            cas_math::expr_nary::build_balanced_mul(ctx, &[two, radicand, denominator_tail]);
        let correction = ctx.add(Expr::Div(radicand_derivative, correction_denominator));
        let compact = ctx.add(Expr::Sub(leading, correction));
        return Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions));
    }

    None
}

fn ln_sqrt_negative_polynomial_gap_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return false;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return false;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    let mut radicand = None;
    let mut polynomial_term_poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in terms {
        if let Some(term_radicand) = extract_square_root_base(ctx, term) {
            if sign == cas_math::expr_nary::Sign::Neg || radicand.replace(term_radicand).is_some() {
                return false;
            }
            continue;
        }

        let Some(mut term_poly) =
            polynomial_radicand_for_calculus_presentation(ctx, term, var_name)
        else {
            return false;
        };
        if sign == cas_math::expr_nary::Sign::Neg {
            term_poly = term_poly.neg();
        }
        polynomial_term_poly = polynomial_term_poly.add(&term_poly);
    }

    let Some(radicand) = radicand else {
        return false;
    };
    if polynomial_term_poly.is_zero() || polynomial_term_poly.derivative().is_zero() {
        return false;
    }
    let Some(radicand_poly) =
        polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)
    else {
        return false;
    };
    let square_gap = polynomial_term_poly
        .mul(&polynomial_term_poly)
        .sub(&radicand_poly);
    square_gap.degree() == 0
        && square_gap
            .coeffs
            .first()
            .is_some_and(|value| value.is_positive())
}

fn nonzero_polynomial_scale_factor(scaled: &Polynomial, base: &Polynomial) -> Option<BigRational> {
    if base.is_zero() || scaled.is_zero() || scaled.var != base.var {
        return None;
    }
    let max_len = scaled.coeffs.len().max(base.coeffs.len());
    let mut scale = None;
    for index in 0..max_len {
        let scaled_coeff = scaled
            .coeffs
            .get(index)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let base_coeff = base
            .coeffs
            .get(index)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if base_coeff.is_zero() {
            if !scaled_coeff.is_zero() {
                return None;
            }
            continue;
        }
        let local_scale = scaled_coeff / base_coeff;
        if let Some(existing) = &scale {
            if existing != &local_scale {
                return None;
            }
        } else {
            scale = Some(local_scale);
        }
    }

    scale.filter(|value| !value.is_zero())
}

fn unit_interval_bounded_inverse_trig_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut outer_coeff = BigRational::one();
    let mut inverse_trig = None;
    let mut target = target;

    if let Expr::Neg(inner) = ctx.get(target).clone() {
        outer_coeff = -outer_coeff;
        target = inner;
    }

    if let Expr::Div(numerator, denominator) = ctx.get(target).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, denominator, 8)?;
        if denominator.is_zero() {
            return None;
        }
        outer_coeff /= denominator;
        target = numerator;
    }

    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            outer_coeff *= value;
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
            return None;
        };
        let derivative_sign = match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
            Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
            _ => return None,
        };
        if args.len() != 1 || inverse_trig.is_some() {
            return None;
        }
        inverse_trig = Some((derivative_sign, args[0]));
    }

    let (derivative_sign, arg) = inverse_trig?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let offset = arg_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let two = BigRational::from_integer(2.into());
    let unit_interval_arg = (offset == -BigRational::one() && slope == two)
        || (offset == BigRational::one() && slope == -two);
    if !unit_interval_arg {
        return None;
    }

    let coefficient = outer_coeff * derivative_sign * slope / BigRational::from_integer(2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let var = ctx.var(var_name);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var]);
    let one = ctx.num(1);
    let var = ctx.var(var_name);
    let one_minus_var = ctx.add(Expr::Sub(one, var));
    let sqrt_one_minus_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![one_minus_var]);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_var, sqrt_one_minus_var]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn ln_power_derivative_numeric_presentation(
    ctx: &mut Context,
    target: ExprId,
    result: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let Expr::Pow(_, exp) = ctx.get(args[0]) else {
        return None;
    };
    let exp_value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    if !exp_value.is_integer() || exp_value.is_zero() {
        return None;
    }

    let compact = fold_numeric_mul_constants_for_hold(ctx, result);
    (compact != result).then_some(compact)
}

fn arctan_arg_matches_for_calculus_presentation(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    if compare_expr(ctx, left, right) == std::cmp::Ordering::Equal {
        return true;
    }

    let Ok(left_poly) = Polynomial::from_expr(ctx, left, var_name) else {
        return false;
    };
    let Ok(right_poly) = Polynomial::from_expr(ctx, right, var_name) else {
        return false;
    };
    left_poly == right_poly
}

fn extract_arctan_term_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(ExprId, ExprId)> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        let (arg, coeff) = extract_arctan_term_for_calculus_presentation(ctx, inner)?;
        let coeff = ctx.add(Expr::Neg(coeff));
        return Some((arg, coeff));
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let (arg, coeff) = extract_arctan_term_for_calculus_presentation(ctx, num)?;
        let coeff = ctx.add(Expr::Div(coeff, den));
        return Some((arg, coeff));
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut arctan_arg = None;
    let mut coefficient_factors = Vec::new();

    for factor in factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        match ctx.get(factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    ) =>
            {
                if arctan_arg.is_some() {
                    return None;
                }
                arctan_arg = Some(args[0]);
            }
            Expr::Div(num, den) => {
                let Expr::Function(fn_id, args) = ctx.get(num).clone() else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    )
                {
                    if arctan_arg.is_some() {
                        return None;
                    }
                    arctan_arg = Some(args[0]);
                    let one = ctx.num(1);
                    coefficient_factors.push(ctx.add(Expr::Div(one, den)));
                } else {
                    coefficient_factors.push(factor);
                }
            }
            Expr::Neg(inner) => {
                let inner = unwrap_internal_hold_for_calculus(ctx, inner);
                let Expr::Function(fn_id, args) = ctx.get(inner).clone() else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    )
                {
                    if arctan_arg.is_some() {
                        return None;
                    }
                    arctan_arg = Some(args[0]);
                    let minus_one = ctx.num(-1);
                    coefficient_factors.push(minus_one);
                } else {
                    coefficient_factors.push(factor);
                }
            }
            _ => coefficient_factors.push(factor),
        }
    }

    let arg = arctan_arg?;
    let coeff = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &coefficient_factors)
    };
    Some((arg, coeff))
}

fn negate_term_for_calculus_presentation(ctx: &mut Context, term: ExprId) -> ExprId {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Number(value) = ctx.get(term).clone() {
        return ctx.add(Expr::Number(-value));
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let num = negate_term_for_calculus_presentation(ctx, num);
        return ctx.add(Expr::Div(num, den));
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    if factors.len() > 1 {
        let mut replaced = false;
        let mut negated_factors = Vec::with_capacity(factors.len());
        for factor in factors {
            if !replaced {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    negated_factors.push(ctx.add(Expr::Number(-value)));
                    replaced = true;
                    continue;
                }
            }
            negated_factors.push(factor);
        }
        if replaced {
            return cas_math::expr_nary::build_balanced_mul(ctx, &negated_factors);
        }
    }

    ctx.add(Expr::Neg(term))
}

struct LnTermForCalculusPresentation {
    arg: ExprId,
    arg_poly: Polynomial,
    coefficient: BigRational,
}

fn extract_ln_term_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<LnTermForCalculusPresentation> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        let mut extracted = extract_ln_term_for_calculus_presentation(ctx, inner, var_name)?;
        extracted.coefficient = -extracted.coefficient;
        return Some(extracted);
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut extracted = extract_ln_term_for_calculus_presentation(ctx, num, var_name)?;
        extracted.coefficient /= denominator;
        return Some(extracted);
    }

    let mut ln_arg = None;
    let mut coefficient = BigRational::one();
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        coefficient *= factor_value;
    }

    let arg = ln_arg?;
    Some(LnTermForCalculusPresentation {
        arg,
        arg_poly: Polynomial::from_expr(ctx, arg, var_name).ok()?,
        coefficient,
    })
}

fn build_scaled_ln_for_calculus_presentation(
    ctx: &mut Context,
    coefficient: &BigRational,
    arg: ExprId,
) -> Option<ExprId> {
    if coefficient.is_zero() {
        return None;
    }

    let ln = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
    if coefficient.is_one() {
        return Some(ln);
    }
    if *coefficient == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(ln)));
    }

    let coefficient = ctx.add(Expr::Number(coefficient.clone()));
    Some(ctx.add(Expr::Mul(coefficient, ln)))
}

fn ln_polynomial_coefficient_degree_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<usize> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        return ln_polynomial_coefficient_degree_for_calculus_presentation(ctx, inner, var_name);
    }

    let mut ln_seen = false;
    let mut coefficient_factors = Vec::new();
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_seen {
                    return None;
                }
                ln_seen = true;
                continue;
            }
        }
        coefficient_factors.push(factor);
    }
    if !ln_seen {
        return None;
    }

    let coefficient = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &coefficient_factors)
    };
    Some(
        Polynomial::from_expr(ctx, coefficient, var_name)
            .ok()?
            .degree(),
    )
}

fn compact_arctan_presentation_other_terms(
    ctx: &mut Context,
    terms: Vec<ExprId>,
    var_name: &str,
) -> Vec<ExprId> {
    let mut polynomial_sum = Polynomial::zero(var_name.to_string());
    let mut ln_groups: Vec<LnTermForCalculusPresentation> = Vec::new();
    let mut passthrough = Vec::new();

    for term in terms {
        if let Some(ln_term) = extract_ln_term_for_calculus_presentation(ctx, term, var_name) {
            if let Some(existing) = ln_groups
                .iter_mut()
                .find(|existing| existing.arg_poly == ln_term.arg_poly)
            {
                existing.coefficient += ln_term.coefficient;
            } else {
                ln_groups.push(ln_term);
            }
            continue;
        }

        if let Ok(poly) = Polynomial::from_expr(ctx, term, var_name) {
            polynomial_sum = polynomial_sum.add(&poly);
            continue;
        }

        passthrough.push(term);
    }

    let mut out = Vec::new();
    for ln_term in ln_groups {
        if let Some(term) =
            build_scaled_ln_for_calculus_presentation(ctx, &ln_term.coefficient, ln_term.arg)
        {
            out.push(term);
        }
    }
    if !polynomial_sum.is_zero() {
        out.push(polynomial_sum.to_expr(ctx));
    }
    out.extend(passthrough);
    out
}

fn contains_nontrivial_arctan_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let mut stack = vec![cas_ast::hold::unwrap_internal_hold(ctx, expr)];
    while let Some(current) = stack.pop() {
        match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, current)) {
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(fn_id, args) => {
                if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Arctan)
                    && args.len() == 1
                    && !variable_named(ctx, args[0], var_name)
                {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            _ => {}
        }
    }
    false
}

fn flatten_subtracting_additive_group_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Sub(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    let right = unwrap_internal_hold_for_calculus(ctx, right);
    if !matches!(ctx.get(right), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }
    if !contains_nontrivial_arctan_for_calculus_presentation(ctx, right, var_name) {
        return None;
    }
    if ln_polynomial_coefficient_degree_for_calculus_presentation(ctx, left, var_name)? > 5 {
        return None;
    }

    let mut additive_terms = Vec::new();
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        left,
        cas_math::expr_nary::Sign::Pos,
        &mut additive_terms,
    );
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        right,
        cas_math::expr_nary::Sign::Neg,
        &mut additive_terms,
    );
    if additive_terms.len() < 3 {
        return None;
    }

    let terms = additive_terms
        .into_iter()
        .map(|(term, sign)| {
            let signed = match sign {
                cas_math::expr_nary::Sign::Pos => term,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, term),
            };
            fold_numeric_mul_constants_for_hold(ctx, signed)
        })
        .collect::<Vec<_>>();

    Some(cas_math::expr_nary::build_balanced_add(ctx, &terms))
}

fn linear_hyperbolic_integer_slope_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, expr).factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
            continue;
        };
        if args.len() != 1
            || !matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Sinh | BuiltinFn::Cosh)
            )
        {
            continue;
        }
        let Ok(arg_poly) = Polynomial::from_expr(ctx, args[0], var_name) else {
            return false;
        };
        if arg_poly.degree() != 1 {
            return false;
        }
        let slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        return !slope.is_zero() && slope.is_integer();
    }
    false
}

fn compact_arctan_additive_terms_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);

    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            if cas_ast::views::as_rational_const(ctx, left, 8).is_some() {
                let compact =
                    compact_arctan_additive_terms_for_calculus_presentation(ctx, right, var_name)?;
                let compact = cas_ast::hold::wrap_hold(ctx, compact);
                return Some(ctx.add(Expr::Mul(left, compact)));
            }
            if cas_ast::views::as_rational_const(ctx, right, 8).is_some() {
                let compact =
                    compact_arctan_additive_terms_for_calculus_presentation(ctx, left, var_name)?;
                let compact = cas_ast::hold::wrap_hold(ctx, compact);
                return Some(ctx.add(Expr::Mul(compact, right)));
            }
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            let compact =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, num, var_name)?;
            let compact = cas_ast::hold::wrap_hold(ctx, compact);
            return Some(ctx.add(Expr::Div(compact, den)));
        }
        _ => {}
    }

    let mut additive_terms = Vec::new();
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        expr,
        cas_math::expr_nary::Sign::Pos,
        &mut additive_terms,
    );
    if additive_terms.len() < 2 {
        return None;
    }

    let mut arctan_arg = None;
    let mut arctan_coefficients = Vec::new();
    let mut other_terms = Vec::new();
    let mut arctan_term_count = 0usize;

    for (term, sign) in additive_terms {
        if let Some((arg, coeff)) = extract_arctan_term_for_calculus_presentation(ctx, term) {
            if let Some(existing_arg) = arctan_arg {
                if !arctan_arg_matches_for_calculus_presentation(ctx, existing_arg, arg, var_name) {
                    return None;
                }
            } else {
                arctan_arg = Some(arg);
            }
            arctan_term_count += 1;
            let coeff = match sign {
                cas_math::expr_nary::Sign::Pos => coeff,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, coeff),
            };
            let coeff = fold_numeric_mul_constants_for_hold(ctx, coeff);
            arctan_coefficients.push(coeff);
        } else {
            let signed_term = match sign {
                cas_math::expr_nary::Sign::Pos => term,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, term),
            };
            let signed_term = fold_numeric_mul_constants_for_hold(ctx, signed_term);
            other_terms.push(signed_term);
        }
    }

    if arctan_term_count < 2 {
        return None;
    }

    let arg = arctan_arg?;
    if Polynomial::from_expr(ctx, arg, var_name).is_err() {
        return None;
    };
    let coeff = cas_math::expr_nary::build_balanced_add(ctx, &arctan_coefficients);
    let coeff = cas_ast::hold::wrap_hold(ctx, coeff);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    let arctan_term = ctx.add(Expr::Mul(coeff, arctan));

    let mut terms = vec![arctan_term];
    terms.extend(compact_arctan_presentation_other_terms(
        ctx,
        other_terms,
        var_name,
    ));
    Some(cas_math::expr_nary::build_balanced_add(ctx, &terms))
}

fn collect_additive_terms_for_arctan_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    sign: cas_math::expr_nary::Sign,
    out: &mut Vec<(ExprId, cas_math::expr_nary::Sign)>,
) {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, left, sign, out);
            collect_additive_terms_for_arctan_calculus_presentation(ctx, right, sign, out);
        }
        Expr::Sub(left, right) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, left, sign, out);
            collect_additive_terms_for_arctan_calculus_presentation(ctx, right, sign.negate(), out);
        }
        Expr::Neg(inner) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, inner, sign.negate(), out);
        }
        _ => out.push((expr, sign)),
    }
}

fn polynomial_times_arctan_affine_integrand_for_diff_shortcut(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, expr).terms;
    !terms.is_empty()
        && terms.into_iter().all(|(term, _)| {
            polynomial_times_arctan_affine_term_for_diff_shortcut(ctx, term, var_name)
        })
}

fn polynomial_times_arctan_affine_term_for_diff_shortcut(
    ctx: &Context,
    term: ExprId,
    var_name: &str,
) -> bool {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    match ctx.get(term).clone() {
        Expr::Neg(inner) => {
            return polynomial_times_arctan_affine_term_for_diff_shortcut(ctx, inner, var_name);
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            return polynomial_times_arctan_affine_term_for_diff_shortcut(ctx, num, var_name);
        }
        _ => {}
    }

    let mut arctan_arg = None;
    let mut polynomial_factor = Polynomial::one(var_name.to_string());
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                )
            {
                if arctan_arg.replace(args[0]).is_some() {
                    return false;
                }
                continue;
            }
        }

        let Ok(factor_poly) = Polynomial::from_expr(ctx, factor, var_name) else {
            return false;
        };
        polynomial_factor = polynomial_factor.mul(&factor_poly);
    }

    let Some(arg) = arctan_arg else {
        return false;
    };
    let Ok(arg_poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return false;
    };
    arg_poly.degree() == 1 && !arg_poly.derivative().is_zero() && !polynomial_factor.is_zero()
}

pub(crate) fn try_post_calculus_presentation(
    ctx: &mut Context,
    source: ExprId,
    result: ExprId,
) -> Option<ExprId> {
    if let Some(call) = try_extract_integrate_call(ctx, source) {
        if cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
            ctx,
            call.target,
            &call.var_name,
        ) {
            if let Some(compact) = compact_arctan_additive_terms_for_calculus_presentation(
                ctx,
                result,
                &call.var_name,
            ) {
                return Some(compact);
            }
        }
        if cas_math::symbolic_integration_support::integrate_symbolic_is_fractional_denominator_power_substitution_target(
            ctx,
            call.target,
            &call.var_name,
        ) {
            let allow_conditional_positive_quadratic =
                !integrate_required_positive_conditions(ctx, call.target, &call.var_name)
                    .is_empty();
            if let Some(compact) =
                compact_negative_three_half_power_result_for_integration_presentation(
                    ctx,
                    result,
                    &call.var_name,
                    allow_conditional_positive_quadratic,
                )
            {
                return Some(compact);
            }
            if let Some(compact) =
                compact_negative_half_power_result_for_integration_presentation(ctx, result)
            {
                return Some(compact);
            }
        }
        if let Some(compact) =
            compact_positive_half_power_result_for_integration_presentation(ctx, result)
        {
            return Some(compact);
        }
        if let Some(compact) =
            compact_acosh_surd_width_arg_for_integration_presentation(ctx, result)
        {
            return Some(compact);
        }
    }

    let call = try_extract_diff_call(ctx, source)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if let Some((compact, _, _)) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _, _)) =
        arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _, _)) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _, _)) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _)) =
        ln_constant_shifted_tan_sqrt_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if try_extract_integrate_call(ctx, target).is_some() {
        if let Some(compact) =
            compact_sqrt_trig_log_derivative_integrand(ctx, result, &call.var_name)
        {
            return Some(compact);
        }
        if let Some(compact) =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, &call.var_name)
        {
            return Some(compact);
        }
    }
    if let Some(compact) = sqrt_cosh_log_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if ln_sqrt_negative_polynomial_gap_target(ctx, target, &call.var_name) {
        if let Some(compact) =
            compact_negative_half_power_result_for_integration_presentation(ctx, result)
        {
            return Some(compact);
        }
    }
    if let Some((compact, _)) =
        sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = supported_integral_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_trig_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = reciprocal_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        sqrt_over_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_over_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_polynomial_quotient_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_of_polynomial_quotient_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_exp_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) = sqrt_shifted_ln_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _, _)) =
        sqrt_small_additive_elementary_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) =
        sqrt_elementary_function_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        sqrt_reciprocal_trig_function_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        signed_elementary_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_arctan_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_arctan_surd_quotient_scaled_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        constant_scaled_acosh_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_sqrt_affine_quotient_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_reciprocal_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_bounded_inverse_trig_sqrt_polynomial_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) = unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        arctan_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_sqrt_affine_quotient_positive_gap_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        arctan_self_normalized_surd_reciprocal_compact_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        -BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = arctan_sqrt_affine_partition_quotient_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = arctan_sqrt_polynomial_quotient_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        asinh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_asinh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_atanh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        acosh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_acosh_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) = constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_sqrt_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_sqrt_quadratic_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_affine_abs_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_positive_quadratic_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_reciprocal_trig_positive_quadratic_surd_quotient_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_reciprocal_trig_positive_quadratic_square_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = ln_sqrt_shift_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        variable_base_constant_argument_log_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) = ln_power_derivative_numeric_presentation(ctx, target, result) {
        return Some(compact);
    }
    if let Some(compact) =
        unit_interval_bounded_inverse_trig_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        arctan_rational_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_rational_affine_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = asinh_polynomial_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) = acosh_affine_derivative_presentation(ctx, target, &call.var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        acosh_strictly_positive_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        asinh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some((compact, _)) =
        atanh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }
    if let Some(compact) = arctan_sqrt_constant_over_polynomial_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        arccot_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        arccot_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = arctan_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        &call.var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }

    let (radicand, derivative_scale) =
        arctan_sqrt_scaled_variable_arg(ctx, target, &call.var_name)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let half = BigRational::new(1.into(), 2.into());
    let coefficient = derivative_scale * half;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let denominator_head = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add(Expr::Mul(denominator_scale, sqrt_radicand))
    };
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let denominator = ctx.add(Expr::Mul(denominator_head, radicand_plus_one));
    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some(compact)
}

pub(crate) fn try_calculus_result_presentation(
    ctx: &mut Context,
    result: ExprId,
) -> Option<ExprId> {
    let result = unwrap_internal_hold_for_calculus(ctx, result);
    compact_sqrt_trig_log_derivative_integrand(ctx, result, "x")
        .or_else(|| compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, "x"))
        .or_else(|| {
            compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation(ctx, result)
        })
        .or_else(|| compact_negative_half_power_product_for_calculus_presentation(ctx, result))
        .or_else(|| {
            has_compactable_ln_abs_cosh_sqrt(ctx, result, "x").then(|| {
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, result, "x")
            })
        })
        .or_else(|| {
            compact_half_power_sum_root_product_for_integration_presentation(ctx, result, "x")
        })
        .or_else(|| {
            compact_trig_square_reduction_primitive_for_integration_presentation(ctx, result, "x")
        })
        .or_else(|| {
            compact_trig_odd_power_reduction_primitive_for_integration_presentation(ctx, result)
        })
        .or_else(|| compact_acosh_surd_width_arg_for_integration_presentation(ctx, result))
        .or_else(|| compact_arctan_additive_terms_for_calculus_presentation(ctx, result, "x"))
}

fn compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };

    let mut numerator_factors = cas_math::expr_nary::mul_leaves(ctx, numerator);
    for idx in 0..numerator_factors.len() {
        let Some(base) = extract_square_root_base(ctx, numerator_factors[idx]) else {
            continue;
        };
        let Some(denominator_scale) = positive_rational_scale_between_exprs(ctx, denominator, base)
        else {
            continue;
        };

        numerator_factors.remove(idx);
        let compact_numerator = match numerator_factors.as_slice() {
            [] => ctx.num(1),
            [single] => *single,
            _ => cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors),
        };

        let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
        let mut denominator_factors = Vec::new();
        if !denominator_scale.is_one() {
            denominator_factors.push(ctx.add(Expr::Number(denominator_scale)));
        }
        denominator_factors.push(sqrt_base);
        let compact_denominator =
            cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
        return Some(ctx.add(Expr::Div(compact_numerator, compact_denominator)));
    }

    None
}

fn positive_rational_scale_between_exprs(
    ctx: &mut Context,
    scaled: ExprId,
    base: ExprId,
) -> Option<BigRational> {
    let budget = PolyBudget {
        max_terms: 24,
        max_total_degree: 8,
        max_pow_exp: 4,
    };
    if let (Ok(scaled_poly), Ok(base_poly)) = (
        multipoly_from_expr(ctx, scaled, &budget),
        multipoly_from_expr(ctx, base, &budget),
    ) {
        if scaled_poly.vars == base_poly.vars
            && !scaled_poly.terms.is_empty()
            && scaled_poly.terms.len() == base_poly.terms.len()
        {
            let scale = scaled_poly.terms[0].0.clone() / base_poly.terms[0].0.clone();
            if scale.is_positive() && base_poly.mul_scalar(&scale) == scaled_poly {
                return Some(scale);
            }
        }
    }

    positive_rational_scale_between_structural_additions(ctx, scaled, base)
}

fn positive_rational_scale_between_structural_additions(
    ctx: &mut Context,
    scaled: ExprId,
    base: ExprId,
) -> Option<BigRational> {
    let (direct_scale, direct_core) =
        split_numeric_scale_product_for_calculus_presentation(ctx, scaled);
    if direct_scale.is_positive() && structurally_equivalent_for_calculus(ctx, direct_core, base) {
        return Some(direct_scale);
    }

    let scaled_terms = scaled_additive_terms_for_calculus_presentation(ctx, scaled);
    let base_terms = scaled_additive_terms_for_calculus_presentation(ctx, base);
    if scaled_terms.is_empty() || scaled_terms.len() != base_terms.len() {
        return None;
    }

    let mut matched = vec![false; base_terms.len()];
    let mut common_scale = None;
    for (scaled_coeff, scaled_core) in scaled_terms {
        let (index, base_coeff, _base_core) =
            base_terms
                .iter()
                .enumerate()
                .find_map(|(index, (base_coeff, base_core))| {
                    (!matched[index]
                        && structurally_equivalent_for_calculus(ctx, scaled_core, *base_core))
                    .then_some((index, base_coeff, base_core))
                })?;
        if base_coeff.is_zero() {
            return None;
        }
        let scale = scaled_coeff / base_coeff.clone();
        if common_scale.as_ref().is_some_and(|common| common != &scale) {
            return None;
        }
        common_scale = Some(scale);
        matched[index] = true;
    }

    common_scale.filter(|scale: &BigRational| scale.is_positive())
}

fn scaled_additive_terms_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Vec<(BigRational, ExprId)> {
    cas_math::expr_nary::add_terms_signed(ctx, expr)
        .into_iter()
        .map(|(term, sign)| {
            let (scale, core) = split_numeric_scale_product_for_calculus_presentation(ctx, term);
            let scale = if sign == cas_math::expr_nary::Sign::Neg {
                -scale
            } else {
                scale
            };
            (scale, core)
        })
        .collect()
}

fn split_numeric_scale_product_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> (BigRational, ExprId) {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return (value, ctx.num(1));
    }

    let mut scale = BigRational::one();
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

fn structurally_equivalent_for_calculus(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    left == right || cas_math::expr_domain::exprs_equivalent(ctx, left, right)
}

fn compact_negative_half_power_product_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_half_power_product_for_calculus_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    collect_fraction_product_factors_for_calculus_presentation(
        ctx,
        expr,
        &mut numerator_factors,
        &mut denominator_factors,
    );

    let mut base = None;
    let mut retained_numerator = Vec::new();
    for factor in numerator_factors {
        if base.is_none() {
            if let Some(pow_base) = negative_half_power_base_for_calculus_presentation(ctx, factor)
            {
                base = Some(pow_base);
                continue;
            }
        }
        retained_numerator.push(factor);
    }

    let base = base?;
    denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![base]));

    let one = ctx.num(1);
    let numerator = if retained_numerator.is_empty() {
        one
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &retained_numerator)
    };
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

fn collect_fraction_product_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                left,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                right,
                numerator_factors,
                denominator_factors,
            );
        }
        Expr::Div(num, den) => {
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                num,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                den,
                denominator_factors,
            );
        }
        _ if split_rational_factor_for_calculus_presentation(
            ctx,
            expr,
            numerator_factors,
            denominator_factors,
        ) => {}
        _ if !is_calculus_presentation_one(ctx, expr) => numerator_factors.push(expr),
        _ => {}
    }
}

fn collect_fraction_denominator_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    denominator_factors: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                left,
                denominator_factors,
            );
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                right,
                denominator_factors,
            );
        }
        _ if split_rational_factor_for_calculus_presentation(
            ctx,
            expr,
            denominator_factors,
            &mut Vec::new(),
        ) => {}
        _ if !is_calculus_presentation_one(ctx, expr) => denominator_factors.push(expr),
        _ => {}
    }
}

fn split_rational_factor_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) -> bool {
    let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) else {
        return false;
    };
    if value.is_zero() {
        numerator_factors.push(expr);
        return true;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    if !numerator.is_one() {
        numerator_factors.push(ctx.add(Expr::Number(numerator)));
    }
    if !denominator.is_one() {
        denominator_factors.push(ctx.add(Expr::Number(denominator)));
    }
    true
}

fn is_calculus_presentation_one(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value.is_one())
}

fn negative_half_power_base_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exponent = small_rational_const_for_calculus_presentation(ctx, *exp)?;
    (exponent == BigRational::new((-1).into(), 2.into())).then_some(*base)
}

fn small_rational_const_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return Some(value);
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => Some(
            small_rational_const_for_calculus_presentation(ctx, *left)?
                + small_rational_const_for_calculus_presentation(ctx, *right)?,
        ),
        Expr::Sub(left, right) => Some(
            small_rational_const_for_calculus_presentation(ctx, *left)?
                - small_rational_const_for_calculus_presentation(ctx, *right)?,
        ),
        Expr::Neg(inner) => Some(-small_rational_const_for_calculus_presentation(
            ctx, *inner,
        )?),
        _ => None,
    }
}

fn compact_acosh_surd_width_arg_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let unwrapped = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(unwrapped).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Acosh) || args.len() != 1 {
        return None;
    }

    let compact_arg =
        compact_rationalized_sqrt_denominator_arg_for_calculus_presentation(ctx, args[0])?;
    let compact = ctx.call_builtin(BuiltinFn::Acosh, vec![compact_arg]);
    if unwrapped == expr {
        Some(compact)
    } else {
        Some(cas_ast::hold::wrap_hold(ctx, compact))
    }
}

fn compact_rationalized_sqrt_denominator_arg_for_calculus_presentation(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(arg).clone() {
        let denominator_value = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if !denominator_value.is_positive() {
            return None;
        }
        return compact_rationalized_sqrt_product_for_calculus_presentation(
            ctx,
            num,
            denominator_value,
        );
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    let (scale_idx, denominator_value) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let value = cas_ast::views::as_rational_const(ctx, *factor, 8)?;
        if !value.is_positive() || value >= BigRational::one() {
            return None;
        }
        let denominator_value = BigRational::one() / value;
        Some((idx, denominator_value))
    })?;
    let mut numerator_factors = factors;
    numerator_factors.remove(scale_idx);
    let numerator = match numerator_factors.as_slice() {
        [] => ctx.num(1),
        [single] => *single,
        _ => cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors),
    };
    compact_rationalized_sqrt_product_for_calculus_presentation(ctx, numerator, denominator_value)
}

fn compact_rationalized_sqrt_product_for_calculus_presentation(
    ctx: &mut Context,
    num: ExprId,
    denominator_value: BigRational,
) -> Option<ExprId> {
    let mut factors = cas_math::expr_nary::mul_leaves(ctx, num);
    for idx in 0..factors.len() {
        let Some(sqrt_value) =
            sqrt_positive_rational_factor_value_for_calculus_presentation(ctx, factors[idx])
        else {
            continue;
        };
        if sqrt_value != denominator_value {
            continue;
        }

        factors.remove(idx);
        let numerator = match factors.as_slice() {
            [] => ctx.num(1),
            [single] => *single,
            _ => cas_math::expr_nary::build_balanced_mul(ctx, &factors),
        };
        let denominator =
            sqrt_positive_rational_expr_for_calculus_presentation(ctx, denominator_value);
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    None
}

fn sqrt_positive_rational_factor_value_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, expr)) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            let value = cas_ast::views::as_rational_const(ctx, args[0], 8)?;
            value.is_positive().then_some(value)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new(1.into(), 2.into())) =>
        {
            let value = cas_ast::views::as_rational_const(ctx, *base, 8)?;
            value.is_positive().then_some(value)
        }
        _ => None,
    }
}

fn collect_scaled_trig_square_terms_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    scale: BigRational,
    linear_coeff: &mut BigRational,
    sin_terms: &mut Vec<(ExprId, BigRational)>,
) -> Option<()> {
    if scale.is_zero() {
        return Some(());
    }

    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(value) if value.is_zero() => Some(()),
        Expr::Add(left, right) => {
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                left,
                var_name,
                scale.clone(),
                linear_coeff,
                sin_terms,
            )?;
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                right,
                var_name,
                scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Sub(left, right) => {
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                left,
                var_name,
                scale.clone(),
                linear_coeff,
                sin_terms,
            )?;
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                right,
                var_name,
                -scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Neg(inner) => collect_scaled_trig_square_terms_for_integration_presentation(
            ctx,
            inner,
            var_name,
            -scale,
            linear_coeff,
            sin_terms,
        ),
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                num,
                var_name,
                scale / den_scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Mul(_, _) => {
            let mut term_scale = scale;
            let mut non_numeric = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    term_scale *= value;
                    continue;
                }
                if non_numeric.replace(factor).is_some() {
                    return None;
                }
            }
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                non_numeric?,
                var_name,
                term_scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var_name => {
            *linear_coeff += scale;
            Some(())
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sin) =>
        {
            sin_terms.push((args[0], scale));
            Some(())
        }
        _ => None,
    }
}

fn compact_trig_square_reduction_primitive_for_integration_presentation(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut linear_coeff = BigRational::zero();
    let mut sin_terms = Vec::new();
    collect_scaled_trig_square_terms_for_integration_presentation(
        ctx,
        result,
        var_name,
        BigRational::one(),
        &mut linear_coeff,
        &mut sin_terms,
    )?;

    if linear_coeff != BigRational::new(1.into(), 2.into()) {
        return None;
    }

    let (sin_arg, sin_coeff) = sin_terms
        .iter()
        .find_map(|(arg, coeff)| (!coeff.is_zero()).then_some((*arg, coeff.clone())))?;
    if sin_terms.iter().any(|(arg, coeff)| {
        !coeff.is_zero() && compare_expr(ctx, *arg, sin_arg) != std::cmp::Ordering::Equal
    }) {
        return None;
    }

    let var = ctx.var(var_name);
    let linear = scale_expr_for_calculus_presentation(ctx, linear_coeff, var);
    let sin = ctx.call_builtin(BuiltinFn::Sin, vec![sin_arg]);
    let trig = scale_expr_for_calculus_presentation(ctx, sin_coeff, sin);
    let compact = ctx.add(Expr::Add(linear, trig));

    (compact != result).then_some(compact)
}

fn trig_power_term_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, u32)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    if let Expr::Function(fn_id, args) = ctx.get(expr).clone() {
        if args.len() != 1 {
            return None;
        }
        return match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Sin | BuiltinFn::Cos) => Some((ctx.builtin_of(fn_id)?, args[0], 1)),
            _ => None,
        };
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let power = cas_ast::views::as_rational_const(ctx, exp, 4)?;
    if power.denom() != &1.into() {
        return None;
    }
    let power = power.numer().to_u32()?;
    if !matches!(power, 3 | 5) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin | BuiltinFn::Cos) => Some((ctx.builtin_of(fn_id)?, args[0], power)),
        _ => None,
    }
}

fn collect_scaled_trig_power_terms_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    scale: BigRational,
    terms: &mut Vec<(BuiltinFn, ExprId, u32, BigRational)>,
) -> Option<()> {
    if scale.is_zero() {
        return Some(());
    }

    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(value) if value.is_zero() => Some(()),
        Expr::Add(left, right) => {
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                left,
                scale.clone(),
                terms,
            )?;
            collect_scaled_trig_power_terms_for_integration_presentation(ctx, right, scale, terms)
        }
        Expr::Sub(left, right) => {
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                left,
                scale.clone(),
                terms,
            )?;
            collect_scaled_trig_power_terms_for_integration_presentation(ctx, right, -scale, terms)
        }
        Expr::Neg(inner) => {
            collect_scaled_trig_power_terms_for_integration_presentation(ctx, inner, -scale, terms)
        }
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                num,
                scale / den_scale,
                terms,
            )
        }
        Expr::Mul(_, _) => {
            let mut term_scale = scale;
            let mut non_numeric = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    term_scale *= value;
                    continue;
                }
                if non_numeric.replace(factor).is_some() {
                    return None;
                }
            }
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                non_numeric?,
                term_scale,
                terms,
            )
        }
        _ => {
            let (builtin, arg, power) = trig_power_term_for_integration_presentation(ctx, expr)?;
            terms.push((builtin, arg, power, scale));
            Some(())
        }
    }
}

fn trig_power_presentation_coeff(
    terms: &[(BuiltinFn, ExprId, u32, BigRational)],
    power: u32,
) -> BigRational {
    terms
        .iter()
        .filter_map(|(_, _, term_power, coeff)| (*term_power == power).then_some(coeff.clone()))
        .fold(BigRational::zero(), |acc, coeff| acc + coeff)
}

fn compact_trig_odd_power_reduction_primitive_for_integration_presentation(
    ctx: &mut Context,
    result: ExprId,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_trig_power_terms_for_integration_presentation(
        ctx,
        result,
        BigRational::one(),
        &mut terms,
    )?;
    if terms.len() < 2 {
        return None;
    }

    let (builtin, arg) = terms
        .iter()
        .find_map(|(builtin, arg, _, coeff)| (!coeff.is_zero()).then_some((*builtin, *arg)))?;
    if terms.iter().any(|(term_builtin, term_arg, power, coeff)| {
        !coeff.is_zero()
            && (*term_builtin != builtin
                || compare_expr(ctx, *term_arg, arg) != std::cmp::Ordering::Equal
                || !matches!(*power, 1 | 3 | 5))
    }) {
        return None;
    }

    let linear_coeff = trig_power_presentation_coeff(&terms, 1);
    if linear_coeff.is_zero() {
        return None;
    }

    let scale = match builtin {
        BuiltinFn::Cos if linear_coeff.is_negative() => -linear_coeff.clone(),
        BuiltinFn::Sin if linear_coeff.is_positive() => linear_coeff.clone(),
        _ => return None,
    };
    if scale.is_zero() {
        return None;
    }

    let cubic_coeff = trig_power_presentation_coeff(&terms, 3);
    if cubic_coeff.is_zero() {
        return None;
    }
    let fifth_coeff = trig_power_presentation_coeff(&terms, 5);
    let has_fifth = !fifth_coeff.is_zero();

    let one_third = BigRational::new(1.into(), 3.into());
    let two_thirds = BigRational::new(2.into(), 3.into());
    let one_fifth = BigRational::new(1.into(), 5.into());
    let expected_cubic = if has_fifth {
        match builtin {
            BuiltinFn::Cos => two_thirds.clone() * scale.clone(),
            BuiltinFn::Sin => -two_thirds.clone() * scale.clone(),
            _ => return None,
        }
    } else {
        match builtin {
            BuiltinFn::Cos => one_third.clone() * scale.clone(),
            BuiltinFn::Sin => -one_third.clone() * scale.clone(),
            _ => return None,
        }
    };
    let expected_fifth = if has_fifth {
        match builtin {
            BuiltinFn::Cos => -one_fifth.clone() * scale.clone(),
            BuiltinFn::Sin => one_fifth.clone() * scale.clone(),
            _ => return None,
        }
    } else {
        BigRational::zero()
    };

    if cubic_coeff != expected_cubic || fifth_coeff != expected_fifth {
        return None;
    }

    let three = ctx.num(3);
    let five = ctx.num(5);
    let base = ctx.call_builtin(builtin, vec![arg]);
    let base_cubed = ctx.add(Expr::Pow(base, three));
    let base_fifth = ctx.add(Expr::Pow(base, five));

    let linear = scale_expr_for_calculus_presentation(ctx, linear_coeff, base);
    let cubic = scale_expr_for_calculus_presentation(ctx, expected_cubic, base_cubed);

    let compact = if has_fifth {
        let fifth = scale_expr_for_calculus_presentation(ctx, expected_fifth, base_fifth);
        match builtin {
            BuiltinFn::Cos => {
                let first_two = ctx.add(Expr::Add(cubic, linear));
                ctx.add(Expr::Add(first_two, fifth))
            }
            BuiltinFn::Sin => {
                let first_two = ctx.add(Expr::Add(linear, fifth));
                ctx.add(Expr::Add(first_two, cubic))
            }
            _ => return None,
        }
    } else {
        match builtin {
            BuiltinFn::Cos => ctx.add(Expr::Add(cubic, linear)),
            BuiltinFn::Sin => ctx.add(Expr::Add(linear, cubic)),
            _ => return None,
        }
    };

    (compact != result).then_some(compact)
}

fn half_power_term_for_integration_presentation(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<(ExprId, BigRational, u32)> {
    let mut coefficient = BigRational::one();
    let mut base_and_offset = None;

    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            coefficient *= value;
            continue;
        }

        let (base, power) = match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
            {
                (args[0], BigRational::new(1.into(), 2.into()))
            }
            Expr::Pow(base, exp) => {
                let power = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                (*base, power)
            }
            _ => return None,
        };

        polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
        let offset = power - BigRational::new(1.into(), 2.into());
        if !offset.is_integer() || offset.is_negative() {
            return None;
        }
        let offset = offset.to_integer().to_u32()?;
        if offset > 4 || base_and_offset.replace((base, offset)).is_some() {
            return None;
        }
    }

    if sign == cas_math::expr_nary::Sign::Neg {
        coefficient = -coefficient;
    }

    let (base, offset) = base_and_offset?;
    (!coefficient.is_zero()).then_some((base, coefficient, offset))
}

fn compact_half_power_sum_root_product_for_integration_presentation(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, result).terms;
    if terms.len() < 2 || terms.len() > 4 {
        return None;
    }

    let mut common_base = None;
    let mut polynomial_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (base, coefficient, offset) =
            half_power_term_for_integration_presentation(ctx, term, sign, var_name)?;
        if let Some(existing) = common_base {
            if compare_expr(ctx, existing, base) != std::cmp::Ordering::Equal {
                return None;
            }
        } else {
            common_base = Some(base);
        }

        let power_free = match offset {
            0 => ctx.num(1),
            1 => base,
            _ => {
                let exp = ctx.num(offset as i64);
                ctx.add(Expr::Pow(base, exp))
            }
        };
        polynomial_terms.push(scale_expr_for_calculus_presentation(
            ctx,
            coefficient,
            power_free,
        ));
    }

    let base = common_base?;
    let raw_polynomial = cas_math::expr_nary::build_balanced_add(ctx, &polynomial_terms);
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };
    let polynomial = multipoly_from_expr(ctx, raw_polynomial, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw_polynomial);
    let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_base, polynomial],
    ))
}

fn signed_elementary_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Neg(inner) = ctx.get(target).clone() else {
        return elementary_sqrt_polynomial_derivative_presentation(ctx, target, var_name);
    };

    let compact = elementary_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)?;
    Some(negate_calculus_presentation(ctx, compact))
}

fn negate_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inner,
        Expr::Div(numerator, denominator) => {
            let numerator = match ctx.get(numerator).clone() {
                Expr::Neg(inner) => inner,
                _ => ctx.add(Expr::Neg(numerator)),
            };
            ctx.add(Expr::Div(numerator, denominator))
        }
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn reciprocal_trig_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (arg, first, second, sign) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            let (first, second, sign) = match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sec) => (BuiltinFn::Sec, BuiltinFn::Tan, BigRational::one()),
                Some(BuiltinFn::Csc) => (BuiltinFn::Csc, BuiltinFn::Cot, -BigRational::one()),
                _ => return None,
            };
            (args[0], first, second, sign)
        }
        Expr::Div(numerator, denominator)
            if cas_ast::views::as_rational_const(ctx, numerator, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            let Expr::Function(den_fn_id, den_args) = ctx.get(denominator).clone() else {
                return None;
            };
            if den_args.len() != 1 {
                return None;
            }
            let (first, second, sign) = match ctx.builtin_of(den_fn_id) {
                Some(BuiltinFn::Cos) => (BuiltinFn::Sec, BuiltinFn::Tan, BigRational::one()),
                Some(BuiltinFn::Sin) => (BuiltinFn::Csc, BuiltinFn::Cot, -BigRational::one()),
                _ => return None,
            };
            (den_args[0], first, second, sign)
        }
        _ => return None,
    };
    let slope = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let coeff = sign * slope;

    let first_arg = ctx.call_builtin(first, vec![arg]);
    let second_arg = ctx.call_builtin(second, vec![arg]);
    let table_core = cas_math::expr_nary::build_balanced_mul(ctx, &[first_arg, second_arg]);
    Some(signed_numerator_for_calculus_presentation(
        ctx, coeff, table_core,
    ))
}

fn scaled_reciprocal_trig_power_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(target).clone() else {
        return None;
    };
    let denominator_scale = cas_ast::views::as_rational_const(ctx, denominator, 8)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let (outer_sign, core) = match ctx.get(numerator).clone() {
        Expr::Neg(inner) => (-BigRational::one(), inner),
        _ => (BigRational::one(), numerator),
    };
    let Expr::Pow(base, exp) = ctx.get(core).clone() else {
        return None;
    };
    let power = cas_ast::views::as_rational_const(ctx, exp, 8)?;
    if !power.is_integer() {
        return None;
    }
    let power = power.to_integer().to_i64()?;
    if !(2..=6).contains(&power) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    let (reciprocal_square_builtin, derivative_sign) = match builtin {
        BuiltinFn::Tan => (BuiltinFn::Sec, BigRational::one()),
        BuiltinFn::Cot => (BuiltinFn::Csc, -BigRational::one()),
        _ => return None,
    };
    let arg = args[0];
    let slope = nonzero_affine_variable_derivative(ctx, arg, var_name)?;

    let base_power = if power == 2 {
        ctx.call_builtin(builtin, vec![arg])
    } else {
        let base_call = ctx.call_builtin(builtin, vec![arg]);
        let next_power = ctx.num(power - 1);
        ctx.add(Expr::Pow(base_call, next_power))
    };
    let reciprocal = ctx.call_builtin(reciprocal_square_builtin, vec![arg]);
    let reciprocal_square = squared_expr(ctx, reciprocal);
    let table_core = cas_math::expr_nary::build_balanced_mul(ctx, &[base_power, reciprocal_square]);
    let coeff = outer_sign * derivative_sign * slope * BigRational::from_integer(power.into())
        / denominator_scale;

    Some(signed_numerator_for_calculus_presentation(
        ctx, coeff, table_core,
    ))
}

fn squared_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
}

fn atanh_arg_over_sqrt_parts(ctx: &mut Context, arg: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg) {
        Expr::Div(num, den) => {
            let radicand = extract_square_root_base(ctx, *den)?;
            Some((*num, radicand))
        }
        Expr::Mul(_, _) => rationalized_arg_over_sqrt_parts(ctx, arg),
        _ => None,
    }
}

fn arg_over_scaled_sqrt_parts(ctx: &mut Context, arg: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, den);
    if factors.len() < 2 {
        return None;
    }

    let mut rational_scale = BigRational::one();
    let mut radicand = None;
    for factor in factors {
        if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
            continue;
        }
        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        rational_scale *= value;
    }

    if !rational_scale.is_positive() {
        return None;
    }
    let scale_square = &rational_scale * &rational_scale;
    let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
    let radicand = ctx.add(Expr::Mul(scale_square_expr, radicand?));
    Some((num, radicand))
}

fn rationalized_arg_over_sqrt_parts(ctx: &mut Context, arg: ExprId) -> Option<(ExprId, ExprId)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    if factors.len() < 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().enumerate() {
        let Some(radicand) = extract_square_root_base(ctx, *sqrt_factor) else {
            continue;
        };
        let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
        if !radicand_value.is_positive() {
            return None;
        }

        let mut rational_scale = BigRational::one();
        let mut numerator_factors = Vec::new();
        for (factor_index, factor) in factors.iter().enumerate() {
            if factor_index == sqrt_index {
                continue;
            }

            if let Some(value) = cas_ast::views::as_rational_const(ctx, *factor, 8) {
                rational_scale *= value;
            } else {
                numerator_factors.push(*factor);
            }
        }

        if numerator_factors.is_empty() {
            continue;
        }

        if rational_scale * &radicand_value == BigRational::one() {
            let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
            return Some((numerator, radicand));
        }
    }

    None
}

fn atanh_open_interval_condition(ctx: &mut Context, arg: ExprId) -> ExprId {
    if let Some((num, radicand)) = atanh_arg_over_sqrt_parts(ctx, arg) {
        let num_square = squared_expr(ctx, num);
        return ctx.add(Expr::Sub(radicand, num_square));
    }

    if let Some(radicand) = extract_square_root_base(ctx, arg) {
        return subtract_from_one_for_calculus_presentation(ctx, radicand);
    }

    let arg_sq = squared_expr(ctx, arg);
    subtract_from_one_for_calculus_presentation(ctx, arg_sq)
}

fn primitive_positive_gap(ctx: &mut Context, gap: ExprId) -> (ExprId, BigRational) {
    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let Ok(poly) = multipoly_from_expr(ctx, gap, &budget) else {
        return (gap, BigRational::one());
    };
    let (content, primitive) = poly.primitive_part();
    if !content.is_positive() {
        return (gap, BigRational::one());
    }
    let primitive_expr = multipoly_to_expr(&primitive, ctx);
    if content.is_one() {
        return (primitive_expr, BigRational::one());
    }

    (primitive_expr, content)
}

fn reciprocal_positive_rational(value: &BigRational) -> BigRational {
    BigRational::new(value.denom().clone(), value.numer().clone())
}

fn shifted_sqrt_positive_constant_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    let left_sqrt_base = extract_square_root_base(ctx, *left);
    let right_sqrt_base = extract_square_root_base(ctx, *right);

    let (radicand, shift_expr) = match (left_sqrt_base, right_sqrt_base) {
        (Some(radicand), None) => (radicand, *right),
        (None, Some(radicand)) => (radicand, *left),
        _ => return None,
    };

    let shift = cas_ast::views::as_rational_const(ctx, shift_expr, 8)?;
    shift.is_positive().then_some((radicand, shift))
}

fn reciprocal_positive_shifted_sqrt_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    if numerator_scale.is_zero() {
        return Some(ctx.num(0));
    }

    let (radicand, shift) = shifted_sqrt_positive_constant_parts(ctx, den)?;
    let d_radicand = differentiate(ctx, radicand, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_radicand, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift);
    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt_squared]);

    if let Some(d_radicand_scale) = cas_ast::views::as_rational_const(ctx, d_radicand, 8) {
        let coefficient = -numerator_scale * d_radicand_scale / BigRational::from_integer(2.into());
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        let denominator = if denominator_coeff == BigRational::one() {
            core_denominator
        } else {
            let denominator_scale =
                rational_const_for_calculus_presentation(ctx, denominator_coeff);
            cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
        };

        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let negative_scale = ctx.add(Expr::Number(-numerator_scale));
    let numerator = ctx.add(Expr::Mul(negative_scale, d_radicand));
    let two = ctx.num(2);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, core_denominator]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    if numerator_scale.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let (radicand, shift) = sqrt_times_nonzero_shifted_sqrt_parts(ctx, den)?;
    let d_radicand = differentiate(ctx, radicand, var_name)?;
    let d_radicand_scale = cas_ast::views::as_rational_const(ctx, d_radicand, 8)?;
    if d_radicand_scale.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let two = ctx.num(2);
    let two_sqrt = ctx.add(Expr::Mul(two, sqrt_radicand));
    let shift_is_positive = shift.is_positive();
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift.clone());
    let numerator_core = ctx.add(Expr::Add(two_sqrt, shift_expr));
    let coefficient = -numerator_scale * d_radicand_scale;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let mut denominator_parts = Vec::new();
    let denominator_scale = denominator_coeff * BigRational::from_integer(2.into());
    if denominator_scale != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_scale,
        ));
    }
    denominator_parts.push(radicand);
    denominator_parts.push(sqrt_radicand);
    denominator_parts.push(shifted_sqrt_squared);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    let result = ctx.add(Expr::Div(numerator, denominator));
    let mut required_conditions = vec![crate::ImplicitCondition::Positive(radicand)];
    if !shift_is_positive {
        required_conditions.push(crate::ImplicitCondition::NonZero(shifted_sqrt));
    }

    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

fn sqrt_times_nonzero_shifted_sqrt_parts(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let factors: Vec<_> = cas_math::expr_nary::mul_leaves(ctx, expr)
        .into_iter()
        .collect();
    if factors.len() != 2 {
        return None;
    }

    for (sqrt_factor, shifted_factor) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        let sqrt_radicand = extract_square_root_base(ctx, sqrt_factor)?;
        let (shifted_radicand, shift) = supported_sqrt_shift_constant_parts(ctx, shifted_factor)?;
        let shifted_radicand = extract_square_root_base(ctx, shifted_radicand)?;
        if !shift.is_zero()
            && cas_math::expr_domain::exprs_equivalent(ctx, sqrt_radicand, shifted_radicand)
        {
            return Some((sqrt_radicand, shift));
        }
    }

    None
}

fn bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    let sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => 1,
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -1,
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])
        .or_else(|| sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation(ctx, args[0]))?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = polynomial_derivative_expr_for_calculus_presentation(ctx, num, var_name)
        .or_else(|| differentiate(ctx, num, var_name))?;
    let (d_num_core, d_num_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_num);
    let numerator_content = if sign < 0 {
        -d_num_content
    } else {
        d_num_content
    };
    let radicand_numer = BigRational::from_integer(radicand_value.numer().clone());
    let radicand_denom = BigRational::from_integer(radicand_value.denom().clone());
    let numerator = if radicand_denom.is_one() {
        signed_numerator_for_calculus_presentation(ctx, numerator_content.clone(), d_num_core)
    } else if let Some(sqrt_content) =
        exact_positive_rational_sqrt_for_calculus_presentation(&radicand_denom)
    {
        signed_numerator_for_calculus_presentation(
            ctx,
            numerator_content * sqrt_content,
            d_num_core,
        )
    } else {
        let base_numerator =
            signed_numerator_for_calculus_presentation(ctx, numerator_content, d_num_core);
        scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            radicand_denom.clone(),
            base_numerator,
        )
    };
    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let scaled_num_square = if radicand_denom.is_one() {
        num_square
    } else {
        scale_expr_for_calculus_presentation(ctx, radicand_denom, num_square)
    };
    let compact_numer = rational_const_for_calculus_presentation(ctx, radicand_numer);
    let gap = ctx.add(Expr::Sub(compact_numer, scaled_num_square));
    let gap = compact_squared_affine_gap_for_calculus_presentation(ctx, gap, var_name);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let compact = ctx.add(Expr::Div(numerator, denominator));
    let compact = fold_numeric_mul_constants_for_hold(ctx, compact);

    Some(compact)
}

fn constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(inner, outer_den) = ctx.get(target).clone() else {
        return None;
    };
    if contains_named_var(ctx, outer_den, var_name) {
        return None;
    }

    let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
    let inner_derivative =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
            .or_else(|| arctan_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))?;
    Some(divide_compact_derivative_by_constant_factor(
        ctx,
        inner_derivative,
        outer_den,
    ))
}

fn remove_unit_mul_factors_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return expr;
    };

    let mut non_unit_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if cas_ast::views::as_rational_const(ctx, factor, 8) == Some(BigRational::one()) {
            continue;
        }
        non_unit_factors.push(factor);
    }

    match non_unit_factors.as_slice() {
        [single] => *single,
        _ => expr,
    }
}

fn reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    for idx in 0..factors.len() {
        let inner = factors[idx];
        let mut constant_factors = factors.clone();
        constant_factors.remove(idx);
        let [constant_factor] = constant_factors.as_slice() else {
            continue;
        };
        let Some(outer_den) = reciprocal_constant_denominator_for_calculus_presentation(
            ctx,
            *constant_factor,
            var_name,
        ) else {
            continue;
        };
        let Some(inner_derivative) =
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
                .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
                .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))
        else {
            continue;
        };
        return Some(divide_compact_derivative_by_constant_factor(
            ctx,
            inner_derivative,
            outer_den,
        ));
    }

    None
}

fn constant_scaled_arctan_surd_quotient_scaled_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Expr::Div(numerator, outer_den) = ctx.get(target).clone() {
        let (scale, inner) = rational_scaled_single_factor(ctx, numerator)?;
        let base = ctx.add(Expr::Div(inner, outer_den));
        let derivative = arctan_surd_quotient_scaled_compact_derivative(ctx, base, var_name)?;
        return Some(scale_compact_derivative_by_rational(ctx, derivative, scale));
    }

    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = arctan_surd_quotient_scaled_compact_derivative(ctx, inner, var_name)?;
    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

fn constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, inner) = rational_scaled_single_factor_allow_unit(ctx, target)?;
    let (derivative, required_conditions) =
        inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
            ctx, inner, var_name,
        )
        .or_else(|| {
            inverse_tangent_reciprocal_sqrt_shifted_sqrt_product_derivative_presentation(
                ctx, inner, var_name,
            )
        })?;
    let derivative = if scale.is_one() {
        unwrap_internal_hold_for_calculus(ctx, derivative)
    } else {
        scale_compact_derivative_by_rational(ctx, derivative, scale)
    };
    Some((ctx.add(Expr::Hold(derivative)), required_conditions))
}

fn rational_scaled_single_factor_allow_unit(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (inner_scale, inner_factor) = rational_scaled_single_factor_allow_unit(ctx, inner)
            .unwrap_or_else(|| (BigRational::one(), inner));
        return Some((-inner_scale, inner_factor));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut non_numeric = Vec::new();

    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    let [inner] = non_numeric.as_slice() else {
        return None;
    };

    Some((scale, *inner))
}

fn rational_scaled_single_factor(ctx: &mut Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (inner_scale, inner_factor) = rational_scaled_single_factor_allow_unit(ctx, inner)
            .unwrap_or_else(|| (BigRational::one(), inner));
        let scale = -inner_scale;
        if scale.is_one() {
            return None;
        }
        return Some((scale, inner_factor));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut non_numeric = Vec::new();

    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    if scale.is_one() {
        return None;
    }

    let [inner] = non_numeric.as_slice() else {
        return None;
    };

    Some((scale, *inner))
}

fn scale_compact_derivative_by_rational(
    ctx: &mut Context,
    derivative: ExprId,
    scale: BigRational,
) -> ExprId {
    if scale.is_one() {
        return derivative;
    }

    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    let scaled = if let Expr::Div(num, den) = ctx.get(derivative).clone() {
        let (num, den) = scale_fraction_for_calculus_presentation(ctx, num, den, scale);
        if let Some(compact) = compact_division_by_positive_denominator_content(ctx, num, den) {
            return compact;
        }
        ctx.add(Expr::Div(num, den))
    } else {
        scale_expr_for_calculus_presentation(ctx, scale, derivative)
    };

    fold_numeric_mul_constants_for_hold(ctx, scaled)
}

fn scale_fraction_for_calculus_presentation(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    scale: BigRational,
) -> (ExprId, ExprId) {
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&scale).unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator);
    let denominator = if denominator_coeff.is_one() {
        denominator
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, denominator])
    };

    (numerator, denominator)
}

fn compact_division_by_positive_denominator_content(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<ExprId> {
    let numerator_value = signed_rational_const_for_calculus_presentation(ctx, num)?;
    let (den_core, den_content) = split_polynomial_content_for_calculus_presentation(ctx, den);
    if !den_content.is_positive() || den_content.is_one() || den_core == den {
        return None;
    }

    let scaled_numerator = numerator_value / den_content;
    let numerator = rational_const_for_calculus_presentation(
        ctx,
        BigRational::from_integer(scaled_numerator.numer().clone()),
    );
    if scaled_numerator.denom().is_one() {
        return Some(ctx.add(Expr::Div(numerator, den_core)));
    }

    let denominator_scale = BigRational::from_integer(scaled_numerator.denom().clone());
    let denominator = scale_expr_for_calculus_presentation(ctx, denominator_scale, den_core);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        if contains_named_var(ctx, outer_den, var_name) {
            return None;
        }

        let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
        let inner_derivative =
            inverse_reciprocal_trig_affine_abs_presentation(ctx, inner, var_name)?;
        let required_conditions =
            inverse_reciprocal_trig_affine_abs_required_conditions(ctx, inner, var_name);
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    for idx in 0..factors.len() {
        let inner = factors[idx];
        let mut constant_factors = factors.clone();
        constant_factors.remove(idx);
        let [constant_factor] = constant_factors.as_slice() else {
            continue;
        };
        let Some(outer_den) = reciprocal_constant_denominator_for_calculus_presentation(
            ctx,
            *constant_factor,
            var_name,
        ) else {
            continue;
        };
        let Some(inner_derivative) =
            inverse_reciprocal_trig_affine_abs_presentation(ctx, inner, var_name)
        else {
            continue;
        };
        let required_conditions =
            inverse_reciprocal_trig_affine_abs_required_conditions(ctx, inner, var_name);
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

    None
}

fn constant_scaled_acosh_affine_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        if contains_named_var(ctx, outer_den, var_name) {
            return None;
        }

        let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
        let (inner_derivative, required_conditions) =
            acosh_affine_derivative_presentation(ctx, inner, var_name)?;
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    for idx in 0..factors.len() {
        let inner = factors[idx];
        let mut constant_factors = factors.clone();
        constant_factors.remove(idx);
        let [constant_factor] = constant_factors.as_slice() else {
            continue;
        };
        let Some(outer_den) = reciprocal_constant_denominator_for_calculus_presentation(
            ctx,
            *constant_factor,
            var_name,
        ) else {
            continue;
        };
        let Some((inner_derivative, required_conditions)) =
            acosh_affine_derivative_presentation(ctx, inner, var_name)
        else {
            continue;
        };
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

    None
}

fn reciprocal_constant_denominator_for_calculus_presentation(
    ctx: &mut Context,
    factor: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if contains_named_var(ctx, factor, var_name) {
        return None;
    }

    match ctx.get(factor).clone() {
        Expr::Number(value) if value.numer() == &BigInt::from(1) && !value.is_zero() => {
            Some(ctx.add(Expr::Number(BigRational::from_integer(
                value.denom().clone(),
            ))))
        }
        Expr::Div(numerator, denominator) => {
            let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
            if numerator_value == BigRational::one() {
                Some(denominator)
            } else {
                None
            }
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            Some(base)
        }
        _ => None,
    }
}

fn divide_compact_derivative_by_constant_factor(
    ctx: &mut Context,
    derivative: ExprId,
    outer_den: ExprId,
) -> ExprId {
    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    if let Expr::Div(num, den) = ctx.get(derivative).clone() {
        if let Some(cancelled_num) = remove_matching_sqrt_like_product_factor(ctx, num, outer_den) {
            return ctx.add(Expr::Div(cancelled_num, den));
        }

        let combined_den = cas_math::expr_nary::build_balanced_mul(ctx, &[outer_den, den]);
        return ctx.add(Expr::Div(num, combined_den));
    }

    ctx.add(Expr::Div(derivative, outer_den))
}

fn remove_matching_sqrt_like_product_factor(
    ctx: &mut Context,
    expr: ExprId,
    factor: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let cancelled = remove_matching_sqrt_like_product_factor(ctx, inner, factor)?;
        return Some(negate_calculus_presentation(ctx, cancelled));
    }

    if same_sqrt_like_argument(ctx, expr, factor) {
        return Some(ctx.num(1));
    }

    let mut factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    for idx in 0..factors.len() {
        if same_sqrt_like_argument(ctx, factors[idx], factor) {
            factors.remove(idx);
            return Some(match factors.as_slice() {
                [] => ctx.num(1),
                [single] => *single,
                _ => cas_math::expr_nary::build_balanced_mul(ctx, &factors),
            });
        }
    }

    None
}

fn asinh_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Asinh) || args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = polynomial_derivative_expr_for_calculus_presentation(ctx, num, var_name)
        .or_else(|| differentiate(ctx, num, var_name))?;
    let (d_num_core, d_num_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_num);
    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let raw_gap = ctx.add(Expr::Add(radicand, num_square));
    let (positive_gap, gap_content) = if radicand_value.is_integer() {
        (raw_gap, BigRational::one())
    } else {
        primitive_positive_gap(ctx, raw_gap)
    };
    let numerator_scale = if gap_content.is_one() {
        d_num_content
    } else if let Some(sqrt_content) =
        exact_positive_rational_sqrt_for_calculus_presentation(&gap_content)
    {
        d_num_content / sqrt_content
    } else {
        let reciprocal_content = reciprocal_positive_rational(&gap_content);
        let numerator = signed_numerator_for_calculus_presentation(ctx, d_num_content, d_num_core);
        let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            reciprocal_content,
            numerator,
        );
        let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![positive_gap]);
        let compact = ctx.add(Expr::Div(numerator, denominator));
        let compact = fold_numeric_mul_constants_for_hold(ctx, compact);
        return Some(compact);
    };
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_scale, d_num_core);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![positive_gap]);
    let compact = ctx.add(Expr::Div(numerator, denominator));
    let compact = fold_numeric_mul_constants_for_hold(ctx, compact);

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn squared_expr_for_compact_gap_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return squared_expr(ctx, expr);
    };
    let Some(exp_value) = cas_ast::views::as_rational_const(ctx, exp, 8) else {
        return squared_expr(ctx, expr);
    };
    let doubled_exp = exp_value * BigRational::from_integer(2.into());
    let doubled_exp = ctx.add(Expr::Number(doubled_exp));
    ctx.add(Expr::Pow(base, doubled_exp))
}

fn arctan_surd_quotient_scaled_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (arctan_expr, outer_den) = match ctx.get(target).clone() {
        Expr::Div(arctan_expr, outer_den) => (arctan_expr, outer_den),
        _ => return None,
    };
    let outer_radicand = extract_square_root_base(ctx, outer_den)?;
    let outer_radicand_value = cas_ast::views::as_rational_const(ctx, outer_radicand, 8)?;
    if !outer_radicand_value.is_positive() {
        return None;
    }

    let (fn_id, args) = match ctx.get(arctan_expr).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (num, inner_radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    if compare_expr(ctx, outer_radicand, inner_radicand) != std::cmp::Ordering::Equal {
        return None;
    }

    let (d_num, square_base) = if let Some(parts) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)
    {
        parts
    } else {
        (differentiate(ctx, num, var_name)?, num)
    };
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let num_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let denominator = ctx.add(Expr::Add(outer_radicand, num_square));
    Some(ctx.add(Expr::Div(d_num, denominator)))
}

fn multiply_by_sqrt_factor_for_calculus_presentation(
    ctx: &mut Context,
    factor: ExprId,
    radicand: ExprId,
) -> ExprId {
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
        if value.is_one() {
            return sqrt_radicand;
        }
        if value == -BigRational::one() {
            return ctx.add(Expr::Neg(sqrt_radicand));
        }
    }

    cas_math::expr_nary::build_balanced_mul(ctx, &[factor, sqrt_radicand])
}

fn arctan_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let (d_num, square_base) = if let Some(parts) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)
    {
        parts
    } else {
        let d_num = polynomial_derivative_expr_for_calculus_presentation(ctx, num, var_name)
            .or_else(|| differentiate(ctx, num, var_name))?;
        let (d_num_core, d_num_content) =
            split_polynomial_content_for_calculus_presentation(ctx, d_num);
        let d_num = signed_numerator_for_calculus_presentation(ctx, d_num_content, d_num_core);
        (d_num, num)
    };
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let num_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let denominator = ctx.add(Expr::Add(radicand, num_square));
    let compact_radicand = rational_const_for_calculus_presentation(ctx, radicand_value);
    let numerator = multiply_by_sqrt_factor_for_calculus_presentation(ctx, d_num, compact_radicand);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn arctan_self_normalized_surd_quotient_parts(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let Some(parts) = atanh_arg_over_sqrt_parts(ctx, arg) {
        return Some(parts);
    }
    if let Some(parts) = arg_over_scaled_sqrt_parts(ctx, arg) {
        return Some(parts);
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    if factors.len() < 2 {
        return None;
    }

    let neg_half = BigRational::new((-1).into(), 2.into());
    let mut radicand = None;
    let mut numerator_factors = Vec::new();
    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(base, exp)
                if cas_ast::views::as_rational_const(ctx, *exp, 8) == Some(neg_half.clone()) =>
            {
                if radicand.replace(*base).is_some() {
                    return None;
                }
            }
            _ => numerator_factors.push(factor),
        }
    }

    let radicand = radicand?;
    if numerator_factors.is_empty() {
        return None;
    }
    let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    Some((numerator, radicand))
}

fn atanh_self_normalized_surd_quotient_positive_gap(
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

fn arctan_self_normalized_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (num, radicand) = arctan_self_normalized_surd_quotient_parts(ctx, args[0])?;
    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let gap_poly = radicand_poly.sub(&num_poly.mul(&num_poly));
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_constant = gap_poly.coeffs.first().cloned()?;
    if !gap_constant.is_positive() {
        return None;
    }

    let (d_num, square_base) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }

    let numerator = scale_expr_for_calculus_presentation(ctx, gap_constant.clone(), d_num);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let num_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let doubled_square =
        scale_expr_for_calculus_presentation(ctx, BigRational::from_integer(2.into()), num_square);
    let gap = rational_const_for_calculus_presentation(ctx, gap_constant);
    let quadratic_factor = ctx.add(Expr::Add(doubled_square, gap));
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, quadratic_factor]);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn atanh_self_normalized_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return None;
    }

    let (num, radicand) = arctan_self_normalized_surd_quotient_parts(ctx, args[0])?;
    atanh_self_normalized_surd_quotient_positive_gap(ctx, args[0], var_name)?;

    let (d_num, _) = compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let compact = ctx.add(Expr::Div(d_num, sqrt_radicand));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn arctan_self_normalized_surd_reciprocal_parts(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg).clone() {
        Expr::Div(numerator, denominator) => {
            let radicand = extract_square_root_base(ctx, numerator)?;
            Some((denominator, radicand))
        }
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
            let mut radicand = None;
            let mut denominator = None;
            let neg_one = BigRational::new((-1).into(), 1.into());

            for factor in factors {
                if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
                    if radicand.replace(factor_radicand).is_some() {
                        return None;
                    }
                    continue;
                }
                match ctx.get(factor) {
                    Expr::Pow(base, exp)
                        if cas_ast::views::as_rational_const(ctx, *exp, 8)
                            == Some(neg_one.clone()) =>
                    {
                        if denominator.replace(*base).is_some() {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }

            Some((denominator?, radicand?))
        }
        _ => None,
    }
}

fn arctan_self_normalized_surd_reciprocal_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (denominator_arg, radicand) = arctan_self_normalized_surd_reciprocal_parts(ctx, args[0])?;
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_arg, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let gap_poly = radicand_poly.sub(&denominator_poly.mul(&denominator_poly));
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_constant = gap_poly.coeffs.first().cloned()?;
    if !gap_constant.is_positive() {
        return None;
    }

    let (d_denominator, square_base) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, denominator_arg, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_denominator, 8).is_some_and(|value| value.is_zero())
    {
        return Some((
            ctx.num(0),
            crate::ImplicitCondition::NonZero(denominator_arg),
        ));
    }

    let numerator = scale_expr_for_calculus_presentation(ctx, -gap_constant.clone(), d_denominator);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let doubled_square = scale_expr_for_calculus_presentation(
        ctx,
        BigRational::from_integer(2.into()),
        denominator_square,
    );
    let gap = rational_const_for_calculus_presentation(ctx, gap_constant);
    let quadratic_factor = ctx.add(Expr::Add(doubled_square, gap));
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, quadratic_factor]);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        crate::ImplicitCondition::NonZero(denominator_arg),
    ))
}

fn compact_surd_quotient_polynomial_presentation_parts(
    ctx: &mut Context,
    num: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    let mut derivative_poly = num_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), num));
    }

    let square_poly = if num_poly.leading_coeff().is_negative() {
        num_poly.neg()
    } else {
        num_poly
    };
    let square_base = square_poly.to_expr(ctx);

    let mut derivative_sign = BigRational::one();
    if derivative_poly.leading_coeff().is_negative() {
        derivative_poly = derivative_poly.neg();
        derivative_sign = -derivative_sign;
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let derivative = signed_numerator_for_calculus_presentation(
        ctx,
        derivative_sign * derivative_content,
        derivative_core,
    );

    Some((derivative, square_base))
}

struct ArctanAffineByPartsTerm {
    arg: ExprId,
    arg_poly: Polynomial,
    cofactor_poly: Polynomial,
}

struct LnAffineByPartsTerm {
    arg_poly: Polynomial,
    coefficient: BigRational,
}

fn apply_additive_sign_to_poly(poly: Polynomial, sign: cas_math::expr_nary::Sign) -> Polynomial {
    match sign {
        cas_math::expr_nary::Sign::Pos => poly,
        cas_math::expr_nary::Sign::Neg => poly.neg(),
    }
}

fn arctan_affine_by_parts_arctan_term(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<ArctanAffineByPartsTerm> {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut term = arctan_affine_by_parts_arctan_term(ctx, num, sign, var_name)?;
        term.cofactor_poly = term.cofactor_poly.div_scalar(&denominator);
        return Some(term);
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut arctan_arg = None;
    let mut cofactor_poly = Polynomial::one(var_name.to_string());

    for factor in factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                )
            {
                if arctan_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    let arg = arctan_arg?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }

    Some(ArctanAffineByPartsTerm {
        arg,
        arg_poly,
        cofactor_poly: apply_additive_sign_to_poly(cofactor_poly, sign),
    })
}

fn arctan_affine_by_parts_ln_term(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<LnAffineByPartsTerm> {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut term = arctan_affine_by_parts_ln_term(ctx, num, sign, var_name)?;
        term.coefficient /= denominator;
        return Some(term);
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut ln_arg = None;
    let mut coefficient_poly = Polynomial::one(var_name.to_string());

    for factor in factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        coefficient_poly = coefficient_poly.mul(&factor_poly);
    }

    let ln_arg = ln_arg?;
    let coefficient_poly = apply_additive_sign_to_poly(coefficient_poly, sign);
    if coefficient_poly.degree() != 0 {
        return None;
    }

    Some(LnAffineByPartsTerm {
        arg_poly: Polynomial::from_expr(ctx, ln_arg, var_name).ok()?,
        coefficient: coefficient_poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero),
    })
}

fn scale_polynomial(poly: &Polynomial, scale: &BigRational) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|coeff| coeff * scale).collect(),
        poly.var.clone(),
    )
}

fn polynomial_arctan_product(ctx: &mut Context, poly: &Polynomial, arg: ExprId) -> ExprId {
    if poly.is_zero() {
        return ctx.num(0);
    }

    let one = Polynomial::one(poly.var.clone());
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    if *poly == one {
        return arctan;
    }
    if *poly == one.neg() {
        return ctx.add(Expr::Neg(arctan));
    }

    let poly_expr = poly.to_expr(ctx);
    ctx.add(Expr::Mul(poly_expr, arctan))
}

fn arctan_affine_by_parts_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    match ctx.get(target).clone() {
        Expr::Mul(left, right) => {
            if cas_ast::views::as_rational_const(ctx, left, 8).is_some() {
                let derivative = arctan_affine_by_parts_compact_derivative(ctx, right, var_name)?;
                let scaled = ctx.add(Expr::Mul(left, derivative));
                return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
            }
            if cas_ast::views::as_rational_const(ctx, right, 8).is_some() {
                let derivative = arctan_affine_by_parts_compact_derivative(ctx, left, var_name)?;
                let scaled = ctx.add(Expr::Mul(derivative, right));
                return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
            }
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            let derivative = arctan_affine_by_parts_compact_derivative(ctx, num, var_name)?;
            let scaled = ctx.add(Expr::Div(derivative, den));
            return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
        }
        _ => {}
    }

    let terms = cas_math::expr_nary::AddView::from_expr(ctx, target).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut arctan_term: Option<ArctanAffineByPartsTerm> = None;
    let mut ln_term: Option<LnAffineByPartsTerm> = None;
    let mut remainder_poly = Polynomial::zero(var_name.to_string());

    for (term, sign) in terms {
        if let Some(term) = arctan_affine_by_parts_arctan_term(ctx, term, sign, var_name) {
            if let Some(existing) = &mut arctan_term {
                if existing.arg_poly != term.arg_poly {
                    return None;
                }
                existing.cofactor_poly = existing.cofactor_poly.add(&term.cofactor_poly);
            } else {
                arctan_term = Some(term);
            }
            continue;
        }

        if let Some(term) = arctan_affine_by_parts_ln_term(ctx, term, sign, var_name) {
            if let Some(existing) = &mut ln_term {
                if existing.arg_poly != term.arg_poly {
                    return None;
                }
                existing.coefficient += term.coefficient;
            } else {
                ln_term = Some(term);
            }
            continue;
        }

        let term_poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
        remainder_poly = remainder_poly.add(&apply_additive_sign_to_poly(term_poly, sign));
    }

    let arctan_term = arctan_term?;
    let ln_term = ln_term?;
    let derivative_poly = arctan_term.arg_poly.derivative();
    if derivative_poly.degree() != 0 || derivative_poly.is_zero() {
        return None;
    }
    let linear_coeff = derivative_poly.coeffs.first()?.clone();
    if linear_coeff.is_zero() {
        return None;
    }

    let expected_ln_arg_poly = arctan_term
        .arg_poly
        .mul(&arctan_term.arg_poly)
        .add(&Polynomial::one(var_name.to_string()));
    if ln_term.arg_poly != expected_ln_arg_poly {
        return None;
    }

    let rational_numerator = scale_polynomial(&arctan_term.cofactor_poly, &linear_coeff)
        .add(&scale_polynomial(
            &arctan_term.arg_poly,
            &(BigRational::from_integer(2.into()) * &ln_term.coefficient * &linear_coeff),
        ))
        .add(&remainder_poly.derivative().mul(&expected_ln_arg_poly));
    if !rational_numerator.is_zero() {
        return None;
    }

    let arctan_cofactor_derivative = arctan_term.cofactor_poly.derivative();
    Some(polynomial_arctan_product(
        ctx,
        &arctan_cofactor_derivative,
        arctan_term.arg,
    ))
}

fn collect_atanh_open_interval_conditions(ctx: &mut Context, root: ExprId) -> Vec<ExprId> {
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

fn atanh_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = polynomial_derivative_expr_for_calculus_presentation(ctx, num, var_name)
        .or_else(|| differentiate(ctx, num, var_name))?;
    let (d_num_core, d_num_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_num);
    let raw_denominator = atanh_open_interval_condition(ctx, args[0]);
    let (compact_denominator, denominator_content) = if radicand_value.is_integer() {
        (raw_denominator, BigRational::one())
    } else {
        primitive_positive_gap(ctx, raw_denominator)
    };
    let compact_denominator =
        compact_squared_affine_gap_for_calculus_presentation(ctx, compact_denominator, var_name);
    let d_num = signed_numerator_for_calculus_presentation(
        ctx,
        d_num_content / denominator_content,
        d_num_core,
    );
    let compact_sqrt_radicand = ctx.add(Expr::Number(radicand_value));
    let compact_sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![compact_sqrt_radicand]);
    let compact_numerator = ctx.add(Expr::Mul(compact_sqrt_radicand, d_num));
    let compact = ctx.add(Expr::Div(compact_numerator, compact_denominator));
    let compact = fold_numeric_mul_constants_for_hold(ctx, compact);

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn fold_numeric_mul_constants_for_hold(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(_, _) => {
            let mut factors = cas_math::expr_nary::mul_leaves(ctx, expr);
            let mut scale = BigRational::one();
            let mut non_numeric = Vec::new();

            while let Some(factor) = factors.pop() {
                let folded = fold_numeric_mul_constants_for_hold(ctx, factor);
                if matches!(ctx.get(folded), Expr::Mul(_, _)) {
                    factors.extend(cas_math::expr_nary::mul_leaves(ctx, folded));
                    continue;
                }
                if let Some(value) = rational_const_for_hold(ctx, folded) {
                    scale *= value;
                } else {
                    non_numeric.push(folded);
                }
            }

            if scale.is_zero() {
                return ctx.num(0);
            }

            if !scale.is_one() && non_numeric.len() == 1 {
                if let Some(radicand) = sqrt_positive_rational_for_hold(ctx, non_numeric[0]) {
                    let sign = if scale.is_negative() {
                        -BigRational::one()
                    } else {
                        BigRational::one()
                    };
                    let scaled_radicand = &scale * &scale * radicand;
                    let folded = if let Some(root) =
                        exact_positive_rational_sqrt_for_calculus_presentation(&scaled_radicand)
                    {
                        ctx.add(Expr::Number(sign * root))
                    } else {
                        let sqrt = sqrt_positive_rational_expr_for_calculus_presentation(
                            ctx,
                            scaled_radicand,
                        );
                        if sign.is_negative() {
                            negate_calculus_presentation(ctx, sqrt)
                        } else {
                            sqrt
                        }
                    };
                    return folded;
                }
                if let Expr::Div(num, den) = ctx.get(non_numeric[0]).clone() {
                    let scale_expr = ctx.add(Expr::Number(scale));
                    let scaled_num = ctx.add(Expr::Mul(scale_expr, num));
                    let folded_num = fold_numeric_mul_constants_for_hold(ctx, scaled_num);
                    return ctx.add(Expr::Div(folded_num, den));
                }
                if scale == -BigRational::one() {
                    if let Expr::Neg(inner) = ctx.get(non_numeric[0]).clone() {
                        return inner;
                    }
                }
            }

            if !scale.is_one() || non_numeric.is_empty() {
                non_numeric.insert(0, ctx.add(Expr::Number(scale)));
            }

            if non_numeric.len() == 1 {
                non_numeric[0]
            } else {
                cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric)
            }
        }
        Expr::Div(num, den) => {
            let num = fold_numeric_mul_constants_for_hold(ctx, num);
            let den = fold_numeric_mul_constants_for_hold(ctx, den);
            if let Some(den_value) = rational_const_for_hold(ctx, den) {
                if den_value.is_zero() {
                    return ctx.add(Expr::Div(num, den));
                }
                if let Some(num_value) = rational_const_for_hold(ctx, num) {
                    return ctx.add(Expr::Number(num_value / den_value));
                }
                let reciprocal = ctx.add(Expr::Number(BigRational::one() / den_value));
                let scaled = ctx.add(Expr::Mul(reciprocal, num));
                return fold_numeric_mul_constants_for_hold(ctx, scaled);
            }
            if let Some(num_value) = rational_const_for_hold(ctx, num)
                .filter(|_| matches!(ctx.get(den), Expr::Mul(_, _)))
            {
                let mut denominator_scale = BigRational::one();
                let mut denominator_factors = Vec::new();
                for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        denominator_scale *= value;
                    } else {
                        denominator_factors.push(factor);
                    }
                }
                if !denominator_scale.is_zero()
                    && !denominator_scale.is_one()
                    && !denominator_factors.is_empty()
                {
                    let scaled_num_value = num_value / denominator_scale;
                    if !scaled_num_value.is_integer() {
                        return ctx.add(Expr::Div(num, den));
                    }
                    let num = ctx.add(Expr::Number(scaled_num_value));
                    let den = if denominator_factors.len() == 1 {
                        denominator_factors[0]
                    } else {
                        cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors)
                    };
                    return ctx.add(Expr::Div(num, den));
                }
            }
            if matches!(ctx.get(den), Expr::Mul(_, _)) {
                let mut numerator_scale = BigRational::one();
                let mut numerator_factors = Vec::new();
                let mut raw_numerator_factors = Vec::new();
                mul_leaves_preserve_order_for_calculus_presentation(
                    ctx,
                    num,
                    &mut raw_numerator_factors,
                );
                for factor in raw_numerator_factors {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        numerator_scale *= value;
                    } else {
                        numerator_factors.push(factor);
                    }
                }

                let mut denominator_scale = BigRational::one();
                let mut denominator_factors = Vec::new();
                let mut raw_denominator_factors = Vec::new();
                mul_leaves_preserve_order_for_calculus_presentation(
                    ctx,
                    den,
                    &mut raw_denominator_factors,
                );
                for factor in raw_denominator_factors {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        denominator_scale *= value;
                    } else {
                        denominator_factors.push(factor);
                    }
                }

                if !denominator_scale.is_zero()
                    && !denominator_scale.is_one()
                    && !numerator_factors.is_empty()
                    && !denominator_factors.is_empty()
                {
                    let scaled_numerator = numerator_scale / denominator_scale;
                    if scaled_numerator.is_integer() {
                        let numerator_core = if numerator_factors.len() == 1 {
                            numerator_factors[0]
                        } else {
                            cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors)
                        };
                        let numerator = signed_numerator_for_calculus_presentation(
                            ctx,
                            scaled_numerator,
                            numerator_core,
                        );
                        let denominator = if denominator_factors.len() == 1 {
                            denominator_factors[0]
                        } else {
                            cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors)
                        };
                        return ctx.add(Expr::Div(numerator, denominator));
                    }
                }
            }
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner = fold_numeric_mul_constants_for_hold(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn mul_leaves_preserve_order_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            mul_leaves_preserve_order_for_calculus_presentation(ctx, left, out);
            mul_leaves_preserve_order_for_calculus_presentation(ctx, right, out);
        }
        _ => out.push(expr),
    }
}

fn fold_numeric_mul_constants_for_hold_additive_terms(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = fold_numeric_mul_constants_for_hold_additive_terms(ctx, left);
            let right = fold_numeric_mul_constants_for_hold_additive_terms(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = fold_numeric_mul_constants_for_hold_additive_terms(ctx, left);
            let right = fold_numeric_mul_constants_for_hold_additive_terms(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        _ => fold_numeric_mul_constants_for_hold(ctx, expr),
    }
}

fn sqrt_positive_rational_for_hold(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let value = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            cas_ast::views::as_rational_const(ctx, args[0], 8)?
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => {
            cas_ast::views::as_rational_const(ctx, *base, 8)?
        }
        _ => return None,
    };
    value.is_positive().then_some(value)
}

fn compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num =
                compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(ctx, num, var_name);
            let den = compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, den, var_name)
                .unwrap_or_else(|| {
                    compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                        ctx, den, var_name,
                    )
                });
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, inner, var_name,
            );
            ctx.add(Expr::Neg(inner))
        }
        _ => compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, expr, var_name)
            .unwrap_or(expr),
    }
}

fn has_compactable_sqrt_hyperbolic_reciprocal_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, expr, var_name).is_some() {
        return true;
    }
    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, left, var_name)
                || has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, num, var_name)
                || has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, inner, var_name),
        _ => false,
    }
}

fn compact_sqrt_hyperbolic_call_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh) || args.len() != 1 {
        return None;
    }

    let radicand = calculus_sqrt_like_radicand(ctx, args[0])?;
    if !contains_named_var(ctx, radicand, var_name) {
        return None;
    }
    polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some(ctx.call_builtin(builtin, vec![sqrt_radicand]))
}

fn compact_positive_cosh_log_abs_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(compact) = compact_ln_abs_cosh_sqrt(ctx, expr, var_name) {
        return compact;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, num, var_name);
            let den =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, den, var_name);
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, inner, var_name);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn compact_ln_abs_cosh_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let cosh_expr = match ctx.get(args[0]).clone() {
        Expr::Function(abs_fn, abs_args)
            if ctx.builtin_of(abs_fn) == Some(BuiltinFn::Abs) && abs_args.len() == 1 =>
        {
            abs_args[0]
        }
        Expr::Function(cosh_fn, cosh_args)
            if ctx.builtin_of(cosh_fn) == Some(BuiltinFn::Cosh) && cosh_args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };
    let Expr::Function(cosh_fn, cosh_args) = ctx.get(cosh_expr).clone() else {
        return None;
    };
    if ctx.builtin_of(cosh_fn) != Some(BuiltinFn::Cosh) || cosh_args.len() != 1 {
        return None;
    }
    let radicand = calculus_sqrt_like_radicand(ctx, cosh_args[0])?;
    if !contains_named_var(ctx, radicand, var_name) {
        return None;
    }
    polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let cosh_expr = ctx.call_builtin(BuiltinFn::Cosh, vec![sqrt_radicand]);
    Some(ctx.call_builtin(BuiltinFn::Ln, vec![cosh_expr]))
}

fn has_compactable_ln_abs_cosh_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_ln_abs_cosh_sqrt(ctx, expr, var_name).is_some() {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_ln_abs_cosh_sqrt(ctx, left, var_name)
                || has_compactable_ln_abs_cosh_sqrt(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_ln_abs_cosh_sqrt(ctx, num, var_name)
                || has_compactable_ln_abs_cosh_sqrt(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_ln_abs_cosh_sqrt(ctx, inner, var_name),
        _ => false,
    }
}

fn compact_sqrt_trig_log_abs_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(compact) = compact_ln_abs_trig_sqrt(ctx, expr, var_name) {
        return compact;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, num, var_name);
            let den = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, den, var_name);
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, inner, var_name);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn compact_ln_abs_trig_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let Expr::Function(abs_fn, abs_args) = ctx.get(args[0]).clone() else {
        return None;
    };
    if ctx.builtin_of(abs_fn) != Some(BuiltinFn::Abs) || abs_args.len() != 1 {
        return None;
    }

    let Expr::Function(trig_fn, trig_args) = ctx.get(abs_args[0]).clone() else {
        return None;
    };
    let trig_builtin = ctx.builtin_of(trig_fn)?;
    if !matches!(trig_builtin, BuiltinFn::Sin | BuiltinFn::Cos) || trig_args.len() != 1 {
        return None;
    }

    let radicand = calculus_sqrt_like_radicand(ctx, trig_args[0])?;
    if !contains_named_var(ctx, radicand, var_name) {
        return None;
    }
    polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig_expr = ctx.call_builtin(trig_builtin, vec![sqrt_radicand]);
    let abs_expr = ctx.call_builtin(BuiltinFn::Abs, vec![trig_expr]);
    Some(ctx.call_builtin(BuiltinFn::Ln, vec![abs_expr]))
}

fn has_compactable_ln_abs_trig_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_ln_abs_trig_sqrt(ctx, expr, var_name).is_some() {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_ln_abs_trig_sqrt(ctx, left, var_name)
                || has_compactable_ln_abs_trig_sqrt(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_ln_abs_trig_sqrt(ctx, num, var_name)
                || has_compactable_ln_abs_trig_sqrt(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_ln_abs_trig_sqrt(ctx, inner, var_name),
        _ => false,
    }
}

fn sqrt_trig_log_antiderivative_derivative_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let mut expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let mut outer_sign = BigRational::one();
    while let Expr::Neg(inner) = ctx.get(expr).clone() {
        expr = inner;
        outer_sign = -outer_sign;
    }

    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let Expr::Function(abs_fn, abs_args) = ctx.get(args[0]).clone() else {
        return None;
    };
    if ctx.builtin_of(abs_fn) != Some(BuiltinFn::Abs) || abs_args.len() != 1 {
        return None;
    }

    let Expr::Function(trig_fn, trig_args) = ctx.get(abs_args[0]).clone() else {
        return None;
    };
    if trig_args.len() != 1 {
        return None;
    }
    let (trig_builtin, derivative_builtin, sign) = match ctx.builtin_of(trig_fn)? {
        BuiltinFn::Cos => (BuiltinFn::Cos, BuiltinFn::Tan, -BigRational::one()),
        BuiltinFn::Sin => (BuiltinFn::Sin, BuiltinFn::Cot, BigRational::one()),
        _ => return None,
    };

    let radicand = calculus_sqrt_like_radicand(ctx, trig_args[0])?;
    let chain_coeff =
        outer_sign * sign * sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let compact =
        build_compact_sqrt_trig_log_integrand(ctx, derivative_builtin, radicand, chain_coeff)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig_display = ctx.call_builtin(trig_builtin, vec![sqrt_radicand]);
    let conditions = vec![
        crate::ImplicitCondition::Positive(radicand),
        crate::ImplicitCondition::NonZero(trig_display),
    ];

    Some((compact, conditions))
}

fn ln_reciprocal_trig_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(ln_fn, ln_args) = ctx.get(target).clone() else {
        return None;
    };
    if ctx.builtin_of(ln_fn) != Some(BuiltinFn::Ln) || ln_args.len() != 1 {
        return None;
    }

    let log_arg = ln_args[0];
    let (sqrt_arg, reciprocal_builtin, denominator_builtin) =
        reciprocal_trig_log_sqrt_parts(ctx, log_arg)?;
    let radicand = calculus_sqrt_like_radicand(ctx, sqrt_arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&chain_coeff)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator_trig = ctx.call_builtin(denominator_builtin, vec![sqrt_radicand]);
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let mut denominator_factors = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(sqrt_radicand);
    denominator_factors.push(denominator_trig);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let derivative = ctx.add(Expr::Div(numerator, denominator));

    let reciprocal = ctx.call_builtin(reciprocal_builtin, vec![sqrt_radicand]);
    let companion_builtin = match reciprocal_builtin {
        BuiltinFn::Sec => BuiltinFn::Tan,
        BuiltinFn::Csc => BuiltinFn::Cot,
        _ => return None,
    };
    let companion = ctx.call_builtin(companion_builtin, vec![sqrt_radicand]);
    let compact_log_arg = match reciprocal_builtin {
        BuiltinFn::Sec => ctx.add(Expr::Add(companion, reciprocal)),
        BuiltinFn::Csc => ctx.add(Expr::Sub(reciprocal, companion)),
        _ => return None,
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, derivative),
        vec![
            crate::ImplicitCondition::Positive(radicand),
            crate::ImplicitCondition::NonZero(denominator_trig),
            crate::ImplicitCondition::Positive(compact_log_arg),
        ],
    ))
}

fn reciprocal_trig_log_sqrt_parts(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BuiltinFn, BuiltinFn)> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => unordered_same_arg_unary_pair_for_calculus(
            ctx,
            left,
            BuiltinFn::Sec,
            right,
            BuiltinFn::Tan,
        )
        .map(|arg| (arg, BuiltinFn::Sec, BuiltinFn::Cos)),
        Expr::Sub(left, right) => {
            same_arg_unary_pair_for_calculus(ctx, left, BuiltinFn::Csc, right, BuiltinFn::Cot)
                .map(|arg| (arg, BuiltinFn::Csc, BuiltinFn::Sin))
        }
        _ => None,
    }
}

fn unordered_same_arg_unary_pair_for_calculus(
    ctx: &mut Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    same_arg_unary_pair_for_calculus(ctx, left, left_builtin, right, right_builtin)
        .or_else(|| same_arg_unary_pair_for_calculus(ctx, right, left_builtin, left, right_builtin))
}

fn same_arg_unary_pair_for_calculus(
    ctx: &mut Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left_arg = unary_builtin_arg_for_calculus(ctx, left, left_builtin)?;
    let right_arg = unary_builtin_arg_for_calculus(ctx, right, right_builtin)?;
    same_sqrt_like_argument(ctx, left_arg, right_arg).then_some(left_arg)
}

fn calculus_sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    extract_square_root_base(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => Some(*base),
        _ => None,
    })
}

fn same_sqrt_like_argument(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    if compare_expr(ctx, left, right) == std::cmp::Ordering::Equal {
        return true;
    }
    let Some(left_base) = calculus_sqrt_like_radicand(ctx, left) else {
        return false;
    };
    let Some(right_base) = calculus_sqrt_like_radicand(ctx, right) else {
        return false;
    };
    compare_expr(ctx, left_base, right_base) == std::cmp::Ordering::Equal
}

fn rational_const_for_hold(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Div(num, den) => {
            let num = rational_const_for_hold(ctx, *num)?;
            let den = rational_const_for_hold(ctx, *den)?;
            (!den.is_zero()).then_some(num / den)
        }
        Expr::Neg(inner) => rational_const_for_hold(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn sqrt_reciprocal_trig_antiderivative_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sec | BuiltinFn::Csc)) =>
        {
            let Some(radicand) = extract_square_root_base(ctx, args[0]) else {
                return false;
            };
            polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name).is_some_and(
                |poly| {
                    let derivative = poly.derivative();
                    !derivative.is_zero() && derivative.degree() == 0
                },
            )
        }
        Expr::Neg(inner) => sqrt_reciprocal_trig_antiderivative_result(ctx, inner, var_name),
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
            let mut non_numeric = Vec::new();
            for factor in factors {
                if rational_const_for_hold(ctx, factor).is_none() {
                    non_numeric.push(factor);
                }
            }
            non_numeric.len() == 1
                && sqrt_reciprocal_trig_antiderivative_result(ctx, non_numeric[0], var_name)
        }
        _ => false,
    }
}

fn inverse_sqrt_quotient_arg_result(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arctan | BuiltinFn::Asinh
                    )
                ) =>
        {
            matches!(ctx.get(args[0]), Expr::Div(_, den) if extract_square_root_base(ctx, *den).is_some())
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            inverse_sqrt_quotient_arg_result(ctx, *left)
                || inverse_sqrt_quotient_arg_result(ctx, *right)
        }
        Expr::Neg(inner) => inverse_sqrt_quotient_arg_result(ctx, *inner),
        _ => false,
    }
}

fn has_sqrt_denominator_result(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            extract_square_root_base(ctx, *den).is_some()
                || has_sqrt_denominator_result(ctx, *num)
                || has_sqrt_denominator_result(ctx, *den)
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_sqrt_denominator_result(ctx, *left) || has_sqrt_denominator_result(ctx, *right)
        }
        Expr::Neg(inner) => has_sqrt_denominator_result(ctx, *inner),
        _ => false,
    }
}

fn compact_negative_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new((-1).into(), 2.into()) {
            let one = ctx.num(1);
            let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
            return Some(ctx.add(Expr::Div(one, sqrt)));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_half_power_result_for_integration_presentation(ctx, inner)?;
        if let Expr::Div(num, den) = ctx.get(compact).clone() {
            let numerator = ctx.add(Expr::Neg(num));
            return Some(ctx.add(Expr::Div(numerator, den)));
        }
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut base = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(pow_base, exp) => {
                let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                if exponent != BigRational::new((-1).into(), 2.into()) || base.is_some() {
                    return None;
                }
                base = Some(*pow_base);
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let base = base?;
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let numerator = ctx.add(Expr::Number(coefficient));
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    Some(ctx.add(Expr::Div(numerator, sqrt)))
}

fn compact_negative_three_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    allow_conditional_positive_quadratic: bool,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if matches_supported_negative_odd_half_power(&exponent) {
            if !quadratic_for_fractional_power_calculus_presentation(
                ctx,
                base,
                var_name,
                allow_conditional_positive_quadratic,
            ) {
                return None;
            }
            let one = ctx.num(1);
            let denominator =
                negative_odd_half_power_denominator_for_presentation(ctx, base, -exponent)?;
            return Some(ctx.add(Expr::Div(one, denominator)));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            ctx,
            inner,
            var_name,
            allow_conditional_positive_quadratic,
        )?;
        if let Expr::Div(num, den) = ctx.get(compact).clone() {
            let numerator = ctx.add(Expr::Neg(num));
            return Some(ctx.add(Expr::Div(numerator, den)));
        }
        return Some(ctx.add(Expr::Neg(compact)));
    }

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let numerator_value = cas_ast::views::as_rational_const(ctx, num, 8)?;
        let mut denominator_scale = BigRational::one();
        let mut base = None;
        for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
            if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                denominator_scale *= value;
                continue;
            }
            let Expr::Pow(pow_base, exp) = ctx.get(factor).clone() else {
                return None;
            };
            let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
            if !matches_supported_positive_odd_half_power(&exponent) || base.is_some() {
                return None;
            }
            base = Some((pow_base, exponent));
        }
        if denominator_scale.is_zero() {
            return None;
        }
        let (base, denominator_power) = base?;
        if !quadratic_for_fractional_power_calculus_presentation(
            ctx,
            base,
            var_name,
            allow_conditional_positive_quadratic,
        ) {
            return None;
        }
        let coefficient = numerator_value / denominator_scale;
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        let one = ctx.num(1);
        let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
        let mut denominator_parts = Vec::new();
        if !denominator_coeff.is_one() {
            denominator_parts.push(rational_const_for_calculus_presentation(
                ctx,
                denominator_coeff,
            ));
        }
        denominator_parts.push(negative_odd_half_power_denominator_for_presentation(
            ctx,
            base,
            denominator_power,
        )?);
        let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut base = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(pow_base, exp) => {
                let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                if !matches_supported_negative_odd_half_power(&exponent) || base.is_some() {
                    return None;
                }
                base = Some((*pow_base, -exponent));
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let (base, denominator_power) = base?;
    if !quadratic_for_fractional_power_calculus_presentation(
        ctx,
        base,
        var_name,
        allow_conditional_positive_quadratic,
    ) {
        return None;
    }
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let mut denominator_parts = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_parts.push(negative_odd_half_power_denominator_for_presentation(
        ctx,
        base,
        denominator_power,
    )?);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn matches_supported_negative_odd_half_power(exponent: &BigRational) -> bool {
    exponent.is_negative() && odd_half_denominator_base_power(&(-exponent.clone())).is_some()
}

fn matches_supported_positive_odd_half_power(exponent: &BigRational) -> bool {
    odd_half_denominator_base_power(exponent).is_some()
}

fn quadratic_for_fractional_power_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    allow_conditional_positive_quadratic: bool,
) -> bool {
    if strictly_positive_quadratic_for_calculus_presentation(ctx, expr, var_name) {
        return true;
    }
    if !allow_conditional_positive_quadratic {
        return false;
    }
    let Some(poly) = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name) else {
        return false;
    };
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return false;
    }
    let leading = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    leading.is_positive()
}

fn negative_odd_half_power_denominator_for_presentation(
    ctx: &mut Context,
    base: ExprId,
    denominator_power: BigRational,
) -> Option<ExprId> {
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    let base_power = odd_half_denominator_base_power(&denominator_power)?;
    let base_factor = if base_power == 1 {
        base
    } else {
        let exponent = ctx.num(base_power);
        ctx.add(Expr::Pow(base, exponent))
    };
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[base_factor, sqrt],
    ))
}

fn odd_half_denominator_base_power(denominator_power: &BigRational) -> Option<i64> {
    if denominator_power.denom() != &BigInt::from(2) {
        return None;
    }
    let numerator = denominator_power.numer();
    if !numerator.is_positive() || !numerator.is_odd() {
        return None;
    }
    let numerator = numerator.to_i64()?;
    if numerator < 3 {
        return None;
    }
    Some((numerator - 1) / 2)
}

fn strictly_positive_quadratic_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Some(poly) = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name) else {
        return false;
    };
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !a.is_positive() {
        return false;
    }
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let discriminant = b.clone() * b - four * a * c;
    discriminant.is_negative()
}

fn compact_positive_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new(1.into(), 2.into()) {
            return Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![base]));
        }
        if exponent == BigRational::new(3.into(), 2.into()) {
            return Some(product_with_sqrt_for_positive_three_half_power(ctx, base));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        if expression_contains_positive_three_half_power(ctx, inner) {
            return None;
        }
        let compact = compact_positive_half_power_result_for_integration_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut base = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(pow_base, exp) => {
                let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                if !(exponent == BigRational::new(1.into(), 2.into())
                    || exponent == BigRational::new(3.into(), 2.into()))
                    || base.is_some()
                {
                    return None;
                }
                base = Some((*pow_base, exponent));
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let (base, exponent) = base?;
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let core = if exponent == BigRational::new(1.into(), 2.into()) {
        ctx.call_builtin(BuiltinFn::Sqrt, vec![base])
    } else if exponent == BigRational::new(3.into(), 2.into()) {
        let product = product_with_sqrt_for_positive_three_half_power(ctx, base);
        return Some(scale_three_half_power_product_for_presentation(
            ctx,
            coefficient,
            product,
        ));
    } else {
        return None;
    };
    Some(scale_expr_for_calculus_presentation(ctx, coefficient, core))
}

fn expression_contains_positive_three_half_power(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Pow(_, exp) => cas_ast::views::as_rational_const(ctx, *exp, 8)
            .is_some_and(|exponent| exponent == BigRational::new(3.into(), 2.into())),
        Expr::Mul(_, _) => cas_math::expr_nary::mul_leaves(ctx, expr)
            .iter()
            .any(|factor| expression_contains_positive_three_half_power(ctx, *factor)),
        _ => false,
    }
}

fn product_with_sqrt_for_positive_three_half_power(ctx: &mut Context, base: ExprId) -> ExprId {
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    cas_math::expr_nary::build_balanced_mul(ctx, &[base, sqrt])
}

fn scale_three_half_power_product_for_presentation(
    ctx: &mut Context,
    coefficient: BigRational,
    product: ExprId,
) -> ExprId {
    if coefficient.is_negative() {
        let positive = scale_expr_for_calculus_presentation(ctx, -coefficient, product);
        return ctx.add(Expr::Neg(positive));
    }
    scale_expr_for_calculus_presentation(ctx, coefficient, product)
}

fn supported_integral_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let integrate_call = try_extract_integrate_call(ctx, target)?;
    if integrate_call.var_name != var_name {
        return None;
    }
    if cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    let supported_sqrt_chain_log_target =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        );
    if supported_sqrt_chain_log_target {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if polynomial_times_arctan_affine_integrand_for_diff_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_positive_quadratic_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_fractional_denominator_power_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_derivative_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        if let Some(compact) =
            compact_inverse_sqrt_product_integrand_for_calculus_presentation(
                ctx,
                integrate_call.target,
            )
        {
            return Some(cas_ast::hold::wrap_hold(ctx, compact));
        }
        return Some(cas_ast::hold::wrap_hold(ctx, integrate_call.target));
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_affine_sqrt_product_derivative_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        if let Some(compact) =
            compact_inverse_sqrt_product_integrand_for_calculus_presentation(
                ctx,
                integrate_call.target,
            )
        {
            return Some(cas_ast::hold::wrap_hold(ctx, compact));
        }
        return Some(cas_ast::hold::wrap_hold(ctx, integrate_call.target));
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_polynomial_substitution_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(cas_ast::hold::wrap_hold(ctx, integrate_call.target));
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_square_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(integrate_call.target);
    }

    if let Some(compact) = compact_direct_sqrt_trig_log_derivative_integrand(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        return Some(compact);
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_log_derivative_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        integrate(ctx, integrate_call.target, &integrate_call.var_name)?;
        if let Some(compact) = compact_sqrt_trig_log_derivative_integrand(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
            return Some(compact);
        }
    }

    let supported_compact_target = sqrt_reciprocal_trig_product_integrand_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    );
    if supported_compact_target {
        return Some(integrate_call.target);
    }

    if expr_contains_direct_trig_with_affine_arg(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) && integrate(ctx, integrate_call.target, &integrate_call.var_name).is_some()
    {
        return Some(integrate_call.target);
    }

    None
}

fn compact_inverse_sqrt_product_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
) -> Option<ExprId> {
    let target = cas_ast::hold::strip_all_holds(ctx, target);

    if let Expr::Pow(base, exp) = ctx.get(target).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new((-1).into(), 2.into()) {
            let radicand_factors = cas_math::expr_nary::mul_leaves(ctx, base);
            if radicand_factors.len() < 2 {
                return None;
            }
            let denominator_factors: Vec<_> = radicand_factors
                .into_iter()
                .map(|factor| ctx.call_builtin(BuiltinFn::Sqrt, vec![factor]))
                .collect();
            let one = ctx.num(1);
            let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
            return Some(ctx.add(Expr::Div(one, denominator)));
        }
    }

    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };

    let mut changed = false;
    let mut denominator_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
        let Some(radicand) = extract_square_root_base(ctx, factor) else {
            denominator_factors.push(factor);
            continue;
        };
        let radicand_factors = cas_math::expr_nary::mul_leaves(ctx, radicand);
        if radicand_factors.len() < 2 {
            denominator_factors.push(factor);
            continue;
        }

        changed = true;
        for radicand_factor in radicand_factors {
            denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand_factor]));
        }
    }

    if !changed {
        return None;
    }

    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    Some(ctx.add(Expr::Div(num, denominator)))
}

fn supported_integral_diff_shortcut_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let integrate_call = try_extract_integrate_call(ctx, target)?;
    if integrate_call.var_name != var_name {
        return None;
    }

    let compact = supported_integral_derivative_presentation(ctx, target, var_name)?;
    let required_conditions =
        integrate_required_nonzero_conditions(ctx, integrate_call.target, &integrate_call.var_name)
            .into_iter()
            .map(crate::ImplicitCondition::NonZero)
            .chain(
                integrate_required_positive_conditions(
                    ctx,
                    integrate_call.target,
                    &integrate_call.var_name,
                )
                .into_iter()
                .map(crate::ImplicitCondition::Positive),
            )
            .collect();

    Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions))
}

fn compact_direct_sqrt_trig_log_derivative_integrand(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_factors, denominator_factors) = match ctx.get(target).clone() {
        Expr::Neg(inner) => {
            let compact = compact_direct_sqrt_trig_log_derivative_integrand(ctx, inner, var_name)?;
            return Some(negate_calculus_presentation(ctx, compact));
        }
        Expr::Div(numerator, denominator) => (
            cas_math::expr_nary::mul_leaves(ctx, numerator)
                .into_iter()
                .collect(),
            cas_math::expr_nary::mul_leaves(ctx, denominator)
                .into_iter()
                .collect(),
        ),
        Expr::Mul(_, _) => split_mul_div_factor_parts(ctx, target).unwrap_or_else(|| {
            (
                cas_math::expr_nary::mul_leaves(ctx, target)
                    .into_iter()
                    .collect(),
                Vec::new(),
            )
        }),
        _ => return None,
    };
    let (trig_index, trig_builtin, arg) =
        numerator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| match ctx.get(*factor) {
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && matches!(
                            ctx.builtin_of(*fn_id),
                            Some(BuiltinFn::Tan | BuiltinFn::Cot)
                        ) =>
                {
                    Some((idx, ctx.builtin_of(*fn_id)?, args[0]))
                }
                _ => None,
            })?;
    let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let remaining_numerator: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
        .collect();
    let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
        ctx,
        &remaining_numerator,
        &denominator_factors,
        radicand,
        var_name,
    )?;
    if observed_coeff != chain_coeff && observed_coeff != -chain_coeff {
        return None;
    }

    build_compact_sqrt_trig_log_integrand(ctx, trig_builtin, radicand, observed_coeff)
}

fn compact_direct_sqrt_hyperbolic_log_derivative_integrand(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_factors, denominator_factors) = match ctx.get(target).clone() {
        Expr::Neg(inner) => {
            let compact =
                compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, inner, var_name)?;
            return Some(negate_calculus_presentation(ctx, compact));
        }
        Expr::Div(numerator, denominator) => (
            signed_mul_leaves_for_calculus_presentation(ctx, numerator),
            signed_mul_leaves_for_calculus_presentation(ctx, denominator),
        ),
        Expr::Mul(_, _) => split_mul_div_factor_parts(ctx, target)?,
        _ => return None,
    };

    if let Some((num_index, den_index, tanh_in_numerator, arg)) =
        hyperbolic_sinh_cosh_quotient_parts(ctx, &numerator_factors, &denominator_factors)
    {
        let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
        let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
        let remaining_numerator: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != num_index).then_some(*factor))
            .collect();
        let remaining_denominator: Vec<_> = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != den_index).then_some(*factor))
            .collect();
        let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
            ctx,
            &remaining_numerator,
            &remaining_denominator,
            radicand,
            var_name,
        )?;
        if observed_coeff == chain_coeff || observed_coeff == -chain_coeff {
            return build_compact_sqrt_hyperbolic_log_integrand(
                ctx,
                tanh_in_numerator,
                radicand,
                observed_coeff,
            );
        }
    }

    if let Some((tanh_index, arg)) =
        numerator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| match ctx.get(*factor) {
                Expr::Function(fn_id, args)
                    if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Tanh) =>
                {
                    Some((idx, args[0]))
                }
                _ => None,
            })
    {
        let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
        let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
        let remaining_numerator: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != tanh_index).then_some(*factor))
            .collect();
        let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
            ctx,
            &remaining_numerator,
            &denominator_factors,
            radicand,
            var_name,
        )?;
        if observed_coeff == chain_coeff || observed_coeff == -chain_coeff {
            return build_compact_sqrt_hyperbolic_log_integrand(
                ctx,
                true,
                radicand,
                observed_coeff,
            );
        }
    }

    let (tanh_index, arg) = denominator_factors
        .iter()
        .enumerate()
        .find_map(|(idx, factor)| match ctx.get(*factor) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Tanh) =>
            {
                Some((idx, args[0]))
            }
            _ => None,
        })?;
    let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != tanh_index).then_some(*factor))
        .collect();
    let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
        ctx,
        &numerator_factors,
        &remaining_denominator,
        radicand,
        var_name,
    )?;
    if observed_coeff != chain_coeff && observed_coeff != -chain_coeff {
        return None;
    }

    build_compact_sqrt_hyperbolic_log_integrand(ctx, false, radicand, observed_coeff)
}

fn signed_mul_leaves_for_calculus_presentation(ctx: &mut Context, root: ExprId) -> Vec<ExprId> {
    let mut negative = false;
    let root = match ctx.get(root).clone() {
        Expr::Neg(inner) => {
            negative = !negative;
            inner
        }
        _ => root,
    };
    let mut factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, root) {
        match ctx.get(factor).clone() {
            Expr::Neg(inner) => {
                negative = !negative;
                factors.extend(cas_math::expr_nary::mul_leaves(ctx, inner));
            }
            _ => factors.push(factor),
        }
    }
    if negative {
        factors.insert(0, ctx.num(-1));
    }
    factors
}

fn hyperbolic_sinh_cosh_quotient_parts(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
) -> Option<(usize, usize, bool, ExprId)> {
    for (num_index, numerator_factor) in numerator_factors.iter().enumerate() {
        let Expr::Function(num_fn, num_args) = ctx.get(*numerator_factor).clone() else {
            continue;
        };
        if num_args.len() != 1 {
            continue;
        }
        let Some(num_builtin) = ctx.builtin_of(num_fn) else {
            continue;
        };
        let expected_den_builtin = match num_builtin {
            BuiltinFn::Sinh => BuiltinFn::Cosh,
            BuiltinFn::Cosh => BuiltinFn::Sinh,
            _ => continue,
        };
        for (den_index, denominator_factor) in denominator_factors.iter().enumerate() {
            let Expr::Function(den_fn, den_args) = ctx.get(*denominator_factor).clone() else {
                continue;
            };
            if den_args.len() != 1 || ctx.builtin_of(den_fn) != Some(expected_den_builtin) {
                continue;
            }
            if !same_sqrt_like_argument(ctx, num_args[0], den_args[0]) {
                continue;
            }
            return Some((
                num_index,
                den_index,
                num_builtin == BuiltinFn::Sinh,
                num_args[0],
            ));
        }
    }
    None
}

fn sqrt_cosh_log_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(ln_fn, ln_args) = ctx.get(target).clone() else {
        return None;
    };
    if ctx.builtin_of(ln_fn) != Some(BuiltinFn::Ln) || ln_args.len() != 1 {
        return None;
    }
    let Expr::Function(cosh_fn, cosh_args) = ctx.get(ln_args[0]).clone() else {
        return None;
    };
    if ctx.builtin_of(cosh_fn) != Some(BuiltinFn::Cosh) || cosh_args.len() != 1 {
        return None;
    }

    let radicand = calculus_sqrt_like_radicand(ctx, cosh_args[0])?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    build_compact_sqrt_hyperbolic_log_integrand(ctx, true, radicand, chain_coeff)
}

fn unary_builtin_arg_for_calculus(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin)).then_some(args[0])
}

fn signed_unary_builtin_arg_for_calculus(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, BigRational)> {
    if let Some(arg) = unary_builtin_arg_for_calculus(ctx, expr, builtin) {
        return Some((arg, BigRational::one()));
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        return unary_builtin_arg_for_calculus(ctx, *inner, builtin)
            .map(|arg| (arg, -BigRational::one()));
    }

    if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
        return None;
    }

    let mut scale = BigRational::one();
    let mut tan_arg = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let arg = unary_builtin_arg_for_calculus(ctx, factor, builtin)?;
        if tan_arg.replace(arg).is_some() {
            return None;
        }
    }

    (scale == -BigRational::one()).then_some((tan_arg?, -BigRational::one()))
}

struct ShiftedTangentLogArg {
    tan_arg: ExprId,
    tan_sign: BigRational,
    shift: BigRational,
}

fn shifted_tangent_log_arg(ctx: &Context, expr: ExprId) -> Option<ShiftedTangentLogArg> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *left, 8).filter(|value| !value.is_zero())
            {
                let (tan_arg, tan_sign) =
                    signed_unary_builtin_arg_for_calculus(ctx, *right, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign,
                    shift,
                })
            } else if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *right, 8).filter(|value| !value.is_zero())
            {
                let (tan_arg, tan_sign) =
                    signed_unary_builtin_arg_for_calculus(ctx, *left, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign,
                    shift,
                })
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *left, 8).filter(|value| !value.is_zero())
            {
                let tan_arg = unary_builtin_arg_for_calculus(ctx, *right, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign: -BigRational::one(),
                    shift,
                })
            } else if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *right, 8).filter(|value| !value.is_zero())
            {
                let tan_arg = unary_builtin_arg_for_calculus(ctx, *left, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign: BigRational::one(),
                    shift: -shift,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn build_compact_shifted_tangent_log_arg(
    ctx: &mut Context,
    tan: ExprId,
    shift: BigRational,
    tan_sign: &BigRational,
) -> Option<ExprId> {
    if shift.is_zero() {
        return None;
    }

    let shift_expr = rational_const_for_calculus_presentation(ctx, shift.abs());
    if tan_sign == &BigRational::one() {
        if shift.is_positive() {
            Some(ctx.add(Expr::Add(tan, shift_expr)))
        } else {
            Some(ctx.add(Expr::Sub(tan, shift_expr)))
        }
    } else if tan_sign == &-BigRational::one() {
        if shift.is_positive() {
            Some(ctx.add(Expr::Sub(shift_expr, tan)))
        } else {
            let neg_tan = ctx.add(Expr::Neg(tan));
            Some(ctx.add(Expr::Sub(neg_tan, shift_expr)))
        }
    } else {
        None
    }
}

fn ln_constant_shifted_tan_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(ln_fn, ln_args) = ctx.get(target).clone() else {
        return None;
    };
    if ctx.builtin_of(ln_fn) != Some(BuiltinFn::Ln) || ln_args.len() != 1 {
        return None;
    }

    let log_arg = ln_args[0];
    let shifted_tan = shifted_tangent_log_arg(ctx, log_arg)?;
    let radicand = calculus_sqrt_like_radicand(ctx, shifted_tan.tan_arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let signed_chain_coeff = shifted_tan.tan_sign.clone() * chain_coeff;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&signed_chain_coeff)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let tan = ctx.call_builtin(BuiltinFn::Tan, vec![sqrt_radicand]);
    let compact_log_arg =
        build_compact_shifted_tangent_log_arg(ctx, tan, shifted_tan.shift, &shifted_tan.tan_sign)?;
    let cos = ctx.call_builtin(BuiltinFn::Cos, vec![sqrt_radicand]);
    let two = ctx.num(2);
    let cos_sq = ctx.add(Expr::Pow(cos, two));

    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let mut denominator_factors = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(sqrt_radicand);
    denominator_factors.push(cos_sq);
    denominator_factors.push(compact_log_arg);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let derivative = ctx.add(Expr::Div(numerator, denominator));

    let cos_guard = ctx.call_builtin(BuiltinFn::Cos, vec![sqrt_radicand]);
    Some((
        cas_ast::hold::wrap_hold(ctx, derivative),
        vec![
            crate::ImplicitCondition::Positive(radicand),
            crate::ImplicitCondition::Positive(compact_log_arg),
            crate::ImplicitCondition::NonZero(cos_guard),
        ],
    ))
}

fn split_mul_div_factor_parts(ctx: &Context, expr: ExprId) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    for factor in factors {
        match ctx.get(factor) {
            Expr::Div(num, den) => {
                numerator_factors.extend(cas_math::expr_nary::mul_leaves(ctx, *num));
                denominator_factors.extend(cas_math::expr_nary::mul_leaves(ctx, *den));
            }
            _ => numerator_factors.push(factor),
        }
    }

    (!denominator_factors.is_empty()).then_some((numerator_factors, denominator_factors))
}

fn compact_sqrt_trig_log_derivative_integrand(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_factors, denominator_factors) = match ctx.get(target).clone() {
        Expr::Neg(inner) => {
            let compact = compact_sqrt_trig_log_derivative_integrand(ctx, inner, var_name)?;
            return Some(negate_calculus_presentation(ctx, compact));
        }
        Expr::Div(numerator, denominator) => (
            cas_math::expr_nary::mul_leaves(ctx, numerator)
                .into_iter()
                .collect(),
            cas_math::expr_nary::mul_leaves(ctx, denominator)
                .into_iter()
                .collect(),
        ),
        Expr::Mul(_, _) => split_mul_div_factor_parts(ctx, target)?,
        _ => return None,
    };
    let (den_index, den_builtin, arg) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| match ctx.get(*factor) {
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && matches!(
                            ctx.builtin_of(*fn_id),
                            Some(BuiltinFn::Cos | BuiltinFn::Sin)
                        ) =>
                {
                    Some((idx, ctx.builtin_of(*fn_id)?, args[0]))
                }
                _ => None,
            })?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    let trig_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Tan,
        BuiltinFn::Sin => BuiltinFn::Cot,
        _ => return None,
    };
    let (num_index, _) = numerator_factors.iter().enumerate().find(|(_, factor)| {
        let Expr::Function(fn_id, args) = ctx.get(**factor).clone() else {
            return false;
        };
        args.len() == 1
            && ctx.builtin_of(fn_id) == Some(numerator_builtin)
            && same_sqrt_like_argument(ctx, args[0], arg)
    })?;
    let radicand = extract_square_root_base(ctx, arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;

    let remaining_numerator: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != num_index).then_some(*factor))
        .collect();
    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != den_index).then_some(*factor))
        .collect();
    let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
        ctx,
        &remaining_numerator,
        &remaining_denominator,
        radicand,
        var_name,
    )?;
    if observed_coeff != chain_coeff && observed_coeff != -chain_coeff {
        return None;
    }

    build_compact_sqrt_trig_log_integrand(ctx, trig_builtin, radicand, observed_coeff)
}

fn sqrt_chain_linear_derivative_coeff(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative = radicand_poly.derivative();
    if derivative.is_zero() || derivative.degree() != 0 {
        return None;
    }
    let derivative_expr = derivative.to_expr(ctx);
    let derivative_coeff = cas_ast::views::as_rational_const(ctx, derivative_expr, 8)?;
    Some(derivative_coeff / BigRational::from_integer(2.into()))
}

fn build_compact_sqrt_trig_log_integrand(
    ctx: &mut Context,
    trig_builtin: BuiltinFn,
    radicand: ExprId,
    chain_coeff: BigRational,
) -> Option<ExprId> {
    if !matches!(trig_builtin, BuiltinFn::Tan | BuiltinFn::Cot) {
        return None;
    }
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig = ctx.call_builtin(trig_builtin, vec![sqrt_radicand]);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&chain_coeff)?;
    let compact_numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, trig);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
    };
    Some(ctx.add(Expr::Div(compact_numerator, denominator)))
}

fn build_compact_sqrt_hyperbolic_log_integrand(
    ctx: &mut Context,
    tanh_in_numerator: bool,
    radicand: ExprId,
    chain_coeff: BigRational,
) -> Option<ExprId> {
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![sqrt_radicand]);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&chain_coeff)?;

    if tanh_in_numerator {
        let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, tanh);
        let denominator = if denominator_coeff == BigRational::one() {
            sqrt_radicand
        } else {
            let denominator_coeff =
                rational_const_for_calculus_presentation(ctx, denominator_coeff);
            cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
        };
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let denominator_core = cas_math::expr_nary::build_balanced_mul(ctx, &[tanh, sqrt_radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        denominator_core
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, denominator_core])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn sqrt_chain_factor_coeff_over_sqrt(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    radicand: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if let Some((idx, _)) = numerator_factors
        .iter()
        .enumerate()
        .find(|(_, factor)| reciprocal_sqrt_factor_matches(ctx, **factor, radicand))
    {
        let numerator_rest: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        return rational_factor_quotient(ctx, &numerator_rest, denominator_factors);
    }

    if let Some((idx, _)) = denominator_factors
        .iter()
        .enumerate()
        .find(|(_, factor)| sqrt_factor_matches(ctx, **factor, radicand))
    {
        let denominator_rest: Vec<_> = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        return rational_factor_quotient(ctx, numerator_factors, &denominator_rest);
    }

    let (sqrt_idx, _) = numerator_factors
        .iter()
        .enumerate()
        .find(|(_, factor)| sqrt_factor_matches(ctx, **factor, radicand))?;
    let (radicand_idx, denominator_radicand_scale) = denominator_factors
        .iter()
        .enumerate()
        .find_map(|(idx, factor)| {
            denominator_factor_radicand_scale(ctx, *factor, radicand, var_name)
                .map(|scale| (idx, scale))
        })?;
    let numerator_rest: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(factor_idx, factor)| (factor_idx != sqrt_idx).then_some(*factor))
        .collect();
    let denominator_rest: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(factor_idx, factor)| (factor_idx != radicand_idx).then_some(*factor))
        .collect();
    rational_factor_quotient(ctx, &numerator_rest, &denominator_rest)
        .map(|coeff| coeff / denominator_radicand_scale)
}

fn sqrt_factor_matches(ctx: &mut Context, factor: ExprId, radicand: ExprId) -> bool {
    calculus_sqrt_like_radicand(ctx, factor)
        .is_some_and(|base| compare_expr(ctx, base, radicand) == std::cmp::Ordering::Equal)
}

fn denominator_factor_radicand_scale(
    ctx: &Context,
    factor: ExprId,
    radicand: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if compare_expr(ctx, factor, radicand) == std::cmp::Ordering::Equal {
        return Some(BigRational::one());
    }

    let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    polynomial_scale(&factor_poly, &radicand_poly)
}

fn polynomial_scale(factor: &Polynomial, base: &Polynomial) -> Option<BigRational> {
    if base.is_zero() {
        return None;
    }

    let len = factor.coeffs.len().max(base.coeffs.len());
    let mut scale = None;
    for idx in 0..len {
        let factor_coeff = factor
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let base_coeff = base
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if base_coeff.is_zero() {
            if !factor_coeff.is_zero() {
                return None;
            }
            continue;
        }

        let candidate = factor_coeff / base_coeff;
        if candidate.is_zero() {
            return None;
        }
        match &scale {
            Some(existing) if *existing != candidate => return None,
            Some(_) => {}
            None => scale = Some(candidate),
        }
    }

    scale
}

fn reciprocal_sqrt_factor_matches(ctx: &mut Context, factor: ExprId, radicand: ExprId) -> bool {
    matches!(
        ctx.get(factor),
        Expr::Pow(base, exp)
            if compare_expr(ctx, *base, radicand) == std::cmp::Ordering::Equal
                && cas_ast::views::as_rational_const(ctx, *exp, 8)
                    == Some(BigRational::new((-1).into(), 2.into()))
    )
}

fn rational_factor_quotient(
    ctx: &Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
) -> Option<BigRational> {
    let numerator = rational_factor_product(ctx, numerator_factors)?;
    let denominator = rational_factor_product(ctx, denominator_factors)?;
    (!denominator.is_zero()).then_some(numerator / denominator)
}

fn rational_factor_product(ctx: &Context, factors: &[ExprId]) -> Option<BigRational> {
    factors.iter().try_fold(BigRational::one(), |acc, factor| {
        cas_ast::views::as_rational_const(ctx, *factor, 8).map(|value| acc * value)
    })
}

fn sqrt_reciprocal_trig_product_integrand_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let mut primary_radicands = Vec::new();
    let mut companion_radicands = Vec::new();
    let mut denominator_radicands = Vec::new();
    collect_sqrt_reciprocal_trig_product_signal(
        ctx,
        target,
        false,
        &mut primary_radicands,
        &mut companion_radicands,
        &mut denominator_radicands,
    );

    primary_radicands.iter().any(|primary| {
        let same_companion = companion_radicands
            .iter()
            .any(|companion| compare_expr(ctx, *primary, *companion) == std::cmp::Ordering::Equal);
        let same_denominator = denominator_radicands.iter().any(|denominator| {
            compare_expr(ctx, *primary, *denominator) == std::cmp::Ordering::Equal
        });
        same_companion
            && same_denominator
            && polynomial_radicand_for_calculus_presentation(ctx, *primary, var_name).is_some_and(
                |poly| {
                    let derivative = poly.derivative();
                    !derivative.is_zero() && derivative.degree() == 0
                },
            )
    })
}

fn collect_sqrt_reciprocal_trig_product_signal(
    ctx: &mut Context,
    root: ExprId,
    in_denominator: bool,
    primary_radicands: &mut Vec<ExprId>,
    companion_radicands: &mut Vec<ExprId>,
    denominator_radicands: &mut Vec<ExprId>,
) {
    if in_denominator {
        if let Some(radicand) = extract_square_root_base(ctx, root) {
            denominator_radicands.push(radicand);
        }
    }

    match ctx.get(root).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() == 1 {
                if let Some(radicand) = extract_square_root_base(ctx, args[0]) {
                    match ctx.builtin_of(fn_id) {
                        Some(BuiltinFn::Sec | BuiltinFn::Csc) => {
                            primary_radicands.push(radicand);
                        }
                        Some(BuiltinFn::Tan | BuiltinFn::Cot) => {
                            companion_radicands.push(radicand);
                        }
                        _ => {}
                    }
                }
            }
            for arg in args {
                collect_sqrt_reciprocal_trig_product_signal(
                    ctx,
                    arg,
                    in_denominator,
                    primary_radicands,
                    companion_radicands,
                    denominator_radicands,
                );
            }
        }
        Expr::Div(numerator, denominator) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                numerator,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                denominator,
                true,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                left,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                right,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Pow(base, exp) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                base,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                exp,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                inner,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Matrix { data, .. } => {
            for item in data {
                collect_sqrt_reciprocal_trig_product_signal(
                    ctx,
                    item,
                    in_denominator,
                    primary_radicands,
                    companion_radicands,
                    denominator_radicands,
                );
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    let call = try_extract_integrate_call(ctx, expr)?;
    if let Some((mut result, required_nonzero)) =
        cas_math::symbolic_integration_support::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        if polynomial_times_arctan_affine_integrand_for_diff_shortcut(
            ctx,
            call.target,
            &call.var_name,
        ) {
            if let Some(compact) =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name)
            {
                result = compact;
            }
        }
        let desc = render_integrate_desc_with(&call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
        });
        return Some(
            Rewrite::new(result)
                .desc(desc)
                .requires(crate::ImplicitCondition::NonZero(required_nonzero)),
        );
    }

    let required_nonzero = integrate_required_nonzero_conditions(ctx, call.target, &call.var_name);
    let mut required_positive =
        integrate_required_positive_conditions(ctx, call.target, &call.var_name);
    let preserve_compact_reciprocal = cas_math::symbolic_integration_support::integrate_symbolic_is_reciprocal_negative_power_denominator_quotient_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_fractional_denominator_power =
        cas_math::symbolic_integration_support::integrate_symbolic_is_fractional_denominator_power_substitution_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_arctan_reciprocal_affine = cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_reciprocal_affine_variable_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_arctan_sqrt_reciprocal = cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_arctan_sqrt_unit_shift_square = cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_arctan_scaled_variable =
        cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_scaled_variable_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_atanh_polynomial = cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_polynomial_substitution_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_asinh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_affine_variable_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_atanh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_affine_variable_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_acosh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_affine_variable_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_bounded_inverse_trig = cas_math::symbolic_integration_support::integrate_symbolic_is_bounded_inverse_trig_variable_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_trig_polynomial = cas_math::symbolic_integration_support::integrate_symbolic_is_trig_polynomial_substitution_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_sqrt_reciprocal_trig_product =
        sqrt_reciprocal_trig_product_integrand_target(ctx, call.target, &call.var_name);
    let preserve_compact_sqrt_trig_reciprocal = cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_reciprocal_derivative_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_sqrt_trig_log = cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_log_derivative_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_sqrt_hyperbolic_log = cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_sqrt_hyperbolic_reciprocal_square =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_sqrt_hyperbolic_reciprocal_derivative =
        cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_inverse_hyperbolic_sqrt_reciprocal =
        cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_affine_sqrt_product_derivative =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_sqrt_product_derivative_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_arcsin_inverse_sqrt_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_log_cube_product = cas_math::symbolic_integration_support::integrate_symbolic_is_log_cube_product_substitution_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_log_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_log_product_substitution_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_rational_linear_partial_fraction =
        cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_affine_hyperbolic_square =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_square_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_hyperbolic_square_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_hyperbolic_square_product_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_linear_exp_by_parts = crate::rule::steps_enabled() && cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_exp_linear_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_linear_trig_by_parts = (crate::rule::steps_enabled()
        || target_has_top_level_negative_orientation(ctx, call.target))
        &&
        cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_trig_linear_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_linear_hyperbolic_by_parts = crate::rule::steps_enabled()
        && cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_hyperbolic_linear_target(
            ctx,
            call.target,
            &call.var_name,
        )
        && linear_hyperbolic_integer_slope_for_calculus_presentation(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_repeated_trig_by_parts =
        cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_trig_linear_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_repeated_hyperbolic_by_parts =
        cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_log_by_parts = crate::rule::steps_enabled()
        && (cas_math::symbolic_integration_support::integrate_symbolic_is_monomial_times_ln_var_by_parts_target(
            ctx,
            call.target,
            &call.var_name,
        ) || (!matches!(ctx.get(call.target), Expr::Add(_, _) | Expr::Sub(_, _))
            && cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_affine_ln_by_parts_target(
            ctx,
            call.target,
            &call.var_name,
        )) || (!matches!(ctx.get(call.target), Expr::Add(_, _) | Expr::Sub(_, _))
            && cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
            ctx,
            call.target,
            &call.var_name,
        )) || (!matches!(ctx.get(call.target), Expr::Add(_, _) | Expr::Sub(_, _))
            && cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
            ctx,
            call.target,
            &call.var_name,
        )));
    let mut result = integrate(ctx, call.target, &call.var_name)?;
    let compact_polynomial_arctan_by_parts_result =
        compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name);
    let preserve_compact_polynomial_arctan_by_parts_result =
        compact_polynomial_arctan_by_parts_result.is_some();
    if required_positive.is_empty() {
        required_positive.extend(collect_atanh_open_interval_conditions(ctx, result));
    }
    let preserve_compact_inverse_sqrt_arg = inverse_sqrt_quotient_arg_result(ctx, result);
    let preserve_compact_sqrt_denominator_result = has_sqrt_denominator_result(ctx, result);
    let compact_negative_half_power_result =
        compact_negative_half_power_result_for_integration_presentation(ctx, result);
    let preserve_compact_negative_half_power_result = compact_negative_half_power_result.is_some();
    let compact_negative_three_half_power_result =
        compact_negative_three_half_power_result_for_integration_presentation(
            ctx,
            result,
            &call.var_name,
            !required_positive.is_empty(),
        );
    let preserve_compact_negative_three_half_power_result =
        compact_negative_three_half_power_result.is_some();
    let compact_positive_half_power_result =
        compact_positive_half_power_result_for_integration_presentation(ctx, result);
    let preserve_compact_positive_half_power_result = compact_positive_half_power_result.is_some();
    let preserve_compact_sqrt_reciprocal_trig_result =
        sqrt_reciprocal_trig_antiderivative_result(ctx, result, &call.var_name);
    let preserve_compact_sqrt_trig_log_result =
        has_compactable_ln_abs_trig_sqrt(ctx, result, &call.var_name);
    let preserve_compact_sqrt_hyperbolic_reciprocal_result =
        has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, result, &call.var_name);
    let compact_half_power_sum_root_product_result =
        compact_half_power_sum_root_product_for_integration_presentation(
            ctx,
            result,
            &call.var_name,
        );
    let preserve_compact_half_power_sum_root_product_result =
        compact_half_power_sum_root_product_result.is_some();
    if preserve_compact_reciprocal
        || preserve_compact_fractional_denominator_power
        || preserve_compact_arctan_reciprocal_affine
        || preserve_compact_arctan_sqrt_reciprocal
        || preserve_compact_arctan_sqrt_unit_shift_square
        || preserve_compact_arctan_scaled_variable
        || preserve_compact_atanh_polynomial
        || preserve_compact_asinh_affine
        || preserve_compact_atanh_affine
        || preserve_compact_acosh_affine
        || preserve_compact_bounded_inverse_trig
        || preserve_compact_trig_polynomial
        || preserve_compact_sqrt_reciprocal_trig_product
        || preserve_compact_sqrt_trig_reciprocal
        || preserve_compact_sqrt_trig_log
        || preserve_compact_sqrt_hyperbolic_log
        || preserve_compact_sqrt_hyperbolic_reciprocal_square
        || preserve_compact_sqrt_hyperbolic_reciprocal_derivative
        || preserve_compact_inverse_hyperbolic_sqrt_reciprocal
        || preserve_compact_affine_sqrt_product_derivative
        || preserve_compact_arcsin_inverse_sqrt_product
        || preserve_compact_log_cube_product
        || preserve_compact_log_product
        || preserve_compact_rational_linear_partial_fraction
        || preserve_compact_affine_hyperbolic_square
        || preserve_compact_hyperbolic_square_product
        || preserve_compact_linear_exp_by_parts
        || preserve_compact_linear_trig_by_parts
        || preserve_compact_linear_hyperbolic_by_parts
        || preserve_compact_repeated_trig_by_parts
        || preserve_compact_repeated_hyperbolic_by_parts
        || preserve_compact_polynomial_arctan_by_parts_result
        || preserve_compact_log_by_parts
        || preserve_compact_inverse_sqrt_arg
        || preserve_compact_sqrt_denominator_result
        || preserve_compact_negative_half_power_result
        || preserve_compact_negative_three_half_power_result
        || preserve_compact_positive_half_power_result
        || preserve_compact_sqrt_reciprocal_trig_result
        || preserve_compact_sqrt_trig_log_result
        || preserve_compact_sqrt_hyperbolic_reciprocal_result
        || preserve_compact_half_power_sum_root_product_result
    {
        if let Some(compact) = compact_half_power_sum_root_product_result {
            result = compact;
        }
        if let Some(compact) = compact_negative_half_power_result {
            result = compact;
        }
        if let Some(compact) = compact_negative_three_half_power_result {
            result = compact;
        }
        if let Some(compact) = compact_positive_half_power_result {
            result = compact;
        }
        if preserve_compact_sqrt_hyperbolic_log {
            result = compact_positive_cosh_log_abs_for_integration_presentation(
                ctx,
                result,
                &call.var_name,
            );
        }
        if preserve_compact_sqrt_trig_log || preserve_compact_sqrt_trig_log_result {
            result =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, result, &call.var_name);
        }
        if preserve_compact_sqrt_hyperbolic_reciprocal_derivative
            || preserve_compact_sqrt_hyperbolic_reciprocal_result
        {
            result = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx,
                result,
                &call.var_name,
            );
        }
        if let Some(compact) = compact_polynomial_arctan_by_parts_result {
            result = compact;
        }
        result = if preserve_compact_rational_linear_partial_fraction {
            fold_numeric_mul_constants_for_hold_additive_terms(ctx, result)
        } else {
            fold_numeric_mul_constants_for_hold(ctx, result)
        };
        if preserve_compact_polynomial_arctan_by_parts_result {
            if let Some(compact) =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name)
            {
                result = compact;
            }
        }
        if preserve_compact_log_by_parts {
            if let Some(compact) = flatten_subtracting_additive_group_for_calculus_presentation(
                ctx,
                result,
                &call.var_name,
            ) {
                result = compact;
            }
        }
        result = if preserve_compact_fractional_denominator_power {
            ctx.add(Expr::Hold(result))
        } else {
            cas_ast::hold::wrap_hold(ctx, result)
        };
    }
    if let Some(compact) = compact_acosh_surd_width_arg_for_integration_presentation(ctx, result) {
        result = compact;
    }
    result = compact_integer_affine_inverse_args_for_integration_presentation(
        ctx,
        result,
        &call.var_name,
    );
    let desc = render_integrate_desc_with(&call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    let required_conditions = required_nonzero
        .into_iter()
        .map(crate::ImplicitCondition::NonZero)
        .chain(
            required_positive
                .into_iter()
                .map(crate::ImplicitCondition::Positive),
        );
    Some(
        Rewrite::new(result)
            .desc(desc)
            .requires_all(required_conditions),
    )
});

fn target_has_top_level_negative_orientation(ctx: &Context, target: ExprId) -> bool {
    match ctx.get(target) {
        Expr::Neg(_) => true,
        Expr::Mul(left, right) => {
            matches!(ctx.get(*left), Expr::Neg(_)) || matches!(ctx.get(*right), Expr::Neg(_))
        }
        _ => false,
    }
}

fn expr_contains_direct_trig_with_affine_arg(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, current)) {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(*fn_id),
                        Some(BuiltinFn::Sin | BuiltinFn::Cos)
                    )
                    && Polynomial::from_expr(ctx, args[0], var_name)
                        .is_ok_and(|poly| poly.degree() == 1) =>
            {
                return true;
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Number(_)
            | Expr::Constant(_)
            | Expr::Variable(_)
            | Expr::SessionRef(_)
            | Expr::Matrix { .. } => {}
        }
    }
    false
}

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    let call = try_extract_diff_call(ctx, expr)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if let Some(open_interval_gap) = atanh_sqrt_known_empty_open_interval_gap(ctx, target) {
        crate::register_blocked_hint(crate::BlockedHint {
            key: crate::AssumptionKey::positive_key(ctx, open_interval_gap),
            expr_id: open_interval_gap,
            rule: "Symbolic Differentiation".to_string(),
            suggestion: "real domain is empty; no real derivative is exposed",
        });
        return None;
    }
    if let Some((open_interval_gap, should_emit_hint)) =
        inverse_reciprocal_trig_bounded_trig_empty_open_interval_gap(ctx, target, &call.var_name)
    {
        if should_emit_hint {
            crate::register_blocked_hint(crate::BlockedHint {
                key: crate::AssumptionKey::positive_key(ctx, open_interval_gap),
                expr_id: open_interval_gap,
                rule: "Symbolic Differentiation".to_string(),
                suggestion: "real domain is empty; no real derivative is exposed",
            });
        }
        return None;
    }
    let mut shortcut_required_conditions = Vec::new();
    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        shortcut_required_conditions.push(crate::ImplicitCondition::Positive(required_positive));
        shortcut_required_conditions.extend(required_conditions);
        let required_conditions = shortcut_required_conditions.into_iter();
        let desc = render_diff_desc_with(&call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
        });
        return Some(
            Rewrite::new(result)
                .desc(desc)
                .requires_all(required_conditions),
        );
    }
    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, &call.var_name)
    {
        shortcut_required_conditions.push(crate::ImplicitCondition::Positive(required_positive));
        shortcut_required_conditions.extend(required_conditions);
        let required_conditions =
            reciprocal_trig_diff_required_conditions(ctx, target, &call.var_name)
                .into_iter()
                .chain(
                    log_reciprocal_abs_or_sqrt_negative_even_power_diff_required_conditions(
                        ctx, target,
                    ),
                )
                .chain(shortcut_required_conditions);
        let desc = render_diff_desc_with(&call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
        });
        return Some(
            Rewrite::new(result)
                .desc(desc)
                .requires_all(required_conditions),
        );
    }
    if let Some((result, required_positive, required_conditions)) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, &call.var_name)
    {
        shortcut_required_conditions.push(crate::ImplicitCondition::Positive(required_positive));
        shortcut_required_conditions.extend(required_conditions);
        let required_conditions = shortcut_required_conditions.into_iter();
        let desc = render_diff_desc_with(&call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
        });
        return Some(
            Rewrite::new(result)
                .desc(desc)
                .requires_all(required_conditions),
        );
    }
    let mut result = ln_reciprocal_trig_sqrt_derivative_presentation(ctx, target, &call.var_name)
        .map(|(result, required_conditions)| {
            shortcut_required_conditions.extend(required_conditions);
            result
        })
        .or_else(|| {
            ln_constant_shifted_tan_sqrt_derivative_presentation(ctx, target, &call.var_name).map(
                |(result, required_conditions)| {
                    shortcut_required_conditions.extend(required_conditions);
                    result
                },
            )
        })
        .or_else(|| {
            supported_integral_diff_shortcut_presentation(ctx, target, &call.var_name).map(
                |(result, required_conditions)| {
                    shortcut_required_conditions.extend(required_conditions);
                    result
                },
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_sqrt_variable_over_positive_affine_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| reciprocal_positive_shifted_sqrt_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_conditions) =
                reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive) =
                sqrt_over_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_conditions) =
                sqrt_of_polynomial_quotient_derivative_presentation_with_domain(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            constant_scaled_arctan_surd_quotient_scaled_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_acosh_affine_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation(ctx, target, &call.var_name)
                .map(|compact| cas_ast::hold::wrap_hold(ctx, compact))
        })
        .or_else(|| {
            bounded_inverse_trig_self_normalized_projection_derivative_presentation(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_condition) =
                asinh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| {
            let (result, required_condition) =
                scaled_asinh_sqrt_constant_over_polynomial_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| arctan_surd_quotient_scaled_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| arctan_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            arctan_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| {
            atanh_self_normalized_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_condition) =
                arctan_self_normalized_surd_reciprocal_compact_derivative(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| arctan_affine_by_parts_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| atanh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| {
            polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_condition) =
                atanh_sqrt_constant_over_polynomial_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.push(required_condition);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                negative_arccot_sqrt_polynomial_derivative_shortcut(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            sqrt_bounded_trig_positive_shift_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_positive, required_conditions) =
                sqrt_small_additive_elementary_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions
                .push(crate::ImplicitCondition::Positive(required_positive));
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let result =
                sqrt_elementary_function_derivative_presentation(ctx, target, &call.var_name)?;
            if let Some(radicand) = extract_square_root_base(ctx, target) {
                shortcut_required_conditions.push(crate::ImplicitCondition::Positive(radicand));
            }
            Some(result)
        })
        .or_else(|| {
            scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            arctan_sqrt_constant_over_polynomial_presentation(
                ctx,
                target,
                &call.var_name,
                BigRational::one(),
            )
        })
        .or_else(|| {
            arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(
                ctx,
                target,
                &call.var_name,
            )
        })
        .or_else(|| {
            let result =
                acosh_sqrt_shifted_quadratic_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(acosh_sqrt_diff_required_conditions(ctx, target));
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_affine_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_strictly_positive_polynomial_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            let (result, required_conditions) =
                acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, &call.var_name)?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            positive_quadratic_square_derivative_result_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            positive_quadratic_quotient_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| {
            let (result, required_conditions) =
                ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, &call.var_name))
        .or_else(|| {
            let (result, required_conditions) =
                ln_sqrt_plus_polynomial_direct_derivative_presentation(
                    ctx,
                    target,
                    &call.var_name,
                )?;
            shortcut_required_conditions.extend(required_conditions);
            Some(result)
        })
        .or_else(|| {
            exp_trig_by_parts_primitive_derivative_presentation(ctx, target, &call.var_name)
        })
        .or_else(|| differentiate(ctx, target, &call.var_name))?;
    if let Some(compact) =
        compact_positive_quadratic_square_derivative_result(ctx, result, &call.var_name)
    {
        result = compact;
    }
    if let Some(compact) = compact_sqrt_var_over_var_times_positive_shift_square_diff_result(
        ctx,
        result,
        &call.var_name,
    ) {
        result = compact;
    }
    let required_conditions = atanh_diff_required_conditions(ctx, target, &call.var_name)
        .into_iter()
        .chain(reciprocal_trig_diff_required_conditions(
            ctx,
            target,
            &call.var_name,
        ))
        .chain(log_reciprocal_abs_or_sqrt_negative_even_power_diff_required_conditions(ctx, target))
        .chain(shortcut_required_conditions);
    let desc = render_diff_desc_with(&call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Some(
        Rewrite::new(result)
            .desc(desc)
            .requires_all(required_conditions),
    )
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(summation::SumRule));
    simplifier.add_rule(Box::new(summation::ProductRule));
}

#[cfg(test)]
mod compact_hold_tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_collapses_rational_noise() {
        let mut ctx = Context::new();
        let expr = parse("(atanh(x^2/sqrt(3)) * 1/2 * 2)/sqrt(3)", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_absorbs_outer_scale_into_quotient() {
        let mut ctx = Context::new();
        let expr = parse("2 * ((atanh(x^2/sqrt(3))/2)/sqrt(3))", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_cancels_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("-1*3/(3*cosh((3*x+1)^(1/2)))", &mut ctx).unwrap();
        let compact =
            compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(&mut ctx, expr, "x");
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "-1 / cosh(sqrt(3 * x + 1))");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_extracts_scaled_sqrt_square_factor() {
        let mut ctx = Context::new();
        let expr = parse("25*sqrt(12/25)", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "10 * sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_keeps_fractional_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("-1/(3*(x^2+x-1))", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "-1 / (3 * (x^2 + x - 1))");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_additive_terms_recurses_into_terms() {
        let mut ctx = Context::new();
        let expr = parse("1/2*ln(abs(x+1)) + 1/2*(x^2/2) - 1/2*x", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold_additive_terms(&mut ctx, expr);

        assert_eq!(
            rendered(&ctx, folded),
            "1/2 * ln(|x + 1|) + 1/4 * x^2 - 1/2 * x"
        );
    }

    #[test]
    fn ln_sum_of_equal_derivative_roots_presentation_accepts_scaled_affines() {
        let mut ctx = Context::new();
        let expr = parse("ln(sqrt(2*x+1)+sqrt(2*x+3))", &mut ctx).unwrap();
        let compact = ln_sum_of_equal_derivative_roots_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| {
                panic!("scaled affine equal-derivative root sum should be recognized")
            });

        assert_eq!(
            rendered(&ctx, compact),
            "1 / (sqrt(2 * x + 1) * sqrt(2 * x + 3))"
        );
    }

    #[test]
    fn bounded_trig_positive_shift_sqrt_derivative_presentation_accepts_multi_function_sum() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(cos(x)+2*sin(x)*cos(x)+4)", &mut ctx).unwrap();
        let compact = sqrt_bounded_trig_positive_shift_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("positive shifted bounded trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(cos(2 * x) - 1/2 * sin(x)) / sqrt(sin(2 * x) + cos(x) + 4)"
        );
    }

    #[test]
    fn additive_trig_reciprocal_subtraction_sqrt_derivative_presentation_is_direct() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sin(2*x)+cos(x)-2/x)", &mut ctx).unwrap();
        let (compact, required_positive, required_conditions) =
            sqrt_additive_trig_polynomial_derivative_presentation(&mut ctx, expr, "x")
                .unwrap_or_else(|| panic!("subtracted reciprocal trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(2 * cos(2 * x) * x^2 + 2 - sin(x) * x^2) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) - 2 / x))"
        );
        assert_eq!(
            rendered(&ctx, required_positive),
            "sin(2 * x) + cos(x) - 2 / x"
        );
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn additive_trig_reciprocal_addition_sqrt_derivative_presentation_stays_direct() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sin(2*x)+cos(x)+1/x)", &mut ctx).unwrap();
        let (compact, required_positive, required_conditions) =
            sqrt_additive_trig_polynomial_derivative_presentation(&mut ctx, expr, "x")
                .unwrap_or_else(|| panic!("added reciprocal trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(2 * cos(2 * x) * x^2 - sin(x) * x^2 - 1) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) + 1 / x))"
        );
        assert_eq!(
            rendered(&ctx, required_positive),
            "sin(2 * x) + cos(x) + 1 / x"
        );
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn inverse_sqrt_quotient_arg_result_detects_compact_inverse_sqrt_substitution() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(x^2/sqrt(3))", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, expr));

        let rationalized = parse("arcsin(1/3 * sqrt(3) * x^2)", &mut ctx).unwrap();

        assert!(!inverse_sqrt_quotient_arg_result(&ctx, rationalized));

        let arctan = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, arctan));
    }

    #[test]
    fn compact_negative_half_power_result_for_integration_presentation_uses_sqrt_denominator() {
        let mut ctx = Context::new();
        let expr = parse("-2*(x^2+x+1)^(-1/2)", &mut ctx).unwrap();
        let compact =
            compact_negative_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "-2 / sqrt(x^2 + x + 1)");
    }

    #[test]
    fn compact_negative_three_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(3*(x^2+x+1)^(3/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))"
        );
    }

    #[test]
    fn compact_negative_five_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(5*(x^2+x+1)^(5/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (5 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^2)"
        );
    }

    #[test]
    fn compact_negative_seven_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(7*(x^2+x+1)^(7/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (7 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^3)"
        );
    }

    #[test]
    fn compact_negative_nine_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(9*(x^2+x+1)^(9/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (9 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^4)"
        );
    }

    #[test]
    fn compact_negative_eleven_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(11*(x^2+x+1)^(11/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (11 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^5)"
        );
    }

    #[test]
    fn compact_negative_thirteen_half_power_result_for_integration_presentation_uses_sqrt_product()
    {
        let mut ctx = Context::new();
        let expr = parse("-2/(13*(x^2+x+1)^(13/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (13 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^6)"
        );
    }

    #[test]
    fn compact_negative_three_half_power_result_requires_conditional_domain_signal() {
        let mut ctx = Context::new();
        let expr = parse("-2/(3*(x^2-1)^(3/2))", &mut ctx).unwrap();

        assert!(
            compact_negative_three_half_power_result_for_integration_presentation(
                &mut ctx, expr, "x", false,
            )
            .is_none()
        );

        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", true,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (3 * sqrt(x^2 - 1) * (x^2 - 1))"
        );
    }

    #[test]
    fn compact_positive_half_power_result_for_integration_presentation_uses_sqrt() {
        let mut ctx = Context::new();
        let expr = parse("2*(x^2+x+1)^(1/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "2 * sqrt(x^2 + x + 1)");
    }

    #[test]
    fn compact_positive_three_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("2/3*(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1)"
        );
    }

    #[test]
    fn compact_negative_three_half_power_result_for_integration_presentation_keeps_outer_sign() {
        let mut ctx = Context::new();
        let expr = parse("-2/3*(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-(2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))"
        );
    }

    #[test]
    fn compact_acosh_surd_width_arg_for_integration_presentation_uses_sqrt_denominator() {
        let mut ctx = Context::new();
        let expr = parse("acosh(sqrt(5)*(x^2+x)/5)", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, expr).unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let normalized = parse("acosh(1/5*sqrt(5)*(x^2+x))", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, normalized)
                .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let normalized_power = parse("acosh(1/5*5^(1/2)*(x^2+x))", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, normalized_power)
                .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");
    }

    #[test]
    fn self_normalized_projection_presentation_accepts_quadratic_numerator() {
        let mut ctx = Context::new();
        let expr = parse("arccos((x^2+x+1)/sqrt((x^2+x+1)^2+5))", &mut ctx).unwrap();
        let derivative = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-sqrt(5) * (2 * x + 1) / ((x^2 + x + 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_self_normalized_surd_quotient_accepts_inverse_sqrt_product_arg() {
        let mut ctx = Context::new();
        let expr = parse("arctan((4*x^2+4*x+2)^(-1/2)*(2*x+1))", &mut ctx).unwrap();
        let derivative =
            arctan_self_normalized_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "2 / (sqrt(4 * x^2 + 4 * x + 2) * (2 * (2 * x + 1)^2 + 1))"
        );
    }

    #[test]
    fn atanh_self_normalized_surd_quotient_accepts_inverse_sqrt_product_arg() {
        let mut ctx = Context::new();
        let expr = parse("atanh(((2*x+1)^2+3)^(-1/2)*(2*x+1))", &mut ctx).unwrap();
        let derivative =
            atanh_self_normalized_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / sqrt((2 * x + 1)^2 + 3)");
    }

    #[test]
    fn arctan_self_normalized_surd_reciprocal_accepts_inverse_denominator_arg() {
        let mut ctx = Context::new();
        let expr = parse("arctan((x^2+1)^(1/2)*x^(-1))", &mut ctx).unwrap();
        let (derivative, required_condition) =
            arctan_self_normalized_surd_reciprocal_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-1 / (sqrt(x^2 + 1) * (2 * x^2 + 1))"
        );
        assert!(matches!(
            required_condition,
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "x"
        ));
    }

    #[test]
    fn self_normalized_projection_presentation_accepts_negated_argument() {
        let mut ctx = Context::new();
        let expr = parse("arccos(-x/sqrt(x^2+1))", &mut ctx).unwrap();
        let derivative = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "1 / (x^2 + 1)");
    }

    #[test]
    fn self_normalized_projection_presentation_normalizes_negated_quadratic_content() {
        let mut ctx = Context::new();
        let expr = parse("arccos(-(x^2+x+1)/sqrt((x^2+x+1)^2+5))", &mut ctx).unwrap();
        let derivative = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "sqrt(5) * (2 * x + 1) / ((x^2 + x + 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_surd_quotient_scaled_compact_derivative_avoids_rationalized_route() {
        let mut ctx = Context::new();
        let expr = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();
        let derivative =
            arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / ((2 * x + 2)^2 + 6)");
    }

    #[test]
    fn constant_scaled_arctan_surd_quotient_scaled_derivative_reuses_compact_route() {
        let mut ctx = Context::new();
        let expr = parse("7*arctan((2*x+1)/sqrt(3))/sqrt(3)", &mut ctx).unwrap();
        let derivative =
            constant_scaled_arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x")
                .unwrap();

        assert_eq!(rendered(&ctx, derivative), "7 / (2 * (x^2 + x + 1))");
    }

    #[test]
    fn constant_scaled_reciprocal_sqrt_product_arctan_derivative_reuses_compact_route() {
        let mut ctx = Context::new();
        let expr = parse("2*arctan(1/(sqrt(x)*(x+1)))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
                &mut ctx, expr, "x",
            )
            .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(3 * x + 1) / ((x * (x + 1)^2 + 1) * sqrt(x))"
        );
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn arctan_surd_quotient_compact_derivative_avoids_rationalized_route() {
        let mut ctx = Context::new();
        let expr = parse("arctan((2*x+2)/sqrt(6))", &mut ctx).unwrap();
        let derivative = arctan_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "2 * sqrt(6) / ((2 * x + 2)^2 + 6)"
        );
    }

    #[test]
    fn arctan_affine_by_parts_compact_derivative_accepts_polynomial_remainder() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3",
            &mut ctx,
        )
        .unwrap();
        let derivative = arctan_affine_by_parts_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "arctan(1 - x) * x^2");

        let normalized = parse(
            "1/6*(2*ln(x^2+2-2*x) + 2*arctan(1-x)*x^3 + 4*arctan(1-x) + x^2 + 4*x)",
            &mut ctx,
        )
        .unwrap();
        let derivative =
            arctan_affine_by_parts_compact_derivative(&mut ctx, normalized, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "x^2 * arctan(1 - x)");
    }

    #[test]
    fn arctan_affine_by_parts_compact_derivative_runs_in_diff_pipeline() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "diff(((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3, x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(rendered(&simplifier.context, result), "arctan(1 - x) * x^2");
    }

    #[test]
    fn sqrt_additive_tan_exp_polynomial_derivative_presentation_accepts_exp_term() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+exp(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(e^x + sec(x)^2 + 1) / (2 * sqrt(tan(x) + e^x + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + e^x + x");
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn sqrt_additive_tan_cos_square_polynomial_derivative_compacts_power_exponent() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+cos(x)^2+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(sec(x)^2 + 1 - 2 * cos(x) * sin(x)) / (2 * sqrt(tan(x) + cos(x)^2 + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + cos(x)^2 + x");
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn sqrt_additive_tan_exp_linear_polynomial_derivative_presentation_accepts_chain_factor() {
        for (input, expected_result, expected_radicand) in [
            (
                "sqrt(tan(x)+exp(2*x)+x)",
                "(sec(x)^2 + 2 * e^(2 * x) + 1) / (2 * sqrt(tan(x) + e^(2 * x) + x))",
                "tan(x) + e^(2 * x) + x",
            ),
            (
                "sqrt(tan(x)+exp(2*x+1)+x)",
                "(sec(x)^2 + 2 * e^(2 * x + 1) + 1) / (2 * sqrt(tan(x) + e^(2 * x + 1) + x))",
                "tan(x) + e^(2 * x + 1) + x",
            ),
            (
                "sqrt(tan(x)+exp(-2*x)+x)",
                "(sec(x)^2 + 1 - 2 * e^(-2 * x)) / (2 * sqrt(tan(x) + e^(-2 * x) + x))",
                "tan(x) + e^(-2 * x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(
                rendered(&ctx, radicand),
                expected_radicand,
                "input: {input}"
            );
            assert_eq!(required_conditions.len(), 1, "input: {input}");
        }
    }

    #[test]
    fn sqrt_additive_tan_reciprocal_sqrt_derivative_presentation_accepts_inverse_sqrt_term() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+1/sqrt(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + 1 / sqrt(x) + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + 1 / sqrt(x) + x");
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn sqrt_additive_tan_negative_reciprocal_sqrt_derivative_presentation_accepts_signed_inverse_sqrt_term(
    ) {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)-1/sqrt(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 1) / (4 * x * sqrt(x) * sqrt(tan(x) - 1 / sqrt(x) + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) - 1 / sqrt(x) + x");
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn sqrt_additive_tan_mixed_sqrt_and_reciprocal_sqrt_derivative_presentation_uses_common_denominator(
    ) {
        for (input, expected_result, expected_radicand) in [
            (
                "sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x))",
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x",
            ),
            (
                "sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x))",
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(rendered(&ctx, radicand), expected_radicand, "input: {input}");
            assert_eq!(required_conditions.len(), 3, "input: {input}");
        }
    }

    #[test]
    fn arctan_sqrt_additive_tan_mixed_sqrt_derivative_presentation_reuses_inner_common_denominator()
    {
        for (input, expected_result, expected_radicand) in [
            (
                "arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x))",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x) * (tan(x) + sqrt(x) + 1 / sqrt(x) + x + 1))",
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x",
            ),
            (
                "arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x))",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x) * (tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x + 1))",
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                arctan_sqrt_additive_tan_polynomial_derivative_presentation(
                    &mut ctx, target, "x",
                )
                .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(rendered(&ctx, radicand), expected_radicand, "input: {input}");
            assert_eq!(required_conditions.len(), 3, "input: {input}");
        }
    }

    #[test]
    fn compact_arctan_additive_terms_accepts_negative_affine_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "1/3*x^3*arctan(1-x) + 1/3*ln(x^2+2-2*x) + 2/3*arctan(1-x) + 1/6*x^2 + 2/3*x",
            &mut ctx,
        )
        .unwrap();
        let compact =
            compact_arctan_additive_terms_for_calculus_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact).matches("arctan(1 - x)").count(), 1);

        let raw_by_parts = parse(
            "1/3*x^3*arctan(1-x) - (-1/3*ln(x^2+2-2*x) - 2/3*arctan(1-x) - 1/6*x^2 - 2/3*x)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            compact_arctan_additive_terms_for_calculus_presentation(&mut ctx, raw_by_parts, "x")
                .unwrap();
        assert_eq!(rendered(&ctx, compact).matches("arctan(1 - x)").count(), 1);

        let duplicate_companions = parse(
            "1/3*ln(x^2+2-2*x) + 1/2*ln(x^2+2-2*x) + 1/3*x^3*arctan(1-x) + 1/2*x^2*arctan(1-x) + 2/3*arctan(1-x) + 1/6*x^2 + 1/2*x + 2/3*x",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_arctan_additive_terms_for_calculus_presentation(
            &mut ctx,
            duplicate_companions,
            "x",
        )
        .unwrap();
        let rendered = rendered(&ctx, compact);
        assert_eq!(rendered.matches("ln(x^2 + 2 - 2 * x)").count(), 1);
        assert!(rendered.contains("5/6 * ln(x^2 + 2 - 2 * x)"));
        assert!(rendered.contains("7/6 * x"));
    }

    #[test]
    fn integrate_pipeline_compacts_negative_affine_arctan_by_parts_result() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("integrate(x^2*arctan(1-x), x)", &mut simplifier.context).unwrap();
        let target = match simplifier.context.get(expr) {
            Expr::Function(_, args) => args[0],
            _ => expr,
        };
        assert!(
            cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
                &mut simplifier.context,
                target,
                "x",
            )
        );
        assert!(polynomial_times_arctan_affine_integrand_for_diff_shortcut(
            &simplifier.context,
            target,
            "x"
        ));
        let raw = integrate(&mut simplifier.context, target, "x").unwrap();
        let raw = fold_numeric_mul_constants_for_hold(&mut simplifier.context, raw);
        let compact = compact_arctan_additive_terms_for_calculus_presentation(
            &mut simplifier.context,
            raw,
            "x",
        )
        .unwrap();
        assert_eq!(
            rendered(&simplifier.context, compact)
                .matches("arctan(1 - x)")
                .count(),
            1
        );
        let (result, _steps) = simplifier.simplify(expr);
        let rendered = rendered(&simplifier.context, result);

        assert_eq!(rendered.matches("arctan(1 - x)").count(), 1);
    }

    #[test]
    fn arctan_surd_quotient_compact_derivative_normalizes_negative_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("arctan(-(x^2+x+1)/sqrt(5))", &mut ctx).unwrap();
        let derivative = arctan_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * x + 1) * sqrt(5) / ((x^2 + x + 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_surd_quotient_scaled_compact_derivative_normalizes_negative_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("arctan((1-x-x^2)/sqrt(5))/sqrt(5)", &mut ctx).unwrap();
        let derivative =
            arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * x + 1) / ((x^2 + x - 1)^2 + 5)"
        );
    }

    #[test]
    fn constant_divisor_compact_derivative_accepts_asinh_surd_quotient() {
        let mut ctx = Context::new();
        let expr = parse("asinh((1-x-x^2)/sqrt(5))/sqrt(5)", &mut ctx).unwrap();
        let derivative = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "(-2 * x - 1) / (sqrt(5) * sqrt((1 - x - x^2)^2 + 5))"
        );
    }

    #[test]
    fn constant_divisor_compact_derivative_accepts_atanh_surd_quotient() {
        let mut ctx = Context::new();
        let expr = parse("atanh((x^2+x+1)/sqrt(7))/sqrt(7)", &mut ctx).unwrap();
        let derivative = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "(2 * x + 1) / (7 - (x^2 + x + 1)^2)"
        );
    }

    #[test]
    fn constant_scaled_acosh_affine_derivative_keeps_compact_roots() {
        let mut ctx = Context::new();
        let expr = parse("acosh(x+1)/2", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_acosh_affine_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (2 * sqrt(x) * sqrt(x + 2))"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(
            required_conditions[0].display(&ctx),
            "x > 0",
            "constant-scaled acosh shortcut must preserve the affine real-domain guard"
        );
    }

    #[test]
    fn post_diff_presentation_compacts_sqrt_over_var_shift_square() {
        let mut ctx = Context::new();
        let target = parse("arctan(sqrt(x)) + sqrt(x)/(x+1)", &mut ctx).unwrap();
        let (direct, _required) =
            arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(rendered(&ctx, direct), "1 / ((x + 1)^2 * sqrt(x))");

        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "diff(arctan(sqrt(x)) + sqrt(x)/(x+1), x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(
            rendered(&simplifier.context, result),
            "1 / ((x + 1)^2 * sqrt(x))"
        );

        let expr = parse(
            "diff(8*arctan(2*sqrt(x)) + 4*sqrt(x)/(x+1/4), x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(
            rendered(&simplifier.context, result),
            "1 / ((x + 1/4)^2 * sqrt(x))"
        );
    }

    #[test]
    fn ln_sqrt_affine_gap_derivative_keeps_compact_radicand() {
        let mut ctx = Context::new();
        let expr = parse("ln(sqrt((2*x+1)^2-4)+(2*x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_plus_polynomial_direct_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / sqrt((2 * x + 1)^2 - 4)");
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 1/2");

        let expr = parse("ln(sqrt((2*x+1)^2-4)-(2*x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_plus_polynomial_direct_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "-2 / sqrt((2 * x + 1)^2 - 4)");
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x < -3/2");
    }

    #[test]
    fn ln_sqrt_positive_shift_nonpolynomial_diff_uses_direct_denominator() {
        let mut ctx = Context::new();
        let target = parse("ln(1+sqrt(sin(x)+2))", &mut ctx).unwrap();
        let (derivative, conditions) =
            ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "cos(x) / (2 * sqrt(sin(x) + 2) * (sqrt(sin(x) + 2) + 1))"
        );
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].display(&ctx), "sin(x) + 2 > 0");
    }

    #[test]
    fn ln_sqrt_positive_shift_exp_diff_does_not_reintroduce_ln_e() {
        let mut ctx = Context::new();
        let target = parse("ln(1+sqrt(exp(x)+1))", &mut ctx).unwrap();
        let (derivative, conditions) =
            ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "e^x / (2 * sqrt(e^x + 1) * (sqrt(e^x + 1) + 1))"
        );
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].display(&ctx), "e^x + 1 > 0");
    }

    #[test]
    fn ln_reciprocal_trig_affine_sqrt_diff_uses_held_compact_derivative() {
        let mut ctx = Context::new();
        let target = parse("ln(sec(sqrt(3*x+1))+tan(sqrt(3*x+1)))", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_reciprocal_trig_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "3 / (2 * sqrt(3 * x + 1) * cos(sqrt(3 * x + 1)))"
        );
        assert_eq!(required_conditions.len(), 3);

        let target = parse("ln(csc(sqrt(3*x+1))-cot(sqrt(3*x+1)))", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_reciprocal_trig_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "3 / (2 * sqrt(3 * x + 1) * sin(sqrt(3 * x + 1)))"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn compact_direct_sqrt_trig_log_derivative_integrand_preserves_tangent_form() {
        let mut ctx = Context::new();
        let expr = parse("tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn ln_constant_shifted_tan_sqrt_diff_uses_held_compact_derivative() {
        let mut ctx = Context::new();
        let target = parse("ln(tan(sqrt(x))+1)", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_constant_shifted_tan_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "1 / (2 * sqrt(x) * cos(sqrt(x))^2 * (tan(sqrt(x)) + 1))"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn ln_constant_shifted_tan_affine_sqrt_diff_uses_held_compact_derivative() {
        let mut ctx = Context::new();
        let target = parse("ln(1+tan(sqrt(2*x+3)))", &mut ctx).unwrap();
        let (result, required_conditions) =
            ln_constant_shifted_tan_sqrt_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "1 / (sqrt(2 * x + 3) * cos(sqrt(2 * x + 3))^2 * (tan(sqrt(2 * x + 3)) + 1))"
        );
        assert_eq!(required_conditions.len(), 3);
    }

    #[test]
    fn compact_direct_sqrt_trig_log_derivative_integrand_accepts_mul_div_chain() {
        let mut ctx = Context::new();
        let expr = parse("tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_half_power_product() {
        let mut ctx = Context::new();
        let expr = parse("1/2*tanh(sqrt(x))*x^(-1/2)", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_tanh_denominator() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)/(2*x*tanh(sqrt(x)))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "1 / (2 * tanh(sqrt(x)) * sqrt(x))");
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_sinh_cosh_quotient() {
        let mut ctx = Context::new();
        let expr = parse("-sinh(sqrt(x))*x^(-1/2)/(2*cosh(sqrt(x)))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "-tanh(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_direct_sqrt_hyperbolic_log_derivative_integrand_accepts_cosh_sinh_quotient() {
        let mut ctx = Context::new();
        let expr = parse("-cosh(sqrt(x))*x^(-1/2)/(2*sinh(sqrt(x)))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_hyperbolic_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "-1 / (2 * tanh(sqrt(x)) * sqrt(x))"
        );
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_preserves_negative_tangent_form() {
        let mut ctx = Context::new();
        let inner = parse("sin(sqrt(x))*x^(-1/2)/(2*cos(sqrt(x)))", &mut ctx).unwrap();
        let expr = ctx.add(Expr::Neg(inner));
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "-tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_accepts_half_power_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "((3*x+1)^(-1/2) * sin((3*x+1)^(1/2)) * 3)/(2 * cos((3*x+1)^(1/2)))",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_accepts_scaled_radicand_denominator() {
        let mut ctx = Context::new();
        let expr = parse(
            "((3*x+1)^(1/2) * sin((3*x+1)^(1/2)) * 3)/(cos((3*x+1)^(1/2)) * (6*x+2))",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn sqrt_trig_log_antiderivative_derivative_presentation_compacts_shifted_chain() {
        let mut ctx = Context::new();
        let expr = parse("-ln(abs(cos(sqrt(3*x+1))))", &mut ctx).unwrap();
        let (compact, conditions) =
            sqrt_trig_log_antiderivative_derivative_presentation(&mut ctx, expr, "x").unwrap();
        let rendered_conditions: Vec<_> = conditions
            .iter()
            .map(|condition| match condition {
                crate::ImplicitCondition::Positive(expr) => {
                    format!("{} > 0", rendered(&ctx, *expr))
                }
                crate::ImplicitCondition::NonZero(expr) => {
                    format!("{} != 0", rendered(&ctx, *expr))
                }
                other => format!("{other:?}"),
            })
            .collect();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
        assert_eq!(
            rendered_conditions,
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) != 0"]
        );
    }

    #[test]
    fn compact_positive_cosh_log_presentation_accepts_half_power_argument() {
        let mut ctx = Context::new();
        let expr = parse("ln(cosh((3*x+1)^(1/2)))", &mut ctx).unwrap();
        let compact =
            compact_positive_cosh_log_abs_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "ln(cosh(sqrt(3 * x + 1)))");
    }

    #[test]
    fn calculus_result_presentation_expands_trig_odd_power_primitive_coefficients() {
        let cases = [
            ("1/3*(cos(x)^3 - 3*cos(x))", "1/3 * cos(x)^3 - cos(x)"),
            ("1/3*(3*sin(x) - sin(x)^3)", "sin(x) - 1/3 * sin(x)^3"),
            (
                "1/6*(cos(2*x+1)^3 - 3*cos(2*x+1))",
                "1/6 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1)",
            ),
            (
                "1/5*(10/3*cos(x)^3 - cos(x)^5 - 5*cos(x))",
                "2/3 * cos(x)^3 - cos(x) - 1/5 * cos(x)^5",
            ),
            (
                "1/5*(sin(x)^5 + 5*sin(x) - 10/3*sin(x)^3)",
                "sin(x) + 1/5 * sin(x)^5 - 2/3 * sin(x)^3",
            ),
            (
                "1/10*(10/3*cos(2*x+1)^3 - cos(2*x+1)^5 - 5*cos(2*x+1))",
                "1/3 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1) - 1/10 * cos(2 * x + 1)^5",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn calculus_result_presentation_expands_trig_square_primitive_coefficients() {
        let cases = [
            ("1/4*(2*x - sin(2*x))", "1/2 * x - 1/4 * sin(2 * x)"),
            ("1/4*(sin(2*x) + 2*x)", "1/4 * sin(2 * x) + 1/2 * x"),
            ("1/8*(4*x - sin(4*x+2))", "1/2 * x - 1/8 * sin(4 * x + 2)"),
            ("1/8*(sin(4*x+2) + 4*x)", "1/8 * sin(4 * x + 2) + 1/2 * x"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn calculus_result_presentation_compacts_negative_half_power_product_denominator() {
        let cases = [
            (
                "cos(x)/2*(sin(x)+1)^(1/2-1)",
                "cos(x) / (2 * sqrt(sin(x) + 1))",
            ),
            (
                "((ln(x)+1)^(1/2-1)/2)*(1/x)",
                "1 / (2 * x * sqrt(ln(x) + 1))",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn calculus_result_presentation_compacts_rationalized_symbolic_sqrt_denominator() {
        let cases = [
            (
                "cos(x)*sqrt(sin(x)+1)/(2*sin(x)+2)",
                "cos(x) / (2 * sqrt(sin(x) + 1))",
            ),
            ("sqrt(ln(x)+1)/(ln(x)+1)", "1 / sqrt(ln(x) + 1)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = try_calculus_result_presentation(&mut ctx, expr).unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }

    #[test]
    fn post_calculus_presentation_compacts_nested_integral_source() {
        let mut ctx = Context::new();
        let source = parse(
            "diff(integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x), x)",
            &mut ctx,
        )
        .unwrap();
        let result = parse(
            "((3*x+1)^(1/2) * sin((3*x+1)^(1/2)) * 3)/(cos((3*x+1)^(1/2)) * (6*x+2))",
            &mut ctx,
        )
        .unwrap();
        let compact = try_post_calculus_presentation(&mut ctx, source, result).unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }
}
