//! Expression transform helpers: binary, pow, div, and function simplification.
//!
//! These methods are extracted with `#[inline(never)]` to reduce the stack frame
//! size of `transform_expr_recursive`.

use super::*;
use crate::rule::Rule;
use cas_math::expr_predicates::contains_named_var;
use cas_math::factoring_support::{
    try_rewrite_difference_of_squares_product_expr, DifferenceOfSquaresProductRewriteKind,
};
use cas_math::logarithm_inverse_support::try_rewrite_exponential_log_inverse_expr;
use cas_math::numeric::as_i64;
use cas_math::polynomial::Polynomial;
use cas_math::pow_preorder_support::{try_plan_sqrt_square_pow_rewrite, SqrtSquarePowRewriteKind};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use smallvec::SmallVec;

fn diff_call_should_preserve_raw_target_for_direct_derivative(
    ctx: &cas_ast::Context,
    name: &str,
    args: &[ExprId],
) -> bool {
    if name != "diff" || args.len() != 2 {
        return false;
    }
    diff_target_should_preserve_raw_derivative_route(ctx, args[0], args[1])
}

fn integrate_call_should_preserve_raw_target_for_direct_integration(
    ctx: &cas_ast::Context,
    name: &str,
    args: &[ExprId],
) -> bool {
    if name != "integrate" {
        return false;
    }

    let var_name = match args.len() {
        1 => "x",
        2 => {
            let Some(var_name) = diff_variable_name(ctx, args[1]) else {
                return false;
            };
            var_name
        }
        _ => return false,
    };

    integrate_target_is_negative_denominator_polynomial_power_substitution(ctx, args[0], var_name)
        || integrate_target_is_reciprocal_quotient_denominator_polynomial_power_substitution(
            ctx, args[0], var_name,
        )
        || integrate_target_is_arctan_sqrt_var_reciprocal(ctx, args[0], var_name)
        || integrate_target_is_inverse_hyperbolic_sqrt_reciprocal(ctx, args[0], var_name)
        || integrate_target_is_sqrt_trig_reciprocal_derivative(ctx, args[0], var_name)
        || integrate_target_is_sqrt_trig_log_derivative(ctx, args[0], var_name)
        || integrate_target_is_sqrt_reciprocal_trig_log_derivative(ctx, args[0], var_name)
        || integrate_target_is_inverse_trig_polynomial_substitution(ctx, args[0], var_name)
        || crate::rules::calculus::is_public_algorithmic_backend_symbolic_positive_quadratic_fallback_shape(
            ctx,
            args[0],
            var_name,
            0,
        )
        || crate::rules::calculus::is_public_algorithmic_backend_symbolic_indefinite_square_fallback_shape(
            ctx,
            args[0],
            var_name,
            0,
        )
        || integrate_target_is_repeated_trig_by_parts_kernel(ctx, args[0], var_name)
        || integrate_target_is_repeated_exp_by_parts_kernel(ctx, args[0], var_name)
        || (crate::rule::steps_enabled()
            && (integrate_target_is_affine_times_basic_by_parts_kernel(ctx, args[0], var_name)
                || integrate_target_is_proportional_affine_ln_by_parts_kernel(
                    ctx, args[0], var_name,
                )))
}

fn integrate_target_is_affine_times_basic_by_parts_kernel(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(target) else {
        return false;
    };

    (expr_is_additive_affine_polynomial(ctx, *left, var_name)
        && expr_is_basic_linear_by_parts_kernel(ctx, *right, var_name))
        || (expr_is_additive_affine_polynomial(ctx, *right, var_name)
            && expr_is_basic_linear_by_parts_kernel(ctx, *left, var_name))
}

fn expr_is_additive_affine_polynomial(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return false;
    };
    poly.degree() == 1
}

fn expr_is_basic_linear_by_parts_kernel(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos | cas_ast::BuiltinFn::Exp
                    )
                ) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) => {
            *exp
        }
        _ => return false,
    };

    let Ok(poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return false;
    };
    poly.degree() == 1
}

fn integrate_target_is_proportional_affine_ln_by_parts_kernel(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(target) else {
        return false;
    };

    expr_is_proportional_affine_ln_pair(ctx, *left, *right, var_name)
        || expr_is_proportional_affine_ln_pair(ctx, *right, *left, var_name)
}

fn expr_is_proportional_affine_ln_pair(
    ctx: &cas_ast::Context,
    cofactor: ExprId,
    log_expr: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(log_expr) else {
        return false;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Ln) {
        return false;
    }

    let Ok(arg_poly) = Polynomial::from_expr(ctx, args[0], var_name) else {
        return false;
    };
    let Ok(cofactor_poly) = Polynomial::from_expr(ctx, cofactor, var_name) else {
        return false;
    };
    if arg_poly.degree() != 1 || cofactor_poly.degree() != 1 {
        return false;
    }

    let arg_offset = arg_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let arg_slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let cofactor_offset = cofactor_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let cofactor_slope = cofactor_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);

    !arg_slope.is_zero()
        && !cofactor_slope.is_zero()
        && cofactor_offset * arg_slope == cofactor_slope * arg_offset
}

fn diff_target_should_preserve_raw_derivative_route(
    ctx: &cas_ast::Context,
    target: ExprId,
    variable: ExprId,
) -> bool {
    if diff_target_is_direct_tan_or_tan_square(ctx, target) {
        return true;
    }

    let Some(var_name) = diff_variable_name(ctx, variable) else {
        return false;
    };
    diff_target_is_repeated_trig_by_parts_integrate_call(ctx, target, var_name)
        || diff_target_is_rational_linear_partial_fraction_integrate_call(ctx, target, var_name)
        || diff_target_is_rational_linear_positive_quadratic_integrate_call(ctx, target, var_name)
        || diff_target_is_reciprocal_positive_quadratic_arctan_integrate_call(ctx, target, var_name)
        || diff_target_is_reciprocal_trig_derivative_product_integrate_call(ctx, target, var_name)
        || diff_target_is_positive_quadratic_power_integrate_call(ctx, target, var_name)
        || diff_target_is_quadratic_affine_ln_by_parts_integrate_call(ctx, target, var_name)
        || diff_target_is_arcsin_inverse_sqrt_product_integrate_call(ctx, target, var_name)
        || diff_target_is_affine_polynomial_times_direct_builtin_derivative_target(
            ctx, target, var_name,
        )
        || diff_target_is_affine_times_arctan_reciprocal_affine(ctx, target, var_name)
        || diff_target_is_arctan_reciprocal_affine_by_parts_sum(ctx, target, var_name)
        || diff_target_is_arctan_affine_by_parts_sum(ctx, target, var_name)
        || diff_target_is_inverse_tangent_direct_trig_linear_arg(ctx, target, var_name)
        || diff_target_is_ln_reciprocal_trig_sqrt(ctx, target, var_name)
        || diff_target_is_ln_constant_shifted_tan_sqrt(ctx, target, var_name)
        || diff_target_is_log_ratio_single_linear_pole(ctx, target, var_name)
        || diff_target_is_separated_linear_log_abs_with_linear_pole(ctx, target, var_name)
        || diff_target_is_positive_quadratic_log_abs_with_linear_pole(ctx, target, var_name)
        || diff_target_is_scaled_reciprocal_trig_shifted_sqrt(ctx, target, var_name)
        || diff_target_is_scaled_inverse_tangent_reciprocal_sqrt_product(ctx, target, var_name)
        || diff_target_is_constant_scaled_inverse_tangent_linear_positive_rational_radius(
            ctx, target, var_name,
        )
        || diff_target_is_inverse_reciprocal_trig_surd_scaled_quadratic(ctx, target, var_name)
        || diff_target_is_reciprocal_positive_shifted_sqrt(ctx, target, var_name)
        || diff_target_is_reciprocal_sqrt_times_nonzero_shifted_sqrt(ctx, target, var_name)
        || diff_target_is_bounded_trig_positive_shift_sqrt(ctx, target)
        || diff_target_is_additive_trig_polynomial_sqrt(ctx, target, var_name)
        || diff_target_is_arctan_sqrt_additive_trig_polynomial(ctx, target, var_name)
        || diff_target_is_scaled_bounded_inverse_trig_surd_quotient(ctx, target, var_name)
        || diff_target_is_bounded_inverse_trig_self_normalized_projection(ctx, target, var_name)
        || expr_is_positive_integer_power_of_low_degree_polynomial(ctx, target, var_name)
        || diff_target_is_constant_scaled_positive_polynomial_power(ctx, target, var_name)
        || diff_target_is_constant_scaled_reciprocal_polynomial_power(ctx, target, var_name)
        || diff_target_is_constant_scaled_atanh_surd_polynomial(ctx, target, var_name)
        || crate::rules::calculus::diff_target_is_tanh_cubic_sech_fourth_primitive(
            ctx, target, var_name,
        )
}

fn diff_target_is_log_ratio_single_linear_pole(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() != 2 {
        return false;
    }

    let mut log_factors: Option<(Polynomial, Polynomial)> = None;
    let mut pole_factor: Option<Polynomial> = None;

    for (term, _sign) in terms {
        if let Some((num, den)) = log_abs_linear_quotient_factors(ctx, term, var_name) {
            if log_factors.replace((num, den)).is_some() {
                return false;
            }
            continue;
        }
        if let Some(linear) = reciprocal_linear_factor_for_raw_diff_target(ctx, term, var_name) {
            if pole_factor.replace(linear).is_some() {
                return false;
            }
            continue;
        }
        return false;
    }

    let Some((num, den)) = log_factors else {
        return false;
    };
    let Some(pole) = pole_factor else {
        return false;
    };
    pole == num || pole == den
}

fn diff_target_is_separated_linear_log_abs_with_linear_pole(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() < 3 {
        return false;
    }

    let mut log_factors = Vec::new();
    let mut pole_factors = Vec::new();

    for (term, _sign) in terms {
        if !contains_named_var(ctx, term, var_name) {
            continue;
        }
        if let Some(linear) = scaled_log_abs_linear_factor_for_raw_diff_target(ctx, term, var_name)
        {
            log_factors.push(linear);
            continue;
        }
        if let Some(linear) = reciprocal_linear_factor_for_raw_diff_target(ctx, term, var_name) {
            pole_factors.push(linear);
            continue;
        }
        return false;
    }

    log_factors.len() >= 2
        && !pole_factors.is_empty()
        && pole_factors.iter().any(|pole| {
            log_factors
                .iter()
                .any(|log_factor| linear_polynomials_are_proportional(pole, log_factor))
        })
}

fn diff_target_is_positive_quadratic_log_abs_with_linear_pole(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() < 3 {
        return false;
    }

    let mut has_positive_quadratic_log = false;
    let mut log_factors = Vec::new();
    let mut pole_factors = Vec::new();

    for (term, _sign) in terms {
        if !contains_named_var(ctx, term, var_name) {
            continue;
        }
        if scaled_log_positive_quadratic_for_raw_diff_target(ctx, term, var_name).is_some() {
            has_positive_quadratic_log = true;
            continue;
        }
        if let Some(linear) = scaled_log_abs_linear_factor_for_raw_diff_target(ctx, term, var_name)
        {
            log_factors.push(linear);
            continue;
        }
        if let Some(linear) = reciprocal_linear_factor_for_raw_diff_target(ctx, term, var_name) {
            pole_factors.push(linear);
            continue;
        }
        return false;
    }

    has_positive_quadratic_log
        && !log_factors.is_empty()
        && pole_factors.iter().any(|pole| {
            log_factors
                .iter()
                .any(|log_factor| linear_polynomials_are_proportional(pole, log_factor))
        })
}

fn scaled_log_positive_quadratic_for_raw_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<Polynomial> {
    let core = strip_rational_scale_for_raw_diff_target(ctx, expr)?;
    let Expr::Function(ln_fn, ln_args) = ctx.get(core) else {
        return None;
    };
    if ln_args.len() != 1 || ctx.builtin_of(*ln_fn) != Some(cas_ast::BuiltinFn::Ln) {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, ln_args[0], var_name).ok()?;
    strictly_positive_quadratic_for_raw_diff_target(&poly).then_some(poly)
}

fn strictly_positive_quadratic_for_raw_diff_target(poly: &Polynomial) -> bool {
    if poly.degree() != 2 {
        return false;
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
    if !a.is_positive() {
        return false;
    }

    let four = BigRational::from_integer(4.into());
    b.clone() * b - four * a * c < BigRational::zero()
}

fn scaled_log_abs_linear_factor_for_raw_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<Polynomial> {
    let core = strip_rational_scale_for_raw_diff_target(ctx, expr)?;
    let Expr::Function(ln_fn, ln_args) = ctx.get(core) else {
        return None;
    };
    if ln_args.len() != 1 || ctx.builtin_of(*ln_fn) != Some(cas_ast::BuiltinFn::Ln) {
        return None;
    }
    let Expr::Function(abs_fn, abs_args) = ctx.get(ln_args[0]) else {
        return None;
    };
    if abs_args.len() != 1 || ctx.builtin_of(*abs_fn) != Some(cas_ast::BuiltinFn::Abs) {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, abs_args[0], var_name).ok()?;
    (poly.degree() == 1).then_some(poly)
}

fn strip_rational_scale_for_raw_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<ExprId> {
    if cas_ast::views::as_rational_const(ctx, expr, 8).is_some() {
        return None;
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => strip_rational_scale_for_raw_diff_target(ctx, *inner),
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, *den, 8).is_some() => {
            strip_rational_scale_for_raw_diff_target(ctx, *num)
        }
        Expr::Mul(_, _) => {
            let mut core = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if cas_ast::views::as_rational_const(ctx, factor, 8).is_some() {
                    continue;
                }
                if core.replace(factor).is_some() {
                    return None;
                }
            }
            core
        }
        _ => Some(expr),
    }
}

fn linear_polynomials_are_proportional(left: &Polynomial, right: &Polynomial) -> bool {
    if left.degree() != 1 || right.degree() != 1 || right.is_zero() {
        return false;
    }

    let mut ratio = None;
    let len = left.coeffs.len().max(right.coeffs.len());
    for idx in 0..len {
        let l = left
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let r = right
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if r.is_zero() {
            if !l.is_zero() {
                return false;
            }
            continue;
        }
        let current = l / r;
        if current.is_zero() {
            return false;
        }
        if let Some(existing) = &ratio {
            if existing != &current {
                return false;
            }
        } else {
            ratio = Some(current);
        }
    }

    ratio.is_some()
}

fn log_abs_linear_quotient_factors(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(Polynomial, Polynomial)> {
    let Expr::Function(ln_fn, ln_args) = ctx.get(expr) else {
        return None;
    };
    if ln_args.len() != 1 || ctx.builtin_of(*ln_fn) != Some(cas_ast::BuiltinFn::Ln) {
        return None;
    }
    let Expr::Function(abs_fn, abs_args) = ctx.get(ln_args[0]) else {
        return None;
    };
    if abs_args.len() != 1 || ctx.builtin_of(*abs_fn) != Some(cas_ast::BuiltinFn::Abs) {
        return None;
    }
    let Expr::Div(num, den) = ctx.get(abs_args[0]) else {
        return None;
    };
    let num_poly = Polynomial::from_expr(ctx, *num, var_name).ok()?;
    let den_poly = Polynomial::from_expr(ctx, *den, var_name).ok()?;
    (num_poly.degree() == 1 && den_poly.degree() == 1).then_some((num_poly, den_poly))
}

fn reciprocal_linear_factor_for_raw_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<Polynomial> {
    let denominator = match ctx.get(expr) {
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, *num, 8).is_some() => *den,
        Expr::Pow(base, exp) if matches!(ctx.get(*exp), Expr::Number(power) if *power == -BigRational::one()) => {
            *base
        }
        _ => return None,
    };
    let poly = Polynomial::from_expr(ctx, denominator, var_name).ok()?;
    (poly.degree() == 1).then_some(poly)
}

fn diff_target_is_bounded_trig_positive_shift_sqrt(ctx: &cas_ast::Context, target: ExprId) -> bool {
    let Some(radicand) = extract_square_root_base(ctx, target) else {
        return false;
    };
    bounded_sin_cos_shift_margin_for_direct_diff_route(ctx, radicand).is_some()
}

fn diff_target_is_additive_trig_polynomial_sqrt(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Some(radicand) = extract_square_root_base(ctx, target) else {
        return false;
    };
    diff_radicand_is_additive_trig_polynomial(ctx, radicand, var_name)
}

fn diff_target_is_arctan_sqrt_additive_trig_polynomial(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Atan | cas_ast::BuiltinFn::Arctan)
        )
    {
        return false;
    }
    let Some(radicand) = extract_square_root_base(ctx, args[0]) else {
        return false;
    };
    diff_radicand_is_additive_trig_polynomial(ctx, radicand, var_name)
}

fn diff_radicand_is_additive_trig_polynomial(
    ctx: &cas_ast::Context,
    radicand: ExprId,
    var_name: &str,
) -> bool {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 6 {
        return false;
    }

    let mut has_trig_term = false;
    let mut has_tan_term = false;
    let mut has_ln_term = false;
    let mut has_sqrt_variable_term = false;
    let mut has_reciprocal_sqrt_variable_term = false;
    let mut has_reciprocal_variable_term = false;
    let mut has_variable_dependency = false;
    for (term, _) in terms {
        has_variable_dependency |= contains_named_var(ctx, term, var_name);
        if bounded_sin_cos_term_bound_for_direct_diff_route(ctx, term).is_some() {
            has_trig_term = true;
            continue;
        }
        if term_is_tan_of_diff_var(ctx, term, var_name) {
            has_trig_term = true;
            has_tan_term = true;
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, term, 8).is_some() {
            continue;
        }
        if term_is_ln_of_diff_var(ctx, term, var_name) {
            has_ln_term = true;
            continue;
        }
        if term_is_exp_of_diff_var(ctx, term, var_name) {
            continue;
        }
        if term_is_sqrt_of_diff_var(ctx, term, var_name) {
            has_sqrt_variable_term = true;
            continue;
        }
        if term_is_reciprocal_sqrt_of_diff_var(ctx, term, var_name) {
            has_reciprocal_sqrt_variable_term = true;
            continue;
        }
        if term_is_reciprocal_of_diff_var(ctx, term, var_name) {
            has_reciprocal_variable_term = true;
            continue;
        }
        let Ok(poly) = Polynomial::from_expr(ctx, term, var_name) else {
            return false;
        };
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return false;
        }
    }

    has_trig_term
        && has_variable_dependency
        && !(has_tan_term
            && has_ln_term
            && (has_sqrt_variable_term || has_reciprocal_sqrt_variable_term))
        && !(has_tan_term && has_reciprocal_variable_term)
}

fn term_is_tan_of_diff_var(ctx: &cas_ast::Context, term: ExprId, var_name: &str) -> bool {
    let term = constant_scaled_single_factor_for_diff_target(ctx, term, var_name).unwrap_or(term);
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    let Expr::Function(fn_id, args) = ctx.get(term) else {
        return false;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Tan) {
        return false;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return false;
    };
    ctx.sym_name(*sym_id) == var_name
}

fn term_is_ln_of_diff_var(ctx: &cas_ast::Context, term: ExprId, var_name: &str) -> bool {
    let term = constant_scaled_single_factor_for_diff_target(ctx, term, var_name).unwrap_or(term);
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    let Expr::Function(fn_id, args) = ctx.get(term) else {
        return false;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Ln) {
        return false;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return false;
    };
    ctx.sym_name(*sym_id) == var_name
}

fn term_is_exp_of_diff_var(ctx: &cas_ast::Context, term: ExprId, var_name: &str) -> bool {
    let term = constant_scaled_single_factor_for_diff_target(ctx, term, var_name).unwrap_or(term);
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    match ctx.get(term) {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Exp) {
                return false;
            }
            let Expr::Variable(sym_id) = ctx.get(args[0]) else {
                return false;
            };
            ctx.sym_name(*sym_id) == var_name
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) => {
            let Expr::Variable(sym_id) = ctx.get(*exp) else {
                return false;
            };
            ctx.sym_name(*sym_id) == var_name
        }
        _ => false,
    }
}

fn term_is_sqrt_of_diff_var(ctx: &cas_ast::Context, term: ExprId, var_name: &str) -> bool {
    let term = constant_scaled_single_factor_for_diff_target(ctx, term, var_name).unwrap_or(term);
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    let Some(radicand) = extract_square_root_base(ctx, term) else {
        return false;
    };
    let Expr::Variable(sym_id) = ctx.get(radicand) else {
        return false;
    };
    ctx.sym_name(*sym_id) == var_name
}

fn term_is_reciprocal_sqrt_of_diff_var(
    ctx: &cas_ast::Context,
    term: ExprId,
    var_name: &str,
) -> bool {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term) {
        return term_is_reciprocal_sqrt_of_diff_var(ctx, *inner, var_name);
    }

    let core = if let Expr::Mul(_, _) = ctx.get(term) {
        let mut nonconstant = None;
        for factor in cas_math::expr_nary::mul_leaves(ctx, term) {
            if cas_ast::views::as_rational_const(ctx, factor, 4).is_some() {
                continue;
            }
            if nonconstant.replace(factor).is_some() {
                return false;
            }
        }
        nonconstant.unwrap_or(term)
    } else {
        term
    };
    let core = cas_ast::hold::unwrap_internal_hold(ctx, core);

    let radicand = match ctx.get(core) {
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, *num, 4).is_some() => {
            let Some(radicand) = extract_square_root_base(ctx, *den) else {
                return false;
            };
            radicand
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            *base
        }
        _ => return false,
    };
    let Expr::Variable(sym_id) = ctx.get(radicand) else {
        return false;
    };
    ctx.sym_name(*sym_id) == var_name
}

fn term_is_reciprocal_of_diff_var(ctx: &cas_ast::Context, term: ExprId, var_name: &str) -> bool {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term) {
        return term_is_reciprocal_of_diff_var(ctx, *inner, var_name);
    }

    let core = if let Expr::Mul(_, _) = ctx.get(term) {
        let mut nonconstant = None;
        for factor in cas_math::expr_nary::mul_leaves(ctx, term) {
            if cas_ast::views::as_rational_const(ctx, factor, 4).is_some() {
                continue;
            }
            if nonconstant.replace(factor).is_some() {
                return false;
            }
        }
        nonconstant.unwrap_or(term)
    } else {
        term
    };
    let core = cas_ast::hold::unwrap_internal_hold(ctx, core);

    let base = match ctx.get(core) {
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, *num, 4).is_some() => *den,
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            *base
        }
        _ => return false,
    };
    let Expr::Variable(sym_id) = ctx.get(base) else {
        return false;
    };
    ctx.sym_name(*sym_id) == var_name
}

fn bounded_sin_cos_shift_margin_for_direct_diff_route(
    ctx: &cas_ast::Context,
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

        let bound = bounded_sin_cos_term_bound_for_direct_diff_route(ctx, term)?;
        trig_bound += bound;
        has_bounded_trig = true;
    }

    if has_bounded_trig && constant_shift > trig_bound {
        Some(constant_shift - trig_bound)
    } else {
        None
    }
}

fn bounded_sin_cos_term_bound_for_direct_diff_route(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if bounded_sin_cos_unit_factor_for_direct_diff_route(ctx, expr) {
        return Some(BigRational::one());
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return bounded_sin_cos_term_bound_for_direct_diff_route(ctx, *inner);
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let mut scale = BigRational::one();
    let mut has_bounded_factor = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if bounded_sin_cos_unit_factor_for_direct_diff_route(ctx, factor) {
            has_bounded_factor = true;
        } else {
            return None;
        }
    }

    has_bounded_factor.then(|| scale.abs())
}

fn bounded_sin_cos_unit_factor_for_direct_diff_route(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                )
        }
        Expr::Pow(base, exp) if bounded_sin_cos_unit_factor_for_direct_diff_route(ctx, *base) => {
            cas_ast::views::as_rational_const(ctx, *exp, 8)
                .is_some_and(|value| value.is_integer() && value > BigRational::zero())
        }
        _ => false,
    }
}

fn diff_target_is_rational_linear_partial_fraction_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    diff_variable_name(ctx, args[1]).is_some_and(|integrate_var| integrate_var == var_name)
        && cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_partial_fraction_target(
            ctx,
            args[0],
            var_name,
        )
}

fn diff_target_is_rational_linear_positive_quadratic_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    diff_variable_name(ctx, args[1]).is_some_and(|integrate_var| integrate_var == var_name)
        && cas_math::symbolic_integration_support::integrate_symbolic_is_rational_linear_positive_quadratic_target(
            ctx,
            args[0],
            var_name,
        )
}

fn diff_target_is_reciprocal_positive_quadratic_arctan_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    if diff_variable_name(ctx, args[1]).is_none_or(|integrate_var| integrate_var != var_name) {
        return false;
    }

    let Expr::Div(num, den) = ctx.get(args[0]) else {
        return false;
    };
    if !cas_ast::views::as_rational_const(ctx, *num, 8).is_some_and(|value| value.is_one()) {
        return false;
    }

    Polynomial::from_expr(ctx, *den, var_name)
        .ok()
        .is_some_and(|denominator| strictly_positive_quadratic_for_raw_diff_target(&denominator))
        || expr_is_linear_square_plus_positive_constant_for_raw_diff_target(ctx, *den, var_name)
}

fn diff_target_is_reciprocal_trig_derivative_product_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    if diff_variable_name(ctx, args[1]).is_none_or(|integrate_var| integrate_var != var_name) {
        return false;
    }

    crate::rules::calculus::constant_scaled_reciprocal_trig_derivative_product_source(
        ctx, args[0], var_name,
    )
}

fn expr_is_linear_square_plus_positive_constant_for_raw_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return false;
    };

    (expr_is_square_of_linear_for_raw_diff_target(ctx, *left, var_name)
        && cas_ast::views::as_rational_const(ctx, *right, 8)
            .is_some_and(|value| value.is_positive()))
        || (expr_is_square_of_linear_for_raw_diff_target(ctx, *right, var_name)
            && cas_ast::views::as_rational_const(ctx, *left, 8)
                .is_some_and(|value| value.is_positive()))
}

fn expr_is_square_of_linear_for_raw_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *exp, 8)
        .is_none_or(|value| value != BigRational::from_integer(2.into()))
    {
        return false;
    }
    let Ok(poly) = Polynomial::from_expr(ctx, *base, var_name) else {
        return expr_is_symbolic_linear_for_raw_diff_target(ctx, *base, var_name);
    };
    poly.degree() == 1
}

fn expr_is_symbolic_linear_for_raw_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == var_name,
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            let left_has_var = contains_named_var(ctx, *left, var_name);
            let right_has_var = contains_named_var(ctx, *right, var_name);
            (left_has_var
                && !right_has_var
                && expr_is_symbolic_linear_for_raw_diff_target(ctx, *left, var_name))
                || (!left_has_var
                    && right_has_var
                    && expr_is_symbolic_linear_for_raw_diff_target(ctx, *right, var_name))
        }
        Expr::Mul(left, right) => {
            let left_has_var = contains_named_var(ctx, *left, var_name);
            let right_has_var = contains_named_var(ctx, *right, var_name);
            (left_has_var
                && !right_has_var
                && expr_is_symbolic_linear_for_raw_diff_target(ctx, *left, var_name))
                || (!left_has_var
                    && right_has_var
                    && expr_is_symbolic_linear_for_raw_diff_target(ctx, *right, var_name))
        }
        Expr::Neg(inner) => expr_is_symbolic_linear_for_raw_diff_target(ctx, *inner, var_name),
        Expr::Hold(inner) => expr_is_symbolic_linear_for_raw_diff_target(ctx, *inner, var_name),
        Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Function(_, _)
        | Expr::Div(_, _)
        | Expr::Pow(_, _)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_) => false,
    }
}

fn diff_target_is_positive_quadratic_power_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    if diff_variable_name(ctx, args[1]).is_none_or(|integrate_var| integrate_var != var_name) {
        return false;
    }

    cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_square_target(
        ctx, args[0], var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
        ctx, args[0], var_name,
    )
}

fn diff_target_is_quadratic_affine_ln_by_parts_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    diff_variable_name(ctx, args[1]).is_some_and(|integrate_var| integrate_var == var_name)
        && integrate_target_is_quadratic_affine_ln_by_parts_kernel(ctx, args[0], var_name)
}

fn diff_target_is_arcsin_inverse_sqrt_product_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    if diff_variable_name(ctx, args[1]).is_none_or(|integrate_var| integrate_var != var_name) {
        return false;
    }

    integrate_target_has_inverse_sqrt_product_shape(ctx, args[0], var_name)
}

fn integrate_target_has_inverse_sqrt_product_shape(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    match ctx.get(target) {
        Expr::Div(num, den) => {
            if contains_named_var(ctx, *num, var_name) {
                return false;
            }
            let mut variable_sqrt_factor_count = 0usize;
            for factor in cas_math::expr_nary::mul_leaves(ctx, *den) {
                let Some(radicand) = extract_square_root_base(ctx, factor) else {
                    continue;
                };
                if cas_math::expr_nary::mul_leaves(ctx, radicand).len() >= 2 {
                    return true;
                }
                if contains_named_var(ctx, radicand, var_name) {
                    variable_sqrt_factor_count += 1;
                }
            }
            variable_sqrt_factor_count >= 2
        }
        Expr::Pow(base, exp) => {
            cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into()))
                && cas_math::expr_nary::mul_leaves(ctx, *base).len() >= 2
        }
        _ => false,
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn preserves_raw_diff_target_for_shifted_tan_affine_sqrt_log() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("ln(1+tan(sqrt(2*x+3)))", &mut ctx).unwrap();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        assert!(diff_target_should_preserve_raw_derivative_route(
            &ctx, target, variable
        ));
    }

    #[test]
    fn preserves_raw_diff_target_for_reciprocal_trig_affine_sqrt_log() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "ln(sec(sqrt(3*x+1))+tan(sqrt(3*x+1)))",
            "ln(csc(sqrt(3*x+1))-cot(sqrt(3*x+1)))",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_target_for_positive_rational_quadratic_arctan_integral() {
        let mut ctx = cas_ast::Context::new();
        let target = cas_parser::parse("integrate(1/((a*x+b)^2+2), x)", &mut ctx).unwrap();
        let variable = ctx.var("x");

        assert!(
            diff_target_is_reciprocal_positive_quadratic_arctan_integrate_call(&ctx, target, "x")
        );
        assert!(diff_target_should_preserve_raw_derivative_route(
            &ctx, target, variable
        ));
    }

    #[test]
    fn preserves_raw_diff_target_for_separated_linear_log_abs_poles() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "(-1/2)*ln(abs(x-1)) - 4/(x-1) + (1/2)*ln(abs(x+1))",
            "1/4*ln(abs(2*x+1)) - 1/(4*(2*x-3)) - 1/4*ln(abs(2*x-3))",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_target_for_positive_quadratic_log_abs_poles() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "1/4*ln(x^2+1)-1/2*ln(abs(x-1))+1/(2*x-2)",
            "1/6*ln(2*x^2+2)-1/2*ln(abs(2*x-2))-1/(2*(2*x-2))",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn avoids_raw_diff_target_for_unrelated_separated_log_abs_sums() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "ln(abs(x-1)) - ln(abs(x+1))",
            "ln(abs(x^2+1)) + 1/(x+1) + ln(abs(x-1))",
            "ln(x^2-1) + ln(abs(x-1)) + 1/(x-1)",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                !diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_target_for_shifted_sqrt_reciprocal_trig_product() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in ["k*sec(b-sqrt(x))", "-k*csc(b-sqrt(x))", "k*sec(sqrt(x)-b)"] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_target_for_positive_shifted_bounded_trig_sqrt() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in ["sqrt(sin(2*x)+cos(x)+4)", "sqrt(cos(x)+2*sin(x)*cos(x)+4)"] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_target_for_integrated_cosecant_cotangent_product() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();
        let target =
            cas_parser::parse("integrate(k*a*csc(a*x+b)*cot(a*x+b), x)", &mut ctx).unwrap();

        assert!(diff_target_should_preserve_raw_derivative_route(
            &ctx, target, variable
        ));
    }

    #[test]
    fn preserves_raw_diff_target_for_integrated_negative_secant_tangent_product() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();
        let target =
            cas_parser::parse("integrate(-k*a*sec(b-a*x)*tan(b-a*x), x)", &mut ctx).unwrap();

        assert!(diff_target_should_preserve_raw_derivative_route(
            &ctx, target, variable
        ));
    }

    #[test]
    fn avoids_raw_diff_target_for_integrated_reciprocal_trig_product_without_derivative_factor() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "integrate(sec(x^2)*tan(x^2), x)",
            "integrate(k*sec(x^2+b)*tan(x^2+b), x)",
            "integrate(csc(x^2)*cot(x^2), x)",
            "integrate(k*csc(x^2+b)*cot(x^2+b), x)",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                !diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_target_for_additive_trig_polynomial_sqrt() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "sqrt(sin(2*x)+cos(x)+x)",
            "sqrt(cos(x)+2*sin(x)*cos(x)+x)",
            "sqrt(tan(x)+x+2)",
            "sqrt(tan(x)+exp(x)+x)",
            "sqrt(tan(x)+2*exp(x)+x)",
            "sqrt(tan(x)+ln(x)+x)",
            "sqrt(tan(x)+2*ln(x)+x)",
            "sqrt(tan(x)-ln(x)+x)",
            "sqrt(tan(x)+sqrt(x)+x)",
            "sqrt(tan(x)+1/sqrt(x)+x)",
            "sqrt(tan(x)+2/sqrt(x)+x)",
            "sqrt(sin(2*x)+cos(x)+x^3)",
            "sqrt(sin(2*x)+cos(x)+ln(x))",
            "sqrt(sin(2*x)+cos(x)+2*ln(x))",
            "sqrt(sin(2*x)+cos(x)-3*ln(x))",
            "sqrt(sin(2*x)+cos(x)+exp(x))",
            "sqrt(sin(2*x)+cos(x)+2*exp(x))",
            "sqrt(sin(2*x)+cos(x)-3*exp(x))",
            "sqrt(sin(2*x)+cos(x)+sqrt(x))",
            "sqrt(sin(2*x)+cos(x)+2*sqrt(x))",
            "sqrt(sin(2*x)+cos(x)-2*sqrt(x))",
            "sqrt(sin(2*x)+cos(x)+1/x)",
            "sqrt(sin(2*x)+cos(x)+2/x)",
            "sqrt(sin(2*x)+cos(x)-2/x)",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_call_for_subtracted_reciprocal_trig_sqrt() {
        let mut ctx = cas_ast::Context::new();
        let call = cas_parser::parse("diff(sqrt(sin(2*x)+cos(x)-2/x), x)", &mut ctx).unwrap();
        let Expr::Function(fn_id, args) = ctx.get(call) else {
            panic!("expected function");
        };
        assert!(diff_call_should_preserve_raw_target_for_direct_derivative(
            &ctx,
            ctx.sym_name(*fn_id),
            args
        ));
    }

    #[test]
    fn avoids_raw_diff_target_for_tan_ln_sqrt_mixed_denominator_sqrt() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "sqrt(tan(x)+ln(x)+sqrt(x)+x)",
            "sqrt(tan(x)+ln(x)+1/sqrt(x)+x)",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                !diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_diff_target_for_arctan_sqrt_additive_trig_polynomial() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x))",
            "arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x))",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                diff_target_should_preserve_raw_derivative_route(&ctx, target, variable),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_integrate_target_for_reciprocal_trig_affine_sqrt_log_kernel() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1)))",
            "3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1)))",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                integrate_call_should_preserve_raw_target_for_direct_integration(
                    &ctx,
                    "integrate",
                    &[target, variable],
                ),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_integrate_target_for_algorithmic_positive_quadratic_backend() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in [
            "(m*(s*x+b)+c)/((s*x+b)^2+a)",
            "(m*s*x+b*m+c)/(s^2*x^2+2*b*s*x+b^2+a)",
        ] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                integrate_call_should_preserve_raw_target_for_direct_integration(
                    &ctx,
                    "integrate",
                    &[target, variable],
                ),
                "{raw}"
            );
        }
    }

    #[test]
    fn preserves_raw_integrate_target_for_algorithmic_indefinite_square_backend() {
        let mut ctx = cas_ast::Context::new();
        let variable = cas_parser::parse("x", &mut ctx).unwrap();

        for raw in ["1/(x^2-a^2)", "(m*(s*x+b)+c)/((s*x+b)^2-a^2)"] {
            let target = cas_parser::parse(raw, &mut ctx).unwrap();
            assert!(
                integrate_call_should_preserve_raw_target_for_direct_integration(
                    &ctx,
                    "integrate",
                    &[target, variable],
                ),
                "{raw}"
            );
        }
    }
}

fn integrate_target_is_quadratic_affine_ln_by_parts_kernel(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let additive_view = cas_math::expr_nary::AddView::from_expr(ctx, target);
    if additive_view.terms.len() >= 2 {
        let mut common_log_arg = None;
        let mut cofactor_sum = Polynomial::zero(var_name.to_string());

        for (term, sign) in additive_view.terms {
            let Some((log_arg, mut cofactor_poly)) =
                quadratic_affine_ln_term_parts_for_transform(ctx, term, var_name)
            else {
                return false;
            };
            if sign == cas_math::expr_nary::Sign::Neg {
                cofactor_poly = cofactor_poly.neg();
            }

            if let Some(existing_arg) = common_log_arg {
                if cas_ast::ordering::compare_expr(ctx, existing_arg, log_arg)
                    != std::cmp::Ordering::Equal
                {
                    return false;
                }
            } else {
                common_log_arg = Some(log_arg);
            }

            cofactor_sum = cofactor_sum.add(&cofactor_poly);
        }

        let Some(log_arg) = common_log_arg else {
            return false;
        };
        return affine_ln_arg_has_positive_slope_for_transform(ctx, log_arg, var_name)
            && cofactor_sum.degree() == 2;
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return false;
    }

    for (log_index, factor) in factors.iter().copied().enumerate() {
        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            continue;
        };
        if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Ln) {
            continue;
        }

        let Ok(arg_poly) = Polynomial::from_expr(ctx, args[0], var_name) else {
            continue;
        };
        if arg_poly.degree() != 1
            || arg_poly
                .coeffs
                .get(1)
                .is_none_or(|slope| !slope.is_positive())
        {
            continue;
        }

        let mut cofactor_poly = Polynomial::new(vec![BigRational::one()], var_name.to_string());
        let mut saw_cofactor = false;
        for (idx, cofactor_factor) in factors.iter().copied().enumerate() {
            if idx == log_index {
                continue;
            }
            let Ok(factor_poly) = Polynomial::from_expr(ctx, cofactor_factor, var_name) else {
                return false;
            };
            cofactor_poly = cofactor_poly.mul(&factor_poly);
            saw_cofactor = true;
        }

        return saw_cofactor && cofactor_poly.degree() == 2;
    }

    false
}

fn quadratic_affine_ln_term_parts_for_transform(
    ctx: &cas_ast::Context,
    term: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, term);

    for (log_index, factor) in factors.iter().copied().enumerate() {
        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            continue;
        };
        if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Ln) {
            continue;
        }

        let mut cofactor_poly = Polynomial::one(var_name.to_string());
        for (idx, cofactor_factor) in factors.iter().copied().enumerate() {
            if idx == log_index {
                continue;
            }
            let Ok(factor_poly) = Polynomial::from_expr(ctx, cofactor_factor, var_name) else {
                return None;
            };
            cofactor_poly = cofactor_poly.mul(&factor_poly);
        }

        return Some((args[0], cofactor_poly));
    }

    None
}

fn affine_ln_arg_has_positive_slope_for_transform(
    ctx: &cas_ast::Context,
    arg: ExprId,
    var_name: &str,
) -> bool {
    let Ok(arg_poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return false;
    };
    arg_poly.degree() == 1
        && arg_poly
            .coeffs
            .get(1)
            .is_some_and(|slope| slope.is_positive())
}

fn diff_target_is_repeated_trig_by_parts_integrate_call(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    diff_variable_name(ctx, args[1]).is_some_and(|integrate_var| integrate_var == var_name)
        && integrate_target_is_repeated_trig_by_parts_kernel(ctx, args[0], var_name)
}

fn integrate_target_is_repeated_trig_by_parts_kernel(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(target) else {
        return false;
    };

    (expr_is_repeated_by_parts_polynomial(ctx, *left, var_name)
        && expr_is_direct_trig_linear_arg(ctx, *right, var_name))
        || (expr_is_repeated_by_parts_polynomial(ctx, *right, var_name)
            && expr_is_direct_trig_linear_arg(ctx, *left, var_name))
}

fn diff_target_is_inverse_tangent_direct_trig_linear_arg(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(
                cas_ast::BuiltinFn::Atan
                    | cas_ast::BuiltinFn::Arctan
                    | cas_ast::BuiltinFn::Acot
                    | cas_ast::BuiltinFn::Arccot
            )
        )
    {
        return false;
    }

    expr_is_direct_trig_linear_arg(ctx, args[0], var_name)
}

fn integrate_target_is_repeated_exp_by_parts_kernel(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(target) else {
        return false;
    };

    (expr_is_repeated_by_parts_polynomial(ctx, *left, var_name)
        && expr_is_direct_exp_linear_arg(ctx, *right, var_name))
        || (expr_is_repeated_by_parts_polynomial(ctx, *right, var_name)
            && expr_is_direct_exp_linear_arg(ctx, *left, var_name))
}

fn expr_is_repeated_by_parts_polynomial(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return false;
    };
    matches!(poly.degree(), 2..=4)
}

fn expr_is_direct_trig_linear_arg(ctx: &cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
        )
    {
        return false;
    }
    let Ok(poly) = Polynomial::from_expr(ctx, args[0], var_name) else {
        return false;
    };
    poly.degree() == 1
}

fn expr_is_direct_exp_linear_arg(ctx: &cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(cas_ast::BuiltinFn::Exp)) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) => {
            *exp
        }
        _ => return false,
    };
    let Ok(poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return false;
    };
    poly.degree() == 1
}

fn diff_variable_name(ctx: &cas_ast::Context, variable: ExprId) -> Option<&str> {
    let Expr::Variable(sym_id) = ctx.get(variable) else {
        return None;
    };
    Some(ctx.sym_name(*sym_id))
}

fn diff_target_is_direct_tan_or_tan_square(ctx: &cas_ast::Context, target: ExprId) -> bool {
    if expr_is_direct_tan_call(ctx, target) {
        return true;
    }
    let Expr::Pow(base, exp) = ctx.get(target) else {
        return false;
    };
    expr_is_direct_tan_call(ctx, *base)
        && matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into())
}

fn diff_target_is_affine_polynomial_times_direct_builtin_derivative_target(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(target) else {
        return false;
    };

    (expr_is_direct_supported_builtin_derivative_call(ctx, *left)
        && expr_is_affine_polynomial_in_named_variable(ctx, *right, var_name))
        || (expr_is_direct_supported_builtin_derivative_call(ctx, *right)
            && expr_is_affine_polynomial_in_named_variable(ctx, *left, var_name))
}

fn expr_is_affine_polynomial_in_named_variable(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return false;
    };
    !poly.is_zero() && poly.degree() <= 1
}

fn expr_is_direct_tan_call(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, fn_args) = ctx.get(expr) else {
        return false;
    };
    fn_args.len() == 1 && ctx.builtin_of(*fn_id) == Some(cas_ast::BuiltinFn::Tan)
}

fn expr_is_direct_supported_builtin_derivative_call(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, fn_args) = ctx.get(expr) else {
        return false;
    };
    fn_args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(
                cas_ast::BuiltinFn::Tan
                    | cas_ast::BuiltinFn::Cot
                    | cas_ast::BuiltinFn::Sec
                    | cas_ast::BuiltinFn::Csc
                    | cas_ast::BuiltinFn::Tanh
            )
        )
}

fn diff_target_is_ln_reciprocal_trig_sqrt(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(ln_fn, ln_args) = ctx.get(target) else {
        return false;
    };
    if ln_args.len() != 1 || ctx.builtin_of(*ln_fn) != Some(cas_ast::BuiltinFn::Ln) {
        return false;
    }

    let Some(sqrt_arg) = reciprocal_trig_log_sqrt_arg_for_diff_target(ctx, ln_args[0]) else {
        return false;
    };
    let Some(radicand) = extract_square_root_base(ctx, sqrt_arg) else {
        return false;
    };
    let Ok(poly) = Polynomial::from_expr(ctx, radicand, var_name) else {
        return false;
    };
    !poly.derivative().is_zero() && poly.degree() <= 1
}

fn reciprocal_trig_log_sqrt_arg_for_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) => unordered_same_arg_unary_pair_for_diff_target(
            ctx,
            *left,
            cas_ast::BuiltinFn::Sec,
            *right,
            cas_ast::BuiltinFn::Tan,
        ),
        Expr::Sub(left, right) => same_arg_unary_pair_for_diff_target(
            ctx,
            *left,
            cas_ast::BuiltinFn::Csc,
            *right,
            cas_ast::BuiltinFn::Cot,
        ),
        _ => None,
    }
}

fn unordered_same_arg_unary_pair_for_diff_target(
    ctx: &cas_ast::Context,
    left: ExprId,
    left_builtin: cas_ast::BuiltinFn,
    right: ExprId,
    right_builtin: cas_ast::BuiltinFn,
) -> Option<ExprId> {
    same_arg_unary_pair_for_diff_target(ctx, left, left_builtin, right, right_builtin).or_else(
        || same_arg_unary_pair_for_diff_target(ctx, right, left_builtin, left, right_builtin),
    )
}

fn same_arg_unary_pair_for_diff_target(
    ctx: &cas_ast::Context,
    left: ExprId,
    left_builtin: cas_ast::BuiltinFn,
    right: ExprId,
    right_builtin: cas_ast::BuiltinFn,
) -> Option<ExprId> {
    let left_arg = direct_unary_arg_for_diff_target(ctx, left, left_builtin)?;
    let right_arg = direct_unary_arg_for_diff_target(ctx, right, right_builtin)?;
    let left_base = extract_square_root_base(ctx, left_arg)?;
    let right_base = extract_square_root_base(ctx, right_arg)?;
    (cas_ast::ordering::compare_expr(ctx, left_base, right_base) == std::cmp::Ordering::Equal)
        .then_some(left_arg)
}

fn direct_unary_arg_for_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    builtin: cas_ast::BuiltinFn,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin)).then_some(args[0])
}

fn diff_target_is_ln_constant_shifted_tan_sqrt(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(ln_fn, ln_args) = ctx.get(target) else {
        return false;
    };
    if ln_args.len() != 1 || ctx.builtin_of(*ln_fn) != Some(cas_ast::BuiltinFn::Ln) {
        return false;
    }

    let Some(tan_arg) = shifted_tan_arg_for_diff_target(ctx, ln_args[0]) else {
        return false;
    };
    let Some(radicand) = extract_square_root_base(ctx, tan_arg) else {
        return false;
    };
    let Ok(poly) = Polynomial::from_expr(ctx, radicand, var_name) else {
        return false;
    };
    !poly.derivative().is_zero() && poly.degree() <= 1
}

fn shifted_tan_arg_for_diff_target(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if cas_ast::views::as_rational_const(ctx, *left, 8)
                .is_some_and(|value| !value.is_zero())
            {
                signed_tan_arg_for_diff_target(ctx, *right)
            } else if cas_ast::views::as_rational_const(ctx, *right, 8)
                .is_some_and(|value| !value.is_zero())
            {
                signed_tan_arg_for_diff_target(ctx, *left)
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            if cas_ast::views::as_rational_const(ctx, *left, 8)
                .is_some_and(|value| !value.is_zero())
            {
                direct_tan_arg_for_diff_target(ctx, *right)
            } else if cas_ast::views::as_rational_const(ctx, *right, 8)
                .is_some_and(|value| !value.is_zero())
            {
                direct_tan_arg_for_diff_target(ctx, *left)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn signed_tan_arg_for_diff_target(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = direct_tan_arg_for_diff_target(ctx, expr) {
        return Some(arg);
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        return direct_tan_arg_for_diff_target(ctx, *inner);
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
        let arg = direct_tan_arg_for_diff_target(ctx, factor)?;
        if tan_arg.replace(arg).is_some() {
            return None;
        }
    }

    (scale == -BigRational::one()).then_some(tan_arg?)
}

fn direct_tan_arg_for_diff_target(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(cas_ast::BuiltinFn::Tan)).then_some(args[0])
}

fn diff_target_is_scaled_reciprocal_trig_shifted_sqrt(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let target = match ctx.get(target) {
        Expr::Neg(inner) => *inner,
        _ => target,
    };

    let mut reciprocal_trig_factor = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        let factor = match ctx.get(factor) {
            Expr::Neg(inner) => *inner,
            _ => factor,
        };
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some() {
            continue;
        }
        if reciprocal_trig_shifted_sqrt_radicand_for_diff_target(ctx, factor, var_name).is_some() {
            if reciprocal_trig_factor.replace(factor).is_some() {
                return false;
            }
            continue;
        }
        if contains_named_var(ctx, factor, var_name) {
            return false;
        }
    }

    reciprocal_trig_factor.is_some()
}

fn reciprocal_trig_shifted_sqrt_radicand_for_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Sec | cas_ast::BuiltinFn::Csc)
        )
    {
        return None;
    }

    let radicand = shifted_sqrt_radicand_for_diff_target(ctx, args[0], var_name)?;
    let Ok(poly) = Polynomial::from_expr(ctx, radicand, var_name) else {
        return None;
    };
    (!poly.derivative().is_zero() && poly.degree() <= 1).then_some(radicand)
}

fn shifted_sqrt_radicand_for_diff_target(
    ctx: &cas_ast::Context,
    arg: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    match ctx.get(arg) {
        Expr::Add(left, right) => {
            if !contains_named_var(ctx, *left, var_name) {
                return signed_sqrt_radicand_for_diff_target(ctx, *right);
            }
            if !contains_named_var(ctx, *right, var_name) {
                return signed_sqrt_radicand_for_diff_target(ctx, *left);
            }
            None
        }
        Expr::Sub(left, right) => {
            if !contains_named_var(ctx, *left, var_name) {
                return signed_sqrt_radicand_for_diff_target(ctx, *right);
            }
            if !contains_named_var(ctx, *right, var_name) {
                return signed_sqrt_radicand_for_diff_target(ctx, *left);
            }
            None
        }
        _ => None,
    }
}

fn signed_sqrt_radicand_for_diff_target(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(radicand) = extract_square_root_base(ctx, expr) {
        return Some(radicand);
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return extract_square_root_base(ctx, *inner);
    }
    None
}

fn diff_target_is_reciprocal_positive_shifted_sqrt(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Div(num, den) = ctx.get(target) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *num, 8).is_none_or(|value| value.is_zero()) {
        return false;
    }

    let Some((radicand, shift)) = shifted_sqrt_positive_constant_parts(ctx, *den) else {
        return false;
    };
    shift.is_positive() && contains_named_var(ctx, radicand, var_name)
}

fn diff_target_is_reciprocal_sqrt_times_nonzero_shifted_sqrt(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Div(num, den) = ctx.get(target) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *num, 8).is_none_or(|value| value.is_zero()) {
        return false;
    }

    let factors: Vec<_> = cas_math::expr_nary::mul_leaves(ctx, *den)
        .into_iter()
        .collect();
    if factors.len() != 2 {
        return false;
    }

    [(factors[0], factors[1]), (factors[1], factors[0])]
        .into_iter()
        .any(|(sqrt_factor, shifted_factor)| {
            let Some(sqrt_radicand) = extract_square_root_base(ctx, sqrt_factor) else {
                return false;
            };
            let Some((shifted_radicand, shift)) =
                shifted_sqrt_positive_constant_parts(ctx, shifted_factor)
            else {
                return false;
            };
            !shift.is_zero()
                && contains_named_var(ctx, sqrt_radicand, var_name)
                && cas_math::expr_domain::exprs_equivalent(ctx, sqrt_radicand, shifted_radicand)
        })
}

fn shifted_sqrt_positive_constant_parts(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        let radicand = extract_square_root_base(ctx, *left)?;
        let shift = -cas_ast::views::as_rational_const(ctx, *right, 8)?;
        return Some((radicand, shift));
    }

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

    cas_ast::views::as_rational_const(ctx, shift_expr, 8).map(|shift| (radicand, shift))
}

fn peel_constant_divisor_for_diff_target(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> ExprId {
    match ctx.get(target) {
        Expr::Neg(inner) => peel_constant_divisor_for_diff_target(ctx, *inner, var_name),
        Expr::Div(num, den) if !contains_named_var(ctx, *den, var_name) => {
            peel_constant_divisor_for_diff_target(ctx, *num, var_name)
        }
        _ => target,
    }
}

fn unit_reciprocal_base_for_diff_target(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if matches!(ctx.get(*num), Expr::Number(n) if n.is_one()) => Some(*den),
        Expr::Pow(base, exp)
            if matches!(
                ctx.get(*exp),
                Expr::Number(n) if n.is_integer() && n.to_integer() == (-1).into()
            ) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn expr_is_arctan_reciprocal_affine_call(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Arctan | cas_ast::BuiltinFn::Atan)
        )
    {
        return false;
    }

    let Some(base) = unit_reciprocal_base_for_diff_target(ctx, args[0]) else {
        return false;
    };
    expr_is_affine_polynomial_in_named_variable(ctx, base, var_name)
}

fn diff_target_is_affine_times_arctan_reciprocal_affine(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let target = peel_constant_divisor_for_diff_target(ctx, target, var_name);
    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.is_empty() {
        return false;
    }

    for (arctan_index, factor) in factors.iter().copied().enumerate() {
        if !expr_is_arctan_reciprocal_affine_call(ctx, factor, var_name) {
            continue;
        }

        let mut cofactor_poly = Polynomial::one(var_name.to_string());
        for cofactor in factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != arctan_index).then_some(factor))
        {
            let Ok(poly) = Polynomial::from_expr(ctx, cofactor, var_name) else {
                return false;
            };
            cofactor_poly = cofactor_poly.mul(&poly);
            if cofactor_poly.degree() > 1 {
                return false;
            }
        }

        return !cofactor_poly.is_zero();
    }

    false
}

fn collect_additive_terms_for_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    terms: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            collect_additive_terms_for_diff_target(ctx, *left, terms);
            collect_additive_terms_for_diff_target(ctx, *right, terms);
        }
        Expr::Neg(inner) => collect_additive_terms_for_diff_target(ctx, *inner, terms),
        _ => terms.push(expr),
    }
}

fn constant_scaled_single_factor_for_diff_target(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = peel_constant_divisor_for_diff_target(ctx, expr, var_name);
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut nonconstant = None;

    for factor in factors {
        if cas_ast::views::as_rational_const(ctx, factor, 4).is_some() {
            continue;
        }
        if nonconstant.replace(factor).is_some() {
            return None;
        }
    }

    nonconstant
}

fn diff_target_term_is_quadratic_ln(ctx: &cas_ast::Context, term: ExprId, var_name: &str) -> bool {
    let Some(term) = constant_scaled_single_factor_for_diff_target(ctx, term, var_name) else {
        return false;
    };
    let Expr::Function(fn_id, args) = ctx.get(term) else {
        return false;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Ln) {
        return false;
    }

    let Ok(poly) = Polynomial::from_expr(ctx, args[0], var_name) else {
        return false;
    };
    !poly.is_zero() && poly.degree() <= 2
}

fn diff_target_is_arctan_reciprocal_affine_by_parts_sum(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let target = peel_constant_divisor_for_diff_target(ctx, target, var_name);
    let mut terms = Vec::new();
    collect_additive_terms_for_diff_target(ctx, target, &mut terms);
    if terms.len() < 2 {
        return false;
    }

    let has_ln_term = terms
        .iter()
        .copied()
        .any(|term| diff_target_term_is_quadratic_ln(ctx, term, var_name));
    let has_arctan_term = terms
        .iter()
        .copied()
        .any(|term| diff_target_is_affine_times_arctan_reciprocal_affine(ctx, term, var_name));

    has_ln_term && has_arctan_term
}

fn expr_is_arctan_affine_call(ctx: &cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Arctan | cas_ast::BuiltinFn::Atan)
        )
    {
        return false;
    }

    expr_is_affine_polynomial_in_named_variable(ctx, args[0], var_name)
}

fn diff_target_is_affine_times_arctan_affine(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let target = peel_constant_divisor_for_diff_target(ctx, target, var_name);
    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.is_empty() {
        return false;
    }

    for (arctan_index, factor) in factors.iter().copied().enumerate() {
        if !expr_is_arctan_affine_call(ctx, factor, var_name) {
            continue;
        }

        let mut cofactor_poly = Polynomial::one(var_name.to_string());
        for cofactor in factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != arctan_index).then_some(factor))
        {
            let Ok(poly) = Polynomial::from_expr(ctx, cofactor, var_name) else {
                return false;
            };
            cofactor_poly = cofactor_poly.mul(&poly);
            if cofactor_poly.degree() > 1 {
                return false;
            }
        }

        return !cofactor_poly.is_zero();
    }

    false
}

fn diff_target_is_arctan_affine_by_parts_sum(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let target = peel_constant_divisor_for_diff_target(ctx, target, var_name);
    let mut terms = Vec::new();
    collect_additive_terms_for_diff_target(ctx, target, &mut terms);
    if terms.len() < 2 {
        return false;
    }

    let has_ln_term = terms
        .iter()
        .copied()
        .any(|term| diff_target_term_is_quadratic_ln(ctx, term, var_name));
    let has_arctan_term = terms
        .iter()
        .copied()
        .any(|term| diff_target_is_affine_times_arctan_affine(ctx, term, var_name));

    has_ln_term && has_arctan_term
}

fn diff_target_is_inverse_reciprocal_trig_surd_scaled_quadratic(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, fn_args) = ctx.get(target) else {
        return false;
    };
    if fn_args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(
                cas_ast::BuiltinFn::Arcsec
                    | cas_ast::BuiltinFn::Asec
                    | cas_ast::BuiltinFn::Arccsc
                    | cas_ast::BuiltinFn::Acsc
            )
        )
    {
        return false;
    }

    expr_is_positive_surd_times_positive_quadratic(ctx, fn_args[0], var_name)
}

fn diff_target_is_constant_scaled_reciprocal_polynomial_power(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let target = match ctx.get(target) {
        Expr::Neg(inner) => *inner,
        _ => target,
    };

    let Expr::Div(num, den) = ctx.get(target) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *num, 4).is_none() {
        return false;
    }

    let mut matched_power = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, *den) {
        if expr_is_positive_integer_power_of_low_degree_polynomial(ctx, factor, var_name) {
            if matched_power {
                return false;
            }
            matched_power = true;
            continue;
        }

        let Some(scale) = cas_ast::views::as_rational_const(ctx, factor, 4) else {
            return false;
        };
        if scale.is_zero() {
            return false;
        }
    }

    matched_power
}

fn diff_target_is_constant_scaled_positive_polynomial_power(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let target = match ctx.get(target) {
        Expr::Neg(inner) => *inner,
        _ => target,
    };

    match ctx.get(target) {
        Expr::Mul(_, _) => {
            let mut matched_power = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
                if expr_is_positive_integer_power_of_low_degree_polynomial(ctx, factor, var_name) {
                    if matched_power {
                        return false;
                    }
                    matched_power = true;
                    continue;
                }

                let Some(scale) = cas_ast::views::as_rational_const(ctx, factor, 4) else {
                    return false;
                };
                if scale.is_zero() {
                    return false;
                }
            }
            matched_power
        }
        Expr::Div(num, den) => {
            cas_ast::views::as_rational_const(ctx, *den, 4).is_some_and(|scale| !scale.is_zero())
                && expr_is_positive_integer_power_of_low_degree_polynomial(ctx, *num, var_name)
        }
        _ => false,
    }
}

fn diff_target_is_constant_scaled_atanh_surd_polynomial(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Some(arg) = constant_scaled_atanh_arg(ctx, target, var_name) else {
        return false;
    };
    expr_is_surd_scaled_polynomial_arg(ctx, arg, var_name)
}

fn diff_target_is_scaled_bounded_inverse_trig_surd_quotient(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    if diff_target_is_bounded_inverse_trig_surd_quotient(ctx, target, var_name) {
        return true;
    }

    match ctx.get(target) {
        Expr::Neg(inner) => {
            diff_target_is_scaled_bounded_inverse_trig_surd_quotient(ctx, *inner, var_name)
        }
        Expr::Div(num, den) if diff_preserve_factor_is_constant_scale(ctx, *den, var_name) => {
            diff_target_is_scaled_bounded_inverse_trig_surd_quotient(ctx, *num, var_name)
        }
        Expr::Mul(_, _) => {
            let mut matched_target = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
                if diff_target_is_bounded_inverse_trig_surd_quotient(ctx, factor, var_name) {
                    if matched_target {
                        return false;
                    }
                    matched_target = true;
                    continue;
                }

                if !diff_preserve_factor_is_constant_scale(ctx, factor, var_name) {
                    return false;
                }
            }
            matched_target
        }
        _ => false,
    }
}

fn diff_target_is_scaled_inverse_tangent_reciprocal_sqrt_product(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    if diff_target_is_inverse_tangent_reciprocal_sqrt_product(ctx, target, var_name) {
        return true;
    }

    match ctx.get(target) {
        Expr::Neg(inner) => {
            diff_target_is_scaled_inverse_tangent_reciprocal_sqrt_product(ctx, *inner, var_name)
        }
        Expr::Div(num, den) if diff_preserve_factor_is_constant_scale(ctx, *den, var_name) => {
            diff_target_is_scaled_inverse_tangent_reciprocal_sqrt_product(ctx, *num, var_name)
        }
        Expr::Mul(_, _) => {
            let mut matched_target = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
                if diff_target_is_inverse_tangent_reciprocal_sqrt_product(ctx, factor, var_name) {
                    if matched_target {
                        return false;
                    }
                    matched_target = true;
                    continue;
                }

                if !diff_preserve_factor_is_constant_scale(ctx, factor, var_name) {
                    return false;
                }
            }
            matched_target
        }
        _ => false,
    }
}

fn diff_target_is_inverse_tangent_reciprocal_sqrt_product(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(
                cas_ast::BuiltinFn::Atan
                    | cas_ast::BuiltinFn::Arctan
                    | cas_ast::BuiltinFn::Acot
                    | cas_ast::BuiltinFn::Arccot
            )
        )
    {
        return false;
    }

    if diff_target_is_reciprocal_sqrt_times_nonzero_shifted_sqrt(ctx, args[0], var_name) {
        return true;
    }

    let Expr::Div(num, den) = ctx.get(args[0]) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *num, 8).is_none_or(|value| value.is_zero()) {
        return false;
    }

    let mut saw_sqrt = false;
    let mut saw_polynomial = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, *den) {
        if let Some(radicand) = extract_square_root_base(ctx, factor) {
            if saw_sqrt {
                return false;
            }
            let Ok(poly) = Polynomial::from_expr(ctx, radicand, var_name) else {
                return false;
            };
            if poly.is_zero() || poly.degree() > 4 {
                return false;
            }
            saw_sqrt = true;
            continue;
        }

        if !contains_named_var(ctx, factor, var_name) {
            if cas_ast::views::as_rational_const(ctx, factor, 8).is_none_or(|value| value.is_zero())
            {
                return false;
            }
            continue;
        }

        if saw_polynomial {
            return false;
        }
        let Ok(poly) = Polynomial::from_expr(ctx, factor, var_name) else {
            return false;
        };
        if poly.is_zero() || poly.degree() > 4 {
            return false;
        }
        saw_polynomial = true;
    }

    saw_sqrt && saw_polynomial
}

fn diff_target_is_constant_scaled_inverse_tangent_linear_positive_rational_radius(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Div(numerator, outer_denominator) = ctx.get(target) else {
        return false;
    };
    let Expr::Function(fn_id, args) = ctx.get(*numerator) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Atan | cas_ast::BuiltinFn::Arctan)
        )
    {
        return false;
    }

    let Some(arg_radius) = diff_arg_is_positive_rational_sqrt_times_dependent_factor_over_radius(
        ctx, args[0], var_name,
    ) else {
        return false;
    };
    diff_denominator_is_positive_rational_sqrt_times_independent_factor(
        ctx,
        *outer_denominator,
        var_name,
    )
    .is_some_and(|outer_radius| outer_radius == arg_radius)
}

fn diff_arg_is_positive_rational_sqrt_times_dependent_factor_over_radius(
    ctx: &cas_ast::Context,
    arg: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let Expr::Div(numerator, denominator) = ctx.get(arg) else {
        return None;
    };
    let radius = cas_ast::views::as_rational_const(ctx, *denominator, 8)?;
    if !radius.is_positive() {
        return None;
    }

    let mut saw_sqrt = false;
    let mut saw_dependent = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, *numerator) {
        if let Some(radicand) = extract_square_root_base(ctx, factor) {
            if saw_sqrt {
                return None;
            }
            let value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
            if value != radius {
                return None;
            }
            saw_sqrt = true;
            continue;
        }

        if contains_named_var(ctx, factor, var_name) {
            if saw_dependent {
                return None;
            }
            saw_dependent = true;
            continue;
        }

        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        if !value.is_one() {
            return None;
        }
    }

    (saw_sqrt && saw_dependent).then_some(radius)
}

fn diff_denominator_is_positive_rational_sqrt_times_independent_factor(
    ctx: &cas_ast::Context,
    denominator: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let mut sqrt_radicand = None;
    let mut saw_independent_factor = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, denominator) {
        if let Some(radicand) = extract_square_root_base(ctx, factor) {
            if sqrt_radicand.is_some() {
                return None;
            }
            let value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
            if !value.is_positive() {
                return None;
            }
            sqrt_radicand = Some(value);
            continue;
        }

        if contains_named_var(ctx, factor, var_name) {
            return None;
        }
        saw_independent_factor = true;
    }

    saw_independent_factor.then_some(sqrt_radicand?)
}

fn diff_target_is_bounded_inverse_trig_surd_quotient(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(
                cas_ast::BuiltinFn::Arcsin
                    | cas_ast::BuiltinFn::Asin
                    | cas_ast::BuiltinFn::Arccos
                    | cas_ast::BuiltinFn::Acos
            )
        )
    {
        return false;
    }

    let arg = args[0];
    if let Expr::Div(num, den) = ctx.get(arg) {
        return diff_target_is_single_low_degree_polynomial_power(ctx, *num, var_name)
            && expr_is_constant_sqrt_scale(ctx, *den);
    }

    let mut saw_power = false;
    let mut saw_surd_scale = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, arg) {
        if contains_named_var(ctx, factor, var_name) {
            if saw_power
                || !diff_target_is_single_low_degree_polynomial_power(ctx, factor, var_name)
            {
                return false;
            }
            saw_power = true;
            continue;
        }

        if !expr_is_constant_sqrt_scale(ctx, factor) {
            return false;
        }
        saw_surd_scale = true;
    }

    saw_power && saw_surd_scale
}

fn diff_target_is_bounded_inverse_trig_self_normalized_projection(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(
                cas_ast::BuiltinFn::Arcsin
                    | cas_ast::BuiltinFn::Asin
                    | cas_ast::BuiltinFn::Arccos
                    | cas_ast::BuiltinFn::Acos
            )
        )
    {
        return false;
    }

    diff_arg_is_self_normalized_projection(ctx, args[0], var_name)
}

fn diff_arg_is_self_normalized_projection(
    ctx: &cas_ast::Context,
    arg: ExprId,
    var_name: &str,
) -> bool {
    let arg = match ctx.get(arg) {
        Expr::Neg(inner) => *inner,
        _ => arg,
    };

    let Expr::Div(numerator, denominator) = ctx.get(arg) else {
        return false;
    };
    let Some(denominator_radicand) = extract_square_root_base(ctx, *denominator) else {
        return false;
    };
    let Ok(numerator_poly) = Polynomial::from_expr(ctx, *numerator, var_name) else {
        return false;
    };
    let Ok(denominator_poly) = Polynomial::from_expr(ctx, denominator_radicand, var_name) else {
        return false;
    };

    !numerator_poly.is_zero()
        && (1..=2).contains(&numerator_poly.degree())
        && denominator_poly.degree() <= 4
}

fn diff_target_is_single_low_degree_polynomial_power(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    let Some(power) = as_i64(ctx, *exp) else {
        return false;
    };
    if !(2..=8).contains(&power) {
        return false;
    }

    let Ok(poly) = Polynomial::from_expr(ctx, *base, var_name) else {
        return false;
    };
    (1..=2).contains(&poly.degree())
}

fn diff_preserve_factor_is_constant_scale(
    ctx: &cas_ast::Context,
    factor: ExprId,
    var_name: &str,
) -> bool {
    if contains_named_var(ctx, factor, var_name) {
        return false;
    }
    if expr_is_constant_sqrt_scale(ctx, factor) {
        return true;
    }

    match ctx.get(factor) {
        Expr::Div(num, den) => {
            cas_ast::views::as_rational_const(ctx, *num, 8).is_some_and(|value| !value.is_zero())
                && expr_is_constant_sqrt_scale(ctx, *den)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            expr_is_constant_sqrt_scale(ctx, *base)
        }
        _ => false,
    }
}

fn constant_scaled_atanh_arg(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.builtin_of(*fn_id) == Some(cas_ast::BuiltinFn::Atanh) && args.len() == 1 {
            return Some(args[0]);
        }
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => constant_scaled_atanh_arg(ctx, *inner, var_name),
        Expr::Div(num, den) => {
            if contains_named_var(ctx, *den, var_name) || !expr_is_constant_sqrt_scale(ctx, *den) {
                return None;
            }
            constant_scaled_atanh_arg(ctx, *num, var_name)
        }
        Expr::Mul(_, _) => {
            let mut atanh_arg = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Expr::Function(fn_id, args) = ctx.get(factor) {
                    if ctx.builtin_of(*fn_id) == Some(cas_ast::BuiltinFn::Atanh) && args.len() == 1
                    {
                        if atanh_arg.replace(args[0]).is_some() {
                            return None;
                        }
                        continue;
                    }
                }

                if contains_named_var(ctx, factor, var_name)
                    || !expr_is_constant_sqrt_scale(ctx, factor)
                {
                    return None;
                }
            }
            atanh_arg
        }
        _ => None,
    }
}

fn expr_is_surd_scaled_polynomial_arg(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if contains_named_var(ctx, *den, var_name) || !expr_is_constant_sqrt_scale(ctx, *den) {
            return false;
        }
        return expr_has_single_polynomial_factor(ctx, *num, var_name);
    }

    let mut saw_polynomial = false;
    let mut saw_nontrivial_scale = false;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if !contains_named_var(ctx, factor, var_name) {
            if !expr_is_constant_sqrt_scale(ctx, factor) {
                return false;
            }
            saw_nontrivial_scale = true;
            continue;
        }

        if saw_polynomial {
            return false;
        }
        let Ok(poly) = Polynomial::from_expr(ctx, factor, var_name) else {
            return false;
        };
        if poly.is_zero() {
            return false;
        }
        saw_polynomial = true;
    }

    saw_polynomial && saw_nontrivial_scale
}

fn expr_has_single_polynomial_factor(ctx: &cas_ast::Context, expr: ExprId, var_name: &str) -> bool {
    let mut saw_polynomial = false;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if !contains_named_var(ctx, factor, var_name) {
            if !expr_is_constant_sqrt_scale(ctx, factor) {
                return false;
            }
            continue;
        }

        if saw_polynomial {
            return false;
        }
        let Ok(poly) = Polynomial::from_expr(ctx, factor, var_name) else {
            return false;
        };
        if poly.is_zero() {
            return false;
        }
        saw_polynomial = true;
    }

    saw_polynomial
}

fn expr_is_constant_sqrt_scale(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    if cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| !value.is_zero()) {
        return true;
    }

    expr_is_positive_rational_sqrt(ctx, expr)
}

fn expr_is_positive_integer_power_of_low_degree_polynomial(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    let Some(power) = as_i64(ctx, *exp) else {
        return false;
    };
    if !(2..=8).contains(&power) {
        return false;
    }

    let Ok(poly) = Polynomial::from_expr(ctx, *base, var_name) else {
        return false;
    };
    (1..=2).contains(&poly.degree())
}

fn integrate_target_is_negative_denominator_polynomial_power_substitution(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_bounded_negative_syntactic_denominator_power_substitution_target(
        ctx,
        target,
        var_name,
        8,
    )
}

fn integrate_target_is_reciprocal_quotient_denominator_polynomial_power_substitution(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_bounded_reciprocal_quotient_denominator_power_substitution_target(
        ctx,
        target,
        var_name,
        8,
    )
}

fn integrate_target_is_arctan_sqrt_var_reciprocal(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx, target, var_name,
    )
}

fn integrate_target_is_inverse_hyperbolic_sqrt_reciprocal(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
        ctx, target, var_name,
    )
}

fn integrate_target_is_sqrt_trig_reciprocal_derivative(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let mut probe_ctx = ctx.clone();
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_reciprocal_derivative_target(
        &mut probe_ctx,
        target,
        var_name,
    )
}

fn integrate_target_is_sqrt_trig_log_derivative(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let mut probe_ctx = ctx.clone();
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_log_derivative_target(
        &mut probe_ctx,
        target,
        var_name,
    )
}

fn integrate_target_is_sqrt_reciprocal_trig_log_derivative(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let mut probe_ctx = ctx.clone();
    cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_reciprocal_trig_log_derivative_target(
        &mut probe_ctx,
        target,
        var_name,
    )
}

fn integrate_target_is_inverse_trig_polynomial_substitution(
    ctx: &cas_ast::Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let mut probe_ctx = ctx.clone();
    cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_polynomial_substitution_target(
        &mut probe_ctx,
        target,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_polynomial_substitution_target(
        &mut probe_ctx,
        target,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_nested_inverse_polynomial_substitution_target(
        &mut probe_ctx,
        target,
        var_name,
    )
}

fn expr_is_positive_surd_times_positive_quadratic(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return false;
    };

    (expr_is_positive_rational_sqrt(ctx, *left)
        && expr_is_strictly_positive_quadratic_in_named_variable(ctx, *right, var_name))
        || (expr_is_positive_rational_sqrt(ctx, *right)
            && expr_is_strictly_positive_quadratic_in_named_variable(ctx, *left, var_name))
}

fn expr_is_positive_rational_sqrt(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let Some(radicand) = extract_square_root_base(ctx, expr) else {
        return false;
    };
    cas_ast::views::as_rational_const(ctx, radicand, 8).is_some_and(|value| value.is_positive())
}

fn expr_is_strictly_positive_quadratic_in_named_variable(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
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

fn format_difference_of_squares_product_desc(
    kind: DifferenceOfSquaresProductRewriteKind,
) -> &'static str {
    match kind {
        DifferenceOfSquaresProductRewriteKind::Basic => "(a-b)(a+b) = a² - b²",
        DifferenceOfSquaresProductRewriteKind::NaryConjugateProduct => {
            "(U+V)(U-V) = U² - V² (conjugate product)"
        }
        DifferenceOfSquaresProductRewriteKind::NaryScan => "(a-b)(a+b)·… = (a²-b²)·… (n-ary scan)",
    }
}

fn format_sqrt_square_pow_plan(kind: SqrtSquarePowRewriteKind) -> (&'static str, &'static str) {
    match kind {
        SqrtSquarePowRewriteKind::PowSquare => {
            ("sqrt(u^2) = |u|", "Simplify Square Root of Square")
        }
        SqrtSquarePowRewriteKind::RepeatedMul => {
            ("sqrt(u * u) = |u|", "Simplify Square Root of Product")
        }
    }
}

fn polynomial_identity_preorder_desc(
    kind: crate::polynomial_identity_support::PolynomialIdentityProofKind,
) -> &'static str {
    match kind {
        crate::polynomial_identity_support::PolynomialIdentityProofKind::Direct => {
            "Polynomial identity: normalize and cancel to 0"
        }
        crate::polynomial_identity_support::PolynomialIdentityProofKind::OpaqueSubstitution => {
            "Polynomial identity (opaque substitution): cancel to 0"
        }
        crate::polynomial_identity_support::PolynomialIdentityProofKind::OpaqueRootRelation => {
            "Polynomial identity (opaque root relation): cancel to 0"
        }
    }
}

fn is_symbolic_atom(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Variable(_) | Expr::Constant(_))
}

fn is_plain_symbolic_binomial(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            is_symbolic_atom(ctx, *left) && is_symbolic_atom(ctx, *right)
        }
        Expr::Neg(inner) => is_plain_symbolic_binomial(ctx, *inner),
        _ => false,
    }
}

fn is_same_symbolic_atom_fraction(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    is_symbolic_atom(ctx, left)
        && is_symbolic_atom(ctx, right)
        && cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn is_symbolic_power_over_same_atom_noop(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(left) else {
        return false;
    };

    is_symbolic_atom(ctx, *base)
        && is_symbolic_atom(ctx, *exp)
        && is_symbolic_atom(ctx, right)
        && cas_ast::ordering::compare_expr(ctx, *base, right) == std::cmp::Ordering::Equal
}

fn collect_positive_add_terms_3(
    ctx: &Context,
    expr: ExprId,
    out: &mut SmallVec<[ExprId; 3]>,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_positive_add_terms_3(ctx, *left, out)
                && collect_positive_add_terms_3(ctx, *right, out)
        }
        Expr::Sub(_, _) | Expr::Neg(_) => false,
        _ => {
            if out.len() == 3 {
                return false;
            }
            out.push(expr);
            true
        }
    }
}

fn collect_signed_add_terms_3(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    out: &mut SmallVec<[(ExprId, bool); 3]>,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_signed_add_terms_3(ctx, *left, positive, out)
                && collect_signed_add_terms_3(ctx, *right, positive, out)
        }
        Expr::Sub(left, right) => {
            collect_signed_add_terms_3(ctx, *left, positive, out)
                && collect_signed_add_terms_3(ctx, *right, !positive, out)
        }
        Expr::Neg(inner) => collect_signed_add_terms_3(ctx, *inner, !positive, out),
        _ => {
            if out.len() == 3 {
                return false;
            }
            out.push((expr, positive));
            true
        }
    }
}

fn collect_mul_factors_3(ctx: &Context, expr: ExprId, out: &mut SmallVec<[ExprId; 3]>) -> bool {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_3(ctx, *left, out) && collect_mul_factors_3(ctx, *right, out)
        }
        _ => {
            if out.len() == 3 {
                return false;
            }
            out.push(expr);
            true
        }
    }
}

fn multiset_matches_exact(ctx: &Context, actual: &[ExprId], expected: &[ExprId]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }

    let mut used = [false; 3];
    for wanted in expected {
        let mut matched = false;
        for (idx, candidate) in actual.iter().enumerate() {
            if used[idx] {
                continue;
            }
            if *candidate == *wanted
                || cas_ast::ordering::compare_expr(ctx, *candidate, *wanted)
                    == std::cmp::Ordering::Equal
            {
                used[idx] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

fn is_exact_two_ab_product(ctx: &mut Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    let mut factors = SmallVec::<[ExprId; 3]>::new();
    if !collect_mul_factors_3(ctx, expr, &mut factors) || factors.len() != 3 {
        return false;
    }

    let two = ctx.num(2);
    multiset_matches_exact(ctx, &factors, &[two, a, b])
}

fn is_exact_binomial_square_fraction_preorder(ctx: &mut Context, num: ExprId, den: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(den) else {
        return false;
    };
    if as_i64(ctx, *exp) != Some(2) {
        return false;
    }
    let Expr::Add(a, b) = ctx.get(*base) else {
        return false;
    };
    let a = *a;
    let b = *b;

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let mut terms = SmallVec::<[ExprId; 3]>::new();
    if !collect_positive_add_terms_3(ctx, num, &mut terms) || terms.len() != 3 {
        return false;
    }

    let mut squares = SmallVec::<[ExprId; 2]>::new();
    let mut middle = None;
    for term in terms {
        if term == a_sq
            || term == b_sq
            || cas_ast::ordering::compare_expr(ctx, term, a_sq) == std::cmp::Ordering::Equal
            || cas_ast::ordering::compare_expr(ctx, term, b_sq) == std::cmp::Ordering::Equal
        {
            squares.push(term);
        } else if middle.is_none() {
            middle = Some(term);
        } else {
            return false;
        }
    }

    squares.len() == 2
        && multiset_matches_exact(ctx, &squares, &[a_sq, b_sq])
        && middle.is_some_and(|term| is_exact_two_ab_product(ctx, term, a, b))
}

fn try_exact_perfect_square_minus_fraction_preorder(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(a, b) = ctx.get(den) else {
        return None;
    };
    let a = *a;
    let b = *b;

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let mut terms = SmallVec::<[(ExprId, bool); 3]>::new();
    if !collect_signed_add_terms_3(ctx, num, true, &mut terms) || terms.len() != 3 {
        return None;
    }

    let mut positives = SmallVec::<[ExprId; 2]>::new();
    let mut negative = None;
    for (term, positive) in terms {
        if positive {
            if positives.len() == 2 {
                return None;
            }
            positives.push(term);
        } else if negative.is_none() {
            negative = Some(term);
        } else {
            return None;
        }
    }

    if positives.len() != 2
        || !multiset_matches_exact(ctx, &positives, &[a_sq, b_sq])
        || !negative.is_some_and(|term| is_exact_two_ab_product(ctx, term, a, b))
    {
        return None;
    }

    Some(den)
}

impl<'a> LocalSimplificationTransformer<'a> {
    /// Transform binary expression (Add/Sub/Mul) by simplifying children.
    /// Extracted to reduce stack frame size in transform_expr_recursive.
    #[inline(never)]
    pub(super) fn transform_binary(
        &mut self,
        id: ExprId,
        l: ExprId,
        r: ExprId,
        op: BinaryOp,
    ) -> ExprId {
        // PRE-ORDER: For Add/Sub, try exact polynomial-identity closure before
        // children are simplified. The bottom-up pipeline can otherwise
        // normalize opaque rational/root atoms into bulky residual fractions
        // that the same proof helper would have closed immediately.
        if matches!(op, BinaryOp::Add | BinaryOp::Sub)
            && matches!(
                self.current_phase,
                crate::phase::SimplifyPhase::Core
                    | crate::phase::SimplifyPhase::Transform
                    | crate::phase::SimplifyPhase::PostCleanup
            )
            && !self.initial_parent_ctx.is_solve_context()
        {
            if let Some(plan) =
                crate::polynomial_identity_support::try_prove_polynomial_identity_zero_expr(
                    self.context,
                    id,
                )
            {
                let zero = self.context.num(0);
                self.record_step(
                    polynomial_identity_preorder_desc(plan.kind),
                    "Polynomial Identity",
                    id,
                    zero,
                );
                return zero;
            }
        }
        // PRE-ORDER: Exact fraction-pair cancellation must run before child
        // simplification can expand denominator products and destroy the compact
        // pair shape produced by calculus rules.
        if matches!(op, BinaryOp::Add | BinaryOp::Sub)
            && !self.initial_parent_ctx.is_solve_context()
        {
            let parent_ctx = self.build_parent_context();
            if matches!(op, BinaryOp::Add) {
                let rule = crate::rules::algebra::fractions::CancelOppositeFractionsRule;
                if let Some(rewrite) = rule.apply(self.context, id, &parent_ctx) {
                    let result = self.record_rewrite_step(&rule, id, rewrite);
                    return result;
                }
            } else {
                let rule = crate::rules::algebra::fractions::CancelEqualFractionsDifferenceRule;
                if let Some(rewrite) = rule.apply(self.context, id, &parent_ctx) {
                    let result = self.record_rewrite_step(&rule, id, rewrite);
                    return result;
                }
            }
        }
        // PRE-ORDER: For Mul, detect conjugate pairs in the factor chain BEFORE
        // child simplification. This prevents canonicalization (sqrt→Pow) from
        // breaking structural matching, and prevents DistributeRule from splitting
        // the conjugate pair across inner Mul nodes after factor reordering.
        // Pattern: (a+b)*(a-b)*... → (a²-b²)*...
        if matches!(op, BinaryOp::Mul) {
            if let Some(result) = self.try_conjugate_pair_contraction(id) {
                return result;
            }
        }

        if matches!(op, BinaryOp::Sub) {
            if let Some(zero) =
                crate::calculus_residual_support::try_diff_hyperbolic_residual_zero_preorder(
                    self.context,
                    l,
                    r,
                )
                .or_else(|| {
                    crate::calculus_residual_support::try_diff_hyperbolic_residual_zero_preorder(
                        self.context,
                        r,
                        l,
                    )
                })
            {
                self.record_step(
                    "Cancel matching hyperbolic derivative residual",
                    "Hyperbolic Diff Residual",
                    id,
                    zero,
                );
                return zero;
            }

            if let Some(rewrite) =
                crate::rules::arithmetic::try_build_direct_trig_power_reduction_equivalence_rewrite(
                    self.context,
                    l,
                    r,
                )
            {
                self.record_step(
                    "Power Reduction Identity",
                    "Power Reduction Identity",
                    id,
                    rewrite.new_expr,
                );
                return rewrite.new_expr;
            }
        }

        if matches!(op, BinaryOp::Add | BinaryOp::Sub) && !self.collect_steps_enabled() {
            if let Some(zero) =
                crate::fraction_residual_support::try_polynomial_denominator_fraction_residual_zero(
                    self.context,
                    id,
                )
            {
                self.record_step(
                    "Cancel fraction residual with equivalent polynomial denominators",
                    "Polynomial-Denominator Fraction Residual",
                    id,
                    zero,
                );
                return zero;
            }
        }

        // PRE-ORDER: Collapse small exact-zero additive combinations before
        // simplifying children. This prevents recursive trig power rewrite loops
        // such as 8*sin(x)^4 - (3 - 4*cos(2*x) + cos(4*x)).
        if matches!(op, BinaryOp::Add | BinaryOp::Sub)
            && crate::rules::arithmetic::maybe_direct_small_zero_additive_combination_candidate(
                self.context,
                id,
            )
        {
            if let Some(rewrite) =
                crate::rules::arithmetic::try_build_direct_small_zero_additive_combination_rewrite(
                    self.context,
                    id,
                )
            {
                if rewrite.description == "Power Reduction Identity"
                    || (rewrite.required_conditions.is_empty()
                        && rewrite.assumption_events.is_empty()
                        && matches!(
                            self.context.get(rewrite.new_expr),
                            cas_ast::Expr::Number(ref n) if n.is_zero()
                        ))
                {
                    let step_name = if rewrite.description == "Power Reduction Identity" {
                        "Power Reduction Identity"
                    } else {
                        "Collapse Exact Zero Additive Subexpression"
                    };
                    self.record_step(step_name, step_name, id, rewrite.new_expr);
                    return rewrite.new_expr;
                }
            }
        }

        let new_l = self.transform_child_at(id, crate::step::PathStep::Left, l);
        if matches!(op, BinaryOp::Add | BinaryOp::Sub) && new_l != l {
            let partially_rebuilt = match op {
                BinaryOp::Add => self.context.add(Expr::Add(new_l, r)),
                BinaryOp::Sub => self.context.add(Expr::Sub(new_l, r)),
                _ => unreachable!(),
            };
            let parent_ctx = self.build_parent_context();
            if matches!(op, BinaryOp::Add) {
                let rule = crate::rules::algebra::fractions::CancelOppositeFractionsRule;
                if let Some(rewrite) = rule.apply(self.context, partially_rebuilt, &parent_ctx) {
                    let result = self.record_rewrite_step(&rule, partially_rebuilt, rewrite);
                    return result;
                }
            } else {
                let rule = crate::rules::algebra::fractions::CancelEqualFractionsDifferenceRule;
                if let Some(rewrite) = rule.apply(self.context, partially_rebuilt, &parent_ctx) {
                    let result = self.record_rewrite_step(&rule, partially_rebuilt, rewrite);
                    return result;
                }
            }

            let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
            if let Some(rewrite) = rule.apply(self.context, partially_rebuilt, &parent_ctx) {
                let result = self.record_rewrite_step(&rule, partially_rebuilt, rewrite);
                return result;
            }
        }
        let new_r = self.transform_child_at(id, crate::step::PathStep::Right, r);

        let rebuilt = if new_l != l || new_r != r {
            let expr = match op {
                BinaryOp::Add => Expr::Add(new_l, new_r),
                BinaryOp::Sub => Expr::Sub(new_l, new_r),
                BinaryOp::Mul => Expr::Mul(new_l, new_r),
                BinaryOp::Div => Expr::Div(new_l, new_r),
            };
            self.context.add(expr)
        } else {
            id
        };

        if matches!(op, BinaryOp::Add | BinaryOp::Sub)
            && rebuilt != id
            && !self.collect_steps_enabled()
        {
            if let Some(zero) =
                crate::fraction_residual_support::try_polynomial_denominator_fraction_residual_zero(
                    self.context,
                    rebuilt,
                )
            {
                self.record_step(
                    "Cancel fraction residual with equivalent polynomial denominators",
                    "Polynomial-Denominator Fraction Residual",
                    rebuilt,
                    zero,
                );
                return zero;
            }
        }
        if matches!(op, BinaryOp::Add | BinaryOp::Sub) && rebuilt != id {
            let parent_ctx = self.build_parent_context();
            if matches!(op, BinaryOp::Add) {
                let rule = crate::rules::algebra::fractions::CancelOppositeFractionsRule;
                if let Some(rewrite) = rule.apply(self.context, rebuilt, &parent_ctx) {
                    let result = self.record_rewrite_step(&rule, rebuilt, rewrite);
                    return result;
                }
            } else {
                let rule = crate::rules::algebra::fractions::CancelEqualFractionsDifferenceRule;
                if let Some(rewrite) = rule.apply(self.context, rebuilt, &parent_ctx) {
                    let result = self.record_rewrite_step(&rule, rebuilt, rewrite);
                    return result;
                }
            }

            if let Some(rewrite) =
                crate::rules::arithmetic::try_build_direct_small_zero_additive_combination_rewrite(
                    self.context,
                    rebuilt,
                )
            {
                if rewrite.required_conditions.is_empty() && rewrite.assumption_events.is_empty() {
                    self.record_step(
                        "Collapse Exact Zero Additive Subexpression",
                        "Collapse Exact Zero Additive Subexpression",
                        rebuilt,
                        rewrite.new_expr,
                    );
                    return rewrite.new_expr;
                }
            }
        }

        rebuilt
    }

    /// PRE-ORDER: Flatten a Mul chain and detect conjugate factor pairs.
    ///
    /// If found, contracts (a+b)*(a-b) → (a²-b²), rebuilds the product with
    /// remaining factors, records a step, and re-enters simplification.
    /// Returns None if no conjugate pair is found.
    #[inline(never)]
    fn try_conjugate_pair_contraction(&mut self, id: ExprId) -> Option<ExprId> {
        let rewrite = try_rewrite_difference_of_squares_product_expr(self.context, id)?;
        self.record_step(
            format_difference_of_squares_product_desc(rewrite.kind),
            "Difference of Squares",
            id,
            rewrite.rewritten,
        );
        Some(self.transform_expr_recursive(rewrite.rewritten))
    }

    /// Transform Pow expression with early detection for sqrt-of-square patterns.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_pow(&mut self, id: ExprId, base: ExprId, exp: ExprId) -> ExprId {
        let allow_hidden_solve_pow_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict();
        if allow_hidden_solve_pow_preorder {
            if let Some(cas_math::power_identity_support::PowerIdentityPolicyPattern::PowZero {
                base,
                base_is_literal_zero: false,
            }) = cas_math::power_identity_support::classify_power_identity_policy_pattern(
                self.context,
                id,
            ) {
                if is_symbolic_atom(self.context, base) {
                    return self.context.num(1);
                }
            }

            if let Some(rewrite) = try_rewrite_exponential_log_inverse_expr(self.context, id) {
                if is_symbolic_atom(self.context, rewrite.rewritten) {
                    return rewrite.rewritten;
                }
            }
        }

        // EARLY DETECTION: sqrt-of-square pattern (u^2)^(1/2) -> |u|
        // Must check BEFORE recursing into children to prevent binomial expansion
        if let Some(plan) = try_plan_sqrt_square_pow_rewrite(self.context, base, exp) {
            let (identity_desc, rule_name) = format_sqrt_square_pow_plan(plan.kind);
            self.record_step(identity_desc, rule_name, id, plan.rewritten);
            return self.transform_expr_recursive(plan.rewritten);
        }

        // Check if this Pow is canonical before recursing into children
        if crate::canonical_forms::is_canonical_form(self.context, id) {
            debug!(
                "Skipping simplification of canonical Pow: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        let new_b = self.transform_child_at(id, crate::step::PathStep::Base, base);
        let new_e = self.transform_child_at(id, crate::step::PathStep::Exponent, exp);

        if new_b != base || new_e != exp {
            self.context.add(Expr::Pow(new_b, new_e))
        } else {
            id
        }
    }

    /// Transform Function expression by simplifying children.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_function(
        &mut self,
        id: ExprId,
        fn_id: SymbolId,
        args: Vec<ExprId>,
    ) -> ExprId {
        let name = self.context.sym_name(fn_id);
        // Check if this function is canonical before recursing into children
        if (name == "sqrt" || name == "abs")
            && crate::canonical_forms::is_canonical_form(self.context, id)
        {
            debug!(
                "Skipping simplification of canonical Function: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // HoldAll semantics: do NOT simplify arguments for these functions
        if is_hold_all_function(name) {
            debug!(
                "HoldAll function, skipping child simplification: {:?}",
                self.context.get(id)
            );
            return id;
        }

        if diff_call_should_preserve_raw_target_for_direct_derivative(self.context, name, &args) {
            let new_var = self.transform_child_at(id, crate::step::PathStep::Arg(1), args[1]);
            if new_var != args[1] {
                return self
                    .context
                    .add(Expr::Function(fn_id, vec![args[0], new_var]));
            }
            return id;
        }

        if integrate_call_should_preserve_raw_target_for_direct_integration(
            self.context,
            name,
            &args,
        ) {
            if args.len() == 2 {
                let new_var = self.transform_child_at(id, crate::step::PathStep::Arg(1), args[1]);
                if new_var != args[1] {
                    return self
                        .context
                        .add(Expr::Function(fn_id, vec![args[0], new_var]));
                }
            }
            return id;
        }

        // Simplify children
        let mut new_args = Vec::with_capacity(args.len());
        let mut changed = false;
        for (i, arg) in args.iter().enumerate() {
            let new_arg = self.transform_child_at(id, crate::step::PathStep::Arg(i), *arg);

            if new_arg != *arg {
                changed = true;
            }
            new_args.push(new_arg);
        }

        if changed {
            self.context.add(Expr::Function(fn_id, new_args))
        } else {
            id
        }
    }

    /// Transform Div expression with early detection for difference-of-squares pattern.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_div(&mut self, id: ExprId, l: ExprId, r: ExprId) -> ExprId {
        let allow_identical_atom_fraction_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict()
            && is_same_symbolic_atom_fraction(self.context, l, r);
        if allow_identical_atom_fraction_preorder {
            return self.context.num(1);
        }

        let allow_same_atom_power_noop_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict()
            && is_symbolic_power_over_same_atom_noop(self.context, l, r);
        if allow_same_atom_power_noop_preorder {
            return id;
        }

        let allow_scalar_multiple_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && match self.initial_parent_ctx.simplify_purpose() {
                crate::SimplifyPurpose::Eval => {
                    self.initial_parent_ctx.context_mode() == crate::options::ContextMode::Solve
                }
                crate::SimplifyPurpose::SolvePrepass => {
                    cas_solver_core::solve_safety_policy::safe_for_prepass(
                        crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                    )
                }
                crate::SimplifyPurpose::SolveTactic => {
                    let domain_mode = self.initial_parent_ctx.domain_mode();
                    cas_solver_core::solve_safety_policy::safe_for_tactic_with_domain_flags(
                        crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                        matches!(domain_mode, crate::DomainMode::Assume),
                        matches!(domain_mode, crate::DomainMode::Strict),
                    )
                }
            };
        if allow_scalar_multiple_preorder {
            if let Some(early_result) =
                crate::rules::algebra::try_structural_scalar_multiple_preorder(
                    self.context,
                    l,
                    r,
                    self.initial_parent_ctx.domain_mode(),
                    self.initial_parent_ctx.value_domain(),
                )
            {
                return early_result;
            }
        }

        let allow_exact_binomial_square_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict();
        if allow_exact_binomial_square_preorder
            && is_exact_binomial_square_fraction_preorder(self.context, l, r)
        {
            return self.context.num(1);
        }

        // EARLY DETECTION: (A² - B²) / (A ± B) pattern
        let allow_difference_of_squares_preorder = match self.initial_parent_ctx.simplify_purpose()
        {
            crate::SimplifyPurpose::Eval => true,
            crate::SimplifyPurpose::SolvePrepass => false,
            crate::SimplifyPurpose::SolveTactic => {
                let domain_mode = self.initial_parent_ctx.domain_mode();
                cas_solver_core::solve_safety_policy::safe_for_tactic_with_domain_flags(
                    crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                    matches!(domain_mode, crate::DomainMode::Assume),
                    matches!(domain_mode, crate::DomainMode::Strict),
                )
            }
        };
        if allow_difference_of_squares_preorder
            && self.event_listener.is_none()
            && !self.initial_parent_ctx.domain_mode().is_strict()
        {
            let step_start = self.steps.len();
            if let Some(early_result) =
                crate::rules::algebra::try_exact_common_factor_mul_fraction_preorder(
                    self.context,
                    id,
                    l,
                    r,
                    self.initial_parent_ctx.domain_mode(),
                    self.initial_parent_ctx.value_domain(),
                    self.collect_steps_enabled(),
                    &mut self.steps,
                    &self.current_path,
                )
            {
                self.rebuild_recent_preorder_steps_at_current_path(step_start);
                if !self.collect_steps_enabled()
                    && self.current_phase == crate::SimplifyPhase::Core
                    && self.initial_parent_ctx.is_solve_context()
                {
                    return early_result;
                }
                return self.transform_expr_recursive(early_result);
            }
        }
        if allow_difference_of_squares_preorder {
            let step_start = self.steps.len();
            if let Some(early_result) = crate::rules::algebra::try_difference_of_squares_preorder(
                self.context,
                id,
                l,
                r,
                self.initial_parent_ctx.domain_mode(),
                self.initial_parent_ctx.value_domain(),
                self.initial_parent_ctx.value_domain() == crate::semantics::ValueDomain::RealOnly,
                self.collect_steps_enabled(),
                &mut self.steps,
                &self.current_path,
            ) {
                self.rebuild_recent_preorder_steps_at_current_path(step_start);
                if !self.collect_steps_enabled()
                    && self.event_listener.is_none()
                    && self.current_phase == crate::SimplifyPhase::Core
                    && self.initial_parent_ctx.is_solve_context()
                    && is_plain_symbolic_binomial(self.context, early_result)
                {
                    return early_result;
                }
                // Note: don't decrement depth here - transform_expr_recursive manages it
                return self.transform_expr_recursive(early_result);
            }
        }

        // Similar pre-order fast path for perfect-square-minus fractions, but only
        // when no listener is attached so we don't widen the existing event-gap
        // behavior beyond the hidden hot path.
        if allow_difference_of_squares_preorder && self.event_listener.is_none() {
            if !self.collect_steps_enabled()
                && self.current_phase == crate::SimplifyPhase::Core
                && self.initial_parent_ctx.is_solve_context()
                && !self.initial_parent_ctx.domain_mode().is_strict()
            {
                if let Some(early_result) =
                    try_exact_perfect_square_minus_fraction_preorder(self.context, l, r)
                {
                    return early_result;
                }
            }

            let step_start = self.steps.len();
            if let Some(early_result) = crate::rules::algebra::try_perfect_square_minus_preorder(
                self.context,
                id,
                l,
                r,
                self.collect_steps_enabled(),
                &mut self.steps,
                &self.current_path,
            ) {
                self.rebuild_recent_preorder_steps_at_current_path(step_start);
                if !self.collect_steps_enabled()
                    && self.current_phase == crate::SimplifyPhase::Core
                    && self.initial_parent_ctx.is_solve_context()
                    && is_plain_symbolic_binomial(self.context, early_result)
                {
                    return early_result;
                }
                return self.transform_expr_recursive(early_result);
            }
        }

        // Exact-shape hidden fast path for `(a^3 ± b^3)/(a±b)`. This avoids the
        // child-recursion sign/canonicalization churn on the raw hotspot inputs
        // without paying the broader planner cost that regressed earlier.
        if allow_difference_of_squares_preorder
            && !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
        {
            if let Some(early_result) =
                crate::rules::algebra::try_exact_sum_diff_of_cubes_preorder(self.context, l, r)
            {
                return early_result;
            }
        }

        // Simplify children
        let new_l = self.transform_child_at(id, crate::step::PathStep::Left, l);
        let numerator_is_literal_zero =
            matches!(self.context.get(l), Expr::Number(n) if n.is_zero());
        let new_r =
            if numerator_is_literal_zero && !self.initial_parent_ctx.domain_mode().is_strict() {
                let previous = self.suppress_depth_overflow_warnings;
                self.suppress_depth_overflow_warnings = true;
                let transformed = self.transform_child_at(id, crate::step::PathStep::Right, r);
                self.suppress_depth_overflow_warnings = previous;
                transformed
            } else {
                self.transform_child_at(id, crate::step::PathStep::Right, r)
            };

        if new_l != l || new_r != r {
            self.context.add(Expr::Div(new_l, new_r))
        } else {
            id
        }
    }
}
