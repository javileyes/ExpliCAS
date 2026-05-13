//! Expression transform helpers: binary, pow, div, and function simplification.
//!
//! These methods are extracted with `#[inline(never)]` to reduce the stack frame
//! size of `transform_expr_recursive`.

use super::*;
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
        || integrate_target_is_repeated_trig_by_parts_kernel(ctx, args[0], var_name)
        || integrate_target_is_repeated_exp_by_parts_kernel(ctx, args[0], var_name)
        || (crate::rule::steps_enabled()
            && integrate_target_is_affine_times_basic_by_parts_kernel(ctx, args[0], var_name))
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
        || diff_target_is_inverse_reciprocal_trig_surd_scaled_quadratic(ctx, target, var_name)
        || diff_target_is_reciprocal_positive_shifted_sqrt(ctx, target, var_name)
        || diff_target_is_scaled_bounded_inverse_trig_surd_quotient(ctx, target, var_name)
        || diff_target_is_bounded_inverse_trig_self_normalized_projection(ctx, target, var_name)
        || expr_is_positive_integer_power_of_low_degree_polynomial(ctx, target, var_name)
        || diff_target_is_constant_scaled_positive_polynomial_power(ctx, target, var_name)
        || diff_target_is_constant_scaled_reciprocal_polynomial_power(ctx, target, var_name)
        || diff_target_is_constant_scaled_atanh_surd_polynomial(ctx, target, var_name)
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

fn shifted_sqrt_positive_constant_parts(
    ctx: &cas_ast::Context,
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
