//! Symbolic integration helpers shared by integration-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_extract::extract_abs_argument_view;
use crate::expr_nary::{build_balanced_add, build_balanced_mul, mul_leaves, AddView, Sign};
use crate::expr_predicates::contains_named_var;
use crate::factor::factor;
use crate::polynomial::Polynomial;
use crate::root_forms::{try_rewrite_simplify_square_root_expr, SimplifySquareRootRewriteKind};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

fn ln_abs(ctx: &mut Context, arg: ExprId) -> ExprId {
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    ctx.call_builtin(BuiltinFn::Ln, vec![abs_arg])
}

fn compact_single_power_polynomial_arg(ctx: &mut Context, arg: ExprId) -> ExprId {
    let factored = factor(ctx, arg);
    if factored == arg {
        return arg;
    }

    match ctx.get(factored) {
        Expr::Pow(_, exp) if is_integer_power_at_least_two(ctx, *exp) => factored,
        _ => arg,
    }
}

fn is_integer_power_at_least_two(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n)
            if n.denom().is_one() && *n >= BigRational::from_integer(2.into())
    )
}

fn is_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(value.into()))
}

fn scale_rational_term(ctx: &mut Context, scale: BigRational, term: ExprId) -> ExprId {
    if scale.is_one() {
        term
    } else if scale == BigRational::from_integer((-1).into()) {
        ctx.add(Expr::Neg(term))
    } else {
        let scale = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale, term)
    }
}

fn is_negative_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == BigRational::new((-1).into(), 2.into()),
        Expr::Div(num, den) => is_number(ctx, *num, -1) && is_number(ctx, *den, 2),
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
            Expr::Div(num, den) => is_number(ctx, *num, 1) && is_number(ctx, *den, 2),
            _ => false,
        },
        _ => false,
    }
}

fn is_positive_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => is_number(ctx, *num, 1) && is_number(ctx, *den, 2),
        _ => false,
    }
}

fn sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if is_positive_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

fn reciprocal_sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_negative_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

fn var_power(ctx: &Context, expr: ExprId, var: &str) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some(BigRational::one()),
        Expr::Pow(base, exp) if is_var(ctx, *base, var) => rational_constant_value(ctx, *exp),
        _ => None,
    }
}

fn scaled_var_power_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational)> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut power = None;

    for factor in factors {
        if let Some(factor_power) = var_power(ctx, factor, var) {
            if power.is_some() {
                return None;
            }
            power = Some(factor_power);
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    Some((scale, power?))
}

#[derive(Clone)]
struct SqrtLinearDenominator {
    scale: BigRational,
    slope: BigRational,
    offset: BigRational,
}

fn positive_linear_polynomial_coeffs(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational)> {
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() != 1 {
        return None;
    }

    let offset = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    (slope.is_positive() && offset.is_positive()).then_some((slope, offset))
}

fn sqrt_var_times_positive_linear_denominator(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<SqrtLinearDenominator> {
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut saw_sqrt_var = false;
    let mut linear_coeffs = None;

    for factor in factors {
        if sqrt_like_radicand(ctx, factor).is_some_and(|radicand| is_var(ctx, radicand, var)) {
            if saw_sqrt_var {
                return None;
            }
            saw_sqrt_var = true;
        } else if let Some(coeffs) = positive_linear_polynomial_coeffs(ctx, factor, var) {
            if linear_coeffs.is_some() {
                return None;
            }
            linear_coeffs = Some(coeffs);
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    let (slope, offset) = linear_coeffs?;
    saw_sqrt_var.then_some(SqrtLinearDenominator {
        scale,
        slope,
        offset,
    })
}

fn expanded_sqrt_var_times_positive_linear_denominator(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<SqrtLinearDenominator> {
    let (left, right) = match ctx.get(den) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };
    let (left_scale, left_power) = scaled_var_power_term(ctx, left, var)?;
    let (right_scale, right_power) = scaled_var_power_term(ctx, right, var)?;

    let half = BigRational::new(1.into(), 2.into());
    let three_halves = BigRational::new(3.into(), 2.into());
    let (offset, slope) = if left_power == half && right_power == three_halves {
        (left_scale, right_scale)
    } else if left_power == three_halves && right_power == half {
        (right_scale, left_scale)
    } else {
        return None;
    };

    (slope.is_positive() && offset.is_positive()).then_some(SqrtLinearDenominator {
        scale: BigRational::one(),
        slope,
        offset,
    })
}

fn sqrt_var_times_positive_linear_parts(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<SqrtLinearDenominator> {
    sqrt_var_times_positive_linear_denominator(ctx, den, var)
        .or_else(|| expanded_sqrt_var_times_positive_linear_denominator(ctx, den, var))
}

fn reciprocal_sqrt_var_over_positive_linear_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<SqrtLinearDenominator> {
    let (slope, offset) = positive_linear_polynomial_coeffs(ctx, den, var)?;

    let factors = mul_leaves(ctx, num);
    let mut scale = BigRational::one();
    let mut saw_reciprocal_sqrt_var = false;

    for factor in factors {
        if reciprocal_sqrt_like_radicand(ctx, factor)
            .is_some_and(|radicand| is_var(ctx, radicand, var))
        {
            if saw_reciprocal_sqrt_var {
                return None;
            }
            saw_reciprocal_sqrt_var = true;
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    saw_reciprocal_sqrt_var.then_some(SqrtLinearDenominator {
        scale,
        slope,
        offset,
    })
}

fn arctan_sqrt_var_reciprocal_antiderivative_from_parts(
    ctx: &mut Context,
    scale: BigRational,
    slope: BigRational,
    offset: BigRational,
    var: &str,
) -> Option<ExprId> {
    if scale.is_zero() {
        return None;
    }
    if !slope.is_positive() || !offset.is_positive() {
        return None;
    }

    let product = slope.clone() * offset.clone();
    let product_root = exact_rational_sqrt(&product)?;
    let ratio = slope / offset;
    let ratio_root = exact_rational_sqrt(&ratio)?;

    let var_expr = ctx.var(var);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var_expr]);
    let arctan_arg = scale_factor(ctx, ratio_root, sqrt_var);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    Some(scale_factor(
        ctx,
        scale * BigRational::from_integer(2.into()) / product_root,
        arctan,
    ))
}

fn arctan_sqrt_var_reciprocal_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(parts) = sqrt_var_times_positive_linear_parts(ctx, den, var) {
        let num_scale = rational_constant_value(ctx, num)?;
        return arctan_sqrt_var_reciprocal_antiderivative_from_parts(
            ctx,
            num_scale / parts.scale,
            parts.slope,
            parts.offset,
            var,
        );
    }

    let parts = reciprocal_sqrt_var_over_positive_linear_parts(ctx, num, den, var)?;
    arctan_sqrt_var_reciprocal_antiderivative_from_parts(
        ctx,
        parts.scale,
        parts.slope,
        parts.offset,
        var,
    )
}

fn arctan_sqrt_var_reciprocal_required_positive_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let matches_denominator_form = sqrt_var_times_positive_linear_parts(ctx, *den, var).is_some()
        && rational_constant_value(ctx, *num).is_some();
    let matches_numerator_form =
        reciprocal_sqrt_var_over_positive_linear_parts(ctx, *num, *den, var).is_some();
    if matches_denominator_form || matches_numerator_form {
        return Some(ctx.var(var));
    }
    None
}

struct ArctanSqrtAffineDerivativeParts {
    radicand: ExprId,
    scale: BigRational,
}

fn arctan_sqrt_affine_derivative_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ArctanSqrtAffineDerivativeParts> {
    let mut num_scale = BigRational::one();
    let mut numerator_sqrt_radicand = None;

    for factor in mul_leaves(ctx, num) {
        if let Some(radicand) = sqrt_like_radicand(ctx, factor) {
            if numerator_sqrt_radicand.is_some() {
                return None;
            }
            numerator_sqrt_radicand = Some(radicand);
        } else {
            num_scale *= rational_constant_value(ctx, factor)?;
        }
    }

    let mut den_scale = BigRational::one();
    let mut denominator_sqrt_radicand = None;
    let mut polynomial_factors = Vec::new();

    for factor in mul_leaves(ctx, den) {
        if let Some(radicand) = sqrt_like_radicand(ctx, factor) {
            if denominator_sqrt_radicand.is_some() {
                return None;
            }
            denominator_sqrt_radicand = Some(radicand);
        } else if let Some(scale) = rational_constant_value(ctx, factor) {
            den_scale *= scale;
        } else {
            polynomial_factors.push(factor);
        }
    }

    if den_scale.is_zero() {
        return None;
    }

    if let Some(radicand) = numerator_sqrt_radicand {
        if denominator_sqrt_radicand.is_some() || polynomial_factors.len() != 2 {
            return None;
        }
        return arctan_sqrt_affine_derivative_parts_from_normalized_factors(
            ctx,
            radicand,
            &polynomial_factors,
            num_scale,
            den_scale,
            var,
        );
    }

    let radicand = denominator_sqrt_radicand?;
    if polynomial_factors.len() != 1 {
        return None;
    }
    arctan_sqrt_affine_derivative_parts_from_direct_factors(
        ctx,
        radicand,
        polynomial_factors[0],
        num_scale,
        den_scale,
        var,
    )
}

fn arctan_sqrt_affine_derivative_parts_from_direct_factors(
    ctx: &Context,
    radicand: ExprId,
    gap_factor: ExprId,
    num_scale: BigRational,
    den_scale: BigRational,
    var: &str,
) -> Option<ArctanSqrtAffineDerivativeParts> {
    let radicand_poly = affine_radicand_polynomial(ctx, radicand, var)?;
    let gap_poly = radicand_plus_one_polynomial(&radicand_poly);
    let gap_ratio = polynomial_ratio_to_expr_factor(ctx, &gap_poly, gap_factor, var)?;
    let kernel_scale = num_scale * gap_ratio / den_scale;
    let scale = arctan_sqrt_affine_output_scale(&radicand_poly, kernel_scale)?;
    Some(ArctanSqrtAffineDerivativeParts { radicand, scale })
}

fn arctan_sqrt_affine_derivative_parts_from_normalized_factors(
    ctx: &Context,
    radicand: ExprId,
    factors: &[ExprId],
    num_scale: BigRational,
    den_scale: BigRational,
    var: &str,
) -> Option<ArctanSqrtAffineDerivativeParts> {
    let radicand_poly = affine_radicand_polynomial(ctx, radicand, var)?;
    let gap_poly = radicand_plus_one_polynomial(&radicand_poly);
    let (radicand_ratio, gap_ratio) =
        normalized_affine_radicand_and_gap_ratios(ctx, &radicand_poly, &gap_poly, factors, var)?;
    let kernel_scale = num_scale * radicand_ratio * gap_ratio / den_scale;
    let scale = arctan_sqrt_affine_output_scale(&radicand_poly, kernel_scale)?;
    Some(ArctanSqrtAffineDerivativeParts { radicand, scale })
}

fn affine_radicand_polynomial(ctx: &Context, radicand: ExprId, var: &str) -> Option<Polynomial> {
    let poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    (poly.degree() == 1).then_some(poly)
}

fn radicand_plus_one_polynomial(radicand: &Polynomial) -> Polynomial {
    radicand.add(&Polynomial::one(radicand.var.clone()))
}

fn polynomial_ratio_to_expr_factor(
    ctx: &Context,
    target: &Polynomial,
    factor: ExprId,
    var: &str,
) -> Option<BigRational> {
    let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
    constant_polynomial_ratio(target, &factor_poly)
}

fn normalized_affine_radicand_and_gap_ratios(
    ctx: &Context,
    radicand: &Polynomial,
    gap: &Polynomial,
    factors: &[ExprId],
    var: &str,
) -> Option<(BigRational, BigRational)> {
    if factors.len() != 2 {
        return None;
    }

    polynomial_ratio_to_expr_factor(ctx, radicand, factors[0], var)
        .and_then(|radicand_ratio| {
            polynomial_ratio_to_expr_factor(ctx, gap, factors[1], var)
                .map(|gap_ratio| (radicand_ratio, gap_ratio))
        })
        .or_else(|| {
            polynomial_ratio_to_expr_factor(ctx, radicand, factors[1], var).and_then(
                |radicand_ratio| {
                    polynomial_ratio_to_expr_factor(ctx, gap, factors[0], var)
                        .map(|gap_ratio| (radicand_ratio, gap_ratio))
                },
            )
        })
}

fn arctan_sqrt_affine_output_scale(
    radicand: &Polynomial,
    kernel_scale: BigRational,
) -> Option<BigRational> {
    if kernel_scale.is_zero() {
        return None;
    }
    let slope = radicand.coeffs.get(1).cloned()?;
    if slope.is_zero() {
        return None;
    }
    Some(kernel_scale * BigRational::from_integer(2.into()) / slope)
}

fn arctan_sqrt_affine_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = arctan_sqrt_affine_derivative_parts(ctx, num, den, var)?;
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![parts.radicand]);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![sqrt]);
    Some(scale_factor(ctx, parts.scale, arctan))
}

fn arctan_sqrt_affine_derivative_required_positive_radicand(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    arctan_sqrt_affine_derivative_parts(ctx, *num, *den, var).map(|parts| parts.radicand)
}

pub fn integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    (sqrt_var_times_positive_linear_parts(ctx, *den, var).is_some()
        && rational_constant_value(ctx, *num).is_some())
        || reciprocal_sqrt_var_over_positive_linear_parts(ctx, *num, *den, var).is_some()
}

fn is_var_square(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => is_var(ctx, *base, var) && is_number(ctx, *exp, 2),
        _ => false,
    }
}

fn is_var_square_plus_one(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            (is_var_square(ctx, *l, var) && is_number(ctx, *r, 1))
                || (is_number(ctx, *l, 1) && is_var_square(ctx, *r, var))
        }
        _ => false,
    }
}

fn reciprocal_trig_square_parts(ctx: &Context, den: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let (base, exp) = match ctx.get(den) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !is_number(ctx, exp, 2) {
        return None;
    }

    let (builtin, arg) = match ctx.get(base) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    match builtin {
        BuiltinFn::Cos | BuiltinFn::Sin => Some((builtin, arg)),
        _ => None,
    }
}

fn reciprocal_trig_square_antiderivative(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;

    let integral = match builtin {
        BuiltinFn::Cos => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            ctx.add(Expr::Div(sin_arg, cos_arg))
        }
        BuiltinFn::Sin => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cot_arg = ctx.add(Expr::Div(cos_arg, sin_arg));
            ctx.add(Expr::Neg(cot_arg))
        }
        _ => return None,
    };

    let is_a_one = if let Expr::Number(n) = ctx.get(a) {
        n.is_one()
    } else {
        false
    };
    if is_a_one {
        Some(integral)
    } else {
        Some(ctx.add(Expr::Div(integral, a)))
    }
}

fn polynomial_reciprocal_trig_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let integral = match builtin {
        BuiltinFn::Cos => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            ctx.add(Expr::Div(sin_arg, cos_arg))
        }
        BuiltinFn::Sin => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cot_arg = ctx.add(Expr::Div(cos_arg, sin_arg));
            ctx.add(Expr::Neg(cot_arg))
        }
        _ => return None,
    };

    if scale.is_one() {
        return Some(integral);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

fn reciprocal_hyperbolic_square_arg(
    ctx: &Context,
    den: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let (base, exp) = match ctx.get(den) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !is_number(ctx, exp, 2) {
        return None;
    }

    match ctx.get(base) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn reciprocal_hyperbolic_cosh_square_arg(ctx: &Context, den: ExprId) -> Option<ExprId> {
    reciprocal_hyperbolic_square_arg(ctx, den, BuiltinFn::Cosh)
}

fn reciprocal_hyperbolic_sinh_square_arg(ctx: &Context, den: ExprId) -> Option<ExprId> {
    reciprocal_hyperbolic_square_arg(ctx, den, BuiltinFn::Sinh)
}

fn hyperbolic_tanh_reciprocal_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_cosh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    if scale.is_one() {
        return Some(tanh_arg);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, tanh_arg))
}

fn hyperbolic_coth_reciprocal_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_sinh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let quotient = ctx.add(Expr::Div(cosh_arg, sinh_arg));
    if scale.is_one() {
        return Some(ctx.add(Expr::Neg(quotient)));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    let negative_scale_expr = ctx.add(Expr::Neg(scale_expr));
    Some(mul2_raw(ctx, negative_scale_expr, quotient))
}

fn hyperbolic_log_derivative_ratio_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = match ctx.get(den) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cosh => BuiltinFn::Sinh,
        BuiltinFn::Sinh => BuiltinFn::Cosh,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let log_abs_den = ln_abs(ctx, den);
    if scale.is_one() {
        return Some(log_abs_den);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, log_abs_den))
}

fn trig_log_derivative_ratio_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (_, _, scaled) = trig_log_derivative_ratio_scale(ctx, num, den, var)?;
    let log_abs_den = ln_abs(ctx, den);
    if scaled.is_one() {
        return Some(log_abs_den);
    }
    if scaled == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(log_abs_den)));
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, log_abs_den))
}

fn trig_log_derivative_ratio_scale(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    let (den_builtin, arg) = match ctx.get(den) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = trig_log_derivative_ratio_cofactor(ctx, num, numerator_builtin, arg)?;
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let scaled = match den_builtin {
        BuiltinFn::Cos => -scale,
        BuiltinFn::Sin => scale,
        _ => return None,
    };
    Some((den_builtin, arg, scaled))
}

fn trig_log_derivative_ratio_cofactor(
    ctx: &mut Context,
    num: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Some(cofactor) =
        trig_log_derivative_ratio_term_cofactor(ctx, num, numerator_builtin, arg)
    {
        return Some(cofactor);
    }

    let add_view = AddView::from_expr(ctx, num);
    if add_view.terms.len() < 2 {
        return None;
    }

    let mut cofactor_terms = Vec::with_capacity(add_view.terms.len());
    for (term, sign) in add_view.terms {
        let cofactor = trig_log_derivative_ratio_term_cofactor(ctx, term, numerator_builtin, arg)?;
        cofactor_terms.push(signed_term(ctx, cofactor, sign));
    }

    Some(build_balanced_add(ctx, &cofactor_terms))
}

fn trig_log_derivative_ratio_term_cofactor(
    ctx: &mut Context,
    term: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, term);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    Some(cofactor)
}

fn trig_log_derivative_ratio_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let (den_builtin, arg, _) = trig_log_derivative_ratio_scale(ctx, num, den, var)?;
    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn hyperbolic_tanh_reciprocal_log_sinh_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = unary_builtin_arg(ctx, den, BuiltinFn::Tanh)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let log_abs_sinh = ln_abs(ctx, sinh_arg);
    if scale.is_one() {
        return Some(log_abs_sinh);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, log_abs_sinh))
}

fn hyperbolic_cosh_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_cosh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (sinh_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, BuiltinFn::Sinh)
            .is_some_and(|sinh_arg| compare_expr(ctx, sinh_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sinh_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let minus_one = BigRational::from_integer((-1).into());
    let minus_one_expr = ctx.add(Expr::Number(minus_one));
    let reciprocal_cosh = ctx.add(Expr::Pow(cosh_arg, minus_one_expr));
    let scale_expr = ctx.add(Expr::Number(scale));
    let negative_scale_expr = ctx.add(Expr::Neg(scale_expr));
    Some(mul2_raw(ctx, negative_scale_expr, reciprocal_cosh))
}

fn hyperbolic_sinh_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_sinh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (cosh_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, BuiltinFn::Cosh)
            .is_some_and(|cosh_arg| compare_expr(ctx, cosh_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != cosh_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let minus_one = BigRational::from_integer((-1).into());
    let minus_one_expr = ctx.add(Expr::Number(minus_one));
    let reciprocal_sinh = ctx.add(Expr::Pow(sinh_arg, minus_one_expr));
    let scale_expr = ctx.add(Expr::Number(scale));
    let negative_scale_expr = ctx.add(Expr::Neg(scale_expr));
    Some(mul2_raw(ctx, negative_scale_expr, reciprocal_sinh))
}

fn hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    hyperbolic_tanh_reciprocal_square_antiderivative(ctx, num, den, var)
        .or_else(|| hyperbolic_coth_reciprocal_square_antiderivative(ctx, num, den, var))
        .or_else(|| hyperbolic_cosh_reciprocal_derivative_antiderivative(ctx, num, den, var))
        .or_else(|| hyperbolic_sinh_reciprocal_derivative_antiderivative(ctx, num, den, var))
}

fn constant_scaled_hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    constant: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    hyperbolic_reciprocal_square_antiderivative(ctx, scaled_num, den, var)
}

fn reciprocal_trig_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    if !is_number(ctx, num, 1) {
        return None;
    }

    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    get_linear_coeffs(ctx, arg, var)?;
    Some(match builtin {
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Cos, vec![arg]),
        BuiltinFn::Sin => ctx.call_builtin(BuiltinFn::Sin, vec![arg]),
        _ => return None,
    })
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn signed_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            unary_builtin_arg(ctx, *inner, builtin).map(|arg| (arg, -BigRational::one()))
        }
        _ => unary_builtin_arg(ctx, expr, builtin).map(|arg| (arg, BigRational::one())),
    }
}

fn divide_by_coeff_unless_one(ctx: &mut Context, integral: ExprId, coeff: ExprId) -> ExprId {
    let is_coeff_one = if let Expr::Number(n) = ctx.get(coeff) {
        n.is_one()
    } else {
        false
    };

    if is_coeff_one {
        integral
    } else {
        ctx.add(Expr::Div(integral, coeff))
    }
}

fn trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    let numerator_arg = unary_builtin_arg(ctx, num, numerator_builtin)?;
    if compare_expr(ctx, numerator_arg, arg) != Ordering::Equal {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let integral = match den_builtin {
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Sec, vec![arg]),
        BuiltinFn::Sin => {
            let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
            ctx.add(Expr::Neg(csc_arg))
        }
        _ => return None,
    };
    Some(divide_by_coeff_unless_one(ctx, integral, a))
}

fn sqrt_compact_reciprocal_trig_antiderivative(
    ctx: &mut Context,
    den_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let radicand = sqrt_like_radicand(ctx, arg)?;
    let sqrt_arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    match den_builtin {
        BuiltinFn::Cos => Some(ctx.call_builtin(BuiltinFn::Sec, vec![sqrt_arg])),
        BuiltinFn::Sin => {
            let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![sqrt_arg]);
            Some(ctx.add(Expr::Neg(csc_arg)))
        }
        _ => None,
    }
}

fn negate_integration_result(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inner,
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn polynomial_trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let integral =
        if let Some(compact) = sqrt_compact_reciprocal_trig_antiderivative(ctx, den_builtin, arg) {
            compact
        } else {
            match den_builtin {
                BuiltinFn::Cos => {
                    let one = ctx.num(1);
                    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    ctx.add(Expr::Div(one, cos_arg))
                }
                BuiltinFn::Sin => {
                    let one = ctx.num(1);
                    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                    let reciprocal_sin = ctx.add(Expr::Div(one, sin_arg));
                    ctx.add(Expr::Neg(reciprocal_sin))
                }
                _ => return None,
            }
        };
    if scale.is_one() {
        return Some(integral);
    }
    if scale == -BigRational::one() {
        return Some(negate_integration_result(ctx, integral));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

fn polynomial_product_from_factors(
    ctx: &mut Context,
    factors: &[ExprId],
    var: &str,
) -> Option<Polynomial> {
    if factors.is_empty() {
        return Some(Polynomial::one(var.to_string()));
    }

    let product = if factors.len() == 1 {
        factors[0]
    } else {
        build_balanced_mul(ctx, factors)
    };
    Polynomial::from_expr(ctx, product, var).ok()
}

fn quotient_scale_against_polynomial(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    target: &Polynomial,
    var: &str,
) -> Option<BigRational> {
    let numerator = polynomial_product_from_factors(ctx, numerator_factors, var)?;
    let denominator = polynomial_product_from_factors(ctx, denominator_factors, var)?;
    let expected = denominator.mul(target);
    constant_polynomial_ratio(&numerator, &expected)
}

fn remove_matching_factor(
    ctx: &Context,
    factors: &[ExprId],
    target: ExprId,
) -> Option<Vec<ExprId>> {
    let (index, _) = factors
        .iter()
        .enumerate()
        .find(|(_, factor)| compare_expr(ctx, **factor, target) == Ordering::Equal)?;
    Some(
        factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != index).then_some(*factor))
            .collect(),
    )
}

fn sqrt_polynomial_derivative_quotient_scale(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    sqrt_arg: ExprId,
    var: &str,
) -> Option<BigRational> {
    let radicand = sqrt_like_radicand(ctx, sqrt_arg)?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let two = BigRational::from_integer(2.into());
    let half_derivative = radicand_poly.derivative().div_scalar(&two);
    if half_derivative.is_zero() {
        return None;
    }

    for (idx, factor) in numerator_factors.iter().enumerate() {
        let Some(factor_radicand) = reciprocal_sqrt_like_radicand(ctx, *factor) else {
            continue;
        };
        if compare_expr(ctx, factor_radicand, radicand) != Ordering::Equal {
            continue;
        }

        let remaining_numerator: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        return quotient_scale_against_polynomial(
            ctx,
            &remaining_numerator,
            denominator_factors,
            &half_derivative,
            var,
        );
    }

    for (idx, factor) in numerator_factors.iter().enumerate() {
        let Some(factor_radicand) = sqrt_like_radicand(ctx, *factor) else {
            continue;
        };
        if compare_expr(ctx, factor_radicand, radicand) != Ordering::Equal {
            continue;
        }

        let Some(remaining_denominator) =
            remove_matching_factor(ctx, denominator_factors, radicand)
        else {
            continue;
        };
        let remaining_numerator: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        return quotient_scale_against_polynomial(
            ctx,
            &remaining_numerator,
            &remaining_denominator,
            &half_derivative,
            var,
        );
    }

    for (idx, factor) in denominator_factors.iter().enumerate() {
        let Some(factor_radicand) = sqrt_like_radicand(ctx, *factor) else {
            continue;
        };
        if compare_expr(ctx, factor_radicand, radicand) != Ordering::Equal {
            continue;
        }

        let remaining_denominator: Vec<_> = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        return quotient_scale_against_polynomial(
            ctx,
            numerator_factors,
            &remaining_denominator,
            &half_derivative,
            var,
        );
    }

    None
}

fn sqrt_trig_reciprocal_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, ExprId, BigRational)> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (den_builtin, arg, radicand, scale) =
                sqrt_trig_reciprocal_derivative_parts(ctx, inner, var)?;
            return Some((den_builtin, arg, radicand, -scale));
        }
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (denominator_index, (den_builtin, arg)) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| {
                reciprocal_trig_square_parts(ctx, *factor).map(|parts| (idx, parts))
            })?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };

    let (numerator_index, numerator_sign) =
        numerator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| {
                let (numerator_arg, sign) =
                    signed_unary_builtin_arg(ctx, *factor, numerator_builtin)?;
                (compare_expr(ctx, numerator_arg, arg) == Ordering::Equal).then_some((idx, sign))
            })?;

    let remaining_numerator: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
        .collect();
    let scale = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        &remaining_numerator,
        &remaining_denominator,
        arg,
        var,
    )? * numerator_sign;
    if scale.is_zero() {
        return None;
    }

    let radicand = sqrt_like_radicand(ctx, arg)?;
    Some((den_builtin, arg, radicand, scale))
}

fn sqrt_trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, _, radicand, scale) = sqrt_trig_reciprocal_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let integral = match den_builtin {
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Sec, vec![arg]),
        BuiltinFn::Sin => {
            let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
            ctx.add(Expr::Neg(csc_arg))
        }
        _ => return None,
    };
    Some(scale_rational_term(ctx, scale, integral))
}

pub fn integrate_symbolic_is_sqrt_trig_reciprocal_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_trig_reciprocal_derivative_parts(ctx, expr, var).is_some()
}

fn sqrt_trig_log_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, ExprId, BigRational)> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (den_builtin, arg, radicand, scale) =
                sqrt_trig_log_derivative_parts(ctx, inner, var)?;
            return Some((den_builtin, arg, radicand, -scale));
        }
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (denominator_index, (den_builtin, arg)) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| {
                reciprocal_trig_factor_parts(ctx, *factor).map(|parts| (idx, parts))
            })?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };

    let (numerator_index, _) = numerator_factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let remaining_numerator: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
        .collect();
    let scale = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        &remaining_numerator,
        &remaining_denominator,
        arg,
        var,
    )?;
    if scale.is_zero() {
        return None;
    }

    let radicand = sqrt_like_radicand(ctx, arg)?;
    Some((den_builtin, arg, radicand, scale))
}

fn sqrt_trig_log_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, _, radicand, scale) = sqrt_trig_log_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let den_arg = ctx.call_builtin(den_builtin, vec![arg]);
    let log_abs_den = ln_abs(ctx, den_arg);
    let integral = match den_builtin {
        BuiltinFn::Cos => ctx.add(Expr::Neg(log_abs_den)),
        BuiltinFn::Sin => log_abs_den,
        _ => return None,
    };
    Some(scale_rational_term(ctx, scale, integral))
}

pub fn integrate_symbolic_is_sqrt_trig_log_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_trig_log_derivative_parts(ctx, expr, var).is_some()
}

fn sqrt_hyperbolic_log_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    let (numerator_factors, denominator_factors) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (mul_leaves(ctx, num), mul_leaves(ctx, den)),
        _ => (mul_leaves(ctx, expr), Default::default()),
    };

    for (idx, factor) in numerator_factors.iter().enumerate() {
        let Some(arg) = unary_builtin_arg(ctx, *factor, BuiltinFn::Tanh) else {
            continue;
        };
        let remaining_numerator: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        let scale = sqrt_polynomial_derivative_quotient_scale(
            ctx,
            &remaining_numerator,
            &denominator_factors,
            arg,
            var,
        )?;
        if scale.is_zero() {
            return None;
        }
        let radicand = sqrt_like_radicand(ctx, arg)?;
        return Some((BuiltinFn::Cosh, radicand, scale));
    }

    for (idx, factor) in denominator_factors.iter().enumerate() {
        let Some(arg) = unary_builtin_arg(ctx, *factor, BuiltinFn::Tanh) else {
            continue;
        };
        let remaining_denominator: Vec<_> = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        let scale = sqrt_polynomial_derivative_quotient_scale(
            ctx,
            &numerator_factors,
            &remaining_denominator,
            arg,
            var,
        )?;
        if scale.is_zero() {
            return None;
        }
        let radicand = sqrt_like_radicand(ctx, arg)?;
        return Some((BuiltinFn::Sinh, radicand, scale));
    }

    None
}

fn sqrt_hyperbolic_log_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (log_builtin, radicand, scale) = sqrt_hyperbolic_log_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let log_arg = ctx.call_builtin(log_builtin, vec![arg]);
    let integral = ln_abs(ctx, log_arg);
    Some(scale_rational_term(ctx, scale, integral))
}

pub fn integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_hyperbolic_log_derivative_parts(ctx, expr, var).is_some()
}

fn reciprocal_hyperbolic_square_factor_parts(
    ctx: &Context,
    factor: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Some(arg) = reciprocal_hyperbolic_cosh_square_arg(ctx, factor) {
        return Some((BuiltinFn::Cosh, arg));
    }
    if let Some(arg) = reciprocal_hyperbolic_sinh_square_arg(ctx, factor) {
        return Some((BuiltinFn::Sinh, arg));
    }
    None
}

fn sqrt_hyperbolic_reciprocal_square_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (denominator_index, (den_builtin, arg)) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| {
                reciprocal_hyperbolic_square_factor_parts(ctx, *factor).map(|parts| (idx, parts))
            })?;

    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
        .collect();
    let scale = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        &numerator_factors,
        &remaining_denominator,
        arg,
        var,
    )?;
    if scale.is_zero() {
        return None;
    }

    let radicand = sqrt_like_radicand(ctx, arg)?;
    Some((den_builtin, radicand, scale))
}

fn sqrt_hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, radicand, scale) = sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let integral = match den_builtin {
        BuiltinFn::Cosh => ctx.call_builtin(BuiltinFn::Tanh, vec![arg]),
        BuiltinFn::Sinh => {
            let one = ctx.num(1);
            let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
            let reciprocal_tanh = ctx.add(Expr::Div(one, tanh_arg));
            ctx.add(Expr::Neg(reciprocal_tanh))
        }
        _ => return None,
    };
    Some(scale_rational_term(ctx, scale, integral))
}

pub fn integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var).is_some()
}

fn sqrt_hyperbolic_reciprocal_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (denominator_index, (den_builtin, arg)) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| {
                reciprocal_hyperbolic_square_factor_parts(ctx, *factor).map(|parts| (idx, parts))
            })?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cosh => BuiltinFn::Sinh,
        BuiltinFn::Sinh => BuiltinFn::Cosh,
        _ => return None,
    };

    let (numerator_index, _) = numerator_factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let remaining_numerator: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
        .collect();
    let scale = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        &remaining_numerator,
        &remaining_denominator,
        arg,
        var,
    )?;
    if scale.is_zero() {
        return None;
    }

    let radicand = sqrt_like_radicand(ctx, arg)?;
    Some((den_builtin, radicand, scale))
}

fn sqrt_hyperbolic_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, radicand, scale) =
        sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let den_arg = ctx.call_builtin(den_builtin, vec![arg]);
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, den_arg));
    let integral = ctx.add(Expr::Neg(reciprocal));
    Some(scale_rational_term(ctx, scale, integral))
}

pub fn integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var).is_some()
}

fn reciprocal_trig_factor_parts(ctx: &Context, den: ExprId) -> Option<(BuiltinFn, ExprId)> {
    match ctx.get(den) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(*fn_id)?;
            match builtin {
                BuiltinFn::Cos | BuiltinFn::Sin => Some((builtin, args[0])),
                _ => None,
            }
        }
        _ => None,
    }
}

fn polynomial_trig_reciprocal_factor_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_factor_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Tan,
        BuiltinFn::Sin => BuiltinFn::Cot,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let integral =
        if let Some(compact) = sqrt_compact_reciprocal_trig_antiderivative(ctx, den_builtin, arg) {
            compact
        } else {
            match den_builtin {
                BuiltinFn::Cos => {
                    let one = ctx.num(1);
                    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    ctx.add(Expr::Div(one, cos_arg))
                }
                BuiltinFn::Sin => {
                    let one = ctx.num(1);
                    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                    let reciprocal_sin = ctx.add(Expr::Div(one, sin_arg));
                    ctx.add(Expr::Neg(reciprocal_sin))
                }
                _ => return None,
            }
        };
    if scale.is_one() {
        return Some(integral);
    }
    if scale == -BigRational::one() {
        return Some(negate_integration_result(ctx, integral));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

fn constant_scaled_trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    constant: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    polynomial_trig_reciprocal_derivative_antiderivative(ctx, scaled_num, den, var)
}

fn trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    let numerator_arg = unary_builtin_arg(ctx, num, numerator_builtin)?;
    if compare_expr(ctx, numerator_arg, arg) != Ordering::Equal {
        return None;
    }
    get_linear_coeffs(ctx, arg, var)?;
    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    if cofactor_factors.is_empty() {
        return None;
    }
    let cofactor = build_balanced_mul(ctx, &cofactor_factors);

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn polynomial_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(ctx, num, den, var)
}

fn sqrt_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, _, radicand, _) = sqrt_trig_reciprocal_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn sqrt_trig_log_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, _, radicand, _) = sqrt_trig_log_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn sqrt_hyperbolic_log_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (log_builtin, radicand, _) = sqrt_hyperbolic_log_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some(ctx.call_builtin(log_builtin, vec![arg]))
}

fn sqrt_hyperbolic_reciprocal_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, radicand, _) = sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn sqrt_hyperbolic_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, radicand, _) = sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn polynomial_trig_reciprocal_factor_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let (den_builtin, arg) = reciprocal_trig_factor_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Tan,
        BuiltinFn::Sin => BuiltinFn::Cot,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;
    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn constant_scaled_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (constant, expr) = match ctx.get(expr).clone() {
        Expr::Mul(l, r) if !contains_named_var(ctx, l, var) => (l, r),
        Expr::Mul(l, r) if !contains_named_var(ctx, r, var) => (r, l),
        _ => return None,
    };
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(ctx, scaled_num, den, var)
}

fn trig_log_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let log_arg_builtin = match builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        BuiltinFn::Sec | BuiltinFn::Csc => {
            return sec_csc_log_antiderivative(ctx, builtin, arg, a);
        }
        _ => return None,
    };
    let log_arg = ctx.call_builtin(log_arg_builtin, vec![arg]);
    let log_abs = ln_abs(ctx, log_arg);
    let integral = match builtin {
        BuiltinFn::Tan => ctx.add(Expr::Neg(log_abs)),
        BuiltinFn::Cot => log_abs,
        _ => return None,
    };

    let is_a_one = if let Expr::Number(n) = ctx.get(a) {
        n.is_one()
    } else {
        false
    };
    if is_a_one {
        Some(integral)
    } else {
        Some(ctx.add(Expr::Div(integral, a)))
    }
}

fn sec_csc_log_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    coeff: ExprId,
) -> Option<ExprId> {
    let log_arg = match builtin {
        BuiltinFn::Sec => {
            let primary = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
            let companion = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
            ctx.add(Expr::Add(primary, companion))
        }
        BuiltinFn::Csc => csc_log_quotient_argument(ctx, arg),
        _ => return None,
    };
    let log_abs = ln_abs(ctx, log_arg);
    Some(divide_by_coeff_unless_one(ctx, log_abs, coeff))
}

fn csc_log_quotient_argument(ctx: &mut Context, arg: ExprId) -> ExprId {
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let one = ctx.num(1);
    let numerator = ctx.add(Expr::Sub(cos_arg, one));
    let denominator = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    ctx.add(Expr::Div(numerator, denominator))
}

fn reciprocal_trig_log_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, num, 1) {
        return None;
    }

    let (builtin, arg) = unary_builtin_arg(ctx, den, BuiltinFn::Cos)
        .map(|arg| (BuiltinFn::Sec, arg))
        .or_else(|| unary_builtin_arg(ctx, den, BuiltinFn::Sin).map(|arg| (BuiltinFn::Csc, arg)))?;
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    sec_csc_log_antiderivative(ctx, builtin, arg, a)
}

fn trig_log_required_nonzero(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let (builtin, arg) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(fn_id)?, args[0]),
        _ => return None,
    };
    let nonzero_builtin = match builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        BuiltinFn::Sec => BuiltinFn::Cos,
        BuiltinFn::Csc => BuiltinFn::Sin,
        _ => return None,
    };
    get_linear_coeffs(ctx, arg, var)?;
    Some(ctx.call_builtin(nonzero_builtin, vec![arg]))
}

fn polynomial_trig_log_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (trig_index, builtin, arg) =
        factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| match ctx.get(*factor) {
                Expr::Function(fn_id, args) if args.len() == 1 => {
                    let builtin = ctx.builtin_of(*fn_id)?;
                    matches!(builtin, BuiltinFn::Tan | BuiltinFn::Cot)
                        .then_some((idx, builtin, args[0]))
                }
                _ => None,
            })?;

    if factors.iter().enumerate().any(|(idx, factor)| {
        idx != trig_index
            && matches!(
                ctx.get(*factor),
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && ctx
                            .builtin_of(*fn_id)
                            .is_some_and(|builtin| matches!(builtin, BuiltinFn::Tan | BuiltinFn::Cot))
            )
    }) {
        return None;
    }

    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let nonzero_builtin = match builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        _ => return None,
    };
    Some(ctx.call_builtin(nonzero_builtin, vec![arg]))
}

fn reciprocal_trig_log_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    if !is_number(ctx, num, 1) {
        return None;
    }

    let (nonzero_builtin, arg) = unary_builtin_arg(ctx, den, BuiltinFn::Cos)
        .map(|arg| (BuiltinFn::Cos, arg))
        .or_else(|| unary_builtin_arg(ctx, den, BuiltinFn::Sin).map(|arg| (BuiltinFn::Sin, arg)))?;
    get_linear_coeffs(ctx, arg, var)?;
    Some(ctx.call_builtin(nonzero_builtin, vec![arg]))
}

fn positive_one_plus_square_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_number(ctx, *left, 1) {
        return square_base(ctx, *right);
    }
    if is_number(ctx, *right, 1) {
        return square_base(ctx, *left);
    }

    None
}

fn square_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    is_number(ctx, *exp, 2).then_some(*base)
}

fn arctan_unary_derivative_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(num).clone() {
        let integral = arctan_unary_derivative_substitution_antiderivative(ctx, inner, den, var)?;
        return Some(ctx.add(Expr::Neg(integral)));
    }

    let arg = positive_one_plus_square_arg(ctx, den)?;
    let (fn_id, args) = match ctx.get(arg).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let inner = args[0];
    if !contains_named_var(ctx, inner, var) {
        return None;
    }

    let (companion_builtin, derivative_sign) = match ctx.builtin_of(fn_id)? {
        BuiltinFn::Sin => (BuiltinFn::Cos, BigRational::one()),
        BuiltinFn::Cos => (BuiltinFn::Sin, -BigRational::one()),
        BuiltinFn::Sinh => (BuiltinFn::Cosh, BigRational::one()),
        BuiltinFn::Cosh => (BuiltinFn::Sinh, BigRational::one()),
        _ => return None,
    };

    let factors = mul_leaves(ctx, num);
    let (companion_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, companion_builtin)
            .is_some_and(|companion_arg| compare_expr(ctx, companion_arg, inner) == Ordering::Equal)
    })?;
    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != companion_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let inner_poly = Polynomial::from_expr(ctx, inner, var).ok()?;
    let derivative = inner_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)? * derivative_sign;
    if scale.is_zero() {
        return None;
    }

    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    if scale.is_one() {
        return Some(arctan);
    }
    if scale == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(arctan)));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, arctan))
}

fn arctan_scaled_variable_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let zero_offset = is_number(ctx, offset, 0);
    let presentation_arg = if zero_offset && coeff.is_one() {
        arg
    } else {
        cas_ast::hold::wrap_hold(ctx, arg)
    };
    let arctan_arg = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    let leading_linear = if zero_offset {
        ctx.var(var)
    } else {
        let scale = BigRational::one() / coeff.clone();
        scale_rational_term(ctx, scale, presentation_arg)
    };
    let leading_term = mul2_raw(ctx, leading_linear, arctan_arg);

    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(presentation_arg, two));
    let one = ctx.num(1);
    let log_arg = ctx.add(Expr::Add(arg_sq, one));
    let log_term = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);

    let two_coeff = coeff * BigRational::from_integer(2.into());
    let log_scale = BigRational::one() / two_coeff;
    let signed_log_term = scale_rational_term(ctx, -log_scale, log_term);
    let leading_term = cas_ast::hold::wrap_hold(ctx, leading_term);
    let signed_log_term = cas_ast::hold::wrap_hold(ctx, signed_log_term);
    Some(ctx.add(Expr::Add(leading_term, signed_log_term)))
}

fn reciprocal_affine_variable_denominator(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, bool)> {
    let denominator = match ctx.get(arg) {
        Expr::Div(num, den) if is_number(ctx, *num, 1) => *den,
        Expr::Pow(base, exp) if is_number(ctx, *exp, -1) => *base,
        _ => return None,
    };
    let (coeff, offset) = get_linear_coeffs(ctx, denominator, var)?;
    let has_zero_offset = is_number(ctx, offset, 0);

    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    Some((denominator, coeff, has_zero_offset))
}

fn arctan_reciprocal_affine_variable_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (denominator, coeff, has_zero_offset) =
        reciprocal_affine_variable_denominator(ctx, arg, var)?;

    let arctan_arg = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    let leading_linear = if has_zero_offset {
        ctx.var(var)
    } else {
        let scale = BigRational::one() / coeff.clone();
        if scale.is_one() {
            denominator
        } else {
            let scale = ctx.add(Expr::Number(scale));
            mul2_raw(ctx, scale, denominator)
        }
    };
    let leading_term = mul2_raw(ctx, leading_linear, arctan_arg);

    let two = ctx.num(2);
    let denominator_sq = ctx.add(Expr::Pow(denominator, two));
    let one = ctx.num(1);
    let log_arg = ctx.add(Expr::Add(denominator_sq, one));
    let log_term = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);

    let two_coeff = coeff * BigRational::from_integer(2.into());
    let log_scale = BigRational::one() / two_coeff;
    let scaled_log_term = if log_scale.is_one() {
        log_term
    } else {
        let scale = ctx.add(Expr::Number(log_scale));
        mul2_raw(ctx, scale, log_term)
    };

    Some(ctx.add(Expr::Add(leading_term, scaled_log_term)))
}

pub fn integrate_symbolic_is_arctan_reciprocal_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
            reciprocal_affine_variable_denominator(ctx, args[0], var)
                .is_some_and(|(_, _, has_zero_offset)| !has_zero_offset)
        }
        Some(BuiltinFn::Arccot | BuiltinFn::Acot) => {
            let Some((coeff, offset)) = get_linear_coeffs(ctx, args[0], var) else {
                return false;
            };
            if is_number(ctx, offset, 0) {
                return false;
            }
            rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
        }
        _ => false,
    }
}

fn asinh_affine_antiderivative(ctx: &mut Context, arg: ExprId, var: &str) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let asinh_arg = ctx.call_builtin(BuiltinFn::Asinh, vec![arg]);
    let leading_term = mul2_raw(ctx, arg, asinh_arg);

    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let one = ctx.num(1);
    let sqrt_arg = ctx.add(Expr::Add(arg_sq, one));
    let sqrt_arg = if is_number(ctx, offset, 0) {
        sqrt_arg
    } else {
        cas_ast::hold::wrap_hold(ctx, sqrt_arg)
    };
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_arg]);
    let primitive = ctx.add(Expr::Sub(leading_term, sqrt_term));

    let scale = BigRational::one() / coeff;
    if scale.is_one() {
        Some(primitive)
    } else {
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, asinh_arg)
        } else {
            scale_rational_term(ctx, scale.clone(), leading_term)
        };
        let sqrt_scale = -scale;
        let scaled_sqrt = scale_rational_term(ctx, sqrt_scale, sqrt_term);
        Some(ctx.add(Expr::Add(scaled_leading, scaled_sqrt)))
    }
}

fn unit_minus_square(ctx: &mut Context, arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    ctx.add(Expr::Sub(one, arg_sq))
}

fn atanh_affine_antiderivative(ctx: &mut Context, arg: ExprId, var: &str) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let atanh_arg = ctx.call_builtin(BuiltinFn::Atanh, vec![arg]);
    let leading_term = mul2_raw(ctx, arg, atanh_arg);

    let log_arg = unit_minus_square(ctx, arg);
    let log_term = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let half_log = mul2_raw(ctx, half, log_term);
    let primitive = ctx.add(Expr::Add(leading_term, half_log));

    let scale = BigRational::one() / coeff;
    if scale.is_one() {
        Some(primitive)
    } else {
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, atanh_arg)
        } else {
            scale_rational_term(ctx, scale.clone(), leading_term)
        };
        let half_scale = scale / BigRational::from_integer(2.into());
        let scaled_log = scale_rational_term(ctx, half_scale, log_term);
        Some(ctx.add(Expr::Add(scaled_log, scaled_leading)))
    }
}

fn acosh_radicands(ctx: &mut Context, arg: ExprId) -> (ExprId, ExprId) {
    let one = ctx.num(1);
    let left = ctx.add(Expr::Sub(arg, one));
    let right = ctx.add(Expr::Add(arg, one));
    (left, right)
}

fn acosh_polynomial_radicands(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let one_poly = Polynomial::one(var.to_string());
    let left = arg_poly.sub(&one_poly).to_expr(ctx);
    let right = arg_poly.add(&one_poly).to_expr(ctx);
    Some((left, right))
}

fn acosh_affine_antiderivative(ctx: &mut Context, arg: ExprId, var: &str) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let acosh_arg = ctx.call_builtin(BuiltinFn::Acosh, vec![arg]);
    let leading_term = mul2_raw(ctx, arg, acosh_arg);

    let (left, right) =
        acosh_polynomial_radicands(ctx, arg, var).unwrap_or_else(|| acosh_radicands(ctx, arg));
    let sqrt_left = ctx.call_builtin(BuiltinFn::Sqrt, vec![left]);
    let sqrt_right = ctx.call_builtin(BuiltinFn::Sqrt, vec![right]);
    let sqrt_product = mul2_raw(ctx, sqrt_left, sqrt_right);

    if coeff.is_negative() {
        let held_sqrt_product = cas_ast::hold::wrap_hold(ctx, sqrt_product);
        let held_leading_term = cas_ast::hold::wrap_hold(ctx, leading_term);
        let primitive = ctx.add(Expr::Sub(held_sqrt_product, held_leading_term));
        let scale = -BigRational::one() / coeff;
        if scale.is_one() {
            return Some(primitive);
        }
        let scaled_sqrt = scale_rational_term(ctx, scale.clone(), held_sqrt_product);
        let leading_scale = -scale;
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, acosh_arg)
        } else {
            scale_rational_term(ctx, leading_scale, held_leading_term)
        };
        return Some(ctx.add(Expr::Add(scaled_sqrt, scaled_leading)));
    }

    let primitive = ctx.add(Expr::Sub(leading_term, sqrt_product));

    let scale = BigRational::one() / coeff;
    if scale.is_one() {
        Some(primitive)
    } else {
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, acosh_arg)
        } else {
            scale_rational_term(ctx, scale.clone(), leading_term)
        };
        let sqrt_scale = -scale;
        let scaled_sqrt = scale_rational_term(ctx, sqrt_scale, sqrt_product);
        Some(ctx.add(Expr::Add(scaled_leading, scaled_sqrt)))
    }
}

fn scaled_unit_minus_square_linear_radicand(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
    coeff: &BigRational,
) -> Option<ExprId> {
    if coeff.is_zero() {
        return None;
    }
    if coeff.is_one() {
        return Some(unit_minus_square(ctx, arg));
    }

    let raw = unit_minus_square(ctx, arg);
    let poly = Polynomial::from_expr(ctx, raw, var).ok()?;
    let coeff_square = coeff * coeff;
    Some(poly.div_scalar(&coeff_square).to_expr(ctx))
}

fn bounded_inverse_trig_scaled_sqrt_term(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
    coeff: &BigRational,
) -> Option<(ExprId, ExprId)> {
    if coeff.is_negative() {
        let radicand = unit_minus_square(ctx, arg);
        let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        let abs_reciprocal = -BigRational::one() / coeff.clone();
        let sqrt_term = if abs_reciprocal.is_one() {
            sqrt
        } else {
            let scale = ctx.add(Expr::Number(abs_reciprocal));
            mul2_raw(ctx, scale, sqrt)
        };
        return Some((radicand, sqrt_term));
    }

    let radicand = scaled_unit_minus_square_linear_radicand(ctx, arg, var, coeff)?;
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some((radicand, sqrt_term))
}

fn bounded_inverse_trig_linear_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (coeff_expr, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff_expr)?;
    if coeff.is_zero() {
        return None;
    }

    let zero_offset = is_number(ctx, offset, 0);
    let inverse = ctx.call_builtin(builtin, vec![arg]);
    if !zero_offset && coeff.is_negative() {
        let product = mul2_raw(ctx, arg, inverse);
        let radicand = scaled_unit_minus_square_linear_radicand(ctx, arg, var, &coeff)?;
        let coeff_square = &coeff * &coeff;
        let factored_radicand = if coeff_square.is_one() {
            radicand
        } else {
            let coeff_square_expr = ctx.add(Expr::Number(coeff_square));
            mul2_raw(ctx, coeff_square_expr, radicand)
        };
        let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let sqrt_term = ctx.add(Expr::Pow(factored_radicand, half));

        if matches!(builtin, BuiltinFn::Arccos | BuiltinFn::Acos) {
            let sqrt_term = cas_ast::hold::wrap_hold(ctx, sqrt_term);
            let product = cas_ast::hold::wrap_hold(ctx, product);
            let primitive = ctx.add(Expr::Sub(sqrt_term, product));
            let scale = -BigRational::one() / coeff;
            if scale.is_one() {
                return Some(primitive);
            }
            let scaled_sqrt = scale_rational_term(ctx, scale.clone(), sqrt_term);
            let scaled_product = scale_rational_term(ctx, -scale, product);
            return Some(ctx.add(Expr::Add(scaled_sqrt, scaled_product)));
        }

        if !matches!(builtin, BuiltinFn::Arcsin | BuiltinFn::Asin) {
            return None;
        }
        let scale = BigRational::one() / coeff;
        let scaled_product = scale_rational_term(ctx, scale.clone(), product);
        let scaled_sqrt = scale_rational_term(ctx, scale, sqrt_term);
        return Some(ctx.add(Expr::Add(scaled_product, scaled_sqrt)));
    }

    let (product, sqrt_term) = if zero_offset {
        let leading_linear = ctx.var(var);
        let product = mul2_raw(ctx, leading_linear, inverse);
        let (_, sqrt_term) = bounded_inverse_trig_scaled_sqrt_term(ctx, arg, var, &coeff)?;
        (product, sqrt_term)
    } else {
        if !coeff.is_positive() {
            return None;
        }
        let product = mul2_raw(ctx, arg, inverse);
        let radicand = unit_minus_square(ctx, arg);
        let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        (product, sqrt_term)
    };

    let primitive = match builtin {
        BuiltinFn::Arcsin | BuiltinFn::Asin if coeff.is_positive() => {
            ctx.add(Expr::Add(product, sqrt_term))
        }
        BuiltinFn::Arcsin | BuiltinFn::Asin => ctx.add(Expr::Sub(product, sqrt_term)),
        BuiltinFn::Arccos | BuiltinFn::Acos if coeff.is_positive() => {
            ctx.add(Expr::Sub(product, sqrt_term))
        }
        BuiltinFn::Arccos | BuiltinFn::Acos => ctx.add(Expr::Add(product, sqrt_term)),
        _ => return None,
    };

    if zero_offset || coeff.is_one() {
        Some(primitive)
    } else {
        let scale = BigRational::one() / coeff;
        let scaled_product = scale_rational_term(ctx, scale.clone(), product);
        let sqrt_scale = match builtin {
            BuiltinFn::Arcsin | BuiltinFn::Asin => scale,
            BuiltinFn::Arccos | BuiltinFn::Acos => -scale,
            _ => return None,
        };
        let scaled_sqrt = scale_rational_term(ctx, sqrt_scale, sqrt_term);
        Some(ctx.add(Expr::Add(scaled_product, scaled_sqrt)))
    }
}

fn bounded_inverse_trig_linear_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos
    ) {
        return None;
    }
    let (coeff_expr, offset) = get_linear_coeffs(ctx, args[0], var)?;
    let coeff = rational_constant_value(ctx, coeff_expr)?;
    if coeff.is_zero() {
        return None;
    }

    if !is_number(ctx, offset, 0) {
        bounded_inverse_trig_linear_antiderivative(ctx, builtin, args[0], var)?;
        if coeff.is_negative() {
            return scaled_unit_minus_square_linear_radicand(ctx, args[0], var, &coeff);
        }
        return Some(unit_minus_square(ctx, args[0]));
    }

    let (radicand, _) = bounded_inverse_trig_scaled_sqrt_term(ctx, args[0], var, &coeff)?;
    Some(radicand)
}

pub fn integrate_symbolic_is_bounded_inverse_trig_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    bounded_inverse_trig_linear_radicand(ctx, expr, var).is_some()
}

pub fn integrate_symbolic_is_asinh_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Asinh) {
        return false;
    }

    let Some((coeff, _)) = get_linear_coeffs(ctx, args[0], var) else {
        return false;
    };
    rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
}

pub fn integrate_symbolic_is_atanh_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) {
        return false;
    }

    let Some((coeff, _)) = get_linear_coeffs(ctx, args[0], var) else {
        return false;
    };
    rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
}

pub fn integrate_symbolic_is_acosh_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Acosh) {
        return false;
    }

    let Some((coeff, _)) = get_linear_coeffs(ctx, args[0], var) else {
        return false;
    };
    rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PolynomialSubstitutionKernel {
    Exp,
    Sin,
    Cos,
    Sinh,
    Cosh,
    Tanh,
    Tan,
    Cot,
}

fn polynomial_substitution_kernel(
    ctx: &Context,
    expr: ExprId,
) -> Option<(PolynomialSubstitutionKernel, ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let kernel = match ctx.builtin_of(*fn_id)? {
                BuiltinFn::Exp => PolynomialSubstitutionKernel::Exp,
                BuiltinFn::Sin => PolynomialSubstitutionKernel::Sin,
                BuiltinFn::Cos => PolynomialSubstitutionKernel::Cos,
                BuiltinFn::Sinh => PolynomialSubstitutionKernel::Sinh,
                BuiltinFn::Cosh => PolynomialSubstitutionKernel::Cosh,
                BuiltinFn::Tanh => PolynomialSubstitutionKernel::Tanh,
                BuiltinFn::Tan => PolynomialSubstitutionKernel::Tan,
                BuiltinFn::Cot => PolynomialSubstitutionKernel::Cot,
                _ => return None,
            };
            Some((kernel, args[0]))
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => {
            Some((PolynomialSubstitutionKernel::Exp, *exp))
        }
        _ => None,
    }
}

fn polynomial_substitution_kernel_antiderivative(
    ctx: &mut Context,
    kernel: PolynomialSubstitutionKernel,
    arg: ExprId,
    kernel_factor: ExprId,
) -> ExprId {
    match kernel {
        PolynomialSubstitutionKernel::Exp => kernel_factor,
        PolynomialSubstitutionKernel::Sin => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            ctx.add(Expr::Neg(cos_arg))
        }
        PolynomialSubstitutionKernel::Cos => ctx.call_builtin(BuiltinFn::Sin, vec![arg]),
        PolynomialSubstitutionKernel::Sinh => ctx.call_builtin(BuiltinFn::Cosh, vec![arg]),
        PolynomialSubstitutionKernel::Cosh => ctx.call_builtin(BuiltinFn::Sinh, vec![arg]),
        PolynomialSubstitutionKernel::Tanh => {
            let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
            ln_abs(ctx, cosh_arg)
        }
        PolynomialSubstitutionKernel::Tan => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let log_abs = ln_abs(ctx, cos_arg);
            ctx.add(Expr::Neg(log_abs))
        }
        PolynomialSubstitutionKernel::Cot => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            ln_abs(ctx, sin_arg)
        }
    }
}

fn constant_polynomial_ratio(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<BigRational> {
    if denominator.is_zero() {
        return None;
    }

    let pivot = denominator
        .coeffs
        .iter()
        .position(|coeff| !coeff.is_zero())?;
    let numerator_pivot = numerator
        .coeffs
        .get(pivot)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let scale = numerator_pivot / denominator.coeffs[pivot].clone();
    let len = numerator.coeffs.len().max(denominator.coeffs.len());

    for idx in 0..len {
        let left = numerator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = denominator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            * scale.clone();
        if left != right {
            return None;
        }
    }

    Some(scale)
}

fn rational_constant_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Add(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left + right)
        }
        Expr::Sub(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left - right)
        }
        Expr::Mul(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left * right)
        }
        Expr::Div(num, den) => {
            let numerator = rational_constant_value(ctx, *num)?;
            let denominator = rational_constant_value(ctx, *den)?;
            if denominator.is_zero() {
                None
            } else {
                Some(numerator / denominator)
            }
        }
        Expr::Neg(inner) => rational_constant_value(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn polynomial_power_factor(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let exponent = rational_constant_value(ctx, *exp)?;
            Some((*base, exponent))
        }
        _ => None,
    }
}

fn rational_coefficient_times_reciprocal_power(
    ctx: &mut Context,
    coefficient: BigRational,
    base: ExprId,
    positive_exponent: BigRational,
) -> ExprId {
    let numerator = BigRational::from_integer(coefficient.numer().clone());
    let numerator = ctx.add(Expr::Number(numerator));

    let denominator_power = if positive_exponent.is_one() {
        base
    } else {
        let exponent = ctx.add(Expr::Number(positive_exponent));
        ctx.add(Expr::Pow(base, exponent))
    };

    let denominator_scale = BigRational::from_integer(coefficient.denom().clone());
    let denominator = if denominator_scale.is_one() {
        denominator_power
    } else {
        let denominator_scale = ctx.add(Expr::Number(denominator_scale));
        mul2_raw(ctx, denominator_scale, denominator_power)
    };

    ctx.add(Expr::Div(numerator, denominator))
}

fn polynomial_power_substitution_from_base(
    ctx: &mut Context,
    cofactor: ExprId,
    base: ExprId,
    exponent: BigRational,
    var: &str,
) -> Option<ExprId> {
    let negative_one = BigRational::from_integer((-1).into());
    if exponent == negative_one {
        return None;
    }

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let base_poly = Polynomial::from_expr(ctx, base, var).ok()?;
    let derivative = base_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let new_exponent = exponent + BigRational::one();
    if new_exponent.is_zero() {
        return None;
    }

    let coefficient = scale / new_exponent.clone();
    if new_exponent.is_integer() && new_exponent < BigRational::zero() {
        return Some(rational_coefficient_times_reciprocal_power(
            ctx,
            coefficient,
            base,
            -new_exponent,
        ));
    }

    let power_exp = ctx.add(Expr::Number(new_exponent.clone()));
    let power = ctx.add(Expr::Pow(base, power_exp));
    if coefficient.is_one() {
        return Some(power);
    }

    let coefficient_expr = ctx.add(Expr::Number(coefficient));
    Some(mul2_raw(ctx, coefficient_expr, power))
}

fn polynomial_power_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (power_index, factor) in factors.iter().enumerate() {
        let Some((base, exponent)) = polynomial_power_factor(ctx, *factor) else {
            continue;
        };

        if !contains_named_var(ctx, base, var) {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != power_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };

        if let Some(integral) =
            polynomial_power_substitution_from_base(ctx, cofactor, base, exponent, var)
        {
            return Some(integral);
        }
    }

    None
}

fn polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    if let Expr::Pow(base, exp) = ctx.get(den) {
        let base = *base;
        let exp = *exp;
        let exponent = rational_constant_value(ctx, exp)?;
        if !exponent.is_integer() || exponent <= BigRational::one() {
            return None;
        }
        if !contains_named_var(ctx, base, var) {
            return None;
        }

        return Some((base, exponent, BigRational::one()));
    }

    scaled_syntactic_polynomial_denominator_power_parts(ctx, den, var)
        .or_else(|| {
            expanded_square_denominator_base(ctx, den, var).map(|base| {
                (
                    base,
                    BigRational::from_integer(2.into()),
                    BigRational::one(),
                )
            })
        })
        .or_else(|| expanded_polynomial_denominator_power_parts(ctx, den, var))
}

fn scaled_syntactic_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    let factors = mul_leaves(ctx, den);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut power_part = None;

    for factor in factors {
        if power_part.is_none() {
            let candidate = match ctx.get(factor) {
                Expr::Pow(base, exp) => Some((*base, *exp)),
                _ => None,
            };

            if let Some((base, exp)) = candidate {
                let exponent = rational_constant_value(ctx, exp)?;
                if exponent.is_integer()
                    && exponent > BigRational::one()
                    && contains_named_var(ctx, base, var)
                    && Polynomial::from_expr(ctx, base, var).is_ok()
                {
                    power_part = Some((base, exponent));
                    continue;
                }
            }
        }

        scale *= rational_constant_value(ctx, factor)?;
    }

    if scale.is_zero() {
        return None;
    }

    power_part.map(|(base, exponent)| (base, exponent, scale))
}

fn negative_syntactic_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    negative_syntactic_polynomial_denominator_power_parts_view(ctx, den, var)
}

fn negative_syntactic_polynomial_denominator_power_parts_view(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    negative_syntactic_polynomial_denominator_power_parts_by(
        ctx,
        den,
        var,
        rational_constant_value,
        rational_constant_value,
    )
}

fn negative_syntactic_polynomial_denominator_power_parts_by<ExponentValue, ScaleValue>(
    ctx: &Context,
    den: ExprId,
    var: &str,
    exponent_value: ExponentValue,
    scale_value: ScaleValue,
) -> Option<(ExprId, BigRational, BigRational)>
where
    ExponentValue: Fn(&Context, ExprId) -> Option<BigRational>,
    ScaleValue: Fn(&Context, ExprId) -> Option<BigRational>,
{
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut power_part = None;

    for factor in factors {
        if power_part.is_none() {
            let candidate = match ctx.get(factor) {
                Expr::Pow(base, exp) => Some((*base, *exp)),
                _ => None,
            };

            if let Some((base, exp)) = candidate {
                let exponent = exponent_value(ctx, exp)?;
                if exponent.is_integer()
                    && exponent < BigRational::zero()
                    && contains_named_var(ctx, base, var)
                    && Polynomial::from_expr(ctx, base, var).is_ok()
                {
                    power_part = Some((base, -exponent));
                    continue;
                }
            }
        }

        scale *= scale_value(ctx, factor)?;
    }

    if scale.is_zero() {
        return None;
    }

    power_part.map(|(base, exponent)| (base, exponent, scale))
}

fn bounded_negative_syntactic_polynomial_denominator_power_parts_view(
    ctx: &Context,
    den: ExprId,
    var: &str,
    max_abs_power: i64,
) -> Option<(ExprId, BigRational, BigRational)> {
    let (base, exponent, scale) = negative_syntactic_polynomial_denominator_power_parts_by(
        ctx,
        den,
        var,
        |ctx, exp| {
            crate::numeric::as_i64(ctx, exp).map(|value| BigRational::from_integer(value.into()))
        },
        |ctx, factor| cas_ast::views::as_rational_const(ctx, factor, 4),
    )?;

    let bound = BigRational::from_integer(max_abs_power.into());
    if exponent > bound {
        return None;
    }

    Some((base, exponent, scale))
}

fn bounded_negative_denominator_power_substitution_target_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    max_abs_power: i64,
) -> Option<(ExprId, BigRational)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (base, exponent, _) = bounded_negative_syntactic_polynomial_denominator_power_parts_view(
        ctx,
        *den,
        var,
        max_abs_power,
    )?;

    let Ok(numerator) = Polynomial::from_expr(ctx, *num, var) else {
        return None;
    };
    let Ok(base_poly) = Polynomial::from_expr(ctx, base, var) else {
        return None;
    };
    let derivative = base_poly.derivative();
    if constant_polynomial_ratio(&numerator, &derivative).is_none_or(|scale| scale.is_zero()) {
        return None;
    }

    Some((base, exponent))
}

pub fn integrate_symbolic_is_bounded_negative_syntactic_denominator_power_substitution_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    max_abs_power: i64,
) -> bool {
    bounded_negative_denominator_power_substitution_target_parts(ctx, expr, var, max_abs_power)
        .is_some()
}

fn reciprocal_quotient_polynomial_denominator_power_parts_view(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    let (scale_expr, reciprocal_den) = match ctx.get(den) {
        Expr::Div(scale, reciprocal_den) => (*scale, *reciprocal_den),
        _ => return None,
    };
    let scale = rational_constant_value(ctx, scale_expr)?;
    if scale.is_zero() {
        return None;
    }

    let (base, exponent) = match ctx.get(reciprocal_den) {
        Expr::Pow(base, exp) => {
            let exponent = rational_constant_value(ctx, *exp)?;
            let negative_two = BigRational::from_integer((-2).into());
            if !exponent.is_integer() || (exponent < BigRational::one() && exponent > negative_two)
            {
                return None;
            }
            (*base, exponent)
        }
        _ => (reciprocal_den, BigRational::one()),
    };

    if !contains_named_var(ctx, base, var) || Polynomial::from_expr(ctx, base, var).is_err() {
        return None;
    }

    Some((base, exponent, scale))
}

fn reciprocal_quotient_denominator_power_substitution_target_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (base, exponent, _) =
        reciprocal_quotient_polynomial_denominator_power_parts_view(ctx, *den, var)?;

    let Ok(numerator) = Polynomial::from_expr(ctx, *num, var) else {
        return None;
    };
    let Ok(base_poly) = Polynomial::from_expr(ctx, base, var) else {
        return None;
    };
    let derivative = base_poly.derivative();
    if constant_polynomial_ratio(&numerator, &derivative).is_none_or(|scale| scale.is_zero()) {
        return None;
    }

    Some((base, exponent))
}

pub fn integrate_symbolic_is_bounded_reciprocal_quotient_denominator_power_substitution_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    max_abs_power: i64,
) -> bool {
    let Some((_, exponent)) =
        reciprocal_quotient_denominator_power_substitution_target_parts(ctx, expr, var)
    else {
        return false;
    };
    let bound = BigRational::from_integer(max_abs_power.into());
    exponent >= -bound.clone() && exponent <= bound
}

pub fn integrate_symbolic_is_reciprocal_negative_power_denominator_quotient_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Some((_, exponent)) =
        reciprocal_quotient_denominator_power_substitution_target_parts(ctx, expr, var)
    else {
        return false;
    };
    exponent < BigRational::zero()
}

fn reciprocal_quotient_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    reciprocal_quotient_polynomial_denominator_power_parts_view(ctx, den, var)
}

fn expanded_square_denominator_base(ctx: &mut Context, den: ExprId, var: &str) -> Option<ExprId> {
    Polynomial::from_expr(ctx, den, var).ok()?;

    let sqrt_den = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
    let rewrite = try_rewrite_simplify_square_root_expr(ctx, sqrt_den)?;
    if rewrite.kind != SimplifySquareRootRewriteKind::PerfectSquare {
        return None;
    }

    let base = extract_abs_argument_view(ctx, rewrite.rewritten).unwrap_or(rewrite.rewritten);
    if !contains_named_var(ctx, base, var) {
        return None;
    }

    Some(base)
}

fn expanded_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    const MAX_EXPANDED_DENOMINATOR_POWER: usize = 5;

    let den_poly = Polynomial::from_expr(ctx, den, var).ok()?;
    if den_poly.degree() < 2 {
        return None;
    }

    let derivative = den_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let repeated_factor = den_poly.gcd(&derivative);
    if repeated_factor.is_zero() || repeated_factor.degree() == 0 {
        return None;
    }

    let (mut base_poly, remainder) = den_poly.div_rem(&repeated_factor).ok()?;
    if !remainder.is_zero() || base_poly.is_zero() || base_poly.degree() == 0 {
        return None;
    }

    let base_lc = base_poly.leading_coeff();
    if base_lc.is_zero() {
        return None;
    }
    base_poly = base_poly.div_scalar(&base_lc);

    let base_degree = base_poly.degree();
    if den_poly.degree() % base_degree != 0 {
        return None;
    }

    let exponent = den_poly.degree() / base_degree;
    if !(2..=MAX_EXPANDED_DENOMINATOR_POWER).contains(&exponent) {
        return None;
    }

    let mut reconstructed = Polynomial::one(var.to_string());
    for _ in 0..exponent {
        reconstructed = reconstructed.mul(&base_poly);
    }
    let denominator_scale = constant_polynomial_ratio(&den_poly, &reconstructed)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let base = base_poly.to_expr(ctx);
    if !contains_named_var(ctx, base, var) {
        return None;
    }

    Some((
        base,
        BigRational::from_integer((exponent as i64).into()),
        denominator_scale,
    ))
}

fn polynomial_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (base, denominator_exponent, denominator_scale) =
        polynomial_denominator_power_parts(ctx, den, var)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let adjusted_num = if denominator_scale.is_one() {
        num
    } else {
        let reciprocal_scale = BigRational::one() / denominator_scale;
        let reciprocal_scale = ctx.add(Expr::Number(reciprocal_scale));
        mul2_raw(ctx, reciprocal_scale, num)
    };

    polynomial_power_substitution_from_base(ctx, adjusted_num, base, -denominator_exponent, var)
}

fn polynomial_negative_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (base, numerator_exponent, denominator_scale) =
        negative_syntactic_polynomial_denominator_power_parts(ctx, den, var)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let adjusted_num = if denominator_scale.is_one() {
        num
    } else {
        let reciprocal_scale = BigRational::one() / denominator_scale;
        let reciprocal_scale = ctx.add(Expr::Number(reciprocal_scale));
        mul2_raw(ctx, reciprocal_scale, num)
    };

    polynomial_power_substitution_from_base(ctx, adjusted_num, base, numerator_exponent, var)
}

fn polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (base, numerator_exponent, denominator_scale) =
        reciprocal_quotient_polynomial_denominator_power_parts(ctx, den, var)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let adjusted_num = if denominator_scale.is_one() {
        num
    } else {
        let reciprocal_scale = BigRational::one() / denominator_scale;
        let reciprocal_scale = ctx.add(Expr::Number(reciprocal_scale));
        mul2_raw(ctx, reciprocal_scale, num)
    };

    polynomial_power_substitution_from_base(ctx, adjusted_num, base, numerator_exponent, var)
}

fn constant_scaled_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    scale: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let scale = rational_constant_value(ctx, scale)?;
    if scale.is_zero() {
        return None;
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (num, den) = (*num, *den);
    let scale_expr = ctx.add(Expr::Number(scale));
    let scaled_num = mul2_raw(ctx, scale_expr, num);

    polynomial_denominator_power_substitution_antiderivative(ctx, scaled_num, den, var)
        .or_else(|| {
            polynomial_negative_denominator_power_substitution_antiderivative(
                ctx, scaled_num, den, var,
            )
        })
        .or_else(|| {
            polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
                ctx, scaled_num, den, var,
            )
        })
}

fn polynomial_denominator_power_substitution_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (base, _, _) = polynomial_denominator_power_parts(ctx, den, var)?;
    polynomial_denominator_power_substitution_antiderivative(ctx, num, den, var)?;
    Some(base)
}

fn polynomial_negative_denominator_power_substitution_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (base, _, _) = negative_syntactic_polynomial_denominator_power_parts(ctx, den, var)?;
    polynomial_negative_denominator_power_substitution_antiderivative(ctx, num, den, var)?;
    Some(base)
}

fn polynomial_reciprocal_quotient_denominator_power_substitution_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (base, _, _) = reciprocal_quotient_polynomial_denominator_power_parts(ctx, den, var)?;
    polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
        ctx, num, den, var,
    )?;
    Some(base)
}

fn polynomial_log_product_substitution_from_base(
    ctx: &mut Context,
    cofactor: ExprId,
    base: ExprId,
    log_factor: ExprId,
    var: &str,
) -> Option<ExprId> {
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let base_poly = Polynomial::from_expr(ctx, base, var).ok()?;
    if base_poly.degree() == 0 {
        return None;
    }

    let derivative = base_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let one = ctx.num(1);
    let log_minus_one = ctx.add(Expr::Sub(log_factor, one));
    let integral = mul2_raw(ctx, base, log_minus_one);
    if scale.is_one() {
        return Some(integral);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

fn polynomial_log_product_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (log_index, factor) in factors.iter().enumerate() {
        let Expr::Function(fn_id, args) = ctx.get(*factor) else {
            continue;
        };
        if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) {
            continue;
        }
        let base = args[0];
        if !contains_named_var(ctx, base, var) {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };

        if let Some(integral) =
            polynomial_log_product_substitution_from_base(ctx, cofactor, base, *factor, var)
        {
            return Some(integral);
        }
    }

    None
}

fn positive_integer_power_value(ctx: &Context, expr: ExprId) -> Option<u32> {
    match ctx.get(expr) {
        Expr::Number(n) if n.denom().is_one() && n.is_positive() => n.to_integer().to_u32(),
        _ => None,
    }
}

fn natural_log_power_factor_parts(ctx: &Context, factor: ExprId) -> Option<(ExprId, ExprId, u32)> {
    let (log_expr, power) = match ctx.get(factor) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) =>
        {
            (factor, 1)
        }
        Expr::Pow(base, exp) => {
            let power = positive_integer_power_value(ctx, *exp)?;
            let Expr::Function(fn_id, args) = ctx.get(*base) else {
                return None;
            };
            if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) {
                return None;
            }
            (*base, power)
        }
        _ => return None,
    };

    let Expr::Function(_, args) = ctx.get(log_expr) else {
        return None;
    };
    let log_arg = args[0];
    let log_base = extract_abs_argument_view(ctx, log_arg).unwrap_or(log_arg);
    Some((log_expr, log_base, power))
}

fn natural_log_argument(ctx: &Context, log_expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(log_expr) else {
        return None;
    };
    if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) {
        Some(args[0])
    } else {
        None
    }
}

fn log_square_by_parts_integral(ctx: &mut Context, base: ExprId, log_expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    let log_square = ctx.add(Expr::Pow(log_expr, two));
    let two_log = mul2_raw(ctx, two, log_expr);
    let trailing = ctx.add(Expr::Sub(log_square, two_log));
    let two = ctx.num(2);
    let by_parts_factor = ctx.add(Expr::Add(trailing, two));
    mul2_raw(ctx, base, by_parts_factor)
}

fn log_cube_by_parts_integral(ctx: &mut Context, base: ExprId, log_expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    let three = ctx.num(3);
    let six = ctx.num(6);
    let log_square = ctx.add(Expr::Pow(log_expr, two));
    let log_cube = ctx.add(Expr::Pow(log_expr, three));

    let three_log_square = mul2_raw(ctx, three, log_square);
    let six_log = mul2_raw(ctx, six, log_expr);
    let head = ctx.add(Expr::Sub(log_cube, three_log_square));
    let tail = ctx.add(Expr::Sub(six_log, six));
    let by_parts_factor = ctx.add(Expr::Add(head, tail));
    mul2_raw(ctx, base, by_parts_factor)
}

fn build_polynomial_log_power_product_substitution_integral(
    ctx: &mut Context,
    log_expr: ExprId,
    log_base: ExprId,
    power: u32,
    cofactor: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !matches!(power, 2 | 3) {
        return None;
    }
    let log_arg = natural_log_argument(ctx, log_expr)?;
    if extract_abs_argument_view(ctx, log_arg).is_some() {
        return None;
    }

    let base_poly = Polynomial::from_expr(ctx, log_base, var).ok()?;
    if base_poly.degree() == 0 {
        return None;
    }
    let derivative = base_poly.derivative();

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let base = base_poly.to_expr(ctx);
    let integral = match power {
        2 => log_square_by_parts_integral(ctx, base, log_expr),
        3 => log_cube_by_parts_integral(ctx, base, log_expr),
        _ => return None,
    };
    if scale.is_one() {
        return Some(integral);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

fn polynomial_log_power_product_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.is_empty() {
        return None;
    }

    for (log_index, factor) in factors.iter().enumerate() {
        let Some((log_expr, log_base, power)) = natural_log_power_factor_parts(ctx, *factor) else {
            continue;
        };

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };

        if let Some(integral) = build_polynomial_log_power_product_substitution_integral(
            ctx, log_expr, log_base, power, cofactor, var,
        ) {
            return Some(integral);
        }
    }

    let (log_expr, log_base, power, cofactor) = additive_common_log_power_cofactor(ctx, expr)?;
    build_polynomial_log_power_product_substitution_integral(
        ctx, log_expr, log_base, power, cofactor, var,
    )
}

fn polynomial_log_product_substitution_power(ctx: &mut Context, expr: ExprId) -> Option<u32> {
    let factors = mul_leaves(ctx, expr);
    for factor in factors {
        if let Some((_, _, power)) = natural_log_power_factor_parts(ctx, factor) {
            return Some(power);
        }
    }

    additive_common_log_power_cofactor(ctx, expr).map(|(_, _, power, _)| power)
}

pub fn integrate_symbolic_is_log_cube_product_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if polynomial_log_product_substitution_power(ctx, expr) != Some(3) {
        return false;
    }

    polynomial_log_power_product_substitution_antiderivative(ctx, expr, var).is_some()
}

fn signed_term(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn additive_common_log_power_cofactor(
    ctx: &mut Context,
    num: ExprId,
) -> Option<(ExprId, ExprId, u32, ExprId)> {
    let add_view = AddView::from_expr(ctx, num);
    if add_view.terms.len() < 2 {
        return None;
    }

    let mut common: Option<(ExprId, ExprId, u32)> = None;
    let mut cofactor_terms = Vec::with_capacity(add_view.terms.len());

    for (term, sign) in add_view.terms {
        let factors = mul_leaves(ctx, term);
        let mut term_cofactor = None;

        for (log_index, factor) in factors.iter().enumerate() {
            let Some((log_expr, log_base, power)) = natural_log_power_factor_parts(ctx, *factor)
            else {
                continue;
            };

            if let Some((common_log, _, common_power)) = common {
                if power != common_power
                    || compare_expr(ctx, log_expr, common_log) != Ordering::Equal
                {
                    continue;
                }
            } else {
                common = Some((log_expr, log_base, power));
            }

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            term_cofactor = Some(signed_term(ctx, cofactor, sign));
            break;
        }

        cofactor_terms.push(term_cofactor?);
    }

    let (log_expr, log_base, power) = common?;
    let cofactor = build_balanced_add(ctx, &cofactor_terms);
    Some((log_expr, log_base, power, cofactor))
}

fn build_polynomial_log_derivative_power_integral(
    ctx: &mut Context,
    denominator: &Polynomial,
    log_expr: ExprId,
    log_base: ExprId,
    power: u32,
    cofactor: ExprId,
    var: &str,
) -> Option<ExprId> {
    let log_base_poly = Polynomial::from_expr(ctx, log_base, var).ok()?;
    if log_base_poly.degree() == 0 {
        return None;
    }
    let denominator_scale = constant_polynomial_ratio(denominator, &log_base_poly)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let derivative = denominator.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let next_power = BigRational::from_integer((power + 1).into());
    let next_power_expr = ctx.add(Expr::Number(next_power.clone()));
    let log_power = ctx.add(Expr::Pow(log_expr, next_power_expr));
    let scaled = scale / next_power;
    if scaled.is_one() {
        return Some(log_power);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, log_power))
}

fn polynomial_log_derivative_power_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    if denominator.degree() == 0 {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    for (log_index, factor) in factors.iter().enumerate() {
        let Some((log_expr, log_base, power)) = natural_log_power_factor_parts(ctx, *factor) else {
            continue;
        };

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };

        if let Some(integral) = build_polynomial_log_derivative_power_integral(
            ctx,
            &denominator,
            log_expr,
            log_base,
            power,
            cofactor,
            var,
        ) {
            return Some(integral);
        }
    }

    let (log_expr, log_base, power, cofactor) = additive_common_log_power_cofactor(ctx, num)?;
    build_polynomial_log_derivative_power_integral(
        ctx,
        &denominator,
        log_expr,
        log_base,
        power,
        cofactor,
        var,
    )
}

fn exp_like_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

fn linear_times_exp_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (exp_index, factor) in factors.iter().enumerate() {
        let Some(exp_arg) = exp_like_arg(ctx, *factor) else {
            continue;
        };

        let arg_poly = Polynomial::from_expr(ctx, exp_arg, var).ok()?;
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != exp_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() != 1 {
            continue;
        }

        let cofactor_slope = cofactor_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if cofactor_slope.is_zero() {
            continue;
        }

        let arg_slope_expr = ctx.add(Expr::Number(arg_slope.clone()));
        let quotient = if arg_slope.is_one() {
            cofactor
        } else {
            ctx.add(Expr::Div(cofactor, arg_slope_expr))
        };
        let correction = cofactor_slope / (arg_slope.clone() * arg_slope);
        let correction_expr = ctx.add(Expr::Number(correction));
        let inner = ctx.add(Expr::Sub(quotient, correction_expr));
        return Some(mul2_raw(ctx, *factor, inner));
    }

    None
}

fn trig_like_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
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

fn hyperbolic_like_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    match ctx.builtin_of(*fn_id) {
        Some(builtin @ (BuiltinFn::Sinh | BuiltinFn::Cosh)) => Some((builtin, args[0])),
        _ => None,
    }
}

fn scale_factor(ctx: &mut Context, scale: BigRational, expr: ExprId) -> ExprId {
    if scale.is_one() {
        return expr;
    }
    let scale_expr = ctx.add(Expr::Number(scale));
    mul2_raw(ctx, scale_expr, expr)
}

fn linear_times_trig_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (trig_index, factor) in factors.iter().enumerate() {
        let Some((builtin, trig_arg)) = trig_like_factor(ctx, *factor) else {
            continue;
        };

        let Ok(arg_poly) = Polynomial::from_expr(ctx, trig_arg, var) else {
            continue;
        };
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };
        let Ok(cofactor_poly) = Polynomial::from_expr(ctx, cofactor, var) else {
            continue;
        };
        if cofactor_poly.degree() != 1 {
            continue;
        }

        let cofactor_slope = cofactor_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if cofactor_slope.is_zero() {
            continue;
        }

        let arg_slope_expr = ctx.add(Expr::Number(arg_slope.clone()));
        let quotient = if arg_slope.is_one() {
            cofactor
        } else {
            ctx.add(Expr::Div(cofactor, arg_slope_expr))
        };
        let correction = cofactor_slope / (arg_slope.clone() * arg_slope);

        return match builtin {
            BuiltinFn::Sin => {
                let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![trig_arg]);
                let quotient_cos = mul2_raw(ctx, quotient, cos_arg);
                let correction_sin = scale_factor(ctx, correction, *factor);
                Some(ctx.add(Expr::Sub(correction_sin, quotient_cos)))
            }
            BuiltinFn::Cos => {
                let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![trig_arg]);
                let quotient_sin = mul2_raw(ctx, quotient, sin_arg);
                let correction_cos = scale_factor(ctx, correction, *factor);
                Some(ctx.add(Expr::Add(quotient_sin, correction_cos)))
            }
            _ => None,
        };
    }

    None
}

fn linear_times_hyperbolic_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (hyperbolic_index, factor) in factors.iter().enumerate() {
        let Some((builtin, hyperbolic_arg)) = hyperbolic_like_factor(ctx, *factor) else {
            continue;
        };

        let Ok(arg_poly) = Polynomial::from_expr(ctx, hyperbolic_arg, var) else {
            continue;
        };
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != hyperbolic_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };
        let Ok(cofactor_poly) = Polynomial::from_expr(ctx, cofactor, var) else {
            continue;
        };
        if cofactor_poly.degree() != 1 {
            continue;
        }

        let cofactor_slope = cofactor_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if cofactor_slope.is_zero() {
            continue;
        }

        let arg_slope_expr = ctx.add(Expr::Number(arg_slope.clone()));
        let quotient = if arg_slope.is_one() {
            cofactor
        } else {
            ctx.add(Expr::Div(cofactor, arg_slope_expr))
        };
        let correction = cofactor_slope / (arg_slope.clone() * arg_slope);

        return match builtin {
            BuiltinFn::Sinh => {
                let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_arg]);
                let quotient_cosh = mul2_raw(ctx, quotient, cosh_arg);
                let correction_sinh = scale_factor(ctx, correction, *factor);
                Some(ctx.add(Expr::Sub(quotient_cosh, correction_sinh)))
            }
            BuiltinFn::Cosh => {
                let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_arg]);
                let quotient_sinh = mul2_raw(ctx, quotient, sinh_arg);
                let correction_cosh = scale_factor(ctx, correction, *factor);
                Some(ctx.add(Expr::Sub(quotient_sinh, correction_cosh)))
            }
            _ => None,
        };
    }

    None
}

fn exact_rational_sqrt(value: &BigRational) -> Option<BigRational> {
    if value < &BigRational::zero() {
        return None;
    }

    let sqrt_num = value.numer().sqrt();
    let sqrt_den = value.denom().sqrt();
    if &sqrt_num * &sqrt_num == value.numer().clone()
        && &sqrt_den * &sqrt_den == value.denom().clone()
    {
        Some(BigRational::new(sqrt_num, sqrt_den))
    } else {
        None
    }
}

fn positive_rational_sqrt_expr(ctx: &mut Context, value: &BigRational) -> Option<ExprId> {
    if value <= &BigRational::zero() {
        return None;
    }

    if let Some(root) = exact_rational_sqrt(value) {
        return Some(ctx.add(Expr::Number(root)));
    }

    let radicand = ctx.add(Expr::Number(value.clone()));
    Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]))
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn square_factor_root_i64(mut n: i64) -> i64 {
    let mut root = 1_i64;
    let mut p = 2_i64;
    while p <= n / p {
        let mut exponent = 0;
        while n % p == 0 {
            n /= p;
            exponent += 1;
        }
        for _ in 0..(exponent / 2) {
            root *= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    root
}

fn integer_polynomial_content(poly: &Polynomial) -> Option<i64> {
    let mut content = 0_i64;
    for coeff in &poly.coeffs {
        if coeff.is_zero() {
            continue;
        }
        if !coeff.denom().is_one() {
            return None;
        }
        let value = coeff.to_integer().to_i64()?.checked_abs()?;
        content = if content == 0 {
            value
        } else {
            gcd_i64(content, value)
        };
    }
    (content > 1).then_some(content)
}

fn reduce_surd_offset_by_common_square_factor(
    arg_poly: &Polynomial,
    offset_square: &BigRational,
) -> Option<(Polynomial, BigRational, BigRational)> {
    if !offset_square.is_integer() || offset_square <= &BigRational::zero() {
        return None;
    }

    let offset_integer = offset_square.to_integer().to_i64()?;
    let square_root_factor = square_factor_root_i64(offset_integer);
    if square_root_factor <= 1 {
        return None;
    }

    let arg_content = integer_polynomial_content(arg_poly)?;
    let common_factor = gcd_i64(arg_content, square_root_factor);
    if common_factor <= 1 {
        return None;
    }

    let common = BigRational::from_integer(common_factor.into());
    let reduced_arg = arg_poly.div_scalar(&common);
    let reduced_offset_square = offset_square / (&common * &common);
    Some((reduced_arg, reduced_offset_square, common))
}

fn constant_polynomial_value(poly: &Polynomial) -> Option<BigRational> {
    if poly.is_zero() {
        return Some(BigRational::zero());
    }
    if poly.degree() != 0 {
        return None;
    }

    poly.coeffs.first().cloned()
}

fn exact_polynomial_square_plus_positive_constant(
    poly: &Polynomial,
) -> Option<(Polynomial, BigRational)> {
    if poly.is_zero() {
        return None;
    }

    let degree = poly.degree();
    if degree == 0 || !degree.is_multiple_of(2) {
        return None;
    }

    let root_degree = degree / 2;
    let leading = poly
        .coeffs
        .get(degree)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let leading_root = exact_rational_sqrt(&leading)?;
    if leading_root.is_zero() {
        return None;
    }

    let mut root_coeffs = vec![BigRational::zero(); root_degree + 1];
    root_coeffs[root_degree] = leading_root.clone();
    let two = BigRational::from_integer(2.into());

    for k in (0..root_degree).rev() {
        let target_degree = root_degree + k;
        let target = poly
            .coeffs
            .get(target_degree)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let mut known = BigRational::zero();

        for i in 0..=root_degree {
            if let Some(j) = target_degree.checked_sub(i) {
                if j <= root_degree && i != k && j != k {
                    known += root_coeffs[i].clone() * root_coeffs[j].clone();
                }
            }
        }

        root_coeffs[k] = (target - known) / (two.clone() * leading_root.clone());
    }

    let root = Polynomial::new(root_coeffs, poly.var.clone());
    let square = root.mul(&root);
    let len = poly.coeffs.len().max(square.coeffs.len());

    for idx in 1..len {
        let left = poly
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = square
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if left != right {
            return None;
        }
    }

    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        - square
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
    if constant > BigRational::zero() {
        Some((root, constant))
    } else {
        None
    }
}

fn exact_positive_constant_minus_polynomial_square(
    poly: &Polynomial,
) -> Option<(Polynomial, BigRational)> {
    if poly.is_zero() {
        return None;
    }

    let degree = poly.degree();
    if degree == 0 || !degree.is_multiple_of(2) {
        return None;
    }

    let root_degree = degree / 2;
    let leading = poly
        .coeffs
        .get(degree)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let leading_root = exact_rational_sqrt(&(-leading))?;
    if leading_root.is_zero() {
        return None;
    }

    let mut root_coeffs = vec![BigRational::zero(); root_degree + 1];
    root_coeffs[root_degree] = leading_root.clone();
    let two = BigRational::from_integer(2.into());

    for k in (0..root_degree).rev() {
        let target_degree = root_degree + k;
        let target = -poly
            .coeffs
            .get(target_degree)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let mut known = BigRational::zero();

        for i in 0..=root_degree {
            if let Some(j) = target_degree.checked_sub(i) {
                if j <= root_degree && i != k && j != k {
                    known += root_coeffs[i].clone() * root_coeffs[j].clone();
                }
            }
        }

        root_coeffs[k] = (target - known) / (two.clone() * leading_root.clone());
    }

    let root = Polynomial::new(root_coeffs, poly.var.clone());
    let square = root.mul(&root);
    let len = poly.coeffs.len().max(square.coeffs.len());

    for idx in 1..len {
        let left = poly
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = square
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if left != -right {
            return None;
        }
    }

    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        + square
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
    if constant > BigRational::zero() {
        Some((root, constant))
    } else {
        None
    }
}

fn polynomial_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (kernel_index, kernel, kernel_arg) =
        factors.iter().enumerate().find_map(|(idx, factor)| {
            polynomial_substitution_kernel(ctx, *factor).map(|(kernel, arg)| (idx, kernel, arg))
        })?;

    if !contains_named_var(ctx, kernel_arg, var) {
        return None;
    }

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        if matches!(
            kernel,
            PolynomialSubstitutionKernel::Tan | PolynomialSubstitutionKernel::Cot
        ) {
            return None;
        }
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, kernel_arg, var).ok()?;
    let derivative_poly = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative_poly)?;
    if scale.is_zero() {
        return None;
    }

    let antiderivative = polynomial_substitution_kernel_antiderivative(
        ctx,
        kernel,
        kernel_arg,
        factors[kernel_index],
    );
    if scale.is_one() {
        return Some(antiderivative);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, antiderivative))
}

fn has_trig_polynomial_substitution_kernel(ctx: &Context, expr: ExprId, var: &str) -> bool {
    mul_leaves(ctx, expr).into_iter().any(|factor| {
        let Some((kernel, arg)) = polynomial_substitution_kernel(ctx, factor) else {
            return false;
        };
        matches!(
            kernel,
            PolynomialSubstitutionKernel::Sin
                | PolynomialSubstitutionKernel::Cos
                | PolynomialSubstitutionKernel::Tan
                | PolynomialSubstitutionKernel::Cot
        ) && contains_named_var(ctx, arg, var)
    })
}

pub fn integrate_symbolic_is_trig_polynomial_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if additive_common_trig_polynomial_substitution_antiderivative(ctx, expr, var).is_some() {
        return true;
    }

    if has_trig_polynomial_substitution_kernel(ctx, expr, var)
        && polynomial_substitution_antiderivative(ctx, expr, var).is_some()
    {
        return true;
    }

    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    trig_log_derivative_ratio_scale(ctx, num, den, var).is_some()
}

fn additive_common_trig_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let add_view = AddView::from_expr(ctx, expr);
    if add_view.terms.len() < 2 {
        return None;
    }

    let mut common: Option<(PolynomialSubstitutionKernel, ExprId, ExprId)> = None;
    let mut cofactor_terms = Vec::with_capacity(add_view.terms.len());

    for (term, sign) in add_view.terms {
        let factors = mul_leaves(ctx, term);
        let mut term_cofactor = None;

        for (kernel_index, factor) in factors.iter().enumerate() {
            let Some((kernel, arg)) = polynomial_substitution_kernel(ctx, *factor) else {
                continue;
            };
            if !matches!(
                kernel,
                PolynomialSubstitutionKernel::Sin | PolynomialSubstitutionKernel::Cos
            ) {
                continue;
            }

            if let Some((common_kernel, common_arg, _)) = common {
                if kernel != common_kernel || compare_expr(ctx, arg, common_arg) != Ordering::Equal
                {
                    continue;
                }
            } else {
                common = Some((kernel, arg, *factor));
            }

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            term_cofactor = Some(signed_term(ctx, cofactor, sign));
            break;
        }

        cofactor_terms.push(term_cofactor?);
    }

    let (kernel, arg, kernel_factor) = common?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = build_balanced_add(ctx, &cofactor_terms);
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative_poly = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative_poly)?;
    if scale.is_zero() {
        return None;
    }

    let antiderivative =
        polynomial_substitution_kernel_antiderivative(ctx, kernel, arg, kernel_factor);
    if scale.is_one() {
        return Some(antiderivative);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, antiderivative))
}

fn arctan_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let Some((arg_poly, offset_square)) =
        exact_polynomial_square_plus_positive_constant(&denominator)
    else {
        return arctan_scaled_quadratic_antiderivative(ctx, &numerator, &denominator);
    };
    let Some(offset) = exact_rational_sqrt(&offset_square) else {
        return arctan_surd_offset_antiderivative(ctx, &numerator, &arg_poly, &offset_square);
    };
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let arctan_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset.clone()));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    let scaled = scale / offset;
    if scaled.is_one() {
        return Some(arctan);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, arctan))
}

fn arctan_surd_offset_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    arg_poly: &Polynomial,
    offset_square: &BigRational,
) -> Option<ExprId> {
    let offset_expr = positive_rational_sqrt_expr(ctx, offset_square)?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let arctan_arg = ctx.add(Expr::Div(arg, offset_expr));
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    let numerator = if scale.is_one() {
        arctan
    } else {
        let scale_num = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_num, arctan)
    };
    Some(ctx.add(Expr::Div(numerator, offset_expr)))
}

fn arctan_scaled_quadratic_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<ExprId> {
    if denominator.degree() != 2 {
        return None;
    }

    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return None;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let discriminant = four.clone() * a.clone() * c - b.clone() * b.clone();
    if discriminant <= BigRational::zero() {
        return None;
    }

    let arg_poly = Polynomial::new(vec![b, two * a.clone()], denominator.var.clone());
    let Some(offset) = exact_rational_sqrt(&discriminant) else {
        return arctan_scaled_quadratic_surd_antiderivative(
            ctx,
            numerator,
            &arg_poly,
            &discriminant,
        );
    };
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let denominator_scale = offset.clone() / (four * a);
    let scaled_derivative = Polynomial::new(
        derivative
            .coeffs
            .iter()
            .map(|coeff| coeff.clone() * denominator_scale.clone())
            .collect(),
        denominator.var.clone(),
    );
    let scale = constant_polynomial_ratio(numerator, &scaled_derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg_over_offset = Polynomial::new(
        arg_poly
            .coeffs
            .iter()
            .map(|coeff| coeff.clone() / offset.clone())
            .collect(),
        denominator.var.clone(),
    )
    .to_expr(ctx);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg_over_offset]);
    if scale.is_one() {
        return Some(arctan);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, arctan))
}

fn arctan_scaled_quadratic_surd_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    arg_poly: &Polynomial,
    discriminant: &BigRational,
) -> Option<ExprId> {
    let numerator_constant = constant_polynomial_value(numerator)?;
    if numerator_constant.is_zero() {
        return None;
    }

    let mut scale = BigRational::from_integer(2.into()) * numerator_constant;
    let (arg_poly, offset_square) =
        if let Some((reduced_arg, reduced_offset_square, common_factor)) =
            reduce_surd_offset_by_common_square_factor(arg_poly, discriminant)
        {
            scale /= common_factor;
            (reduced_arg, reduced_offset_square)
        } else {
            (arg_poly.clone(), discriminant.clone())
        };

    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;
    let arg = arg_poly.to_expr(ctx);
    let arctan_arg = ctx.add(Expr::Div(arg, offset_expr));
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);

    let numerator = if scale.is_one() {
        arctan
    } else {
        let scale_num = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_num, arctan)
    };
    Some(ctx.add(Expr::Div(numerator, offset_expr)))
}

fn atanh_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&denominator)?;
    let Some(offset) = exact_rational_sqrt(&offset_square) else {
        return atanh_surd_offset_antiderivative(ctx, &numerator, &arg_poly, &offset_square);
    };
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let atanh_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset.clone()));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let atanh = ctx.call_builtin(BuiltinFn::Atanh, vec![atanh_arg]);
    let scaled = scale / offset;
    if scaled.is_one() {
        return Some(atanh);
    }
    if scaled == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(atanh)));
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, atanh))
}

fn atanh_surd_offset_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    arg_poly: &Polynomial,
    offset_square: &BigRational,
) -> Option<ExprId> {
    let derivative = arg_poly.derivative();
    let mut scale = constant_polynomial_ratio(numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let (arg_poly, offset_square) =
        if let Some((reduced_arg, reduced_offset_square, common_factor)) =
            reduce_surd_offset_by_common_square_factor(arg_poly, offset_square)
        {
            scale /= common_factor;
            (reduced_arg, reduced_offset_square)
        } else {
            (arg_poly.clone(), offset_square.clone())
        };

    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;
    let arg = arg_poly.to_expr(ctx);
    let atanh_arg = ctx.add(Expr::Div(arg, offset_expr));
    let atanh = ctx.call_builtin(BuiltinFn::Atanh, vec![atanh_arg]);
    let numerator = if scale.is_one() {
        atanh
    } else {
        let scale_num = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_num, atanh)
    };
    Some(ctx.add(Expr::Div(numerator, offset_expr)))
}

fn polynomial_square_minus_constant_log_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) =
        exact_positive_constant_minus_polynomial_square(&denominator.neg())?;
    let offset = exact_rational_sqrt(&offset_square)?;
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let offset_expr = ctx.add(Expr::Number(offset.clone()));
    let numerator_arg = ctx.add(Expr::Sub(arg, offset_expr));
    let denominator_arg = ctx.add(Expr::Add(arg, offset_expr));
    let ratio = ctx.add(Expr::Div(numerator_arg, denominator_arg));
    let log_abs_ratio = ln_abs(ctx, ratio);

    let two = BigRational::from_integer(2.into());
    let scaled = scale / (two * offset);
    if scaled.is_one() {
        return Some(log_abs_ratio);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, log_abs_ratio))
}

fn polynomial_log_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    if denominator.degree() == 0 {
        return None;
    }

    let derivative = denominator.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let log_abs_den = ln_abs(ctx, den);
    if scale.is_one() {
        return Some(log_abs_den);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, log_abs_den))
}

fn arcsin_polynomial_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&radicand)?;
    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let raw_arg = arg_poly.to_expr(ctx);
    let arg = compact_single_power_polynomial_arg(ctx, raw_arg);
    let arcsin_arg = if offset_square.is_one() {
        arg
    } else {
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let arcsin = ctx.call_builtin(BuiltinFn::Arcsin, vec![arcsin_arg]);
    if scale.is_one() {
        return Some(arcsin);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, arcsin))
}

fn asinh_polynomial_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let (arg_poly, offset_square) = exact_polynomial_square_plus_positive_constant(&radicand)?;
    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let raw_arg = arg_poly.to_expr(ctx);
    let arg = compact_single_power_polynomial_arg(ctx, raw_arg);
    let asinh_arg = if offset_square.is_one() {
        arg
    } else {
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let asinh = ctx.call_builtin(BuiltinFn::Asinh, vec![asinh_arg]);
    if scale.is_one() {
        return Some(asinh);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, asinh))
}

fn sqrt_derivative_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let derivative = radicand_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let two = BigRational::from_integer(2.into());
    let scaled = scale * two;
    if scaled.is_one() {
        return Some(sqrt_radicand);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, sqrt_radicand))
}

fn sqrt_product_derivative_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let derivative = radicand_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let half_exp = ctx.rational(3, 2);
    let power = ctx.add(Expr::Pow(radicand, half_exp));
    let two = BigRational::from_integer(2.into());
    let three_r = BigRational::from_integer(3.into());
    let scaled = scale * two / three_r;
    if scaled.is_one() {
        return Some(power);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, power))
}

fn sqrt_derivative_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        return sqrt_derivative_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn sqrt_derivative_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn sqrt_product_derivative_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    sqrt_product_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn arcsin_polynomial_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        return arcsin_polynomial_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn asinh_polynomial_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        return asinh_polynomial_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    asinh_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn arcsin_polynomial_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn asinh_polynomial_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    asinh_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn sqrt_derivative_substitution_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            if let Some(radicand) = sqrt_like_radicand(ctx, den) {
                sqrt_derivative_substitution_from_radicand(ctx, num, radicand, var)?;
                return Some(radicand);
            }

            let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
            let factors = mul_leaves(ctx, num);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                let radicand = sqrt_like_radicand(ctx, *factor)?;
                let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
                (radicand_poly == denominator).then_some((idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        _ => None,
    }
}

fn sqrt_trig_reciprocal_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (_, _, radicand, _) = sqrt_trig_reciprocal_derivative_parts(ctx, expr, var)?;
    Some(radicand)
}

fn sqrt_trig_log_derivative_radicand(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let (_, _, radicand, _) = sqrt_trig_log_derivative_parts(ctx, expr, var)?;
    Some(radicand)
}

fn sqrt_hyperbolic_log_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (_, radicand, _) = sqrt_hyperbolic_log_derivative_parts(ctx, expr, var)?;
    Some(radicand)
}

fn sqrt_hyperbolic_reciprocal_square_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (_, radicand, _) = sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var)?;
    Some(radicand)
}

fn sqrt_hyperbolic_reciprocal_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (_, radicand, _) = sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var)?;
    Some(radicand)
}

fn arcsin_polynomial_substitution_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            if let Some(radicand) = sqrt_like_radicand(ctx, den) {
                arcsin_polynomial_substitution_from_radicand(ctx, num, radicand, var)?;
                return Some(radicand);
            }

            let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
            let factors = mul_leaves(ctx, num);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                let radicand = sqrt_like_radicand(ctx, *factor)?;
                let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
                (radicand_poly == denominator).then_some((idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        _ => None,
    }
}

fn atanh_polynomial_substitution_denominator(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    atanh_polynomial_substitution_antiderivative(ctx, num, den, var)?;
    Some(den)
}

fn atanh_polynomial_substitution_target_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<()> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&denominator)?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() || offset_square.is_zero() {
        return None;
    }

    Some(())
}

pub fn integrate_symbolic_is_atanh_polynomial_substitution_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_is_atanh_polynomial_substitution_target(ctx, inner, var);
    }

    atanh_polynomial_substitution_target_parts(ctx, expr, var).is_some()
}

fn constant_scaled_integrand_inner(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Mul(left, right) => {
            let left_depends_on_var = contains_named_var(ctx, *left, var);
            let right_depends_on_var = contains_named_var(ctx, *right, var);
            match (left_depends_on_var, right_depends_on_var) {
                (false, true) => Some(*right),
                (true, false) => Some(*left),
                _ => None,
            }
        }
        _ => None,
    }
}

pub fn integrate_symbolic_required_nonzero_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_required_nonzero_conditions(ctx, inner, var);
    }

    let conditions: Vec<_> = trig_log_required_nonzero(ctx, expr, var)
        .into_iter()
        .chain(polynomial_trig_log_required_nonzero(ctx, expr, var))
        .chain(reciprocal_trig_log_required_nonzero(ctx, expr, var))
        .chain(trig_log_derivative_ratio_required_nonzero(ctx, expr, var))
        .chain(trig_reciprocal_derivative_required_nonzero(ctx, expr, var))
        .chain(polynomial_trig_reciprocal_derivative_required_nonzero(
            ctx, expr, var,
        ))
        .chain(sqrt_trig_reciprocal_derivative_required_nonzero(
            ctx, expr, var,
        ))
        .chain(sqrt_trig_log_derivative_required_nonzero(ctx, expr, var))
        .chain(sqrt_hyperbolic_log_derivative_required_nonzero(
            ctx, expr, var,
        ))
        .chain(sqrt_hyperbolic_reciprocal_square_required_nonzero(
            ctx, expr, var,
        ))
        .chain(sqrt_hyperbolic_reciprocal_derivative_required_nonzero(
            ctx, expr, var,
        ))
        .chain(polynomial_trig_reciprocal_factor_required_nonzero(
            ctx, expr, var,
        ))
        .chain(constant_scaled_trig_reciprocal_derivative_required_nonzero(
            ctx, expr, var,
        ))
        .chain(reciprocal_trig_square_required_nonzero(ctx, expr, var))
        .chain(polynomial_denominator_power_substitution_required_nonzero(
            ctx, expr, var,
        ))
        .chain(polynomial_negative_denominator_power_substitution_required_nonzero(ctx, expr, var))
        .chain(
            polynomial_reciprocal_quotient_denominator_power_substitution_required_nonzero(
                ctx, expr, var,
            ),
        )
        .collect();

    dedup_required_conditions(ctx, conditions)
}

fn dedup_required_conditions(ctx: &Context, conditions: Vec<ExprId>) -> Vec<ExprId> {
    let mut unique = Vec::new();
    for condition in conditions {
        if unique
            .iter()
            .any(|existing| compare_expr(ctx, *existing, condition) == Ordering::Equal)
        {
            continue;
        }
        unique.push(condition);
    }
    unique
}

pub fn integrate_symbolic_required_positive_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let mut conditions: Vec<ExprId> = bounded_inverse_trig_linear_radicand(ctx, expr, var)
        .into_iter()
        .collect();
    conditions.extend(arctan_sqrt_var_reciprocal_required_positive_radicand(
        ctx, expr, var,
    ));
    conditions.extend(arctan_sqrt_affine_derivative_required_positive_radicand(
        ctx, expr, var,
    ));
    conditions.extend(arcsin_polynomial_substitution_radicand(ctx, expr, var));
    conditions.extend(sqrt_derivative_substitution_radicand(ctx, expr, var));
    conditions.extend(sqrt_trig_reciprocal_derivative_radicand(ctx, expr, var));
    conditions.extend(sqrt_trig_log_derivative_radicand(ctx, expr, var));
    conditions.extend(sqrt_hyperbolic_log_derivative_radicand(ctx, expr, var));
    conditions.extend(sqrt_hyperbolic_reciprocal_square_radicand(ctx, expr, var));
    conditions.extend(sqrt_hyperbolic_reciprocal_derivative_radicand(
        ctx, expr, var,
    ));
    conditions.extend(atanh_polynomial_substitution_denominator(ctx, expr, var));
    conditions.extend(acosh_affine_radicands(ctx, expr, var));
    conditions
}

fn acosh_affine_radicands(ctx: &mut Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return vec![],
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Acosh) {
        return vec![];
    }

    if !integrate_symbolic_is_acosh_affine_variable_target(ctx, expr, var) {
        return vec![];
    }

    let (left, right) = acosh_radicands(ctx, args[0]);
    vec![left, right]
}

/// Integrate `expr` with respect to `var` using a small set of symbolic rules.
pub fn integrate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    // Extract variant info in one borrow, then process with owned ExprId values.
    enum IntKind {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Neg(ExprId),
        Mul(ExprId, ExprId),
        Pow(ExprId, ExprId),
        Variable(usize),
        Div(ExprId, ExprId),
        Function(usize, Vec<ExprId>),
        Other,
    }
    let kind = match ctx.get(expr) {
        Expr::Add(l, r) => IntKind::Add(*l, *r),
        Expr::Sub(l, r) => IntKind::Sub(*l, *r),
        Expr::Neg(inner) => IntKind::Neg(*inner),
        Expr::Mul(l, r) => IntKind::Mul(*l, *r),
        Expr::Pow(b, e) => IntKind::Pow(*b, *e),
        Expr::Variable(s) => IntKind::Variable(*s),
        Expr::Div(n, d) => IntKind::Div(*n, *d),
        Expr::Function(f, args) => IntKind::Function(*f, args.clone()),
        _ => IntKind::Other,
    };

    if matches!(kind, IntKind::Add(_, _) | IntKind::Sub(_, _)) {
        if let Some(integral) =
            additive_common_trig_polynomial_substitution_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_log_power_product_substitution_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }
    }

    if let IntKind::Add(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Add(int_l, int_r)));
    }

    if let IntKind::Sub(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Sub(int_l, int_r)));
    }

    if let IntKind::Neg(inner) = kind {
        let inner_integral = integrate_symbolic_expr(ctx, inner, var)?;
        return Some(negate_integration_result(ctx, inner_integral));
    }

    if let Some(integral) = polynomial_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_derivative_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_product_derivative_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_trig_reciprocal_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_trig_log_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_hyperbolic_log_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_hyperbolic_reciprocal_square_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_hyperbolic_reciprocal_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_power_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_log_product_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_log_power_product_substitution_antiderivative(ctx, expr, var)
    {
        return Some(integral);
    }

    if let Some(integral) = linear_times_exp_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = linear_times_trig_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = linear_times_hyperbolic_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = arcsin_polynomial_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = asinh_polynomial_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let IntKind::Mul(l, r) = kind {
        if !contains_named_var(ctx, l, var) {
            if let Some(integral) =
                constant_scaled_trig_reciprocal_derivative_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_hyperbolic_reciprocal_square_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_denominator_power_substitution_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(int_r) = integrate_symbolic_expr(ctx, r, var) {
                return Some(mul2_raw(ctx, l, int_r));
            }
        }
        if !contains_named_var(ctx, r, var) {
            if let Some(integral) =
                constant_scaled_trig_reciprocal_derivative_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_hyperbolic_reciprocal_square_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_denominator_power_substitution_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(int_l) = integrate_symbolic_expr(ctx, l, var) {
                return Some(mul2_raw(ctx, r, int_l));
            }
        }
    }

    if !contains_named_var(ctx, expr, var) {
        let var_expr = ctx.var(var);
        return Some(mul2_raw(ctx, expr, var_expr));
    }

    if let IntKind::Pow(base, exp) = kind {
        if is_negative_half(ctx, exp) && is_var_square_plus_one(ctx, base, var) {
            let var_expr = ctx.var(var);
            return Some(ctx.call_builtin(BuiltinFn::Asinh, vec![var_expr]));
        }

        if let Some((a, _)) = get_linear_coeffs(ctx, base, var) {
            if !contains_named_var(ctx, exp, var) {
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == BigRational::from_integer((-1).into()) {
                        let ln_u = ln_abs(ctx, base);
                        return Some(ctx.add(Expr::Div(ln_u, a)));
                    }
                }

                let one = ctx.num(1);
                let new_exp = ctx.add(Expr::Add(exp, one));

                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };
                let new_denom = if is_a_one {
                    new_exp
                } else {
                    mul2_raw(ctx, a, new_exp)
                };

                let pow_expr = ctx.add(Expr::Pow(base, new_exp));
                return Some(ctx.add(Expr::Div(pow_expr, new_denom)));
            }
        }

        if !contains_named_var(ctx, base, var) {
            if let Some((a, _)) = get_linear_coeffs(ctx, exp, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                let is_e = if let Expr::Constant(c) = ctx.get(base) {
                    c == &cas_ast::Constant::E
                } else {
                    false
                };

                if is_e {
                    if is_a_one {
                        return Some(expr);
                    }
                    return Some(ctx.add(Expr::Div(expr, a)));
                }

                let ln_c = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let denom = if is_a_one {
                    ln_c
                } else {
                    mul2_raw(ctx, a, ln_c)
                };
                return Some(ctx.add(Expr::Div(expr, denom)));
            }
        }
    }

    if let IntKind::Variable(sym_id) = kind {
        if ctx.sym_name(sym_id) == var {
            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let pow_expr = ctx.add(Expr::Pow(var_expr, two));
            return Some(ctx.add(Expr::Div(pow_expr, two)));
        }
    }

    if let IntKind::Div(num, den) = kind {
        if let Some(integral) = arctan_sqrt_var_reciprocal_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = arctan_sqrt_affine_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = reciprocal_trig_log_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_reciprocal_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_trig_reciprocal_derivative_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_trig_reciprocal_factor_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_log_derivative_ratio_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_log_derivative_ratio_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            hyperbolic_tanh_reciprocal_log_sinh_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_reciprocal_square_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            arctan_unary_derivative_substitution_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = sqrt_derivative_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            arcsin_polynomial_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = asinh_polynomial_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = atanh_polynomial_substitution_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_denominator_power_substitution_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_negative_denominator_power_substitution_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
                ctx, num, den, var,
            )
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_square_minus_constant_log_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_log_derivative_power_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = arctan_polynomial_substitution_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                if is_var_square_plus_one(ctx, den, var) {
                    let var_expr = ctx.var(var);
                    return Some(ctx.call_builtin(BuiltinFn::Arctan, vec![var_expr]));
                }

                if let Some(integral) = reciprocal_trig_square_antiderivative(ctx, den, var) {
                    return Some(integral);
                }

                if let Some((a, _)) = get_linear_coeffs(ctx, den, var) {
                    let ln_den = ln_abs(ctx, den);
                    return Some(ctx.add(Expr::Div(ln_den, a)));
                }
            }
        }

        if let Some(integral) = polynomial_reciprocal_trig_square_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_log_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }
    }

    if let IntKind::Function(fn_id, args) = kind {
        if args.len() == 1 {
            let arg = args[0];
            if let Some(
                builtin @ (BuiltinFn::Tan | BuiltinFn::Cot | BuiltinFn::Sec | BuiltinFn::Csc),
            ) = ctx.builtin_of(fn_id)
            {
                return trig_log_antiderivative(ctx, builtin, arg, var);
            }

            if matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Arctan | BuiltinFn::Atan)
            ) {
                if let Some(integral) =
                    arctan_reciprocal_affine_variable_antiderivative(ctx, arg, var)
                {
                    return Some(integral);
                }
                if let Some(integral) = arctan_scaled_variable_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Asinh) {
                if let Some(integral) = asinh_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Atanh) {
                if let Some(integral) = atanh_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Acosh) {
                if let Some(integral) = acosh_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if let Some(
                builtin @ (BuiltinFn::Arcsin
                | BuiltinFn::Asin
                | BuiltinFn::Arccos
                | BuiltinFn::Acos),
            ) = ctx.builtin_of(fn_id)
            {
                if let Some(integral) =
                    bounded_inverse_trig_linear_antiderivative(ctx, builtin, arg, var)
                {
                    return Some(integral);
                }
            }

            if let Some((a, _)) = get_linear_coeffs(ctx, arg, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                match ctx.builtin_of(fn_id) {
                    Some(BuiltinFn::Sin) => {
                        let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                        let integral = ctx.add(Expr::Neg(cos_arg));
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Cos) => {
                        let integral = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Exp) => {
                        if is_a_one {
                            return Some(expr);
                        }
                        return Some(ctx.add(Expr::Div(expr, a)));
                    }
                    Some(BuiltinFn::Ln) => {
                        if is_a_one && is_var(ctx, arg, var) {
                            let product = mul2_raw(ctx, arg, expr);
                            return Some(ctx.add(Expr::Sub(product, arg)));
                        }

                        let one = ctx.num(1);
                        let log_minus_one = ctx.add(Expr::Sub(expr, one));
                        let integral = mul2_raw(ctx, arg, log_minus_one);
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

/// Returns `(a, b)` such that `expr = a*var + b`.
pub fn get_linear_coeffs(ctx: &mut Context, expr: ExprId, var: &str) -> Option<(ExprId, ExprId)> {
    if !contains_named_var(ctx, expr, var) {
        return Some((ctx.num(0), expr));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some((ctx.num(1), ctx.num(0))),
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if !contains_named_var(ctx, l, var) && is_var(ctx, r, var) {
                return Some((l, ctx.num(0)));
            }
            if !contains_named_var(ctx, l, var) {
                let (a, b) = get_linear_coeffs(ctx, r, var)?;
                if !contains_named_var(ctx, a, var) && !contains_named_var(ctx, b, var) {
                    return Some((
                        multiply_linear_part(ctx, l, a),
                        multiply_linear_part(ctx, l, b),
                    ));
                }
            }
            if is_var(ctx, l, var) && !contains_named_var(ctx, r, var) {
                return Some((r, ctx.num(0)));
            }
            if !contains_named_var(ctx, r, var) {
                let (a, b) = get_linear_coeffs(ctx, l, var)?;
                if !contains_named_var(ctx, a, var) && !contains_named_var(ctx, b, var) {
                    return Some((
                        multiply_linear_part(ctx, r, a),
                        multiply_linear_part(ctx, r, b),
                    ));
                }
            }
            None
        }
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            if contains_named_var(ctx, den, var) {
                return None;
            }
            let (a, b) = get_linear_coeffs(ctx, num, var)?;
            if !contains_named_var(ctx, a, var) && !contains_named_var(ctx, b, var) {
                return Some((
                    divide_linear_part(ctx, a, den),
                    divide_linear_part(ctx, b, den),
                ));
            }
            None
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);

            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = ctx.add(Expr::Add(a1, a2));
                    let b = ctx.add(Expr::Add(b1, b2));
                    return Some((a, b));
                }
            }
            None
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);
            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = ctx.add(Expr::Sub(a1, a2));
                    let b = ctx.add(Expr::Sub(b1, b2));
                    return Some((a, b));
                }
            }
            None
        }
        Expr::Neg(inner) => {
            let (a, b) = get_linear_coeffs(ctx, *inner, var)?;
            let neg_a = ctx.add(Expr::Neg(a));
            let neg_b = ctx.add(Expr::Neg(b));
            Some((neg_a, neg_b))
        }
        _ => None,
    }
}

fn multiply_linear_part(ctx: &mut Context, factor: ExprId, part: ExprId) -> ExprId {
    if is_number(ctx, part, 0) {
        ctx.num(0)
    } else if is_number(ctx, factor, 1) {
        part
    } else {
        mul2_raw(ctx, factor, part)
    }
}

fn divide_linear_part(ctx: &mut Context, part: ExprId, denominator: ExprId) -> ExprId {
    if is_number(ctx, part, 0) {
        ctx.num(0)
    } else if is_number(ctx, denominator, 1) {
        part
    } else {
        ctx.add(Expr::Div(part, denominator))
    }
}

fn is_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        ctx.sym_name(*sym_id) == var
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{get_linear_coeffs, integrate_symbolic_expr};
    use crate::polynomial::Polynomial;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use num_rational::BigRational;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn assert_constant_expr(ctx: &Context, id: cas_ast::ExprId, numerator: i64, denominator: i64) {
        let poly = Polynomial::from_expr(ctx, id, "x").expect("constant polynomial");
        assert_eq!(
            poly.coeffs,
            vec![BigRational::new(numerator.into(), denominator.into())]
        );
    }

    #[test]
    fn integrates_simple_power() {
        let mut ctx = Context::new();
        let expr = parse("x^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x^(1 + 2) / (1 + 2)");
    }

    #[test]
    fn integrates_linear_trig_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2 * cos(2 * x)");
    }

    #[test]
    fn integrates_linear_log_table() {
        let mut ctx = Context::new();
        let expr = parse("ln(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x * ln(x) - x");

        let expr = parse("ln(2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(2 * x + 1) * (ln(2 * x + 1) - 1) / (0 + 2)"
        );
    }

    #[test]
    fn integrates_polynomial_derivative_times_log_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*ln(x^2+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(x^2 + 1) * (ln(x^2 + 1) - 1)");

        let expr = parse("(2*x+1)*ln(x^2+x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(x^2 + x + 1) * (ln(x^2 + x + 1) - 1)");

        let expr = parse("4*x*ln(x^2+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2 * (x^2 + 1) * (ln(x^2 + 1) - 1)");
    }

    #[test]
    fn integrates_polynomial_log_derivative_power_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*ln(x^2+1)^2/(x^2+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/3 * ln(x^2 + 1)^3");

        let expr = parse("(2*x+1)*ln(abs(x^2+x-1))^2/(x^2+x-1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/3 * ln(|x^2 + x - 1|)^3");
    }

    #[test]
    fn integrates_polynomial_derivative_times_log_square_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("2*x*ln(x^2+1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^2 + 1) * (ln(x^2 + 1)^2 - 2 * ln(x^2 + 1) + 2)"
        );

        let expr = parse("ln(2*x+1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * (2 * x + 1) * (ln(2 * x + 1)^2 - 2 * ln(2 * x + 1) + 2)"
        );

        let expr = parse("(2*x+1)*ln(x^2+x+1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^2 + x + 1) * (ln(x^2 + x + 1)^2 - 2 * ln(x^2 + x + 1) + 2)"
        );

        let expr = parse("2*x*ln(x^2-1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^2 - 1) * (ln(x^2 - 1)^2 - 2 * ln(x^2 - 1) + 2)"
        );

        let expr = parse("(2*x+1)*ln(x^2+x-1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^2 + x - 1) * (ln(x^2 + x - 1)^2 - 2 * ln(x^2 + x - 1) + 2)"
        );

        let expr = parse("(3*x^2-1)*ln(x^3-x)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^3 - x) * (ln(x^3 - x)^2 - 2 * ln(x^3 - x) + 2)"
        );

        let expr = parse("(4*x^3-2*x)*ln(x^4-x^2-1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^4 - x^2 - 1) * (ln(x^4 - x^2 - 1)^2 - 2 * ln(x^4 - x^2 - 1) + 2)"
        );

        let expr = parse("2*x*ln(x^2+1)^3", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^2 + 1) * (ln(x^2 + 1)^3 - 3 * ln(x^2 + 1)^2 + 6 * ln(x^2 + 1) - 6)"
        );

        let expr = parse("(2*x+1)*ln(x^2+x+1)^3", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(x^2 + x + 1) * (ln(x^2 + x + 1)^3 - 3 * ln(x^2 + x + 1)^2 + 6 * ln(x^2 + x + 1) - 6)"
        );
    }

    #[test]
    fn integrates_linear_times_exp_linear_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("x*exp(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "e^x * (x - 1)");

        let expr = parse("(2*x+3)*exp(2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "e^(2 * x + 1) * ((2 * x + 3) / 2 - 1/2)"
        );

        let expr = parse("(x+1)*exp((3*x+2)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "e^((3 * x + 2) / 2) * ((x + 1) / 3/2 - 4/9)"
        );

        let expr = parse("(x+1)*exp((2-3*x)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "e^((2 - 3 * x) / 2) * (-(x + 1) / 3/2 - 4/9)"
        );
    }

    #[test]
    fn integrates_linear_times_trig_linear_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("x*sin(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x) - x * cos(x)");

        let expr = parse("x*cos(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "cos(x) + x * sin(x)");

        let expr = parse("(2*x+3)*sin(2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * sin(2 * x + 1) - (cos(2 * x + 1) * (2 * x + 3))/2"
        );

        let expr = parse("(2*x+3)*cos(2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * cos(2 * x + 1) + (sin(2 * x + 1) * (2 * x + 3))/2"
        );

        let expr = parse("(x+1)*sin((3*x+2)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "4/9 * sin((3 * x + 2) / 2) - (cos((3 * x + 2) / 2) * (x + 1))/3/2"
        );

        let expr = parse("(x+1)*cos((3*x+2)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "4/9 * cos((3 * x + 2) / 2) + (sin((3 * x + 2) / 2) * (x + 1))/3/2"
        );

        let expr = parse("(x+1)*sin((2-3*x)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "4/9 * sin((2 - 3 * x) / 2) - -cos((2 - 3 * x) / 2) * (x + 1)/3/2"
        );

        let expr = parse("(x+1)*cos((2-3*x)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "4/9 * cos((2 - 3 * x) / 2) + -sin((2 - 3 * x) / 2) * (x + 1)/3/2"
        );
    }

    #[test]
    fn integrates_linear_times_hyperbolic_linear_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("x*sinh(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x * cosh(x) - sinh(x)");

        let expr = parse("x*cosh(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x * sinh(x) - cosh(x)");

        let expr = parse("(2*x+3)*sinh(2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(cosh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * sinh(2 * x + 1)"
        );

        let expr = parse("(2*x+3)*cosh(2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(sinh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * cosh(2 * x + 1)"
        );

        let expr = parse("(x+1)*sinh((3*x+2)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(cosh((3 * x + 2) / 2) * (x + 1))/3/2 - 4/9 * sinh((3 * x + 2) / 2)"
        );

        let expr = parse("(x+1)*cosh((3*x+2)/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(sinh((3 * x + 2) / 2) * (x + 1))/3/2 - 4/9 * cosh((3 * x + 2) / 2)"
        );
    }

    #[test]
    fn integrates_explicit_negation_by_linearity() {
        let mut ctx = Context::new();
        let expr = parse("-sin(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "cos(x)");

        let expr = parse("-(x*sin(x^2)/cos(x^2)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-((1 * 1/2)/cos(x^2))");

        let expr = parse("-(x^2*cos(x^3)/sin(x^3)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-(-1 * 1/3/sin(x^3))");

        let expr = parse("-tan(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    }

    #[test]
    fn integrates_polynomial_derivative_exp_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*exp(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "e^(x^2)");
    }

    #[test]
    fn integrates_polynomial_derivative_trig_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*cos(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x^2)");

        let expr = parse("2*x*sin(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-cos(x^2)");

        let expr = parse("4*x^3*cos(x^4-x^2)-2*x*cos(x^4-x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x^4 - x^2)");

        let expr = parse("4*x^3*sin(x^4-x^2)-2*x*sin(x^4-x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-cos(x^4 - x^2)");

        let expr = parse(
            "(4*x^3*sin(x^4-x^2)-2*x*sin(x^4-x^2))/cos(x^4-x^2)",
            &mut ctx,
        )
        .expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-ln(|cos(x^4 - x^2)|)");
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(x^4 - x^2)");

        let expr = parse(
            "(4*x^3*cos(x^4-x^2)-2*x*cos(x^4-x^2))/sin(x^4-x^2)",
            &mut ctx,
        )
        .expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sin(x^4 - x^2)|)");
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(x^4 - x^2)");
    }

    #[test]
    fn integrates_polynomial_derivative_hyperbolic_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sinh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * cosh(2 * x + 1)");

        let expr = parse("cosh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * sinh(2 * x + 1)");

        let expr = parse("2*x*sinh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "cosh(x^2)");

        let expr = parse("2*x*cosh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sinh(x^2)");

        let expr = parse("tanh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|cosh(2 * x + 1)|)");

        let expr = parse("2*x*tanh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|cosh(x^2)|)");
    }

    #[test]
    fn integrates_hyperbolic_log_derivative_ratio_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sinh(2*x + 1)/cosh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|cosh(2 * x + 1)|)");

        let expr = parse("cosh(2*x + 1)/sinh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(2 * x + 1)|)");

        let expr = parse("2*x*cosh(x^2)/sinh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sinh(x^2)|)");
    }

    #[test]
    fn integrates_hyperbolic_tanh_reciprocal_log_sinh_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/tanh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(2 * x + 1)|)");

        let expr = parse("2*x/tanh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sinh(x^2)|)");

        let expr = parse("x/tanh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(x^2)|)");
    }

    #[test]
    fn integrates_hyperbolic_tanh_reciprocal_square_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/cosh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * tanh(2 * x + 1)");

        let expr = parse("2*x/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "tanh(x^2)");

        let expr = parse("x/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * tanh(x^2)");
    }

    #[test]
    fn integrates_hyperbolic_coth_reciprocal_square_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/sinh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "-1/2 * cosh(2 * x + 1)/sinh(2 * x + 1)"
        );

        let expr = parse("2*x/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-cosh(x^2) / sinh(x^2)");

        let expr = parse("x/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2 * cosh(x^2)/sinh(x^2)");
    }

    #[test]
    fn integrates_hyperbolic_cosh_reciprocal_derivative_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sinh(2*x + 1)/cosh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/cosh(2 * x + 1)");

        let expr = parse("2*x*sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/cosh(x^2)");

        let expr = parse("x*sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/cosh(x^2)");
    }

    #[test]
    fn integrates_hyperbolic_sinh_reciprocal_derivative_substitution() {
        let mut ctx = Context::new();
        let expr = parse("cosh(2*x + 1)/sinh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/sinh(2 * x + 1)");

        let expr = parse("2*x*cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/sinh(x^2)");

        let expr = parse("x*cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/sinh(x^2)");
    }

    #[test]
    fn integrates_arctan_unary_derivative_substitution() {
        let mut ctx = Context::new();
        let expr = parse("cosh(x)/(1+sinh(x)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(sinh(x))");

        let expr = parse("2*cosh(2*x + 1)/(1+sinh(2*x + 1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(sinh(2 * x + 1))");

        let expr = parse("sinh(x)/(1+cosh(x)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(cosh(x))");

        let expr = parse("-sinh(x)/(1+cosh(x)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-arctan(cosh(x))");
    }

    #[test]
    fn hyperbolic_substitution_rejects_missing_polynomial_cofactor() {
        let mut ctx = Context::new();
        let expr = parse("sinh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
        assert!(
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty()
        );

        let expr = parse("tanh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
        assert!(
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty()
        );

        let expr = parse("1/cosh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("cosh(x^2)/sinh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("1/sinh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("1/tanh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("1/(x^4-1)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    }

    #[test]
    fn integrates_reciprocal_linear_with_absolute_log() {
        let mut ctx = Context::new();
        let expr = parse("1/(3*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|3 * x|) / 3");
    }

    #[test]
    fn integrates_linear_power_minus_one_with_absolute_log() {
        let mut ctx = Context::new();
        let expr = parse("(2*x + 1)^-1", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|2 * x + 1|) / (0 + 2)");
    }

    #[test]
    fn integrates_arctan_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/(x^2+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(x)");
    }

    #[test]
    fn integrates_arctan_sqrt_reciprocal_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/(2*sqrt(x)*(x+1))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(sqrt(x))");

        let expr = parse("1/(sqrt(x)*(x+1))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2 * arctan(sqrt(x))");

        let expr = parse("x^(-1/2)/(x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2 * arctan(sqrt(x))");

        let expr = parse("1/(sqrt(x)*(4*x+1))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(2 * sqrt(x))");

        let expr = parse("1/(sqrt(x)*(x+4))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(1/2 * sqrt(x))");

        let expr = parse("1/(2*sqrt(x)*(4*x+1))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arctan(2 * sqrt(x))");
    }

    #[test]
    fn integrates_arctan_sqrt_affine_derivative_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/(sqrt(4*x+1)*(2*x+1))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(sqrt(4 * x + 1))");

        let expr = parse("(4*x+1)^(1/2)/((2*x+1)*(4*x+1))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(sqrt(4 * x + 1))");

        let expr = parse("-1/(2*sqrt(5-3*x)*(2-x))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(sqrt(5 - 3 * x))");
    }

    #[test]
    fn integrates_arctan_scaled_variable_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("arctan(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2 * ln(x^2 + 1) + x * arctan(x)");

        let expr = parse("arctan(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "-1/4 * ln((2 * x)^2 + 1) + x * arctan(2 * x)"
        );

        let expr = parse("arctan(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "-1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(2 * x + 1)"
        );

        let expr = parse("arctan(1 - 2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/4 * ln((1 - 2 * x)^2 + 1) + -1/2 * (1 - 2 * x) * arctan(1 - 2 * x)"
        );
    }

    #[test]
    fn integrates_arctan_reciprocal_scaled_variable_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("arctan(1/x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(x^2 + 1) + x * arctan(1 / x)");

        let expr = parse("arctan(1/(2*x))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/4 * ln((2 * x)^2 + 1) + x * arctan(1 / (2 * x))"
        );

        let expr = parse("arctan(1/(2*x + 1))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(1 / (2 * x + 1))"
        );
    }

    #[test]
    fn integrates_bounded_inverse_trig_scaled_variable_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sqrt(1 - x^2) + x * arcsin(x)");

        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "1 - x^2");

        let expr = parse("arccos(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x * arccos(x) - sqrt(1 - x^2)");

        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "1 - x^2");

        let expr = parse("arcsin(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sqrt(1/4 - x^2) + x * arcsin(2 * x)");

        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "1/4 - x^2");

        let expr = parse("arccos(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x * arccos(2 * x) - sqrt(1/4 - x^2)");

        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "1/4 - x^2");

        let expr = parse("arcsin(-2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "x * arcsin(-2 * x) - 1/2 * sqrt(1 - (-2 * x)^2)"
        );

        let expr = parse("arcsin(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * sqrt(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * arcsin(2 * x + 1)"
        );

        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "1 - (2 * x + 1)^2");

        let expr = parse("arccos(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * (2 * x + 1) * arccos(2 * x + 1) - 1/2 * sqrt(1 - (2 * x + 1)^2)"
        );

        let expr = parse("arcsin(1 - 2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "-1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arcsin(1 - 2 * x)"
        );
        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "x - x^2");

        let expr = parse("arccos(1 - 2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arccos(1 - 2 * x)"
        );

        let expr = parse("arcsin(a*x)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty()
        );
    }

    #[test]
    fn integrates_asinh_affine_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("asinh(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x * asinh(x) - sqrt(x^2 + 1)");

        let expr = parse("asinh(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "x * asinh(2 * x) - 1/2 * sqrt((2 * x)^2 + 1)"
        );

        let expr = parse("asinh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * (2 * x + 1) * asinh(2 * x + 1) - 1/2 * sqrt((2 * x + 1)^2 + 1)"
        );
    }

    #[test]
    fn integrates_atanh_affine_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("atanh(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(1 - x^2) + x * atanh(x)");

        let expr = parse("atanh(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/4 * ln(1 - (2 * x)^2) + x * atanh(2 * x)"
        );

        let expr = parse("atanh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/4 * ln(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * atanh(2 * x + 1)"
        );
    }

    #[test]
    fn integrates_acosh_affine_by_parts() {
        let mut ctx = Context::new();
        let expr = parse("acosh(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "x * acosh(x) - sqrt(x - 1) * sqrt(x + 1)"
        );

        let expr = parse("acosh(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "x * acosh(2 * x) - 1/2 * sqrt(2 * x - 1) * sqrt(2 * x + 1)"
        );

        let expr = parse("acosh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * (2 * x + 1) * acosh(2 * x + 1) - 1/2 * sqrt(2 * x) * sqrt(2 * x + 2)"
        );

        let expr = parse("acosh(1 - 2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "1/2 * sqrt(-2 * x) * sqrt(2 - 2 * x) - 1/2 * (1 - 2 * x) * acosh(1 - 2 * x)"
        );
    }

    #[test]
    fn integrates_polynomial_derivative_arctan_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(x^2)");

        let expr = parse("x/(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arctan(x^2)");

        let expr = parse("2*x/(4+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arctan(x^2 / 2)");

        let expr = parse("2*x/(3+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(x^2 / sqrt(3)) / sqrt(3)");

        let expr = parse("1/(4+(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arctan((x + 1) / 2)");

        let expr = parse("1/(2*x^2+2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(2 * x + 1)");

        let expr = parse("1/(2*x^2+4*x+5)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "arctan((2 * x + 2) / sqrt(6)) / sqrt(6)"
        );
    }

    #[test]
    fn integrates_polynomial_derivative_atanh_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * atanh(x^2 / 2)");

        let expr = parse("-2*x/(1-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-atanh(x^2)");

        let expr = parse("x/(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/4 * atanh(x^2 / 2)");

        let expr = parse("2*x/(3-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "atanh(x^2 / sqrt(3)) / sqrt(3)");

        let expr = parse("1/(12-4*x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/4 * atanh(x / sqrt(3)) / sqrt(3)");

        let expr = parse("(2*x+2)/(5-(x^2+2*x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "atanh((x^2 + 2 * x + 1) / sqrt(5)) / sqrt(5)"
        );

        let expr = parse("1/(4-(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * atanh((x + 1) / 2)");
    }

    #[test]
    fn integrates_inverse_sqrt_substitution_with_surd_width() {
        let mut ctx = Context::new();

        let expr = parse("2*x/sqrt(3-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin(x^2 / sqrt(3))");

        let expr = parse("2*x/sqrt(3+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x^2 / sqrt(3))");

        let expr = parse("(2*x+2)/sqrt(2 - x^4 - 4*x^3 - 6*x^2 - 4*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin((x + 1)^2 / sqrt(3))");

        let expr = parse("(2*x+2)/sqrt(4 + 4*x + 6*x^2 + 4*x^3 + x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh((x + 1)^2 / sqrt(3))");
    }

    #[test]
    fn integrates_polynomial_derivative_square_minus_constant_log_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/(x^2-1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|(x - 1) / (x + 1)|)");

        let expr = parse("2*x/(x^4-4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/4 * ln(|(x^2 - 2) / (x^2 + 2)|)");

        let expr = parse("x/(x^4-4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/8 * ln(|(x^2 - 2) / (x^2 + 2)|)");
    }

    #[test]
    fn integrates_polynomial_log_derivative_substitution() {
        let mut ctx = Context::new();
        let expr = parse("(2*x+1)/(x^2+x-1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|x^2 + x - 1|)");

        let expr = parse("(4*x+2)/(x^2+x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2 * ln(|x^2 + x + 1|)");

        let expr = parse("(2*x+3)/((x+1)*(x+2))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|(x + 1) * (x + 2)|)");
    }

    #[test]
    fn atanh_substitution_reports_open_interval_condition() {
        let mut ctx = Context::new();
        let expr = parse("2*x/(4-x^4)", &mut ctx).expect("parse");
        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");

        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "4 - x^4");

        let factored = parse("(2*x)/((2-x^2)*(x^2+2))", &mut ctx).expect("parse");
        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, factored, "x");

        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "(x^2 + 2) * (2 - x^2)");
    }

    #[test]
    fn integrates_polynomial_derivative_arcsin_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/sqrt(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin(x^2 / 2)");

        let expr = parse("x/sqrt(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arcsin(x^2 / 2)");

        let expr = parse("1/sqrt(4-(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin((x + 1) / 2)");

        let expr = parse("(2*x*sqrt(4-x^4))/(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin(x^2 / 2)");
    }

    #[test]
    fn integrates_polynomial_derivative_asinh_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/sqrt(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x^2)");

        let expr = parse("x/sqrt(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * asinh(x^2)");

        let expr = parse("2*x/sqrt(4+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x^2 / 2)");

        let expr = parse("1/sqrt(4+(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh((x + 1) / 2)");

        let expr = parse("(2*x*sqrt(1+x^4))/(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x^2)");
    }

    #[test]
    fn integrates_polynomial_derivative_over_square_root_substitution() {
        let mut ctx = Context::new();
        let expr = parse("x/sqrt(x^2+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sqrt(x^2 + 1)");

        let expr = parse("(2*x+1)/sqrt(x^2+x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2 * sqrt(x^2 + x + 1)");

        let expr = parse("((2*x+1)*sqrt(x^2+x+1))/(x^2+x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2 * sqrt(x^2 + x + 1)");
    }

    #[test]
    fn sqrt_derivative_substitution_reports_positive_radicand_condition() {
        let mut ctx = Context::new();
        let expr = parse("2*x/sqrt(x^2-1)", &mut ctx).expect("parse");
        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");

        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "x^2 - 1");
    }

    #[test]
    fn integrates_polynomial_derivative_times_square_root_substitution() {
        let mut ctx = Context::new();
        let expr = parse("x*sqrt(x^2+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/3 * (x^2 + 1)^(3/2)");

        let expr = parse("(2*x+1)*sqrt(x^2+x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2/3 * (x^2 + x + 1)^(3/2)");
    }

    #[test]
    fn integrates_polynomial_derivative_times_power_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*(x^2+1)^3", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/4 * (x^2 + 1)^4");

        let expr = parse("2*x*(x^2+1)^(3/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2/5 * (x^2 + 1)^(5/2)");

        let expr = parse("(2*x+1)*(x^2+x+1)^3", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/4 * (x^2 + x + 1)^4");
    }

    #[test]
    fn integrates_polynomial_derivative_over_denominator_power_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/(x^2+1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (x^2 + 1)");

        let expr = parse("2*x/(x^2-1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (x^2 - 1)");

        let expr = parse("2*x/(x^2+1)^3", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (2 * (x^2 + 1)^2)");

        let expr = parse("(2*x+1)/(x^4+2*x^3-x^2-2*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (x^2 + x - 1)");

        let expr = parse("(2*x+1)/(3*(x^2+x-1)^2)", &mut ctx).expect("parse");
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

        let expr = parse("(2*x+1)/(3*x^4+6*x^3-3*x^2-6*x+3)", &mut ctx).expect("parse");
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (3 * (x^2 + x - 1))");

        let expr = parse("1/3*((2*x+1)/(x^2+x-1)^2)", &mut ctx).expect("parse");
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

        let expr = parse("(2*x+1)/(x^6+3*x^5-5*x^3+3*x-1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (2 * (x^2 + x - 1)^2)");

        let expr = parse("(2*x+1)/(4*x^6+12*x^5-20*x^3+12*x-4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (8 * (x^2 + x - 1)^2)");

        let expr = parse("(2*x+1)/(3/((x^2+x-1)^(-2)))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (3 * (x^2 + x - 1))");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

        let expr = parse("1/(x^5+5*x^4+10*x^3+10*x^2+5*x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / (4 * (x + 1)^4)");
    }

    #[test]
    fn detects_bounded_reciprocal_quotient_denominator_power_substitution_targets() {
        let mut ctx = Context::new();

        for input in ["(2*x+1)/(3/(x^2+x-1)^2)", "(2*x+1)/(3/((x^2+x-1)^(-2)))"] {
            let expr = parse(input, &mut ctx).expect("parse");
            assert!(
                super::integrate_symbolic_is_bounded_reciprocal_quotient_denominator_power_substitution_target(
                    &ctx, expr, "x", 8
                ),
                "expected bounded reciprocal quotient detector to accept {input}"
            );
        }

        for input in [
            "(2*x+1)/(3/((x^2+x-1)^(-1)))",
            "(2*x+1)/(3/(x^2+x-1)^9)",
            "(2*x+1)/(3/((x^2+x-1)^(-9)))",
        ] {
            let expr = parse(input, &mut ctx).expect("parse");
            assert!(
                !super::integrate_symbolic_is_bounded_reciprocal_quotient_denominator_power_substitution_target(
                    &ctx, expr, "x", 8
                ),
                "expected bounded reciprocal quotient detector to reject {input}"
            );
        }
    }

    #[test]
    fn detects_bounded_negative_syntactic_denominator_power_substitution_targets() {
        let mut ctx = Context::new();

        for input in ["(2*x+1)/(3*(x^2+x-1)^(-1))", "(2*x+1)/(3*(x^2+x-1)^(-2))"] {
            let expr = parse(input, &mut ctx).expect("parse");
            assert!(
                super::integrate_symbolic_is_bounded_negative_syntactic_denominator_power_substitution_target(
                    &ctx, expr, "x", 8
                ),
                "expected bounded negative denominator power detector to accept {input}"
            );
        }

        for input in [
            "(2*x+1)/(3*(x^2+x-1)^(-9))",
            "(x+1)/(3*(x^2+x-1)^(-2))",
            "(2*x+1)/(3*(x^2+x-1)^2)",
        ] {
            let expr = parse(input, &mut ctx).expect("parse");
            assert!(
                !super::integrate_symbolic_is_bounded_negative_syntactic_denominator_power_substitution_target(
                    &ctx, expr, "x", 8
                ),
                "expected bounded negative denominator power detector to reject {input}"
            );
        }
    }

    #[test]
    fn integrates_asinh_kernel() {
        let mut ctx = Context::new();
        let expr = parse("(x^2+1)^(-1/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x)");
    }

    #[test]
    fn integrates_secant_squared_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/cos(x)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x) / cos(x)");

        let expr = parse("2*x/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x^2) / cos(x^2)");

        let expr = parse("x/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(sin(x^2) * 1/2)/cos(x^2)");
    }

    #[test]
    fn integrates_cosecant_squared_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/sin(x)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-cos(x) / sin(x)");

        let expr = parse("3*x^2/sin(x^3)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-cos(x^3) / sin(x^3)");
    }

    #[test]
    fn integrates_canonical_sec_tan_and_csc_cot_quotients() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x + 1)/cos(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sec(2 * x + 1) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

        let expr = parse("cos(2*x + 1)/sin(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-csc(2 * x + 1) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

        let expr = parse("tan(2*x + 1)/cos(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(1 * 1/2)/cos(2 * x + 1)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

        let expr = parse("cot(2*x + 1)/sin(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 * 1/2/sin(2 * x + 1)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

        let expr = parse("sec(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "ln(|tan(2 * x + 1) + sec(2 * x + 1)|) / (0 + 2)"
        );

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

        let expr = parse("csc(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "ln(|(cos(2 * x + 1) - 1) / sin(2 * x + 1)|) / (0 + 2)"
        );

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");
    }

    #[test]
    fn integrates_polynomial_sec_tan_and_csc_cot_quotients() {
        let mut ctx = Context::new();
        let expr = parse("2*x*sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1 / cos(x^2)");

        let expr = parse("x*sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(1 * 1/2)/cos(x^2)");

        let expr = parse("3*x^2*cos(x^3)/sin(x^3)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / sin(x^3)");

        let expr = parse("2*(x*sin(x^2)/cos(x^2)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1 / cos(x^2)");

        let expr = parse("3*(x^2*cos(x^3)/sin(x^3)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / sin(x^3)");
    }

    #[test]
    fn integrates_sqrt_chain_sec_tan_and_csc_cot_quotients() {
        let mut ctx = Context::new();
        let expr = parse("sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sec(sqrt(x))");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr =
            parse("cos((2*x)^(1/2))*(2*x)^(-1/2)/sin((2*x)^(1/2))^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-csc(sqrt(2 * x))");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(sqrt(2 * x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "2 * x");
    }

    #[test]
    fn integrates_sqrt_chain_tangent_cotangent_log_quotients() {
        let mut ctx = Context::new();
        let expr = parse("sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x)))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-ln(|cos(sqrt(x))|)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr =
            parse("cos((2*x)^(1/2))*(2*x)^(-1/2)/sin((2*x)^(1/2))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sin(sqrt(2 * x))|)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(sqrt(2 * x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "2 * x");
    }

    #[test]
    fn integrates_sqrt_chain_hyperbolic_tangent_logs() {
        let mut ctx = Context::new();
        let expr = parse("tanh(sqrt(x))/(2*sqrt(x))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|cosh(sqrt(x))|)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cosh(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr = parse("sqrt(x)/(2*x*tanh(sqrt(x)))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sinh(sqrt(x))|)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr = parse("tanh((2*x)^(1/2))*(2*x)^(-1/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|cosh(sqrt(2 * x))|)");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "2 * x");
    }

    #[test]
    fn integrates_sqrt_chain_hyperbolic_reciprocal_squares() {
        let mut ctx = Context::new();
        let expr = parse("1/(2*sqrt(x)*cosh(sqrt(x))^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "tanh(sqrt(x))");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cosh(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr = parse("1/(2*sqrt(x)*sinh(sqrt(x))^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / tanh(sqrt(x))");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr = parse("(2*x)^(-1/2)/cosh((2*x)^(1/2))^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "tanh(sqrt(2 * x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "2 * x");
    }

    #[test]
    fn integrates_sqrt_chain_hyperbolic_reciprocal_derivatives() {
        let mut ctx = Context::new();
        let expr = parse("sinh(sqrt(x))/(2*sqrt(x)*cosh(sqrt(x))^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / cosh(sqrt(x))");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cosh(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr = parse("cosh(sqrt(x))/(2*sqrt(x)*sinh(sqrt(x))^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / sinh(sqrt(x))");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "x");

        let expr = parse(
            "sinh((2*x)^(1/2))*(2*x)^(-1/2)/cosh((2*x)^(1/2))^2",
            &mut ctx,
        )
        .expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1 / cosh(sqrt(2 * x))");

        let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
        assert_eq!(positive.len(), 1);
        assert_eq!(rendered(&ctx, positive[0]), "2 * x");
    }

    #[test]
    fn canonical_sec_tan_quotient_rejects_non_linear_argument() {
        let mut ctx = Context::new();
        let expr = parse("sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
    }

    #[test]
    fn integrates_trig_log_linear_substitution() {
        let mut ctx = Context::new();
        let expr = parse("tan(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-ln(|cos(2 * x + 1)|) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

        let expr = parse("cot(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sin(2 * x + 1)|) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");
    }

    #[test]
    fn integrates_polynomial_trig_log_ratio_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*tan(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-ln(|cos(x^2)|)");
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(x^2)");

        let expr = parse("x*tan(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2 * ln(|cos(x^2)|)");

        let expr = parse("3*x^2*cot(x^3)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sin(x^3)|)");
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(x^3)");
    }

    #[test]
    fn trig_log_integration_rejects_non_linear_argument_without_condition() {
        let mut ctx = Context::new();
        let expr = parse("tan(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
    }

    #[test]
    fn extracts_linear_coeffs() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 3", &mut ctx).expect("parse");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("coeffs");
        let a_text = rendered(&ctx, a);
        assert!(a_text == "2" || a_text == "0 + 2");
        let b_text = rendered(&ctx, b);
        assert!(b_text == "3" || b_text == "0 + 3");

        let expr = parse("1 + -2*x", &mut ctx).expect("parse negated linear term");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("negated term coeffs");
        let a_text = rendered(&ctx, a);
        assert!(a_text == "-2" || a_text == "0 - 2", "{a_text}");
        let b_text = rendered(&ctx, b);
        assert!(
            b_text == "1" || b_text == "0 + 1" || b_text == "1 + -0",
            "{b_text}"
        );

        let expr = parse("1 - 2*x", &mut ctx).expect("parse negative slope affine");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("negative slope coeffs");
        let a_text = rendered(&ctx, a);
        assert!(a_text == "-2" || a_text == "0 - 2", "{a_text}");
        let b_text = rendered(&ctx, b);
        assert!(b_text == "1" || b_text == "1 - 0", "{b_text}");

        let expr = parse("1/2*(3*x + 2)", &mut ctx).expect("parse scaled affine");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("scaled affine coeffs");
        assert_constant_expr(&ctx, a, 3, 2);
        assert_constant_expr(&ctx, b, 1, 1);

        let expr = parse("(3*x + 2)/2", &mut ctx).expect("parse divided affine");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("divided affine coeffs");
        assert_constant_expr(&ctx, a, 3, 2);
        assert_constant_expr(&ctx, b, 1, 1);

        let expr = parse("asinh(1 - 2*x)", &mut ctx).expect("parse shifted asinh");
        assert!(super::integrate_symbolic_is_asinh_affine_variable_target(
            &mut ctx, expr, "x"
        ));

        let expr = parse("asinh(2*x)", &mut ctx).expect("parse scaled asinh");
        assert!(super::integrate_symbolic_is_asinh_affine_variable_target(
            &mut ctx, expr, "x"
        ));

        let expr = parse("-(2*x + 1)", &mut ctx).expect("parse negated affine");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("negated affine coeffs");
        assert_eq!(rendered(&ctx, a), "-(0 + 2)");
        assert_eq!(rendered(&ctx, b), "-(0 + 1)");
    }
}
