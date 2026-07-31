//! `symbolic_integration_support`: familia `rational`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn rational_over_expr(
    ctx: &mut Context,
    numerator: BigRational,
    denominator: ExprId,
) -> ExprId {
    if numerator.is_zero() {
        return ctx.num(0);
    }
    if numerator == BigRational::from_integer((-1).into()) {
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, denominator));
        return ctx.add(Expr::Neg(reciprocal));
    }
    let numerator_expr = ctx.add(Expr::Number(numerator));
    ctx.add(Expr::Div(numerator_expr, denominator))
}

/// `r^mult` for a rational `r` and integer `mult` (negative ⇒ reciprocal). Caller ensures `r != 0`.
pub(super) fn int_pow_rational(r: &BigRational, mult: i64) -> BigRational {
    let pow_pos = |e: u32| -> BigRational {
        BigRational::new(
            num_traits::pow(r.numer().clone(), e as usize),
            num_traits::pow(r.denom().clone(), e as usize),
        )
    };
    if mult >= 0 {
        pow_pos(mult as u32)
    } else {
        pow_pos((-mult) as u32).recip()
    }
}

pub(super) fn scale_by_reciprocal_linear_coeff(
    ctx: &mut Context,
    integral: ExprId,
    coeff: ExprId,
) -> ExprId {
    if let Some(coeff) = rational_constant_value(ctx, coeff) {
        if !coeff.is_zero() {
            return scale_rational_term(ctx, BigRational::one() / coeff, integral);
        }
    }

    divide_by_coeff_unless_one(ctx, integral, coeff)
}

pub(super) fn scale_reciprocal_integration_result(
    ctx: &mut Context,
    scale: BigRational,
    expr: ExprId,
) -> ExprId {
    if scale.is_one() {
        return expr;
    }
    if scale == -BigRational::one() {
        return negate_integration_result(ctx, expr);
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => scale_reciprocal_integration_result(ctx, -scale, inner),
        Expr::Div(num, den) => {
            let numerator_scale = BigRational::from_integer(scale.numer().clone());
            let scaled_num = if is_number(ctx, num, 1) {
                ctx.add(Expr::Number(numerator_scale))
            } else if numerator_scale.is_one() {
                num
            } else if numerator_scale == BigRational::from_integer((-1).into()) {
                ctx.add(Expr::Neg(num))
            } else {
                let numerator_scale = ctx.add(Expr::Number(numerator_scale));
                multiply_rational_factor_if_possible(ctx, numerator_scale, num)
                    .unwrap_or_else(|| mul2_raw(ctx, numerator_scale, num))
            };

            let denominator_scale = BigRational::from_integer(scale.denom().clone());
            let scaled_den = if denominator_scale.is_one() {
                den
            } else {
                let denominator_scale = ctx.add(Expr::Number(denominator_scale));
                mul2_raw(ctx, denominator_scale, den)
            };

            ctx.add(Expr::Div(scaled_num, scaled_den))
        }
        _ => scale_rational_term(ctx, scale, expr),
    }
}

pub(super) fn scale_expr_reciprocal_integration_result_preserving_presentation(
    ctx: &mut Context,
    scale: ExprId,
    expr: ExprId,
    preserve_presentation: bool,
) -> ExprId {
    let scaled = scale_expr_reciprocal_integration_result(ctx, scale, expr);
    if preserve_presentation {
        cas_ast::hold::wrap_hold(ctx, scaled)
    } else {
        scaled
    }
}

pub(super) fn scale_reciprocal_integration_result_with_unit_presentation(
    ctx: &mut Context,
    scale: BigRational,
    expr: ExprId,
    preserve_unit_scale_presentation: bool,
) -> ExprId {
    let scaled =
        scale_reciprocal_integration_result_preserving_presentation(ctx, scale.clone(), expr);
    if preserve_unit_scale_presentation && (scale.is_one() || scale == -BigRational::one()) {
        cas_ast::hold::wrap_hold(ctx, scaled)
    } else {
        scaled
    }
}

pub(super) fn quotient_scale_against_polynomial(
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

pub(super) fn reciprocal_affine_variable_denominator(
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

/// True when expr is built purely from rational operations over
/// sin(k*var)/cos(k*var) atoms and var-free subtrees; collects every k.
/// Any other occurrence of var (bare x, tan, nested args) refuses.
pub(super) fn collect_weierstrass_rational_slopes(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    slopes: &mut Vec<BigRational>,
) -> bool {
    if let Some((slope, _)) = weierstrass_trig_atom(ctx, expr, var) {
        slopes.push(slope);
        return true;
    }
    if !contains_named_var(ctx, expr, var) {
        return true;
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_weierstrass_rational_slopes(ctx, l, var, slopes)
                && collect_weierstrass_rational_slopes(ctx, r, var, slopes)
        }
        Expr::Neg(inner) => collect_weierstrass_rational_slopes(ctx, inner, var, slopes),
        Expr::Pow(base, exponent) => {
            if contains_named_var(ctx, exponent, var) {
                return false;
            }
            let Some(value) = crate::numeric_eval::as_rational_const(ctx, exponent) else {
                return false;
            };
            if !value.is_integer() {
                return false;
            }
            collect_weierstrass_rational_slopes(ctx, base, var, slopes)
        }
        _ => false,
    }
}

pub(super) fn scale_polynomial_rational(poly: &Polynomial, scale: &BigRational) -> Polynomial {
    let mut scaled = poly.clone();
    for coeff in &mut scaled.coeffs {
        *coeff *= scale;
    }
    scaled
}

pub(super) fn rational_coefficient_times_reciprocal_power(
    ctx: &mut Context,
    coefficient: BigRational,
    base: ExprId,
    positive_exponent: BigRational,
) -> ExprId {
    let numerator = BigRational::from_integer(coefficient.numer().clone());
    let numerator = ctx.add(Expr::Number(numerator));

    let denominator_power = if positive_exponent == BigRational::new(1.into(), 2.into()) {
        ctx.call_builtin(BuiltinFn::Sqrt, vec![base])
    } else if positive_exponent.is_one() {
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

pub(super) fn expanded_square_denominator_base(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    Polynomial::from_expr(ctx, den, var).ok()?;

    // DETECTOR use, deliberately domain-neutral (audit 2026-07-30, S4-002
    // corrección (c)): only the PerfectSquare KIND is consulted; the |·|
    // rewrite is not published from here.
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

pub(super) fn positive_integer_power_rational(base: &BigRational, exponent: u32) -> BigRational {
    let mut acc = BigRational::one();
    for _ in 0..exponent {
        acc *= base.clone();
    }
    acc
}

pub(super) fn build_linear_expr_from_rationals(
    ctx: &mut Context,
    var: &str,
    slope: BigRational,
    offset: BigRational,
) -> Option<ExprId> {
    if slope.is_zero() && offset.is_zero() {
        return None;
    }

    let mut terms = Vec::new();
    if !slope.is_zero() {
        let var_expr = ctx.var(var);
        terms.push(scale_rational_term(ctx, slope, var_expr));
    }
    if !offset.is_zero() {
        terms.push(ctx.add(Expr::Number(offset)));
    }

    Some(build_balanced_add(ctx, &terms))
}

pub(super) fn split_unit_reciprocal_factor(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = mul_leaves(ctx, term);
    if factors.len() < 2 {
        return None;
    }

    let mut reciprocal_index = None;
    let mut denominator = None;
    for (index, factor) in factors.iter().copied().enumerate() {
        let Expr::Div(num, den) = ctx.get(factor) else {
            continue;
        };
        if !is_number(ctx, *num, 1) {
            continue;
        }
        if reciprocal_index.is_some() {
            return None;
        }
        reciprocal_index = Some(index);
        denominator = Some(*den);
    }

    let reciprocal_index = reciprocal_index?;
    let numerator_factors: Vec<_> = factors
        .into_iter()
        .enumerate()
        .filter_map(|(index, factor)| (index != reciprocal_index).then_some(factor))
        .collect();
    let numerator = match numerator_factors.as_slice() {
        [] => ctx.num(1),
        [single] => *single,
        _ => build_balanced_mul(ctx, &numerator_factors),
    };
    Some((numerator, denominator?))
}

pub(super) fn scale_by_rational_over_variable_free_slope(
    ctx: &mut Context,
    numerator_scale: BigRational,
    slope: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    if numerator_scale.is_zero() || contains_named_var(ctx, slope, var) {
        return None;
    }
    if let Some(slope_scale) = rational_constant_value(ctx, slope) {
        if slope_scale.is_zero() {
            return None;
        }
        return Some(scale_factor(ctx, numerator_scale / slope_scale, expr));
    }

    let scaled = scale_factor(ctx, numerator_scale, expr);
    Some(ctx.add(Expr::Div(scaled, slope)))
}

pub(super) fn split_rational_denominator_scale(
    ctx: &mut Context,
    expr: ExprId,
) -> (BigRational, ExprId) {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (scale, core) = split_rational_denominator_scale(ctx, inner);
            (-scale, core)
        }
        Expr::Div(num, den) => {
            if let Some(den_scale) = rational_constant_value(ctx, den) {
                if !den_scale.is_zero() {
                    return (BigRational::one() / den_scale, num);
                }
            }
            (BigRational::one(), expr)
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut core_factors = Vec::new();
            for factor in mul_leaves(ctx, expr) {
                if let Some(factor_scale) = rational_constant_value(ctx, factor) {
                    scale *= factor_scale;
                } else {
                    core_factors.push(factor);
                }
            }
            let core = match core_factors.len() {
                0 => ctx.num(1),
                1 => core_factors[0],
                _ => build_balanced_mul(ctx, &core_factors),
            };
            (scale, core)
        }
        _ => (BigRational::one(), expr),
    }
}

pub(super) fn positive_quadratic_linear_numerator_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if Polynomial::from_expr(ctx, den, var).is_err() {
        if let Some(integral) =
            positive_constant_radius_quadratic_linear_numerator_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }
    }

    let mut numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let mut denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }

    let mut log_den_expr = den;
    if denominator.leading_coeff().is_negative() {
        numerator = numerator.neg();
        denominator = denominator.neg();
        log_den_expr = denominator.to_expr(ctx);
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
    let discriminant_gap = four * a.clone() * c - b.clone() * b.clone();
    if discriminant_gap <= BigRational::zero() {
        return None;
    }

    let (quotient, remainder) = if numerator.degree() >= denominator.degree() {
        numerator.div_rem(&denominator).ok()?
    } else {
        (Polynomial::zero(numerator.var.clone()), numerator)
    };
    if remainder.degree() > 1 {
        return None;
    }

    let mut terms = Vec::new();
    if !quotient.is_zero() {
        terms.push(polynomial_antiderivative_expr(ctx, &quotient));
    }
    if remainder.is_zero() {
        return match terms.len() {
            0 => None,
            1 => Some(terms[0]),
            _ => Some(build_balanced_add(ctx, &terms)),
        };
    }

    let numerator_constant = remainder
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let numerator_linear = remainder
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let log_scale = numerator_linear / (two * a);
    let arctan_scale = numerator_constant - log_scale.clone() * b;

    if !log_scale.is_zero() {
        let log_arg = positive_quadratic_log_argument(ctx, &denominator, log_den_expr);
        let log_den = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);
        terms.push(scale_rational_term(ctx, log_scale, log_den));
    }
    if !arctan_scale.is_zero() {
        let constant_numerator = Polynomial::new(vec![arctan_scale], denominator.var.clone());
        terms.push(arctan_scaled_quadratic_antiderivative(
            ctx,
            &constant_numerator,
            &denominator,
        )?);
    }

    match terms.len() {
        0 => None,
        1 => Some(terms[0]),
        _ => {
            let sum = build_balanced_add(ctx, &terms);
            Some(cas_ast::hold::wrap_hold(ctx, sum))
        }
    }
}

fn positive_constant_radius_quadratic_linear_numerator_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    if numerator.degree() > 1 {
        return None;
    }

    let (arg, radius) = positive_square_constant_plus_square_arg(ctx, den, var)
        .or_else(|| positive_square_constant_plus_expanded_square_arg(ctx, den, var))?;
    let (linear_arg, slope_expr) = nonzero_linear_arg_and_slope(ctx, arg, var)?;
    let slope = rational_constant_value(ctx, slope_expr)?;
    let arg_poly = Polynomial::from_expr(ctx, linear_arg, var).ok()?;

    let two = BigRational::from_integer(2.into());
    let derivative_poly = scale_polynomial(&arg_poly, two * slope);
    if derivative_poly.degree() != 1 {
        return None;
    }
    let derivative_linear = derivative_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if derivative_linear.is_zero() {
        return None;
    }

    let numerator_linear = numerator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let log_scale = numerator_linear / derivative_linear;
    let scaled_derivative_poly = scale_polynomial(&derivative_poly, log_scale.clone());
    let remainder = numerator.sub(&scaled_derivative_poly);
    if remainder.degree() > 0 {
        return None;
    }
    let remainder_constant = remainder
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let mut terms = Vec::new();
    if !log_scale.is_zero() {
        let log_den = ctx.call_builtin(BuiltinFn::Ln, vec![den]);
        terms.push(scale_rational_term(ctx, log_scale, log_den));
    }
    if !remainder_constant.is_zero() {
        let (arctan_arg, arctan_scale) =
            arctan_positive_quadratic_arg_and_scale(ctx, linear_arg, radius, var)?;
        let arctan = ctx.call_builtin(BuiltinFn::Atan, vec![arctan_arg]);
        let arctan_integral = ctx.add(Expr::Div(arctan, arctan_scale));
        terms.push(scale_rational_term(
            ctx,
            remainder_constant,
            arctan_integral,
        ));
    }

    match terms.len() {
        0 => None,
        1 => Some(terms[0]),
        _ => {
            let sum = build_balanced_add(ctx, &terms);
            Some(cas_ast::hold::wrap_hold(ctx, sum))
        }
    }
}

pub fn integrate_symbolic_positive_quadratic_linear_numerator_decomposition_expr(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    if Polynomial::from_expr(ctx, expr, var).is_err() {
        if let Some(decomposition) =
            positive_constant_radius_quadratic_linear_numerator_decomposition_expr(ctx, expr, var)
        {
            return Some(decomposition);
        }
    }

    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let mut numerator = Polynomial::from_expr(ctx, num, var).ok()?;

    let mut denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }
    if denominator.leading_coeff().is_negative() {
        numerator = numerator.neg();
        denominator = denominator.neg();
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
    let discriminant_gap = four * a.clone() * c - b.clone() * b.clone();
    if discriminant_gap <= BigRational::zero() {
        return None;
    }

    let (quotient, remainder) = if numerator.degree() >= denominator.degree() {
        numerator.div_rem(&denominator).ok()?
    } else {
        (Polynomial::zero(numerator.var.clone()), numerator)
    };
    if remainder.degree() > 1 {
        return None;
    }

    let numerator_constant = remainder
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let numerator_linear = remainder
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let log_scale = numerator_linear / (two * a);
    let remainder_constant = numerator_constant - log_scale.clone() * b;

    let mut decomposition_terms = Vec::new();
    if !quotient.is_zero() {
        decomposition_terms.push(quotient.to_expr(ctx));
    }
    if !log_scale.is_zero() {
        let derivative_part = scale_polynomial(&denominator.derivative(), log_scale);
        decomposition_terms.push(quadratic_partial_fraction_term_expr(
            ctx,
            &derivative_part,
            &denominator,
        )?);
    }
    if !remainder_constant.is_zero() {
        let remainder_part = Polynomial::new(vec![remainder_constant], denominator.var.clone());
        decomposition_terms.push(quadratic_partial_fraction_term_expr(
            ctx,
            &remainder_part,
            &denominator,
        )?);
    }

    if decomposition_terms.len() <= 1 {
        return None;
    }
    Some(build_balanced_add(ctx, &decomposition_terms))
}

fn positive_constant_radius_quadratic_linear_numerator_decomposition_expr(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    if numerator.degree() > 1 {
        return None;
    }

    let (arg, _radius) = positive_square_constant_plus_square_arg(ctx, den, var)
        .or_else(|| positive_square_constant_plus_expanded_square_arg(ctx, den, var))?;
    let (linear_arg, slope_expr) = nonzero_linear_arg_and_slope(ctx, arg, var)?;
    let slope = rational_constant_value(ctx, slope_expr)?;
    let arg_poly = Polynomial::from_expr(ctx, linear_arg, var).ok()?;

    let two = BigRational::from_integer(2.into());
    let derivative_poly = scale_polynomial(&arg_poly, two * slope);
    if derivative_poly.degree() != 1 {
        return None;
    }
    let derivative_linear = derivative_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if derivative_linear.is_zero() {
        return None;
    }

    let numerator_linear = numerator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let log_scale = numerator_linear / derivative_linear;
    let scaled_derivative_poly = scale_polynomial(&derivative_poly, log_scale.clone());
    let remainder = numerator.sub(&scaled_derivative_poly);
    if remainder.degree() > 0 {
        return None;
    }
    let remainder_constant = remainder
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let mut decomposition_terms = Vec::new();
    if !log_scale.is_zero() {
        let derivative_part = scaled_derivative_poly.to_expr(ctx);
        decomposition_terms.push(ctx.add(Expr::Div(derivative_part, den)));
    }
    if !remainder_constant.is_zero() {
        let remainder_part =
            Polynomial::new(vec![remainder_constant], numerator.var.clone()).to_expr(ctx);
        decomposition_terms.push(ctx.add(Expr::Div(remainder_part, den)));
    }

    if decomposition_terms.len() <= 1 {
        return None;
    }
    Some(build_balanced_add(ctx, &decomposition_terms))
}

pub fn integrate_symbolic_is_positive_constant_radius_quadratic_linear_numerator_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    positive_constant_radius_quadratic_linear_numerator_antiderivative(ctx, num, den, var).is_some()
}

pub(super) fn positive_quadratic_over_positive_quadratic_rational_term(
    ctx: &mut Context,
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> ExprId {
    positive_quadratic_power_rational_term(ctx, numerator, denominator, 1)
}

pub(super) fn positive_quadratic_power_rational_term(
    ctx: &mut Context,
    numerator: &Polynomial,
    denominator: &Polynomial,
    power: i64,
) -> ExprId {
    let extract_negative = polynomial_has_only_negative_terms(numerator);
    let numerator = if extract_negative {
        numerator.neg()
    } else {
        numerator.clone()
    };
    let q_expr = denominator.to_expr(ctx);
    let q_expr = if power == 1 {
        q_expr
    } else {
        let power_expr = ctx.num(power);
        ctx.add(Expr::Pow(q_expr, power_expr))
    };
    let denominator_lcm = rational_polynomial_denominator_lcm(&numerator);
    let (numerator, rational_denominator) = if denominator_lcm > num_bigint::BigInt::one() {
        let denominator_scale = BigRational::from_integer(denominator_lcm);
        let numerator = scale_polynomial(&numerator, denominator_scale.clone()).to_expr(ctx);
        let denominator_scale = ctx.add(Expr::Number(denominator_scale));
        (numerator, mul2_raw(ctx, denominator_scale, q_expr))
    } else {
        (numerator.to_expr(ctx), q_expr)
    };

    let rational_term = ctx.add(Expr::Div(numerator, rational_denominator));
    if extract_negative {
        ctx.add(Expr::Neg(rational_term))
    } else {
        rational_term
    }
}

fn rational_polynomial_denominator_lcm(poly: &Polynomial) -> num_bigint::BigInt {
    poly.coeffs
        .iter()
        .filter(|coeff| !coeff.is_zero())
        .fold(num_bigint::BigInt::one(), |acc, coeff| {
            acc.lcm(coeff.denom())
        })
}

pub(super) fn positive_rational_polynomial_content(poly: &Polynomial) -> BigRational {
    let mut numerator_gcd: Option<num_bigint::BigInt> = None;
    let mut denominator_lcm = num_bigint::BigInt::one();

    for coeff in poly.coeffs.iter().filter(|coeff| !coeff.is_zero()) {
        let numerator_abs = coeff.numer().abs();
        numerator_gcd = Some(match numerator_gcd {
            Some(gcd) => gcd.gcd(&numerator_abs),
            None => numerator_abs,
        });
        denominator_lcm = denominator_lcm.lcm(coeff.denom());
    }

    numerator_gcd
        .map(|gcd| BigRational::new(gcd, denominator_lcm))
        .unwrap_or_else(BigRational::zero)
}

pub(super) fn normalized_partial_fraction_linear_factor(factor: &Polynomial) -> Option<Polynomial> {
    if factor.degree() != 1 {
        return None;
    }

    let content = positive_rational_polynomial_content(factor);
    let mut normalized = if content.is_positive() && !content.is_one() {
        factor.div_scalar(&content)
    } else {
        factor.clone()
    };

    if normalized.leading_coeff().is_negative() {
        normalized = normalized.neg();
    }

    Some(normalized)
}

fn grouped_linear_rational_factors(
    denominator: &Polynomial,
) -> Option<Vec<PartialFractionLinearFactor>> {
    const MAX_PARTIAL_FRACTION_DEGREE: usize = 6;

    if denominator.degree() < 1 || denominator.degree() > MAX_PARTIAL_FRACTION_DEGREE {
        return None;
    }

    let mut groups: Vec<PartialFractionLinearFactor> = Vec::new();
    for factor in denominator.factor_rational_roots() {
        let factor = normalized_partial_fraction_linear_factor(&factor)?;

        if let Some(group) = groups.iter_mut().find(|group| group.factor == factor) {
            group.multiplicity += 1;
        } else {
            groups.push(PartialFractionLinearFactor {
                factor,
                multiplicity: 1,
            });
        }
    }

    let total_degree: usize = groups.iter().map(|group| group.multiplicity).sum();
    (total_degree == denominator.degree()).then_some(groups)
}

/// Shared with the algorithmic integration backend's multi-quadratic
/// partial-fraction probe; the decomposition family owns this solver.
pub(crate) fn solve_rational_linear_system(
    mut matrix: Vec<Vec<BigRational>>,
    mut rhs: Vec<BigRational>,
) -> Option<Vec<BigRational>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    for col in 0..n {
        let pivot = (col..n).find(|row| !matrix[*row][col].is_zero())?;
        if pivot != col {
            matrix.swap(pivot, col);
            rhs.swap(pivot, col);
        }

        let pivot_value = matrix[col][col].clone();
        for entry in matrix[col].iter_mut().skip(col) {
            *entry /= pivot_value.clone();
        }
        rhs[col] /= pivot_value;

        for row in 0..n {
            if row == col || matrix[row][col].is_zero() {
                continue;
            }

            let factor = matrix[row][col].clone();
            let pivot_tail: Vec<_> = matrix[col].iter().skip(col).cloned().collect();
            for (entry, pivot_entry) in matrix[row].iter_mut().skip(col).zip(pivot_tail) {
                *entry -= factor.clone() * pivot_entry;
            }
            let pivot_rhs = rhs[col].clone();
            rhs[row] -= factor * pivot_rhs;
        }
    }

    Some(rhs)
}

fn proper_rational_linear_partial_fraction_terms(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<LinearPartialFractionTerms> {
    if numerator.degree() >= denominator.degree() {
        return None;
    }

    let groups = grouped_linear_rational_factors(denominator)?;
    let mut bases = Vec::new();
    for group in &groups {
        for power in 1..=group.multiplicity {
            let divisor = polynomial_pow(&group.factor, power);
            let (quotient, remainder) = denominator.div_rem(&divisor).ok()?;
            if !remainder.is_zero() {
                return None;
            }
            bases.push((group.factor.clone(), power, quotient));
        }
    }

    let unknown_count = bases.len();
    if unknown_count != denominator.degree() {
        return None;
    }

    let mut matrix = vec![vec![BigRational::zero(); unknown_count]; unknown_count];
    for (col, (_, _, quotient)) in bases.iter().enumerate() {
        for (row, coeff) in quotient.coeffs.iter().enumerate().take(unknown_count) {
            matrix[row][col] = coeff.clone();
        }
    }

    let rhs = (0..unknown_count)
        .map(|idx| {
            numerator
                .coeffs
                .get(idx)
                .cloned()
                .unwrap_or_else(BigRational::zero)
        })
        .collect();
    let coefficients = solve_rational_linear_system(matrix, rhs)?;

    Some(
        coefficients
            .into_iter()
            .zip(bases)
            .filter_map(|(coefficient, (factor, power, _))| {
                (!coefficient.is_zero()).then_some((coefficient, factor, power))
            })
            .collect(),
    )
}

fn rational_linear_partial_fraction_decomposition(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<(Polynomial, LinearPartialFractionTerms)> {
    grouped_linear_rational_factors(denominator)?;

    let (quotient, remainder) = if numerator.degree() >= denominator.degree() {
        numerator.div_rem(denominator).ok()?
    } else {
        (Polynomial::zero(numerator.var.clone()), numerator.clone())
    };

    let terms = if remainder.is_zero() {
        Vec::new()
    } else {
        proper_rational_linear_partial_fraction_terms(&remainder, denominator)?
    };

    (!quotient.is_zero() || !terms.is_empty()).then_some((quotient, terms))
}

fn linear_partial_fraction_term_expr(
    ctx: &mut Context,
    coefficient: BigRational,
    factor: &Polynomial,
    power: usize,
) -> Option<ExprId> {
    let factor_expr = factor.to_expr(ctx);
    let denominator = if power == 1 {
        factor_expr
    } else {
        let exponent = ctx.num(power as i64);
        ctx.add(Expr::Pow(factor_expr, exponent))
    };
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, denominator));
    Some(scale_reciprocal_integration_result(
        ctx,
        coefficient,
        reciprocal,
    ))
}

pub fn integrate_symbolic_rational_linear_partial_fraction_decomposition_expr(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (quotient, terms) =
        rational_linear_partial_fraction_decomposition(&numerator, &denominator)?;

    let mut decomposition_terms = Vec::new();
    if !quotient.is_zero() {
        decomposition_terms.push(quotient.to_expr(ctx));
    }
    for (coefficient, factor, power) in terms {
        decomposition_terms.push(linear_partial_fraction_term_expr(
            ctx,
            coefficient,
            &factor,
            power,
        )?);
    }

    match decomposition_terms.len() {
        0 => None,
        1 => Some(decomposition_terms[0]),
        _ => Some(build_balanced_add(ctx, &decomposition_terms)),
    }
}

fn quadratic_partial_fraction_term_expr(
    ctx: &mut Context,
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<ExprId> {
    if numerator.degree() == 0 {
        let coefficient = numerator
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let denominator_expr = denominator.to_expr(ctx);
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, denominator_expr));
        return Some(scale_reciprocal_integration_result(
            ctx,
            coefficient,
            reciprocal,
        ));
    }

    let numerator_expr = if numerator.leading_coeff().is_negative() {
        let positive_numerator = numerator.neg().to_expr(ctx);
        let denominator_expr = denominator.to_expr(ctx);
        let positive_fraction = ctx.add(Expr::Div(positive_numerator, denominator_expr));
        return Some(ctx.add(Expr::Neg(positive_fraction)));
    } else {
        numerator.to_expr(ctx)
    };
    let denominator_expr = denominator.to_expr(ctx);
    Some(ctx.add(Expr::Div(numerator_expr, denominator_expr)))
}

pub fn integrate_symbolic_rational_linear_positive_quadratic_partial_fraction_decomposition_expr(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;

    let mut decomposition_terms = Vec::new();
    if let Some(parts) =
        rational_linear_positive_quadratic_partial_fraction_decomposition(&numerator, &denominator)
    {
        if !parts.quotient.is_zero() {
            decomposition_terms.push(parts.quotient.to_expr(ctx));
        }
        for (coefficient, power) in parts.linear_terms {
            decomposition_terms.push(linear_partial_fraction_term_expr(
                ctx,
                coefficient,
                &parts.linear_factor,
                power,
            )?);
        }
        if !parts.quadratic_numerator.is_zero() {
            decomposition_terms.push(quadratic_partial_fraction_term_expr(
                ctx,
                &parts.quadratic_numerator,
                &parts.quadratic_factor,
            )?);
        }
    } else if let Some(parts) =
        rational_multi_linear_positive_quadratic_partial_fraction_decomposition(
            &numerator,
            &denominator,
        )
    {
        if !parts.quotient.is_zero() {
            decomposition_terms.push(parts.quotient.to_expr(ctx));
        }
        for (factor, coefficient, power) in parts.linear_terms {
            decomposition_terms.push(linear_partial_fraction_term_expr(
                ctx,
                coefficient,
                &factor,
                power,
            )?);
        }
        if !parts.quadratic_numerator.is_zero() {
            decomposition_terms.push(quadratic_partial_fraction_term_expr(
                ctx,
                &parts.quadratic_numerator,
                &parts.quadratic_factor,
            )?);
        }
    } else {
        return None;
    }

    match decomposition_terms.len() {
        0 => None,
        1 => Some(decomposition_terms[0]),
        _ => Some(build_balanced_add(ctx, &decomposition_terms)),
    }
}

fn partial_fraction_linear_term_antiderivative(
    ctx: &mut Context,
    coefficient: BigRational,
    factor: &Polynomial,
    power: usize,
) -> Option<ExprId> {
    let slope = factor.coeffs.get(1)?.clone();
    if slope.is_zero() {
        return None;
    }

    let factor_expr = factor.to_expr(ctx);
    if power == 1 {
        let scale = coefficient / slope;
        let log_arg = partial_fraction_log_factor_expr(ctx, factor);
        let log_abs = ln_abs(ctx, log_arg);
        return Some(scale_rational_term(ctx, scale, log_abs));
    }

    let scale = -coefficient / (slope * BigRational::from_integer((power as i64 - 1).into()));
    let denominator = if power == 2 {
        factor_expr
    } else {
        let exponent = ctx.num((power as i64) - 1);
        ctx.add(Expr::Pow(factor_expr, exponent))
    };
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, denominator));
    Some(scale_reciprocal_integration_result(ctx, scale, reciprocal))
}

pub(super) fn rational_linear_partial_fraction_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (quotient, terms) =
        rational_linear_partial_fraction_decomposition(&numerator, &denominator)?;

    let mut integral_terms = Vec::new();
    if !quotient.is_zero() {
        let quotient_expr = quotient.to_expr(ctx);
        integral_terms.push(integrate_symbolic_expr(ctx, quotient_expr, var)?);
    }

    if quotient.is_zero() {
        if let Some(compact_log_ratio) =
            compact_opposite_simple_log_partial_fraction_antiderivative(ctx, &terms)
        {
            return Some(compact_log_ratio);
        }
        if let Some((compact_log_ratio, first_log_index, second_log_index)) =
            compact_opposite_simple_log_partial_fraction_pair(ctx, &terms)
        {
            integral_terms.push(compact_log_ratio);
            for (index, (coefficient, factor, power)) in terms.into_iter().enumerate() {
                if index == first_log_index || index == second_log_index {
                    continue;
                }
                integral_terms.push(partial_fraction_linear_term_antiderivative(
                    ctx,
                    coefficient,
                    &factor,
                    power,
                )?);
            }
            return match integral_terms.len() {
                0 => None,
                1 => Some(integral_terms[0]),
                _ => {
                    let sum = build_balanced_add(ctx, &integral_terms);
                    Some(cas_ast::hold::wrap_hold(ctx, sum))
                }
            };
        }
    }

    for (coefficient, factor, power) in terms {
        integral_terms.push(partial_fraction_linear_term_antiderivative(
            ctx,
            coefficient,
            &factor,
            power,
        )?);
    }

    match integral_terms.len() {
        0 => None,
        1 => Some(integral_terms[0]),
        _ => {
            let sum = build_balanced_add(ctx, &integral_terms);
            Some(cas_ast::hold::wrap_hold(ctx, sum))
        }
    }
}

pub(super) fn rational_linear_partial_fraction_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return vec![],
    };

    let Ok(numerator) = Polynomial::from_expr(ctx, num, var) else {
        return vec![];
    };
    let Ok(denominator) = Polynomial::from_expr(ctx, den, var) else {
        return vec![];
    };
    if rational_linear_partial_fraction_decomposition(&numerator, &denominator).is_none() {
        return vec![];
    };

    grouped_linear_rational_factors(&denominator)
        .into_iter()
        .flatten()
        .map(|group| group.factor.to_expr(ctx))
        .collect()
}

pub fn integrate_symbolic_is_rational_linear_partial_fraction_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    let Ok(numerator) = Polynomial::from_expr(ctx, num, var) else {
        return false;
    };
    let Ok(denominator) = Polynomial::from_expr(ctx, den, var) else {
        return false;
    };

    rational_linear_partial_fraction_decomposition(&numerator, &denominator).is_some()
}

fn rational_linear_positive_quadratic_partial_fraction_decomposition(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<LinearPositiveQuadraticPartialFraction> {
    let (linear_factor, linear_multiplicity, quadratic_factor) =
        linear_positive_quadratic_factors(denominator)?;

    let (quotient, remainder) = if numerator.degree() >= denominator.degree() {
        numerator.div_rem(denominator).ok()?
    } else {
        (Polynomial::zero(numerator.var.clone()), numerator.clone())
    };

    if remainder.is_zero() {
        return Some(LinearPositiveQuadraticPartialFraction {
            quotient,
            linear_factor,
            linear_terms: Vec::new(),
            quadratic_factor: quadratic_factor.clone(),
            quadratic_numerator: Polynomial::zero(numerator.var.clone()),
        });
    }

    let x_poly = Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        numerator.var.clone(),
    );
    let linear_power = polynomial_pow(&linear_factor, linear_multiplicity);
    let mut bases = Vec::new();
    for power in 1..=linear_multiplicity {
        let divisor = polynomial_pow(&linear_factor, power);
        let (basis, remainder) = denominator.div_rem(&divisor).ok()?;
        if !remainder.is_zero() {
            return None;
        }
        bases.push((Some(power), basis));
    }
    bases.push((None, linear_power.mul(&x_poly)));
    bases.push((None, linear_power));

    let unknown_count = denominator.degree();
    if bases.len() != unknown_count {
        return None;
    }
    let mut matrix = vec![vec![BigRational::zero(); unknown_count]; unknown_count];
    for (col, (_, basis)) in bases.iter().enumerate() {
        for (row, coeff) in basis.coeffs.iter().enumerate().take(unknown_count) {
            matrix[row][col] = coeff.clone();
        }
    }
    let rhs = (0..unknown_count)
        .map(|idx| {
            remainder
                .coeffs
                .get(idx)
                .cloned()
                .unwrap_or_else(BigRational::zero)
        })
        .collect();
    let coefficients = solve_rational_linear_system(matrix, rhs)?;
    let linear_terms = coefficients
        .iter()
        .take(linear_multiplicity)
        .cloned()
        .zip(1..=linear_multiplicity)
        .filter(|(coefficient, _)| !coefficient.is_zero())
        .collect();
    let quadratic_numerator = Polynomial::new(
        vec![
            coefficients[linear_multiplicity + 1].clone(),
            coefficients[linear_multiplicity].clone(),
        ],
        numerator.var.clone(),
    );

    Some(LinearPositiveQuadraticPartialFraction {
        quotient,
        linear_factor,
        linear_terms,
        quadratic_factor,
        quadratic_numerator,
    })
}

fn rational_multi_linear_positive_quadratic_partial_fraction_decomposition(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<MultiLinearPositiveQuadraticPartialFraction> {
    let (groups, quadratic_factor) = multi_linear_positive_quadratic_factors(denominator)?;

    let (quotient, remainder) = if numerator.degree() >= denominator.degree() {
        numerator.div_rem(denominator).ok()?
    } else {
        (Polynomial::zero(numerator.var.clone()), numerator.clone())
    };

    if remainder.is_zero() {
        return Some(MultiLinearPositiveQuadraticPartialFraction {
            quotient,
            linear_factors: groups.iter().map(|group| group.factor.clone()).collect(),
            linear_terms: Vec::new(),
            quadratic_factor,
            quadratic_numerator: Polynomial::zero(numerator.var.clone()),
        });
    }

    let mut linear_power_product = Polynomial::one(numerator.var.clone());
    let mut bases = Vec::new();
    for group in &groups {
        linear_power_product =
            linear_power_product.mul(&polynomial_pow(&group.factor, group.multiplicity));
        for power in 1..=group.multiplicity {
            let divisor = polynomial_pow(&group.factor, power);
            let (basis, remainder) = denominator.div_rem(&divisor).ok()?;
            if !remainder.is_zero() {
                return None;
            }
            bases.push((Some((group.factor.clone(), power)), basis));
        }
    }

    let x_poly = Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        numerator.var.clone(),
    );
    bases.push((None, linear_power_product.mul(&x_poly)));
    bases.push((None, linear_power_product));

    let unknown_count = denominator.degree();
    if bases.len() != unknown_count {
        return None;
    }

    let mut matrix = vec![vec![BigRational::zero(); unknown_count]; unknown_count];
    for (col, (_, basis)) in bases.iter().enumerate() {
        for (row, coeff) in basis.coeffs.iter().enumerate().take(unknown_count) {
            matrix[row][col] = coeff.clone();
        }
    }
    let rhs = (0..unknown_count)
        .map(|idx| {
            remainder
                .coeffs
                .get(idx)
                .cloned()
                .unwrap_or_else(BigRational::zero)
        })
        .collect();
    let coefficients = solve_rational_linear_system(matrix, rhs)?;

    let linear_count: usize = groups.iter().map(|group| group.multiplicity).sum();
    let linear_terms = coefficients
        .iter()
        .take(linear_count)
        .cloned()
        .zip(bases.into_iter().take(linear_count))
        .filter_map(|(coefficient, (linear, _))| {
            let (factor, power) = linear?;
            (!coefficient.is_zero()).then_some((factor, coefficient, power))
        })
        .collect();
    let quadratic_numerator = Polynomial::new(
        vec![
            coefficients[linear_count + 1].clone(),
            coefficients[linear_count].clone(),
        ],
        numerator.var.clone(),
    );

    Some(MultiLinearPositiveQuadraticPartialFraction {
        quotient,
        linear_factors: groups.into_iter().map(|group| group.factor).collect(),
        linear_terms,
        quadratic_factor,
        quadratic_numerator,
    })
}

pub(super) fn rational_linear_positive_quadratic_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let parts = rational_linear_positive_quadratic_partial_fraction_decomposition(
        &numerator,
        &denominator,
    )?;

    let mut integral_terms = Vec::new();
    if !parts.quotient.is_zero() {
        let quotient_expr = parts.quotient.to_expr(ctx);
        integral_terms.push(integrate_symbolic_expr(ctx, quotient_expr, var)?);
    }
    for (coefficient, power) in parts.linear_terms {
        integral_terms.push(partial_fraction_linear_term_antiderivative(
            ctx,
            coefficient,
            &parts.linear_factor,
            power,
        )?);
    }
    if !parts.quadratic_numerator.is_zero() {
        let quadratic_num = parts.quadratic_numerator.to_expr(ctx);
        let quadratic_den = parts.quadratic_factor.to_expr(ctx);
        integral_terms.push(positive_quadratic_linear_numerator_antiderivative(
            ctx,
            quadratic_num,
            quadratic_den,
            var,
        )?);
    }

    match integral_terms.len() {
        0 => None,
        1 => Some(integral_terms[0]),
        _ => {
            let sum = build_balanced_add(ctx, &integral_terms);
            Some(cas_ast::hold::wrap_hold(ctx, sum))
        }
    }
}

pub(super) fn rational_multi_linear_positive_quadratic_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let parts = rational_multi_linear_positive_quadratic_partial_fraction_decomposition(
        &numerator,
        &denominator,
    )?;

    let mut integral_terms = Vec::new();
    if !parts.quotient.is_zero() {
        let quotient_expr = parts.quotient.to_expr(ctx);
        integral_terms.push(integrate_symbolic_expr(ctx, quotient_expr, var)?);
    }
    for (factor, coefficient, power) in parts.linear_terms {
        integral_terms.push(partial_fraction_linear_term_antiderivative(
            ctx,
            coefficient,
            &factor,
            power,
        )?);
    }
    if !parts.quadratic_numerator.is_zero() {
        let quadratic_num = parts.quadratic_numerator.to_expr(ctx);
        let quadratic_den = parts.quadratic_factor.to_expr(ctx);
        integral_terms.push(positive_quadratic_linear_numerator_antiderivative(
            ctx,
            quadratic_num,
            quadratic_den,
            var,
        )?);
    }

    match integral_terms.len() {
        0 => None,
        1 => Some(integral_terms[0]),
        _ => {
            let sum = build_balanced_add(ctx, &integral_terms);
            Some(cas_ast::hold::wrap_hold(ctx, sum))
        }
    }
}

pub(super) fn rational_linear_positive_quadratic_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return vec![],
    };

    let Ok(numerator) = Polynomial::from_expr(ctx, num, var) else {
        return vec![];
    };
    let Ok(denominator) = Polynomial::from_expr(ctx, den, var) else {
        return vec![];
    };
    if let Some(parts) =
        rational_linear_positive_quadratic_partial_fraction_decomposition(&numerator, &denominator)
    {
        return vec![parts.linear_factor.to_expr(ctx)];
    }

    rational_multi_linear_positive_quadratic_partial_fraction_decomposition(
        &numerator,
        &denominator,
    )
    .map(|parts| {
        parts
            .linear_factors
            .into_iter()
            .map(|factor| factor.to_expr(ctx))
            .collect()
    })
    .unwrap_or_default()
}

pub fn integrate_symbolic_is_rational_linear_positive_quadratic_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    let Ok(numerator) = Polynomial::from_expr(ctx, num, var) else {
        return false;
    };
    let Ok(denominator) = Polynomial::from_expr(ctx, den, var) else {
        return false;
    };

    rational_linear_positive_quadratic_partial_fraction_decomposition(&numerator, &denominator)
        .is_some()
        || rational_multi_linear_positive_quadratic_partial_fraction_decomposition(
            &numerator,
            &denominator,
        )
        .is_some()
}

pub fn integrate_symbolic_rational_linear_positive_quadratic_required_nonzero_if_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<Vec<ExprId>> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let Ok(numerator) = Polynomial::from_expr(ctx, num, var) else {
        return None;
    };
    let Ok(denominator) = Polynomial::from_expr(ctx, den, var) else {
        return None;
    };
    if let Some(parts) =
        rational_linear_positive_quadratic_partial_fraction_decomposition(&numerator, &denominator)
    {
        return Some(vec![parts.linear_factor.to_expr(ctx)]);
    }

    rational_multi_linear_positive_quadratic_partial_fraction_decomposition(
        &numerator,
        &denominator,
    )
    .map(|parts| {
        parts
            .linear_factors
            .into_iter()
            .map(|factor| factor.to_expr(ctx))
            .collect()
    })
}

pub(super) fn positive_constant_radius_quadratic_denominator_is_structurally_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let parts = positive_square_constant_plus_square_arg(ctx, expr, var)
        .or_else(|| positive_square_constant_plus_expanded_square_arg(ctx, expr, var));
    let Some((arg, _radius)) = parts else {
        return false;
    };

    nonzero_linear_arg_and_slope(ctx, arg, var).is_some()
}

pub(super) fn multiply_rational_fraction_integral_result(
    ctx: &mut Context,
    scale: ExprId,
    num: ExprId,
    den: ExprId,
) -> Option<ExprId> {
    let mut numerator_scale = rational_constant_value(ctx, scale)?;
    let numerator_base = match ctx.get(num).clone() {
        Expr::Number(value) => {
            numerator_scale *= value;
            None
        }
        Expr::Neg(inner) => {
            numerator_scale = -numerator_scale;
            Some(inner)
        }
        _ => Some(num),
    };

    let mut denominator_factors = Vec::new();
    let mut cancelled_numeric_denominator = false;
    for factor in mul_leaves(ctx, den) {
        if !cancelled_numeric_denominator {
            if let Some(factor_scale) = rational_constant_value(ctx, factor) {
                if !factor_scale.is_zero() {
                    numerator_scale /= factor_scale;
                    cancelled_numeric_denominator = true;
                    continue;
                }
            }
        }
        denominator_factors.push(factor);
    }

    let numerator_coeff = BigRational::from_integer(numerator_scale.numer().clone());
    let denominator_coeff = BigRational::from_integer(numerator_scale.denom().clone());
    let numerator = match numerator_base {
        Some(base) if numerator_coeff.is_one() => base,
        Some(base) if numerator_coeff == BigRational::from_integer((-1).into()) => {
            ctx.add(Expr::Neg(base))
        }
        Some(base) => {
            let coeff = ctx.add(Expr::Number(numerator_coeff));
            mul2_raw(ctx, coeff, base)
        }
        None => ctx.add(Expr::Number(numerator_coeff)),
    };

    if !denominator_coeff.is_one() {
        let denominator_coeff = ctx.add(Expr::Number(denominator_coeff));
        denominator_factors.insert(0, denominator_coeff);
    }

    if denominator_factors.is_empty() {
        Some(numerator)
    } else {
        let denominator = build_balanced_mul(ctx, &denominator_factors);
        Some(ctx.add(Expr::Div(numerator, denominator)))
    }
}

pub(super) fn multiply_rational_factor_if_possible(
    ctx: &mut Context,
    scale: ExprId,
    expr: ExprId,
) -> Option<ExprId> {
    let scale = match ctx.get(scale).clone() {
        Expr::Number(scale) => scale,
        _ => return None,
    };

    let mut factors = mul_leaves(ctx, expr);
    for idx in 0..factors.len() {
        let factor = factors[idx];
        let Expr::Number(value) = ctx.get(factor).clone() else {
            continue;
        };

        let combined = scale.clone() * value;
        if combined.is_zero() {
            return Some(ctx.add(Expr::Number(combined)));
        }
        if combined.is_one() {
            factors.remove(idx);
        } else {
            factors[idx] = ctx.add(Expr::Number(combined));
        }
        return Some(build_product_from_factors(ctx, &factors));
    }

    None
}

/// Split `expr` as `(c, core)` with `expr == c · core`, pulling ALL rational multiplicative constants
/// out of a product tree (recursing through `Neg`/`Mul`, including a constant nested inside a factor,
/// as the chain rule produces — `e^(cos 2x)·(−2·sin 2x)`). A pure rational constant gives `(c, 1)`.
/// Used by the chain-substitution gate so a constant-free core compare recovers a scale `≠ ±1`.
pub(super) fn strip_rational_coefficient(ctx: &mut Context, expr: ExprId) -> (BigRational, ExprId) {
    if let Some(c) = crate::numeric_eval::as_rational_const(ctx, expr) {
        let one = ctx.num(1);
        return (c, one);
    }
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (c, core) = strip_rational_coefficient(ctx, inner);
            (-c, core)
        }
        Expr::Mul(a, b) => {
            let (ca, core_a) = strip_rational_coefficient(ctx, a);
            let (cb, core_b) = strip_rational_coefficient(ctx, b);
            let core = ctx.add(Expr::Mul(core_a, core_b));
            (ca * cb, core)
        }
        _ => (BigRational::one(), expr),
    }
}

/// Multiply `expr` by a rational constant `k`, folding the trivial `±1` cases to keep the output tidy.
pub(super) fn scale_expr_by_rational(ctx: &mut Context, expr: ExprId, k: BigRational) -> ExprId {
    use num_traits::One;
    if k.is_one() {
        expr
    } else if k == -BigRational::one() {
        negate_scalar_expr(ctx, expr)
    } else {
        let coeff = ctx.add(Expr::Number(k));
        ctx.add(Expr::Mul(coeff, expr))
    }
}
