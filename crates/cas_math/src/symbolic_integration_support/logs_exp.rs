//! `symbolic_integration_support`: familia `logs_exp`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn valid_constant_log_base_ln(
    ctx: &mut Context,
    base: ExprId,
) -> Option<Option<ExprId>> {
    valid_constant_log_base_ln_from_rational_value(ctx, base, rational_constant_value(ctx, base))
}

pub(super) fn affine_constant_base_log_antiderivative(
    ctx: &mut Context,
    log_expr: ExprId,
    arg: ExprId,
    base_ln: Option<ExprId>,
    var: &str,
) -> Option<ExprId> {
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    if contains_named_var(ctx, a, var) {
        return None;
    }
    affine_constant_base_log_antiderivative_from_slope(
        ctx,
        log_expr,
        arg,
        base_ln,
        a,
        rational_constant_value(ctx, a),
    )
}

/// `int e^sqrt(x) dx = 2 (sqrt(x) - 1) e^sqrt(x)`, via `u = sqrt(x)`
/// (`int e^sqrt(x) dx = 2 int u e^u du`). `e^sqrt(x)` is `Pow(E, sqrt(x))`, not a
/// `Function`, so it is dispatched from the Pow arm. Self-gates if the delegated
/// tail `int u e^u du` does not resolve.
pub(super) fn exp_of_sqrt_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !matches!(ctx.get(base), Expr::Constant(Constant::E)) {
        return None;
    }
    let radicand = sqrt_like_radicand(ctx, exp)?;
    if !is_var(ctx, radicand, var) {
        return None;
    }

    // u_integrand = u * e^u.
    let var_expr = ctx.var(var);
    let e = ctx.add(Expr::Constant(Constant::E));
    let e_pow_var = ctx.add(Expr::Pow(e, var_expr));
    let var_expr = ctx.var(var);
    let u_integrand = mul2_raw(ctx, var_expr, e_pow_var);
    complete_sqrt_substitution(ctx, u_integrand, exp, var)
}

/// |e^t| = e^t for real t: drop Abs wrappers the back-substitution
/// leaves around exponentials (ln(|u|) -> ln(|e^(c x)|)).
pub(super) fn strip_redundant_exponential_abs(ctx: &mut Context, expr: ExprId) -> ExprId {
    let node = ctx.get(expr).clone();
    if let Expr::Function(fn_id, args) = &node {
        if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Abs)) {
            let inner = ctx.get(args[0]).clone();
            let is_exponential = match inner {
                Expr::Pow(base, _) => {
                    matches!(ctx.get(base), Expr::Constant(cas_ast::Constant::E))
                }
                Expr::Function(inner_fn, inner_args) => {
                    inner_args.len() == 1
                        && matches!(ctx.builtin_of(inner_fn), Some(BuiltinFn::Exp))
                }
                _ => false,
            };
            if is_exponential {
                return strip_redundant_exponential_abs(ctx, args[0]);
            }
        }
    }
    match node {
        Expr::Add(l, r) => {
            let l = strip_redundant_exponential_abs(ctx, l);
            let r = strip_redundant_exponential_abs(ctx, r);
            ctx.add(Expr::Add(l, r))
        }
        Expr::Sub(l, r) => {
            let l = strip_redundant_exponential_abs(ctx, l);
            let r = strip_redundant_exponential_abs(ctx, r);
            ctx.add(Expr::Sub(l, r))
        }
        Expr::Mul(l, r) => {
            let l = strip_redundant_exponential_abs(ctx, l);
            let r = strip_redundant_exponential_abs(ctx, r);
            ctx.add(Expr::Mul(l, r))
        }
        Expr::Div(l, r) => {
            let l = strip_redundant_exponential_abs(ctx, l);
            let r = strip_redundant_exponential_abs(ctx, r);
            ctx.add(Expr::Div(l, r))
        }
        Expr::Neg(inner) => {
            let inner = strip_redundant_exponential_abs(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Pow(base, exponent) => {
            let base = strip_redundant_exponential_abs(ctx, base);
            let exponent = strip_redundant_exponential_abs(ctx, exponent);
            ctx.add(Expr::Pow(base, exponent))
        }
        Expr::Function(fn_id, args) => {
            let args: Vec<_> = args
                .iter()
                .map(|arg| strip_redundant_exponential_abs(ctx, *arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    }
}

/// True when expr is built purely from rational operations over
/// e^(k*var) atoms (k rational nonzero) and var-free subtrees; collects
/// every k. Any other occurrence of var refuses.
pub(super) fn collect_exponential_rational_slopes(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    slopes: &mut Vec<BigRational>,
) -> bool {
    if let Some(slope) = exponential_atom_slope(ctx, expr, var) {
        slopes.push(slope);
        return true;
    }
    if !contains_named_var(ctx, expr, var) {
        return true;
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_exponential_rational_slopes(ctx, l, var, slopes)
                && collect_exponential_rational_slopes(ctx, r, var, slopes)
        }
        Expr::Neg(inner) => collect_exponential_rational_slopes(ctx, inner, var, slopes),
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
            collect_exponential_rational_slopes(ctx, base, var, slopes)
        }
        _ => false,
    }
}

/// e^(k*var) (or exp(k*var)) with rational nonzero k and zero offset.
pub(super) fn exponential_atom_slope(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<BigRational> {
    let exponent = match ctx.get(expr).clone() {
        Expr::Pow(base, exponent)
            if matches!(ctx.get(base), Expr::Constant(cas_ast::Constant::E)) =>
        {
            exponent
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp)) =>
        {
            args[0]
        }
        _ => return None,
    };
    let (slope_expr, offset) = get_linear_coeffs(ctx, exponent, var)?;
    if !is_number(ctx, offset, 0) {
        return None;
    }
    let slope = rational_constant_value(ctx, slope_expr)?;
    (!slope.is_zero()).then_some(slope)
}

pub(super) fn multiply_log_derivative_correction(
    ctx: &mut Context,
    base: ExprId,
    correction: ExprId,
) -> ExprId {
    if let Expr::Div(num, den) = ctx.get(correction).clone() {
        if matches!(ctx.get(num), Expr::Number(value) if value.is_one()) {
            return ctx.add(Expr::Div(base, den));
        }
    }

    mul2_raw(ctx, base, correction)
}

pub(super) fn log_power_term(ctx: &mut Context, log_expr: ExprId, degree: u32) -> ExprId {
    match degree {
        0 => ctx.num(1),
        1 => log_expr,
        _ => {
            let degree = ctx.num(i64::from(degree));
            ctx.add(Expr::Pow(log_expr, degree))
        }
    }
}

fn matching_ln_argument_polynomial(
    ctx: &Context,
    expr: ExprId,
    target_arg_poly: &Polynomial,
    var: &str,
) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(cas_ast::hold::unwrap_hold(ctx, expr)) else {
        return false;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) {
        return false;
    }
    Polynomial::from_expr(ctx, args[0], var)
        .ok()
        .is_some_and(|arg_poly| arg_poly == *target_arg_poly)
}

pub(super) fn scaled_matching_ln_coefficient(
    ctx: &Context,
    expr: ExprId,
    target_arg_poly: &Polynomial,
    var: &str,
) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    if matching_ln_argument_polynomial(ctx, expr, target_arg_poly, var) {
        return Some(BigRational::one());
    }

    match ctx.get(expr) {
        Expr::Mul(left, right) => match (ctx.get(*left), ctx.get(*right)) {
            (Expr::Number(scale), _) => {
                scaled_matching_ln_coefficient(ctx, *right, target_arg_poly, var)
                    .map(|coeff| scale.clone() * coeff)
            }
            (_, Expr::Number(scale)) => {
                scaled_matching_ln_coefficient(ctx, *left, target_arg_poly, var)
                    .map(|coeff| scale.clone() * coeff)
            }
            _ => None,
        },
        Expr::Div(num, den) => {
            let Expr::Number(den) = ctx.get(*den) else {
                return None;
            };
            if den.is_zero() {
                return None;
            }
            scaled_matching_ln_coefficient(ctx, *num, target_arg_poly, var)
                .map(|scale| scale / den.clone())
        }
        Expr::Neg(inner) => {
            scaled_matching_ln_coefficient(ctx, *inner, target_arg_poly, var).map(|scale| -scale)
        }
        _ => None,
    }
}

pub(super) fn natural_log_argument(ctx: &Context, log_expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(log_expr) else {
        return None;
    };
    if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) {
        Some(args[0])
    } else {
        None
    }
}

pub(super) fn polynomial_log_power_product_required_positive_condition(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    if !integrate_symbolic_is_log_power_product_substitution_target(ctx, expr, var) {
        return Vec::new();
    }

    let log_base = polynomial_log_power_product_log_base(ctx, expr);
    let Some(log_base) = log_base else {
        return Vec::new();
    };
    let Ok(base_poly) = Polynomial::from_expr(ctx, log_base, var) else {
        return vec![log_base];
    };
    if is_strictly_positive_quadratic(&base_poly) {
        Vec::new()
    } else {
        vec![log_base]
    }
}

fn polynomial_log_power_product_log_base(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    for factor in mul_leaves(ctx, expr) {
        if let Some((_, log_base, _, _)) = log_power_substitution_factor_parts(ctx, factor) {
            return Some(log_base);
        }
    }

    let (_, log_base, _, _, _) = additive_common_log_power_cofactor_with_correction(ctx, expr)?;
    Some(log_base)
}

pub(super) fn log_derivative_correction_power(
    ctx: &mut Context,
    correction: ExprId,
    power: u32,
) -> ExprId {
    match power {
        0 => ctx.num(1),
        1 => correction,
        _ if is_number(ctx, correction, 1) => ctx.num(1),
        _ => match ctx.get(correction).clone() {
            Expr::Div(num, den) if is_number(ctx, num, 1) => {
                let exponent = ctx.num(i64::from(power));
                let denominator_power = ctx.add(Expr::Pow(den, exponent));
                let one = ctx.num(1);
                ctx.add(Expr::Div(one, denominator_power))
            }
            _ => {
                let exponent = ctx.num(i64::from(power));
                ctx.add(Expr::Pow(correction, exponent))
            }
        },
    }
}

pub(super) fn scale_log_power_term(ctx: &mut Context, scale: BigRational, term: ExprId) -> ExprId {
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        if is_number(ctx, num, 1) {
            let scale_expr = ctx.add(Expr::Number(scale));
            return ctx.add(Expr::Div(scale_expr, den));
        }
    }

    if let Some((numerator, denominator)) = split_unit_reciprocal_factor(ctx, term) {
        let numerator = scale_rational_term(ctx, scale, numerator);
        return ctx.add(Expr::Div(numerator, denominator));
    }

    scale_rational_term(ctx, scale, term)
}

pub(super) fn additive_common_log_power_cofactor(
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

pub(super) fn additive_common_log_power_cofactor_with_correction(
    ctx: &mut Context,
    num: ExprId,
) -> Option<(ExprId, ExprId, ExprId, u32, ExprId)> {
    let add_view = AddView::from_expr(ctx, num);
    if add_view.terms.len() < 2 {
        return None;
    }

    let mut common: Option<(ExprId, ExprId, ExprId, u32)> = None;
    let mut cofactor_terms = Vec::with_capacity(add_view.terms.len());

    for (term, sign) in add_view.terms {
        let factors = mul_leaves(ctx, term);
        let mut term_cofactor = None;

        for (log_index, factor) in factors.iter().enumerate() {
            let Some((log_expr, log_base, correction, power)) =
                log_power_substitution_factor_parts(ctx, *factor)
            else {
                continue;
            };

            if let Some((common_log, _, common_correction, common_power)) = common {
                if power != common_power
                    || compare_expr(ctx, log_expr, common_log) != Ordering::Equal
                    || compare_expr(ctx, correction, common_correction) != Ordering::Equal
                {
                    continue;
                }
            } else {
                common = Some((log_expr, log_base, correction, power));
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

    let (log_expr, log_base, correction, power) = common?;
    let cofactor = build_balanced_add(ctx, &cofactor_terms);
    Some((log_expr, log_base, correction, power, cofactor))
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

pub(super) fn polynomial_log_derivative_power_antiderivative(
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

pub(super) fn exp_like_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

/// integrate(c / e^(a*x+b), x) = -(c/a) / e^(a*x+b): the simplifier
/// normalizes e^(-x) to 1/e^x, so the reciprocal-exponential shape must
/// be owned here (numeric nonzero slope only; e^(x^2)-style exponents
/// and variable numerators fall through).
pub(super) fn reciprocal_exp_linear_antiderivative(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    var: &str,
) -> Option<ExprId> {
    let exponent = exp_like_arg(ctx, denominator)?;
    if contains_named_var(ctx, numerator, var) {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, exponent, var).ok()?;
    let slope = nonzero_linear_polynomial_slope(&poly)?;
    let scale = ctx.add(Expr::Number(-slope.recip()));
    let scaled_numerator = mul2_raw(ctx, scale, numerator);
    Some(ctx.add(Expr::Div(scaled_numerator, denominator)))
}

pub(super) fn linear_times_exp_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = linear_exp_linear_product_parts(ctx, expr, var).ok()??;
    let cofactor_slope = nonzero_linear_polynomial_slope(&parts.cofactor_poly)?;

    let correction = cofactor_slope / (parts.arg_slope.clone() * parts.arg_slope.clone());
    let correction_poly = Polynomial::new(vec![correction], var.to_string());
    let inner_poly = parts
        .cofactor_poly
        .div_scalar(&parts.arg_slope)
        .sub(&correction_poly);
    let inner = inner_poly.to_expr(ctx);
    Some(mul2_raw(ctx, parts.exp_factor, inner))
}

pub fn integrate_symbolic_is_linear_times_exp_linear_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    linear_times_exp_linear_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn polynomial_times_exp_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts =
        polynomial_exp_linear_product_parts(ctx, expr, var, ExpByPartsCofactorFailure::Stop)
            .ok()??;

    let inner_poly = polynomial_exp_by_parts_inner(&parts.cofactor_poly, &parts.arg_slope);
    let exp_factor = ctx.call_builtin(BuiltinFn::Exp, vec![parts.exp_arg]);
    Some(exp_times_polynomial_with_rational_content_factored(
        ctx,
        exp_factor,
        &inner_poly,
    ))
}

fn is_polynomial_times_exp_linear_target(ctx: &mut Context, expr: ExprId, var: &str) -> bool {
    let Ok(Some(_parts)) =
        polynomial_exp_linear_product_parts(ctx, expr, var, ExpByPartsCofactorFailure::Skip)
    else {
        return false;
    };

    true
}

pub fn integrate_symbolic_is_polynomial_times_exp_linear_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    is_polynomial_times_exp_linear_target(ctx, expr, var)
}

fn exp_times_polynomial_with_rational_content_factored(
    ctx: &mut Context,
    exp_factor: ExprId,
    poly: &Polynomial,
) -> ExprId {
    let mut denominator_lcm = num_bigint::BigInt::one();
    for coeff in &poly.coeffs {
        denominator_lcm = denominator_lcm.lcm(coeff.denom());
    }

    if denominator_lcm.is_one() {
        let inner = poly.to_expr(ctx);
        return mul2_raw(ctx, exp_factor, inner);
    }

    let denominator = BigRational::from_integer(denominator_lcm);
    let scaled_poly = poly.mul(&Polynomial::new(
        vec![denominator.clone()],
        poly.var.clone(),
    ));
    let inner = scaled_poly.to_expr(ctx);
    let numerator = mul2_raw(ctx, exp_factor, inner);
    let denominator_expr = ctx.add(Expr::Number(denominator));
    ctx.add(Expr::Div(numerator, denominator_expr))
}

pub(super) fn polynomial_square_minus_constant_log_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) =
        exact_positive_constant_minus_polynomial_square(&denominator.neg())?;
    if offset_square.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let Some(offset) = exact_rational_sqrt(&offset_square) else {
        // Irrational sqrt(c): build the log ratio with a symbolic
        // radical. p'(x)/(p(x)^2 - c) -> scale/(2 sqrt(c)) *
        // ln|(p - sqrt(c))/(p + sqrt(c))| for sqrt(c) the positive
        // surd. Covers 1/(x^2 - 2), 1/(x^2 + 2x - 1), and the
        // 1/(u^2 - 2) kernel the x^4+1 substitution needs.
        let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;
        let arg_expr = arg_poly.to_expr(ctx);
        let numerator_arg = ctx.add(Expr::Sub(arg_expr, offset_expr));
        let denominator_arg = ctx.add(Expr::Add(arg_expr, offset_expr));
        let ratio = ctx.add(Expr::Div(numerator_arg, denominator_arg));
        let log_abs_ratio = ln_abs(ctx, ratio);
        let two = BigRational::from_integer(2.into());
        let half_scale = scale / two;
        if half_scale.is_one() {
            return Some(ctx.add(Expr::Div(log_abs_ratio, offset_expr)));
        }
        let half_scale_expr = ctx.add(Expr::Number(half_scale));
        let coefficient = ctx.add(Expr::Div(half_scale_expr, offset_expr));
        return Some(mul2_raw(ctx, coefficient, log_abs_ratio));
    };
    if offset.is_zero() {
        return None;
    }

    let offset_poly = Polynomial::new(vec![offset.clone()], arg_poly.var.clone());
    let mut numerator_poly = arg_poly.sub(&offset_poly);
    let mut denominator_poly = arg_poly.add(&offset_poly);
    let numerator_content = positive_rational_polynomial_content(&numerator_poly);
    let denominator_content = positive_rational_polynomial_content(&denominator_poly);
    if numerator_content == denominator_content
        && numerator_content.is_positive()
        && !numerator_content.is_one()
    {
        numerator_poly = numerator_poly.div_scalar(&numerator_content);
        denominator_poly = denominator_poly.div_scalar(&denominator_content);
    }

    let numerator_arg = numerator_poly.to_expr(ctx);
    let denominator_arg = denominator_poly.to_expr(ctx);
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

pub(super) fn polynomial_log_derivative_antiderivative(
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

fn build_polynomial_log_reciprocal_derivative_integral(
    ctx: &mut Context,
    num: ExprId,
    log_base_arg: ExprId,
    log_expr: ExprId,
    log_power: u32,
    denominator_scale: BigRational,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let base_poly = Polynomial::from_expr(ctx, log_base_arg, var).ok()?;
    if base_poly.degree() == 0 {
        return None;
    }
    if denominator_scale.is_zero() {
        return None;
    }
    let scale = constant_polynomial_ratio(&numerator, &base_poly.derivative())? / denominator_scale;
    if scale.is_zero() {
        return None;
    }

    if log_power > 1 {
        let denominator = if log_power == 2 {
            log_expr
        } else {
            let power = ctx.add(Expr::Number(BigRational::from_integer(
                (log_power - 1).into(),
            )));
            ctx.add(Expr::Pow(log_expr, power))
        };
        let denominator_scale = BigRational::from_integer((log_power - 1).into());
        let result_scale = -scale / denominator_scale;
        let numerator = ctx.add(Expr::Number(BigRational::from_integer(
            result_scale.numer().clone(),
        )));
        let denominator_scale = BigRational::from_integer(result_scale.denom().clone());
        let denominator = if denominator_scale.is_one() {
            denominator
        } else {
            let denominator_scale = ctx.add(Expr::Number(denominator_scale));
            mul2_raw(ctx, denominator_scale, denominator)
        };
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let integral = ln_abs(ctx, log_expr);
    if scale.is_one() {
        return Some(integral);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

pub(super) fn polynomial_log_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (denominator_scale, den_core) = split_rational_denominator_scale(ctx, den);
    if let Some((log_expr, log_base_arg, log_power, cofactor)) =
        additive_common_log_power_cofactor(ctx, den_core)
    {
        if log_power > 1 && polynomial_exprs_equal(ctx, log_base_arg, cofactor, var) {
            return build_polynomial_log_reciprocal_derivative_integral(
                ctx,
                num,
                log_base_arg,
                log_expr,
                log_power,
                denominator_scale,
                var,
            );
        }
    }

    let factored_den = factor(ctx, den_core);
    let factors = mul_leaves(ctx, factored_den);
    if factors.len() == 2 {
        let candidate = match (
            natural_log_power_factor_parts(ctx, factors[0]),
            natural_log_power_factor_parts(ctx, factors[1]),
        ) {
            (Some((log_expr, log_base_arg, power)), None)
                if polynomial_exprs_equal(ctx, log_base_arg, factors[1], var) =>
            {
                Some((factors[1], log_expr, power))
            }
            (None, Some((log_expr, log_base_arg, power)))
                if polynomial_exprs_equal(ctx, factors[0], log_base_arg, var) =>
            {
                Some((factors[0], log_expr, power))
            }
            _ => None,
        };

        if let Some((log_base_arg, log_expr, log_power)) = candidate {
            if let Some(integral) = build_polynomial_log_reciprocal_derivative_integral(
                ctx,
                num,
                log_base_arg,
                log_expr,
                log_power,
                denominator_scale,
                var,
            ) {
                return Some(integral);
            }
        }
    }

    None
}

pub fn integrate_symbolic_is_polynomial_log_reciprocal_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    polynomial_log_reciprocal_derivative_antiderivative(ctx, *num, *den, var).is_some()
}

pub(super) fn positive_quadratic_log_argument(
    ctx: &mut Context,
    denominator: &Polynomial,
    original_den: ExprId,
) -> ExprId {
    let content = positive_rational_polynomial_content(denominator);
    if content.is_positive() && !content.is_one() {
        return denominator.div_scalar(&content).to_expr(ctx);
    }

    original_den
}

pub(super) fn partial_fraction_log_factor_expr(ctx: &mut Context, factor: &Polynomial) -> ExprId {
    let content = positive_rational_polynomial_content(factor);
    let mut normalized = if content.is_positive() && !content.is_one() {
        factor.div_scalar(&content)
    } else {
        factor.clone()
    };

    if normalized.leading_coeff().is_negative() {
        normalized = normalized.neg();
    }

    normalized.to_expr(ctx)
}

pub(super) fn compact_opposite_simple_log_partial_fraction_pair(
    ctx: &mut Context,
    terms: &LinearPartialFractionTerms,
) -> Option<(ExprId, usize, usize)> {
    let simple_indices: Vec<_> = terms
        .iter()
        .enumerate()
        .filter_map(|(index, (_, _, power))| (*power == 1).then_some(index))
        .collect();
    if simple_indices.len() != 2 {
        return None;
    }

    let index_a = simple_indices[0];
    let index_b = simple_indices[1];
    let (coefficient_a, factor_a, _) = &terms[index_a];
    let (coefficient_b, factor_b, _) = &terms[index_b];

    let slope_a = factor_a.coeffs.get(1)?.clone();
    let slope_b = factor_b.coeffs.get(1)?.clone();
    if slope_a.is_zero() || slope_b.is_zero() {
        return None;
    }

    let scale_a = coefficient_a.clone() / slope_a;
    let scale_b = coefficient_b.clone() / slope_b;
    if scale_a.is_zero() || scale_a != -scale_b.clone() {
        return None;
    }

    let (numerator_factor, denominator_factor, scale) = if scale_a.is_positive() {
        (factor_a, factor_b, scale_a)
    } else {
        (factor_b, factor_a, -scale_a)
    };
    let numerator_arg = partial_fraction_log_factor_expr(ctx, numerator_factor);
    let denominator_arg = partial_fraction_log_factor_expr(ctx, denominator_factor);
    let ratio = ctx.add(Expr::Div(numerator_arg, denominator_arg));
    let log_abs_ratio = ln_abs(ctx, ratio);

    Some((
        scale_rational_term(ctx, scale, log_abs_ratio),
        index_a,
        index_b,
    ))
}

pub(super) fn compact_opposite_simple_log_partial_fraction_antiderivative(
    ctx: &mut Context,
    terms: &LinearPartialFractionTerms,
) -> Option<ExprId> {
    if terms.len() != 2 {
        return None;
    }

    let (compact_log_ratio, _, _) = compact_opposite_simple_log_partial_fraction_pair(ctx, terms)?;
    Some(compact_log_ratio)
}

/// Recursively fold `ln(E) → 1` and unit products (`a·1 → a`, `1·a → a`) in `expr`.
/// `ln(E) = 1` is exact, so this never changes the value; it only cancels the artefact the
/// general power rule leaves when differentiating a `Pow(E, g)` candidate antiderivative.
pub(super) fn reduce_ln_e_and_unit_products(ctx: &mut Context, expr: ExprId) -> ExprId {
    let is_one = |ctx: &Context, e: ExprId| matches!(ctx.get(e), Expr::Number(n) if n.numer() == &1.into() && n.denom() == &1.into());
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() == 1
                && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln)
                && matches!(ctx.get(args[0]), Expr::Constant(cas_ast::Constant::E))
            {
                return ctx.num(1);
            }
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&arg| reduce_ln_e_and_unit_products(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, new_args))
        }
        Expr::Mul(a, b) => {
            let a = reduce_ln_e_and_unit_products(ctx, a);
            let b = reduce_ln_e_and_unit_products(ctx, b);
            // Bubble any sign to the top so `e^g · (−sin g')` and `−(sin g' · e^g)` compare equal.
            let (a, a_neg) = usub_strip_top_neg(ctx, a);
            let (b, b_neg) = usub_strip_top_neg(ctx, b);
            let product = if is_one(ctx, a) {
                b
            } else if is_one(ctx, b) {
                a
            } else {
                ctx.add(Expr::Mul(a, b))
            };
            if a_neg ^ b_neg {
                ctx.add(Expr::Neg(product))
            } else {
                product
            }
        }
        Expr::Add(a, b) => {
            let a = reduce_ln_e_and_unit_products(ctx, a);
            let b = reduce_ln_e_and_unit_products(ctx, b);
            ctx.add(Expr::Add(a, b))
        }
        Expr::Sub(a, b) => {
            let a = reduce_ln_e_and_unit_products(ctx, a);
            let b = reduce_ln_e_and_unit_products(ctx, b);
            ctx.add(Expr::Sub(a, b))
        }
        Expr::Div(a, b) => {
            let a = reduce_ln_e_and_unit_products(ctx, a);
            let b = reduce_ln_e_and_unit_products(ctx, b);
            ctx.add(Expr::Div(a, b))
        }
        Expr::Pow(a, b) => {
            let a = reduce_ln_e_and_unit_products(ctx, a);
            let b = reduce_ln_e_and_unit_products(ctx, b);
            ctx.add(Expr::Pow(a, b))
        }
        Expr::Neg(inner) => {
            let inner = reduce_ln_e_and_unit_products(ctx, inner);
            let (core, negated) = usub_strip_top_neg(ctx, inner);
            if negated {
                core
            } else {
                ctx.add(Expr::Neg(core))
            }
        }
        _ => expr,
    }
}
