//! `symbolic_integration_support`: familia `radicals`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn signed_sqrt_like_radicand(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let radicand = sqrt_like_radicand(ctx, *inner)?;
            Some((radicand, -BigRational::one()))
        }
        _ => sqrt_like_radicand(ctx, expr).map(|radicand| (radicand, BigRational::one())),
    }
}

/// `(q)^(-1/2)` whose radicand `q` is a degree-2 polynomial in `var` — the reciprocal-sqrt factor of
/// the `p(x)/√(quadratic)` family (`x²+1`, `1-x²`, `x²-1`, `x²+2x+5`).
fn reciprocal_sqrt_quadratic_radicand(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let base = reciprocal_sqrt_like_radicand(ctx, expr)?;
    let poly = Polynomial::from_expr(ctx, base, var).ok()?;
    (poly.degree() == 2).then_some(base)
}

/// `p(x)·(q)^(-1/2)` with a SUM numerator `p` and a degree-2 radicand `q`: distribute the sum over
/// the radical so each `term·(q)^(-1/2)` hits the existing sqrt-quadratic antiderivatives
/// (`x/√q → √q`, `c/√q → asinh/arcsin/acosh`, `x²/√q → reduction`), then add by linearity. Only a
/// genuine multi-term numerator splits (a single term already has its dedicated owner); bails unless
/// EVERY piece integrates, so a stray non-integrable term leaves the case for later rules.
pub(super) fn linear_numerator_over_reciprocal_sqrt_quadratic_antiderivative(
    ctx: &mut Context,
    l: ExprId,
    r: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (radical, numerator) = if reciprocal_sqrt_quadratic_radicand(ctx, l, var).is_some() {
        (l, r)
    } else if reciprocal_sqrt_quadratic_radicand(ctx, r, var).is_some() {
        (r, l)
    } else {
        return None;
    };
    let terms = signed_additive_terms(ctx, numerator);
    if terms.len() < 2 {
        return None;
    }
    let mut acc: Option<ExprId> = None;
    for (positive, term) in terms {
        let piece = ctx.add(Expr::Mul(term, radical));
        let integral = integrate_symbolic_expr(ctx, piece, var)?;
        let signed = if positive {
            integral
        } else {
            ctx.add(Expr::Neg(integral))
        };
        acc = Some(match acc {
            None => signed,
            Some(previous) => ctx.add(Expr::Add(previous, signed)),
        });
    }
    acc
}

pub(super) fn sqrt_var_times_symbolic_square_shift_denominator(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<SymbolicSquareShiftDenominator> {
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut saw_sqrt_var = false;
    let mut square_shift = None;

    for factor in factors {
        if sqrt_like_radicand(ctx, factor).is_some_and(|radicand| is_var(ctx, radicand, var)) {
            if saw_sqrt_var {
                return None;
            }
            saw_sqrt_var = true;
        } else if let Some(parts) = symbolic_square_shift_argument_parts(ctx, factor, var) {
            if square_shift.is_some() {
                return None;
            }
            square_shift = Some(parts);
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    let (parameter, argument, argument_scale) = square_shift?;
    saw_sqrt_var.then_some(SymbolicSquareShiftDenominator {
        scale,
        parameter,
        argument,
        argument_scale,
    })
}

pub(super) fn sqrt_var_times_positive_linear_denominator(
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

pub(super) fn expanded_sqrt_var_times_positive_linear_denominator(
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

pub(super) fn scale_rational_sqrt_term(
    ctx: &mut Context,
    scale: BigRational,
    sqrt_factor: BigRational,
    expr: ExprId,
) -> Option<ExprId> {
    let sqrt_expr = positive_rational_sqrt_expr(ctx, &sqrt_factor)?;
    let scaled_sqrt = scale_rational_term(ctx, scale, sqrt_expr);
    Some(mul2_raw(ctx, scaled_sqrt, expr))
}

/// `int f(sqrt(x)) dx` for a unary builtin `f`, via the substitution
/// `u = sqrt(x)` (`x = u^2`, `dx = 2u du`):
///
/// ```text
/// int f(sqrt(x)) dx = 2 int u f(u) du = 2 H(sqrt(x)),  H(u) = int u f(u) du
/// ```
///
/// `H` is delegated to the integrator (the monomial-times-{trig, hyperbolic,
/// inverse-trig} owners) and the result is back-substituted `u -> sqrt(x)`.
/// Self-gates to an honest residual if `H` does not resolve (e.g. `f = tan`,
/// where `int u tan(u) du` is non-elementary). Covers sin/cos/sinh/cosh and the
/// inverse-trig family.
pub(super) fn function_of_sqrt_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    // The argument must be sqrt of the bare variable.
    let radicand = sqrt_like_radicand(ctx, arg)?;
    if !is_var(ctx, radicand, var) {
        return None;
    }

    // int f(sqrt(x)) dx = 2 int u f(u) du: delegate the monomial-times-f tail.
    let var_expr = ctx.var(var);
    let f_of_var = ctx.call_builtin(builtin, vec![var_expr]);
    let u_integrand = mul2_raw(ctx, var_expr, f_of_var);
    complete_sqrt_substitution(ctx, u_integrand, arg, var)
}

/// `int H(sqrt(x)) / sqrt(x) dx = 2 int H(u) du`, via `u = sqrt(x)`: the
/// `1/sqrt(x)` cofactor cancels the `u` from `dx = 2u du`, so the delegate
/// integrand is just `H(u)`. The engine normalizes `H(sqrt(x))/sqrt(x)` to the
/// product `H(sqrt(x)) * x^(-1/2)`, so this is dispatched from the Mul arm with
/// the two factors. Covers `H = exp` (`Pow(E, .)`) and the unary builtins whose
/// primitive resolves (e.g. `int e^sqrt(x)/sqrt(x) = 2 e^sqrt(x)`,
/// `int sin(sqrt(x))/sqrt(x) = -2 cos(sqrt(x))`). Self-gates otherwise.
pub(super) fn function_over_sqrt_antiderivative(
    ctx: &mut Context,
    factor_a: ExprId,
    factor_b: ExprId,
    var: &str,
) -> Option<ExprId> {
    // One factor must be the reciprocal root x^(-1/2); the other is H(sqrt(x)).
    let h_expr = if is_negative_half_power_of_var(ctx, factor_a, var) {
        factor_b
    } else if is_negative_half_power_of_var(ctx, factor_b, var) {
        factor_a
    } else {
        return None;
    };
    let (h_of_var, sqrt_arg) = function_of_sqrt_parts(ctx, h_expr, var)?;
    // u_integrand = u * (H(u)/u) = H(u); back-substitute against the sqrt(x) arg.
    complete_sqrt_substitution(ctx, h_of_var, sqrt_arg, var)
}

/// `int H(sqrt(x)) / sqrt(x) dx` when the integrand survives as a literal
/// `Div(H(sqrt(x)), sqrt(x))` (e.g. on recursive integrator entry, before the
/// `A/sqrt(x) -> A*x^(-1/2)` normalization). Same `u = sqrt(x)` reduction as the
/// Mul form: `2 int H(u) du`.
pub(super) fn function_over_sqrt_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let den_radicand = sqrt_like_radicand(ctx, den)?;
    if !is_var(ctx, den_radicand, var) {
        return None;
    }
    let (h_of_var, sqrt_arg) = function_of_sqrt_parts(ctx, num, var)?;
    complete_sqrt_substitution(ctx, h_of_var, sqrt_arg, var)
}

pub(super) fn sqrt_polynomial_derivative_quotient_scale(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    sqrt_arg: ExprId,
    var: &str,
) -> Option<BigRational> {
    let (radicand, derivative_sign) = sqrt_chain_argument_derivative_parts(ctx, sqrt_arg, var)?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let two = BigRational::from_integer(2.into());
    let half_derivative =
        scale_polynomial(&radicand_poly.derivative(), derivative_sign).div_scalar(&two);
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

pub(super) fn sqrt_polynomial_derivative_quotient_scale_expr(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    sqrt_arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(scale) = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        numerator_factors,
        denominator_factors,
        sqrt_arg,
        var,
    ) {
        return Some(ctx.add(Expr::Number(scale)));
    }

    let mut symbolic_scale_factors = Vec::new();
    let mut derivative_factors = Vec::new();
    for factor in numerator_factors {
        if contains_named_var(ctx, *factor, var) {
            derivative_factors.push(*factor);
        } else {
            symbolic_scale_factors.push(*factor);
        }
    }
    if symbolic_scale_factors.is_empty() {
        return None;
    }

    let rational_scale = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        &derivative_factors,
        denominator_factors,
        sqrt_arg,
        var,
    )?;
    if rational_scale.is_zero() {
        return None;
    }

    let symbolic_scale = if symbolic_scale_factors.len() == 1 {
        symbolic_scale_factors[0]
    } else {
        build_balanced_mul(ctx, &symbolic_scale_factors)
    };
    Some(scale_rational_term(ctx, rational_scale, symbolic_scale))
}

fn positive_rational_constant_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let value = rational_constant_value(ctx, expr)?;
    positive_rational_sqrt_expr(ctx, &value)
}

pub(super) fn positive_constant_like_root(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(root) = positive_rational_constant_root(ctx, expr) {
        return Some(root);
    }
    if contains_named_var(ctx, expr, var) {
        return None;
    }
    if !crate::calculus_domain_support::positive_condition_is_proven_over_reals(
        ctx,
        expr,
        SYMBOLIC_INTEGRATION_DOMAIN_PROOF_DEPTH,
    ) {
        return None;
    }
    Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]))
}

/// c * x^n / sqrt(a - b*x^2) for rational a, b > 0 and 2 <= n <= 6: the
/// textbook reduction I_n = a(n-1)/(bn) * I_{n-2} - x^(n-1)
/// sqrt(a - b x^2)/(bn), delegating the n = 0/1 bases to their owners
/// (arcsin table, derivative substitution).
pub(super) fn monomial_over_sqrt_negative_quadratic_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    // Normalization artifact (sixth Div-arm instance): the raw surface
    // is Div(c*x^n, sqrt(a - b x^2)); rebuild as a product with the
    // negative half power and reuse the same matcher.
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let rad = match ctx.get(den).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt)) =>
            {
                args[0]
            }
            Expr::Pow(base, exponent)
                if crate::numeric_eval::as_rational_const(ctx, exponent)
                    .is_some_and(|value| value == BigRational::new(1.into(), 2.into())) =>
            {
                base
            }
            _ => return None,
        };
        let neg_half = ctx.add(Expr::Number(BigRational::new((-1).into(), 2.into())));
        let reciprocal = ctx.add(Expr::Pow(rad, neg_half));
        let product = mul2_raw(ctx, num, reciprocal);
        return monomial_over_sqrt_negative_quadratic_antiderivative(ctx, product, var);
    }

    let mut scale = BigRational::one();
    let mut power: Option<BigRational> = None;
    let mut radicand: Option<ExprId> = None;
    for factor in mul_leaves(ctx, expr) {
        if let Some(value) = rational_constant_value(ctx, factor) {
            scale *= value;
            continue;
        }
        if let Some(factor_power) = var_power(ctx, factor, var) {
            if power.is_some() {
                return None;
            }
            power = Some(factor_power);
            continue;
        }
        let Expr::Pow(base, exponent) = ctx.get(factor).clone() else {
            return None;
        };
        let exponent_value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
        if exponent_value != BigRational::new((-1).into(), 2.into()) || radicand.is_some() {
            return None;
        }
        radicand = Some(base);
    }
    let radicand = radicand?;
    let power = power?;
    if !power.denom().is_one() {
        return None;
    }
    let n = power.to_integer();
    if n < 2.into() || n > 6.into() {
        return None;
    }
    let n = usize::try_from(i64::try_from(&n).ok()?).ok()?;

    let quad = Polynomial::from_expr(ctx, radicand, var).ok()?;
    if quad.degree() != 2 {
        return None;
    }
    let a = quad
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let linear = quad
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let neg_b = quad
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !linear.is_zero() || neg_b.is_zero() || a.is_zero() {
        return None;
    }
    let integral = if neg_b.is_negative() {
        // Circle family a - b x^2 (a > 0): the arcsin-flavored recurrence.
        let b = -neg_b;
        if !a.is_positive() {
            return None;
        }
        monomial_over_sqrt_reduction(ctx, n, &a, &b, radicand, var)?
    } else {
        // Hyperbolic family b x^2 + a (any nonzero a): the asinh/acosh
        // flavored recurrence I_n = x^(n-1) sqrt(.)/(bn) - a(n-1)/(bn)
        // I_{n-2}, delegating n = 0/1 bases to their owners.
        monomial_over_sqrt_hyperbolic_reduction(ctx, n, &a, &neg_b, radicand, var)?
    };
    Some(scale_rational_term(ctx, scale, integral))
}

/// (alpha x + beta) / sqrt(q) for a full quadratic q with NONZERO
/// linear coefficient (pure-quadratic radicands keep their owners):
/// split alpha x + beta = (alpha/(2 a2)) q' + c0, giving
/// (alpha/a2) sqrt(q) + c0 * integral of 1/sqrt(q) (delegated).
pub(super) fn linear_over_sqrt_shifted_quadratic_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (numerator, radicand) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let rad = match ctx.get(den).clone() {
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt)) =>
                {
                    args[0]
                }
                Expr::Pow(base, exponent)
                    if crate::numeric_eval::as_rational_const(ctx, exponent)
                        .is_some_and(|value| value == BigRational::new(1.into(), 2.into())) =>
                {
                    base
                }
                _ => return None,
            };
            (num, rad)
        }
        Expr::Mul(_, _) | Expr::Neg(_) => {
            let mut numerator_factors = Vec::new();
            let mut radicand = None;
            for factor in mul_leaves(ctx, expr) {
                if let Expr::Pow(base, exponent) = ctx.get(factor).clone() {
                    if crate::numeric_eval::as_rational_const(ctx, exponent)
                        .is_some_and(|value| value == BigRational::new((-1).into(), 2.into()))
                        && radicand.is_none()
                    {
                        radicand = Some(base);
                        continue;
                    }
                }
                numerator_factors.push(factor);
            }
            let radicand = radicand?;
            if numerator_factors.is_empty() {
                return None;
            }
            let numerator = build_balanced_mul(ctx, &numerator_factors);
            (numerator, radicand)
        }
        _ => return None,
    };

    let quad = Polynomial::from_expr(ctx, radicand, var).ok()?;
    if quad.degree() != 2 {
        return None;
    }
    let a1 = quad
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let a2 = quad
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a1.is_zero() || a2.is_zero() {
        return None;
    }
    let numerator_poly = Polynomial::from_expr(ctx, numerator, var).ok()?;
    if numerator_poly.degree() != 1 {
        return None;
    }
    let beta = numerator_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let alpha = numerator_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if alpha.is_zero() {
        return None;
    }

    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let head = scale_rational_term(ctx, &alpha / &a2, sqrt_term);
    let c0 = beta - &alpha * &a1 / (BigRational::from_integer(2.into()) * &a2);
    if c0.is_zero() {
        return Some(head);
    }
    let one = ctx.num(1);
    let sqrt_again = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let base_integrand = ctx.add(Expr::Div(one, sqrt_again));
    let base = integrate_symbolic_expr(ctx, base_integrand, var)?;
    let tail = scale_rational_term(ctx, c0, base);
    Some(ctx.add(Expr::Add(head, tail)))
}

/// p(x)/sqrt(q) for ANY quadratic q (a2 != 0) and 2 <= deg p <= 6 via
/// the Hermite-style split p = r' q + r q'/2 + c with deg r = deg p - 1:
/// the triangular system solves top-down (diagonal k*a2), giving
/// r(x) sqrt(q) + c * integral of 1/sqrt(q) (delegated). Ordered AFTER
/// the pure-radicand reduction families so their displays stay owned.
pub(super) fn polynomial_over_sqrt_quadratic_hermite_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (numerator, radicand) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let rad = match ctx.get(den).clone() {
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt)) =>
                {
                    args[0]
                }
                Expr::Pow(base, exponent)
                    if crate::numeric_eval::as_rational_const(ctx, exponent)
                        .is_some_and(|value| value == BigRational::new(1.into(), 2.into())) =>
                {
                    base
                }
                _ => return None,
            };
            (num, rad)
        }
        Expr::Mul(_, _) | Expr::Neg(_) => {
            let mut numerator_factors = Vec::new();
            let mut radicand = None;
            for factor in mul_leaves(ctx, expr) {
                if let Expr::Pow(base, exponent) = ctx.get(factor).clone() {
                    if crate::numeric_eval::as_rational_const(ctx, exponent)
                        .is_some_and(|value| value == BigRational::new((-1).into(), 2.into()))
                        && radicand.is_none()
                    {
                        radicand = Some(base);
                        continue;
                    }
                }
                numerator_factors.push(factor);
            }
            let radicand = radicand?;
            if numerator_factors.is_empty() {
                return None;
            }
            let numerator = build_balanced_mul(ctx, &numerator_factors);
            (numerator, radicand)
        }
        _ => return None,
    };

    let quad = Polynomial::from_expr(ctx, radicand, var).ok()?;
    if quad.degree() != 2 {
        return None;
    }
    let a0 = quad
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let a1 = quad
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let a2 = quad
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a2.is_zero() {
        return None;
    }
    let p = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let m = p.degree();
    if !(2..=6).contains(&m) {
        return None;
    }
    let coeff = |poly: &Polynomial, k: usize| {
        poly.coeffs
            .get(k)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };

    // Back-substitution: r_{k-1} = [p_k - (2k+1) a1/2 r_k - (k+1) a0
    // r_{k+1}] / (k a2) for k = m..1; then c from the degree-0 row.
    let mut r = vec![BigRational::zero(); m];
    for k in (1..=m).rev() {
        let r_k = if k < m {
            r[k].clone()
        } else {
            BigRational::zero()
        };
        let r_k1 = if k + 1 < m {
            r[k + 1].clone()
        } else {
            BigRational::zero()
        };
        let half = BigRational::new(1.into(), 2.into());
        let numerator_value = coeff(&p, k)
            - BigRational::from_integer(((2 * k + 1) as i64).into()) * &a1 * &half * &r_k
            - BigRational::from_integer(((k + 1) as i64).into()) * &a0 * &r_k1;
        r[k - 1] = numerator_value / (BigRational::from_integer((k as i64).into()) * &a2);
    }
    let r1 = if m > 1 {
        r[1].clone()
    } else {
        BigRational::zero()
    };
    let c = coeff(&p, 0) - &r1 * &a0 - &r[0] * &a1 * BigRational::new(1.into(), 2.into());

    let var_expr = ctx.var(var);
    let mut r_terms = Vec::new();
    for (degree, value) in r.iter().enumerate() {
        if value.is_zero() {
            continue;
        }
        let term = match degree {
            0 => ctx.num(1),
            1 => var_expr,
            _ => {
                let exponent = ctx.num(degree as i64);
                ctx.add(Expr::Pow(var_expr, exponent))
            }
        };
        r_terms.push(scale_rational_term(ctx, value.clone(), term));
    }
    if r_terms.is_empty() {
        return None;
    }
    let r_expr = build_balanced_add(ctx, &r_terms);
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let head = mul2_raw(ctx, r_expr, sqrt_term);
    if c.is_zero() {
        return Some(head);
    }
    let one = ctx.num(1);
    let sqrt_again = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let base_integrand = ctx.add(Expr::Div(one, sqrt_again));
    let base = integrate_symbolic_expr(ctx, base_integrand, var)?;
    let tail = scale_rational_term(ctx, c, base);
    Some(ctx.add(Expr::Add(head, tail)))
}

/// Quotients k * x^p * sqrt(q)^J / (x^m q^n) with q = a x^2 + c a pure
/// quadratic (rational a != 0, c != 0) and net radical power J odd:
/// the x-in-denominator side of the trig-substitution chapter, on both
/// the raw surface (1/(x sqrt(q))) and the rationalized one
/// (sqrt(q)/(x q), expanded denominators like x^4 + 4 x^2). The form
/// is normalized to u/(x^M q^N) via u^J = u q^((J-1)/2). Odd M
/// substitutes u = sqrt(q) (x dx = u du / a), giving the rational
/// integrand a^((M+1)/2-1) u^(2-2N)/(u^2-c)^((M+1)/2) delegated to the
/// rational owners (kernel 1/(x sqrt(q)) -> arctan/atanh of sqrt(q):
/// the arcsec chapter). Even M covers the two textbook shapes:
/// 1/(x^2 sqrt(q)) = -sqrt(q)/(c x) and sqrt(q)/x^2 = -sqrt(q)/x +
/// a * integral(1/sqrt(q)) via the owned inverse-sqrt tables.
pub(super) fn quadratic_radical_over_monomial_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    // Split the Div/Mul tree into numerator and denominator factors.
    let mut num_factors: Vec<ExprId> = Vec::new();
    let mut den_factors: Vec<ExprId> = Vec::new();
    fn split_factors(
        ctx: &Context,
        expr: ExprId,
        inverted: bool,
        num: &mut Vec<ExprId>,
        den: &mut Vec<ExprId>,
    ) {
        match ctx.get(expr) {
            Expr::Mul(l, r) => {
                let (l, r) = (*l, *r);
                split_factors(ctx, l, inverted, num, den);
                split_factors(ctx, r, inverted, num, den);
            }
            Expr::Div(l, r) => {
                let (l, r) = (*l, *r);
                split_factors(ctx, l, inverted, num, den);
                split_factors(ctx, r, !inverted, num, den);
            }
            _ => {
                if inverted {
                    den.push(expr);
                } else {
                    num.push(expr);
                }
            }
        }
    }
    split_factors(ctx, expr, false, &mut num_factors, &mut den_factors);
    if den_factors.is_empty() {
        return None;
    }

    // Classify factors: half-integer powers of one shared radicand vs
    // polynomial cofactors.
    let mut radicand: Option<(ExprId, Polynomial)> = None;
    let mut net_j: i64 = 0;
    let mut num_polys: Vec<Polynomial> = Vec::new();
    let mut den_polys: Vec<Polynomial> = Vec::new();
    for (factors, sign) in [(num_factors.clone(), 1i64), (den_factors.clone(), -1i64)] {
        for factor in factors {
            let atom = match ctx.get(factor).clone() {
                Expr::Pow(base, exponent) => crate::numeric_eval::as_rational_const(ctx, exponent)
                    .filter(|value| *value.denom() == 2.into())
                    .and_then(|value| i64::try_from(value.numer()).ok())
                    .map(|j| (base, j)),
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt)) =>
                {
                    Some((args[0], 1))
                }
                _ => None,
            };
            if let Some((base, j)) = atom {
                let base_poly = Polynomial::from_expr(ctx, base, var).ok()?;
                if base_poly.degree() != 2 || !base_poly.coeffs[1].is_zero() {
                    return None;
                }
                match &radicand {
                    None => radicand = Some((base, base_poly)),
                    Some((_, existing)) => {
                        if existing.coeffs != base_poly.coeffs {
                            return None;
                        }
                    }
                }
                net_j += sign * j;
                continue;
            }
            let poly = Polynomial::from_expr(ctx, factor, var).ok()?;
            if poly.is_zero() {
                return None;
            }
            if sign > 0 {
                num_polys.push(poly);
            } else {
                den_polys.push(poly);
            }
        }
    }
    let (radicand_expr, radicand_poly) = radicand?;
    if net_j <= -7 || net_j >= 7 || net_j % 2 == 0 {
        return None;
    }
    let a = radicand_poly.coeffs[2].clone();
    let c = radicand_poly.coeffs[0].clone();
    if a.is_zero() || c.is_zero() {
        return None;
    }

    // Numerator polynomial part must be a monomial k x^p.
    let mut num_poly = Polynomial::one(var.to_string());
    for poly in &num_polys {
        num_poly = num_poly.mul(poly);
    }
    let p = num_poly
        .coeffs
        .iter()
        .take_while(|coeff| coeff.is_zero())
        .count();
    if p >= num_poly.coeffs.len()
        || num_poly.coeffs[p + 1..]
            .iter()
            .any(|coeff| !coeff.is_zero())
    {
        return None;
    }
    let k_num = num_poly.coeffs[p].clone();

    // Denominator polynomial part: scale * x^m * q^n.
    let mut den_poly = Polynomial::one(var.to_string());
    for poly in &den_polys {
        den_poly = den_poly.mul(poly);
    }
    let m_raw = den_poly
        .coeffs
        .iter()
        .take_while(|coeff| coeff.is_zero())
        .count();
    if m_raw >= den_poly.coeffs.len() {
        return None;
    }
    let even_part: Vec<BigRational> = den_poly.coeffs[m_raw..].to_vec();
    if even_part
        .iter()
        .skip(1)
        .step_by(2)
        .any(|coeff| !coeff.is_zero())
    {
        return None;
    }
    let y_coeffs: Vec<BigRational> = even_part.iter().cloned().step_by(2).collect();
    let n = y_coeffs.len() - 1;
    if n > 3 {
        return None;
    }
    let mut q_power = vec![BigRational::from_integer(1.into())];
    for _ in 0..n {
        let mut next = vec![BigRational::zero(); q_power.len() + 1];
        for (idx, coeff) in q_power.iter().enumerate() {
            next[idx] += coeff * &c;
            next[idx + 1] += coeff * &a;
        }
        q_power = next;
    }
    let lead = q_power.last()?.clone();
    if lead.is_zero() {
        return None;
    }
    let scale = y_coeffs.last()? / &lead;
    if scale.is_zero() || (0..=n).any(|idx| y_coeffs[idx] != &q_power[idx] * &scale) {
        return None;
    }

    // Canonical form k_total * u / (x^M q^N): u^J = u * q^((J-1)/2).
    let m_eff = i64::try_from(m_raw).ok()? - i64::try_from(p).ok()?;
    if !(1..=7).contains(&m_eff) {
        return None;
    }
    let n_eff = i64::try_from(n).ok()? + (1 - net_j) / 2;
    if !(0..=4).contains(&n_eff) {
        return None;
    }
    let k_total = k_num / (&scale);
    if k_total.is_zero() {
        return None;
    }
    let half = BigRational::new(1.into(), 2.into());

    if m_eff % 2 == 1 {
        let used = cas_ast::collect_variables(ctx, expr);
        let u_name = ["u", "u_", "u_sub"]
            .iter()
            .find(|candidate| !used.contains(**candidate) && *candidate != &var)?
            .to_string();
        let half_m = u32::try_from((m_eff + 1) / 2).ok()?;
        let mut factor = k_total;
        for _ in 0..(half_m - 1) {
            factor *= &a;
        }
        let u_exp = 2 - 2 * n_eff;
        let mut numer_u = Polynomial::zero(u_name.clone());
        let numer_degree = usize::try_from(u_exp.max(0)).ok()?;
        numer_u.coeffs = vec![BigRational::zero(); numer_degree + 1];
        numer_u.coeffs[numer_degree] = factor;
        let mut den_u = Polynomial::one(u_name.clone());
        let mut u2_minus_c = Polynomial::zero(u_name.clone());
        u2_minus_c.coeffs = vec![
            -c.clone(),
            BigRational::zero(),
            BigRational::from_integer(1.into()),
        ];
        for _ in 0..half_m {
            den_u = den_u.mul(&u2_minus_c);
        }
        if u_exp < 0 {
            let extra = usize::try_from(-u_exp).ok()?;
            let mut mono = Polynomial::zero(u_name.clone());
            mono.coeffs = vec![BigRational::zero(); extra + 1];
            mono.coeffs[extra] = BigRational::from_integer(1.into());
            den_u = den_u.mul(&mono);
        }
        if numer_u.degree() > 10 || den_u.degree() > 10 {
            return None;
        }
        let numerator_expr = polynomial_to_expr(ctx, &numer_u, &u_name);
        let denominator_expr = polynomial_to_expr(ctx, &den_u, &u_name);
        let integrand_u = ctx.add(Expr::Div(numerator_expr, denominator_expr));
        let integral_u = integrate_symbolic_expr(ctx, integrand_u, &u_name).or_else(|| {
            let config = crate::general_integration_backend::AlgorithmicIntegrationBackendConfig::residual_fallback();
            let candidate = crate::general_integration_backend::try_algorithmic_integration_backend(
                ctx, integrand_u, &u_name, config,
            );
            if !candidate.required_conditions.is_empty() {
                return None;
            }
            candidate.fallback_antiderivative(config)
        })?;
        let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);
        let half_expr = ctx.add(Expr::Number(half));
        let replacement = ctx.add(Expr::Pow(radicand_expr, half_expr));
        let target = ctx.var(&u_name);
        let substituted = crate::substitute::substitute_power_aware(
            ctx,
            integral_u,
            target,
            replacement,
            crate::substitute::SubstituteOptions::exact(),
        );
        return Some(strip_redundant_sqrt_abs(ctx, substituted));
    }
    if m_eff == 2 {
        let half_expr = ctx.add(Expr::Number(half));
        let sqrt_q = ctx.add(Expr::Pow(radicand_expr, half_expr));
        let var_expr = ctx.var(var);
        if n_eff == 1 {
            // u/(x^2 q) = 1/(x^2 sqrt(q)): antiderivative -sqrt(q)/(c x).
            let coeff = ctx.add(Expr::Number(-(&k_total / &c)));
            let scaled_sqrt = ctx.add(Expr::Mul(coeff, sqrt_q));
            return Some(ctx.add(Expr::Div(scaled_sqrt, var_expr)));
        }
        if n_eff == 0 {
            // sqrt(q)/x^2 = d/dx[-sqrt(q)/x] + a/sqrt(q): delegate the
            // owned inverse-sqrt tail.
            let one = ctx.num(1);
            let inv_sqrt = ctx.add(Expr::Div(one, sqrt_q));
            let tail = integrate_symbolic_expr(ctx, inv_sqrt, var)?;
            let scaled_tail = scale_rational_term(ctx, &a * &k_total, tail);
            let neg_k = ctx.add(Expr::Number(-k_total));
            let scaled_sqrt = ctx.add(Expr::Mul(neg_k, sqrt_q));
            let head = ctx.add(Expr::Div(scaled_sqrt, var_expr));
            return Some(ctx.add(Expr::Add(head, scaled_tail)));
        }
    }
    None
}

/// True when expr is built purely from rational operations over the
/// variable, sqrt(linear) atoms (half-integer powers of a linear
/// radicand), and var-free subtrees; collects every radicand with its
/// rational (slope, offset). Any other occurrence of var refuses.
pub(super) fn collect_linear_radical_radicands(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    radicands: &mut Vec<(ExprId, BigRational, BigRational)>,
) -> bool {
    if let Some(atom) = linear_radical_atom(ctx, expr, var) {
        radicands.push((atom.radicand, atom.slope, atom.offset));
        return true;
    }
    if !contains_named_var(ctx, expr, var) {
        return true;
    }
    if matches!(ctx.get(expr), Expr::Variable(sym) if ctx.sym_name(*sym) == var) {
        return true;
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_linear_radical_radicands(ctx, l, var, radicands)
                && collect_linear_radical_radicands(ctx, r, var, radicands)
        }
        Expr::Neg(inner) => collect_linear_radical_radicands(ctx, inner, var, radicands),
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
            collect_linear_radical_radicands(ctx, base, var, radicands)
        }
        _ => false,
    }
}

/// (a x + b)^(k/2) with k odd (sqrt(...) counts as k = 1) and rational
/// nonzero slope a.
pub(super) fn linear_radical_atom(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<LinearRadicalAtom> {
    let (radicand, half_power) = match ctx.get(expr).clone() {
        Expr::Pow(base, exponent) => {
            let value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if *value.denom() != 2.into() {
                return None;
            }
            (base, i64::try_from(value.numer()).ok()?)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt)) =>
        {
            (args[0], 1)
        }
        _ => return None,
    };
    if !contains_named_var(ctx, radicand, var) {
        return None;
    }
    let (slope_expr, offset_expr) = get_linear_coeffs(ctx, radicand, var)?;
    let slope = rational_constant_value(ctx, slope_expr)?;
    if slope.is_zero() {
        return None;
    }
    let offset = rational_constant_value(ctx, offset_expr)?;
    Some(LinearRadicalAtom {
        radicand,
        slope,
        offset,
        half_power,
    })
}

/// |t^(k/2)| = t^(k/2) for positive half-integer powers (the radical
/// is the nonnegative square root): drop Abs wrappers the
/// back-substitution leaves around radical atoms (ln(|u|) ->
/// ln(|sqrt(a x + b)|)).
pub(super) fn strip_redundant_sqrt_abs(ctx: &mut Context, expr: ExprId) -> ExprId {
    let node = ctx.get(expr).clone();
    if let Expr::Function(fn_id, args) = &node {
        if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Abs)) {
            let inner = ctx.get(args[0]).clone();
            let is_nonnegative_radical = match inner {
                Expr::Pow(_, exponent) => crate::numeric_eval::as_rational_const(ctx, exponent)
                    .is_some_and(|value| *value.denom() == 2.into() && value > BigRational::zero()),
                Expr::Function(inner_fn, inner_args) => {
                    inner_args.len() == 1
                        && matches!(ctx.builtin_of(inner_fn), Some(BuiltinFn::Sqrt))
                }
                _ => false,
            };
            if is_nonnegative_radical {
                return strip_redundant_sqrt_abs(ctx, args[0]);
            }
        }
    }
    match node {
        Expr::Add(l, r) => {
            let l = strip_redundant_sqrt_abs(ctx, l);
            let r = strip_redundant_sqrt_abs(ctx, r);
            ctx.add(Expr::Add(l, r))
        }
        Expr::Sub(l, r) => {
            let l = strip_redundant_sqrt_abs(ctx, l);
            let r = strip_redundant_sqrt_abs(ctx, r);
            ctx.add(Expr::Sub(l, r))
        }
        Expr::Mul(l, r) => {
            let l = strip_redundant_sqrt_abs(ctx, l);
            let r = strip_redundant_sqrt_abs(ctx, r);
            ctx.add(Expr::Mul(l, r))
        }
        Expr::Div(l, r) => {
            let l = strip_redundant_sqrt_abs(ctx, l);
            let r = strip_redundant_sqrt_abs(ctx, r);
            ctx.add(Expr::Div(l, r))
        }
        Expr::Neg(inner) => {
            let inner = strip_redundant_sqrt_abs(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Pow(base, exponent) => {
            let base = strip_redundant_sqrt_abs(ctx, base);
            let exponent = strip_redundant_sqrt_abs(ctx, exponent);
            ctx.add(Expr::Pow(base, exponent))
        }
        Expr::Function(fn_id, args) => {
            let args: Vec<_> = args
                .iter()
                .map(|arg| strip_redundant_sqrt_abs(ctx, *arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    }
}

pub(super) fn radical_numerator_polynomial_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut cofactor_factors: Vec<ExprId> = Vec::new();
    let mut radicand: Option<ExprId> = None;
    for factor in mul_leaves(ctx, expr) {
        match ctx.get(factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt))
                    && radicand.is_none()
                    && Polynomial::from_expr(ctx, args[0], var).is_ok_and(|p| p.degree() == 2) =>
            {
                radicand = Some(args[0]);
            }
            Expr::Pow(base, exponent)
                if crate::numeric_eval::as_rational_const(ctx, exponent)
                    .is_some_and(|value| value == BigRational::new(1.into(), 2.into()))
                    && radicand.is_none()
                    && Polynomial::from_expr(ctx, base, var).is_ok_and(|p| p.degree() == 2) =>
            {
                radicand = Some(base);
            }
            _ => cofactor_factors.push(factor),
        }
    }
    let radicand = radicand?;
    let quad = Polynomial::from_expr(ctx, radicand, var).ok()?;

    let cofactor_poly = if cofactor_factors.is_empty() {
        Polynomial::one(var.to_string())
    } else {
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        Polynomial::from_expr(ctx, cofactor, var).ok()?
    };
    let numerator_poly = cofactor_poly.mul(&quad);
    if numerator_poly.degree() > 6 || numerator_poly.is_zero() {
        return None;
    }

    let var_expr = ctx.var(var);
    let mut terms = Vec::new();
    for (degree, coeff) in numerator_poly.coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let term = match degree {
            0 => ctx.num(1),
            1 => var_expr,
            _ => {
                let exponent = ctx.num(degree as i64);
                ctx.add(Expr::Pow(var_expr, exponent))
            }
        };
        terms.push(scale_rational_term(ctx, coeff.clone(), term));
    }
    let numerator_expr = build_balanced_add(ctx, &terms);
    let neg_half = ctx.add(Expr::Number(BigRational::new((-1).into(), 2.into())));
    let reciprocal = ctx.add(Expr::Pow(radicand, neg_half));
    let rebuilt = mul2_raw(ctx, numerator_expr, reciprocal);
    integrate_symbolic_expr(ctx, rebuilt, var)
}

pub(super) fn monomial_over_sqrt_reduction(
    ctx: &mut Context,
    n: usize,
    a: &BigRational,
    b: &BigRational,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    if n <= 1 {
        // Delegate the base cases to their existing owners.
        let var_expr = ctx.var(var);
        let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        let one = ctx.num(1);
        let base_integrand = if n == 0 {
            ctx.add(Expr::Div(one, sqrt_term))
        } else {
            ctx.add(Expr::Div(var_expr, sqrt_term))
        };
        return integrate_symbolic_expr(ctx, base_integrand, var);
    }
    let n_rational = BigRational::from_integer((n as i64).into());
    let n_minus_one = BigRational::from_integer(((n - 1) as i64).into());
    let lower = monomial_over_sqrt_reduction(ctx, n - 2, a, b, radicand, var)?;
    let lower_scale = a * &n_minus_one / (b * &n_rational);
    let lower_term = scale_rational_term(ctx, lower_scale, lower);

    let var_expr = ctx.var(var);
    let head_power = ctx.num((n - 1) as i64);
    let head_monomial = ctx.add(Expr::Pow(var_expr, head_power));
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let head_raw = mul2_raw(ctx, head_monomial, sqrt_term);
    let head_scale = -BigRational::one() / (b * &n_rational);
    let head_term = scale_rational_term(ctx, head_scale, head_raw);
    Some(ctx.add(Expr::Add(lower_term, head_term)))
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

pub(super) fn reduce_surd_offset_by_common_square_factor(
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

pub(super) fn reduce_surd_offset_by_square_denominator(
    arg_poly: &Polynomial,
    offset_square: &BigRational,
) -> Option<(Polynomial, BigRational, BigRational)> {
    if offset_square.is_integer() || offset_square <= &BigRational::zero() {
        return None;
    }

    let denominator_root = offset_square.denom().sqrt();
    if &denominator_root * &denominator_root != offset_square.denom().clone() {
        return None;
    }

    let denominator_scale = BigRational::from_integer(denominator_root);
    if denominator_scale.is_one() {
        return None;
    }

    let reduced_offset_square = offset_square * &denominator_scale * &denominator_scale;
    if !reduced_offset_square.is_integer() {
        return None;
    }

    Some((
        scale_polynomial(arg_poly, denominator_scale.clone()),
        reduced_offset_square,
        denominator_scale,
    ))
}

pub(super) fn normalize_surd_ratio_arg(
    arg_poly: Polynomial,
    offset_square: BigRational,
) -> (Polynomial, BigRational) {
    let mut arg_poly = arg_poly;
    let mut offset_square = offset_square;

    if let Some((normalized_arg, normalized_offset_square, _)) =
        reduce_surd_offset_by_square_denominator(&arg_poly, &offset_square)
    {
        arg_poly = normalized_arg;
        offset_square = normalized_offset_square;
    }
    if let Some((reduced_arg, reduced_offset_square, _)) =
        reduce_surd_offset_by_common_square_factor(&arg_poly, &offset_square)
    {
        arg_poly = reduced_arg;
        offset_square = reduced_offset_square;
    }

    (arg_poly, offset_square)
}

pub(super) fn exact_polynomial_square_root(poly: &Polynomial) -> Option<Polynomial> {
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
    (root.mul(&root) == *poly).then_some(root)
}

pub(super) fn symbolic_radius_inverse_sqrt_primitive(
    ctx: &mut Context,
    numerator: ExprId,
    radius_square: ExprId,
    arg: ExprId,
    var: &str,
    inverse_fn: BuiltinFn,
) -> Option<ExprId> {
    let numerator_scale = rational_constant_value(ctx, numerator)?;
    let (slope, _) = get_linear_coeffs(ctx, arg, var)?;

    let radius = ctx.call_builtin(BuiltinFn::Sqrt, vec![radius_square]);
    let inverse_arg = ctx.add(Expr::Div(arg, radius));
    let primitive = ctx.call_builtin(inverse_fn, vec![inverse_arg]);
    scale_by_rational_over_variable_free_slope(ctx, numerator_scale, slope, primitive, var)
}

pub(super) fn explicit_sqrt_linear_domain_sample(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Div(_, den) => return explicit_sqrt_linear_domain_sample(ctx, *den, var),
        Expr::Neg(inner) => return explicit_sqrt_linear_domain_sample(ctx, *inner, var),
        _ => {}
    }

    let mut lower: Option<BigRational> = None;
    let mut upper: Option<BigRational> = None;
    let mut saw_linear_sqrt = false;

    for factor in mul_leaves(ctx, expr) {
        let Some(radicand) =
            sqrt_like_radicand(ctx, factor).or_else(|| reciprocal_sqrt_like_radicand(ctx, factor))
        else {
            continue;
        };
        let poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let constant = poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let slope = poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if slope.is_zero() {
            return None;
        }
        let bound = -constant / slope.clone();
        if slope.is_positive() {
            lower = Some(match lower {
                Some(current) if current > bound => current,
                _ => bound,
            });
        } else {
            upper = Some(match upper {
                Some(current) if current < bound => current,
                _ => bound,
            });
        }
        saw_linear_sqrt = true;
    }

    if !saw_linear_sqrt {
        return None;
    }
    if let (Some(low), Some(high)) = (&lower, &upper) {
        if low >= high {
            return None;
        }
        return Some((low.clone() + high.clone()) / BigRational::from_integer(2.into()));
    }

    let one = BigRational::one();
    if let Some(low) = lower {
        return Some(low + one);
    }
    if let Some(high) = upper {
        return Some(high - one);
    }

    Some(BigRational::zero())
}

pub(super) fn affine_sqrt_product_derivative_solution(
    radicand: &Polynomial,
    numerator: &Polynomial,
) -> Option<Polynomial> {
    if radicand.degree() != 1 || radicand.coeffs.len() < 2 {
        return None;
    }

    let offset = radicand
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = radicand.coeffs[1].clone();
    if slope.is_zero() {
        return None;
    }

    let degree = numerator.degree();
    let two = BigRational::from_integer(2.into());
    let mut coeffs = vec![BigRational::zero(); degree + 1];
    for k in (0..=degree).rev() {
        let numerator_coeff = numerator
            .coeffs
            .get(k)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let next_coeff = coeffs.get(k + 1).cloned().unwrap_or_else(BigRational::zero);
        let next_power = BigRational::from_integer((k + 1).into());
        let current_weight = BigRational::from_integer((2 * k + 1).into());
        let carried = two.clone() * offset.clone() * next_power * next_coeff;
        let denominator = slope.clone() * current_weight;
        if denominator.is_zero() {
            return None;
        }
        coeffs[k] = (numerator_coeff - carried) / denominator;
    }

    let candidate = Polynomial::new(coeffs, numerator.var.clone());
    if candidate.is_zero() {
        return None;
    }

    let reconstructed = scale_polynomial(&candidate.derivative().mul(radicand), two)
        .add(&candidate.mul(&radicand.derivative()));
    (reconstructed == *numerator).then_some(candidate)
}

pub(super) fn affine_sqrt_product_derivative_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (numerator, radicand) = affine_sqrt_product_derivative_div_parts(ctx, num, den, var)?;
    affine_sqrt_product_derivative_from_parts(ctx, numerator, radicand, var)
}

pub(super) fn affine_sqrt_product_derivative_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (numerator, radicand) = affine_sqrt_product_derivative_product_parts(ctx, expr, var)?;
    affine_sqrt_product_derivative_from_parts(ctx, numerator, radicand, var)
}

fn affine_sqrt_product_derivative_radicand(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (numerator, radicand) = match ctx.get(expr) {
        Expr::Div(num, den) => affine_sqrt_product_derivative_div_parts(ctx, *num, *den, var)?,
        _ => affine_sqrt_product_derivative_product_parts(ctx, expr, var)?,
    };
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    affine_sqrt_product_derivative_solution(&radicand_poly, &numerator)?;
    Some(radicand)
}

pub fn integrate_symbolic_is_affine_sqrt_product_derivative_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    affine_sqrt_product_derivative_radicand(ctx, expr, var).is_some()
}

pub(super) fn sqrt_product_denominator_radicand(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let mut radicands = Vec::new();
    let mut denominator_scale_factors = Vec::new();
    for factor in mul_leaves(ctx, den) {
        if let Some(scale) = rational_constant_value(ctx, factor) {
            if scale.is_zero() {
                return None;
            }
            denominator_scale_factors.push(ctx.add(Expr::Number(scale)));
            continue;
        }
        let radicand = sqrt_like_radicand(ctx, factor)?;
        let (radicand, scale) =
            split_positive_rational_content_from_sqrt_radicand(ctx, radicand, var)?;
        radicands.push(radicand);
        if let Some(scale) = scale {
            denominator_scale_factors.push(scale);
        }
    }

    if radicands.len() < 2 {
        return None;
    }

    let denominator_scale = if denominator_scale_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &denominator_scale_factors)
    };

    Some((build_balanced_mul(ctx, &radicands), denominator_scale))
}

pub(super) fn split_positive_rational_content_from_sqrt_radicand(
    ctx: &mut Context,
    radicand: ExprId,
    var: &str,
) -> Option<(ExprId, Option<ExprId>)> {
    let poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let content = poly.content();
    if content.is_zero() || !content.is_positive() {
        return None;
    }
    if content.is_one() {
        return Some((radicand, None));
    }

    let normalized = poly.div_scalar(&content).to_expr(ctx);
    let scale = positive_rational_sqrt_expr(ctx, &content)?;
    Some((normalized, Some(scale)))
}

pub(super) fn divide_by_sqrt_product_denominator_scale(
    ctx: &mut Context,
    expr: ExprId,
    denominator_scale: ExprId,
) -> ExprId {
    if matches!(ctx.get(denominator_scale), Expr::Number(n) if n.is_one()) {
        return expr;
    }
    ctx.add(Expr::Div(expr, denominator_scale))
}

fn is_sqrt_of_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    sqrt_like_radicand(ctx, expr).is_some_and(|radicand| is_var(ctx, radicand, var))
}

fn is_sqrt_var_minus_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Sub(left, right) => is_sqrt_of_var(ctx, *left, var) && is_var(ctx, *right, var),
        Expr::Add(left, right) => {
            (is_sqrt_of_var(ctx, *left, var) && is_neg_var(ctx, *right, var))
                || (is_neg_var(ctx, *left, var) && is_sqrt_of_var(ctx, *right, var))
        }
        _ => false,
    }
}

pub(super) fn is_var_times_sqrt_var_minus_var_radicand(
    ctx: &Context,
    radicand: ExprId,
    var: &str,
) -> bool {
    let factors = mul_leaves(ctx, radicand);
    match factors.as_slice() {
        [left, right] => {
            (is_var(ctx, *left, var) && is_sqrt_var_minus_var(ctx, *right, var))
                || (is_sqrt_var_minus_var(ctx, *left, var) && is_var(ctx, *right, var))
        }
        _ => false,
    }
}

pub(super) fn collect_fraction_factors_for_inverse_sqrt_product(
    ctx: &Context,
    expr: ExprId,
    in_denominator: bool,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_fraction_factors_for_inverse_sqrt_product(
                ctx,
                *left,
                in_denominator,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_factors_for_inverse_sqrt_product(
                ctx,
                *right,
                in_denominator,
                numerator_factors,
                denominator_factors,
            );
        }
        Expr::Div(left, right) => {
            collect_fraction_factors_for_inverse_sqrt_product(
                ctx,
                *left,
                in_denominator,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_factors_for_inverse_sqrt_product(
                ctx,
                *right,
                !in_denominator,
                numerator_factors,
                denominator_factors,
            );
        }
        _ if in_denominator => denominator_factors.push(expr),
        _ => numerator_factors.push(expr),
    }
}

pub(super) fn affine_sqrt_product_derivative_radicand_from_mut_context(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    affine_sqrt_product_derivative_radicand(ctx, expr, var)
}

/// Rewrite `cbrt(u)` as `u^(1/3)` and `cbrt(u)^n` as `u^(n/3)` in the integrable
/// positions (the algebraic skeleton), returning the rewritten expr when any
/// cube root was lowered. `cbrt` is kept as `Function(Cbrt)` everywhere else
/// (display + the cube-root limit rules); this lowering is local to integration,
/// where `cbrt(u) = u^(1/3)` lets the ordinary power rule do the work. It does
/// NOT descend into transcendental function arguments (e.g. `sin(cbrt(x))`),
/// which stay residual either way.
pub(super) fn lower_cbrt_for_integration(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (rewritten, changed) = rewrite_cbrt_to_pow_for_integration(ctx, expr);
    changed.then_some(rewritten)
}

fn rewrite_cbrt_to_pow_for_integration(ctx: &mut Context, expr: ExprId) -> (ExprId, bool) {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Cbrt) =>
        {
            let (inner, _) = rewrite_cbrt_to_pow_for_integration(ctx, args[0]);
            let third = ctx.add(Expr::Number(BigRational::new(1.into(), 3.into())));
            (ctx.add(Expr::Pow(inner, third)), true)
        }
        Expr::Pow(base, exp) => {
            // cbrt(a)^n -> a^(n/3): the nested power does not auto-flatten.
            if let Expr::Function(fn_id, args) = ctx.get(base).clone() {
                if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Cbrt) {
                    if let Expr::Number(n) = ctx.get(exp).clone() {
                        let (inner, _) = rewrite_cbrt_to_pow_for_integration(ctx, args[0]);
                        let new_exp =
                            ctx.add(Expr::Number(n / BigRational::new(3.into(), 1.into())));
                        return (ctx.add(Expr::Pow(inner, new_exp)), true);
                    }
                }
            }
            let (b, cb) = rewrite_cbrt_to_pow_for_integration(ctx, base);
            let (e, ce) = rewrite_cbrt_to_pow_for_integration(ctx, exp);
            if cb || ce {
                (ctx.add(Expr::Pow(b, e)), true)
            } else {
                (expr, false)
            }
        }
        Expr::Add(l, r) => {
            let (l2, cl) = rewrite_cbrt_to_pow_for_integration(ctx, l);
            let (r2, cr) = rewrite_cbrt_to_pow_for_integration(ctx, r);
            if cl || cr {
                (ctx.add(Expr::Add(l2, r2)), true)
            } else {
                (expr, false)
            }
        }
        Expr::Sub(l, r) => {
            let (l2, cl) = rewrite_cbrt_to_pow_for_integration(ctx, l);
            let (r2, cr) = rewrite_cbrt_to_pow_for_integration(ctx, r);
            if cl || cr {
                (ctx.add(Expr::Sub(l2, r2)), true)
            } else {
                (expr, false)
            }
        }
        Expr::Mul(l, r) => {
            let (l2, cl) = rewrite_cbrt_to_pow_for_integration(ctx, l);
            let (r2, cr) = rewrite_cbrt_to_pow_for_integration(ctx, r);
            if cl || cr {
                (ctx.add(Expr::Mul(l2, r2)), true)
            } else {
                (expr, false)
            }
        }
        Expr::Div(n, d) => {
            let (n2, cn) = rewrite_cbrt_to_pow_for_integration(ctx, n);
            let (d2, cd) = rewrite_cbrt_to_pow_for_integration(ctx, d);
            if cn || cd {
                (ctx.add(Expr::Div(n2, d2)), true)
            } else {
                (expr, false)
            }
        }
        Expr::Neg(inner) => {
            let (inner2, c) = rewrite_cbrt_to_pow_for_integration(ctx, inner);
            if c {
                (ctx.add(Expr::Neg(inner2)), true)
            } else {
                (expr, false)
            }
        }
        _ => (expr, false),
    }
}
