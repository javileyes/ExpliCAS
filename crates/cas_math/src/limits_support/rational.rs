//! `limits_support`: familia `rational`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

pub(super) fn apply_finite_rational_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(ctx.add(Expr::Number(value)))
}

pub(super) fn apply_finite_one_sided_rational_polynomial_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let (den_order, den_derivative) =
        finite_polynomial_local_order_and_derivative(&denominator, &point_value)?;
    if den_order == 0 {
        return None;
    }

    let Some((num_order, num_derivative)) =
        finite_polynomial_local_order_and_derivative(&numerator, &point_value)
    else {
        return Some(ctx.num(0));
    };
    if num_order >= den_order {
        return None;
    }

    let num_sign = finite_local_tail_sign(&num_derivative, num_order, side)?;
    let den_sign = finite_local_tail_sign(&den_derivative, den_order, side)?;
    Some(signed_abs_ratio_infinity(ctx, num_sign, den_sign))
}

pub(super) fn apply_finite_bilateral_rational_polynomial_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left = apply_finite_one_sided_rational_polynomial_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Left,
    )?;
    let right = apply_finite_one_sided_rational_polynomial_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Right,
    )?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

pub(super) fn rational_tail_sign(value: &BigRational) -> InfSign {
    if value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    }
}

pub(super) fn apply_finite_zero_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    scaled_argument: fn(&Context, ExprId) -> Option<(BigRational, ExprId)>,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (scale, zero_argument) = scaled_argument(ctx, num)?;
    let argument = Polynomial::from_expr(ctx, zero_argument, &var_name).ok()?;
    if !argument.eval(&point_value).is_zero() {
        return None;
    }

    let scale_poly = Polynomial::new(vec![scale], var_name.to_string());
    let numerator = argument.mul(&scale_poly);
    let denominator = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(ctx.add(Expr::Number(value)))
}

pub(super) fn rational_pow_nonnegative(base: &BigRational, exponent: u64) -> BigRational {
    let mut result = BigRational::one();
    let mut factor = base.clone();
    let mut remaining = exponent;

    while remaining > 0 {
        if remaining % 2 == 1 {
            result *= factor.clone();
        }
        remaining /= 2;
        if remaining > 0 {
            factor = factor.clone() * factor;
        }
    }

    result
}

/// A rational function of the variable (polynomial, or a ratio of two
/// polynomials such as `1/x`). Real wherever defined, hence bounded near
/// any point for a saturating outer function.
pub(super) fn argument_is_real_rational_function(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
) -> bool {
    if Polynomial::from_expr(ctx, arg, var_name).is_ok() {
        return true;
    }
    if let Expr::Div(num, den) = ctx.get(arg) {
        // The denominator must be a NONZERO polynomial. An identically-zero
        // denominator (1/(x - x) = 1/0) makes the quotient undefined on the
        // WHOLE punctured neighbourhood, so sin/cos/... of it is nowhere
        // defined and has no limit - it must not count as bounded. A
        // denominator that merely vanishes at isolated points (1/x) is fine:
        // the quotient is defined on the punctured neighbourhood.
        return Polynomial::from_expr(ctx, *num, var_name).is_ok()
            && Polynomial::from_expr(ctx, *den, var_name)
                .is_ok_and(|den_poly| !den_poly.is_zero());
    }
    false
}

/// Higher-order 0/0 limits at a finite point via Taylor series:
/// `(1 - cos x)/x^2 -> 1/2`, `(sin x - x)/x^3 -> -1/6`,
/// `(e^x - 1 - x)/x^2 -> 1/2`. Both numerator and denominator are expanded
/// to a bounded order; the limit is the ratio of the lowest-order
/// coefficients when the numerator does not vanish slower than the
/// denominator. Coefficients up to the denominator's order are EXACT
/// (truncation only drops higher orders), so the value is exact.
///
/// Runs after the first-order equivalent engine, which owns the simple
/// `sin x / x` cases; this rule resolves the cancellation cases that need
/// the second/third Taylor term.
pub(super) fn apply_finite_taylor_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;
    // Only the point 0 (the Taylor series are expanded at 0).
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let num_series = taylor_at_zero(ctx, num, &var_name, TAYLOR_QUOTIENT_MAX_ORDER)?;
    let den_series = taylor_at_zero(ctx, den, &var_name, TAYLOR_QUOTIENT_MAX_ORDER)?;
    let num_low = lowest_nonzero_order(&num_series.coeffs);
    let den_low = lowest_nonzero_order(&den_series.coeffs)?;
    // Genuine 0/0 only: the denominator must vanish at 0.
    if den_low == 0 {
        return None;
    }
    let den_coeff = den_series.coeffs[den_low].clone();
    match num_low {
        // Numerator is identically 0 up to the tracked order: 0 / (x^d) -> 0
        // only if it truly vanishes to order > den_low. We only know that when
        // den_low <= tracked order, which it is here; a higher-order numerator
        // term cannot change a 0 leading behaviour against x^den_low.
        None => Some(ctx.num(0)),
        Some(m) if m > den_low => Some(ctx.num(0)),
        Some(m) if m == den_low => {
            let value = &num_series.coeffs[m] / &den_coeff;
            Some(ctx.add(Expr::Number(value)))
        }
        // m < den_low: numerator vanishes slower -> divergent (DNE bilateral).
        Some(_) => None,
    }
}

/// The rational value of a fully evaluated limit result, if it is a plain number.
pub(super) fn limit_result_rational(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        _ => None,
    }
}

/// Power-series reciprocal `1/den` to `order`, via the standard recurrence
/// `r_0 = 1/d_0`, `r_k = -(1/d_0)·Σ_{i=1}^{k} d_i·r_{k-i}`. Requires `den(0) ≠ 0`
/// (returns `None` otherwise — a pole at 0 has no Maclaurin expansion).
fn reciprocal_series(den: &Polynomial, order: usize, var_name: &str) -> Option<Polynomial> {
    use num_traits::{One, Zero};
    let d0 = den.coeffs.first().cloned().filter(|c| !c.is_zero())?;
    let mut r = vec![BigRational::zero(); order + 1];
    r[0] = BigRational::one() / d0.clone();
    for k in 1..=order {
        let mut acc = BigRational::zero();
        for i in 1..=k {
            let di = den.coeffs.get(i).cloned().unwrap_or_else(BigRational::zero);
            acc += di * r[k - i].clone();
        }
        r[k] = -acc / d0.clone();
    }
    Some(Polynomial::new(r, var_name.to_string()))
}

/// Maclaurin expansion extended to RATIONAL summands: the analytic `taylor_at_zero`
/// plus quotients `num/den` and negative integer powers `base^(-m)` whose denominator
/// is non-zero at 0 (so the function is analytic there). Kept SEPARATE from
/// `taylor_at_zero` so the limit evaluator's series path is unaffected — only the public
/// `taylor()`/`series()` command sees the rational extension.
pub(super) fn taylor_at_zero_with_rational(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    order: usize,
) -> Option<Polynomial> {
    if let Some(series) = taylor_at_zero(ctx, expr, var_name, order) {
        return Some(series);
    }
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let den_series = taylor_at_zero_with_rational(ctx, den, var_name, order)?;
            // Valuation of the denominator series: den(0) = 0 means the plain
            // reciprocal has no Maclaurin expansion, but a REMOVABLE singularity
            // (num vanishing at least as fast — `sin(x)/x`) cancels the common
            // power of the variable. Re-expand both to `order + s` first: the
            // truncated tails of the raw series become the LOW-order terms after
            // the shift (`sin(x)` to order 4 loses the x⁵ term that `sin(x)/x`
            // needs for its x⁴ coefficient).
            let den_valuation = den_series.coeffs.iter().position(|c| !c.is_zero())?;
            if den_valuation == 0 {
                let num_series = taylor_at_zero_with_rational(ctx, num, var_name, order)?;
                let recip = reciprocal_series(&den_series, order, var_name)?;
                return Some(truncate_polynomial(
                    &num_series.mul(&recip),
                    order,
                    var_name,
                ));
            }
            let extended = order + den_valuation;
            let num_series = taylor_at_zero_with_rational(ctx, num, var_name, extended)?;
            let den_series = taylor_at_zero_with_rational(ctx, den, var_name, extended)?;
            let num_valuation = num_series.coeffs.iter().position(|c| !c.is_zero())?;
            if num_valuation < den_valuation {
                // Genuine pole: no Maclaurin expansion — decline honestly.
                return None;
            }
            let num_shifted = Polynomial::new(
                num_series.coeffs[den_valuation..].to_vec(),
                var_name.to_string(),
            );
            let den_shifted = Polynomial::new(
                den_series.coeffs[den_valuation..].to_vec(),
                var_name.to_string(),
            );
            let recip = reciprocal_series(&den_shifted, order, var_name)?;
            Some(truncate_polynomial(
                &num_shifted.mul(&recip),
                order,
                var_name,
            ))
        }
        Expr::Pow(base, exponent) => {
            // base^(negative integer) = 1 / base^|n|.
            let exp_value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !exp_value.is_integer() || !exp_value.is_negative() {
                return None;
            }
            let m: u32 = (-exp_value.to_integer()).try_into().ok()?;
            let base_series = taylor_at_zero_with_rational(ctx, base, var_name, order)?;
            let mut den_pow = Polynomial::one(var_name.to_string());
            for _ in 0..m {
                den_pow = truncate_polynomial(&den_pow.mul(&base_series), order, var_name);
            }
            reciprocal_series(&den_pow, order, var_name)
        }
        _ => None,
    }
}

/// Rational-exponent power at infinity: `lim_{x->+∞} x^q` for a non-integer
/// rational `q`.  The bare integer case is owned by [`apply_power_rule`] (which
/// also resolves the `x->-∞` parity); this rule covers fractional exponents,
/// where the antiderivatives of fractional-power integrands surface
/// (`∫x^(-3/2) = -2/√x`, etc.):
///
/// * `q > 0` → `+∞`
/// * `q < 0` → `0`
///
/// For `x->-∞` the base is negative and `x^q` with non-integer `q` is not real,
/// so we decline and leave the limit as an honest residual.
pub(crate) fn apply_rational_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };

    // Base must be exactly the limit variable.
    if base != var {
        return None;
    }

    // Lo único que hace falta decidir es el SIGNO del exponente. `as_rational_const`
    // solo casa literales plegables, así que un exponente constante pero irracional
    // (`pi`, `pi - 3`, `e`) dejaba el límite sin evaluar — no era una respuesta
    // incorrecta, era una NO-respuesta, y la capa que decide ese signo de forma
    // exacta ya existe (`provable_const_sign`, superset con e/π).
    let sign = match crate::numeric_eval::as_rational_const(ctx, exp) {
        Some(q) => {
            // Los exponentes ENTEROS son de `apply_power_rule`, que además cuida la
            // paridad para `x -> -∞`.
            if q.is_integer() {
                return None;
            }
            if q.is_negative() {
                crate::const_sign::ConstSign::Negative
            } else {
                crate::const_sign::ConstSign::Positive
            }
        }
        // Sin valor racional el exponente no puede ser entero, así que no invade a
        // `apply_power_rule`; solo se acepta si su signo es DECIDIBLE.
        None => crate::const_sign::provable_const_sign(ctx, exp)?,
    };

    // A non-integer power of a negative magnitude is not real-valued.
    if approach == InfSign::Neg {
        return None;
    }

    match sign {
        crate::const_sign::ConstSign::Negative => Some(ctx.num(0)),
        crate::const_sign::ConstSign::Positive => Some(mk_infinity(ctx, InfSign::Pos)),
        // `x^0 = 1` tiene dueño en otra regla; aquí declinar es lo honesto.
        crate::const_sign::ConstSign::Zero => None,
    }
}

/// `x^a / x^b` con exponentes CONSTANTES reducido a `x^(a-b)`, delegando en las
/// reglas de potencia existentes (entera con paridad, racional, y constante
/// decidible vía `provable_const_sign`).
///
/// El camino de límites recibía el cociente CRUDO y ninguna regla veía a través:
/// `limit(x^(5/2)/x^(3/2), x, ∞)` quedaba residual aunque la expresión suelta
/// simplifique a `x`, y `limit(x^pi/x^3, x, ∞)` ni con la regla de exponente
/// constante (que ya decide `x^(pi-3)` a solas). La resta se pliega EXACTA si
/// ambos exponentes son racionales; si no, queda como `Sub` estructurado, que es
/// justo lo que la capa de signo constante sabe decidir.
pub(super) fn apply_same_base_power_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };
    let split = |ctx: &Context, side: ExprId| -> (ExprId, Option<ExprId>) {
        match ctx.get(side) {
            Expr::Pow(base, exponent) => (*base, Some(*exponent)),
            _ => (side, None),
        }
    };
    let (base1, exp1) = split(ctx, numerator);
    let (base2, exp2) = split(ctx, denominator);
    if cas_ast::ordering::compare_expr(ctx, base1, var) != std::cmp::Ordering::Equal
        || cas_ast::ordering::compare_expr(ctx, base2, var) != std::cmp::Ordering::Equal
    {
        return None;
    }
    // Al menos un lado debe traer exponente explícito: `x/x` tiene otro dueño.
    if exp1.is_none() && exp2.is_none() {
        return None;
    }
    let one = ctx.num(1);
    let exp1 = exp1.unwrap_or(one);
    let exp2 = exp2.unwrap_or(one);
    // Exponentes constantes (la variable en el exponente es otra familia).
    if depends_on(ctx, exp1, var) || depends_on(ctx, exp2, var) {
        return None;
    }

    let rational_exponents = (
        crate::numeric_eval::as_rational_const(ctx, exp1),
        crate::numeric_eval::as_rational_const(ctx, exp2),
    );
    // En −∞ la reducción solo es sound con exponentes ENTEROS: `x^(5/2)/x^(3/2)`
    // no es real para x<0 (la original no existe donde el límite mira) y
    // reducirla a `x` fabricaría un −∞ de una expresión indefinida. Con enteros,
    // `apply_power_rule` ya cuida la paridad.
    if approach == InfSign::Neg
        && !matches!(
            &rational_exponents,
            (Some(a), Some(b)) if a.is_integer() && b.is_integer()
        )
    {
        return None;
    }
    let difference = match rational_exponents {
        (Some(a), Some(b)) => ctx.add(Expr::Number(a - b)),
        _ => ctx.add(Expr::Sub(exp1, exp2)),
    };
    let reduced = ctx.add(Expr::Pow(var, difference));
    apply_power_rule(ctx, reduced, var, approach)
        .or_else(|| apply_rational_power_rule(ctx, reduced, var, approach))
}

/// Rule 4: Reciprocal power - lim c/x^n = 0 for n > 0 and c independent of x.
pub(crate) fn apply_reciprocal_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Numerator must be constant wrt variable.
    if depends_on(ctx, num, var) {
        return None;
    }

    // Denominator must be x^n with n > 0, or plain x.
    let power = if den == var {
        1
    } else if let Some((base, n)) = parse_pow_int(ctx, den) {
        if base != var || n <= 0 {
            return None;
        }
        n
    } else {
        return None;
    };

    if power > 0 {
        Some(ctx.num(0))
    } else {
        None
    }
}

/// Combine `lhs + rhs` over a common denominator into a single fraction
/// `(num_l·den_r + num_r·den_l) / (den_l·den_r)` — the additive companion of
/// [`combine_difference_over_common_denominator`]. `None` when neither side
/// is a fraction. Resolves an `∞ + (−∞)` finite limit: `x/(x−1) + 1/(1−x)`
/// becomes `(x·(1−x) + (x−1))/((x−1)(1−x))`, which reduces to `1`.
pub(super) fn combine_sum_over_common_denominator(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
    let lhs_frac = as_fraction(ctx, lhs);
    let rhs_frac = as_fraction(ctx, rhs);
    if lhs_frac.is_none() && rhs_frac.is_none() {
        return None;
    }
    let one = ctx.num(1);
    let (num_l, den_l) = lhs_frac.unwrap_or((lhs, one));
    let (num_r, den_r) = rhs_frac.unwrap_or((rhs, one));
    let t1 = ctx.add(Expr::Mul(num_l, den_r));
    let t2 = ctx.add(Expr::Mul(num_r, den_l));
    let num = ctx.add(Expr::Add(t1, t2));
    let den = ctx.add(Expr::Mul(den_l, den_r));
    Some(ctx.add(Expr::Div(num, den)))
}

/// Combine `lhs - rhs` over a common denominator into a single fraction
/// `(num_l·den_r - num_r·den_l) / (den_l·den_r)`, treating a non-fraction term as denominator 1.
/// `None` when NEITHER side is a fraction (combining a pure polynomial difference yields no new
/// structure). Used to resolve a `∞ - ∞` finite limit: `1/sin²x - 1/x²` becomes
/// `(x² - sin²x)/(x²·sin²x)`, which the limit engine evaluates to `1/3`.
/// Decompose `expr` into `(numerator, denominator)` for fraction-like forms, so an `∞ - ∞`
/// difference can be put over a common denominator. Handles an explicit `Div`, a power of a
/// quotient `(a/b)^k → (a^k, b^k)`, and the reciprocal-trig functions `csc`/`sec`/`cot` and their
/// powers: `csc(u)^k → (1, sin(u)^k)`, `sec(u)^k → (1, cos(u)^k)`, `cot(u)^k → (cos(u)^k, sin(u)^k)`.
/// `None` for anything else (the caller then treats it as denominator 1).
pub(super) fn as_fraction(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    use cas_ast::BuiltinFn;
    let (base, exp) = match ctx.get(expr).clone() {
        Expr::Pow(b, e) => (b, Some(e)),
        _ => (expr, None),
    };
    let (num_base, den_base) = match ctx.get(base).clone() {
        Expr::Div(n, d) => (n, d),
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let arg = args[0];
            if ctx.is_builtin(fn_id, BuiltinFn::Csc) {
                let one = ctx.num(1);
                let sin = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                (one, sin)
            } else if ctx.is_builtin(fn_id, BuiltinFn::Sec) {
                let one = ctx.num(1);
                let cos = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                (one, cos)
            } else if ctx.is_builtin(fn_id, BuiltinFn::Cot) {
                let cos = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                let sin = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                (cos, sin)
            } else {
                return None;
            }
        }
        _ => return None,
    };
    match exp {
        Some(e) => {
            let num = ctx.add(Expr::Pow(num_base, e));
            let den = ctx.add(Expr::Pow(den_base, e));
            Some((num, den))
        }
        None => Some((num_base, den_base)),
    }
}

pub(super) fn combine_difference_over_common_denominator(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<ExprId> {
    let lhs_frac = as_fraction(ctx, lhs);
    let rhs_frac = as_fraction(ctx, rhs);
    if lhs_frac.is_none() && rhs_frac.is_none() {
        return None;
    }
    let one = ctx.num(1);
    let (num_l, den_l) = lhs_frac.unwrap_or((lhs, one));
    let (num_r, den_r) = rhs_frac.unwrap_or((rhs, one));
    let t1 = ctx.add(Expr::Mul(num_l, den_r));
    let t2 = ctx.add(Expr::Mul(num_r, den_l));
    let num = ctx.add(Expr::Sub(t1, t2));
    let den = ctx.add(Expr::Mul(den_l, den_r));
    Some(ctx.add(Expr::Div(num, den)))
}

pub(super) fn finite_denominator_proven_nonzero(ctx: &Context, expr: ExprId) -> bool {
    if numeric_limit_value(ctx, expr).is_some_and(|value| !value.is_zero()) {
        return true;
    }
    if finite_expr_proven_positive(ctx, expr) {
        return true;
    }
    match ctx.get(expr) {
        Expr::Neg(inner) => finite_expr_proven_positive(ctx, *inner),
        _ => false,
    }
}

/// `r^m` for a rational `r` and small `m`.
pub(super) fn pow_rational(r: &BigRational, m: u32) -> BigRational {
    let mut acc = BigRational::from_integer(BigInt::from(1));
    for _ in 0..m {
        acc *= r;
    }
    acc
}

/// Binomial coefficient `C(n, k)` as a rational.
pub(super) fn binomial_rational(n: u32, k: u32) -> BigRational {
    let mut acc = BigRational::from_integer(BigInt::from(1));
    for i in 0..k {
        acc *= BigRational::new(BigInt::from((n - i) as i64), BigInt::from((i + 1) as i64));
    }
    acc
}

pub(super) fn rationalized_surd_product(
    ctx: &mut Context,
    coeff: BigRational,
    radicand: BigRational,
) -> ExprId {
    if coeff.is_zero() {
        return ctx.add(Expr::Number(coeff));
    }

    let sqrt_radicand = ctx.add(Expr::Number(radicand));
    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_radicand]);
    let abs_coeff = coeff.abs();
    let one_int = BigInt::from(1);

    let numerator = if abs_coeff.numer() == &one_int {
        sqrt_expr
    } else {
        let multiplier = ctx.add(Expr::Number(BigRational::from_integer(
            abs_coeff.numer().clone(),
        )));
        ctx.add(Expr::Mul(multiplier, sqrt_expr))
    };

    let unsigned = if abs_coeff.denom() == &one_int {
        numerator
    } else {
        let denominator = ctx.add(Expr::Number(BigRational::from_integer(
            abs_coeff.denom().clone(),
        )));
        ctx.add(Expr::Div(numerator, denominator))
    };

    if coeff.is_negative() {
        ctx.add(Expr::Neg(unsigned))
    } else {
        unsigned
    }
}

pub(super) fn rational_polynomial_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let num_growth = polynomial_or_constant_growth_info(ctx, num, var)?;
    let den_growth = polynomial_or_constant_growth_info(ctx, den, var)?;

    if num_growth.degree <= den_growth.degree {
        return None;
    }

    let degree_delta = num_growth.degree - den_growth.degree;
    let leading_ratio = num_growth.leading_coeff / den_growth.leading_coeff;
    Some(limit_growth_sign(&leading_ratio, degree_delta, approach))
}

pub(super) fn rational_polynomial_argument_zero_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let num_growth = polynomial_or_constant_growth_info(ctx, num, var)?;
    let den_growth = polynomial_or_constant_growth_info(ctx, den, var)?;

    if num_growth.degree >= den_growth.degree {
        return None;
    }

    let degree_delta = den_growth.degree - num_growth.degree;
    let leading_ratio = num_growth.leading_coeff / den_growth.leading_coeff;
    Some(limit_growth_sign(&leading_ratio, degree_delta, approach))
}

pub(super) fn rational_polynomial_argument_finite_tail_value(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
) -> Option<BigRational> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let num_growth = polynomial_or_constant_growth_info(ctx, num, var)?;
    let den_growth = polynomial_or_constant_growth_info(ctx, den, var)?;

    if num_growth.degree != den_growth.degree {
        return None;
    }

    Some(num_growth.leading_coeff / den_growth.leading_coeff)
}

pub(super) fn is_rational_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value == &rational_one())
}

/// Rational polynomial limit rule for `P(x)/Q(x)` as `x -> ±∞`.
///
/// Compares polynomial degrees in `var`:
/// - `deg(P) < deg(Q) -> 0`
/// - `deg(P) = deg(Q) -> lc(P)/lc(Q)` when both leading coefficients are numeric
/// - `deg(P) > deg(Q) -> ±∞` according to leading coefficient sign and parity
pub(crate) fn rational_poly_limit(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    // Match Div(num, den)
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Get variable name
    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    // Conservative budget for polynomial conversion
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    // Convert numerator and denominator to polynomials
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Get variable index in polynomial
    // If var not in poly, it's constant wrt var (degree 0)
    let var_idx_num = p_num.var_index(var_name);
    let var_idx_den = p_den.var_index(var_name);

    // If neither contains the variable, constant rule handles it
    if var_idx_num.is_none() && var_idx_den.is_none() {
        return None; // Let constant rule handle it
    }

    // Check for zero denominator polynomial
    if p_den.is_zero() {
        return None; // Division by zero - don't handle here
    }

    // Get degrees
    let deg_p = var_idx_num.map(|idx| p_num.degree_in(idx)).unwrap_or(0);
    let deg_q = var_idx_den.map(|idx| p_den.degree_in(idx)).unwrap_or(0);

    // Get leading coefficients
    let lc_p = var_idx_num
        .map(|idx| p_num.leading_coeff_in(idx))
        .unwrap_or_else(|| p_num.clone());
    let lc_q = var_idx_den
        .map(|idx| p_den.leading_coeff_in(idx))
        .unwrap_or_else(|| p_den.clone());

    // Both leading coefficients must be numeric constants
    let lc_p_val = lc_p.constant_value()?;
    let lc_q_val = lc_q.constant_value()?;

    // Case 1: deg(P) < deg(Q) -> 0
    if deg_p < deg_q {
        return Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))));
    }

    // Case 2: deg(P) = deg(Q) -> lc(P)/lc(Q)
    if deg_p == deg_q {
        let ratio = &lc_p_val / &lc_q_val;
        return Some(ctx.add(Expr::Number(ratio)));
    }

    // Case 3: deg(P) > deg(Q) -> ±∞
    // Sign = sign(lc_p/lc_q) * sign(x^k) where k = deg_p - deg_q
    let k = deg_p - deg_q;
    let ratio = &lc_p_val / &lc_q_val;
    let sign = limit_growth_sign(&ratio, k, approach);
    Some(mk_infinity(ctx, sign))
}

/// Combine an expression into a single fraction `(numerator, denominator)` of
/// polynomial-buildable parts, putting Add/Sub over a common denominator and
/// flattening Mul/Div. Lets a downstream polynomial reader expand and cancel
/// (e.g. `x * ((1 + 1/x) - 1)` rationalizes to `(x * (x + 1 - x)) / x`).
/// Non-rational atoms (functions, irrational powers) stay opaque over 1, so the
/// reader's own preconditions reject them.
pub(super) fn rationalize_to_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId) {
    let one = |ctx: &mut Context| ctx.num(1);
    match ctx.get(expr).clone() {
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            let subtract = matches!(ctx.get(expr), Expr::Sub(_, _));
            let (na, da) = rationalize_to_fraction(ctx, a);
            let (nb, db) = rationalize_to_fraction(ctx, b);
            let left = ctx.add(Expr::Mul(na, db));
            let right = ctx.add(Expr::Mul(nb, da));
            let num = if subtract {
                ctx.add(Expr::Sub(left, right))
            } else {
                ctx.add(Expr::Add(left, right))
            };
            let den = ctx.add(Expr::Mul(da, db));
            (num, den)
        }
        Expr::Mul(a, b) => {
            let (na, da) = rationalize_to_fraction(ctx, a);
            let (nb, db) = rationalize_to_fraction(ctx, b);
            (ctx.add(Expr::Mul(na, nb)), ctx.add(Expr::Mul(da, db)))
        }
        Expr::Div(a, b) => {
            let (na, da) = rationalize_to_fraction(ctx, a);
            let (nb, db) = rationalize_to_fraction(ctx, b);
            // (na/da) / (nb/db) = (na db) / (da nb).
            (ctx.add(Expr::Mul(na, db)), ctx.add(Expr::Mul(da, nb)))
        }
        Expr::Neg(inner) => {
            let (n, d) = rationalize_to_fraction(ctx, inner);
            (ctx.add(Expr::Neg(n)), d)
        }
        _ => {
            let d = one(ctx);
            (expr, d)
        }
    }
}

/// `(numerator, denominator)` of an expression viewed as a fraction: a Div
/// splits, anything else is over 1.
pub(super) fn rational_numerator_denominator(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId) {
    if let Expr::Div(num, den) = ctx.get(expr) {
        (*num, *den)
    } else {
        let one = ctx.num(1);
        (expr, one)
    }
}

/// Every `Div` denominator in the tree, substituted at the point, must
/// evaluate to a NONZERO Gaussian rational — the exact meromorphy gate of the
/// direct-substitution case. A transcendental denominator (cosh at a Gaussian
/// point) is not decidable here and fails conservatively.
pub(super) fn all_denominators_provably_nonzero_at(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    use num_traits::Zero;
    let node = ctx.get(expr).clone();
    match node {
        Expr::Div(l, r) => {
            let substituted = cas_ast::substitute_expr_by_id(ctx, r, var, point);
            let Some((re, im)) = eval_gaussian_const_deep(ctx, substituted) else {
                return false;
            };
            if re.is_zero() && im.is_zero() {
                return false;
            }
            all_denominators_provably_nonzero_at(ctx, l, var, point)
                && all_denominators_provably_nonzero_at(ctx, r, var, point)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            all_denominators_provably_nonzero_at(ctx, l, var, point)
                && all_denominators_provably_nonzero_at(ctx, r, var, point)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            all_denominators_provably_nonzero_at(ctx, inner, var, point)
        }
        Expr::Pow(base, _) => all_denominators_provably_nonzero_at(ctx, base, var, point),
        Expr::Function(_, args) => args
            .iter()
            .all(|&a| all_denominators_provably_nonzero_at(ctx, a, var, point)),
        _ => true,
    }
}

/// True when `sub_expr` with the point substituted folds to a NONZERO exact
/// rational.
pub(super) fn substituted_folds_to_nonzero_rational(
    ctx: &mut Context,
    sub_expr: ExprId,
    var_ids: &[ExprId],
    points: &[ExprId],
) -> bool {
    use num_traits::Zero;
    let mut value = sub_expr;
    for (var_id, point) in var_ids.iter().zip(points) {
        value = cas_ast::substitute_expr_by_id(ctx, value, *var_id, *point);
    }
    eval_rational_const_deep(ctx, value).is_some_and(|v| !v.is_zero())
}

/// Exact rational evaluation of a fully-numeric tree INCLUDING integer powers
/// (`1^2 + 1^2` → 2), which `as_rational_const` deliberately does not fold.
/// Fourth appearance of the "only literals match" lesson — this is the local
/// DECISION evaluator for the continuity prover; the shared fold stays
/// untouched (its L'Hôpital lane pins exact shapes).
pub(super) fn eval_rational_const_deep(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    use num_traits::{ToPrimitive, Zero};
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Add(l, r) => {
            Some(eval_rational_const_deep(ctx, *l)? + eval_rational_const_deep(ctx, *r)?)
        }
        Expr::Sub(l, r) => {
            Some(eval_rational_const_deep(ctx, *l)? - eval_rational_const_deep(ctx, *r)?)
        }
        Expr::Mul(l, r) => {
            Some(eval_rational_const_deep(ctx, *l)? * eval_rational_const_deep(ctx, *r)?)
        }
        Expr::Div(l, r) => {
            let d = eval_rational_const_deep(ctx, *r)?;
            if d.is_zero() {
                return None;
            }
            Some(eval_rational_const_deep(ctx, *l)? / d)
        }
        Expr::Neg(inner) => Some(-eval_rational_const_deep(ctx, *inner)?),
        Expr::Hold(inner) => eval_rational_const_deep(ctx, *inner),
        Expr::Pow(base, exp) => {
            let b = eval_rational_const_deep(ctx, *base)?;
            let e = eval_rational_const_deep(ctx, *exp)?;
            if !e.is_integer() {
                return None;
            }
            let n = e.to_integer().to_i64().filter(|n| n.unsigned_abs() <= 64)?;
            if n >= 0 {
                Some(num_traits::pow::Pow::pow(b, n as u64))
            } else {
                if b.is_zero() {
                    return None;
                }
                Some(
                    BigRational::from_integer(1.into()) / num_traits::pow::Pow::pow(b, (-n) as u64),
                )
            }
        }
        _ => None,
    }
}

/// Bottom-up cleanup of the artifacts that `x ↦ 1/x` substitution introduces: nested reciprocals
/// (`a/(b/c) → (a·c)/b`, so `1/(1/x) → x`), products by a reciprocal (`a·(1/d) → a/d`), and unit
/// `Mul` factors (`1·e → e`). Purely structural and value-preserving, so it does not affect the
/// limit; it only puts the substituted expression in the shape the finite evaluator recognises.
pub(super) fn reduce_reciprocal_substitution_artifacts(ctx: &mut Context, expr: ExprId) -> ExprId {
    let rebuilt = match *ctx.get(expr) {
        Expr::Add(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Add(a, b))
        }
        Expr::Sub(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Sub(a, b))
        }
        Expr::Mul(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Mul(a, b))
        }
        Expr::Div(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Div(a, b))
        }
        Expr::Pow(a, b) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            let b = reduce_reciprocal_substitution_artifacts(ctx, b);
            ctx.add(Expr::Pow(a, b))
        }
        Expr::Neg(a) => {
            let a = reduce_reciprocal_substitution_artifacts(ctx, a);
            ctx.add(Expr::Neg(a))
        }
        Expr::Function(fn_id, ref args) => {
            let args: Vec<ExprId> = args.clone();
            let reduced: Vec<ExprId> = args
                .into_iter()
                .map(|arg| reduce_reciprocal_substitution_artifacts(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, reduced))
        }
        _ => expr,
    };
    // Local rewrites on the rebuilt (children-reduced) node.
    match *ctx.get(rebuilt) {
        // a / (b/c) = (a·c)/b  →  in particular 1/(1/x) = x.
        Expr::Div(num, den) => {
            if let Expr::Div(inner_num, inner_den) = *ctx.get(den) {
                let new_num = mul_drop_unit(ctx, num, inner_den);
                if expr_is_one(ctx, inner_num) {
                    return new_num; // (a·c)/1
                }
                return ctx.add(Expr::Div(new_num, inner_num));
            }
            rebuilt
        }
        // a·(1/d) = a/d, and drop unit factors.
        Expr::Mul(a, b) => {
            if expr_is_one(ctx, a) {
                return b;
            }
            if expr_is_one(ctx, b) {
                return a;
            }
            if let Expr::Div(bn, bd) = *ctx.get(b) {
                if expr_is_one(ctx, bn) {
                    return ctx.add(Expr::Div(a, bd));
                }
            }
            if let Expr::Div(an, ad) = *ctx.get(a) {
                if expr_is_one(ctx, an) {
                    return ctx.add(Expr::Div(b, ad));
                }
            }
            rebuilt
        }
        _ => rebuilt,
    }
}
