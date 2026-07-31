//! `limits_support`: familia `logs_exp`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

fn finite_unit_log_base(
    ctx: &mut Context,
    base_expr: ExprId,
    var: ExprId,
    point: ExprId,
    var_name: &str,
    point_value: &BigRational,
    unit_boundary_side: Option<FiniteLimitSide>,
) -> Option<UnitLogBase> {
    let base = if let Some(base) = constant_rational_value(ctx, base_expr) {
        base
    } else if let Ok(base_poly) = Polynomial::from_expr(ctx, base_expr, var_name) {
        base_poly.eval(point_value)
    } else {
        let base_limit = try_limit_rules_at_finite(ctx, base_expr, var, point)?;
        constant_rational_value(ctx, base_limit)?
    };

    if !base.is_positive() {
        return None;
    }
    if base == rational_one() {
        let side = unit_boundary_side?;
        let base_tail =
            finite_endpoint_unit_base_tail_sign(ctx, base_expr, var_name, point_value, side)?;
        return Some(UnitLogBase::UnitBoundary(base_tail));
    }
    Some(UnitLogBase::Fixed(base))
}

fn scaled_unit_log_argument(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    var_name: &str,
    point_value: &BigRational,
    unit_boundary_side: Option<FiniteLimitSide>,
) -> Option<(BigRational, ExprId, UnitLogBase)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id)? {
            BuiltinFn::Ln => Some((BigRational::one(), args[0], UnitLogBase::Natural)),
            BuiltinFn::Log2 => Some((
                BigRational::one(),
                args[0],
                UnitLogBase::Fixed(BigRational::from_integer(BigInt::from(2))),
            )),
            BuiltinFn::Log10 => Some((
                BigRational::one(),
                args[0],
                UnitLogBase::Fixed(BigRational::from_integer(BigInt::from(10))),
            )),
            _ => None,
        },
        Expr::Function(fn_id, args) if args.len() == 2 && ctx.is_builtin(fn_id, BuiltinFn::Log) => {
            let base = finite_unit_log_base(
                ctx,
                args[0],
                var,
                point,
                var_name,
                point_value,
                unit_boundary_side,
            )?;
            Some((BigRational::one(), args[1], base))
        }
        Expr::Neg(inner) => {
            let (scale, argument, base) = scaled_unit_log_argument(
                ctx,
                inner,
                var,
                point,
                var_name,
                point_value,
                unit_boundary_side,
            )?;
            Some((-scale, argument, base))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, argument, base) = scaled_unit_log_argument(
                    ctx,
                    rhs,
                    var,
                    point,
                    var_name,
                    point_value,
                    unit_boundary_side,
                )?;
                return Some((scale * inner_scale, argument, base));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, argument, base) = scaled_unit_log_argument(
                    ctx,
                    lhs,
                    var,
                    point,
                    var_name,
                    point_value,
                    unit_boundary_side,
                )?;
                return Some((scale * inner_scale, argument, base));
            }
            None
        }
        _ => None,
    }
}

fn finite_log_unit_quotient_result(
    ctx: &mut Context,
    value: BigRational,
    base: UnitLogBase,
) -> ExprId {
    match base {
        UnitLogBase::Natural => ctx.add(Expr::Number(value)),
        UnitLogBase::Fixed(base) => {
            if value.is_zero() {
                return ctx.add(Expr::Number(value));
            }
            let numerator = ctx.add(Expr::Number(value));
            let base_expr = ctx.add(Expr::Number(base));
            let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
            ctx.add(Expr::Div(numerator, ln_base))
        }
        UnitLogBase::UnitBoundary(_) => {
            unreachable!("unit-boundary log bases are only valid for endpoint limits")
        }
    }
}

fn unit_log_base_tail_coeff(base: &UnitLogBase) -> BigRational {
    match base {
        UnitLogBase::Natural => rational_one(),
        UnitLogBase::Fixed(base) if base > &rational_one() => rational_one(),
        UnitLogBase::Fixed(_) => -rational_one(),
        UnitLogBase::UnitBoundary(InfSign::Pos) => rational_one(),
        UnitLogBase::UnitBoundary(InfSign::Neg) => -rational_one(),
    }
}

pub(super) fn apply_finite_one_sided_log_endpoint_rule(
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
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();

    let (scale, log_argument, base) =
        scaled_unit_log_argument(ctx, expr, var, point, &var_name, &point_value, Some(side))?;
    if scale.is_zero() {
        return None;
    }

    if finite_endpoint_argument_zero_tail_sign(ctx, log_argument, &var_name, &point_value, side)?
        != InfSign::Pos
    {
        return None;
    }

    let total_scale = scale * unit_log_base_tail_coeff(&base);
    scale_infinity(ctx, &total_scale, InfSign::Neg)
}

pub(super) fn apply_finite_bilateral_log_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left =
        apply_finite_one_sided_log_endpoint_rule(ctx, expr, var, point, FiniteLimitSide::Left)?;
    let right =
        apply_finite_one_sided_log_endpoint_rule(ctx, expr, var, point, FiniteLimitSide::Right)?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

fn exp_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Exp) => {
            Some(args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => Some(exp),
        _ => None,
    }
}

pub(super) fn scaled_exp_zero_offset_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, rhs) => {
            Some((BigRational::one(), exp_argument(ctx, lhs)?))
        }
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, lhs) => {
            Some((-BigRational::one(), exp_argument(ctx, rhs)?))
        }
        Expr::Add(lhs, rhs) => {
            if constant_rational_value(ctx, rhs).is_some_and(|value| value == -BigRational::one()) {
                return Some((BigRational::one(), exp_argument(ctx, lhs)?));
            }
            if constant_rational_value(ctx, lhs).is_some_and(|value| value == -BigRational::one()) {
                return Some((BigRational::one(), exp_argument(ctx, rhs)?));
            }
            None
        }
        Expr::Neg(inner) => {
            let (scale, argument) = scaled_exp_zero_offset_argument(ctx, inner)?;
            Some((-scale, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, argument) = scaled_exp_zero_offset_argument(ctx, rhs)?;
                return Some((scale * inner_scale, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, argument) = scaled_exp_zero_offset_argument(ctx, lhs)?;
                return Some((scale * inner_scale, argument));
            }
            None
        }
        _ => None,
    }
}

pub(super) fn apply_finite_exp_zero_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    apply_finite_zero_quotient_rule(ctx, expr, var, point, scaled_exp_zero_offset_argument)
}

/// `(a^(g) - 1) / h(x) -> ln(a) * lim(g/h)` as `var -> point`, the
/// derivative-of-`a^x`-at-0 family: `(2^x - 1)/x = ln 2`, `(3^x - 1)/x = ln 3`,
/// `(2^(3x) - 1)/x = 3 ln 2`. The base `a` must be a positive rational != 1
/// (the natural base `e^x - 1` is left to apply_finite_exp_zero_quotient_rule,
/// which gives the cleaner 1 instead of ln(e)). Since `a^g ~ 1 + g ln a` as
/// `g -> 0`, the numerator's first-order equivalent is `g ln a`, and the limit
/// is `ln(a)` times the rational limit of `g/h`.
pub(super) fn apply_finite_general_exp_zero_quotient_rule(
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
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (scale, base, exponent) = scaled_general_power_zero_offset(ctx, num)?;
    // The natural base stays with apply_finite_exp_zero_quotient_rule (which
    // gives the cleaner 1 instead of ln(e)); any other base — rational or
    // provable-constant like π — must be provably positive and != 1.
    if matches!(ctx.get(base), Expr::Constant(Constant::E)) {
        return None;
    }
    if constant_base_vs_one(ctx, base)? == std::cmp::Ordering::Equal {
        return None;
    }
    let exponent_poly = Polynomial::from_expr(ctx, exponent, &var_name).ok()?;
    // a^g -> 1 (so a^g - 1 -> 0) requires the exponent to vanish at the point.
    if !exponent_poly.eval(&point_value).is_zero() {
        return None;
    }
    let scale_poly = Polynomial::new(vec![scale], var_name.clone());
    let numerator = exponent_poly.mul(&scale_poly);
    let denominator = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    let ratio = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    // 0 * ln(a) = 0 (g vanishes faster than h); fold it rather than emit 0*ln(a).
    if ratio.is_zero() {
        return Some(ctx.num(0));
    }
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    if ratio.is_one() {
        return Some(ln_base);
    }
    let ratio_expr = ctx.add(Expr::Number(ratio));
    Some(ctx.add(Expr::Mul(ratio_expr, ln_base)))
}

/// An exponential atom `base^(g)`: returns `(Some(rational_base), g)` for a
/// numeric base != 1, or `(None, g)` for the natural base e (ln(e) = 1). Both
/// `Pow(E, g)` and `exp(g)` are recognized.
fn exponential_base_and_exponent(
    ctx: &Context,
    expr: ExprId,
) -> Option<(Option<BigRational>, ExprId)> {
    use num_traits::One;
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Exp) =>
        {
            Some((None, args[0]))
        }
        Expr::Pow(base, exponent) => {
            if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
                return Some((None, *exponent));
            }
            let base_value = constant_rational_value(ctx, *base)?;
            if !base_value.is_positive() || base_value.is_one() {
                return None;
            }
            Some((Some(base_value), *exponent))
        }
        _ => None,
    }
}

/// Accumulate `expr` (under overall `sign`) as a linear combination of
/// exponentials `c * a^(g)` (a a positive rational != 1, or e) plus constants,
/// reading off the value at 0 and the first derivative at 0. Each exponential
/// requires `g` a polynomial with `g(0) = 0` and degree >= 1, so `a^g -> 1`
/// and the derivative contribution is `c * g'(0) * ln(a)` (or `c * g'(0)` for
/// the base e). Returns None for any term outside the class.
#[allow(clippy::too_many_arguments)]
fn accumulate_exp_combination(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    sign: &BigRational,
    value: &mut BigRational,
    log_terms: &mut Vec<(BigRational, BigRational)>,
    const_deriv: &mut BigRational,
    saw_exp: &mut bool,
) -> Option<()> {
    use num_traits::Zero;
    // A pure constant contributes to the value, nothing to the derivative.
    if let Some(c) = constant_rational_value(ctx, expr) {
        *value += sign * &c;
        return Some(());
    }
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let neg = -sign.clone();
            accumulate_exp_combination(
                ctx,
                inner,
                var_name,
                &neg,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )
        }
        Expr::Add(a, b) => {
            accumulate_exp_combination(
                ctx,
                a,
                var_name,
                sign,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )?;
            accumulate_exp_combination(
                ctx,
                b,
                var_name,
                sign,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )
        }
        Expr::Sub(a, b) => {
            accumulate_exp_combination(
                ctx,
                a,
                var_name,
                sign,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )?;
            let neg = -sign.clone();
            accumulate_exp_combination(
                ctx,
                b,
                var_name,
                &neg,
                value,
                log_terms,
                const_deriv,
                saw_exp,
            )
        }
        Expr::Mul(a, b) => {
            // Exactly one factor must be an x-free rational scale.
            if let Some(scale) = constant_rational_value(ctx, a) {
                let s = sign * &scale;
                return accumulate_exp_combination(
                    ctx,
                    b,
                    var_name,
                    &s,
                    value,
                    log_terms,
                    const_deriv,
                    saw_exp,
                );
            }
            if let Some(scale) = constant_rational_value(ctx, b) {
                let s = sign * &scale;
                return accumulate_exp_combination(
                    ctx,
                    a,
                    var_name,
                    &s,
                    value,
                    log_terms,
                    const_deriv,
                    saw_exp,
                );
            }
            None
        }
        _ => {
            let (base_opt, g) = exponential_base_and_exponent(ctx, expr)?;
            let g_poly = Polynomial::from_expr(ctx, g, var_name).ok()?;
            if !g_poly.eval(&BigRational::zero()).is_zero() || g_poly.degree() < 1 {
                return None;
            }
            let slope = g_poly
                .coeffs
                .get(1)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            *saw_exp = true;
            // a^(g(0)) = a^0 = 1 contributes to the value at 0.
            *value += sign.clone();
            let coeff = sign * &slope;
            match base_opt {
                Some(base) => {
                    if !coeff.is_zero() {
                        log_terms.push((coeff, base));
                    }
                }
                None => *const_deriv += coeff, // ln(e) = 1
            }
            Some(())
        }
    }
}

/// Run the exponential-combination accumulator and return its first-derivative
/// `(log terms, constant)` only when the expression genuinely vanishes at 0
/// over a real exponential combination.
pub(super) fn exp_combination_first_derivative(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(Vec<(BigRational, BigRational)>, BigRational)> {
    use num_traits::Zero;
    let mut value = BigRational::zero();
    let mut log_terms: Vec<(BigRational, BigRational)> = Vec::new();
    let mut const_deriv = BigRational::zero();
    let mut saw_exp = false;
    accumulate_exp_combination(
        ctx,
        expr,
        var_name,
        &rational_one(),
        &mut value,
        &mut log_terms,
        &mut const_deriv,
        &mut saw_exp,
    )?;
    if !saw_exp || !value.is_zero() {
        return None;
    }
    Some((log_terms, const_deriv))
}

/// EXACT zero test for `constant + Σ cᵢ·ln(bᵢ)` with rational `cᵢ` and
/// positive rational bases `bᵢ`: factor every base numerator/denominator
/// over a pairwise-coprime refinement and read the sum as
/// `constant + Σₘ dₘ·ln(m)`. The logs of pairwise-coprime integers > 1 are
/// linearly independent over Q, and for `constant != 0` the sum can never
/// vanish (it would make `e^q` rational for a rational `q != 0`, against
/// Lindemann–Weierstrass) — so the sum is zero IFF `constant = 0` AND every
/// `dₘ = 0`. Never a float: this gate EMITS (fold-to-0) and DECLINES
/// (division by zero) on soundness paths. `None` only if a base fails to
/// factor over the refined set (internal safety net) — callers must then
/// treat zero-ness as unknown.
pub(super) fn exact_log_combination_is_zero(
    log_terms: &[(BigRational, BigRational)],
    constant: &BigRational,
) -> Option<bool> {
    let mut contributions: Vec<(BigRational, BigInt)> = Vec::new();
    for (coeff, base) in log_terms {
        if !base.is_positive() {
            return None; // no real log — upstream gates exclude this
        }
        if coeff.is_zero() {
            continue;
        }
        let numer = base.numer().clone();
        let denom = base.denom().clone();
        if !numer.is_one() {
            contributions.push((coeff.clone(), numer));
        }
        if !denom.is_one() {
            contributions.push((-coeff.clone(), denom));
        }
    }
    let factors: Vec<BigInt> = contributions.iter().map(|(_, v)| v.clone()).collect();
    let base_set = coprime_refinement(&factors);
    // Per-element exponent coefficients: d_m = Σ cᵢ · (multiplicity of m in vᵢ).
    let mut totals: Vec<BigRational> = vec![BigRational::zero(); base_set.len()];
    for (coeff, value) in contributions {
        let mut rest = value;
        for (m, total) in base_set.iter().zip(totals.iter_mut()) {
            let mut multiplicity = 0i64;
            while (&rest % m).is_zero() {
                rest /= m;
                multiplicity += 1;
            }
            if multiplicity != 0 {
                *total += &coeff * BigRational::from_integer(multiplicity.into());
            }
        }
        if !rest.is_one() {
            return None; // incomplete factorization — zero-ness unknown
        }
    }
    Some(constant.is_zero() && totals.iter().all(|t| t.is_zero()))
}

/// Build the expression `const + sum coeff_i ln(base_i)`.
pub(super) fn build_log_combination_expr(
    ctx: &mut Context,
    log_terms: Vec<(BigRational, BigRational)>,
    constant: BigRational,
) -> ExprId {
    use num_traits::{One, Zero};
    let mut result: Option<ExprId> = None;
    if !constant.is_zero() {
        result = Some(ctx.add(Expr::Number(constant)));
    }
    for (coeff, base) in log_terms {
        if coeff.is_zero() {
            continue;
        }
        let base_expr = ctx.add(Expr::Number(base));
        let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
        let term = if coeff.is_one() {
            ln_base
        } else if coeff == -rational_one() {
            ctx.add(Expr::Neg(ln_base))
        } else {
            let coeff_expr = ctx.add(Expr::Number(coeff));
            ctx.add(Expr::Mul(coeff_expr, ln_base))
        };
        result = Some(match result {
            Some(acc) => ctx.add(Expr::Add(acc, term)),
            None => term,
        });
    }
    result.unwrap_or_else(|| ctx.num(0))
}

pub(super) fn apply_finite_exp_linear_combination_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
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

    let mut value = BigRational::zero();
    let mut log_terms: Vec<(BigRational, BigRational)> = Vec::new();
    let mut const_deriv = BigRational::zero();
    let mut saw_exp = false;
    accumulate_exp_combination(
        ctx,
        num,
        &var_name,
        &rational_one(),
        &mut value,
        &mut log_terms,
        &mut const_deriv,
        &mut saw_exp,
    )?;
    // A genuine 0/0 over a real exponential combination.
    if !saw_exp || !value.is_zero() {
        return None;
    }
    // Denominator: a polynomial vanishing to exactly first order at 0.
    let den_poly = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    if !den_poly.eval(&BigRational::zero()).is_zero() {
        return None;
    }
    let den_slope = den_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if den_slope.is_zero() {
        return None;
    }
    // N'(0) exactly 0 (e.g. 12^(2x) − 144^x): fold to a clean 0 instead of
    // emitting a zero-valued log combination. Exact decision, never a float.
    if exact_log_combination_is_zero(&log_terms, &const_deriv) == Some(true) {
        return Some(ctx.num(0));
    }

    // result = (const_deriv + sum coeff_i ln(base_i)) / den_slope.
    let mut result: Option<ExprId> = None;
    let scaled_const = &const_deriv / &den_slope;
    if !scaled_const.is_zero() {
        result = Some(ctx.add(Expr::Number(scaled_const)));
    }
    for (coeff, base) in log_terms {
        let scaled = &coeff / &den_slope;
        if scaled.is_zero() {
            continue;
        }
        let base_expr = ctx.add(Expr::Number(base));
        let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
        let term = if scaled.is_one() {
            ln_base
        } else if scaled == -rational_one() {
            ctx.add(Expr::Neg(ln_base))
        } else {
            let coeff_expr = ctx.add(Expr::Number(scaled));
            ctx.add(Expr::Mul(coeff_expr, ln_base))
        };
        result = Some(match result {
            Some(acc) => ctx.add(Expr::Add(acc, term)),
            None => term,
        });
    }
    Some(result.unwrap_or_else(|| ctx.num(0)))
}

pub(super) fn apply_finite_log_unit_quotient_rule(
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
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (scale, log_argument, base) =
        scaled_unit_log_argument(ctx, num, var, point, &var_name, &point_value, None)?;
    let argument = Polynomial::from_expr(ctx, log_argument, &var_name).ok()?;
    if argument.eval(&point_value) != BigRational::one() {
        return None;
    }

    let unit_offset = argument.sub(&Polynomial::one(var_name.clone()));
    let scale_poly = Polynomial::new(vec![scale], var_name.clone());
    let numerator = unit_offset.mul(&scale_poly);
    let denominator = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(finite_log_unit_quotient_result(ctx, value, base))
}

pub(super) fn finite_exp_exact_expr_result(
    ctx: &mut Context,
    argument_limit: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(argument_limit).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let inner = args[0];
    finite_expr_proven_positive(ctx, inner).then_some(inner)
}

pub(super) fn finite_ln_exact_expr_result(
    ctx: &mut Context,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if matches!(ctx.get(argument_limit), Expr::Constant(Constant::E)) {
        return Some(ctx.num(1));
    }

    let Expr::Function(fn_id, args) = ctx.get(argument_limit).clone() else {
        return None;
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Exp) && args.len() == 1 {
        Some(args[0])
    } else {
        None
    }
}

fn finite_log_base_limit_is_valid(ctx: &Context, base_limit: ExprId) -> bool {
    let Some(base_value) = numeric_limit_value(ctx, base_limit) else {
        return false;
    };
    base_value.is_positive() && base_value != rational_one()
}

pub(super) fn finite_log_result(
    ctx: &mut Context,
    base_limit: ExprId,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if !finite_log_base_limit_is_valid(ctx, base_limit) {
        return None;
    }
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if !argument_value.is_positive() {
            return None;
        }
        if let Some(exact_result) =
            finite_log_exact_numeric_result(ctx, base_limit, &argument_value)
        {
            return Some(exact_result);
        }
        let value_expr = ctx.add(Expr::Number(argument_value));
        return Some(ctx.call_builtin(BuiltinFn::Log, vec![base_limit, value_expr]));
    }

    finite_expr_proven_positive(ctx, argument_limit)
        .then(|| ctx.call_builtin(BuiltinFn::Log, vec![base_limit, argument_limit]))
}

fn finite_log_exact_numeric_result(
    ctx: &mut Context,
    base_limit: ExprId,
    argument_value: &BigRational,
) -> Option<ExprId> {
    if argument_value.is_one() {
        return Some(ctx.num(0));
    }

    let base_value = numeric_limit_value(ctx, base_limit)?;
    if base_value == *argument_value {
        return Some(ctx.num(1));
    }

    finite_exact_rational_log_result(ctx, &base_value, argument_value)
}

pub(super) fn finite_exact_rational_log_result(
    ctx: &mut Context,
    base_value: &BigRational,
    argument_value: &BigRational,
) -> Option<ExprId> {
    let exponent = exact_rational_log_result(base_value, argument_value)?;
    Some(ctx.add(Expr::Number(exponent)))
}

fn exact_rational_log_result(
    base_value: &BigRational,
    argument_value: &BigRational,
) -> Option<BigRational> {
    if !base_value.is_positive()
        || base_value.is_one()
        || !argument_value.is_positive()
        || argument_value.is_one()
    {
        return None;
    }

    for denominator in 1..=FINITE_LOG_EXACT_RATIONAL_DENOMINATOR_LIMIT {
        let argument_power = rational_pow_nonnegative(argument_value, denominator as u64);
        for numerator in 1..=FINITE_LOG_EXACT_RATIONAL_NUMERATOR_LIMIT {
            let base_power = rational_pow_nonnegative(base_value, numerator as u64);
            if argument_power == base_power {
                return Some(BigRational::new(
                    BigInt::from(numerator),
                    BigInt::from(denominator),
                ));
            }
            if argument_power == BigRational::one() / base_power {
                return Some(BigRational::new(
                    BigInt::from(-numerator),
                    BigInt::from(denominator),
                ));
            }
        }
    }

    None
}

pub(super) fn apply_finite_binary_log_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Log) || args.len() != 2 {
        return None;
    }

    let base_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    let argument_limit = try_limit_rules_at_finite(ctx, args[1], var, point)?;
    finite_log_result(ctx, base_limit, argument_limit)
}

/// u^p * ln(u)^q -> 0 as u -> 0 from the side where u > 0, for p > 0 and
/// q >= 1, with u = var - point (or var itself at point 0).
pub(super) fn power_log_dominance_zero_limit(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    // u must approach 0 from the positive side: right of the point.
    if !matches!(side, FiniteLimitSide::Right) {
        return None;
    }
    let (power_factor, log_factor) =
        if one_sided_log_power_of_shift(ctx, right, var, point).is_some() {
            (left, right)
        } else if one_sided_log_power_of_shift(ctx, left, var, point).is_some() {
            (right, left)
        } else {
            return None;
        };
    one_sided_log_power_of_shift(ctx, log_factor, var, point)?;
    let exponent = one_sided_positive_power_of_shift(ctx, power_factor, var, point)?;
    if exponent.is_positive() {
        Some(ctx.num(0))
    } else {
        None
    }
}

/// `sum_i c_i (var-point)^{a_i} P_i(ln(var-point)) -> 0` as `var -> point+`,
/// where every additive term carries a STRICTLY POSITIVE total power of
/// `(var-point)` and otherwise only a polynomial in `ln(var-point)` and
/// var-free constants. A positive power dominates any polynomial in the
/// logarithm, so each term -> 0 and the sum -> 0.
///
/// This generalizes `power_log_dominance_zero_limit` (a single
/// `u^p * ln(u)^q` product) to the antiderivatives of `x^a ln(x)^b`, e.g.
/// `int ln(x)^2 dx = x(ln(x)^2 - 2 ln(x) + 2)` and
/// `int ln(x)/sqrt(x) dx = 2 sqrt(x) ln(x) - 4 sqrt(x)`, whose lower
/// endpoint touches 0 and whose boundary value the definite integrator
/// needs as a one-sided limit.
pub(super) fn apply_finite_one_sided_power_log_polynomial_zero(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    // ln(var - point) is real only to the right of the point.
    if !matches!(side, FiniteLimitSide::Right) {
        return None;
    }
    if !depends_on(ctx, expr, var) {
        return None;
    }
    power_log_polynomial_sum_to_zero(ctx, expr, var, point).then(|| ctx.num(0))
}

/// Every additive term of `expr` is power-log dominated to zero.
fn power_log_polynomial_sum_to_zero(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            power_log_polynomial_sum_to_zero(ctx, lhs, var, point)
                && power_log_polynomial_sum_to_zero(ctx, rhs, var, point)
        }
        Expr::Neg(inner) => power_log_polynomial_sum_to_zero(ctx, inner, var, point),
        _ => power_log_term_dominated_to_zero(ctx, expr, var, point),
    }
}

/// A single multiplicative term tends to 0 from the right: its factors are
/// powers of `(var-point)`, polynomials in `ln(var-point)`, and var-free
/// constants, and the total `(var-point)` exponent is strictly positive.
fn power_log_term_dominated_to_zero(
    ctx: &Context,
    term: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    let mut total_power = BigRational::zero();
    let mut saw_power = false;
    for factor in collect_mul_factors(ctx, term) {
        if let Some(exponent) = shift_power_exponent(ctx, factor, var, point) {
            total_power += exponent;
            saw_power = true;
        } else if is_var_shift_log_polynomial(ctx, factor, var, point) {
            // ln-polynomial growth is dominated by any positive power.
        } else if depends_on(ctx, factor, var) {
            // A factor that is neither a (var-point) power nor a log
            // polynomial (e.g. sin, exp, a foreign variable) is unclassified.
            return false;
        }
        // A var-free constant factor neither vanishes nor adds power.
    }
    saw_power && total_power.is_positive()
}

/// `expr` is a polynomial in `ln(var-point)`: var-free constants, the bare
/// `ln(var-point)`, its non-negative integer powers, and their sums and
/// products. Restricted to integer powers because `ln(var-point) < 0` near
/// the point makes fractional powers leave the reals.
fn is_var_shift_log_polynomial(ctx: &Context, expr: ExprId, var: ExprId, point: ExprId) -> bool {
    if !depends_on(ctx, expr, var) {
        return true;
    }
    let is_ln_of_shift = |candidate: ExprId| -> bool {
        matches!(ctx.get(candidate), Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln))
                && is_var_shift(ctx, args[0], var, point))
    };
    if is_ln_of_shift(expr) {
        return true;
    }
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_ln_of_shift(*base) => {
            crate::numeric_eval::as_rational_const(ctx, *exp)
                .is_some_and(|value| value.is_integer() && !value.is_negative())
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_var_shift_log_polynomial(ctx, *lhs, var, point)
                && is_var_shift_log_polynomial(ctx, *rhs, var, point)
        }
        Expr::Neg(inner) => is_var_shift_log_polynomial(ctx, *inner, var, point),
        _ => false,
    }
}

/// Recognize ln(var - point)^q (q >= 1 rational; q = 1 for the bare ln).
fn one_sided_log_power_of_shift(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<BigRational> {
    let is_ln_of_shift = |ctx: &Context, candidate: ExprId| -> bool {
        matches!(ctx.get(candidate), Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln))
                && is_var_shift(ctx, args[0], var, point))
    };
    if is_ln_of_shift(ctx, expr) {
        return Some(rational_one());
    }
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) if is_ln_of_shift(ctx, base) => {
            let value = crate::numeric_eval::as_rational_const(ctx, exp)?;
            (value >= rational_one()).then_some(value)
        }
        _ => None,
    }
}

pub(super) fn linear_exp_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp)) =>
        {
            linear_argument_tail_sign(ctx, args[0], var, approach)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            linear_argument_tail_sign(ctx, exp, var, approach)
        }
        Expr::Pow(_, _) => {
            general_base_pow_tail_sign(ctx, expr, var, approach, linear_argument_tail_sign)
        }
        _ => None,
    }
}

fn polynomial_exp_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp)) =>
        {
            polynomial_argument_tail_sign(ctx, args[0], var, approach)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            polynomial_argument_tail_sign(ctx, exp, var, approach)
        }
        Expr::Pow(_, _) => {
            general_base_pow_tail_sign(ctx, expr, var, approach, polynomial_argument_tail_sign)
        }
        _ => None,
    }
}

pub(super) fn scaled_polynomial_exp_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledPolynomialExpTailInfo> {
    if let Some(tail) = polynomial_exp_tail_sign(ctx, expr, var, approach) {
        return Some(ScaledPolynomialExpTailInfo {
            coeff: BigRational::from_integer(BigInt::from(1)),
            tail,
        });
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut info = scaled_polynomial_exp_tail_info(ctx, inner, var, approach)?;
            info.coeff = -info.coeff;
            Some(info)
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(lhs_scale) = numeric_limit_value(ctx, lhs) {
                if let Some(mut rhs_info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach)
                {
                    rhs_info.coeff *= lhs_scale;
                    return Some(rhs_info);
                }
            }
            if let Some(rhs_scale) = numeric_limit_value(ctx, rhs) {
                if let Some(mut lhs_info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach)
                {
                    lhs_info.coeff *= rhs_scale;
                    return Some(lhs_info);
                }
            }
            None
        }
        _ => None,
    }
}

pub(super) fn nonzero_scaled_polynomial_exp_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledPolynomialExpTailInfo> {
    let info = scaled_polynomial_exp_tail_info(ctx, expr, var, approach)?;
    if info.coeff.is_zero() {
        None
    } else {
        Some(info)
    }
}

fn exact_named_log_base_tail_sign(ctx: &Context, base: ExprId) -> Option<InfSign> {
    if known_positive_constant_exceeds_one(ctx, base) {
        return Some(InfSign::Pos);
    }

    if let Expr::Div(num, den) = ctx.get(base).clone() {
        if is_rational_one(ctx, num) {
            return match exact_named_log_base_tail_sign(ctx, den)? {
                InfSign::Pos => Some(InfSign::Neg),
                InfSign::Neg => Some(InfSign::Pos),
            };
        }
    }

    if let Expr::Pow(pow_base, exp) = ctx.get(base).clone() {
        let base_sign = exact_named_log_base_tail_sign(ctx, pow_base)?;
        let exponent = crate::expr_extract::extract_i64_integer(ctx, exp)?;
        if exponent == 0 {
            return None;
        }
        return match (base_sign, exponent.is_positive()) {
            (InfSign::Pos, true) | (InfSign::Neg, false) => Some(InfSign::Pos),
            (InfSign::Pos, false) | (InfSign::Neg, true) => Some(InfSign::Neg),
        };
    }

    None
}

fn log_base_tail_coeff_from_sign(sign: InfSign) -> BigRational {
    match sign {
        InfSign::Pos => rational_one(),
        InfSign::Neg => -rational_one(),
    }
}

pub(super) fn positive_log_base_tail_coeff(ctx: &Context, base: ExprId) -> Option<BigRational> {
    if let Some(sign) = exact_named_log_base_tail_sign(ctx, base) {
        return Some(log_base_tail_coeff_from_sign(sign));
    }

    let base_value = constant_rational_value(ctx, base)?;
    let one = rational_one();
    if !base_value.is_positive() || base_value == one {
        return None;
    }
    if base_value > one {
        Some(one)
    } else {
        Some(-one)
    }
}

pub(super) fn log_argument_tail_coeff(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<BigRational> {
    if unbounded_argument_tail_sign(ctx, arg, var, approach) == Some(InfSign::Pos) {
        return Some(rational_one());
    }

    if rational_polynomial_argument_zero_tail_sign(ctx, arg, var, approach) == Some(InfSign::Pos) {
        return Some(-rational_one());
    }

    None
}

/// The argument `P` of a bare `ln(P)` (natural logarithm, single argument).
pub(super) fn bare_natural_log_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln)) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

/// Accumulate `expr` (scaled by `sign`) into signed natural-log terms and signed arctangent terms,
/// each recorded as `(coeff, argument)`. A log leaf is a rational multiple of a bare `ln(P)` (argument
/// optionally under `abs`); an arctan leaf is a rational multiple of `arctan(Q)`/`atan(Q)`. Any leaf
/// outside those two shapes declines the whole decomposition.
pub(super) fn collect_signed_log_and_arctan_terms(
    ctx: &Context,
    expr: ExprId,
    sign: &BigRational,
    log_terms: &mut Vec<(BigRational, ExprId)>,
    arctan_terms: &mut Vec<(BigRational, ExprId)>,
) -> Option<()> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => collect_signed_log_and_arctan_terms(
            ctx,
            inner,
            &(-sign.clone()),
            log_terms,
            arctan_terms,
        ),
        Expr::Add(a, b) => {
            collect_signed_log_and_arctan_terms(ctx, a, sign, log_terms, arctan_terms)?;
            collect_signed_log_and_arctan_terms(ctx, b, sign, log_terms, arctan_terms)
        }
        Expr::Sub(a, b) => {
            collect_signed_log_and_arctan_terms(ctx, a, sign, log_terms, arctan_terms)?;
            collect_signed_log_and_arctan_terms(ctx, b, &(-sign.clone()), log_terms, arctan_terms)
        }
        Expr::Mul(a, b) => {
            if let Some(c) = constant_rational_value(ctx, a) {
                collect_signed_log_and_arctan_terms(ctx, b, &(sign * &c), log_terms, arctan_terms)
            } else if let Some(c) = constant_rational_value(ctx, b) {
                collect_signed_log_and_arctan_terms(ctx, a, &(sign * &c), log_terms, arctan_terms)
            } else {
                None
            }
        }
        Expr::Div(a, b) => {
            let c = constant_rational_value(ctx, b)?;
            if c.is_zero() {
                return None;
            }
            collect_signed_log_and_arctan_terms(ctx, a, &(sign / &c), log_terms, arctan_terms)
        }
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Ln) => {
                log_terms.push((sign.clone(), strip_single_abs(ctx, args[0])));
                Some(())
            }
            Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
                arctan_terms.push((sign.clone(), args[0]));
                Some(())
            }
            _ => None,
        },
        _ => None,
    }
}

/// Accumulate `expr` (under `sign`) as a sum of rational-base exponentials
/// `c * b^(s x)` recorded as `(c, effective base b^s)`, plus constants
/// (effective base 1). Returns None for any term outside the class.
pub(super) fn collect_rational_exp_terms(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    sign: &BigRational,
    terms: &mut Vec<(BigRational, BigRational)>,
) -> Option<()> {
    use num_traits::{One, Signed, Zero};
    if let Some(c) = constant_rational_value(ctx, expr) {
        terms.push((sign * &c, BigRational::one()));
        return Some(());
    }
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let neg = -sign.clone();
            collect_rational_exp_terms(ctx, inner, var_name, &neg, terms)
        }
        Expr::Add(a, b) => {
            collect_rational_exp_terms(ctx, a, var_name, sign, terms)?;
            collect_rational_exp_terms(ctx, b, var_name, sign, terms)
        }
        Expr::Sub(a, b) => {
            collect_rational_exp_terms(ctx, a, var_name, sign, terms)?;
            let neg = -sign.clone();
            collect_rational_exp_terms(ctx, b, var_name, &neg, terms)
        }
        Expr::Mul(a, b) => {
            if let Some(scale) = constant_rational_value(ctx, a) {
                let s = sign * &scale;
                return collect_rational_exp_terms(ctx, b, var_name, &s, terms);
            }
            if let Some(scale) = constant_rational_value(ctx, b) {
                let s = sign * &scale;
                return collect_rational_exp_terms(ctx, a, var_name, &s, terms);
            }
            None
        }
        _ => {
            // b^(s x): rational base b > 0, exponent a positive-integer multiple
            // of x, so the effective base b^s is a comparable rational.
            let (base, exponent) = numeric_base_power(ctx, expr)?;
            if !base.is_positive() {
                return None;
            }
            let exp_poly = Polynomial::from_expr(ctx, exponent, var_name).ok()?;
            if exp_poly.degree() != 1 || !exp_poly.coeffs.first().is_some_and(|c| c.is_zero()) {
                return None;
            }
            let slope = exp_poly.coeffs.get(1).cloned()?;
            if !slope.is_integer() || !slope.is_positive() {
                return None;
            }
            let s = u32::try_from(slope.to_integer()).ok()?;
            if s > 64 {
                return None;
            }
            let effective_base = num_traits::pow::pow(base, s as usize);
            terms.push((sign.clone(), effective_base));
            Some(())
        }
    }
}

/// `c * ln(x)^a` (a >= 1 integer, c != 0) with the logarithm's argument
/// tending to +inf; returns (c, a). Recognizes the bare log, its integer
/// powers, a numeric scale, and a negation.
pub(super) fn constant_times_log_power(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<(BigRational, i64)> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (c, a) = constant_times_log_power(ctx, inner, var, approach)?;
            Some((-c, a))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                let (c, a) = constant_times_log_power(ctx, rhs, var, approach)?;
                return Some((scale * c, a));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                let (c, a) = constant_times_log_power(ctx, lhs, var, approach)?;
                return Some((scale * c, a));
            }
            None
        }
        Expr::Pow(base, exponent) if is_unbounded_log(ctx, base, var, approach) => {
            let exp = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if exp.is_integer() && exp.is_positive() {
                Some((rational_one(), exp.to_integer().try_into().ok()?))
            } else {
                None
            }
        }
        _ if is_unbounded_log(ctx, expr, var, approach) => Some((rational_one(), 1)),
        _ => None,
    }
}

pub(super) fn scaled_exp_subpoly_product_limit(
    ctx: &mut Context,
    exp_info: ScaledPolynomialExpTailInfo,
    subpoly_info: ScaledSubpolynomialTailInfo,
) -> Option<ExprId> {
    if exp_info.coeff.is_zero() || subpoly_info.coeff.is_zero() || exp_info.tail == InfSign::Neg {
        return Some(ctx.num(0));
    }

    scale_infinity(ctx, &(exp_info.coeff * subpoly_info.coeff), InfSign::Pos)
}

/// The sign of the dominant term of a sum of rational-base exponentials at
/// `+inf`, but only when a genuinely growing base (`> 1`) dominates with a
/// nonzero coefficient. None when the expression is not such a sum.
pub(super) fn exp_sum_dominant_sign(ctx: &Context, expr: ExprId, var_name: &str) -> Option<i32> {
    use num_traits::{One, Signed, Zero};
    let mut terms: Vec<(BigRational, BigRational)> = Vec::new();
    collect_rational_exp_terms(ctx, expr, var_name, &rational_one(), &mut terms)?;
    let one = BigRational::one();
    let max_base = terms.iter().map(|(_, b)| b.clone()).max()?;
    if max_base <= one {
        return None;
    }
    let dominant: BigRational = terms
        .iter()
        .filter(|(_, b)| *b == max_base)
        .map(|(c, _)| c.clone())
        .sum();
    if dominant.is_zero() {
        return None;
    }
    Some(if dominant.is_positive() { 1 } else { -1 })
}

/// An exponential atom with a var-free base: `exp(u)` (base `e`) or `b^u`.
pub(super) fn constant_exponential_atom(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Exp) => {
            let e = ctx.add(Expr::Constant(Constant::E));
            Some((e, args[0]))
        }
        Expr::Pow(base, exponent) if !depends_on(ctx, base, var) => Some((base, exponent)),
        _ => None,
    }
}

/// `exp(L)` for a resolved inner limit L, folding the cases an exponential
/// indeterminate form produces: e^(+inf)=inf, e^(-inf)=0, e^0=1, e^1=e,
/// e^(ln c)=c, otherwise e^L.
pub(super) fn exp_of_limit_value(ctx: &mut Context, l_limit: ExprId) -> Option<ExprId> {
    use num_traits::{One, Zero};
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            if let Some(value) = crate::numeric_eval::as_rational_const(ctx, l_limit) {
                if value.is_zero() {
                    return Some(ctx.num(1));
                }
                if value.is_one() {
                    return Some(ctx.add(Expr::Constant(Constant::E)));
                }
            }
            // e^(ln c) = c.
            if let Some(arg) = bare_natural_log_argument(ctx, l_limit) {
                return Some(arg);
            }
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, l_limit)))
        }
    }
}
