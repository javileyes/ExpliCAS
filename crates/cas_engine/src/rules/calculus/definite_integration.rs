//! Definite integration over the real domain via the fundamental theorem
//! (block 13 first rung). The mathematical core is the INTERVAL
//! CERTIFICATE: before any substitution, every condition required by the
//! indefinite antiderivative must be certified on [lower, upper] with
//! numeric rational bounds — a pole inside the closed interval makes the
//! integral undefined (divergent for the supported rational families),
//! and anything the certificate cannot decide stays an honest residual.

use super::integration_conditions::IntegrationRequiredConditions;
use super::integration_result_pipeline::integrate_with_result_preservation;
use crate::symbolic_calculus_call_support::DefiniteIntegralCall;
use crate::ImplicitCondition;
use crate::Rewrite;
use cas_ast::{Constant, Context, Expr, ExprId};
use cas_math::limit_types::{Approach, LimitOptions};
use cas_math::numeric_eval::as_rational_const;
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

enum IntervalCertificate {
    Certified,
    /// Every obstruction is a root exactly AT a boundary endpoint of the
    /// sorted interval: the integral may converge as an improper one,
    /// decided by one-sided limits of the antiderivative.
    BoundaryTouch {
        lower: bool,
        upper: bool,
    },
    Undefined,
    Unknown,
}

enum DefiniteBound {
    Finite(Endpoint),
    PosInfinity,
    NegInfinity,
    Symbolic,
}

/// Exact interval endpoint of the form `rational + pi_multiple * pi`,
/// covering both rational bounds and the exam-standard rational multiples
/// of pi. Comparisons are exact whenever the pi parts agree (in
/// particular for two pi-pure values, which is how trig zeros at
/// k*pi/2 compare against pi-multiple bounds); mixed comparisons fall
/// back to the rational pi enclosure and refuse when undecidable.
#[derive(Clone, PartialEq)]
struct Endpoint {
    rational: BigRational,
    pi_multiple: BigRational,
    e_multiple: BigRational,
}

impl Endpoint {
    fn from_rational(value: BigRational) -> Self {
        Endpoint {
            rational: value,
            pi_multiple: BigRational::from_integer(0.into()),
            e_multiple: BigRational::from_integer(0.into()),
        }
    }

    fn from_pi_multiple(multiple: BigRational) -> Self {
        Endpoint {
            rational: BigRational::from_integer(0.into()),
            pi_multiple: multiple,
            e_multiple: BigRational::from_integer(0.into()),
        }
    }

    fn from_e_multiple(multiple: BigRational) -> Self {
        Endpoint {
            rational: BigRational::from_integer(0.into()),
            pi_multiple: BigRational::from_integer(0.into()),
            e_multiple: multiple,
        }
    }

    fn enclosure(&self) -> (BigRational, BigRational) {
        let signed = |multiple: &BigRational, (low, high): (BigRational, BigRational)| {
            if *multiple >= BigRational::from_integer(0.into()) {
                (multiple * low, multiple * high)
            } else {
                (multiple * high, multiple * low)
            }
        };
        let (pi_low, pi_high) = signed(&self.pi_multiple, pi_enclosure());
        let (e_low, e_high) = signed(&self.e_multiple, e_enclosure());
        (
            &self.rational + &pi_low + &e_low,
            &self.rational + &pi_high + &e_high,
        )
    }

    fn try_cmp(&self, other: &Endpoint) -> Option<std::cmp::Ordering> {
        if self.pi_multiple == other.pi_multiple && self.e_multiple == other.e_multiple {
            return Some(self.rational.cmp(&other.rational));
        }
        let (self_low, self_high) = self.enclosure();
        let (other_low, other_high) = other.enclosure();
        if self_high < other_low {
            return Some(std::cmp::Ordering::Less);
        }
        if self_low > other_high {
            return Some(std::cmp::Ordering::Greater);
        }
        None
    }
}

/// Recognize rational multiples of pi: pi, q*pi, pi/n and negations.
fn pi_multiple_of(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Pi) => Some(BigRational::from_integer(1.into())),
        Expr::Neg(inner) => pi_multiple_of(ctx, *inner).map(|value| -value),
        Expr::Mul(l, r) => {
            if let Some(scale) = as_rational_const(ctx, *l) {
                return pi_multiple_of(ctx, *r).map(|value| scale * value);
            }
            if let Some(scale) = as_rational_const(ctx, *r) {
                return pi_multiple_of(ctx, *l).map(|value| scale * value);
            }
            None
        }
        Expr::Div(numerator, denominator) => {
            let divisor = as_rational_const(ctx, *denominator)?;
            if divisor.is_zero() {
                return None;
            }
            pi_multiple_of(ctx, *numerator).map(|value| value / divisor)
        }
        _ => None,
    }
}

/// Recognize rational multiples of e: e, q*e, e/n and negations.
fn e_multiple_of(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Constant(Constant::E) => Some(BigRational::from_integer(1.into())),
        Expr::Neg(inner) => e_multiple_of(ctx, *inner).map(|value| -value),
        Expr::Mul(l, r) => {
            if let Some(scale) = as_rational_const(ctx, *l) {
                return e_multiple_of(ctx, *r).map(|value| scale * value);
            }
            if let Some(scale) = as_rational_const(ctx, *r) {
                return e_multiple_of(ctx, *l).map(|value| scale * value);
            }
            None
        }
        Expr::Div(numerator, denominator) => {
            let divisor = as_rational_const(ctx, *denominator)?;
            if divisor.is_zero() {
                return None;
            }
            e_multiple_of(ctx, *numerator).map(|value| value / divisor)
        }
        _ => None,
    }
}

/// Gaussian moment integrals over a half-line or the full line:
///   int_0^inf x^(2n) e^(-a x^2) dx = (1/2) (2n)!/(4^n n!) sqrt(pi) / a^(n+1/2)
/// for rational a > 0 and even cofactor degree 2n. The full line doubles
/// it (even integrand); (-inf, 0] equals [0, inf). Only these exact
/// infinite-bound patterns resolve - the indefinite integral and any
/// finite-bound or odd-cofactor variant stay residual / honest.
fn gaussian_definite_integral_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    use num_bigint::BigInt;
    use num_traits::{One, Zero};

    // The interval must be a half-line from 0 or the full real line.
    #[derive(Clone, Copy)]
    enum GaussianInterval {
        HalfLine, // [0, inf) or (-inf, 0]
        FullLine, // (-inf, inf)
    }
    let interval = match (lower_bound, upper_bound) {
        (DefiniteBound::Finite(low), DefiniteBound::PosInfinity)
            if low.rational.is_zero() && low.pi_multiple.is_zero() && low.e_multiple.is_zero() =>
        {
            GaussianInterval::HalfLine
        }
        (DefiniteBound::NegInfinity, DefiniteBound::Finite(high))
            if high.rational.is_zero()
                && high.pi_multiple.is_zero()
                && high.e_multiple.is_zero() =>
        {
            GaussianInterval::HalfLine
        }
        (DefiniteBound::NegInfinity, DefiniteBound::PosInfinity) => GaussianInterval::FullLine,
        _ => return None,
    };

    let (a, cofactor_coeff, two_n) = match_gaussian_integrand(ctx, call.target, &call.var_name)?;
    if !a.is_positive() || two_n % 2 != 0 {
        return None;
    }
    let n = (two_n / 2) as u32;

    // coeff = (cofactor leading coeff) * (1/2) * (2n)! / (4^n n!).
    let factorial = |k: u32| -> BigInt {
        let mut acc = BigInt::one();
        for i in 1..=k {
            acc *= BigInt::from(i);
        }
        acc
    };
    let two = BigRational::from_integer(BigInt::from(2));
    let mut coeff =
        BigRational::new(factorial(2 * n), factorial(n) * BigInt::from(4).pow(n)) / &two;
    coeff *= &cofactor_coeff; // the integrand's leading constant factor.
    if matches!(interval, GaussianInterval::FullLine) {
        coeff *= &two; // even integrand over the symmetric interval.
    }

    // value = coeff * sqrt(pi) / a^(n + 1/2) = coeff/a^n * sqrt(pi)/sqrt(a)
    //       = coeff/a^n * sqrt(pi/a).
    let pi = ctx.add(Expr::Constant(Constant::Pi));
    let ratio_const = if a.is_one() {
        pi
    } else {
        let a_expr = ctx.add(Expr::Number(a.clone()));
        ctx.add(Expr::Div(pi, a_expr))
    };
    let half = ctx.add(Expr::Number(BigRational::new(
        BigInt::one(),
        BigInt::from(2),
    )));
    let sqrt_ratio = ctx.add(Expr::Pow(ratio_const, half));

    // Fold the rational a^n into coeff.
    let a_pow_n = {
        let mut acc = BigRational::one();
        for _ in 0..n {
            acc *= &a;
        }
        acc
    };
    let scalar = coeff / a_pow_n;
    let result = if scalar.is_one() {
        sqrt_ratio
    } else {
        let scalar_expr = ctx.add(Expr::Number(scalar));
        ctx.add(Expr::Mul(scalar_expr, sqrt_ratio))
    };
    Some(Rewrite::new(result).desc("Gaussian moment integral (table)"))
}

/// Half-integer Gamma moments `int_0^inf c x^(m-1/2) e^(-a x) dx` with
/// rational `a > 0` and `m >= 0` integer, evaluating to
/// `c (2m)!/(4^m m!) / a^m * sqrt(pi / a)` (since `Gamma(m+1/2) =
/// (2m)!/(4^m m!) sqrt(pi)` and `int_0^inf x^s e^(-a x) = Gamma(s+1)/a^(s+1)`).
/// Examples: `e^(-x)/sqrt(x) = sqrt(pi)`, `sqrt(x) e^(-x) = sqrt(pi)/2`,
/// `e^(-2x)/sqrt(x) = sqrt(pi/2)`.
///
/// Runs after the Gaussian table and before the antiderivative attempt.
/// Gated to `[0, inf)` only, a pure-linear decay exponent, and a HALF-integer
/// total power of the variable, so the integer moments (`x^n e^(-x) = n!`,
/// which DO have an elementary antiderivative) and the indefinite/divergent
/// forms stay residual.
fn gamma_half_integer_definite_integral_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    use num_bigint::BigInt;
    use num_traits::{One, Zero};

    // Only the half-line [0, inf): x^(m-1/2) is not real for x < 0, and the
    // integrand is not even, so the full line / (-inf, 0] do not apply.
    match (lower_bound, upper_bound) {
        (DefiniteBound::Finite(low), DefiniteBound::PosInfinity)
            if low.rational.is_zero() && low.pi_multiple.is_zero() && low.e_multiple.is_zero() => {}
        _ => return None,
    }

    let (a, coeff_lead, m) = match_gamma_integrand(ctx, call.target, &call.var_name)?;
    if !a.is_positive() {
        return None;
    }

    // scalar = c * (2m)! / (4^m m!) / a^m, result = scalar * sqrt(pi / a).
    let factorial = |k: u32| -> BigInt {
        let mut acc = BigInt::one();
        for i in 1..=k {
            acc *= BigInt::from(i);
        }
        acc
    };
    let mut scalar =
        BigRational::new(factorial(2 * m), factorial(m) * BigInt::from(4).pow(m)) * &coeff_lead;
    let mut a_pow_m = BigRational::one();
    for _ in 0..m {
        a_pow_m *= &a;
    }
    scalar /= &a_pow_m;

    let pi = ctx.add(Expr::Constant(Constant::Pi));
    let ratio_const = if a.is_one() {
        pi
    } else {
        let a_expr = ctx.add(Expr::Number(a.clone()));
        ctx.add(Expr::Div(pi, a_expr))
    };
    let half = ctx.add(Expr::Number(BigRational::new(
        BigInt::one(),
        BigInt::from(2),
    )));
    let sqrt_ratio = ctx.add(Expr::Pow(ratio_const, half));

    let result = if scalar.is_one() {
        sqrt_ratio
    } else {
        let scalar_expr = ctx.add(Expr::Number(scalar));
        ctx.add(Expr::Mul(scalar_expr, sqrt_ratio))
    };
    Some(Rewrite::new(result).desc("Half-integer Gamma moment integral (table)"))
}

/// Match `c x^s e^(-a x)` with rational `a > 0` and `s = m - 1/2` a half
/// integer (so `m = s + 1/2` is a non-negative integer). Returns
/// `(a, c, m)`. Accumulates the net power of the variable and the linear
/// decay coefficient across Mul/Div factors; rejects a non-linear exponent,
/// an integer power of the variable, `s < -1/2` (divergent), a foreign
/// variable, or any factor that is neither a variable power nor `e^(linear)`.
fn match_gamma_integrand(
    ctx: &Context,
    integrand: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational, u32)> {
    use num_traits::{One, Zero};
    let mut x_exponent = BigRational::zero();
    let mut decay = BigRational::zero();
    let mut saw_exp = false;
    let mut constant = BigRational::one();
    collect_gamma_factors(
        ctx,
        integrand,
        var,
        true,
        &mut x_exponent,
        &mut decay,
        &mut saw_exp,
        &mut constant,
    )?;
    if !saw_exp {
        return None;
    }
    // integrand ~ e^(decay x); convergence on [0, inf) needs decay < 0.
    let a = -decay;
    if !a.is_positive() {
        return None;
    }
    // s = x_exponent must be a HALF-integer with m = s + 1/2 a non-negative
    // integer (s >= -1/2). Integer s is left to the elementary antiderivative.
    let m_rational = &x_exponent + BigRational::new(num_bigint::BigInt::from(1), 2.into());
    if !m_rational.is_integer() || m_rational.is_negative() {
        return None;
    }
    let m = m_rational.to_integer().try_into().ok()?;
    Some((a, constant, m))
}

/// Walk a Mul/Div chain, accumulating the net power of `var`, the linear
/// decay coefficient of any `e^(linear)` factor, and the var-free constant.
/// `positive` tracks whether the current factor sits in a numerator (Div
/// inverts it). Returns None on any unclassifiable factor.
#[allow(clippy::too_many_arguments)]
fn collect_gamma_factors(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    positive: bool,
    x_exponent: &mut BigRational,
    decay: &mut BigRational,
    saw_exp: &mut bool,
    constant: &mut BigRational,
) -> Option<()> {
    match ctx.get(expr).clone() {
        Expr::Mul(l, r) => {
            collect_gamma_factors(ctx, l, var, positive, x_exponent, decay, saw_exp, constant)?;
            collect_gamma_factors(ctx, r, var, positive, x_exponent, decay, saw_exp, constant)
        }
        Expr::Div(num, den) => {
            collect_gamma_factors(
                ctx, num, var, positive, x_exponent, decay, saw_exp, constant,
            )?;
            collect_gamma_factors(
                ctx, den, var, !positive, x_exponent, decay, saw_exp, constant,
            )
        }
        // A unit-magnitude negative coefficient normalizes to a top-level
        // Neg wrapper (e.g. -e^(-x)/sqrt(x) = Neg(...)); -1 is its own
        // reciprocal, so it negates the constant whether in num or den.
        Expr::Neg(inner) => {
            *constant = -std::mem::take(constant);
            collect_gamma_factors(
                ctx, inner, var, positive, x_exponent, decay, saw_exp, constant,
            )
        }
        _ => collect_gamma_leaf(
            ctx, expr, var, positive, x_exponent, decay, saw_exp, constant,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_gamma_leaf(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    positive: bool,
    x_exponent: &mut BigRational,
    decay: &mut BigRational,
    saw_exp: &mut bool,
    constant: &mut BigRational,
) -> Option<()> {
    use num_traits::{One, Zero};
    let sign = if positive {
        BigRational::one()
    } else {
        -BigRational::one()
    };
    // e^(c x): a pure-linear exponent contributes c to the decay.
    if let Expr::Pow(base, exponent) = ctx.get(expr) {
        if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
            let poly = Polynomial::from_expr(ctx, *exponent, var).ok()?;
            if poly.degree() != 1 {
                return None;
            }
            // Pure linear: no constant term (e^(c x + d) carries a transcendental e^d).
            if !poly.coeffs.first().map(|c| c.is_zero()).unwrap_or(true) {
                return None;
            }
            let c = poly.coeffs.get(1)?.clone();
            *decay += &sign * c;
            *saw_exp = true;
            return Some(());
        }
        // x^k: rational power of the variable.
        if is_var_named(ctx, *base, var) {
            let k = as_rational_const(ctx, *exponent)?;
            *x_exponent += &sign * k;
            return Some(());
        }
    }
    // sqrt(x) = x^(1/2).
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1
            && matches!(ctx.builtin_of(*fn_id), Some(cas_ast::BuiltinFn::Sqrt))
            && is_var_named(ctx, args[0], var)
        {
            *x_exponent += &sign * BigRational::new(num_bigint::BigInt::from(1), 2.into());
            return Some(());
        }
    }
    // Bare variable: x^1.
    if is_var_named(ctx, expr, var) {
        *x_exponent += &sign;
        return Some(());
    }
    // A var-free rational constant factor scales the result.
    if let Some(value) = as_rational_const(ctx, expr) {
        if value.is_zero() {
            return None;
        }
        if positive {
            *constant *= value;
        } else {
            *constant /= value;
        }
        return Some(());
    }
    None
}

fn is_var_named(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym) if ctx.sym_name(*sym) == var)
}

/// Match `x^(2n) e^(-a x^2)` with rational a > 0, in either engine form:
///   `x^(2n) / e^(a x^2)`  (Div, e^(positive a x^2) in the denominator)
///   `x^(2n) e^(-a x^2)`   (Mul, e^(negative coefficient x^2))
/// Returns (a, cofactor_leading_coeff, 2n). Refuses non-pure-quadratic
/// exponents (any linear or constant term), odd cofactor degrees, and
/// foreign variable factors.
fn match_gaussian_integrand(
    ctx: &Context,
    integrand: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational, i64)> {
    use num_traits::Zero;
    if let Expr::Div(num, den) = ctx.get(integrand) {
        let (num, den) = (*num, *den);
        // integrand = num / e^(c x^2) = num e^(-c x^2), so the decay rate
        // a (integrand ~ e^(-a x^2)) equals the denominator coefficient c.
        let a = gaussian_exp_coefficient(ctx, den, var)?;
        if !a.is_positive() {
            return None;
        }
        let (coeff, two_n) = monomial_even_degree(ctx, std::slice::from_ref(&num), var)?;
        return Some((a, coeff, two_n));
    }
    // Mul form (or a bare exponential): one e^(c x^2) factor with c < 0.
    let factors = cas_math::expr_nary::mul_factors(ctx, integrand);
    let mut cofactor = Vec::new();
    let mut coefficient: Option<BigRational> = None;
    for f in &factors {
        if let Some(c) = gaussian_exp_coefficient(ctx, *f, var) {
            if coefficient.is_some() {
                return None;
            }
            coefficient = Some(c);
        } else {
            cofactor.push(*f);
        }
    }
    let c = coefficient?;
    let a = -c;
    if !a.is_positive() || a.is_zero() {
        return None;
    }
    let (coeff, two_n) = monomial_even_degree(ctx, &cofactor, var)?;
    Some((a, coeff, two_n))
}

/// Coefficient c when `factor == e^(c * var^2)` (pure quadratic exponent,
/// no linear or constant term); None otherwise.
fn gaussian_exp_coefficient(ctx: &Context, factor: ExprId, var: &str) -> Option<BigRational> {
    let Expr::Pow(base, exponent) = ctx.get(factor) else {
        return None;
    };
    if !matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
        return None;
    }
    let exponent = *exponent;
    let poly = Polynomial::from_expr(ctx, exponent, var).ok()?;
    if poly.degree() != 2 {
        return None;
    }
    // Pure quadratic: no x^1 or x^0 term.
    if !poly.coeffs.first().map(|c| c.is_zero()).unwrap_or(true)
        || !poly.coeffs.get(1).map(|c| c.is_zero()).unwrap_or(true)
    {
        return None;
    }
    poly.coeffs.get(2).cloned()
}

/// The product of `factors` must be `c * var^(2n)` with an EVEN
/// nonnegative degree; returns (c, 2n). A bare constant gives (c, 0).
fn monomial_even_degree(
    ctx: &Context,
    factors: &[ExprId],
    var: &str,
) -> Option<(BigRational, i64)> {
    use num_traits::Zero;
    let mut product = Polynomial::one(var.to_string());
    for &f in factors {
        let poly = Polynomial::from_expr(ctx, f, var).ok()?;
        product = product.mul(&poly);
    }
    // Must be a single monomial term c * x^(2n): all lower coeffs zero.
    let degree = product.degree();
    if !degree.is_multiple_of(2) {
        return None;
    }
    let leading = product.coeffs.get(degree)?.clone();
    if leading.is_zero() {
        return None;
    }
    if product
        .coeffs
        .iter()
        .take(degree)
        .any(|coeff| !coeff.is_zero())
    {
        return None;
    }
    Some((leading, degree as i64))
}

/// Flatten a multiplicative chain into leaf factors.
fn classify_bound(ctx: &Context, bound: ExprId) -> DefiniteBound {
    if let Some(value) = as_rational_const(ctx, bound) {
        return DefiniteBound::Finite(Endpoint::from_rational(value));
    }
    if let Some(multiple) = pi_multiple_of(ctx, bound) {
        return DefiniteBound::Finite(Endpoint::from_pi_multiple(multiple));
    }
    if let Some(multiple) = e_multiple_of(ctx, bound) {
        return DefiniteBound::Finite(Endpoint::from_e_multiple(multiple));
    }
    match ctx.get(bound) {
        Expr::Constant(Constant::Infinity) => DefiniteBound::PosInfinity,
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            DefiniteBound::NegInfinity
        }
        _ => DefiniteBound::Symbolic,
    }
}

/// A rational point strictly inside the (possibly pi-valued) interval,
/// via the enclosures; None when the enclosures overlap.
fn interval_rational_probe(low: &Endpoint, high: &Endpoint) -> Option<BigRational> {
    let (_, low_high) = low.enclosure();
    let (high_low, _) = high.enclosure();
    if low_high < high_low {
        Some((low_high + high_low) / BigRational::from_integer(2.into()))
    } else {
        None
    }
}

enum RootPosition {
    Outside,
    AtLower,
    AtUpper,
    Inside,
}

/// Position of a rational root relative to the closed interval; None
/// when undecidable.
fn root_position(low: &Endpoint, high: &Endpoint, value: &BigRational) -> Option<RootPosition> {
    let point = Endpoint::from_rational(value.clone());
    match point.try_cmp(low)? {
        std::cmp::Ordering::Less => return Some(RootPosition::Outside),
        std::cmp::Ordering::Equal => return Some(RootPosition::AtLower),
        std::cmp::Ordering::Greater => {}
    }
    match point.try_cmp(high)? {
        std::cmp::Ordering::Greater => Some(RootPosition::Outside),
        std::cmp::Ordering::Equal => Some(RootPosition::AtUpper),
        std::cmp::Ordering::Less => Some(RootPosition::Inside),
    }
}

/// Parity of an expression in the integration variable. A sound, conservative
/// classifier: every arm is a parity identity and anything undecidable is None.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VarParity {
    Odd,
    Even,
}

/// Parity behaviour of a single-argument builtin as the OUTER function.
enum OuterParity {
    /// f(-y) = -f(y): f(g) inherits the parity of g.
    Odd,
    /// f(-y) = f(y): f(g) is even for any g of defined parity.
    Even,
    /// Neither, but f(g) is even when g is even (f(g(-x)) = f(g(x))); an odd
    /// inner argument yields no usable parity.
    EvenWhenInnerEven,
}

fn builtin_outer_parity(builtin: cas_ast::BuiltinFn) -> Option<OuterParity> {
    use cas_ast::BuiltinFn::{
        Abs, Arcsin, Arctan, Asin, Asinh, Atan, Atanh, Cbrt, Cos, Cosh, Cot, Csc, Exp, Ln, Log,
        Log10, Log2, Sec, Sin, Sinh, Sqrt, Tan, Tanh,
    };
    Some(match builtin {
        Sin | Tan | Csc | Cot | Sinh | Tanh | Asin | Arcsin | Atan | Arctan | Asinh | Atanh
        | Cbrt => OuterParity::Odd,
        Cos | Sec | Cosh | Abs => OuterParity::Even,
        Exp | Ln | Log | Log2 | Log10 | Sqrt => OuterParity::EvenWhenInnerEven,
        _ => return None,
    })
}

/// Parity of `expr` in `var_name`, or None when undecidable. Sound: a foreign
/// symbol is a constant in the integration variable (even); a sum keeps a
/// parity only when both terms share it; a product/quotient adds parities
/// (odd*odd = even); an integer power carries the base's parity by exponent
/// parity; and a composition follows the outer function's parity class.
fn parity_in_var(ctx: &Context, expr: ExprId, var_name: &str) -> Option<VarParity> {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) => Some(VarParity::Even),
        Expr::Variable(sym) => Some(if ctx.sym_name(*sym) == var_name {
            VarParity::Odd
        } else {
            VarParity::Even
        }),
        Expr::Neg(inner) => parity_in_var(ctx, *inner, var_name),
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            let pa = parity_in_var(ctx, *a, var_name)?;
            let pb = parity_in_var(ctx, *b, var_name)?;
            (pa == pb).then_some(pa)
        }
        Expr::Mul(a, b) | Expr::Div(a, b) => {
            let pa = parity_in_var(ctx, *a, var_name)?;
            let pb = parity_in_var(ctx, *b, var_name)?;
            Some(if pa == pb {
                VarParity::Even
            } else {
                VarParity::Odd
            })
        }
        Expr::Pow(base, exp) => {
            use num_integer::Integer;
            // A positive x-free base is an exponential b^g = e^{g ln b}: even
            // exactly when the exponent g is even (e^(x^2)), undecidable when g
            // is odd. Covers e^g and rational bases like 2^(x^2).
            let base_is_positive_constant = match as_rational_const(ctx, *base) {
                Some(value) => value.is_positive(),
                None => matches!(ctx.get(*base), Expr::Constant(Constant::E)),
            };
            if base_is_positive_constant {
                let exp_parity = parity_in_var(ctx, *exp, var_name)?;
                return (exp_parity == VarParity::Even).then_some(VarParity::Even);
            }
            // Otherwise the base carries the variable: an x-free integer
            // exponent carries the base's parity by the exponent's parity.
            let exponent = as_rational_const(ctx, *exp)?;
            if !exponent.is_integer() {
                return None;
            }
            let base_parity = parity_in_var(ctx, *base, var_name)?;
            Some(match base_parity {
                VarParity::Even => VarParity::Even,
                VarParity::Odd if exponent.to_integer().is_even() => VarParity::Even,
                VarParity::Odd => VarParity::Odd,
            })
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let inner = parity_in_var(ctx, args[0], var_name)?;
            match builtin_outer_parity(ctx.builtin_of(*fn_id)?)? {
                OuterParity::Odd => Some(inner),
                OuterParity::Even => Some(VarParity::Even),
                OuterParity::EvenWhenInnerEven => {
                    (inner == VarParity::Even).then_some(VarParity::Even)
                }
            }
        }
        _ => None,
    }
}

/// Structural fallback when no antiderivative exists: an ODD integrand over a
/// symmetric interval [-a, a] integrates to 0 with no antiderivative needed.
/// Soundness rests on three independent obligations — symmetric finite bounds,
/// provable oddness in the variable, and integrability (no interior
/// singularity, certified by the same scan that makes int(1/x, x, -1, 1)
/// undefined). The orientation of the bounds does not matter: a reversed
/// interval only flips the sign of 0.
fn odd_symmetric_definite_integral_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    let (DefiniteBound::Finite(low), DefiniteBound::Finite(high)) = (lower_bound, upper_bound)
    else {
        return None;
    };
    // Symmetric interval: lower = -upper component by component.
    let symmetric = (&low.rational + &high.rational).is_zero()
        && (&low.pi_multiple + &high.pi_multiple).is_zero()
        && (&low.e_multiple + &high.e_multiple).is_zero();
    if !symmetric {
        return None;
    }
    if parity_in_var(ctx, call.target, &call.var_name) != Some(VarParity::Odd) {
        return None;
    }
    // Sort the endpoints for the certificate's closed interval, then require
    // full continuity (Certified) - a boundary touch or an interior pole is
    // not integrable-to-zero by symmetry.
    let (interval_low, interval_high) = match low.try_cmp(high) {
        Some(std::cmp::Ordering::Greater) => (high, low),
        Some(_) => (low, high),
        None => return None,
    };
    if !matches!(
        integrand_risks_certified(
            ctx,
            call.target,
            &call.var_name,
            interval_low,
            interval_high
        ),
        IntervalCertificate::Certified
    ) {
        return None;
    }
    let zero = ctx.num(0);
    Some(Rewrite::new(zero).desc("odd integrand over symmetric interval [-a, a] = 0"))
}

/// `integral_a^b |c x + d| dx` over finite RATIONAL bounds, split at the root
/// `r = -d/c` of the linear inner. The antiderivative of `c x + d` is
/// `G(x) = c x^2/2 + d x`; on each side of `r` the inner has constant sign, so
/// the integral is `|G(r) - G(lo)| + |G(hi) - G(r)|` when `r` lies strictly
/// inside the interval and `|G(hi) - G(lo)|` otherwise. Pure rational arithmetic,
/// no antiderivative search. The integrand must be exactly `|linear|` (`x*|x|`
/// and other products are left to the odd-symmetry / FTC owners). Resolves
/// `integral|x| [-1,1] = 1`, `integral|x-1| [0,2] = 1`, `integral|2x-1| [0,1] = 1/2`.
fn abs_linear_definite_integral_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    let (DefiniteBound::Finite(low), DefiniteBound::Finite(high)) = (lower_bound, upper_bound)
    else {
        return None;
    };
    // Pure rational bounds only (a pi/e component is out of scope).
    let pure_rational = |endpoint: &Endpoint| -> Option<BigRational> {
        (endpoint.pi_multiple.is_zero() && endpoint.e_multiple.is_zero())
            .then(|| endpoint.rational.clone())
    };
    let lo = pure_rational(low)?;
    let hi = pure_rational(high)?;

    // The integrand must be exactly |g(x)| with g a nonzero-slope linear poly.
    let Expr::Function(fn_id, args) = ctx.get(call.target).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(cas_ast::BuiltinFn::Abs) {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, args[0], &call.var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let slope = poly.coeffs.get(1)?.clone();
    let intercept = poly.coeffs.first()?.clone();
    if slope.is_zero() {
        return None;
    }

    let two = BigRational::from_integer(2.into());
    let antiderivative = |x: &BigRational| -> BigRational {
        let x_sq = x * x;
        &slope * &x_sq / &two + &intercept * x
    };
    let root = -&intercept / &slope;

    // Sort to a positive-orientation interval; restore the sign at the end.
    let (left, right, sign) = if lo <= hi {
        (lo.clone(), hi.clone(), BigRational::from_integer(1.into()))
    } else {
        (
            hi.clone(),
            lo.clone(),
            BigRational::from_integer((-1).into()),
        )
    };
    let positive = if left < root && root < right {
        (antiderivative(&root) - antiderivative(&left)).abs()
            + (antiderivative(&right) - antiderivative(&root)).abs()
    } else {
        (antiderivative(&right) - antiderivative(&left)).abs()
    };
    let value = ctx.add(Expr::Number(sign * positive));
    Some(Rewrite::new(value).desc("integral of |linear| split at its root"))
}

pub(super) fn definite_integration_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
) -> Option<Rewrite> {
    let lower_bound = classify_bound(ctx, call.lower);
    let upper_bound = classify_bound(ctx, call.upper);
    if let (DefiniteBound::Finite(lower), DefiniteBound::Finite(upper)) =
        (&lower_bound, &upper_bound)
    {
        if lower == upper {
            let zero = ctx.num(0);
            return Some(Rewrite::new(zero).desc("integrate(f, x, a, a) = 0"));
        }
    }

    // Gaussian table for the famous non-elementary improper integrals,
    // BEFORE the antiderivative attempt (which has no closed form). The
    // INDEFINITE integral stays residual; only these exact infinite-bound
    // patterns resolve.
    if let Some(rewrite) = gaussian_definite_integral_rewrite(ctx, call, &lower_bound, &upper_bound)
    {
        return Some(rewrite);
    }

    // Gamma table for the half-integer moments int_0^inf x^(m-1/2) e^(-a x),
    // whose antiderivative is non-elementary (Gamma function). Same gating
    // discipline: only [0, inf) with a pure-linear decay; the indefinite
    // form and integer/divergent powers stay residual.
    if let Some(rewrite) =
        gamma_half_integer_definite_integral_rewrite(ctx, call, &lower_bound, &upper_bound)
    {
        return Some(rewrite);
    }

    // |linear| has no single elementary antiderivative, so split it at the root
    // before the FTC attempt (the absolute value is continuous, no certificate
    // needed); a product like x*|x| is not a bare |linear| and is left to the
    // odd-symmetry / FTC owners.
    if let Some(rewrite) =
        abs_linear_definite_integral_rewrite(ctx, call, &lower_bound, &upper_bound)
    {
        return Some(rewrite);
    }

    let Some((antiderivative, conditions)) = resolve_indefinite_for_definite(ctx, call) else {
        // No elementary antiderivative: try the structural symmetry fallback
        // before conceding a residual. This keeps the FTC path's ownership of
        // everything that DOES have an antiderivative (x^3, tan x, ...) intact.
        return odd_symmetric_definite_integral_rewrite(ctx, call, &lower_bound, &upper_bound);
    };

    if matches!(
        lower_bound,
        DefiniteBound::PosInfinity | DefiniteBound::NegInfinity
    ) || matches!(
        upper_bound,
        DefiniteBound::PosInfinity | DefiniteBound::NegInfinity
    ) {
        return improper_integration_rewrite(
            ctx,
            call,
            antiderivative,
            conditions,
            &lower_bound,
            &upper_bound,
        );
    }

    let (DefiniteBound::Finite(lower), DefiniteBound::Finite(upper)) = (lower_bound, upper_bound)
    else {
        // Symbolic bounds: sound exactly when the antiderivative is
        // unconditional - every condition-free antiderivative the engine
        // emits is continuous on all of R, so there is no interval to
        // certify (the curriculum "area function" integrate(f, x, a, t)).
        if !conditions.is_empty() {
            return None;
        }
        let mut antiderivative = antiderivative;
        loop {
            let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
            if unwrapped == antiderivative {
                break;
            }
            antiderivative = unwrapped;
        }
        let at_upper =
            cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.upper);
        let at_lower =
            cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.lower);
        // A genuinely non-finite boundary value means the improper integral
        // DIVERGES at that endpoint (e.g. int_0^x ln(t)/t dt: the antiderivative
        // (ln t)^2/2 -> +inf at t = 0). Returning F(upper) - F(lower) would carry
        // a silent infinity^k term that a later `diff` drops into a false finite
        // derivative. The check is structural and domain-aware: ln(0), c/0 and
        // 0^neg are non-finite, but a literal-zero factor kills a removable
        // singularity (0*ln(0) -> 0, so int_0^x ln(t) dt -> x ln(x) - x is kept)
        // and ln(0^2+1) = ln(1) is finite (so int_0^x arctan(t) dt is kept).
        if boundary_is_genuinely_nonfinite(ctx, at_upper)
            || boundary_is_genuinely_nonfinite(ctx, at_lower)
        {
            let undefined = ctx.add(Expr::Constant(Constant::Undefined));
            return Some(
                Rewrite::new(undefined).desc("integrate(f, x, a, t) diverges at an endpoint"),
            );
        }
        let result = ctx.add(Expr::Sub(at_upper, at_lower));
        return Some(Rewrite::new(result).desc("integrate(f, x, a, b)"));
    };
    // F(upper) - F(lower) is already orientation-aware; the swap is only
    // for the certificate's closed interval.
    let (interval_low, interval_high) = match lower.try_cmp(&upper) {
        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => (lower, upper),
        Some(std::cmp::Ordering::Greater) => (upper, lower),
        None => return None,
    };
    let mut antiderivative = antiderivative;
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
        if unwrapped == antiderivative {
            break;
        }
        antiderivative = unwrapped;
    }

    match combine_certificates(
        certify_interval(
            ctx,
            &conditions,
            &call.var_name,
            &interval_low,
            &interval_high,
        ),
        integrand_risks_certified(
            ctx,
            call.target,
            &call.var_name,
            &interval_low,
            &interval_high,
        ),
    ) {
        IntervalCertificate::Certified => {}
        IntervalCertificate::BoundaryTouch {
            lower: touch_low,
            upper: touch_high,
        } => {
            return boundary_touch_evaluation(
                ctx,
                call,
                antiderivative,
                &interval_low,
                &interval_high,
                touch_low,
                touch_high,
            );
        }
        IntervalCertificate::Undefined => {
            let undefined = ctx.add(Expr::Constant(Constant::Undefined));
            return Some(
                Rewrite::new(undefined)
                    .desc("integrate(f, x, a, b) diverges: pole inside the interval"),
            );
        }
        IntervalCertificate::Unknown => return None,
    }

    let at_upper = cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.upper);
    let at_lower = cas_ast::substitute_expr_by_id(ctx, antiderivative, call.var_expr, call.lower);
    let result = ctx.add(Expr::Sub(at_upper, at_lower));
    Some(Rewrite::new(result).desc("integrate(f, x, a, b)"))
}

/// Boundary-touched endpoints: the obstruction sits exactly at an
/// endpoint of the sorted interval, so the boundary value is the
/// ONE-SIDED LIMIT of the antiderivative approaching from inside the
/// interval (curriculum improper integrals like the unit-interval
/// natural-log integral evaluating to -1). Finite limits converge;
/// signed infinities report honest
/// divergence; unresolved limits stay residual.
#[allow(clippy::too_many_arguments)]
fn boundary_touch_evaluation(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    antiderivative: ExprId,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
    touch_low: bool,
    touch_high: bool,
) -> Option<Rewrite> {
    let bound_value = |ctx: &mut Context, bound_expr: ExprId| -> Option<ExprId> {
        let endpoint = match classify_bound(ctx, bound_expr) {
            DefiniteBound::Finite(endpoint) => endpoint,
            _ => return None,
        };
        let touched =
            (touch_low && endpoint == *interval_low) || (touch_high && endpoint == *interval_high);
        if !touched {
            return Some(cas_ast::substitute_expr_by_id(
                ctx,
                antiderivative,
                call.var_expr,
                bound_expr,
            ));
        }
        // Approach from inside the sorted interval.
        let side = if endpoint == *interval_low {
            cas_math::limit_types::FiniteLimitSide::Right
        } else {
            cas_math::limit_types::FiniteLimitSide::Left
        };
        let opts = LimitOptions::default();
        let mut budget = crate::budget::Budget::preset_cli();
        let outcome = crate::limits::limit(
            ctx,
            antiderivative,
            call.var_expr,
            Approach::FiniteOneSided(bound_expr, side),
            &opts,
            &mut budget,
        )
        .ok()?;
        if outcome.warning.is_some() || expr_contains_limit_call(ctx, outcome.expr) {
            return None;
        }
        if matches!(ctx.get(outcome.expr), Expr::Constant(Constant::Undefined)) {
            return None;
        }
        Some(outcome.expr)
    };

    let upper_value = bound_value(ctx, call.upper)?;
    let lower_value = bound_value(ctx, call.lower)?;

    let upper_sign = infinite_sign(ctx, upper_value);
    let lower_sign = infinite_sign(ctx, lower_value);
    let result = match (upper_sign, lower_sign) {
        (Some(_), Some(_)) => return None, // infinity - infinity: indeterminate
        (Some(sign), None) => build_signed_infinity(ctx, sign),
        (None, Some(sign)) => build_signed_infinity(ctx, -sign),
        (None, None) => ctx.add(Expr::Sub(upper_value, lower_value)),
    };
    Some(Rewrite::new(result).desc("integrate(f, x, a, b)"))
}

/// Improper integrals: at an infinite bound the boundary value is the
/// LIMIT of the antiderivative (never a substitution - the symbolic path
/// used to leak the infinity constant into F, producing forms like
/// arctan(infinity)). Divergence to +-infinity is reported as the honest
/// infinite value; indeterminate combinations stay residual.
fn improper_integration_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    antiderivative: ExprId,
    conditions: Vec<ImplicitCondition>,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    match combine_certificates(
        certify_unbounded_interval(ctx, &conditions, &call.var_name, lower_bound, upper_bound),
        integrand_risks_certified_unbounded(
            ctx,
            call.target,
            &call.var_name,
            lower_bound,
            upper_bound,
        ),
    ) {
        IntervalCertificate::Certified => {}
        // Touches at the finite endpoint of an unbounded interval need
        // mixed one-sided/at-infinity evaluation: next rung, residual.
        IntervalCertificate::BoundaryTouch { .. } => return None,
        IntervalCertificate::Undefined => {
            let undefined = ctx.add(Expr::Constant(Constant::Undefined));
            return Some(
                Rewrite::new(undefined)
                    .desc("integrate(f, x, a, b) diverges: pole inside the interval"),
            );
        }
        IntervalCertificate::Unknown => return None,
    }

    let mut antiderivative = antiderivative;
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
        if unwrapped == antiderivative {
            break;
        }
        antiderivative = unwrapped;
    }

    let upper_value = boundary_value(ctx, antiderivative, call.var_expr, call.upper, upper_bound)?;
    let lower_value = boundary_value(ctx, antiderivative, call.var_expr, call.lower, lower_bound)?;

    let upper_sign = infinite_sign(ctx, upper_value);
    let lower_sign = infinite_sign(ctx, lower_value);
    let result = match (upper_sign, lower_sign) {
        (Some(_), Some(_)) => return None, // infinity - infinity: indeterminate
        (Some(sign), None) => build_signed_infinity(ctx, sign),
        (None, Some(sign)) => build_signed_infinity(ctx, -sign),
        (None, None) => ctx.add(Expr::Sub(upper_value, lower_value)),
    };
    Some(Rewrite::new(result).desc("integrate(f, x, a, b)"))
}

/// Boundary value of the antiderivative: substitution at finite or
/// symbolic-free bounds, the limits engine at infinite ones. None when
/// the limit is unresolved or unsafe (honest residual).
fn boundary_value(
    ctx: &mut Context,
    antiderivative: ExprId,
    var_expr: ExprId,
    bound_expr: ExprId,
    bound: &DefiniteBound,
) -> Option<ExprId> {
    let approach = match bound {
        DefiniteBound::Finite(_) => {
            return Some(cas_ast::substitute_expr_by_id(
                ctx,
                antiderivative,
                var_expr,
                bound_expr,
            ));
        }
        DefiniteBound::PosInfinity => Approach::PosInfinity,
        DefiniteBound::NegInfinity => Approach::NegInfinity,
        DefiniteBound::Symbolic => return None,
    };
    let opts = LimitOptions::default();
    let mut budget = crate::budget::Budget::preset_cli();
    let outcome =
        crate::limits::limit(ctx, antiderivative, var_expr, approach, &opts, &mut budget).ok()?;
    if outcome.warning.is_some() || expr_contains_limit_call(ctx, outcome.expr) {
        return None;
    }
    if matches!(ctx.get(outcome.expr), Expr::Constant(Constant::Undefined)) {
        return None;
    }
    Some(outcome.expr)
}

fn expr_contains_limit_call(ctx: &mut Context, expr: ExprId) -> bool {
    let limit_symbol = ctx.intern_symbol("limit");
    expr_contains_call_to(ctx, expr, limit_symbol)
}

fn expr_contains_call_to(ctx: &Context, expr: ExprId, target: cas_ast::symbol::SymbolId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            *fn_id == target
                || args
                    .iter()
                    .any(|arg| expr_contains_call_to(ctx, *arg, target))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            expr_contains_call_to(ctx, *l, target) || expr_contains_call_to(ctx, *r, target)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_call_to(ctx, *inner, target),
        _ => false,
    }
}

fn infinite_sign(ctx: &Context, expr: ExprId) -> Option<i32> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity) => Some(1),
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            Some(-1)
        }
        _ => None,
    }
}

fn build_signed_infinity(ctx: &mut Context, sign: i32) -> ExprId {
    let infinity = ctx.add(Expr::Constant(Constant::Infinity));
    if sign >= 0 {
        infinity
    } else {
        ctx.add(Expr::Neg(infinity))
    }
}

/// Certificate over a (half-)infinite interval: linear poles must lie
/// strictly outside the unbounded closed interval.
fn certify_unbounded_interval(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    let mut outcome = IntervalCertificate::Certified;
    for condition in conditions {
        match condition {
            ImplicitCondition::NonZero(expr) => {
                // cos/sin have zeros in every unbounded interval.
                if trig_condition_target(ctx, *expr, var_name) {
                    return IntervalCertificate::Undefined;
                }
                match nonzero_on_unbounded_interval(ctx, *expr, var_name, lower_bound, upper_bound)
                {
                    IntervalCertificate::Undefined => return IntervalCertificate::Undefined,
                    other => outcome = combine_certificates(outcome, other),
                }
            }
            ImplicitCondition::Positive(expr) | ImplicitCondition::NonNegative(expr) => {
                match globally_positive(ctx, *expr, var_name) {
                    true => {}
                    false => outcome = IntervalCertificate::Unknown,
                }
            }
            _ => outcome = IntervalCertificate::Unknown,
        }
    }
    outcome
}

fn trig_condition_target(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(cas_ast::BuiltinFn::Cos | cas_ast::BuiltinFn::Sin)
            ) && matches!(ctx.get(args[0]),
                Expr::Variable(sym) if ctx.sym_name(*sym) == var_name)
        }
        _ => false,
    }
}

/// Positivity on all of R: variable-free positive numerics, or quadratics
/// with negative discriminant and positive leading coefficient.
fn globally_positive(ctx: &mut Context, expr: ExprId, var_name: &str) -> bool {
    if let Some(value) = as_rational_const(ctx, expr) {
        return value.is_positive();
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return false;
    };
    if poly.degree() != 2 {
        return false;
    }
    let a = poly.coeffs[2].clone();
    let b = poly.coeffs[1].clone();
    let c = poly.coeffs[0].clone();
    let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
    discriminant.is_negative() && a.is_positive()
}

fn nonzero_on_unbounded_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    if let Some(value) = as_rational_const(ctx, expr) {
        return if value.is_zero() {
            IntervalCertificate::Undefined
        } else {
            IntervalCertificate::Certified
        };
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return IntervalCertificate::Unknown;
    };
    match poly.degree() {
        0 => {
            if poly.coeffs.first().is_none_or(Zero::is_zero) {
                IntervalCertificate::Undefined
            } else {
                IntervalCertificate::Certified
            }
        }
        1 => {
            let root = -&poly.coeffs[0] / &poly.coeffs[1];
            let point = Endpoint::from_rational(root);
            let above_lower = match lower_bound {
                DefiniteBound::NegInfinity => true,
                DefiniteBound::Finite(lower) => match point.try_cmp(lower) {
                    Some(std::cmp::Ordering::Less) => false,
                    Some(_) => true,
                    None => return IntervalCertificate::Unknown,
                },
                _ => return IntervalCertificate::Unknown,
            };
            let below_upper = match upper_bound {
                DefiniteBound::PosInfinity => true,
                DefiniteBound::Finite(upper) => match point.try_cmp(upper) {
                    Some(std::cmp::Ordering::Greater) => false,
                    Some(_) => true,
                    None => return IntervalCertificate::Unknown,
                },
                _ => return IntervalCertificate::Unknown,
            };
            if above_lower && below_upper {
                IntervalCertificate::Undefined
            } else {
                IntervalCertificate::Certified
            }
        }
        2 => {
            let a = poly.coeffs[2].clone();
            let b = poly.coeffs[1].clone();
            let c = poly.coeffs[0].clone();
            let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
            if discriminant.is_negative() {
                IntervalCertificate::Certified
            } else {
                IntervalCertificate::Unknown
            }
        }
        _ => IntervalCertificate::Unknown,
    }
}

/// Decide every required condition on the closed interval. Conservative by
/// construction: only conditions provably independent of the interval (or
/// provably violated inside it) are decided; everything else is Unknown
/// and the call stays residual.
/// Resolve the indefinite antiderivative for a definite call, mirroring
/// the indefinite rule's route order: the derivative-cofactor route
/// first (it owns u'/sqrt(u)-style shapes the standard pipeline
/// declines), then the standard pipeline.
fn resolve_indefinite_for_definite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
) -> Option<(ExprId, Vec<ImplicitCondition>)> {
    if let Some((result, condition)) =
        super::integration_derivative_cofactor_routes::polynomial_trig_reciprocal_derivative_root_gate_route(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        return Some((result, vec![condition]));
    }
    let mut required_conditions =
        IntegrationRequiredConditions::from_target(ctx, call.target, &call.var_name);
    let antiderivative = integrate_with_result_preservation(
        ctx,
        call.target,
        &call.var_name,
        &mut required_conditions,
    )?;
    Some((
        antiderivative,
        required_conditions.into_implicit_conditions().collect(),
    ))
}

/// True if `expr` (a substituted boundary value F(endpoint)) is genuinely
/// non-finite: an explicit `infinity`/`undefined`, `ln` of a non-positive
/// constant, a division by zero with a non-zero numerator, or `0^(negative)`.
///
/// Domain-aware so it does NOT over-flag removable singularities: a literal-zero
/// factor kills a singular cofactor (`0*ln(0)` is finite, keeping the convergent
/// improper `int_0^x ln(t) dt`), and `ln(0^2+1) = ln(1)` is finite (keeping the
/// proper `int_0^x arctan(t) dt`). A symbolic argument (`as_rational_const`
/// None) is never statically flagged.
fn boundary_is_genuinely_nonfinite(ctx: &Context, expr: ExprId) -> bool {
    // Numeric value of a CONSTANT subexpression (no free variables), used to fold
    // function-of-zero forms a rational extractor cannot (sinh(0), e^0-1,
    // |sinh(0)|, ...). Only ever applied to ln arguments / single factors, never
    // to a `0 * singular` product, so there is no 0*inf indeterminacy.
    let numeric = |ctx: &Context, e: ExprId| -> Option<f64> {
        let vars = std::collections::HashMap::new();
        cas_math::evaluator_f64::eval_f64(ctx, e, &vars).filter(|v| v.is_finite())
    };
    let is_zero = |ctx: &Context, e: ExprId| -> bool {
        as_rational_const(ctx, e).is_some_and(|r| r.is_zero())
            || numeric(ctx, e).is_some_and(|v| v.abs() < 1e-12)
    };
    let nonpositive = |ctx: &Context, e: ExprId| -> bool {
        as_rational_const(ctx, e).is_some_and(|r| r <= BigRational::from_integer(0.into()))
            || numeric(ctx, e).is_some_and(|v| v <= 1e-9)
    };
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Neg(a) | Expr::Hold(a) => boundary_is_genuinely_nonfinite(ctx, *a),
        Expr::Mul(a, b) => {
            // 0 * anything = 0: a literal-zero factor kills a singular cofactor.
            if is_zero(ctx, *a) || is_zero(ctx, *b) {
                return false;
            }
            boundary_is_genuinely_nonfinite(ctx, *a) || boundary_is_genuinely_nonfinite(ctx, *b)
        }
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            boundary_is_genuinely_nonfinite(ctx, *a) || boundary_is_genuinely_nonfinite(ctx, *b)
        }
        Expr::Div(a, b) => {
            (is_zero(ctx, *b) && !is_zero(ctx, *a))
                || boundary_is_genuinely_nonfinite(ctx, *a)
                || boundary_is_genuinely_nonfinite(ctx, *b)
        }
        Expr::Pow(base, exp) => {
            (is_zero(ctx, *base) && as_rational_const(ctx, *exp).is_some_and(|r| r.is_negative()))
                || boundary_is_genuinely_nonfinite(ctx, *base)
        }
        Expr::Function(fn_id, args) => {
            if matches!(ctx.builtin_of(*fn_id), Some(cas_ast::BuiltinFn::Ln))
                && args.len() == 1
                && nonpositive(ctx, args[0])
            {
                return true;
            }
            args.iter()
                .any(|&a| boundary_is_genuinely_nonfinite(ctx, a))
        }
        _ => false,
    }
}

fn certify_interval(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    let mut outcome = IntervalCertificate::Certified;
    for condition in conditions {
        match condition {
            ImplicitCondition::NonZero(expr) => {
                match nonzero_on_interval(ctx, *expr, var_name, interval_low, interval_high) {
                    IntervalCertificate::Undefined => return IntervalCertificate::Undefined,
                    other => outcome = combine_certificates(outcome, other),
                }
            }
            ImplicitCondition::Positive(expr) | ImplicitCondition::NonNegative(expr) => {
                match positive_on_interval(ctx, *expr, var_name, interval_low, interval_high) {
                    IntervalCertificate::Undefined => return IntervalCertificate::Undefined,
                    other => outcome = combine_certificates(outcome, other),
                }
            }
            _ => outcome = IntervalCertificate::Unknown,
        }
    }
    outcome
}

fn combine_certificates(
    first: IntervalCertificate,
    second: IntervalCertificate,
) -> IntervalCertificate {
    use IntervalCertificate::*;
    match (first, second) {
        (Undefined, _) | (_, Undefined) => Undefined,
        (Unknown, _) | (_, Unknown) => Unknown,
        (BoundaryTouch { lower: a, upper: b }, BoundaryTouch { lower: c, upper: d }) => {
            BoundaryTouch {
                lower: a || c,
                upper: b || d,
            }
        }
        (touch @ BoundaryTouch { .. }, Certified) | (Certified, touch @ BoundaryTouch { .. }) => {
            touch
        }
        (Certified, Certified) => Certified,
    }
}

/// SELF-CONTAINED risk scan of the integrand: the condition collectors
/// are not guaranteed complete (adversarial review found ln-denominator
/// conditions systematically absent), so certification additionally
/// requires every risky subterm of the integrand itself to be certified
/// on the interval - denominators factor by factor, ln arguments
/// positive AND away from 1 (ln(u) = 0 at u = 1), fractional-power bases
/// positive, trig denominators via the pi enclosure. Anything the scan
/// cannot decide refuses certification.
fn integrand_risks_certified(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    scan_expr_risks(ctx, expr, var_name, &mut |ctx, risk| match risk {
        RiskKind::DenominatorNonZero(factor) => {
            nonzero_on_interval(ctx, factor, var_name, interval_low, interval_high)
        }
        RiskKind::MustBePositive(arg) => {
            positive_on_interval(ctx, arg, var_name, interval_low, interval_high)
        }
        RiskKind::DefinedOnUnitInterval(arg) => {
            unit_interval_certificate(ctx, arg, var_name, interval_low, interval_high)
        }
    })
}

/// -1 <= u <= 1 over the closed interval: certify 1 - u >= 0 and
/// 1 + u >= 0, where an endpoint TOUCH (u = +-1 exactly at a bound) is
/// still certified because arcsin/arccos are defined there.
fn unit_interval_certificate(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    let one = ctx.num(1);
    let upper_slack = ctx.add(Expr::Sub(one, arg));
    let one = ctx.num(1);
    let lower_slack = ctx.add(Expr::Add(one, arg));
    let mut outcome = IntervalCertificate::Certified;
    for slack in [upper_slack, lower_slack] {
        let cert = match positive_on_interval(ctx, slack, var_name, interval_low, interval_high) {
            IntervalCertificate::BoundaryTouch { .. } => IntervalCertificate::Certified,
            other => other,
        };
        outcome = combine_certificates(outcome, cert);
    }
    outcome
}

fn integrand_risks_certified_unbounded(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    scan_expr_risks(ctx, expr, var_name, &mut |ctx, risk| match risk {
        RiskKind::DenominatorNonZero(factor) => {
            if trig_condition_target(ctx, factor, var_name) {
                return IntervalCertificate::Undefined;
            }
            nonzero_on_unbounded_interval(ctx, factor, var_name, lower_bound, upper_bound)
        }
        RiskKind::MustBePositive(arg) => {
            positive_on_unbounded_interval(ctx, arg, var_name, lower_bound, upper_bound)
        }
        // arcsin/arccos cannot stay within [-1, 1] on an unbounded
        // interval for any nonconstant argument the scan certifies today.
        RiskKind::DefinedOnUnitInterval(_) => IntervalCertificate::Unknown,
    })
}

enum RiskKind {
    DenominatorNonZero(ExprId),
    MustBePositive(ExprId),
    /// arcsin/arccos argument: defined on the CLOSED unit interval, so
    /// endpoint touches certify (only the derivative is singular there).
    DefinedOnUnitInterval(ExprId),
}

fn scan_expr_risks(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    certify: &mut dyn FnMut(&mut Context, RiskKind) -> IntervalCertificate,
) -> IntervalCertificate {
    let node = ctx.get(expr).clone();
    let mut outcome = IntervalCertificate::Certified;
    let merge = |certificate: IntervalCertificate, outcome: &mut IntervalCertificate| {
        *outcome = combine_certificates(
            std::mem::replace(outcome, IntervalCertificate::Certified),
            certificate,
        );
    };
    match node {
        Expr::Div(numerator, denominator) => {
            merge(
                certify_denominator_factors(ctx, denominator, var_name, certify),
                &mut outcome,
            );
            merge(
                scan_expr_risks(ctx, numerator, var_name, certify),
                &mut outcome,
            );
            merge(
                scan_expr_risks(ctx, denominator, var_name, certify),
                &mut outcome,
            );
        }
        Expr::Pow(base, exponent) => {
            let exponent_value = as_rational_const(ctx, exponent);
            match exponent_value {
                Some(value) if value.is_integer() && value.is_positive() => {}
                Some(value) if value.is_integer() => {
                    merge(
                        certify_denominator_factors(ctx, base, var_name, certify),
                        &mut outcome,
                    );
                }
                Some(_) => {
                    // Fractional exponent: real-domain base positivity.
                    merge(certify(ctx, RiskKind::MustBePositive(base)), &mut outcome);
                }
                None => {
                    // Variable exponent: total and positive only for a
                    // positive constant base (e included).
                    let base_safe = matches!(ctx.get(base), Expr::Constant(Constant::E))
                        || as_rational_const(ctx, base).is_some_and(|value| value.is_positive());
                    if !base_safe {
                        merge(IntervalCertificate::Unknown, &mut outcome);
                    }
                    merge(
                        scan_expr_risks(ctx, exponent, var_name, certify),
                        &mut outcome,
                    );
                }
            }
            merge(scan_expr_risks(ctx, base, var_name, certify), &mut outcome);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            merge(scan_expr_risks(ctx, l, var_name, certify), &mut outcome);
            merge(scan_expr_risks(ctx, r, var_name, certify), &mut outcome);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            merge(scan_expr_risks(ctx, inner, var_name, certify), &mut outcome);
        }
        Expr::Function(fn_id, args) => {
            let recurse_args =
                |ctx: &mut Context,
                 certify: &mut dyn FnMut(&mut Context, RiskKind) -> IntervalCertificate,
                 outcome: &mut IntervalCertificate| {
                    for arg in &args {
                        let cert = scan_expr_risks(ctx, *arg, var_name, certify);
                        *outcome = combine_certificates(
                            std::mem::replace(outcome, IntervalCertificate::Certified),
                            cert,
                        );
                    }
                };
            match ctx.builtin_of(fn_id) {
                Some(
                    cas_ast::BuiltinFn::Sin
                    | cas_ast::BuiltinFn::Cos
                    | cas_ast::BuiltinFn::Exp
                    | cas_ast::BuiltinFn::Arctan
                    | cas_ast::BuiltinFn::Atan
                    | cas_ast::BuiltinFn::Sinh
                    | cas_ast::BuiltinFn::Cosh
                    | cas_ast::BuiltinFn::Tanh
                    | cas_ast::BuiltinFn::Abs,
                ) => recurse_args(ctx, certify, &mut outcome),
                Some(
                    cas_ast::BuiltinFn::Arcsin
                    | cas_ast::BuiltinFn::Asin
                    | cas_ast::BuiltinFn::Arccos
                    | cas_ast::BuiltinFn::Acos,
                ) if args.len() == 1 => {
                    merge(
                        certify(ctx, RiskKind::DefinedOnUnitInterval(args[0])),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                Some(cas_ast::BuiltinFn::Ln) if args.len() == 1 => {
                    merge(
                        certify(ctx, RiskKind::MustBePositive(args[0])),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                Some(cas_ast::BuiltinFn::Sqrt) if args.len() == 1 => {
                    merge(
                        certify(ctx, RiskKind::MustBePositive(args[0])),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                Some(cas_ast::BuiltinFn::Tan) if args.len() == 1 => {
                    // tan's poles are cos zeros.
                    let cos_arg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![args[0]]);
                    merge(
                        certify(ctx, RiskKind::DenominatorNonZero(cos_arg)),
                        &mut outcome,
                    );
                    recurse_args(ctx, certify, &mut outcome);
                }
                _ => merge(IntervalCertificate::Unknown, &mut outcome),
            }
        }
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {}
        _ => merge(IntervalCertificate::Unknown, &mut outcome),
    }
    outcome
}

/// Certify a denominator factor by factor: polynomials by root location,
/// integer powers by their base, exp-like factors are never zero,
/// ln(u) = 0 exactly at u = 1 (certified via u - 1 nonzero plus the ln
/// domain), trig factors via the pi enclosure.
#[allow(clippy::only_used_in_recursion)]
fn certify_denominator_factors(
    ctx: &mut Context,
    denominator: ExprId,
    var_name: &str,
    certify: &mut dyn FnMut(&mut Context, RiskKind) -> IntervalCertificate,
) -> IntervalCertificate {
    let node = ctx.get(denominator).clone();
    match node {
        Expr::Mul(l, r) => combine_certificates(
            certify_denominator_factors(ctx, l, var_name, certify),
            certify_denominator_factors(ctx, r, var_name, certify),
        ),
        Expr::Neg(inner) | Expr::Hold(inner) => {
            certify_denominator_factors(ctx, inner, var_name, certify)
        }
        Expr::Pow(base, exponent) => {
            // A positive constant base (e included) is never zero,
            // whatever the exponent.
            let base_never_zero = matches!(ctx.get(base), Expr::Constant(Constant::E))
                || as_rational_const(ctx, base).is_some_and(|value| value.is_positive());
            if base_never_zero {
                return IntervalCertificate::Certified;
            }
            match as_rational_const(ctx, exponent) {
                Some(value) if !value.is_zero() => {
                    certify_denominator_factors(ctx, base, var_name, certify)
                }
                _ => IntervalCertificate::Unknown,
            }
        }
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id) {
            Some(cas_ast::BuiltinFn::Exp) => IntervalCertificate::Certified,
            Some(cas_ast::BuiltinFn::Ln) => {
                let one = ctx.num(1);
                let shifted = ctx.add(Expr::Sub(args[0], one));
                combine_certificates(
                    certify(ctx, RiskKind::MustBePositive(args[0])),
                    certify(ctx, RiskKind::DenominatorNonZero(shifted)),
                )
            }
            Some(cas_ast::BuiltinFn::Cos | cas_ast::BuiltinFn::Sin) => {
                certify(ctx, RiskKind::DenominatorNonZero(denominator))
            }
            Some(cas_ast::BuiltinFn::Sqrt) => {
                // sqrt(u) = 0 exactly at u = 0; strict positivity of u
                // certifies both the domain and the nonzero denominator.
                certify(ctx, RiskKind::MustBePositive(args[0]))
            }
            _ => IntervalCertificate::Unknown,
        },
        _ => certify(ctx, RiskKind::DenominatorNonZero(denominator)),
    }
}

/// Strict positivity of a polynomial condition on the closed interval:
/// variable-free numerics decide directly; otherwise every rational root
/// must lie strictly outside [low, high], the non-root residual must have
/// no real roots (negative discriminant or constant), and a sign probe at
/// an interior root-free point confirms the sign. Roots touching or
/// inside the interval are conservatively Unknown (the boundary case may
/// be a convergent improper integral, which this rung does not decide).
fn positive_on_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    if let Some(value) = as_rational_const(ctx, expr) {
        return if value.is_positive() {
            IntervalCertificate::Certified
        } else {
            IntervalCertificate::Unknown
        };
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return IntervalCertificate::Unknown;
    };
    if poly.is_zero() {
        return IntervalCertificate::Unknown;
    }

    let mut residual_has_real_roots_ruled_out = false;
    let mut touches = IntervalCertificate::Certified;
    let factors = poly.factor_rational_roots();
    for factor in &factors {
        match factor.degree() {
            0 => {}
            1 => {
                let root = -&factor.coeffs[0] / &factor.coeffs[1];
                match root_position(interval_low, interval_high, &root) {
                    Some(RootPosition::Outside) => {}
                    Some(RootPosition::AtLower) => {
                        touches = combine_certificates(
                            touches,
                            IntervalCertificate::BoundaryTouch {
                                lower: true,
                                upper: false,
                            },
                        );
                    }
                    Some(RootPosition::AtUpper) => {
                        touches = combine_certificates(
                            touches,
                            IntervalCertificate::BoundaryTouch {
                                lower: false,
                                upper: true,
                            },
                        );
                    }
                    _ => return IntervalCertificate::Unknown,
                }
            }
            2 => {
                let a = factor.coeffs[2].clone();
                let b = factor.coeffs[1].clone();
                let c = factor.coeffs[0].clone();
                let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
                if !discriminant.is_negative() {
                    // Irrational real roots could lie anywhere.
                    return IntervalCertificate::Unknown;
                }
                residual_has_real_roots_ruled_out = true;
            }
            _ => return IntervalCertificate::Unknown,
        }
    }
    let _ = residual_has_real_roots_ruled_out;

    // No root strictly inside: the sign is constant in the interior;
    // probe a rational point strictly inside.
    let Some(probe) = interval_rational_probe(interval_low, interval_high) else {
        return IntervalCertificate::Unknown;
    };
    if poly.eval(&probe).is_positive() {
        touches
    } else {
        IntervalCertificate::Unknown
    }
}

/// Positivity of a polynomial on a (half-)infinite interval: globally
/// positive quadratics certify directly; otherwise every rational root
/// must lie strictly outside, quadratic residuals must have no real
/// roots, infinite tails must point positive (leading-coefficient sign),
/// and a probe inside confirms.
fn positive_on_unbounded_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    if globally_positive(ctx, expr, var_name) {
        return IntervalCertificate::Certified;
    }
    if let Some(value) = as_rational_const(ctx, expr) {
        return if value.is_positive() {
            IntervalCertificate::Certified
        } else {
            IntervalCertificate::Unknown
        };
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return IntervalCertificate::Unknown;
    };
    if poly.is_zero() {
        return IntervalCertificate::Unknown;
    }

    let inside = |root: &BigRational| -> bool {
        let point = Endpoint::from_rational(root.clone());
        let above_lower = match lower_bound {
            DefiniteBound::NegInfinity => true,
            DefiniteBound::Finite(lower) => {
                !matches!(point.try_cmp(lower), Some(std::cmp::Ordering::Less))
            }
            _ => return true, // conservador: trátalo como dentro
        };
        let below_upper = match upper_bound {
            DefiniteBound::PosInfinity => true,
            DefiniteBound::Finite(upper) => {
                !matches!(point.try_cmp(upper), Some(std::cmp::Ordering::Greater))
            }
            _ => return true,
        };
        above_lower && below_upper
    };
    for factor in poly.factor_rational_roots() {
        match factor.degree() {
            0 => {}
            1 => {
                let root = -&factor.coeffs[0] / &factor.coeffs[1];
                if inside(&root) {
                    return IntervalCertificate::Unknown;
                }
            }
            2 => {
                let a = factor.coeffs[2].clone();
                let b = factor.coeffs[1].clone();
                let c = factor.coeffs[0].clone();
                let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
                if !discriminant.is_negative() {
                    return IntervalCertificate::Unknown;
                }
            }
            _ => return IntervalCertificate::Unknown,
        }
    }

    // Infinite tails must be positive.
    let leading = poly.leading_coeff();
    if matches!(upper_bound, DefiniteBound::PosInfinity) && !leading.is_positive() {
        return IntervalCertificate::Unknown;
    }
    if matches!(lower_bound, DefiniteBound::NegInfinity) {
        let degree_even = poly.degree() % 2 == 0;
        let tail_positive = if degree_even {
            leading.is_positive()
        } else {
            leading.is_negative()
        };
        if !tail_positive {
            return IntervalCertificate::Unknown;
        }
    }

    // Probe a point inside the certified root-free region (the enclosure
    // bound is rational even for pi-valued endpoints).
    let probe_point = match (lower_bound, upper_bound) {
        (DefiniteBound::Finite(lower), _) => {
            lower.enclosure().1 + BigRational::from_integer(1.into())
        }
        (_, DefiniteBound::Finite(upper)) => {
            upper.enclosure().0 - BigRational::from_integer(1.into())
        }
        _ => BigRational::from_integer(0.into()),
    };
    if poly.eval(&probe_point).is_positive() {
        IntervalCertificate::Certified
    } else {
        IntervalCertificate::Unknown
    }
}

/// Rational enclosure of pi, tight enough for textbook bounds.
fn pi_enclosure() -> (BigRational, BigRational) {
    let denom = num_bigint::BigInt::from(100_000_000_000_000u64);
    (
        BigRational::new(
            num_bigint::BigInt::from(314_159_265_358_979u64),
            denom.clone(),
        ),
        BigRational::new(num_bigint::BigInt::from(314_159_265_358_980u64), denom),
    )
}

/// Rational enclosure of e, tight enough for textbook bounds.
fn e_enclosure() -> (BigRational, BigRational) {
    let denom = num_bigint::BigInt::from(100_000_000_000_000u64);
    (
        BigRational::new(
            num_bigint::BigInt::from(271_828_182_845_904u64),
            denom.clone(),
        ),
        BigRational::new(num_bigint::BigInt::from(271_828_182_845_905u64), denom),
    )
}

/// Zeros of cos (odd multiples of pi/2) or sin (integer multiples of pi)
/// against the closed rational interval, via the pi enclosure: every zero
/// enclosure disjoint from [low, high] certifies; an enclosure fully
/// inside is a pole; overlap with the boundary stays Unknown.
fn trig_nonzero_on_interval(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> Option<IntervalCertificate> {
    let builtin = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let inner_is_var = matches!(ctx.get(args[0]),
                Expr::Variable(sym) if ctx.sym_name(*sym) == var_name);
            if !inner_is_var {
                return None;
            }
            ctx.builtin_of(*fn_id)?
        }
        _ => return None,
    };
    let half = BigRational::new(1.into(), 2.into());
    // Zeros at multiplier * pi with multiplier in the arithmetic
    // progression below.
    let (start, step) = match builtin {
        cas_ast::BuiltinFn::Cos => (half, BigRational::from_integer(1.into())),
        cas_ast::BuiltinFn::Sin => (
            BigRational::from_integer(0.into()),
            BigRational::from_integer(1.into()),
        ),
        _ => return None,
    };
    let (pi_low, pi_high) = pi_enclosure();

    // Multiplier window covering the interval generously.
    let approx_low = &interval_low.enclosure().0 / &pi_high - BigRational::from_integer(2.into());
    let approx_high = &interval_high.enclosure().1 / &pi_low + BigRational::from_integer(2.into());
    let k_low = approx_low.floor().to_integer();
    let k_high = approx_high.ceil().to_integer();

    let mut k = k_low;
    while k <= k_high {
        let multiplier = &start + &step * BigRational::from_integer(k.clone());
        // The zero is exactly multiplier * pi: pi-pure, so comparisons
        // against pi-multiple bounds are exact rational comparisons.
        let zero = Endpoint::from_pi_multiple(multiplier);
        let before_interval = matches!(zero.try_cmp(interval_low), Some(std::cmp::Ordering::Less));
        let after_interval = matches!(
            zero.try_cmp(interval_high),
            Some(std::cmp::Ordering::Greater)
        );
        if !(before_interval || after_interval) {
            let strictly_inside =
                matches!(
                    zero.try_cmp(interval_low),
                    Some(std::cmp::Ordering::Greater)
                ) && matches!(zero.try_cmp(interval_high), Some(std::cmp::Ordering::Less));
            if strictly_inside {
                return Some(IntervalCertificate::Undefined);
            }
            if matches!(zero.try_cmp(interval_low), Some(std::cmp::Ordering::Equal)) {
                return Some(IntervalCertificate::BoundaryTouch {
                    lower: true,
                    upper: false,
                });
            }
            if matches!(zero.try_cmp(interval_high), Some(std::cmp::Ordering::Equal)) {
                return Some(IntervalCertificate::BoundaryTouch {
                    lower: false,
                    upper: true,
                });
            }
            return Some(IntervalCertificate::Unknown);
        }
        k += 1;
    }
    Some(IntervalCertificate::Certified)
}

fn nonzero_on_interval(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    if let Some(certificate) =
        trig_nonzero_on_interval(ctx, expr, var_name, interval_low, interval_high)
    {
        return certificate;
    }
    if let Some(value) = as_rational_const(ctx, expr) {
        return if value.is_zero() {
            IntervalCertificate::Undefined
        } else {
            IntervalCertificate::Certified
        };
    }
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return IntervalCertificate::Unknown;
    };
    match poly.degree() {
        0 => {
            if poly.coeffs.first().is_none_or(num_traits::Zero::is_zero) {
                IntervalCertificate::Undefined
            } else {
                IntervalCertificate::Certified
            }
        }
        1 => {
            let root = -&poly.coeffs[0] / &poly.coeffs[1];
            match root_position(interval_low, interval_high, &root) {
                // A pole strictly inside the interval: divergent.
                Some(RootPosition::Inside) => IntervalCertificate::Undefined,
                Some(RootPosition::Outside) => IntervalCertificate::Certified,
                Some(RootPosition::AtLower) => IntervalCertificate::BoundaryTouch {
                    lower: true,
                    upper: false,
                },
                Some(RootPosition::AtUpper) => IntervalCertificate::BoundaryTouch {
                    lower: false,
                    upper: true,
                },
                None => IntervalCertificate::Unknown,
            }
        }
        2 => {
            let a = poly.coeffs[2].clone();
            let b = poly.coeffs[1].clone();
            let c = poly.coeffs[0].clone();
            let discriminant = &b * &b - BigRational::from_integer(4.into()) * &a * &c;
            if discriminant.is_negative() {
                IntervalCertificate::Certified
            } else {
                IntervalCertificate::Unknown
            }
        }
        _ => IntervalCertificate::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    pub(super) fn eval_definite(source: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(source, &mut ctx).expect(source);
        let call =
            crate::symbolic_calculus_call_support::try_extract_definite_integrate_call(&ctx, expr)?;
        let rewrite = definite_integration_rewrite(&mut ctx, &call)?;
        Some(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ))
    }

    #[test]
    fn ftc_evaluates_certified_intervals() {
        // The raw rewrite is the unsimplified F(b) - F(a); the engine
        // simplifier folds it to 1/3 (pinned by the matrix row).
        assert!(eval_definite("integrate(x^2, x, 0, 1)").is_some());
        // Orientation is automatic: F(upper) - F(lower) with original bounds.
        assert!(eval_definite("integrate(x, x, 1, 0)").is_some());
    }

    #[test]
    fn abs_linear_definite_integral_splits_at_the_root() {
        // |c x + d| over rational bounds, split at r = -d/c.
        for (source, expected) in [
            ("integrate(abs(x), x, -1, 1)", "1"),
            ("integrate(abs(x-1), x, 0, 2)", "1"),
            ("integrate(abs(2*x-1), x, 0, 1)", "1/2"),
            ("integrate(abs(x), x, -2, 3)", "13/2"),
            ("integrate(abs(x-1), x, 2, 5)", "15/2"), // root outside the interval
            ("integrate(abs(3-x), x, 0, 4)", "5"),
        ] {
            assert_eq!(eval_definite(source).as_deref(), Some(expected), "{source}");
        }
    }

    #[test]
    fn abs_linear_definite_integral_declines_out_of_scope() {
        // Not a bare |linear|: a product (owned by odd symmetry), a quadratic
        // inner, a symbolic (pi) bound, and the indefinite form all decline here.
        // The product still resolves elsewhere (odd symmetry -> 0); the others
        // stay residual.
        assert_eq!(
            eval_definite("integrate(abs(x)*x, x, -1, 1)").as_deref(),
            Some("0")
        );
        assert!(eval_definite("integrate(abs(x^2-1), x, 0, 2)").is_none());
        assert!(eval_definite("integrate(abs(x), x, 0, pi)").is_none());
    }

    #[test]
    fn pole_inside_closed_interval_is_undefined() {
        let result = eval_definite("integrate(1/x, x, -1, 1)").expect("rewrite");
        assert_eq!(result, "undefined");
        // An endpoint pole is a boundary touch: the one-sided limit of
        // ln|x| reports the honest signed divergence instead.
        let endpoint = eval_definite("integrate(1/x, x, 0, 1)").expect("rewrite");
        assert_eq!(endpoint, "infinity");
    }

    #[test]
    fn unknown_certificates_stay_residual() {
        // Symbolic bound over a CONDITIONAL antiderivative: the pole
        // location cannot be certified against a symbolic interval.
        assert!(eval_definite("integrate(1/x, x, 1, t)").is_none());
        // Symbolic positivity condition cannot be certified.
        assert!(eval_definite("integrate(c/((x+b)^2+a), x, 0, 1)").is_none());
    }

    #[test]
    fn symbolic_bounds_evaluate_for_unconditional_antiderivatives() {
        // The curriculum "area function": condition-free antiderivatives
        // are continuous on all of R, so symbolic bounds need no
        // certificate.
        assert!(eval_definite("integrate(x^2, x, 0, t)").is_some());
        assert!(eval_definite("integrate(1/(x^2+1), x, a, b)").is_some());
    }
}

#[cfg(test)]
mod odd_symmetry_tests {
    use super::tests::eval_definite;

    #[test]
    fn odd_integrand_over_symmetric_interval_is_zero() {
        // No elementary antiderivative, but odd + continuous + symmetric -> 0.
        for source in [
            "integrate(sin(x)/(1+x^2), x, -1, 1)",
            "integrate(sin(x)*exp(x^2), x, -1, 1)",
            "integrate(sin(x^3), x, -2, 2)",
            "integrate(x^3*sin(x^2)*exp(x^4), x, -1, 1)",
            // tan is odd and continuous on [-1, 1] (the pole is at pi/2 > 1).
            "integrate(tan(x)*exp(x^2), x, -1, 1)",
            // a rational-base exponential 2^(x^2) = e^(x^2 ln 2) is even.
            "integrate(sin(x)*2^(x^2), x, -1, 1)",
        ] {
            assert_eq!(eval_definite(source).as_deref(), Some("0"), "{source}");
        }
    }

    #[test]
    fn symmetry_declines_unsound_and_inapplicable_shapes() {
        // Interior pole (1/x is odd but NOT integrable to 0).
        assert_eq!(
            eval_definite("integrate(1/x, x, -1, 1)").as_deref(),
            Some("undefined"),
        );
        // tan has a pole at pi/2 inside [-2, 2]: not certified, stays residual.
        assert!(eval_definite("integrate(tan(x)*exp(x^2), x, -2, 2)").is_none());
        // Denominator roots at +-1/2 inside [-1, 1].
        assert!(eval_definite("integrate(sin(x)/(x^2-1/4), x, -1, 1)").is_none());
        // Even integrand, not odd.
        assert!(eval_definite("integrate(cos(x)/(1+x^2), x, -1, 1)").is_none());
        assert!(eval_definite("integrate(exp(x^2), x, -1, 1)").is_none());
        // Asymmetric interval.
        assert!(eval_definite("integrate(sin(x)/(1+x^2), x, 0, 1)").is_none());
        assert!(eval_definite("integrate(sin(x)*exp(x^2), x, -1, 2)").is_none());
        // sin(x)*exp(x) is neither odd nor even (exp(x) is neither); the
        // classifier declines and FTC owns it (elementary antiderivative).
        let ftc = eval_definite("integrate(sin(x)*exp(x), x, -1, 1)").expect("FTC");
        assert_ne!(ftc, "0");
    }
}

#[cfg(test)]
mod improper_tests {
    use super::tests::eval_definite;

    #[test]
    fn improper_integrals_converge_via_limits() {
        assert!(eval_definite("integrate(1/x^2, x, 1, infinity)").is_some());
        assert!(eval_definite("integrate(1/(x^2+1), x, -infinity, infinity)").is_some());
    }

    #[test]
    fn improper_divergence_reports_infinity() {
        let result = eval_definite("integrate(x^2, x, 0, infinity)").expect("rewrite");
        assert_eq!(result, "infinity");
    }

    #[test]
    fn improper_pole_inside_unbounded_interval_is_undefined() {
        let result = eval_definite("integrate(1/x^2, x, -1, infinity)").expect("rewrite");
        assert_eq!(result, "undefined");
    }

    #[test]
    fn logarithmic_divergence_reports_signed_infinity() {
        // F = ln|x| now resolves at both infinities, so the divergent
        // half-line integrals report their honest signed value.
        assert_eq!(
            eval_definite("integrate(1/x, x, 1, infinity)").as_deref(),
            Some("infinity")
        );
        assert_eq!(
            eval_definite("integrate(1/x, x, -infinity, -1)").as_deref(),
            Some("-infinity")
        );
    }
}

#[cfg(test)]
mod certificate_tests {
    use super::tests::eval_definite;

    #[test]
    fn cofactor_route_antiderivatives_evaluate_definitely() {
        assert!(eval_definite("integrate(x/sqrt(x^2+1), x, 0, 1)").is_some());
    }

    #[test]
    fn polynomial_positivity_certifies_away_from_roots() {
        // Positive(1-x^2) with roots +-1 strictly outside [0, 1/2].
        assert!(eval_definite("integrate(x/sqrt(1-x^2), x, 0, 1/2)").is_some());
        // Root on the boundary: now a boundary touch that converges via
        // the one-sided limit of the antiderivative.
        assert!(eval_definite("integrate(x/sqrt(1-x^2), x, 0, 1)").is_some());
    }

    #[test]
    fn trig_nonzero_certificate_locates_cosine_zeros() {
        // [0, 1] is inside (-pi/2, pi/2): certified. (The raw unit
        // context sees the simplifier-normalized 1/cos^2 form.)
        assert!(eval_definite("integrate(1/cos(x)^2, x, 0, 1)").is_some());
        // pi/2 inside [0, 2]: pole, undefined.
        assert_eq!(
            eval_definite("integrate(tan(x), x, 0, 2)").as_deref(),
            Some("undefined")
        );
        // Unbounded interval always contains cosine zeros.
        assert_eq!(
            eval_definite("integrate(tan(x), x, 0, infinity)").as_deref(),
            Some("undefined")
        );
    }
}

#[cfg(test)]
mod pi_bound_tests {
    use super::tests::eval_definite;

    #[test]
    fn pi_multiple_bounds_certify_exactly_against_trig_zeros() {
        // [0, pi/4] inside (-pi/2, pi/2): exact pi-pure comparison.
        assert!(eval_definite("integrate(1/cos(x)^2, x, 0, pi/4)").is_some());
        // pi/2 strictly inside [0, 3pi/4]: undefined, also exactly.
        assert_eq!(
            eval_definite("integrate(tan(x), x, 0, 3*pi/4)").as_deref(),
            Some("undefined")
        );
    }

    #[test]
    fn pi_endpoints_compose_with_polynomial_certificates() {
        // Positive(x^2+1)-style risks certify against the pi enclosure.
        assert!(eval_definite("integrate(x/(x^2+1), x, 0, pi)").is_some());
    }

    #[test]
    fn mixed_undecidable_endpoints_stay_residual() {
        // A bound inside the pi enclosure of pi/2 cannot be decided and
        // must refuse rather than guess.
        assert!(
            eval_definite("integrate(1/cos(x)^2, x, 0, 15707963267948966/10000000000000000)")
                .is_none()
        );
    }
}

#[cfg(test)]
mod boundary_touch_tests {
    use super::tests::eval_definite;

    #[test]
    fn bounded_inverse_trig_integrands_certify_on_the_unit_interval() {
        // arcsin/arccos are defined on the CLOSED unit interval: the
        // endpoint touch at x = 1 certifies, out-of-domain refuses.
        assert!(eval_definite("integrate(x*arcsin(x), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(arccos(x), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(arcsin(x), x, -1, 1)").is_some());
        assert!(eval_definite("integrate(arcsin(x), x, 0, 2)").is_none());
        assert!(eval_definite("integrate(x*arcsin(x), x, -2, 0)").is_none());
    }

    #[test]
    fn boundary_touch_fractional_power_atoms_evaluate() {
        // Positive fractional powers touching x = 0: the antiderivative's
        // one-sided limit resolves via the power atom + product/scale rules.
        assert!(eval_definite("integrate(x^(1/2), x, 0, 4)").is_some());
        assert!(eval_definite("integrate(x^(1/3), x, 0, 8)").is_some());
        assert!(eval_definite("integrate(x^(3/2), x, 0, 1)").is_some());
    }

    #[test]
    fn boundary_convergent_improper_integrals_evaluate() {
        // The textbook trio: F's one-sided limit at the touched endpoint.
        assert!(eval_definite("integrate(ln(x), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(x*(1-x^2)^(-1/2), x, 0, 1)").is_some());
        assert!(eval_definite("integrate(x^(-1/2), x, 0, 1)").is_some());
    }

    #[test]
    fn boundary_power_log_improper_integrals_evaluate() {
        // int_0^1 x^a ln(x)^b: the antiderivative's lower endpoint touches 0
        // through x^a ln(x)^b -> 0 terms (power dominates the log polynomial).
        // The raw rewrite is the unsimplified F(1) - F(0+); the simplifier and
        // the matrix smoke pin the exact values (2, -1/4, -4, -1/9, -4/9). Here
        // we only certify the lower endpoint resolves (no residual limit call).
        // The fractional-power forms (ln(x)/sqrt(x), sqrt(x)*ln(x)) only reach
        // the integrator after the pre-simplifier rewrites their radicals to
        // x^(p/q), so they are pinned by the CLI-driven matrix smoke instead of
        // here; these polynomial-coefficient forms exercise the same boundary.
        for source in [
            "integrate(ln(x)^2, x, 0, 1)",
            "integrate(x*ln(x), x, 0, 1)",
            "integrate(x^2*ln(x), x, 0, 1)",
        ] {
            let result = eval_definite(source).unwrap_or_else(|| panic!("must resolve: {source}"));
            assert!(
                !result.contains("limit("),
                "{source} left a residual: {result}"
            );
        }
    }

    #[test]
    fn boundary_divergence_reports_signed_infinity() {
        assert_eq!(
            eval_definite("integrate(1/x^2, x, 0, 1)").as_deref(),
            Some("infinity")
        );
    }

    #[test]
    fn interior_roots_still_refuse_or_diverge() {
        assert_eq!(
            eval_definite("integrate(1/x, x, -1, 1)").as_deref(),
            Some("undefined")
        );
    }

    #[test]
    fn gaussian_moment_integrals_resolve_to_table_values() {
        for (source, expected) in [
            ("integrate(e^(-x^2), x, 0, infinity)", "1/2 * pi^(1/2)"),
            ("integrate(e^(-x^2), x, -infinity, infinity)", "pi^(1/2)"),
            ("integrate(x^2*e^(-x^2), x, 0, infinity)", "1/4 * pi^(1/2)"),
            (
                "integrate(x^2*e^(-x^2), x, -infinity, infinity)",
                "1/2 * pi^(1/2)",
            ),
            ("integrate(x^4*e^(-x^2), x, 0, infinity)", "3/8 * pi^(1/2)"),
            (
                "integrate(e^(-2*x^2), x, 0, infinity)",
                "1/2 * (pi / 2)^(1/2)",
            ),
            ("integrate(e^(-x^2), x, -infinity, 0)", "1/2 * pi^(1/2)"),
            // Constant numerator coefficients must scale the result
            // (regression: the Div form once dropped the coefficient).
            ("integrate(2/e^(x^2), x, 0, infinity)", "pi^(1/2)"),
            ("integrate(3/e^(x^2), x, 0, infinity)", "3/2 * pi^(1/2)"),
            ("integrate((-2)/e^(x^2), x, 0, infinity)", "-pi^(1/2)"),
        ] {
            assert_eq!(eval_definite(source).as_deref(), Some(expected), "{source}");
        }
    }

    #[test]
    fn gaussian_table_keeps_honesty_residuals() {
        // The indefinite Gaussian, finite bounds, growing exponential,
        // non-quadratic exponents, and a linear-shifted exponent must NOT
        // resolve through the table (honesty list).
        for source in [
            "integrate(e^(-x^2), x, 0, 1)",
            "integrate(e^(x^2), x, 0, infinity)",
            "integrate(e^(-x^3), x, 0, infinity)",
            "integrate(e^(-x^2+x), x, 0, infinity)",
            "integrate(e^(-x^2), x, 1, infinity)",
        ] {
            // eval_definite returns None when the recognizer declines and
            // no antiderivative exists (the residual case).
            assert!(
                eval_definite(source).is_none(),
                "must stay residual: {source}"
            );
        }
    }

    #[test]
    fn gamma_half_integer_moment_integrals_resolve_to_table_values() {
        // int_0^inf x^(m-1/2) e^(-a x) = (2m)!/(4^m m!)/a^m sqrt(pi/a).
        for (source, expected) in [
            ("integrate(e^(-x)/sqrt(x), x, 0, infinity)", "pi^(1/2)"),
            (
                "integrate(sqrt(x)*e^(-x), x, 0, infinity)",
                "1/2 * pi^(1/2)",
            ),
            (
                "integrate(x^(3/2)*e^(-x), x, 0, infinity)",
                "3/4 * pi^(1/2)",
            ),
            (
                "integrate(e^(-2*x)/sqrt(x), x, 0, infinity)",
                "(pi / 2)^(1/2)",
            ),
            (
                "integrate(3*e^(-x)/sqrt(x), x, 0, infinity)",
                "3 * pi^(1/2)",
            ),
            // Unit-magnitude negative coefficient (a top-level Neg wrapper).
            ("integrate(-e^(-x)/sqrt(x), x, 0, infinity)", "-pi^(1/2)"),
            (
                "integrate(-x^(3/2)*e^(-x), x, 0, infinity)",
                "-3/4 * pi^(1/2)",
            ),
        ] {
            assert_eq!(eval_definite(source).as_deref(), Some(expected), "{source}");
        }
    }

    #[test]
    fn gamma_half_integer_table_keeps_honesty_residuals() {
        // The indefinite form, finite bounds, integer powers (elementary
        // antiderivative), a divergent power (s < -1/2), a non-linear decay,
        // and a constant-shifted exponent must NOT resolve through the table.
        for source in [
            "integrate(e^(-x)/sqrt(x), x, 0, 1)",
            "integrate(e^(-x)/x^(3/2), x, 0, infinity)",
            "integrate(e^(-x^2)/sqrt(x), x, 0, infinity)",
            "integrate(e^(-x+1)/sqrt(x), x, 0, infinity)",
            "integrate(sqrt(x)*e^(x), x, 0, infinity)",
        ] {
            assert!(
                eval_definite(source).is_none(),
                "must stay residual: {source}"
            );
        }
        // Integer moments still resolve via the elementary antiderivative
        // (raw F(inf) - F(0) here; the simplifier and CLI fold it to 1).
        assert!(eval_definite("integrate(x*e^(-x), x, 0, infinity)").is_some());
    }

    #[test]
    fn e_multiple_bounds_certify_reciprocal_integrals() {
        // e and rational multiples of e are now finite endpoints with a
        // rational enclosure, so the pole certificate can place poles
        // and the FTC evaluates (1/x to e used to stay residual).
        for source in [
            "integrate(1/x, x, 1, e)",
            "integrate(1/x^2, x, 1, e)",
            "integrate(2/x, x, 1, e)",
            "integrate(1/x, x, 1, 2*e)",
        ] {
            assert!(eval_definite(source).is_some(), "must certify: {source}");
        }
    }

    #[test]
    fn e_bound_places_poles_via_enclosure() {
        // Pole at 2 is inside [1, e] (e ~ 2.718): diverges. Pole at 3 is
        // outside: certifies. The e enclosure decides both.
        assert_eq!(
            eval_definite("integrate(1/(x-2), x, 1, e)").as_deref(),
            Some("undefined")
        );
        assert!(eval_definite("integrate(1/(x-3), x, 1, e)").is_some());
    }
}
