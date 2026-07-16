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

    /// The exact rational value when this endpoint carries no `pi`/`e` part.
    fn as_pure_rational(&self) -> Option<&BigRational> {
        if self.pi_multiple.is_zero() && self.e_multiple.is_zero() {
            Some(&self.rational)
        } else {
            None
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
            // Otherwise the base carries the variable.
            let base_parity = parity_in_var(ctx, *base, var_name)?;
            if base_parity == VarParity::Even {
                // An even base is even for ANY x-free exponent — integer or not
                // (`√(x²-1)`, `(x²+1)^(3/2)`): `base(-x) = base(x)` ⇒ `base^p(-x) = base^p(x)`.
                return Some(VarParity::Even);
            }
            // Odd base: parity carries only through an x-free INTEGER exponent.
            let exponent = as_rational_const(ctx, *exp)?;
            if !exponent.is_integer() {
                return None;
            }
            Some(if exponent.to_integer().is_even() {
                VarParity::Even
            } else {
                VarParity::Odd
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
fn abs_polynomial_definite_integral_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    use num_traits::{One, Zero};
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

    // The integrand must be exactly |g(x)| with g a polynomial in the variable.
    let Expr::Function(fn_id, args) = ctx.get(call.target).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(cas_ast::BuiltinFn::Abs) {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, args[0], &call.var_name).ok()?;
    if poly.degree() < 1 {
        return None;
    }

    // F(x) = ∫ g = Σ aᵢ·xⁱ⁺¹/(i+1) — exact at any rational point. On each interval between
    // consecutive real roots of g, g keeps its sign, so ∫|g| = |∫g| = |F(b) − F(a)|.
    let antiderivative = |x: &BigRational| -> BigRational {
        let mut acc = BigRational::zero();
        for (i, a) in poly.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            let mut power = BigRational::one();
            for _ in 0..=i {
                power *= x;
            }
            acc += a * &power / BigRational::from_integer(((i + 1) as i64).into());
        }
        acc
    };

    // Sign-change points = real roots of g inside the interval. Collect the RATIONAL roots as
    // split points; if any factor carries irrational real roots (a residual quadratic with
    // discriminant ≥ 0, or any residual of degree ≥ 3), decline — splitting there would need an
    // irrational breakpoint, and missing such a sign change would give a wrong value.
    let (left, right, sign) = if lo <= hi {
        (lo.clone(), hi.clone(), BigRational::one())
    } else {
        (hi.clone(), lo.clone(), -BigRational::one())
    };
    let mut breakpoints: Vec<BigRational> = Vec::new();
    for factor in poly.factor_rational_roots() {
        match factor.degree() {
            0 => {}
            1 => {
                let root = -&factor.coeffs[0] / &factor.coeffs[1];
                if left < root && root < right {
                    breakpoints.push(root);
                }
            }
            2 => {
                let a = &factor.coeffs[2];
                let b = &factor.coeffs[1];
                let c = &factor.coeffs[0];
                let discriminant = b * b - BigRational::from_integer(4.into()) * a * c;
                if !discriminant.is_negative() {
                    // Irrational real roots: fine ONLY if both lie outside the interval (no sign
                    // change to split at). A root inside needs an irrational breakpoint — decline.
                    let lo_e = Endpoint::from_rational(left.clone());
                    let hi_e = Endpoint::from_rational(right.clone());
                    match quadratic_real_roots_clear_of_interval(a, b, c, &lo_e, &hi_e) {
                        IntervalCertificate::Certified => {}
                        _ => return None,
                    }
                }
            }
            _ => return None,
        }
    }
    breakpoints.sort();

    // Sum |F(bᵢ₊₁) − F(bᵢ)| over the pieces [left, b₁], …, [bₖ, right].
    let mut cut = vec![left];
    cut.extend(breakpoints);
    cut.push(right);
    let mut total = BigRational::zero();
    for piece in cut.windows(2) {
        total += (antiderivative(&piece[1]) - antiderivative(&piece[0])).abs();
    }

    let value = ctx.add(Expr::Number(sign * total));
    Some(Rewrite::new(value).desc("integral of |polynomial| split at its real roots"))
}

/// `∫_a^b |N/D|` where the rational `N/D` is SIGN-DEFINITE on `[a,b]` — `N` has no root and `D` no
/// pole inside, so the sign never flips. Certify that (reusing `nonzero_on_interval`), read the sign
/// at the rational midpoint, strip the absolute value (negating for a negative integrand), and
/// delegate to the ordinary definite machinery. The non-rational-argument and the sign-changing
/// cases are out of scope and decline.
fn abs_sign_definite_rational_definite_integral_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    use num_traits::Zero;
    let (DefiniteBound::Finite(low), DefiniteBound::Finite(high)) = (lower_bound, upper_bound)
    else {
        return None;
    };
    let lo = low.as_pure_rational()?;
    let hi = high.as_pure_rational()?;
    if lo == hi {
        return None;
    }

    // Integrand must be exactly `|g|` with `g = N/D` a genuine rational (non-constant denominator).
    let Expr::Function(fn_id, args) = ctx.get(call.target).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(cas_ast::BuiltinFn::Abs) {
        return None;
    }
    let inner = args[0];
    let Expr::Div(num, den) = ctx.get(inner).clone() else {
        return None;
    };
    let num_poly = Polynomial::from_expr(ctx, num, &call.var_name).ok()?;
    let den_poly = Polynomial::from_expr(ctx, den, &call.var_name).ok()?;
    if den_poly.degree() < 1 {
        return None;
    }

    // Sign-definite ⟺ neither N nor D vanishes anywhere on the closed interval.
    if !matches!(
        nonzero_on_interval(ctx, num, &call.var_name, low, high),
        IntervalCertificate::Certified
    ) || !matches!(
        nonzero_on_interval(ctx, den, &call.var_name, low, high),
        IntervalCertificate::Certified
    ) {
        return None;
    }

    // Read the (constant) sign at the rational midpoint.
    let mid = (lo + hi) / BigRational::from_integer(2.into());
    let den_mid = den_poly.eval(&mid);
    if den_mid.is_zero() {
        return None;
    }
    let g_mid = num_poly.eval(&mid) / den_mid;
    if g_mid.is_zero() {
        return None;
    }

    let stripped = if g_mid > BigRational::zero() {
        inner
    } else {
        ctx.add(Expr::Neg(inner))
    };
    let stripped_call = DefiniteIntegralCall {
        target: stripped,
        var_expr: call.var_expr,
        var_name: call.var_name.clone(),
        lower: call.lower,
        upper: call.upper,
    };
    definite_integration_rewrite(ctx, &stripped_call)
}

/// `N/(a² − x²)` (constant `N`, `a² > 0`) integrated over an interval strictly
/// OUTSIDE `(−a, a)`. The elementary antiderivative is `atanh(x/a)/a`, real only
/// on `(−a, a)`, so the FTC path attaches the domain condition `−a < x < a` and
/// declines beyond it. But `1/(a² − x²) = −1/(x² − a²)`, whose log-form
/// antiderivative `(1/2a)·ln|(x−a)/(x+a)|` is real on the whole pole-free
/// interval — and the ordinary rational FTC machinery already evaluates that
/// sibling correctly. Flip the sign shape into the `x² − a²` orientation
/// (`N/(c0+c2·x²) = −N/((−c2)·x² − c0)`, an identity) and delegate.
///
/// Gated to intervals strictly outside `(−a, a)` (compared via squares, exact
/// for irrational `a = √(a²)`): an interval inside keeps the elegant `atanh`
/// form, and one straddling a pole still resolves to `undefined` downstream.
fn atanh_form_outside_domain_definite_rewrite(
    ctx: &mut Context,
    call: &DefiniteIntegralCall,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> Option<Rewrite> {
    let (DefiniteBound::Finite(lo), DefiniteBound::Finite(hi)) = (lower_bound, upper_bound) else {
        return None;
    };
    let lo_r = lo.as_pure_rational()?.clone();
    let hi_r = hi.as_pure_rational()?.clone();
    let (low_r, high_r) = if lo_r <= hi_r {
        (lo_r, hi_r)
    } else {
        (hi_r, lo_r)
    };

    // Integrand `N / D` with a var-free constant `N` and `D` a quadratic
    // `c0 + c2·x²` (no linear term), a positive constant minus a positive
    // multiple of `x²` — the `a² − x²` shape that yields the `atanh` form.
    let Expr::Div(num, den) = ctx.get(call.target).clone() else {
        return None;
    };
    as_rational_const(ctx, num)?;
    let den_poly = Polynomial::from_expr(ctx, den, &call.var_name).ok()?;
    if den_poly.degree() != 2 {
        return None;
    }
    let coeff = |i: usize| {
        den_poly
            .coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };
    let c0 = coeff(0);
    let c1 = coeff(1);
    let c2 = coeff(2);
    if !c1.is_zero() || !c2.is_negative() || !c0.is_positive() {
        return None;
    }
    // D = c0 + c2·x² = (−c2)·(a² − x²), with a² = c0 / (−c2) > 0.
    let neg_c2 = -c2;
    let a_squared = &c0 / &neg_c2;
    // Interval strictly outside (−a, a): low > a OR high < −a, via squares so
    // the check stays exact when `a` is irrational. Being outside also
    // guarantees the interval is pole-free (the only poles sit at ±a).
    let outside_right = low_r.is_positive() && &low_r * &low_r > a_squared;
    let outside_left = high_r.is_negative() && &high_r * &high_r > a_squared;
    if !(outside_right || outside_left) {
        return None;
    }

    // Build `−N / ((−c2)·x² − c0)` directly (a subtraction, positive-leading —
    // structurally the `x² − a²` sibling, so it will NOT collapse back to the
    // `atanh` orientation) and delegate to the ordinary FTC path.
    let two = ctx.num(2);
    let x_squared = ctx.add(Expr::Pow(call.var_expr, two));
    let lead_coeff = ctx.add(Expr::Number(neg_c2));
    let lead = ctx.add(Expr::Mul(lead_coeff, x_squared));
    let c0_expr = ctx.add(Expr::Number(c0));
    let new_den = ctx.add(Expr::Sub(lead, c0_expr));
    let new_num = ctx.add(Expr::Neg(num));
    let flipped = ctx.add(Expr::Div(new_num, new_den));
    let delegated = DefiniteIntegralCall {
        target: flipped,
        var_expr: call.var_expr,
        var_name: call.var_name.clone(),
        lower: call.lower,
        upper: call.upper,
    };
    definite_integration_rewrite(ctx, &delegated)
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
        // An EVEN integrand over a strictly-NEGATIVE interval reflects to the positive
        // branch: `f(-x) = f(x)` ⇒ `∫_a^b f = ∫_{-b}^{-a} f`. This moves the integral to
        // where a real antiderivative that only covers `x ≥ a` applies — e.g.
        // `integrate(√(x²-1), x, -3, -2) = integrate(√(x²-1), x, 2, 3)` (the antiderivative
        // uses `acosh`, real only for `arg ≥ 1`, so the negative branch otherwise declines).
        let strictly_negative = |e: &Endpoint| {
            e.pi_multiple.is_zero() && e.e_multiple.is_zero() && e.rational.is_negative()
        };
        if strictly_negative(lower)
            && strictly_negative(upper)
            && parity_in_var(ctx, call.target, &call.var_name) == Some(VarParity::Even)
        {
            let reflected = DefiniteIntegralCall {
                target: call.target,
                var_expr: call.var_expr,
                var_name: call.var_name.clone(),
                lower: ctx.add(Expr::Neg(call.upper)),
                upper: ctx.add(Expr::Neg(call.lower)),
            };
            if let Some(rewrite) = definite_integration_rewrite(ctx, &reflected) {
                return Some(rewrite);
            }
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

    // |polynomial| has no single elementary antiderivative across a sign change, so split it at
    // the real roots before the FTC attempt (the absolute value is continuous, no certificate
    // needed); a product like x*|x| is not a bare |polynomial| and is left to the odd-symmetry /
    // FTC owners. Irrational breakpoints are out of scope and decline.
    if let Some(rewrite) =
        abs_polynomial_definite_integral_rewrite(ctx, call, &lower_bound, &upper_bound)
    {
        return Some(rewrite);
    }

    // |N/D| where the rational `N/D` keeps ONE sign across the whole interval (no root of `N`, no
    // pole of `D`): the absolute value is redundant, so strip it (negating when `N/D < 0`) and reuse
    // the ordinary rational machinery — `integrate(|1/x|, x, 1, 2) = ln(2)`.
    if let Some(rewrite) =
        abs_sign_definite_rational_definite_integral_rewrite(ctx, call, &lower_bound, &upper_bound)
    {
        return Some(rewrite);
    }

    // `N/(a² − x²)` over an interval entirely OUTSIDE (−a, a): the `atanh`
    // antiderivative is real only inside (−a, a), so the FTC path declines
    // there; the equal `−N/(x² − a²)` has a real log antiderivative off the
    // poles. Rewrite and delegate (gated to outside; inside keeps `atanh`).
    if let Some(rewrite) =
        atanh_form_outside_domain_definite_rewrite(ctx, call, &lower_bound, &upper_bound)
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
    let mut antiderivative_was_held = false;
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, antiderivative);
        if unwrapped == antiderivative {
            break;
        }
        antiderivative_was_held = true;
        antiderivative = unwrapped;
    }

    // A rationalization step (`1/(√x·(1+x)) → (√x³−√x)/(x³−x)`) can invent a SPURIOUS denominator
    // root where the numerator also vanishes (here `x = 1`) — a removable singularity, not a pole.
    // Divide such removable roots out of the integrand before the pole certificate so they do not
    // fabricate a false `undefined` on a convergent/regular integral.
    let in_closed_interval = |r: &BigRational| {
        !matches!(
            root_position(&interval_low, &interval_high, r),
            None | Some(RootPosition::Outside)
        )
    };
    let certify_target = reduce_removable_quotient_poles(
        ctx,
        call.target,
        &call.var_name,
        call.var_expr,
        antiderivative,
        &in_closed_interval,
    );
    let conditions_and_integrand = combine_certificates(
        certify_interval(
            ctx,
            &conditions,
            &call.var_name,
            &interval_low,
            &interval_high,
        ),
        integrand_risks_certified(
            ctx,
            certify_target,
            &call.var_name,
            &interval_low,
            &interval_high,
        ),
    );
    // The antiderivative may introduce an `acosh` whose real domain is narrower
    // than the integrand's; refuse to substitute a bound where it would be complex.
    let with_acosh = combine_certificates(
        conditions_and_integrand,
        antiderivative_acosh_domain_certificate(
            ctx,
            antiderivative,
            &call.var_name,
            &interval_low,
            &interval_high,
        ),
    );
    // A Weierstrass-substitution antiderivative is DISCONTINUOUS at the poles of
    // its tan(k·x/2) even where the integrand is smooth; naive FTC across such a
    // jump is a wrong value, so decline unless every carrier is zero-free.
    match combine_certificates(
        with_acosh,
        antiderivative_trig_pole_certificate(
            ctx,
            antiderivative,
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
    let mut result = ctx.add(Expr::Sub(at_upper, at_lower));
    // A HOLD-wrapped antiderivative (the algorithmic backend's contract: its
    // canonical surd/root_sum render must not be re-folded) needs the SAME
    // protection on the substituted difference — without it the simplifier
    // mangles the mixed SURD log/arctan constants (observed: the x^4-4
    // definite lost its arctan term entirely, a wrong value). Surd-free
    // backend results (atanh/log of rational points) fold CORRECTLY and are
    // pinned pre-folded, so only re-wrap when a surd or root_sum survives in
    // the substituted difference.
    if antiderivative_was_held && contains_surd_or_root_sum(ctx, result) {
        result = cas_ast::hold::wrap_hold(ctx, result);
    }
    Some(Rewrite::new(result).desc("integrate(f, x, a, b)"))
}

/// True when the expression contains a `sqrt`/`cbrt` call or a `root_sum`
/// node — the shapes whose substituted definite differences the simplifier
/// is known to mangle without the `__hold` barrier.
fn contains_surd_or_root_sum(ctx: &Context, root: ExprId) -> bool {
    // Only the MIXED forms need the barrier: a root_sum, or ln/arctan
    // coexisting with sqrt/cbrt (the x^4-4 mangling shape). Plain surd
    // powers (arclength) and surd-free atanh/ln fold correctly unwrapped.
    let mut has_surd = false;
    let mut has_log_or_arctan = false;
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if ctx.sym_name(*fn_id) == "root_sum" {
                    return true;
                }
                match ctx.builtin_of(*fn_id) {
                    Some(cas_ast::BuiltinFn::Sqrt) | Some(cas_ast::BuiltinFn::Cbrt) => {
                        has_surd = true;
                    }
                    Some(cas_ast::BuiltinFn::Ln) | Some(cas_ast::BuiltinFn::Arctan) => {
                        has_log_or_arctan = true;
                    }
                    _ => {}
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
    has_surd && has_log_or_arctan
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
        // Poles at BOTH endpoints: the antiderivative diverges at each. Equal signs
        // give an `inf - inf` indeterminate => the (doubly) improper integral
        // diverges with no value (undefined); opposite signs give a definite
        // `±infinity` (consistent with the single-endpoint-pole convention below).
        (Some(us), Some(ls)) => {
            if us == ls {
                ctx.add(Expr::Constant(Constant::Undefined))
            } else {
                build_signed_infinity(ctx, us)
            }
        }
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
    // Drop removable singularities (`(√x³−√x)/(x³−x)`'s spurious `x=1`, or the convergent `√x`
    // boundary at 0) before the pole certificate — a root the continuous antiderivative bridges does
    // not diverge the improper integral. A root is in range when it sits in the closed interval,
    // treating an infinite end as unbounded on that side.
    let in_closed_interval = |r: &BigRational| {
        let r_endpoint = Endpoint::from_rational(r.clone());
        let lower_ok = match lower_bound {
            DefiniteBound::NegInfinity => true,
            DefiniteBound::Finite(low) => matches!(
                low.try_cmp(&r_endpoint),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            _ => false,
        };
        let upper_ok = match upper_bound {
            DefiniteBound::PosInfinity => true,
            DefiniteBound::Finite(high) => matches!(
                r_endpoint.try_cmp(high),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            _ => false,
        };
        lower_ok && upper_ok
    };
    let certify_target = reduce_removable_quotient_poles(
        ctx,
        call.target,
        &call.var_name,
        call.var_expr,
        antiderivative,
        &in_closed_interval,
    );
    match combine_certificates(
        certify_unbounded_interval(ctx, &conditions, &call.var_name, lower_bound, upper_bound),
        integrand_risks_certified_unbounded(
            ctx,
            certify_target,
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

    // Strip ALL holds, not just a root one: a partial-fraction antiderivative can wrap a SUBTREE
    // (`Add(__hold(−½·arctan x − ¼·ln(x²+1)), ½·ln|x−1|)`), and a surviving inner `__hold` blocks the
    // boundary limit (the log/arctan term collector cannot see through it), residualizing a convergent
    // improper integral whose value is otherwise computable.
    let antiderivative = cas_ast::hold::strip_all_holds(ctx, antiderivative);

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
    nonzero_poly_on_interval(&poly, lower_bound, upper_bound)
}

/// Whether the (engine-EXPANDED) denominator polynomial is provably nonzero across the interval.
/// Degree 0/1/2 are decided EXACTLY from the coefficients and the bounds; a higher-degree polynomial
/// is split via its RATIONAL roots (`factor_rational_roots`) and each factor checked, so
/// `1/(x^3-x)`, `1/(x^4-1)`, `1/((x^2-1)(x^2-4))` (which the engine expands to a single polynomial)
/// certify. A leftover factor of degree ≥3 with no rational root stays `Unknown` (its irrational real
/// roots are not located). `combine_certificates` gives the product semantics: a provable interior
/// root (`Undefined`) dominates, an undecided factor (`Unknown`) blocks certification.
fn nonzero_poly_on_interval(
    poly: &Polynomial,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
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
            let a0 = poly.coeffs[2].clone();
            let b0 = poly.coeffs[1].clone();
            let c0 = poly.coeffs[0].clone();
            let four = BigRational::from_integer(4.into());
            let discriminant = &b0 * &b0 - &four * &a0 * &c0;
            if discriminant.is_negative() {
                // No real roots: the quadratic never vanishes anywhere.
                return IntervalCertificate::Certified;
            }
            // Real roots r1 ≤ r2 exist. Normalize to a > 0 (`q` nonzero ⟺ `−q` nonzero), so
            // `q < 0` strictly between the roots and `q > 0` outside `[r1, r2]`. The quadratic is
            // nonzero on the interval iff the interval lies entirely OUTSIDE `[r1, r2]` — decided
            // EXACTLY (no surds) from the vertex `x_v = −b/2a` and the sign of `q` at the finite
            // endpoint (`lo > r2 ⟺ q(lo) > 0 ∧ lo > x_v`). The TAIL convergence is NOT decided here:
            // a slowly-decaying but pole-free integrand still certifies and its boundary limit then
            // reports the honest `±∞` divergence. `∫_2^∞ 1/(x²−1)`: roots ±1 ∉ [2,∞) → certified.
            let (a, b, c) = if a0.is_negative() {
                (-a0, -b0, -c0)
            } else {
                (a0, b0, c0)
            };
            let two = BigRational::from_integer(2.into());
            let vertex = -&b / (&two * &a);
            let q_at = |x: &BigRational| &a * x * x + &b * x + &c;
            match (lower_bound, upper_bound) {
                // [lo, ∞): nonzero ⟺ lo > r2 ⟺ q(lo) > 0 ∧ lo > x_v.
                (DefiniteBound::Finite(lo_ep), DefiniteBound::PosInfinity) => {
                    let Some(lo) = lo_ep.as_pure_rational() else {
                        return IntervalCertificate::Unknown;
                    };
                    let q_lo = q_at(lo);
                    if q_lo.is_zero() {
                        IntervalCertificate::Unknown // root exactly at the boundary: defer
                    } else if q_lo.is_positive() && *lo > vertex {
                        IntervalCertificate::Certified
                    } else {
                        IntervalCertificate::Undefined // a pole lies in [lo, ∞): diverges
                    }
                }
                // (-∞, hi]: nonzero ⟺ hi < r1 ⟺ q(hi) > 0 ∧ hi < x_v.
                (DefiniteBound::NegInfinity, DefiniteBound::Finite(hi_ep)) => {
                    let Some(hi) = hi_ep.as_pure_rational() else {
                        return IntervalCertificate::Unknown;
                    };
                    let q_hi = q_at(hi);
                    if q_hi.is_zero() {
                        IntervalCertificate::Unknown
                    } else if q_hi.is_positive() && *hi < vertex {
                        IntervalCertificate::Certified
                    } else {
                        IntervalCertificate::Undefined
                    }
                }
                // (-∞, ∞) with real roots: poles are inside.
                (DefiniteBound::NegInfinity, DefiniteBound::PosInfinity) => {
                    IntervalCertificate::Undefined
                }
                // Symbolic bounds, or a finite-finite interval (handled by the proper-definite
                // path): stay conservative.
                _ => IntervalCertificate::Unknown,
            }
        }
        // Degree ≥ 3: split off the RATIONAL-root factors and check each (linear/quadratic decided
        // exactly); a leftover degree-≥3 factor is decided by the EXACT interval Sturm count
        // (G1 E-iv-d3 — this is what opens definite integrals over the algorithmic-backend
        // antiderivatives, root_sum included). The product is nonzero iff every factor is, and a
        // factor that vanishes inside makes the integral diverge.
        _ => {
            let factors = poly.factor_rational_roots();
            if factors.len() <= 1 {
                return sturm_nonzero_certificate(poly, lower_bound, upper_bound);
            }
            let mut outcome = IntervalCertificate::Certified;
            for factor in &factors {
                let certificate = if factor.degree() >= 3 {
                    sturm_nonzero_certificate(factor, lower_bound, upper_bound)
                } else {
                    nonzero_poly_on_interval(factor, lower_bound, upper_bound)
                };
                outcome = combine_certificates(outcome, certificate);
            }
            outcome
        }
    }
}

/// EXACT nonzero-on-interval certificate for an arbitrary-degree polynomial
/// via the interval Sturm count (`count_real_roots_in_interval`,
/// BigRational throughout — floats never decide). Semi-infinite rays are
/// reduced to a finite interval through the Cauchy root bound
/// `M = 1 + max|c_i|/|c_n|` (every real root lies in `[-M, M]`). A root
/// EXACTLY at a finite endpoint stays `Unknown` (deferred, mirroring the
/// quadratic case); a root strictly inside is an honest `Undefined`
/// (the integral diverges at that pole).
fn sturm_nonzero_certificate(
    poly: &Polynomial,
    lower_bound: &DefiniteBound,
    upper_bound: &DefiniteBound,
) -> IntervalCertificate {
    use num_traits::One;
    let cauchy_bound = || -> BigRational {
        let lead = poly.leading_coeff();
        let mut max = BigRational::zero();
        for c in poly.coeffs.iter().take(poly.degree()) {
            let ratio = (c / &lead).abs();
            if ratio > max {
                max = ratio;
            }
        }
        max + BigRational::one()
    };
    let finite_rational = |bound: &DefiniteBound| -> Option<BigRational> {
        match bound {
            DefiniteBound::Finite(endpoint) => endpoint.as_pure_rational().cloned(),
            _ => None,
        }
    };
    let (lo, hi) = match (lower_bound, upper_bound) {
        (DefiniteBound::NegInfinity, DefiniteBound::PosInfinity) => {
            return if poly.count_real_roots() == 0 {
                IntervalCertificate::Certified
            } else {
                IntervalCertificate::Undefined
            };
        }
        (DefiniteBound::Finite(_), DefiniteBound::PosInfinity) => {
            let Some(lo) = finite_rational(lower_bound) else {
                return IntervalCertificate::Unknown;
            };
            let bound = cauchy_bound();
            if lo > bound {
                return IntervalCertificate::Certified;
            }
            (lo, bound)
        }
        (DefiniteBound::NegInfinity, DefiniteBound::Finite(_)) => {
            let Some(hi) = finite_rational(upper_bound) else {
                return IntervalCertificate::Unknown;
            };
            let bound = -cauchy_bound();
            if hi < bound {
                return IntervalCertificate::Certified;
            }
            (bound, hi)
        }
        (DefiniteBound::Finite(_), DefiniteBound::Finite(_)) => {
            let (Some(lo), Some(hi)) = (finite_rational(lower_bound), finite_rational(upper_bound))
            else {
                return IntervalCertificate::Unknown;
            };
            if lo > hi {
                return IntervalCertificate::Unknown;
            }
            (lo, hi)
        }
        _ => return IntervalCertificate::Unknown,
    };
    // A root exactly at a finite endpoint: defer (boundary semantics belong
    // to other certificates), matching the quadratic arm's discipline.
    if poly.eval(&lo).is_zero() || poly.eval(&hi).is_zero() {
        return IntervalCertificate::Unknown;
    }
    if poly.count_real_roots_in_interval(&lo, &hi) == 0 {
        IntervalCertificate::Certified
    } else {
        IntervalCertificate::Undefined
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

/// Collect the arguments of every `acosh(·)` subterm of `expr`.
fn collect_acosh_args(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr).clone() {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_acosh_args(ctx, l, out);
            collect_acosh_args(ctx, r, out);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => collect_acosh_args(ctx, inner, out),
        Expr::Function(fn_id, args) => {
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(cas_ast::BuiltinFn::Acosh) {
                out.push(args[0]);
            }
            for arg in args {
                collect_acosh_args(ctx, arg, out);
            }
        }
        _ => {}
    }
}

/// The antiderivative the FTC step substitutes may carry an `acosh` term whose
/// real domain (`arg >= 1`) is NARROWER than the integrand's, so a bound outside
/// that domain yields a NON-REAL boundary value: `integrate(sqrt(x^2-1), x, -3, -2)`
/// has the antiderivative `(x*sqrt(x^2-1) - acosh(x))/2` and used to report
/// `1/2*acosh(-3) + ...` — `acosh(-3)` is complex. The integral itself is finite
/// (≈ 2.2877), so this is NOT a divergence; we DECLINE (Unknown -> residual), never
/// claim Undefined. (`integrate(sqrt(x^2-1), x, 2, 3)` keeps evaluating, and the
/// `arg = 1` endpoint touch in `[1, 2]` certifies because `acosh(1) = 0` is real.)
fn antiderivative_acosh_domain_certificate(
    ctx: &mut Context,
    antiderivative: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    let mut acosh_args = Vec::new();
    collect_acosh_args(ctx, antiderivative, &mut acosh_args);
    let mut outcome = IntervalCertificate::Certified;
    for arg in acosh_args {
        let one = ctx.num(1);
        let slack = ctx.add(Expr::Sub(arg, one));
        // acosh is real exactly where `arg - 1 >= 0`; an endpoint touch (`arg = 1`)
        // is still real (`acosh(1) = 0`). Anything else (proven `< 1`, or
        // unprovable) means our acosh antiderivative is inapplicable -> decline.
        let cert = match positive_on_interval(ctx, slack, var_name, interval_low, interval_high) {
            IntervalCertificate::Certified | IntervalCertificate::BoundaryTouch { .. } => {
                IntervalCertificate::Certified
            }
            _ => IntervalCertificate::Unknown,
        };
        outcome = combine_certificates(outcome, cert);
    }
    outcome
}

/// When the integrand is `N/D` with a polynomial `D`, a SIMPLE interior root `r > 0` of `D` where
/// the numerator `N` ALSO vanishes is a REMOVABLE singularity: the integrand has a finite two-sided
/// limit there (`N` supplies the `(x − r)` that cancels `D`'s simple zero), so it must NOT make the
/// definite integral undefined. This arises when a rationalization step turns `1/(√x·(1+x))` into
/// `(√x³ − √x)/(x³ − x)`, inventing a spurious pole at `x = 1` (where the numerator is also 0).
/// Divide each removable root out of `D` for the pole certificate; genuine poles (`N(r) ≠ 0`) and
/// higher-multiplicity roots are kept. Returns the integrand unchanged when nothing is removable.
fn reduce_removable_quotient_poles(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    var_expr: ExprId,
    antiderivative: ExprId,
    in_closed_interval: &dyn Fn(&BigRational) -> bool,
) -> ExprId {
    let Expr::Div(numerator, denominator) = ctx.get(target).clone() else {
        return target;
    };
    let Ok(den_poly) = Polynomial::from_expr(ctx, denominator, var_name) else {
        return target;
    };
    if den_poly.degree() < 1 {
        return target;
    }
    let factors = den_poly.factor_rational_roots();
    let linear_roots: Vec<BigRational> = factors
        .iter()
        .filter(|f| f.degree() == 1)
        .map(|f| -&f.coeffs[0] / &f.coeffs[1])
        .collect();

    let mut removable: Vec<BigRational> = Vec::new();
    for root in &linear_roots {
        // Only a SIMPLE root (multiplicity 1) can be cancelled by a first-order numerator zero.
        if linear_roots.iter().filter(|r| *r == root).count() != 1 {
            continue;
        }
        // A root strictly OUTSIDE the closed interval never affects the integral; only one inside it
        // (interior, or a boundary the antiderivative bridges) can fabricate the false undefined.
        if !in_closed_interval(root) {
            continue;
        }
        // The (condition-free, continuous) antiderivative stays FINITE across a removable
        // singularity but blows up at a genuine pole (`2·arctan(√1)` is finite; `ln|x−1| → −∞`).
        // A finite `F(root)` certifies the root is removable, so it must not divergence-reject the
        // integral; a non-finite `F(root)` is a real pole and is KEPT.
        let root_expr = ctx.add(Expr::Number(root.clone()));
        let f_at_root = cas_ast::substitute_expr_by_id(ctx, antiderivative, var_expr, root_expr);
        if !boundary_is_genuinely_nonfinite(ctx, f_at_root) && !removable.contains(root) {
            removable.push(root.clone());
        }
    }
    if removable.is_empty() {
        return target;
    }

    // Rebuild the denominator dropping one `(x − r)` factor per removable (simple) root.
    let mut reduced = Polynomial::one(var_name.to_string());
    let mut to_skip = removable;
    for factor in &factors {
        if factor.degree() == 1 {
            let root = -&factor.coeffs[0] / &factor.coeffs[1];
            if let Some(pos) = to_skip.iter().position(|r| *r == root) {
                to_skip.remove(pos);
                continue;
            }
        }
        reduced = reduced.mul(factor);
    }
    let reduced_expr = reduced.to_expr(ctx);
    ctx.add(Expr::Div(numerator, reduced_expr))
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
/// `Some(m)` iff `expr` is exactly `m·var` for a rational `m`: the bare variable
/// (`m = 1`), `Number·var` / `var·Number`, `var/Number`, or a negation thereof.
fn as_rational_multiple_of_var(ctx: &Context, expr: ExprId, var_name: &str) -> Option<BigRational> {
    let is_var =
        |e: ExprId| matches!(ctx.get(e), Expr::Variable(sym) if ctx.sym_name(*sym) == var_name);
    match ctx.get(expr) {
        Expr::Variable(sym) if ctx.sym_name(*sym) == var_name => {
            Some(BigRational::from_integer(1.into()))
        }
        Expr::Neg(inner) => Some(-as_rational_multiple_of_var(ctx, *inner, var_name)?),
        Expr::Mul(l, r) => {
            let (num, v) = if is_var(*r) {
                (*l, *r)
            } else if is_var(*l) {
                (*r, *l)
            } else {
                return None;
            };
            let _ = v;
            match ctx.get(num) {
                Expr::Number(n) => Some(n.clone()),
                _ => None,
            }
        }
        Expr::Div(l, r) => {
            if !is_var(*l) {
                return None;
            }
            match ctx.get(*r) {
                Expr::Number(n) if !num_traits::Zero::is_zero(n) => {
                    Some(BigRational::from_integer(1.into()) / n)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Collect the trig-pole CARRIERS of an antiderivative: for every `tan(u)` term the
/// carrier is `cos(u)` (built here — the tan pole is exactly `cos(u) = 0`), and for
/// every `Div` denominator, any `sin`/`cos` call factor is a carrier as-is (the
/// `arctan(sin(u)/(c·cos(u)))` presentation of the Weierstrass antiderivative).
fn collect_trig_pole_carriers(ctx: &mut Context, expr: ExprId, out: &mut Vec<ExprId>) {
    fn walk(ctx: &Context, expr: ExprId, tan_args: &mut Vec<ExprId>, den_trigs: &mut Vec<ExprId>) {
        match ctx.get(expr).clone() {
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                walk(ctx, l, tan_args, den_trigs);
                walk(ctx, r, tan_args, den_trigs);
            }
            Expr::Div(num, den) => {
                walk(ctx, num, tan_args, den_trigs);
                walk(ctx, den, tan_args, den_trigs);
                collect_trig_call_factors(ctx, den, den_trigs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => walk(ctx, inner, tan_args, den_trigs),
            Expr::Function(fn_id, args) => {
                if args.len() == 1 && ctx.builtin_of(fn_id) == Some(cas_ast::BuiltinFn::Tan) {
                    tan_args.push(args[0]);
                }
                for arg in args {
                    walk(ctx, arg, tan_args, den_trigs);
                }
            }
            _ => {}
        }
    }
    fn collect_trig_call_factors(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
        match ctx.get(expr).clone() {
            Expr::Mul(l, r) => {
                collect_trig_call_factors(ctx, l, out);
                collect_trig_call_factors(ctx, r, out);
            }
            Expr::Neg(inner) => collect_trig_call_factors(ctx, inner, out),
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(cas_ast::BuiltinFn::Sin) | Some(cas_ast::BuiltinFn::Cos)
                    ) =>
            {
                out.push(expr);
            }
            _ => {}
        }
    }
    let mut tan_args = Vec::new();
    let mut den_trigs = Vec::new();
    walk(ctx, expr, &mut tan_args, &mut den_trigs);
    for arg in tan_args {
        let cos_arg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
        if !out.contains(&cos_arg) {
            out.push(cos_arg);
        }
    }
    for carrier in den_trigs {
        if !out.contains(&carrier) {
            out.push(carrier);
        }
    }
}

/// The Weierstrass route substitutes `t = tan(k·x/2)`, so its antiderivative is
/// DISCONTINUOUS at the poles of tan even though the integrand is smooth there:
/// `∫ 1/(2+cos x) dx = (2/√3)·arctan(tan(x/2)/√3)` jumps by `2π/√3` at every
/// `x = (2m+1)π`. Naive FTC endpoint substitution across such a pole reported `0`
/// over `[0, 2π]` for a strictly positive integrand (or a negative value, or a raw
/// leak). Certify the interval only when EVERY trig-pole carrier of the
/// antiderivative is provably zero-free on it; anything else — a pole inside, at an
/// endpoint, or undecidable — DECLINES to an honest residual (`Unknown`), never
/// `Undefined`: the antiderivative's jump is a representation artifact, the
/// integral itself is finite.
fn antiderivative_trig_pole_certificate(
    ctx: &mut Context,
    antiderivative: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    let mut carriers = Vec::new();
    collect_trig_pole_carriers(ctx, antiderivative, &mut carriers);
    for carrier in carriers {
        match trig_nonzero_on_interval(ctx, carrier, var_name, interval_low, interval_high) {
            Some(IntervalCertificate::Certified) => {}
            _ => return IntervalCertificate::Unknown,
        }
    }
    IntervalCertificate::Certified
}

fn trig_nonzero_on_interval(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> Option<IntervalCertificate> {
    let (builtin, scale) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            // Accept `m·x` arguments with a nonzero RATIONAL `m` (`cos(x/2)` from the
            // Weierstrass antiderivative): the zeros of `trig(m·x)` are those of
            // `trig(|m|·x)` (cos is even, sin is odd through zero), at
            // `x = multiplier·π/|m|` — still π-pure rationals, comparisons stay exact.
            let scale = as_rational_multiple_of_var(ctx, args[0], var_name)?;
            if num_traits::Zero::is_zero(&scale) {
                return None;
            }
            (ctx.builtin_of(*fn_id)?, num_traits::Signed::abs(&scale))
        }
        _ => return None,
    };
    let half = BigRational::new(1.into(), 2.into());
    // Zeros at (multiplier / scale) * pi with multiplier in the arithmetic
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

    // Multiplier window covering the interval generously (interval scaled by
    // `scale` so the multiplier progression covers `scale·x/π`).
    let approx_low =
        &interval_low.enclosure().0 * &scale / &pi_high - BigRational::from_integer(2.into());
    let approx_high =
        &interval_high.enclosure().1 * &scale / &pi_low + BigRational::from_integer(2.into());
    let k_low = approx_low.floor().to_integer();
    let k_high = approx_high.ceil().to_integer();

    // Scan EVERY zero in the closed interval rather than returning at the first.
    // A single boundary zero is a removable/improper endpoint, but a second zero
    // (an interior pole, or a pole at BOTH endpoints) makes the integral divergent;
    // returning early on the first endpoint zero misclassified those (e.g. integral
    // of cos/sin^2 over [0, 2*pi] missed the interior pole at pi, and over [0, pi]
    // saw only the lower pole and fabricated a one-sided signed infinity).
    let mut interior_pole = false;
    let mut undecidable = false;
    let mut touch_lower = false;
    let mut touch_upper = false;
    let mut k = k_low;
    while k <= k_high {
        let multiplier = &start + &step * BigRational::from_integer(k.clone());
        // The zero is exactly (multiplier / scale) * pi: pi-pure, so comparisons
        // against pi-multiple bounds are exact rational comparisons.
        let zero = Endpoint::from_pi_multiple(multiplier / &scale);
        let cmp_low = zero.try_cmp(interval_low);
        let cmp_high = zero.try_cmp(interval_high);
        let before_interval = matches!(cmp_low, Some(std::cmp::Ordering::Less));
        let after_interval = matches!(cmp_high, Some(std::cmp::Ordering::Greater));
        if !(before_interval || after_interval) {
            if matches!(cmp_low, Some(std::cmp::Ordering::Greater))
                && matches!(cmp_high, Some(std::cmp::Ordering::Less))
            {
                interior_pole = true;
            } else if matches!(cmp_low, Some(std::cmp::Ordering::Equal)) {
                touch_lower = true;
            } else if matches!(cmp_high, Some(std::cmp::Ordering::Equal)) {
                touch_upper = true;
            } else {
                undecidable = true;
            }
        }
        k += 1;
    }
    if interior_pole {
        return Some(IntervalCertificate::Undefined);
    }
    if undecidable {
        return Some(IntervalCertificate::Unknown);
    }
    if touch_lower || touch_upper {
        return Some(IntervalCertificate::BoundaryTouch {
            lower: touch_lower,
            upper: touch_upper,
        });
    }
    Some(IntervalCertificate::Certified)
}

/// Decide whether the real roots of an irreducible-over-ℚ quadratic `a·x² + b·x + c` (discriminant
/// ≥ 0, so the roots are IRRATIONAL) avoid the closed interval — without ever computing the surds.
/// `g` is evaluated only at the rational endpoints and the vertex `-b/(2a)`: a sign change between
/// the endpoints means exactly one root strictly inside (a pole → divergent); equal endpoint signs
/// with the vertex strictly inside and `g(vertex)` of the opposite sign means BOTH roots are inside;
/// otherwise the interval is root-free. Returns `Unknown` if either endpoint carries a `pi`/`e` part
/// (the surd-vs-irrational comparison is then out of scope). All arithmetic is exact `BigRational`.
fn quadratic_real_roots_clear_of_interval(
    a: &BigRational,
    b: &BigRational,
    c: &BigRational,
    interval_low: &Endpoint,
    interval_high: &Endpoint,
) -> IntervalCertificate {
    let (Some(lo), Some(hi)) = (
        interval_low.as_pure_rational(),
        interval_high.as_pure_rational(),
    ) else {
        return IntervalCertificate::Unknown;
    };
    let g = |x: &BigRational| -> BigRational { (a * x + b) * x + c };
    let g_lo = g(lo);
    let g_hi = g(hi);
    // The roots are irrational, so `g` at a rational point is never zero. A defensive zero (a root
    // that is in fact rational) means a root sits on the interval — decline rather than miscertify.
    if g_lo.is_zero() || g_hi.is_zero() {
        return IntervalCertificate::Unknown;
    }
    if g_lo.is_positive() != g_hi.is_positive() {
        return IntervalCertificate::Undefined; // one root strictly inside
    }
    let two_a = BigRational::from_integer(2.into()) * a;
    let vertex = -b / &two_a;
    if &vertex > lo && &vertex < hi {
        let g_vertex = g(&vertex);
        // Equal endpoint signs but an interior dip past zero ⟹ both roots inside.
        if g_vertex.is_zero() || g_vertex.is_positive() != g_lo.is_positive() {
            return IntervalCertificate::Undefined;
        }
    }
    IntervalCertificate::Certified
}

/// A denominator provably positive OR negative for EVERY real value of the
/// variable (`e^x + 1`, `cosh(x) + 2`, `x² + 1`), hence never zero. Uses the
/// shared real-domain sign prover, which decides `e^x > 0`, `cosh ≥ 1`, and sums
/// thereof. Interval-independent (positivity everywhere subsumes any interval).
fn denominator_provably_nonzero_everywhere(ctx: &mut Context, expr: ExprId) -> bool {
    use cas_math::prove_sign::prove_positive_depth_with;
    use cas_math::tri_proof::TriProof;
    const DEPTH: usize = 4;
    let prove_pos = |ctx: &Context, e: ExprId| {
        prove_positive_depth_with(ctx, e, DEPTH, true, |_, _, _| TriProof::Unknown).is_proven()
    };
    if prove_pos(ctx, expr) {
        return true;
    }
    let neg = ctx.add(Expr::Neg(expr));
    prove_pos(ctx, neg)
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
    let poly = match Polynomial::from_expr(ctx, expr, var_name) {
        Ok(poly) => poly,
        Err(_) => {
            // A TRANSCENDENTAL denominator the polynomial path cannot parse
            // (`e^x + 1`, `cosh(x) + 2`): if it is provably positive — or negative —
            // EVERYWHERE, it has no pole anywhere, so the definite FTC path is safe on
            // any interval. The real-domain sign prover decides `e^x > 0`, `cosh ≥ 1`,
            // etc. Without this, `∫ 1/(e^x+1)` (whose antiderivative IS found) leaked a
            // residual because the pole certificate returned Unknown.
            if denominator_provably_nonzero_everywhere(ctx, expr) {
                return IntervalCertificate::Certified;
            }
            return IntervalCertificate::Unknown;
        }
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
        _ => {
            // Degree ≥ 2: factor out the rational roots and locate each relative to the
            // interval, exactly as `positive_on_interval` does. A rational root strictly inside
            // is a pole (divergent → Undefined); a boundary root is a touch; outside is clear.
            // The leftover (no rational roots) certifies only when it has NO real roots — an
            // irreducible quadratic (negative discriminant) or a constant; anything else
            // (irrational real roots that could sit inside) stays Unknown. This lifts the prior
            // blanket Unknown so an EXPANDED real-root denominator (`x²−1`, `x²+x−2`, `x³−x`)
            // certifies just like its factored spelling `(x−1)(x+1)`.
            let mut outcome = IntervalCertificate::Certified;
            for factor in &poly.factor_rational_roots() {
                match factor.degree() {
                    0 => {}
                    1 => {
                        let root = -&factor.coeffs[0] / &factor.coeffs[1];
                        let cert = match root_position(interval_low, interval_high, &root) {
                            Some(RootPosition::Inside) => return IntervalCertificate::Undefined,
                            Some(RootPosition::Outside) => IntervalCertificate::Certified,
                            Some(RootPosition::AtLower) => IntervalCertificate::BoundaryTouch {
                                lower: true,
                                upper: false,
                            },
                            Some(RootPosition::AtUpper) => IntervalCertificate::BoundaryTouch {
                                lower: false,
                                upper: true,
                            },
                            None => return IntervalCertificate::Unknown,
                        };
                        outcome = combine_certificates(outcome, cert);
                    }
                    2 => {
                        let a = &factor.coeffs[2];
                        let b = &factor.coeffs[1];
                        let c = &factor.coeffs[0];
                        let discriminant = b * b - BigRational::from_integer(4.into()) * a * c;
                        if !discriminant.is_negative() {
                            // Irreducible-over-ℚ quadratic with real (irrational) roots: locate
                            // them by EXACT rational sign analysis at the endpoints and vertex.
                            match quadratic_real_roots_clear_of_interval(
                                a,
                                b,
                                c,
                                interval_low,
                                interval_high,
                            ) {
                                IntervalCertificate::Undefined => {
                                    return IntervalCertificate::Undefined
                                }
                                IntervalCertificate::Unknown => {
                                    return IntervalCertificate::Unknown
                                }
                                other => outcome = combine_certificates(outcome, other),
                            }
                        }
                    }
                    // Degree ≥ 3 leftover (no rational roots): the EXACT
                    // interval Sturm count decides (G1 E-iv-d3 — this opens
                    // definite integrals over every algorithmic-backend
                    // antiderivative, root_sum included). Roots at an endpoint
                    // get the same BoundaryTouch semantics as the linear arm;
                    // a root strictly inside is an honest divergence.
                    _ => {
                        let (Some(lo), Some(hi)) = (
                            interval_low.as_pure_rational(),
                            interval_high.as_pure_rational(),
                        ) else {
                            return IntervalCertificate::Unknown;
                        };
                        let at_lower = factor.eval(lo).is_zero();
                        let at_upper = factor.eval(hi).is_zero();
                        let total = factor.count_real_roots_in_interval(lo, hi);
                        let interior =
                            total.saturating_sub(usize::from(at_lower) + usize::from(at_upper));
                        if interior > 0 {
                            return IntervalCertificate::Undefined;
                        }
                        let cert = if at_lower || at_upper {
                            IntervalCertificate::BoundaryTouch {
                                lower: at_lower,
                                upper: at_upper,
                            }
                        } else {
                            IntervalCertificate::Certified
                        };
                        outcome = combine_certificates(outcome, cert);
                    }
                }
            }
            outcome
        }
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
    fn weierstrass_antiderivative_pole_in_interval_declines() {
        // The Weierstrass antiderivative `arctan(tan(x/2)/√3)` JUMPS at x = (2m+1)π
        // even though the integrand is smooth there; FTC across the jump fabricated
        // 0 / negative values / raw leaks. Any trig-pole carrier zero inside (or at
        // an endpoint of) the closed interval must DECLINE to an honest residual
        // (None), never evaluate and never claim undefined — the integral is finite.
        for src in [
            "integrate(1/(2+cos(x)), x, 0, 2*pi)",
            "integrate(1/(2+cos(x)), x, pi/2, 3*pi/2)",
            "integrate(1/(2+sin(x)), x, 0, 2*pi)",
            "integrate(1/(5+4*cos(x)), x, 0, 2*pi)",
            "integrate(1/(2+cos(x)), x, 0, pi)",
            "integrate(1/(2+cos(x)), x, 0, 4*pi)",
        ] {
            assert!(eval_definite(src).is_none(), "{src} must stay residual");
        }
        // Pole-free windows keep evaluating (the scaled-argument extension of
        // trig_nonzero_on_interval certifies cos(x/2) ≠ 0 on them exactly).
        for src in [
            "integrate(1/(2+cos(x)), x, 0, pi/2)",
            "integrate(1/(2+cos(x)), x, -pi/2, pi/2)",
            "integrate(1/(2+cos(x)), x, 0, 1)",
        ] {
            assert!(eval_definite(src).is_some(), "{src} should evaluate");
        }
    }

    #[test]
    fn expanded_real_root_denominator_certifies_like_its_factored_form() {
        // An EXPANDED denominator with rational roots (`x²−1`, `x²+x−2`, `x³−x`) is
        // certified by factoring out the rational roots and locating each — it must now
        // evaluate exactly like its factored spelling `(x−1)(x+1)`, not stay residual.
        for src in [
            "integrate(1/(x^2-1), x, 2, 3)",
            "integrate(1/(x^2+x-2), x, 2, 4)",
            "integrate(1/(x^2-5*x+6), x, 4, 5)",
            "integrate(x/(x^2-1), x, 2, 3)",
            "integrate(1/(x^3-x), x, 2, 3)",
        ] {
            assert!(eval_definite(src).is_some(), "{src} should evaluate");
        }
        // SOUNDNESS: a rational root strictly inside the interval is a pole → divergent
        // → `undefined`, never a fabricated finite value.
        for src in [
            "integrate(1/(x^2-1), x, -2, 2)",  // pole at ±1 inside
            "integrate(1/(x^2+x-2), x, 0, 3)", // pole at 1 inside
            "integrate(1/(x^3-x), x, -2, 2)",  // poles at -1,0,1 inside
        ] {
            assert_eq!(
                eval_definite(src).as_deref(),
                Some("undefined"),
                "{src} has an interior pole"
            );
        }
        // Irrational roots (`x²−2 → ±√2`, golden `x²−x−1 → φ`) are located by exact rational sign
        // analysis at the endpoints/vertex, so a pole-free interval now EVALUATES…
        for src in [
            "integrate(1/(x^2-2), x, 2, 3)",
            "integrate(1/(x^2-3), x, 2, 3)",
            "integrate(1/(x^2-x-1), x, 2, 3)",
            "integrate(x/(x^2-2), x, 2, 3)",
        ] {
            assert!(eval_definite(src).is_some(), "{src} should evaluate");
        }
        // …and an interval straddling an irrational pole stays divergent (√2 ∈ (1,2), φ ∈ (1,2)).
        for src in [
            "integrate(1/(x^2-2), x, 1, 2)",
            "integrate(1/(x^2-3), x, 1, 2)",
            "integrate(1/(x^2-x-1), x, 1, 2)",
        ] {
            assert_eq!(
                eval_definite(src).as_deref(),
                Some("undefined"),
                "{src} straddles an irrational pole"
            );
        }
    }

    #[test]
    fn abs_of_sign_definite_rational_strips_the_absolute_value() {
        // |N/D| with no root of N and no pole of D inside the interval is sign-definite, so the
        // absolute value drops (negating when the integrand is negative) and the ordinary rational
        // machinery evaluates it. (The low-level helper returns the raw FTC substitution, e.g.
        // `ln(|2|) − ln(|1|)`; the full pipeline folds it — the CLI surfaces `ln(2)`.)
        for src in [
            "integrate(abs(1/x), x, 1, 2)",
            "integrate(abs(1/x), x, -2, -1)", // 1/x < 0 here → −1/x
            "integrate(abs(1/(x^2-1)), x, 2, 3)",
            "integrate(abs((x^2-1)/x), x, 2, 3)",
        ] {
            assert!(eval_definite(src).is_some(), "{src} should evaluate");
        }
        // A sign change or a pole inside is out of scope — decline rather than mis-handle.
        for src in [
            "integrate(abs(1/x), x, -1, 2)",      // pole at 0 inside
            "integrate(abs((x-1)/x), x, 0, 2)",   // numerator root 1 and pole 0 inside
            "integrate(abs(1/(x^2-1)), x, 0, 3)", // poles ±1 inside
        ] {
            assert!(eval_definite(src).is_none(), "{src} should decline");
        }
    }

    #[test]
    fn even_integrand_over_negative_interval_reflects_to_positive() {
        // `integrate(sqrt(x^2-1), x)` = (x*sqrt(x^2-1) - acosh(x))/2; `acosh` is real
        // only for arg >= 1, so the FTC on a NEGATIVE interval would substitute
        // `acosh(-3)` (complex). Since the integrand is EVEN, the interval reflects to
        // the positive branch (`∫_{-3}^{-2} = ∫_2^3`) where the antiderivative is real,
        // so it EVALUATES (to the same value as the positive interval), not declines.
        // (The acosh-domain certificate remains as a safety net for any acosh that
        // reaches a negative bound without reflection applying.)
        assert_eq!(
            eval_definite("integrate(sqrt(x^2-1), x, -3, -2)"),
            eval_definite("integrate(sqrt(x^2-1), x, 2, 3)"),
        );
        assert_eq!(
            eval_definite("integrate(sqrt(x^2-4), x, -5, -3)"),
            eval_definite("integrate(sqrt(x^2-4), x, 3, 5)"),
        );
        assert!(eval_definite("integrate(sqrt(x^2-1), x, -3, -2)").is_some());
        // Positive interval and the boundary touch (`acosh(1) = 0`) still evaluate.
        assert!(eval_definite("integrate(sqrt(x^2-1), x, 2, 3)").is_some());
        assert!(eval_definite("integrate(sqrt(x^2-1), x, 1, 2)").is_some());
        // Even polynomial over a negative interval: same value via reflection.
        assert_eq!(
            eval_definite("integrate(x^2, x, -3, -2)"),
            eval_definite("integrate(x^2, x, 2, 3)"),
        );
        assert!(eval_definite("integrate(x^2, x, 0, 2)").is_some());
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
    fn abs_polynomial_definite_integral_splits_at_real_roots() {
        // |g| over rational bounds, split at every real root of g inside the interval.
        for (source, expected) in [
            ("integrate(abs(x^2-1), x, 0, 2)", "2"),
            ("integrate(abs(x^2-4), x, 0, 3)", "23/3"),
            ("integrate(abs(x^2-3*x+2), x, 0, 3)", "11/6"),
            ("integrate(abs(x^3-x), x, -2, 2)", "5"),
            ("integrate(abs(x^2-1), x, 2, 0)", "-2"), // reversed orientation
            // Irreducible quadratic with roots OUTSIDE the interval: no split, ∫|g| = |∫g|.
            ("integrate(abs(x^2-2), x, 3, 4)", "31/3"),
            ("integrate(abs(x^2-2), x, 0, 1)", "5/3"),
        ] {
            assert_eq!(eval_definite(source).as_deref(), Some(expected), "{source}");
        }
        // Irrational root INSIDE the interval (√2 ∈ (0,2)) needs an irrational breakpoint —
        // out of scope, declines honestly rather than mis-splitting.
        assert!(eval_definite("integrate(abs(x^2-2), x, 0, 2)").is_none());
    }

    #[test]
    fn abs_linear_definite_integral_scope() {
        // A product resolves by odd symmetry (-> 0): `x*|x|` is not a bare `|polynomial|`.
        assert_eq!(
            eval_definite("integrate(abs(x)*x, x, -1, 1)").as_deref(),
            Some("0")
        );
        // A bare |linear| with a symbolic bound now resolves via the affine abs
        // antiderivative x|x|/2 and the FTC. This low-level helper returns the raw
        // F(pi) - F(0) substitution; the full eval pipeline folds it to pi^2/2
        // (|pi| = pi, pi > 0), which the CLI surfaces.
        assert_eq!(
            eval_definite("integrate(abs(x), x, 0, pi)").as_deref(),
            Some("1/2 * pi * |pi| - 0 * 1/2 * |0|")
        );
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

    #[test]
    fn poles_at_both_endpoints_and_interior_are_undefined() {
        // Regression for Round-4 Cluster K: the trig zero scan returned on the FIRST
        // zero, so it saw only the lower pole and fabricated a one-sided signed
        // infinity. sin has zeros at 0 AND pi -> BOTH endpoints of [0, pi] are poles
        // of cos/sin^2 (antiderivative -1/sin -> -inf at each): an inf - inf
        // indeterminate, so the integral diverges (undefined).
        assert_eq!(
            eval_definite("integrate(cos(x)/sin(x)^2, x, 0, pi)").as_deref(),
            Some("undefined")
        );
        // Interior pole at pi inside [0, 2pi]: the second zero was missed when the
        // scan short-circuited on the x=0 endpoint zero.
        assert_eq!(
            eval_definite("integrate(cos(x)/sin(x)^2, x, 0, 2*pi)").as_deref(),
            Some("undefined")
        );
        // Poles at both pi and 2pi (the [pi, 2pi] endpoints).
        assert_eq!(
            eval_definite("integrate(cos(x)/sin(x)^2, x, pi, 2*pi)").as_deref(),
            Some("undefined")
        );
        // A single endpoint pole still yields the honest one-sided divergence.
        assert_eq!(
            eval_definite("integrate(cos(x)/sin(x)^2, x, pi/2, pi)").as_deref(),
            Some("-infinity")
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
