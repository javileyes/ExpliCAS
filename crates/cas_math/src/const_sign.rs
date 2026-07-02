//! Exact rational SIGN oracle for variable-free (constant) expressions.
//!
//! Computes VERIFIED rational value bounds `[lo, hi]` (with `lo <= value <= hi`,
//! both `BigRational`) for a constant expression built from rationals, `pi`, `e`,
//! the golden ratio `phi`, `sqrt`, and `+ - * /` / rational powers (integer, and
//! `p/q` over a nonnegative base via n-th-root bounds), and derives a provable
//! sign from them. Transcendental `ln`/`log`/`exp` get cheap exact sign
//! rules (a base `> 1` log is positive iff its argument exceeds 1; `exp` of a real
//! is always positive).
//!
//! EVERYTHING is exact `BigRational` arithmetic -- never an `f64` gate. When the
//! sign cannot be proven the oracle returns `None`, and callers MUST bail rather
//! than guess (the recurring "soundness gates must be exact" discipline: a float
//! comparison near zero can drop a true value or fabricate a false one).

use crate::numeric_eval::as_rational_const;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::str::FromStr;
use std::sync::OnceLock;

/// Process-wide cache for the fixed transcendental-constant bounds. Each entry is a pair of
/// exact `BigRational`s that never change, so the first computation (a 50-digit string parse for
/// `pi`/`e`, or a 60-term atanh series for `ln(2)`/`ln(10)`) is done once and every later call
/// clones the cached value instead of recomputing it — the exact-bounds oracle runs deep in the
/// sign-proof hot loop (P10 of the saneamiento audit).
type ConstBounds = (BigRational, BigRational);
static PI_BOUNDS_CACHE: OnceLock<ConstBounds> = OnceLock::new();
static E_BOUNDS_CACHE: OnceLock<ConstBounds> = OnceLock::new();
static LN2_BOUNDS_CACHE: OnceLock<ConstBounds> = OnceLock::new();
static LN10_BOUNDS_CACHE: OnceLock<ConstBounds> = OnceLock::new();

/// Provable sign of a real constant expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstSign {
    Negative,
    Zero,
    Positive,
}

#[cfg(test)]
fn ratio(numer: i64, denom: i64) -> BigRational {
    BigRational::new(BigInt::from(numer), BigInt::from(denom))
}

// Hand-verified rational bounds `(lower, upper)` to 50 decimal places, so the
// interval width is 1e-50 and the oracle only fails to decide a comparison whose
// threshold is within 1e-50 of the constant (no realistic input). `phi` is derived
// from the arbitrary-precision `sqrt(5)` bounds instead of being hardcoded.
//
// lo = the 50-decimal truncation (< true value); hi = lo + 1e-50 (> true value).
fn bounds_to_50dp(lo_digits: &str) -> (BigRational, BigRational) {
    let denom = BigInt::from(10).pow(50);
    let lo = BigRational::new(BigInt::from_str(lo_digits).unwrap(), denom.clone());
    let hi = &lo + BigRational::new(BigInt::one(), denom);
    (lo, hi)
}
fn pi_bounds() -> (BigRational, BigRational) {
    // pi = 3.14159265358979323846264338327950288419716939937510 5820974944...
    PI_BOUNDS_CACHE
        .get_or_init(|| bounds_to_50dp("314159265358979323846264338327950288419716939937510"))
        .clone()
}
fn e_bounds() -> (BigRational, BigRational) {
    // e  = 2.71828182845904523536028747135266249775724709369995 9574966968...
    E_BOUNDS_CACHE
        .get_or_init(|| bounds_to_50dp("271828182845904523536028747135266249775724709369995"))
        .clone()
}

/// Cached `ln(2)` bounds (the base-2 range-reduction constant used by every `ln_bounds` call).
fn ln2_bounds() -> ConstBounds {
    LN2_BOUNDS_CACHE
        .get_or_init(|| atanh_ln_bounds(&BigRational::from_integer(BigInt::from(2))))
        .clone()
}

/// Cached `ln(10)` bounds (used by the `log10` value-bounds path).
fn ln10_bounds() -> ConstBounds {
    LN10_BOUNDS_CACHE
        .get_or_init(|| {
            ln_bounds(&BigRational::from_integer(BigInt::from(10)))
                .expect("ln(10) bounds are always computable")
        })
        .clone()
}

fn zero() -> BigRational {
    BigRational::zero()
}
fn one() -> BigRational {
    BigRational::one()
}

/// Round `x` UP to a multiple of `1/denom` (keeps an upper bound an upper bound).
fn round_up(x: &BigRational, denom: &BigInt) -> BigRational {
    let scaled = x * BigRational::from_integer(denom.clone());
    scaled.ceil() / BigRational::from_integer(denom.clone())
}

/// Round `x` DOWN to a multiple of `1/denom` (keeps a lower bound a lower bound).
fn round_down(x: &BigRational, denom: &BigInt) -> BigRational {
    let scaled = x * BigRational::from_integer(denom.clone());
    scaled.floor() / BigRational::from_integer(denom.clone())
}

/// Tight rational bounds `(lo, hi)` with `lo <= sqrt(q) <= hi` for `q >= 0`.
///
/// Newton-from-above: `x_{n+1} = (x_n + q/x_n)/2 >= sqrt(q)` by AM-GM, so every
/// iterate (rounded UP to bounded precision) stays an upper bound; the matching
/// lower bound is `q / hi <= sqrt(q)`. Both are exact rationals.
fn sqrt_bounds(q: &BigRational) -> Option<(BigRational, BigRational)> {
    if q.is_negative() {
        return None;
    }
    if q.is_zero() {
        return Some((zero(), zero()));
    }
    // Perfect square => exact rational root (so `sqrt(4) - 2` proves to exactly 0).
    if let Some(exact) = exact_sqrt(q) {
        return Some((exact.clone(), exact));
    }
    let precision = BigInt::from(10).pow(40); // 1/10^40 resolution
    let two = BigRational::from_integer(BigInt::from(2));
    // x0 >= sqrt(q): max(q, 1) works (q >= 1 => q >= sqrt(q); q < 1 => 1 > sqrt(q)).
    let mut hi = if q >= &one() { q.clone() } else { one() };
    for _ in 0..200 {
        let next = round_up(&((&hi + q / &hi) / &two), &precision);
        if next >= hi {
            break; // converged (monotone non-increasing, bounded below by sqrt(q))
        }
        hi = next;
    }
    let lo = q / &hi; // hi >= sqrt(q)  =>  q/hi <= sqrt(q)
    Some((lo, hi))
}

/// Exact rational square root of `q >= 0` when it is rational (both numerator and
/// denominator of the reduced fraction are perfect squares), else `None`.
fn exact_sqrt(q: &BigRational) -> Option<BigRational> {
    let numer = q.numer();
    let denom = q.denom();
    let sn = numer.sqrt();
    let sd = denom.sqrt();
    if &(&sn * &sn) == numer && &(&sd * &sd) == denom {
        Some(BigRational::new(sn, sd))
    } else {
        None
    }
}

/// `x^n` for a small non-negative integer `n` (exact rational).
fn rational_pow_u32(x: &BigRational, n: u32) -> BigRational {
    let mut acc = one();
    for _ in 0..n {
        acc = &acc * x;
    }
    acc
}

/// Exact rational n-th root of `q >= 0` when it is rational (both numerator and
/// denominator of the reduced fraction are perfect n-th powers), else `None`.
fn exact_nth_root(q: &BigRational, n: u32) -> Option<BigRational> {
    let rn = q.numer().nth_root(n);
    let rd = q.denom().nth_root(n);
    let pow_back = |b: &BigInt| -> BigInt {
        let mut acc = BigInt::one();
        for _ in 0..n {
            acc *= b;
        }
        acc
    };
    if &pow_back(&rn) == q.numer() && &pow_back(&rd) == q.denom() {
        Some(BigRational::new(rn, rd))
    } else {
        None
    }
}

/// Tight rational bounds `(lo, hi)` with `lo <= q^(1/n) <= hi` for `q >= 0`, `n >= 1`.
///
/// Newton-from-above on `x^n = q`: `x_{k+1} = ((n-1)·x_k + q/x_k^(n-1))/n >= q^(1/n)`
/// by AM-GM, so every iterate (rounded UP to bounded precision) stays an upper
/// bound; the matching lower bound is `q / hi^(n-1) <= q^(1/n)`. Both are exact
/// rationals; if the loop stops before converging the bounds are LOOSE but still
/// valid (the caller can only fail to decide, never decide wrongly).
fn nth_root_bounds(q: &BigRational, n: u32) -> Option<(BigRational, BigRational)> {
    if n == 0 || q.is_negative() {
        return None;
    }
    if n == 1 || q.is_zero() || q.is_one() {
        return Some((q.clone(), q.clone()));
    }
    if let Some(exact) = exact_nth_root(q, n) {
        return Some((exact.clone(), exact));
    }
    let precision = BigInt::from(10).pow(40); // 1/10^40 resolution
    let n_r = BigRational::from_integer(BigInt::from(n));
    let n_minus_1 = BigRational::from_integer(BigInt::from(n - 1));
    // x0 >= q^(1/n): max(q, 1) works (q >= 1 => q >= q^(1/n); q < 1 => 1 > q^(1/n)).
    // Rounded UP to bounded precision so the first `hi^(n-1)` cannot blow up when
    // `q` carries a huge exact denominator (round-up keeps an upper bound).
    let mut hi = round_up(&if q >= &one() { q.clone() } else { one() }, &precision);
    for _ in 0..400 {
        let next = round_up(
            &((&n_minus_1 * &hi + q / rational_pow_u32(&hi, n - 1)) / &n_r),
            &precision,
        );
        if next >= hi {
            break; // converged (monotone non-increasing, bounded below by q^(1/n))
        }
        hi = next;
    }
    let lo = q / rational_pow_u32(&hi, n - 1); // hi >= q^(1/n)  =>  q/hi^(n-1) <= q^(1/n)
    Some((lo, hi))
}

/// Tight rational bounds `(lo, hi)` with `lo <= ln(m) <= hi` for `m >= 1`, via the
/// inverse-hyperbolic-tangent series `ln(m) = 2·Σ_{k>=0} y^(2k+1)/(2k+1)` with
/// `y = (m-1)/(m+1) ∈ [0,1)`. The partial sum `2·S_N` is a LOWER bound (every
/// omitted term is positive); the tail `R_N = 2·Σ_{k>N} y^(2k+1)/(2k+1)` is bounded
/// above by `2·y^(2N+3) / ((2N+3)·(1-y²))`, an exact rational, giving the UPPER bound.
fn atanh_ln_bounds(m: &BigRational) -> (BigRational, BigRational) {
    if m.is_one() {
        return (zero(), zero());
    }
    let two = BigRational::from_integer(BigInt::from(2));
    let y = (m - one()) / (m + one()); // ∈ (0, 1) for m > 1
    let y2 = &y * &y;
    // N chosen so the tail < 1e-50 even at the worst reduced argument (y = 1/3,
    // i.e. m = 2): (1/3)^(2N+3) < 1e-50  =>  N >= 52. Use 60 for margin.
    const N: i64 = 60;
    let mut term = y.clone(); // y^(2k+1), starts at k=0 (y^1)
    let mut sum = zero();
    let mut k: i64 = 0;
    while k <= N {
        sum = &sum + &term / BigRational::from_integer(BigInt::from(2 * k + 1));
        term = &term * &y2; // advance to y^(2(k+1)+1)
        k += 1;
    }
    // `term` is now y^(2N+3).
    let lo = &two * &sum;
    let rem = &two * &term / (BigRational::from_integer(BigInt::from(2 * N + 3)) * (one() - &y2));
    let hi = &lo + rem;
    (lo, hi)
}

/// Tight rational bounds `(lo, hi)` with `lo <= ln(x) <= hi` for `x > 0`, else `None`
/// (real `ln` is undefined for `x <= 0`). Range-reduces `x = 2^e · m` with `m ∈ [1,2)`
/// so the series argument stays `<= 1/3` (fast convergence): `ln(x) = e·ln(2) + ln(m)`.
fn ln_bounds(x: &BigRational) -> Option<(BigRational, BigRational)> {
    if !x.is_positive() {
        return None;
    }
    if x.is_one() {
        return Some((zero(), zero()));
    }
    if x < &one() {
        // ln(x) = -ln(1/x); 1/x > 1 reduces to the handled branch.
        let (lo, hi) = ln_bounds(&(one() / x))?;
        return Some((-hi, -lo));
    }
    // x >= 1: reduce to m ∈ [1, 2).
    let two = BigRational::from_integer(BigInt::from(2));
    let mut m = x.clone();
    let mut e: i64 = 0;
    while m >= two {
        m = &m / &two;
        e += 1;
        if e > 1_000_000 {
            return None; // defensive: absurdly large argument
        }
    }
    let (l2_lo, l2_hi) = ln2_bounds(); // ln(2), cached
    let (lm_lo, lm_hi) = atanh_ln_bounds(&m); // ln(m), m ∈ [1, 2)
    let e_r = BigRational::from_integer(BigInt::from(e));
    Some((&e_r * &l2_lo + lm_lo, &e_r * &l2_hi + lm_hi))
}

/// `ln` of a positive interval `[al, ah]`. `ln` is monotone increasing, so
/// `ln([al, ah]) = [ln_lower(al), ln_upper(ah)]`; bails if the lower endpoint is
/// not strictly positive (real-domain `ln`).
fn ln_interval_bounds((al, ah): (BigRational, BigRational)) -> Option<(BigRational, BigRational)> {
    if !al.is_positive() {
        return None;
    }
    let (lo, _) = ln_bounds(&al)?;
    let (_, hi) = ln_bounds(&ah)?;
    Some((lo, hi))
}

/// Tight rational bounds `(lo, hi)` with `lo <= sin(c) <= hi` (or `cos`) for a
/// RATIONAL `c` in radians, via the Taylor series truncated with the RIGOROUS
/// Lagrange remainder `|R_n| <= |c|^(n+1)/(n+1)!` (every derivative of sin/cos is
/// bounded by 1 in absolute value, so this holds for ANY real `c` — no
/// alternating-monotone assumption). `is_sin` selects the odd (sin, start degree 1)
/// or even (cos, start degree 0) series. Bails for `|c| > 100` (range reduction
/// would be needed and the factorials get expensive) — sound, just incomplete.
fn trig_taylor_bounds(c: &BigRational, is_sin: bool) -> Option<(BigRational, BigRational)> {
    let abs_c = c.abs();
    if abs_c > BigRational::from_integer(BigInt::from(100)) {
        return None;
    }
    let neg_c2 = -(c * c);
    // First term: sin -> `c` (degree 1); cos -> `1` (degree 0).
    let (mut term, mut deg) = if is_sin {
        (c.clone(), 1i64)
    } else {
        (one(), 0i64)
    };
    let mut sum = term.clone();
    let eps = BigRational::new(BigInt::one(), BigInt::from(10).pow(60));
    for _ in 0..2000 {
        // Lagrange remainder after the degree-`deg` term: |c|^(deg+1)/(deg+1)! =
        // |term| * |c| / (deg+1)  (since |term| = |c|^deg / deg!).
        let rem = &term.abs() * &abs_c / BigRational::from_integer(BigInt::from(deg + 1));
        if rem < eps {
            return Some((&sum - &rem, &sum + &rem));
        }
        // Advance to the next non-zero term: multiply by -c^2 / ((deg+1)(deg+2)).
        let denom = BigRational::from_integer(BigInt::from((deg + 1) * (deg + 2)));
        term = &term * &neg_c2 / denom;
        sum = &sum + &term;
        deg += 2;
    }
    None
}

fn sin_bounds(c: &BigRational) -> Option<(BigRational, BigRational)> {
    trig_taylor_bounds(c, true)
}
fn cos_bounds(c: &BigRational) -> Option<(BigRational, BigRational)> {
    trig_taylor_bounds(c, false)
}

/// A sound INNER bound for `pi/2`: `pi_lo/2 < pi/2`. Comparing a rational endpoint
/// against this proves it is strictly inside `(-pi/2, pi/2)`.
fn half_pi_inner() -> BigRational {
    let (pl, _) = pi_bounds();
    pl / BigRational::from_integer(BigInt::from(2))
}

/// Bounds for `sin` over an argument interval `[al, ah]`. An EXACT rational POINT
/// argument uses the direct Taylor bound (any `|c| <= 100`); a non-degenerate
/// interval is bounded only when it lies PROVABLY inside the principal increasing
/// piece `(-pi/2, pi/2)` (so `sin` is monotone there), else bails — sin is not
/// monotone across an extremum.
fn sin_arg_bounds((al, ah): (BigRational, BigRational)) -> Option<(BigRational, BigRational)> {
    if al == ah {
        return sin_bounds(&al);
    }
    let hp = half_pi_inner();
    let neg_hp = BigRational::zero() - &hp;
    if al >= neg_hp && ah <= hp {
        let (lo, _) = sin_bounds(&al)?; // increasing: sin(al) is the lower end
        let (_, hi) = sin_bounds(&ah)?;
        return Some((lo, hi));
    }
    None
}

/// Bounds for `cos` over `[al, ah]`. Point args use the direct Taylor bound; an
/// interval is bounded only when provably inside the decreasing piece `[0, pi)`.
fn cos_arg_bounds((al, ah): (BigRational, BigRational)) -> Option<(BigRational, BigRational)> {
    if al == ah {
        return cos_bounds(&al);
    }
    let (pl, _) = pi_bounds();
    if !al.is_negative() && ah <= pl {
        let (lo, _) = cos_bounds(&ah)?; // decreasing: cos(ah) is the lower end
        let (_, hi) = cos_bounds(&al)?;
        return Some((lo, hi));
    }
    None
}

/// Bounds for `tan` over `[al, ah]`. Point args divide sin/cos directly; an interval
/// is bounded only when provably inside the increasing pole-free piece `(-pi/2, pi/2)`.
fn tan_arg_bounds((al, ah): (BigRational, BigRational)) -> Option<(BigRational, BigRational)> {
    if al == ah {
        let s = sin_bounds(&al)?;
        let c = cos_bounds(&al)?;
        return Some(interval_mul(s, interval_recip(c)?));
    }
    let hp = half_pi_inner();
    let neg_hp = BigRational::zero() - &hp;
    if al >= neg_hp && ah <= hp {
        // tan increasing on (-pi/2, pi/2): tan(arg) in [tan(al), tan(ah)].
        let lo = interval_mul(sin_bounds(&al)?, interval_recip(cos_bounds(&al)?)?).0;
        let hi = interval_mul(sin_bounds(&ah)?, interval_recip(cos_bounds(&ah)?)?).1;
        return Some((lo, hi));
    }
    None
}

fn interval_neg((lo, hi): (BigRational, BigRational)) -> (BigRational, BigRational) {
    (-hi, -lo)
}
fn interval_add(
    (al, ah): (BigRational, BigRational),
    (bl, bh): (BigRational, BigRational),
) -> (BigRational, BigRational) {
    (al + bl, ah + bh)
}
fn interval_sub(
    (al, ah): (BigRational, BigRational),
    (bl, bh): (BigRational, BigRational),
) -> (BigRational, BigRational) {
    (al - bh, ah - bl)
}
fn interval_mul(
    (al, ah): (BigRational, BigRational),
    (bl, bh): (BigRational, BigRational),
) -> (BigRational, BigRational) {
    let products = [&al * &bl, &al * &bh, &ah * &bl, &ah * &bh];
    let lo = products.iter().min().unwrap().clone();
    let hi = products.iter().max().unwrap().clone();
    (lo, hi)
}
/// Reciprocal of an interval that does NOT contain zero.
fn interval_recip((lo, hi): (BigRational, BigRational)) -> Option<(BigRational, BigRational)> {
    if lo.is_positive() || hi.is_negative() {
        Some((one() / hi, one() / lo))
    } else {
        None // interval brackets 0: reciprocal is unbounded
    }
}

/// Exact rational value bounds `[lo, hi]` for a constant expression, or `None`.
pub fn const_value_bounds(ctx: &Context, expr: ExprId) -> Option<(BigRational, BigRational)> {
    const_value_bounds_depth(ctx, expr, 0)
}

fn const_value_bounds_depth(
    ctx: &Context,
    expr: ExprId,
    depth: u32,
) -> Option<(BigRational, BigRational)> {
    if depth > 64 {
        return None;
    }
    // Any rational subexpression collapses to a point interval.
    if let Some(r) = as_rational_const(ctx, expr) {
        return Some((r.clone(), r));
    }
    let bounds = |e: ExprId| const_value_bounds_depth(ctx, e, depth + 1);
    match ctx.get(expr) {
        Expr::Constant(Constant::Pi) => Some(pi_bounds()),
        Expr::Constant(Constant::E) => Some(e_bounds()),
        Expr::Constant(Constant::Phi) => {
            // phi = (1 + sqrt(5)) / 2, bounded via the arbitrary-precision sqrt.
            let (slo, shi) = sqrt_bounds(&BigRational::from_integer(BigInt::from(5)))?;
            let two = BigRational::from_integer(BigInt::from(2));
            Some(((one() + slo) / &two, (one() + shi) / two))
        }
        Expr::Neg(a) => Some(interval_neg(bounds(*a)?)),
        Expr::Add(a, b) => Some(interval_add(bounds(*a)?, bounds(*b)?)),
        Expr::Sub(a, b) => Some(interval_sub(bounds(*a)?, bounds(*b)?)),
        Expr::Mul(a, b) => Some(interval_mul(bounds(*a)?, bounds(*b)?)),
        Expr::Div(a, b) => Some(interval_mul(bounds(*a)?, interval_recip(bounds(*b)?)?)),
        Expr::Pow(base, exp) => {
            let exp_r = as_rational_const(ctx, *exp)?;
            let base_b = bounds(*base)?;
            interval_pow(base_b, &exp_r)
        }
        Expr::Function(_, args) => {
            // sqrt(a): bounds(a) must be >= 0.
            if ctx.is_builtin_call(expr, BuiltinFn::Sqrt) && args.len() == 1 {
                let (al, ah) = bounds(args[0])?;
                if al.is_negative() {
                    return None;
                }
                let (_, hi) = sqrt_bounds(&ah)?;
                let (lo, _) = sqrt_bounds(&al)?;
                Some((lo, hi))
            } else if ctx.is_builtin_call(expr, BuiltinFn::Ln) && args.len() == 1 {
                ln_interval_bounds(bounds(args[0])?)
            } else if ctx.is_builtin_call(expr, BuiltinFn::Log2) && args.len() == 1 {
                // log2(c) = ln(c) / ln(2); ln(2) > 0 so the reciprocal is well-defined.
                let num = ln_interval_bounds(bounds(args[0])?)?;
                let den = interval_recip(ln2_bounds())?;
                Some(interval_mul(num, den))
            } else if ctx.is_builtin_call(expr, BuiltinFn::Log10) && args.len() == 1 {
                // log10(c) = ln(c) / ln(10).
                let num = ln_interval_bounds(bounds(args[0])?)?;
                let den = interval_recip(ln10_bounds())?;
                Some(interval_mul(num, den))
            } else if ctx.is_builtin_call(expr, BuiltinFn::Log) && args.len() == 2 {
                // log(base, arg) = ln(arg) / ln(base); the base interval must not
                // straddle 1 (else ln(base) brackets 0 and the ratio is unbounded).
                let num = ln_interval_bounds(bounds(args[1])?)?;
                let den = interval_recip(ln_interval_bounds(bounds(args[0])?)?)?;
                Some(interval_mul(num, den))
            } else if ctx.is_builtin_call(expr, BuiltinFn::Sin) && args.len() == 1 {
                // Rational point arg: direct Taylor (any |c| <= 100). Non-degenerate
                // interval (an irrational arg like `sqrt(2)` or `pi/7`): bounded only
                // when provably inside the principal monotone piece (-pi/2, pi/2).
                sin_arg_bounds(bounds(args[0])?)
            } else if ctx.is_builtin_call(expr, BuiltinFn::Cos) && args.len() == 1 {
                cos_arg_bounds(bounds(args[0])?)
            } else if ctx.is_builtin_call(expr, BuiltinFn::Tan) && args.len() == 1 {
                tan_arg_bounds(bounds(args[0])?)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Bounds of `base ^ exp` where `exp` is a rational constant. Handles integer
/// exponents (positive/negative) and, for a NONNEGATIVE base, any small rational
/// exponent `p/q` via `base^(p/q) = (base^|p|)^(1/q)` (reciprocated for `p < 0`);
/// bails on a negative base with a fractional exponent (odd-root semantics are
/// the caller's business, never guessed here).
fn interval_pow(
    base: (BigRational, BigRational),
    exp: &BigRational,
) -> Option<(BigRational, BigRational)> {
    use num_traits::ToPrimitive;
    if exp.is_zero() {
        return Some((one(), one()));
    }
    if !exp.is_integer() {
        // General rational exponent p/q (reduced, q >= 2) over a nonnegative base.
        // Caps at 16: every realistic textbook exponent (1/2..7/2, p/q with small q)
        // fits, and the n-th-root Newton cost stays milliseconds. Larger exponents
        // return None (honest indecision) rather than multi-second exact arithmetic.
        let root: u32 = exp.denom().to_u32().filter(|d| *d <= 16)?;
        let p = exp.numer();
        if p.abs() > BigInt::from(16) {
            return None;
        }
        let (blo, bhi) = base;
        if blo.is_negative() {
            return None;
        }
        // Round the base OUTWARD to bounded precision before powering: a 50-digit
        // bound raised to p=63 otherwise explodes to ~200k-digit rationals inside
        // the n-th-root Newton (multi-minute gcds). Outward rounding only widens
        // the interval — bounds stay valid, decisions stay sound.
        let precision = BigInt::from(10).pow(40);
        let blo = round_down(&blo, &precision);
        let bhi = round_up(&bhi, &precision);
        // The rounded-down lower bound of a nonnegative base can dip below 0; clamp.
        let blo = if blo.is_negative() { zero() } else { blo };
        // base >= 0 and |p| >= 1  =>  x^|p| is monotone nondecreasing on the interval.
        let times = p.abs().to_u32().unwrap_or(0);
        let (rlo, _) = nth_root_bounds(&rational_pow_u32(&blo, times), root)?;
        let (_, rhi) = nth_root_bounds(&rational_pow_u32(&bhi, times), root)?;
        return if p.is_negative() {
            interval_recip((rlo, rhi))
        } else {
            Some((rlo, rhi))
        };
    }
    let n = exp.to_integer();
    // Bound the magnitude of the exponent to avoid blow-up.
    let abs_n = n.abs();
    if abs_n > BigInt::from(64) {
        return None;
    }
    let times = abs_n.to_u32_digits().1.first().copied().unwrap_or(0);
    let mut acc = (one(), one());
    for _ in 0..times {
        acc = interval_mul(acc, base.clone());
    }
    if n.is_negative() {
        interval_recip(acc)
    } else {
        Some(acc)
    }
}

fn sign_of(r: &BigRational) -> ConstSign {
    if r.is_positive() {
        ConstSign::Positive
    } else if r.is_negative() {
        ConstSign::Negative
    } else {
        ConstSign::Zero
    }
}

/// Cheap exact sign for bare transcendental constants that value-bounds cannot
/// reach: a base-`> 1` logarithm (`ln`, `log2`, `log10`) is positive iff its
/// argument exceeds 1; `exp` of any real constant is strictly positive.
fn transcendental_sign(ctx: &Context, expr: ExprId) -> Option<ConstSign> {
    let single_arg = || match ctx.get(expr) {
        Expr::Function(_, args) if args.len() == 1 => Some(args[0]),
        _ => None,
    };
    for log_fn in [BuiltinFn::Ln, BuiltinFn::Log2, BuiltinFn::Log10] {
        if ctx.is_builtin_call(expr, log_fn) {
            let arg = single_arg()?;
            let (lo, hi) = const_value_bounds(ctx, arg)?;
            // Argument must be strictly positive for a real logarithm.
            if !lo.is_positive() {
                return None;
            }
            let one = one();
            if lo > one {
                return Some(ConstSign::Positive);
            }
            if hi < one {
                return Some(ConstSign::Negative);
            }
            if lo == one && hi == one {
                return Some(ConstSign::Zero); // argument is exactly 1
            }
            return None; // argument straddles 1
        }
    }
    if ctx.is_builtin_call(expr, BuiltinFn::Exp) {
        let arg = single_arg()?;
        // exp(real constant) > 0 always; require the argument to be a real constant.
        if const_value_bounds(ctx, arg).is_some() {
            return Some(ConstSign::Positive);
        }
    }
    None
}

/// Provable sign of a real constant expression via EXACT rational reasoning, or
/// `None` when it cannot be decided (caller must bail; never guess).
pub fn provable_const_sign(ctx: &Context, expr: ExprId) -> Option<ConstSign> {
    if let Some(r) = as_rational_const(ctx, expr) {
        return Some(sign_of(&r));
    }
    if let Some((lo, hi)) = const_value_bounds(ctx, expr) {
        if lo.is_positive() {
            return Some(ConstSign::Positive);
        }
        if hi.is_negative() {
            return Some(ConstSign::Negative);
        }
        if lo.is_zero() && hi.is_zero() {
            return Some(ConstSign::Zero);
        }
        // Interval straddles 0: fall through to the transcendental rules.
    }
    transcendental_sign(ctx, expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn sign(src: &str) -> Option<ConstSign> {
        let mut ctx = Context::new();
        let e = parse(src, &mut ctx).expect("parse");
        provable_const_sign(&ctx, e)
    }

    #[test]
    fn rational_signs_are_exact() {
        assert_eq!(sign("3"), Some(ConstSign::Positive));
        assert_eq!(sign("-5/2"), Some(ConstSign::Negative));
        assert_eq!(sign("0"), Some(ConstSign::Zero));
        assert_eq!(sign("2 - 2"), Some(ConstSign::Zero));
    }

    #[test]
    fn pi_e_phi_combinations() {
        assert_eq!(sign("pi - 4"), Some(ConstSign::Negative)); // pi < 4
        assert_eq!(sign("pi - 3"), Some(ConstSign::Positive)); // pi > 3
        assert_eq!(sign("2*pi - 6"), Some(ConstSign::Positive)); // 2pi ~6.28 > 6
        assert_eq!(sign("e - 2"), Some(ConstSign::Positive));
        assert_eq!(sign("e - 3"), Some(ConstSign::Negative));
        assert_eq!(sign("phi - 1"), Some(ConstSign::Positive)); // phi ~1.618
        assert_eq!(sign("phi - 2"), Some(ConstSign::Negative));
    }

    #[test]
    fn sqrt_combinations() {
        assert_eq!(sign("sqrt(2) - 2"), Some(ConstSign::Negative)); // 1.414 < 2
        assert_eq!(sign("sqrt(2) - 1"), Some(ConstSign::Positive));
        assert_eq!(sign("sqrt(5) - 2"), Some(ConstSign::Positive)); // 2.236 > 2
        assert_eq!(sign("sqrt(4) - 2"), Some(ConstSign::Zero)); // exact
        assert_eq!(sign("sqrt(2) - 7/5"), Some(ConstSign::Positive)); // 1.4142 > 1.4
    }

    #[test]
    fn rational_exponent_powers_decide_comparisons() {
        // base^(p/q) over a nonnegative base via n-th-root bounds: e^(1/3)~1.3956,
        // 2^(1/3)~1.2599, pi^(3/2)~5.568, e^(-1/2)~0.6065.
        assert_eq!(sign("e^(1/3) - 1"), Some(ConstSign::Positive));
        assert_eq!(sign("1 - e^(1/3)"), Some(ConstSign::Negative));
        assert_eq!(sign("e^(1/3) - 7/5"), Some(ConstSign::Negative)); // 1.3956 < 1.4
        assert_eq!(sign("2^(1/3) - 5/4"), Some(ConstSign::Positive)); // 1.2599 > 1.25
        assert_eq!(sign("2^(1/3) - 63/50"), Some(ConstSign::Negative)); // 1.2599 < 1.26
        assert_eq!(sign("8^(1/3) - 2"), Some(ConstSign::Zero)); // exact n-th root
        assert_eq!(sign("(27/8)^(2/3) - 9/4"), Some(ConstSign::Zero)); // ((3/2)^3)^(2/3)
        assert_eq!(sign("pi^(3/2) - 5"), Some(ConstSign::Positive)); // 5.568 > 5
        assert_eq!(sign("pi^(3/2) - 6"), Some(ConstSign::Negative)); // 5.568 < 6
        assert_eq!(sign("e^(-1/2) - 1"), Some(ConstSign::Negative)); // reciprocal branch
        assert_eq!(sign("e^(-1/2) - 3/5"), Some(ConstSign::Positive)); // 0.6065 > 0.6
                                                                       // The P0-F-log root: e^(1/3)/(1 - e^(1/3)) ~ -3.53 must prove negative so
                                                                       // the log-equation domain filter (x > 0) can drop it.
        assert_eq!(sign("e^(1/3) / (1 - e^(1/3))"), Some(ConstSign::Negative));
        // Negative base with a fractional exponent: never guessed.
        assert_eq!(sign("(-2)^(1/3)"), None);
    }

    #[test]
    fn logarithms_by_argument() {
        assert_eq!(sign("ln(2)"), Some(ConstSign::Positive)); // 2 > 1
        assert_eq!(sign("ln(1/2)"), Some(ConstSign::Negative)); // 1/2 < 1
        assert_eq!(sign("ln(1)"), Some(ConstSign::Zero));
        assert_eq!(sign("log2(8)"), Some(ConstSign::Positive));
        assert_eq!(sign("exp(5)"), Some(ConstSign::Positive)); // exp > 0
        assert_eq!(sign("exp(-3)"), Some(ConstSign::Positive));
    }

    #[test]
    fn ln_value_bounds_decide_comparisons() {
        // VALUE (not just sign) comparisons against rational thresholds are now
        // decided by the exact atanh-series bounds: ln(2)~0.693, ln(5)~1.609,
        // ln(10)~2.303, ln(1/2)~-0.693.
        assert_eq!(sign("ln(2) - 1"), Some(ConstSign::Negative)); // 0.693 < 1
        assert_eq!(sign("ln(2) - 1/2"), Some(ConstSign::Positive)); // 0.693 > 0.5
        assert_eq!(sign("ln(5) - 1"), Some(ConstSign::Positive)); // 1.609 > 1
        assert_eq!(sign("ln(10) - 2"), Some(ConstSign::Positive)); // 2.303 > 2
        assert_eq!(sign("ln(10) - 3"), Some(ConstSign::Negative)); // 2.303 < 3
        assert_eq!(sign("ln(1/2) + 1"), Some(ConstSign::Positive)); // -0.693 + 1 > 0
        assert_eq!(sign("ln(1/2) + 1/2"), Some(ConstSign::Negative)); // -0.693 + 0.5 < 0
                                                                      // Composes with the other constants (ln of a bounded irrational argument).
        assert_eq!(sign("ln(pi) - 1"), Some(ConstSign::Positive)); // ln(3.1416)=1.14 > 1
    }

    #[test]
    fn log_base_value_bounds_decide_comparisons() {
        // log_b(c) = ln(c)/ln(b) value comparisons, via the ln bounds + interval
        // division: log2(3)=1.585, log2(10)=3.322, log10(50)=1.699, log_3(10)=2.096.
        assert_eq!(sign("log2(3) - 2"), Some(ConstSign::Negative)); // 1.585 < 2
        assert_eq!(sign("log2(3) - 1"), Some(ConstSign::Positive)); // 1.585 > 1
        assert_eq!(sign("log2(10) - 3"), Some(ConstSign::Positive)); // 3.322 > 3
        assert_eq!(sign("log10(50) - 2"), Some(ConstSign::Negative)); // 1.699 < 2
        assert_eq!(sign("log10(50) - 1"), Some(ConstSign::Positive)); // 1.699 > 1
        assert_eq!(sign("log(3, 10) - 2"), Some(ConstSign::Positive)); // 2.096 > 2
        assert_eq!(sign("log(3, 10) - 5/2"), Some(ConstSign::Negative)); // 2.096 < 2.5
                                                                         // Sign-only cases still decided (log2(8)=3 > 0, log2(1/2)=-1 < 0).
        assert_eq!(sign("log2(8)"), Some(ConstSign::Positive));
        assert_eq!(sign("log2(1/2)"), Some(ConstSign::Negative));
    }

    #[test]
    fn trig_value_bounds_decide_rational_argument_comparisons() {
        // sin/cos/tan at a RATIONAL argument (radians) via the Taylor series with the
        // Lagrange remainder: sin(1)=0.841, sin(2)=0.909, sin(3)=0.141, cos(1)=0.540,
        // cos(2)=-0.416, tan(1)=1.557.
        assert_eq!(sign("sin(1) - 1/2"), Some(ConstSign::Positive)); // 0.841 > 0.5
        assert_eq!(sign("sin(1) - 1"), Some(ConstSign::Negative)); // 0.841 < 1
        assert_eq!(sign("sin(2)"), Some(ConstSign::Positive)); // 0.909 > 0
        assert_eq!(sign("sin(3) - 1/2"), Some(ConstSign::Negative)); // 0.141 < 0.5
        assert_eq!(sign("cos(1) - 1/2"), Some(ConstSign::Positive)); // 0.540 > 0.5
        assert_eq!(sign("cos(2)"), Some(ConstSign::Negative)); // -0.416 < 0
        assert_eq!(sign("tan(1) - 1"), Some(ConstSign::Positive)); // 1.557 > 1
        assert_eq!(sign("sin(0)"), Some(ConstSign::Zero)); // sin(0) is exactly 0
                                                           // A LARGE argument bails (range reduction not implemented yet) -> None.
        assert_eq!(sign("sin(200)"), None);
    }

    #[test]
    fn trig_value_bounds_decide_irrational_argument_in_principal_piece() {
        // An IRRATIONAL (interval) argument inside the principal monotone piece is now
        // decided: sin increasing on (-pi/2, pi/2), cos decreasing on [0, pi].
        // sin(sqrt2)=0.988, cos(sqrt2)=0.156, sin(pi/7)=0.434, tan(sqrt2)=6.34.
        assert_eq!(sign("sin(sqrt(2))"), Some(ConstSign::Positive)); // 0.988 > 0
        assert_eq!(sign("sin(sqrt(2)) - 1"), Some(ConstSign::Negative)); // 0.988 < 1
        assert_eq!(sign("cos(sqrt(2))"), Some(ConstSign::Positive)); // 0.156 > 0
        assert_eq!(sign("cos(sqrt(2)) - 1/2"), Some(ConstSign::Negative)); // 0.156 < 0.5
        assert_eq!(sign("sin(pi/7) - 1/2"), Some(ConstSign::Negative)); // 0.434 < 0.5
        assert_eq!(sign("tan(sqrt(2)) - 5"), Some(ConstSign::Positive)); // 6.34 > 5
                                                                         // An interval argument OUTSIDE the principal piece still bails (sin is not
                                                                         // monotone across the extremum at pi/2): sqrt(8) = 2.83 lies in (pi/2, pi).
        assert_eq!(sign("sin(sqrt(8))"), None);
    }

    #[test]
    fn ln_bounds_bracket_known_values_tightly() {
        // Bounds must be SOUND (bracket the true value) and TIGHT (width < 1e-40).
        let max_width = BigRational::new(BigInt::one(), BigInt::from(10).pow(40));
        for x in [2i64, 3, 5, 7, 10, 100] {
            let (lo, hi) = ln_bounds(&BigRational::from_integer(BigInt::from(x))).unwrap();
            assert!(lo < hi, "ln({x}) bounds must be ordered");
            assert!(&hi - &lo < max_width, "ln({x}) bounds must be tight");
        }
        // Coarse SOUNDNESS sanity (the bounds are far tighter than these intervals,
        // so they must sit strictly inside — catches a grossly wrong series).
        // ln(2)=0.69314…, ln(5)=1.60943…, ln(10)=2.30258…
        let (lo2, hi2) = ln_bounds(&BigRational::from_integer(BigInt::from(2))).unwrap();
        assert!(lo2 > ratio(6931, 10000) && hi2 < ratio(6932, 10000));
        let (lo5, hi5) = ln_bounds(&BigRational::from_integer(BigInt::from(5))).unwrap();
        assert!(lo5 > ratio(16094, 10000) && hi5 < ratio(16095, 10000));
        let (lo10, hi10) = ln_bounds(&BigRational::from_integer(BigInt::from(10))).unwrap();
        assert!(lo10 > ratio(23025, 10000) && hi10 < ratio(23026, 10000));
        // ln(1) is EXACTLY 0 (not merely bracketed); ln of x<1 is symmetric.
        assert_eq!(ln_bounds(&one()), Some((zero(), zero())));
        let (loh, hih) = ln_bounds(&ratio(1, 2)).unwrap(); // ln(1/2) = -ln(2)
        assert!(loh == -hi2 && hih == -lo2);
        // Non-positive arguments have no real ln.
        assert_eq!(ln_bounds(&zero()), None);
        assert_eq!(ln_bounds(&ratio(-1, 1)), None);
    }

    #[test]
    fn undeterminable_returns_none() {
        // Argument straddles 1 with current bounds? ln near 1 with symbolic arg.
        assert_eq!(sign("x"), None); // not a constant
        assert_eq!(sign("ln(x)"), None);
    }

    #[test]
    fn bounds_bracket_high_precision_values() {
        // A 52-decimal reference for each constant must lie strictly inside the
        // 50-decimal bounds (this is what guarantees soundness near the boundary).
        let d52 = BigInt::from(10).pow(52);
        let pi_ref = BigRational::new(
            BigInt::from_str("31415926535897932384626433832795028841971693993751058").unwrap(),
            d52.clone(),
        );
        let (plo, phi_hi) = pi_bounds();
        assert!(plo < pi_ref && pi_ref < phi_hi);
        let e_ref = BigRational::new(
            BigInt::from_str("27182818284590452353602874713526624977572470936999596").unwrap(),
            d52,
        );
        let (elo, ehi) = e_bounds();
        assert!(elo < e_ref && e_ref < ehi);
    }

    #[test]
    fn near_zero_thresholds_are_decided() {
        // Thresholds within ~1e-8 of the constant (the adversarial near-boundary
        // family) must still resolve correctly with the 50-digit bounds.
        assert_eq!(sign("pi - 314159265/100000000"), Some(ConstSign::Positive));
        assert_eq!(sign("e - 271828182/100000000"), Some(ConstSign::Positive));
        assert_eq!(sign("phi - 161803398/100000000"), Some(ConstSign::Positive));
        assert_eq!(
            sign("pi^2 - 986960440/100000000"),
            Some(ConstSign::Positive)
        );
        assert_eq!(
            sign("pi*e - 853973422/100000000"),
            Some(ConstSign::Positive)
        );
    }
}
