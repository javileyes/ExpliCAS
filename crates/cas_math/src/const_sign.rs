//! Exact rational SIGN oracle for variable-free (constant) expressions.
//!
//! Computes VERIFIED rational value bounds `[lo, hi]` (with `lo <= value <= hi`,
//! both `BigRational`) for a constant expression built from rationals, `pi`, `e`,
//! the golden ratio `phi`, `sqrt`, and `+ - * /` / integer powers, and derives a
//! provable sign from them. Transcendental `ln`/`log`/`exp` get cheap exact sign
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

/// Provable sign of a real constant expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstSign {
    Negative,
    Zero,
    Positive,
}

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
    bounds_to_50dp("314159265358979323846264338327950288419716939937510")
}
fn e_bounds() -> (BigRational, BigRational) {
    // e  = 2.71828182845904523536028747135266249775724709369995 9574966968...
    bounds_to_50dp("271828182845904523536028747135266249775724709369995")
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
    let (l2_lo, l2_hi) = atanh_ln_bounds(&two); // ln(2)
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
                let den = interval_recip(ln_bounds(&BigRational::from_integer(BigInt::from(2)))?)?;
                Some(interval_mul(num, den))
            } else if ctx.is_builtin_call(expr, BuiltinFn::Log10) && args.len() == 1 {
                // log10(c) = ln(c) / ln(10).
                let num = ln_interval_bounds(bounds(args[0])?)?;
                let den = interval_recip(ln_bounds(&BigRational::from_integer(BigInt::from(10)))?)?;
                Some(interval_mul(num, den))
            } else if ctx.is_builtin_call(expr, BuiltinFn::Log) && args.len() == 2 {
                // log(base, arg) = ln(arg) / ln(base); the base interval must not
                // straddle 1 (else ln(base) brackets 0 and the ratio is unbounded).
                let num = ln_interval_bounds(bounds(args[1])?)?;
                let den = interval_recip(ln_interval_bounds(bounds(args[0])?)?)?;
                Some(interval_mul(num, den))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Bounds of `base ^ exp` where `exp` is a rational constant. Handles integer
/// exponents (positive/negative) and `1/2` (square root); bails otherwise.
fn interval_pow(
    base: (BigRational, BigRational),
    exp: &BigRational,
) -> Option<(BigRational, BigRational)> {
    if exp.is_zero() {
        return Some((one(), one()));
    }
    // exp == 1/2  =>  sqrt
    if *exp == ratio(1, 2) {
        let (lo, hi) = base;
        if lo.is_negative() {
            return None;
        }
        let (lo2, _) = sqrt_bounds(&lo)?;
        let (_, hi2) = sqrt_bounds(&hi)?;
        return Some((lo2, hi2));
    }
    if !exp.is_integer() {
        return None;
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
