//! Scientific-notation approximation of astronomically large (or small)
//! EXACT numeric expressions — the territory `approx`'s f64 contract cannot
//! reach (|value| beyond ~10^±308), e.g. `5^123456789` or `12345!`.
//!
//! Pure integer/rational arithmetic: a value travels as a normalized triple
//! (sign, mantissa `m` with `1 ≤ m < 10`, decimal exponent: BigInt), the
//! mantissa truncated to [`WORKING_SIG`] significant digits after every
//! operation. The exponent comes from renormalization COUNTING — no floating
//! `log10` ever decides an integer. This is presentation arithmetic for an
//! explicitly-approximate surface; no keep/drop decision rides on it.
//!
//! Accuracy is enforced, not hoped for: every operation also advances a
//! conservative error bound measured in truncation units (`err_ulps`, one
//! unit = one worst-case mantissa truncation). Squaring doubles the bound,
//! so `b^e` costs `~2^bits(e)` units — fine for any single power, and the
//! budget check refuses pathological nests like `(a^(10^18))^(10^18)`
//! instead of publishing garbage digits.

use crate::decimal_display::{round_rational_sig, DECIMAL_DISPLAY_SIG_DIGITS};
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// Working mantissa precision (significant decimal digits). The published
/// mantissa keeps [`DECIMAL_DISPLAY_SIG_DIGITS`] (12); the 28-digit guard
/// absorbs error compounding, bounded by [`ERR_ULPS_BUDGET`] below.
const WORKING_SIG: usize = 40;

/// Highest admissible accumulated error, in truncation units of `10^-39`
/// relative each. `5e23` units keep the true relative error under
/// `5e23 · 10^-39 = 5e-16`, i.e. the 12 published digits stay exact with
/// three digits of headroom. A single `b^e` with `e ≤ 10^18` accumulates
/// `~2^61 ≈ 2.3e18` units and passes; a nested `(b^e)^k` of that size
/// overflows the budget and declines honestly.
const ERR_ULPS_BUDGET: f64 = 5e23;

/// Largest `|e|` accepted in `b^e` (beyond it even the EXPONENT arithmetic
/// stops being meaningful input).
const POW_EXP_MAX: u64 = 1_000_000_000_000_000_000;

/// Largest `n` for `n!` — the O(n) exact product below stays well under a
/// second here (measured; `10^6!` has 5.6M digits and is the practical
/// ceiling for a didactic surface).
const FACT_ARG_MAX: u64 = 1_000_000;

/// A value in scientific notation: `mantissa · 10^exp10`, mantissa already
/// rounded to [`DECIMAL_DISPLAY_SIG_DIGITS`] with `1 ≤ |mantissa| < 10`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SciApprox {
    pub mantissa: BigRational,
    pub exp10: BigInt,
}

/// Working value: `sign · m · 10^exp10` with `1 ≤ m < 10`, plus the running
/// error bound in truncation units.
#[derive(Debug, Clone)]
struct SciValue {
    negative: bool,
    m: BigRational,
    exp10: BigInt,
    err_ulps: f64,
}

impl SciValue {
    /// Renormalize `m` into `[1, 10)` (adjusting `exp10`), truncate to
    /// [`WORKING_SIG`] digits and charge one truncation unit.
    fn normalized(mut m: BigRational, mut exp10: BigInt, negative: bool, err_ulps: f64) -> Self {
        let ten = BigRational::from_integer(BigInt::from(10));
        while m >= ten {
            m /= &ten;
            exp10 += 1;
        }
        let one = BigRational::one();
        while m < one {
            m *= &ten;
            exp10 -= 1;
        }
        m = round_rational_sig(&m, WORKING_SIG);
        // Rounding up can land exactly on 10 — fold it back.
        if m >= ten {
            m /= &ten;
            exp10 += 1;
        }
        SciValue {
            negative,
            m,
            exp10,
            err_ulps: err_ulps + 1.0,
        }
    }

    fn from_rational(r: &BigRational) -> Option<SciValue> {
        if r.is_zero() {
            return None;
        }
        let negative = r.is_negative();
        let v = r.abs();
        // Decimal-exponent estimate from digit counts; normalized() fixes the
        // off-by-one this coarse estimate can leave.
        let digits_num = v.numer().to_string().len() as i64;
        let digits_den = v.denom().to_string().len() as i64;
        let shift = BigInt::from(digits_num - digits_den);
        let m = v / pow10_rational(&shift)?;
        Some(SciValue::normalized(m, shift, negative, 0.0))
    }

    fn mul(&self, other: &SciValue) -> SciValue {
        SciValue::normalized(
            &self.m * &other.m,
            &self.exp10 + &other.exp10,
            self.negative != other.negative,
            self.err_ulps + other.err_ulps,
        )
    }

    fn div(&self, other: &SciValue) -> SciValue {
        SciValue::normalized(
            &self.m / &other.m,
            &self.exp10 - &other.exp10,
            self.negative != other.negative,
            self.err_ulps + other.err_ulps,
        )
    }

    fn reciprocal(&self) -> SciValue {
        SciValue::normalized(self.m.recip(), -&self.exp10, self.negative, self.err_ulps)
    }

    /// `self^e` by binary exponentiation (square-and-multiply); `e = 0` is 1.
    fn powi(&self, e: u64) -> SciValue {
        if e == 0 {
            return SciValue::normalized(BigRational::one(), BigInt::zero(), false, 0.0);
        }
        let mut base = self.clone();
        let mut rest = e;
        // Square up to the lowest set bit and seed the accumulator there — a
        // positive exponent always has one, so no Option (no panic path).
        while rest & 1 == 0 {
            base = base.mul(&base);
            rest >>= 1;
        }
        let mut acc = base.clone();
        rest >>= 1;
        while rest > 0 {
            base = base.mul(&base);
            if rest & 1 == 1 {
                acc = acc.mul(&base);
            }
            rest >>= 1;
        }
        acc
    }
}

/// `10^e` exact, or `None` when `|e|` does not even fit `u32` — an exponent
/// that size means materializing gigabytes of digits, so the caller declines
/// instead of attempting it (the sci lane exists precisely to avoid that).
fn pow10_rational(e: &BigInt) -> Option<BigRational> {
    let magnitude = e.magnitude().to_u32()?;
    let p = BigInt::from(10).pow(magnitude);
    Some(if e.is_negative() {
        BigRational::new(BigInt::one(), p)
    } else {
        BigRational::from_integer(p)
    })
}

/// Exact `n!` as a sci value: running BigInt product, stripped to
/// [`WORKING_SIG`] leading digits whenever it outgrows twice that (each strip
/// truncates once — error grows LINEARLY, never compounds).
fn factorial_sci(n: u64) -> SciValue {
    let mut acc = BigInt::one();
    let mut stripped = BigInt::zero();
    let mut err_ulps = 0.0;
    // ~3.33 bits per decimal digit; strip when the accumulator holds about
    // twice the working precision.
    let strip_bits = (WORKING_SIG as u64) * 2 * 10 / 3;
    for k in 2..=n {
        acc *= k;
        if acc.bits() > strip_bits {
            let digits = acc.to_string().len();
            let drop = digits - WORKING_SIG;
            acc /= BigInt::from(10).pow(drop as u32);
            stripped += drop;
            err_ulps += 1.0;
        }
    }
    let digits = acc.to_string().len() as i64;
    let exp10 = stripped + BigInt::from(digits - 1);
    let m = BigRational::new(acc, BigInt::from(10).pow((digits - 1) as u32));
    SciValue::normalized(m, exp10, false, err_ulps)
}

/// Evaluate a closed numeric expression into scientific notation. Grammar:
/// numbers, `Neg`/`Mul`/`Div`, `Pow` with integer exponent (`|e| ≤ 10^18`),
/// `fact`/`factorial` of an integer literal (`≤ 10^6`), transparently through
/// `__hold` and the `decimal` display node (so `approx(approx(x))`
/// re-derives the same value). Anything else — or a value of zero, or an
/// error bound past budget — declines with `None`.
pub fn try_sci_approx_expr(ctx: &Context, expr: ExprId) -> Option<SciApprox> {
    let value = sci_eval(ctx, expr)?;
    if value.err_ulps > ERR_ULPS_BUDGET {
        return None;
    }
    let mut mantissa = round_rational_sig(&value.m, DECIMAL_DISPLAY_SIG_DIGITS);
    let mut exp10 = value.exp10;
    let ten = BigRational::from_integer(BigInt::from(10));
    if mantissa >= ten {
        mantissa /= &ten;
        exp10 += 1;
    }
    if value.negative {
        mantissa = -mantissa;
    }
    Some(SciApprox { mantissa, exp10 })
}

fn sci_eval(ctx: &Context, expr: ExprId) -> Option<SciValue> {
    match ctx.get(expr) {
        Expr::Number(r) => SciValue::from_rational(r),
        Expr::Neg(inner) => {
            let mut v = sci_eval(ctx, *inner)?;
            v.negative = !v.negative;
            Some(v)
        }
        Expr::Hold(inner) => sci_eval(ctx, *inner),
        Expr::Mul(l, r) => {
            let a = sci_eval(ctx, *l)?;
            let b = sci_eval(ctx, *r)?;
            Some(a.mul(&b))
        }
        Expr::Div(l, r) => {
            let a = sci_eval(ctx, *l)?;
            let b = sci_eval(ctx, *r)?;
            Some(a.div(&b))
        }
        Expr::Pow(base, exp) => {
            let e = extract_bounded_integer(ctx, *exp, POW_EXP_MAX)?;
            let base_value = sci_eval(ctx, *base)?;
            match e.magnitude {
                0 => SciValue::from_rational(&BigRational::one()),
                magnitude => {
                    let powered = base_value.powi(magnitude);
                    Some(if e.negative {
                        powered.reciprocal()
                    } else {
                        powered
                    })
                }
            }
        }
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.sym_name(*fn_id) {
            "fact" | "factorial" => {
                let n = extract_bounded_integer(ctx, args[0], FACT_ARG_MAX)?;
                if n.negative {
                    return None;
                }
                if n.magnitude <= 1 {
                    return SciValue::from_rational(&BigRational::one());
                }
                Some(factorial_sci(n.magnitude))
            }
            // The numeric-presentation display node is transparent.
            "decimal" => sci_eval(ctx, args[0]),
            _ => None,
        },
        _ => None,
    }
}

struct BoundedInteger {
    negative: bool,
    magnitude: u64,
}

/// The argument as an integer literal with `|value| ≤ max`, else `None`.
fn extract_bounded_integer(ctx: &Context, expr: ExprId, max: u64) -> Option<BoundedInteger> {
    let (negative, inner) = match ctx.get(expr) {
        Expr::Neg(inner) => (true, *inner),
        _ => (false, expr),
    };
    let Expr::Number(r) = ctx.get(inner) else {
        return None;
    };
    if !r.is_integer() {
        return None;
    }
    let magnitude = r.numer().magnitude().to_u64()?;
    if magnitude > max {
        return None;
    }
    Some(BoundedInteger {
        negative: negative != r.is_negative(),
        magnitude,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn sci_of(source: &str) -> SciApprox {
        let mut ctx = Context::new();
        let expr = parse(source, &mut ctx).expect("parse");
        try_sci_approx_expr(&ctx, expr).expect("sci approx")
    }

    fn assert_sci(source: &str, mantissa: &str, exp10: i64) {
        let sci = sci_of(source);
        let expected = parse_decimal_rational(mantissa);
        assert_eq!(
            sci.mantissa, expected,
            "{source}: mantissa {} != {mantissa}",
            sci.mantissa
        );
        assert_eq!(sci.exp10, BigInt::from(exp10), "{source}: exponent");
    }

    /// `"2.20110600528"` → exact rational `220110600528/10^11`.
    fn parse_decimal_rational(s: &str) -> BigRational {
        let negative = s.starts_with('-');
        let unsigned = s.trim_start_matches('-');
        let (int_part, frac_part) = unsigned.split_once('.').unwrap_or((unsigned, ""));
        let digits: BigInt = format!("{int_part}{frac_part}").parse().expect("digits");
        let scale = BigInt::from(10).pow(frac_part.len() as u32);
        let r = BigRational::new(digits, scale);
        if negative {
            -r
        } else {
            r
        }
    }

    // References computed externally with 60-digit decimal arithmetic
    // (Python `decimal` for powers, exact `math.factorial` for factorials).

    #[test]
    fn power_beyond_f64_matches_high_precision_reference() {
        assert_sci("5^123456789", "2.20110600528", 86292592);
    }

    #[test]
    fn negative_exponent_gives_reciprocal_scale() {
        assert_sci("5^(-123456789)", "4.54317055880", -86292593);
    }

    #[test]
    fn rational_base_below_one_matches_reference() {
        assert_sci("(3/7)^12345678", "1.25105903156", -4542923);
    }

    #[test]
    fn factorial_matches_exact_leading_digits() {
        // 12345! = 3.443642469186…×10^45150 (exact leading digits).
        assert_sci("12345!", "3.44364246919", 45150);
    }

    #[test]
    fn factorial_at_cap_matches_exact_leading_digits() {
        // 1000000! = 8.263931688331…×10^5565708 (exact, 5.6M digits).
        assert_sci("1000000!", "8.26393168833", 5565708);
    }

    #[test]
    fn agrees_with_the_f64_lane_in_overlapping_range() {
        // 2^500 = 3.2733906078961…×10^150: the exact lane must reproduce the
        // f64 surface digit for digit where both can compute.
        let sci = sci_of("2^500");
        let f64_reference =
            crate::decimal_display::approx_display_rational(2.0f64.powi(500)).expect("finite");
        let scale = pow10_rational(&sci.exp10).expect("test exponent fits u32");
        assert_eq!(&sci.mantissa * &scale, f64_reference, "2^500 lanes differ");
    }

    #[test]
    fn negative_base_carries_parity_sign() {
        let odd = sci_of("(-5)^123456789");
        assert!(odd.mantissa.is_negative(), "odd exponent keeps the sign");
        let even = sci_of("(-5)^123456788");
        assert!(!even.mantissa.is_negative(), "even exponent drops the sign");
    }

    #[test]
    fn products_and_quotients_compose() {
        // 3 · 5^123456789 → mantissa 3·2.20110600528… = 6.60331801585…
        assert_sci("3 * 5^123456789", "6.60331801585", 86292592);
        // 5^123456789 / 5^123456789 stays 1·10^0 (no cancellation smarts,
        // just consistent arithmetic).
        assert_sci("5^123456789 / 5^123456789", "1", 0);
    }

    #[test]
    fn declines_out_of_grammar_and_out_of_budget() {
        let mut ctx = Context::new();
        for source in [
            "x^123456789",                               // symbolic base
            "5^x",                                       // symbolic exponent
            "5^(1/2)",                                   // non-integer exponent
            "(-3)!",                                     // factorial of a negative
            "2000000!",                                  // beyond FACT_ARG_MAX
            "5^1000000000000000001",                     // beyond POW_EXP_MAX
            "(5^999999999999999999)^999999999999999999", // error budget
            "0 * 5^123456789",                           // exact zero
            "5^123456789 + 1",                           // Add is out of grammar
        ] {
            let expr = parse(source, &mut ctx).expect(source);
            assert!(
                try_sci_approx_expr(&ctx, expr).is_none(),
                "{source} must decline"
            );
        }
    }

    #[test]
    fn small_values_still_compute_exactly() {
        // The lane itself is magnitude-agnostic; gating lives in the caller.
        assert_sci("2^100", "1.26765060023", 30);
        assert_sci("10^400", "1", 400);
    }
}
