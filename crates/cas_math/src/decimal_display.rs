//! WYSIWYG payload rounding for the `decimal(Number)` display node.
//!
//! `approx(...)` wraps the ROUNDED 12-significant-digit rational — the same
//! value the formatter shows — so the displayed number IS the stored number.
//! Without this, `approx(3/7)` would carry the raw f64 binary expansion:
//! invisible phantom digits that make `approx(3/7) - 0.428571428571` a tiny
//! nonzero residue instead of exactly 0.
//!
//! Purely presentational: this rounding feeds display payloads only, never a
//! keep/drop decision (the soundness gates keep their exact `as_rational_*`
//! paths untouched).

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Significant digits carried by the `decimal` display surface. Must match
/// the formatter's render width (`format_rational_decimal(_, 12)`), which
/// becomes a no-op re-round of an already-rounded payload.
pub const DECIMAL_DISPLAY_SIG_DIGITS: usize = 12;

/// The exact rational an `approx()` result should wrap for an f64 value:
/// `from_float` then round half-up to 12 significant digits. `None` for
/// non-finite input.
pub fn approx_display_rational(value: f64) -> Option<BigRational> {
    if !value.is_finite() {
        return None;
    }
    let raw = BigRational::from_float(value)?;
    Some(round_rational_sig(&raw, DECIMAL_DISPLAY_SIG_DIGITS))
}

/// Round an exact rational half-up to `sig` significant decimal digits,
/// returning the rounded value as an exact rational. Large magnitudes round
/// too (`10^20/7` becomes `14285714285700000000`, not raw f64 quantization
/// junk); zero stays zero.
pub fn round_rational_sig(value: &BigRational, sig: usize) -> BigRational {
    if value.is_zero() || sig == 0 {
        return BigRational::zero();
    }
    let negative = value.is_negative();
    let v = value.abs();
    let (num, den) = (v.numer().clone(), v.denom().clone());
    let ten = BigInt::from(10);

    // Decimal exponent e with 10^e <= v < 10^(e+1).
    let int_part = &num / &den;
    let e: i64 = if !int_part.is_zero() {
        int_part.to_string().len() as i64 - 1
    } else {
        // v < 1: count how many times ×10 is needed to reach >= 1.
        let mut scaled = num.clone();
        let mut z: i64 = 0;
        while scaled < den {
            scaled *= &ten;
            z += 1;
            if z > 4096 {
                // Unreachably small for f64-born values; fall back to zero.
                return BigRational::zero();
            }
        }
        -z
    };

    // Scale so the kept digits become the integer part, round, scale back.
    let k = sig as i64 - 1 - e;
    let round_half_up = |a: &BigInt, b: &BigInt| -> BigInt {
        // floor(a/b + 1/2) for positive a, b.
        (a * 2 + b) / (b * 2)
    };
    let mut rounded = if k >= 0 {
        let scale = ten.pow(k as u32);
        let n = round_half_up(&(&num * &scale), &den);
        BigRational::new(n, scale)
    } else {
        let scale = ten.pow((-k) as u32);
        let n = round_half_up(&num, &(&den * &scale));
        BigRational::from_integer(n * scale)
    };
    if negative {
        rounded = -rounded;
    }
    rounded
}

/// Render a rational as a plain decimal string at the display width
/// (rounding first — a no-op for already-rounded payloads). Used by fold
/// descriptions so didactic text never leaks `428571428571/1000000000000`.
pub fn decimal_display_string(value: &BigRational) -> String {
    let rounded = round_rational_sig(value, DECIMAL_DISPLAY_SIG_DIGITS);
    if rounded.is_zero() {
        return "0".to_string();
    }
    let negative = rounded.is_negative();
    let v = rounded.abs();
    let (num, den) = (v.numer().clone(), v.denom().clone());
    // Rounded payloads have denominators dividing a power of ten.
    let ten = BigInt::from(10);
    let mut k = 0u32;
    let mut pow = BigInt::from(1);
    while (&pow % &den) != BigInt::from(0) && k < 4096 {
        pow *= &ten;
        k += 1;
    }
    let scaled = &num * (&pow / &den);
    let digits = scaled.to_string();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if k == 0 {
        out.push_str(&digits);
        return out;
    }
    let k = k as usize;
    if digits.len() > k {
        out.push_str(&digits[..digits.len() - k]);
        let frac = digits[digits.len() - k..].trim_end_matches('0');
        if !frac.is_empty() {
            out.push('.');
            out.push_str(frac);
        }
    } else {
        out.push_str("0.");
        for _ in 0..(k - digits.len()) {
            out.push('0');
        }
        out.push_str(digits.trim_end_matches('0'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    #[test]
    fn decimal_string_renders_rounded_payloads() {
        assert_eq!(decimal_display_string(&rat(3, 7)), "0.428571428571");
        assert_eq!(decimal_display_string(&rat(-2, 3)), "-0.666666666667");
        assert_eq!(decimal_display_string(&rat(1, 2)), "0.5");
        assert_eq!(decimal_display_string(&rat(3, 1)), "3");
        assert_eq!(decimal_display_string(&rat(0, 1)), "0");
    }

    #[test]
    fn rounds_to_twelve_significant_digits_half_up() {
        // 3/7 = 0.428571428571428... -> 0.428571428571 (truncation case:
        // 13th significant digit is 4, rounds down).
        assert_eq!(
            round_rational_sig(&rat(3, 7), 12),
            rat(428571428571, 1_000_000_000_000)
        );
        // 2/3 = 0.666666... -> half-up on the 13th digit (6) rounds UP.
        assert_eq!(
            round_rational_sig(&rat(2, 3), 12),
            rat(666666666667, 1_000_000_000_000)
        );
        // Sign is preserved through the abs/round/reapply path.
        assert_eq!(
            round_rational_sig(&rat(-2, 3), 12),
            rat(-666666666667, 1_000_000_000_000)
        );
    }

    #[test]
    fn small_values_keep_significant_not_absolute_digits() {
        // 1/70000 = 0.0000142857142857... -> 12 SIGNIFICANT digits.
        let rounded = round_rational_sig(&rat(1, 70_000), 12);
        assert_eq!(
            rounded,
            BigRational::new(142857142857u64.into(), 10u64.pow(16).into())
        );
    }

    #[test]
    fn large_magnitudes_round_instead_of_keeping_quantization_junk() {
        // 10^20/7 = 14285714285714285714.28... -> 12 sig digits + zeros.
        let big = BigRational::new(num_bigint::BigInt::from(10u64).pow(20), 7.into());
        let rounded = round_rational_sig(&big, 12);
        let expected =
            num_bigint::BigInt::from(142857142857u64) * num_bigint::BigInt::from(10u64).pow(8);
        assert_eq!(rounded, BigRational::from_integer(expected));
    }

    #[test]
    fn approx_display_rational_matches_displayed_value() {
        // The WYSIWYG identity: the payload for f64(3/7) is EXACTLY the
        // rational that parsing the displayed "0.428571428571" produces.
        let payload = approx_display_rational(3.0f64 / 7.0).expect("finite");
        assert_eq!(payload, rat(428571428571, 1_000_000_000_000));
        // Exact-representable values stay exact.
        assert_eq!(approx_display_rational(0.5).unwrap(), rat(1, 2));
        assert_eq!(approx_display_rational(-2.0).unwrap(), rat(-2, 1));
        assert_eq!(approx_display_rational(0.0).unwrap(), BigRational::zero());
        // Non-finite declines.
        assert!(approx_display_rational(f64::NAN).is_none());
        assert!(approx_display_rational(f64::INFINITY).is_none());
        // Round-trip vs from_f64 sanity: rounded value is within 5e-13 rel.
        let v = std::f64::consts::PI;
        let payload = approx_display_rational(v).unwrap();
        let back = BigRational::from_f64(v).unwrap();
        let diff = (payload - back).abs();
        assert!(diff < rat(1, 1_000_000_000_000));
    }
}
