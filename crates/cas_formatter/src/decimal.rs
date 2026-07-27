//! Decimal presentation of exact rationals for `approx(...)` results
//! (G1 Cap. E-iv-d). The engine stays exact everywhere; a `decimal(Number)`
//! wrapper node marks a value the user explicitly asked to see numerically,
//! and THIS formatting is the only place fractions become decimal strings.
//!
//! Large and small magnitudes render in scientific notation like a serious
//! scientific calculator: `approx(2^100)` shows `1.26765060023*10^30`, never
//! a 12-digit mantissa padded with 19 zeros. The switch follows the `%g`/
//! calculator rule: scientific iff the decimal exponent falls outside
//! `[-4, sig)`. Purely presentational and WYSIWYG-preserving: the
//! `mantissa*10^e` reading is EXACTLY the stored rounded rational.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Smallest decimal exponent still rendered fixed: `0.0001` stays fixed,
/// `0.00001` becomes `1*10^(-5)`. Mirrors `%g` (scientific iff e < -4).
/// Must stay in lockstep with `cas_math::decimal_display` (the fold-
/// description twin renderer).
const SCI_LOWER_EXP: i64 = -4;

/// How a rounded rational presents: plain positional notation, or calculator
/// scientific notation `mantissa × 10^exponent` (mantissa carries the sign;
/// each render surface picks its own `×`/`\times` spelling).
pub enum DecimalRendering {
    Fixed(String),
    Scientific { mantissa: String, exponent: i64 },
}

/// Significant digits + decimal exponent of a nonzero rational rounded
/// half-up to `sig` digits: `(negative, digits, e)` with `digits` trailing-
/// zero-trimmed (at least one digit) and `e` the exponent of the leading
/// digit, so `value = ±d.igits · 10^e`. `None` for zero.
fn rounded_digits(value: &BigRational, sig: usize) -> Option<(bool, String, i64)> {
    if value.is_zero() || sig == 0 {
        return None;
    }
    let negative = value.is_negative();
    let v = value.abs();
    let (num, den) = (v.numer().clone(), v.denom().clone());
    let ten = BigInt::from(10);

    // Decimal exponent e with 10^e <= v < 10^(e+1), located from the digit
    // counts (true e is `num_len - den_len` or one less; one exact compare
    // settles it — no scaling loops, so huge magnitudes stay cheap).
    let num_len = num.to_string().len() as i64;
    let den_len = den.to_string().len() as i64;
    let mut e = num_len - den_len;
    let v_at_least_pow10 = |c: i64| -> bool {
        if c >= 0 {
            num >= &den * ten.pow(c as u32)
        } else {
            &num * ten.pow((-c) as u32) >= den
        }
    };
    if !v_at_least_pow10(e) {
        e -= 1;
    }

    // Scale so the kept digits become the integer part, round half-up.
    let k = sig as i64 - 1 - e;
    let round_half_up = |a: &BigInt, b: &BigInt| -> BigInt { (a * 2 + b) / (b * 2) };
    let scaled = if k >= 0 {
        round_half_up(&(&num * ten.pow(k as u32)), &den)
    } else {
        round_half_up(&num, &(&den * ten.pow((-k) as u32)))
    };
    let mut digits = scaled.to_string();
    if digits.len() > sig {
        // Rounding carried over (999…9 -> 1000…0): the extras are zeros.
        e += (digits.len() - sig) as i64;
        digits.truncate(sig);
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    Some((negative, digits, e))
}

/// Decide fixed vs scientific for an exact rational at `sig` significant
/// digits. Rounding here can never feed a decision — display only.
pub fn rational_decimal_rendering(value: &BigRational, sig: usize) -> DecimalRendering {
    let sig = sig.max(1);
    let Some((negative, digits, e)) = rounded_digits(value, sig) else {
        return DecimalRendering::Fixed("0".to_string());
    };
    let sign = if negative { "-" } else { "" };
    if e >= sig as i64 || e < SCI_LOWER_EXP {
        let mantissa = if digits.len() > 1 {
            format!("{sign}{}.{}", &digits[..1], &digits[1..])
        } else {
            format!("{sign}{digits}")
        };
        return DecimalRendering::Scientific {
            mantissa,
            exponent: e,
        };
    }
    let mut out = String::from(sign);
    if e >= 0 {
        let int_len = (e + 1) as usize;
        if digits.len() <= int_len {
            out.push_str(&digits);
            out.push_str(&"0".repeat(int_len - digits.len()));
        } else {
            out.push_str(&digits[..int_len]);
            out.push('.');
            out.push_str(&digits[int_len..]);
        }
    } else {
        out.push_str("0.");
        out.push_str(&"0".repeat((-e - 1) as usize));
        out.push_str(&digits);
    }
    DecimalRendering::Fixed(out)
}

/// True when [`rational_decimal_rendering`] picks scientific notation. The
/// renderers consult this to parenthesize: the scientific string contains
/// `*`/`^`, so as a Pow base or Div denominator it is no longer atomic.
pub fn renders_scientific(value: &BigRational, sig: usize) -> bool {
    matches!(
        rational_decimal_rendering(value, sig),
        DecimalRendering::Scientific { .. }
    )
}

/// Scientific suffix with a caller-chosen multiplication sign: `m×10^30`,
/// `m*10^(-31)` (negative exponents parenthesized so the ASCII form
/// re-parses to the exact stored value).
pub fn sci_with_times(mantissa: &str, exponent: i64, times: &str) -> String {
    if exponent < 0 {
        format!("{mantissa}{times}10^({exponent})")
    } else {
        format!("{mantissa}{times}10^{exponent}")
    }
}

/// Format an exact rational as a decimal string with up to `sig` significant
/// digits (trailing zeros trimmed), in ASCII: fixed notation in range,
/// `m*10^e` outside it. Purely presentational.
pub fn format_rational_decimal(value: &BigRational, sig: usize) -> String {
    match rational_decimal_rendering(value, sig) {
        DecimalRendering::Fixed(s) => s,
        DecimalRendering::Scientific { mantissa, exponent } => {
            sci_with_times(&mantissa, exponent, "*")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    fn big10(exp: u32) -> BigInt {
        BigInt::from(10u32).pow(exp)
    }

    #[test]
    fn in_range_values_keep_fixed_notation() {
        assert_eq!(format_rational_decimal(&rat(3, 7), 12), "0.428571428571");
        assert_eq!(format_rational_decimal(&rat(-2, 3), 12), "-0.666666666667");
        assert_eq!(format_rational_decimal(&rat(1, 2), 12), "0.5");
        assert_eq!(format_rational_decimal(&rat(3, 1), 12), "3");
        assert_eq!(format_rational_decimal(&rat(0, 1), 12), "0");
        // e = 11: twelve integer digits is the largest fixed magnitude.
        assert_eq!(
            format_rational_decimal(&rat(999999999999, 1), 12),
            "999999999999"
        );
        // e = -4 is the smallest fixed magnitude (`%g` rule).
        assert_eq!(format_rational_decimal(&rat(1, 10000), 12), "0.0001");
        assert_eq!(
            format_rational_decimal(&rat(1, 7000), 12),
            "0.000142857142857"
        );
    }

    #[test]
    fn large_magnitudes_switch_to_scientific() {
        // 2^100 rounded to 12 significant digits.
        let payload = BigRational::from_integer(BigInt::from(126765060023u64) * big10(19));
        assert_eq!(format_rational_decimal(&payload, 12), "1.26765060023*10^30");
        // Exactly 10^12 crosses the threshold (13 integer digits).
        assert_eq!(
            format_rational_decimal(&BigRational::from_integer(big10(12)), 12),
            "1*10^12"
        );
        assert_eq!(
            format_rational_decimal(&BigRational::from_integer(-3 * big10(15)), 12),
            "-3*10^15"
        );
        // Unrounded input still rounds to 12 significant digits first.
        assert_eq!(
            format_rational_decimal(
                &BigRational::from_integer(BigInt::from(1234567890123456i64)),
                12
            ),
            "1.23456789012*10^15"
        );
    }

    #[test]
    fn small_magnitudes_switch_to_scientific() {
        assert_eq!(format_rational_decimal(&rat(1, 100000), 12), "1*10^(-5)");
        // 1/2^100 rounded to 12 significant digits.
        let payload = BigRational::new(788860905221u64.into(), big10(42));
        assert_eq!(
            format_rational_decimal(&payload, 12),
            "7.88860905221*10^(-31)"
        );
        assert_eq!(
            format_rational_decimal(&-BigRational::new(1.into(), big10(7)), 12),
            "-1*10^(-7)"
        );
    }

    #[test]
    fn rounding_carry_can_cross_the_threshold() {
        // 999999999999.6 rounds half-up to 1000000000000 = 1*10^12: the carry
        // pushes e from 11 to 12, so the DECIDED form is scientific.
        let v = BigRational::new(9999999999996i64.into(), 10.into());
        assert_eq!(format_rational_decimal(&v, 12), "1*10^12");
        // 0.000099999999999951 rounds to 0.0001: carry pulls e back INTO the
        // fixed range.
        let v = BigRational::new(99999999999951i64.into(), big10(18));
        assert_eq!(format_rational_decimal(&v, 12), "0.0001");
    }

    #[test]
    fn scientific_reading_equals_stored_value_wysiwyg() {
        // `1.26765060023*10^30` re-parsed = mantissa shifted = the payload.
        let payload = BigRational::from_integer(BigInt::from(126765060023u64) * big10(19));
        let mantissa = BigRational::new(126765060023u64.into(), big10(11));
        assert_eq!(mantissa * BigRational::from_integer(big10(30)), payload);
    }
}
