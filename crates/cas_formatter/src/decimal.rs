//! Decimal presentation of exact rationals for `approx(...)` results
//! (G1 Cap. E-iv-d). The engine stays exact everywhere; a `decimal(Number)`
//! wrapper node marks a value the user explicitly asked to see numerically,
//! and THIS formatting is the only place fractions become decimal strings.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Format an exact rational as a decimal string with up to `sig` significant
/// digits (trailing zeros trimmed). Purely presentational — rounding here can
/// never feed a decision.
pub fn format_rational_decimal(value: &BigRational, sig: usize) -> String {
    if value.is_zero() {
        return "0".to_string();
    }
    let negative = value.is_negative();
    let v = value.abs();
    let (num, den) = (v.numer().clone(), v.denom().clone());
    let mut int_part = &num / &den;
    let mut rem = &num % &den;
    let int_digits = if int_part.is_zero() {
        0
    } else {
        int_part.to_string().len()
    };
    let frac_len = sig
        .saturating_sub(int_digits)
        .max(if int_part.is_zero() { sig } else { 0 });
    let mut frac_digits: Vec<u8> = Vec::new();
    let ten = BigInt::from(10);
    // When the integer part is zero, skip leading fractional zeros so `sig`
    // counts SIGNIFICANT digits (0.0000123456789012 keeps 12 digits of 123..).
    let mut leading_zeros_pending = int_part.is_zero();
    let mut produced = 0usize;
    let mut guard = 0usize;
    while produced < frac_len && !rem.is_zero() && guard < 512 {
        guard += 1;
        rem *= &ten;
        let digit = (&rem / &den).to_string().parse::<u8>().unwrap_or(0);
        rem = &rem % &den;
        if leading_zeros_pending && digit == 0 {
            frac_digits.push(0);
            continue;
        }
        leading_zeros_pending = false;
        frac_digits.push(digit);
        produced += 1;
    }
    // Round half-up on the next digit.
    if !rem.is_zero() {
        let next = ((&rem * &ten) / &den)
            .to_string()
            .parse::<u8>()
            .unwrap_or(0);
        if next >= 5 {
            let mut i = frac_digits.len();
            loop {
                if i == 0 {
                    int_part += 1;
                    break;
                }
                i -= 1;
                if frac_digits[i] == 9 {
                    frac_digits[i] = 0;
                } else {
                    frac_digits[i] += 1;
                    break;
                }
            }
        }
    }
    while frac_digits.last() == Some(&0) {
        frac_digits.pop();
    }
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    out.push_str(&int_part.to_string());
    if !frac_digits.is_empty() {
        out.push('.');
        for d in frac_digits {
            out.push(char::from(b'0' + d));
        }
    }
    out
}
