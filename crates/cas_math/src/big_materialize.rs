//! Exact materialization of astronomically large numeric expressions —
//! `bignum(123456!)`, `bignum(2^1234567)` — the deliberate, COSTED opposite
//! of the scientific-notation lane in [`crate::sci_approx`].
//!
//! Everything here is exact BigInt/BigRational arithmetic; the "approximate"
//! part is only the COST GATE: before any digit is computed, each node's
//! result size is bounded (bit counts for powers, Stirling for factorials)
//! and anything past [`BIGNUM_MAX_DIGITS`] declines with `None` — an honest
//! residual, produced in microseconds, instead of minutes of multiplication.
//! The estimates decide resource spending only, never a mathematical value.

use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// Ceiling on the total digits (numerator + denominator) a materialized
/// result may hold. `123456!` (574,965 digits) and `2^1234567` (371,642
/// digits) fit — about 13s and 5s end-to-end respectively; `5^123456789`
/// (86M digits) declines instantly.
///
/// The ceiling is set by the RENDER pipeline, not the arithmetic: the
/// product tree builds 1.5M-digit factorials in under a second, but each
/// binary→decimal conversion downstream is O(n²) in num-bigint and the
/// pipeline performs several (measured: `300000!` = 102s wall). Raising
/// this requires a divide-and-conquer `to_string` or caching the rendered
/// digits — not more arithmetic.
pub const BIGNUM_MAX_DIGITS: u64 = 600_000;

/// ~ bits per decimal digit (1 / log2(10)), with a hair of overestimate so
/// the gate errs toward declining.
const DIGITS_PER_BIT: f64 = 0.302;

/// Materialize `expr` into ONE exact rational, refusing (with `None`) any
/// intermediate whose size estimate crosses `max_digits`. Grammar: numbers,
/// `Neg`/`Mul`/`Div`, `Pow` with an integer literal exponent, and
/// `fact`/`factorial` of an integer literal; transparent through `__hold`
/// and the `decimal` display node. Symbolic content declines.
pub fn try_materialize_exact(ctx: &Context, expr: ExprId, max_digits: u64) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(r) => Some(r.clone()),
        Expr::Neg(inner) => Some(-try_materialize_exact(ctx, *inner, max_digits)?),
        Expr::Hold(inner) => try_materialize_exact(ctx, *inner, max_digits),
        Expr::Mul(l, r) => {
            let a = try_materialize_exact(ctx, *l, max_digits)?;
            let b = try_materialize_exact(ctx, *r, max_digits)?;
            if rational_bits(&a).checked_add(rational_bits(&b))? > bits_budget(max_digits) {
                return None;
            }
            Some(a * b)
        }
        Expr::Div(l, r) => {
            let a = try_materialize_exact(ctx, *l, max_digits)?;
            let b = try_materialize_exact(ctx, *r, max_digits)?;
            if b.is_zero() {
                return None;
            }
            if rational_bits(&a).checked_add(rational_bits(&b))? > bits_budget(max_digits) {
                return None;
            }
            Some(a / b)
        }
        Expr::Pow(base, exp) => {
            let e = integer_literal(ctx, *exp)?;
            let base_value = try_materialize_exact(ctx, *base, max_digits)?;
            pow_exact_bounded(&base_value, &e, max_digits)
        }
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.sym_name(*fn_id) {
            "fact" | "factorial" => {
                let n = integer_literal(ctx, args[0])?;
                if n.is_negative() {
                    return None;
                }
                let n = n.to_u64()?;
                if factorial_digits_estimate(n) > max_digits as f64 {
                    return None;
                }
                Some(BigRational::from_integer(factorial_product_tree(n)))
            }
            "decimal" => try_materialize_exact(ctx, args[0], max_digits),
            _ => None,
        },
        _ => None,
    }
}

fn integer_literal(ctx: &Context, expr: ExprId) -> Option<BigInt> {
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
    let value = r.numer().clone();
    Some(if negative { -value } else { value })
}

/// Would `bignum(expr)` produce NEW digits? True only for a SYMBOLIC numeric
/// expression (a bare number is already materialized) whose estimated size
/// passes the same gates the materialization itself applies. This is the
/// availability predicate for UI affordances (the web's «Calcular bignum»
/// button): sharing the gate formulas keeps the button from ever promising
/// what `bignum` would refuse — `12345!` offers, `300000!` (1.5M digits) and
/// `5^123456789` (86M) stay silent.
pub fn bignum_would_materialize(ctx: &Context, expr: ExprId) -> bool {
    if is_plain_number_form(ctx, expr) {
        return false;
    }
    match estimate_exact_digits(ctx, expr) {
        Some(digits) => digits <= BIGNUM_MAX_DIGITS as f64,
        None => false,
    }
}

/// A value that is ALREADY in materialized display form — a number, a
/// rational `a/b` of two numbers, or a negation of either. `bignum` on
/// these produces no new digits.
fn is_plain_number_form(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Neg(inner) => is_plain_number_form(ctx, *inner),
        Expr::Div(l, r) => {
            matches!(ctx.get(*l), Expr::Number(_)) && matches!(ctx.get(*r), Expr::Number(_))
        }
        _ => false,
    }
}

/// Estimated total digits (numerator + denominator) of the exact value,
/// `None` outside the materialization grammar. Advisory and cheap (O(tree),
/// never multiplies): powers and factorials use the SAME formulas as the
/// materialization gates; products/quotients sum child estimates.
fn estimate_exact_digits(ctx: &Context, expr: ExprId) -> Option<f64> {
    match ctx.get(expr) {
        Expr::Number(r) => Some(rational_bits(r) as f64 * DIGITS_PER_BIT),
        Expr::Neg(inner) | Expr::Hold(inner) => estimate_exact_digits(ctx, *inner),
        Expr::Mul(l, r) | Expr::Div(l, r) => {
            Some(estimate_exact_digits(ctx, *l)? + estimate_exact_digits(ctx, *r)?)
        }
        Expr::Pow(base, exp) => {
            let e = integer_literal(ctx, *exp)?;
            let magnitude = e.magnitude().to_u64()? as f64;
            match ctx.get(*base) {
                // Numeric base: precise via fractional log2 (same as the gate).
                Expr::Number(r) => {
                    let log2_base = log2_estimate(r.numer()) + log2_estimate(r.denom());
                    Some(magnitude * log2_base * DIGITS_PER_BIT + 1.0)
                }
                // Composite base: digits(b^e) ≤ e·digits(b) — conservative.
                _ => Some(magnitude * estimate_exact_digits(ctx, *base)?),
            }
        }
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.sym_name(*fn_id) {
            "fact" | "factorial" => {
                let n = integer_literal(ctx, args[0])?;
                if n.is_negative() {
                    return None;
                }
                Some(factorial_digits_estimate(n.to_u64()?))
            }
            "decimal" => estimate_exact_digits(ctx, args[0]),
            _ => None,
        },
        _ => None,
    }
}

fn rational_bits(r: &BigRational) -> u64 {
    r.numer().bits() + r.denom().bits()
}

fn bits_budget(max_digits: u64) -> u64 {
    // digits / 0.302 ≈ bits; saturate rather than overflow.
    ((max_digits as f64) / DIGITS_PER_BIT) as u64
}

/// `base^e` with the size gate applied BEFORE computing: the result holds
/// `|e| · log2(base parts)` bits, estimated to <0.01% with the fractional
/// log2 below (f64 is fine HERE — this decides resource spending with a
/// margin, never a mathematical value), so the decision costs nothing.
fn pow_exact_bounded(base: &BigRational, e: &BigInt, max_digits: u64) -> Option<BigRational> {
    if e.is_zero() {
        if base.is_zero() {
            return None; // 0^0: not this lane's call to make
        }
        return Some(BigRational::one());
    }
    let magnitude = e.magnitude().to_u64()?;
    if base.is_zero() {
        return if e.is_negative() {
            None // 1/0
        } else {
            Some(BigRational::zero())
        };
    }
    let log2_base = log2_estimate(base.numer()) + log2_estimate(base.denom());
    let projected_bits = (magnitude as f64) * log2_base + 2.0;
    if projected_bits > bits_budget(max_digits) as f64 {
        return None;
    }
    let exp_u32 = u32::try_from(magnitude).ok()?;
    let powered = num_traits::Pow::pow(base, exp_u32);
    Some(if e.is_negative() {
        powered.recip()
    } else {
        powered
    })
}

/// `log2(|x|)` to ~15 significant digits: exponent from the bit length plus
/// the fractional part from the top 53 bits.
fn log2_estimate(x: &BigInt) -> f64 {
    let bits = x.bits();
    if bits == 0 {
        return 0.0;
    }
    if bits <= 53 {
        return (x.magnitude().to_u64().expect("fits") as f64).log2();
    }
    let shift = bits - 53;
    let top = (x.magnitude() >> shift).to_u64().expect("53 bits fit");
    (top as f64).log2() + shift as f64
}

/// Stirling bound on `digits(n!)`, padded up — a resource gate, so
/// overestimating is the safe direction.
fn factorial_digits_estimate(n: u64) -> f64 {
    if n < 2 {
        return 1.0;
    }
    let x = n as f64;
    (x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI * x).ln()) / std::f64::consts::LN_10 + 2.0
}

/// Exact `n!` as a balanced product tree: multiplying similarly-sized
/// halves keeps the big multiplications in num-bigint's subquadratic range
/// (a plain left-fold multiplies a huge accumulator by tiny factors, which
/// is quadratic overall and minutes-slow at this scale).
fn factorial_product_tree(n: u64) -> BigInt {
    if n < 2 {
        return BigInt::one();
    }
    range_product(2, n)
}

fn range_product(lo: u64, hi: u64) -> BigInt {
    debug_assert!(lo <= hi);
    if hi - lo < 16 {
        let mut acc = BigInt::from(lo);
        for k in (lo + 1)..=hi {
            acc *= k;
        }
        return acc;
    }
    let mid = lo + (hi - lo) / 2;
    range_product(lo, mid) * range_product(mid + 1, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn materialize(source: &str) -> Option<BigRational> {
        let mut ctx = Context::new();
        let expr = parse(source, &mut ctx).expect("parse");
        try_materialize_exact(&ctx, expr, BIGNUM_MAX_DIGITS)
    }

    #[test]
    fn factorial_product_tree_matches_known_values() {
        assert_eq!(factorial_product_tree(0), BigInt::one());
        assert_eq!(factorial_product_tree(1), BigInt::one());
        assert_eq!(factorial_product_tree(10), BigInt::from(3_628_800u64));
        // 20! = 2432902008176640000 (fits u64)
        assert_eq!(
            factorial_product_tree(20),
            BigInt::from(2_432_902_008_176_640_000u64)
        );
        // Cross-check the tree against the naive fold at a splitting size.
        let mut naive = BigInt::one();
        for k in 2..=100u64 {
            naive *= k;
        }
        assert_eq!(factorial_product_tree(100), naive);
    }

    #[test]
    fn materializes_the_motivating_giants_with_exact_digits() {
        // 123456! — 574,965 digits; head verified externally.
        let f = materialize("123456!").expect("123456!");
        let s = f.numer().to_string();
        assert_eq!(s.len(), 574_965);
        assert!(s.starts_with("26040699049291378729"), "head: {}", &s[..30]);

        // 2^1234567 — 371,642 digits; head and tail verified externally.
        let p = materialize("2^1234567").expect("2^1234567");
        let s = p.numer().to_string();
        assert_eq!(s.len(), 371_642);
        assert!(s.starts_with("49963964863286026867"), "head: {}", &s[..30]);
        assert!(s.ends_with("70709952737787772928"), "tail");
    }

    #[test]
    fn declines_past_the_digit_ceiling_instantly() {
        // 86M digits — must refuse without computing.
        assert!(materialize("5^123456789").is_none());
        // 1.5M and 5.6M digits — past the render-bound ceiling.
        assert!(materialize("300000!").is_none());
        assert!(materialize("1000000!").is_none());
    }

    #[test]
    fn rational_and_negative_powers_stay_exact() {
        let r = materialize("(3/7)^1000").expect("(3/7)^1000");
        assert_eq!(r.numer().to_string().len(), 478); // 3^1000 digits
        let inv = materialize("2^(-100)").expect("2^-100");
        assert_eq!(inv.numer(), &BigInt::one());
        assert_eq!(inv.denom().to_string().len(), 31); // 2^100 digits

        let neg = materialize("(-2)^101").expect("(-2)^101");
        assert!(neg.is_negative());
    }

    #[test]
    fn bignum_availability_mirrors_the_materialization_gates() {
        let mut ctx = Context::new();
        let check = |ctx: &mut Context, src: &str| {
            let expr = parse(src, ctx).expect(src);
            bignum_would_materialize(ctx, expr)
        };
        // Symbolic giants under the ceiling: offer.
        assert!(check(&mut ctx, "2^1234567"));
        assert!(check(&mut ctx, "12345!"));
        assert!(check(&mut ctx, "-(2^1234567)"));
        assert!(check(&mut ctx, "1/2^1234567"));
        // Over the ceiling or out of grammar: silent.
        assert!(!check(&mut ctx, "300000!"));
        assert!(!check(&mut ctx, "5^123456789"));
        assert!(!check(&mut ctx, "2^(1/2)"));
        assert!(!check(&mut ctx, "x^123456789"));
        // Already materialized numbers: nothing to offer.
        assert!(!check(&mut ctx, "123456789"));
        assert!(!check(&mut ctx, "5/6"));
        assert!(!check(&mut ctx, "-42"));
    }

    #[test]
    fn declines_out_of_grammar_shapes() {
        for source in ["x!", "2^x", "2^(1/2)", "(-3)!", "0^0", "0^(-1)"] {
            assert!(materialize(source).is_none(), "{source} must decline");
        }
    }

    #[test]
    fn products_and_quotients_compose_with_the_gate() {
        let v = materialize("2^1000 * 3^500 / 7^200").expect("composite");
        assert!(!v.is_zero());

        // The Mul gate itself: two ~300-digit literals pass individually but
        // their product crosses a 500-digit ceiling.
        let mut ctx = Context::new();
        let big_literal = format!("1{}", "0".repeat(300));
        let expr = parse(&format!("{big_literal} * {big_literal}"), &mut ctx).expect("parse");
        assert!(try_materialize_exact(&ctx, expr, 500).is_none());
        // Each factor alone fits that same ceiling.
        let single = parse(&big_literal, &mut ctx).expect("parse");
        assert!(try_materialize_exact(&ctx, single, 500).is_some());
    }
}
