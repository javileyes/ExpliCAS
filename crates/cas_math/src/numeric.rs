// =============================================================================
// Numerical Helpers (Zero-Clone for Number inspection)
// =============================================================================

use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;
use num_traits::ToPrimitive;

/// Get a reference to the Number without cloning.
/// Use for inspection only; the reference is tied to the Context's lifetime.
#[inline]
pub fn as_number(ctx: &Context, id: ExprId) -> Option<&num_rational::BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n),
        _ => None,
    }
}

/// Try to extract an i64 value from a Number expression (without cloning).
/// Returns None if not a Number, not an integer, or doesn't fit in i64.
#[inline]
pub fn as_i64(ctx: &Context, id: ExprId) -> Option<i64> {
    match ctx.get(id) {
        Expr::Number(n) if n.is_integer() => n.to_integer().to_i64(),
        _ => None,
    }
}

/// GCD for rational numbers when both are integers; otherwise returns `1`.
pub fn gcd_rational(a: BigRational, b: BigRational) -> BigRational {
    if a.is_integer() && b.is_integer() {
        use num_integer::Integer;
        let num_a = a.to_integer();
        let num_b = b.to_integer();
        let g = num_a.gcd(&num_b);
        return BigRational::from_integer(g);
    }
    BigRational::one()
}

#[cfg(test)]
mod tests {
    use super::gcd_rational;
    use num_rational::BigRational;

    #[test]
    fn gcd_rational_integers() {
        let a = BigRational::from_integer(12.into());
        let b = BigRational::from_integer(18.into());
        let g = gcd_rational(a, b);
        assert_eq!(g, BigRational::from_integer(6.into()));
    }

    #[test]
    fn gcd_rational_non_integer_fallbacks_to_one() {
        let a = BigRational::new(3.into(), 2.into());
        let b = BigRational::from_integer(6.into());
        let g = gcd_rational(a, b);
        assert_eq!(g, BigRational::from_integer(1.into()));
    }
}
