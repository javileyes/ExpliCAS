mod fractions;
mod integers;
mod terms;

use cas_ast::{Context, ExprId};
pub(crate) use integers::IsOne;
use num_bigint::BigInt;
use num_rational::BigRational;

/// Try to interpret an expression as a fraction (BigRational).
/// Handles both Number(n) and Div(Number, Number) patterns.
pub(crate) fn try_as_fraction(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    fractions::try_as_fraction(ctx, expr)
}

/// Collect all terms from an Add chain.
pub(crate) fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    terms::collect_add_terms(ctx, expr, terms)
}

/// Format a BigRational as a LaTeX fraction or integer.
pub(crate) fn format_fraction(r: &BigRational) -> String {
    fractions::format_fraction(r)
}

/// Compute LCM of two BigInts.
pub(crate) fn lcm_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    integers::lcm_bigint(a, b)
}

/// Compute GCD of two BigInts using Euclidean algorithm.
#[cfg(test)]
pub(crate) fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    integers::gcd_bigint(a, b)
}
