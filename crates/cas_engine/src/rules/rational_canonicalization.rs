use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::rational_canonicalization_support::{
    try_rewrite_nested_pow_canonical_expr, try_rewrite_rational_div_canonical_expr,
};

// ──────────────────────────────────────────────────────────────────────
// CanonicalizeRationalDivRule: Div(Number(p), Number(q)) → Number(p/q)
//
// Unifies the two representations of exact rationals that occur in the
// engine: the parser produces `Div(Number(5), Number(6))` while the
// simplifier's `add_exp` merges exponents into `Number(5/6)`.
// Without this rule, structurally different ASTs for the same value
// cause fingerprint mismatches, failed cancellations, and solver cycles.
//
// Guards:
// • q must be non-zero
// • Both operands must be exact Number (BigRational) — no floats
// • Result is reduced (BigRational always normalises gcd + sign)
// ──────────────────────────────────────────────────────────────────────
define_rule!(
    CanonicalizeRationalDivRule,
    "Canonicalize Rational Division",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_rational_div_canonical_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("p / q = p/q (exact rational)"))
    }
);

// ──────────────────────────────────────────────────────────────────────
// CanonicalizeNestedPowRule: Pow(Pow(base, k), r) → Pow(base, k*r)
//
// Folds nested exponentiation when it is DOMAIN-SAFE in ℝ.
//
// The rewrite (x^k)^r = x^(k·r) is ONLY valid in ℝ when:
//   • r is integer (q=1)                           → always safe
//   • r = p/q with q odd                           → safe (odd root)
//   • r = p/q with q even AND k is odd             → safe
//   • r = p/q with q even AND k is even            → NOT SAFE
//     (classic example: (x^2)^(1/2) = |x| ≠ x)
//
// This avoids the `(x^2)^(1/2) → x` bug while still folding safe cases
// like `sqrt(x^3) = (x^3)^(1/2) → x^(3/2)`.
// ──────────────────────────────────────────────────────────────────────
define_rule!(
    CanonicalizeNestedPowRule,
    "Canonicalize Nested Power",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_nested_pow_canonical_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("(x^k)^r = x^(k·r)"))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CanonicalizeRationalDivRule));
    simplifier.add_rule(Box::new(CanonicalizeNestedPowRule));
}
