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
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CanonicalizeRationalDivRule));
    simplifier.add_rule(Box::new(CanonicalizeNestedPowRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, Expr};

    #[test]
    fn test_rational_div_canonicalization() {
        let mut ctx = Context::new();
        let rule = CanonicalizeRationalDivRule;
        let five = ctx.num(5);
        let six = ctx.num(6);
        let expr = ctx.add(Expr::Div(five, six));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
            assert_eq!(n.to_string(), "5/6");
        } else {
            panic!("Expected Number, got {:?}", ctx.get(rewrite.new_expr));
        }
    }

    #[test]
    fn test_rational_div_zero_denominator() {
        let mut ctx = Context::new();
        let rule = CanonicalizeRationalDivRule;
        let five = ctx.num(5);
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Div(five, zero));
        // Should NOT rewrite — division by zero
        assert!(rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .is_none());
    }

    #[test]
    fn test_nested_pow_safe_odd_k() {
        // (x^3)^(1/2) → x^(3/2) — safe because k=3 is odd
        let mut ctx = Context::new();
        let rule = CanonicalizeNestedPowRule;
        let x = ctx.var("x");
        let three = ctx.num(3);
        let inner = ctx.add(Expr::Pow(x, three));
        let half = ctx.rational(1, 2);
        let expr = ctx.add(Expr::Pow(inner, half));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should be Pow(x, 3/2)
        if let Expr::Pow(base, exp) = ctx.get(rewrite.new_expr) {
            assert!(matches!(ctx.get(*base), Expr::Variable(_)));
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(n.to_string(), "3/2");
            } else {
                panic!("Expected Number exponent");
            }
        } else {
            panic!("Expected Pow");
        }
    }

    #[test]
    fn test_nested_pow_unsafe_even_k_even_q() {
        // (x^2)^(1/2) — MUST NOT rewrite (would give x instead of |x|)
        let mut ctx = Context::new();
        let rule = CanonicalizeNestedPowRule;
        let x = ctx.var("x");
        let two = ctx.num(2);
        let inner = ctx.add(Expr::Pow(x, two));
        let half = ctx.rational(1, 2);
        let expr = ctx.add(Expr::Pow(inner, half));
        assert!(rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .is_none());
    }

    #[test]
    fn test_nested_pow_integer_exponent() {
        // (x^3)^2 → x^6 — safe because outer exponent is integer
        let mut ctx = Context::new();
        let rule = CanonicalizeNestedPowRule;
        let x = ctx.var("x");
        let three = ctx.num(3);
        let inner = ctx.add(Expr::Pow(x, three));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(inner, two));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        if let Expr::Pow(_, exp) = ctx.get(rewrite.new_expr) {
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(n.to_string(), "6");
            } else {
                panic!("Expected Number exponent");
            }
        } else {
            panic!("Expected Pow");
        }
    }
}
