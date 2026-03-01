//! Binomial conjugate rationalization and difference canonicalization rules.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::difference_product_rule_support::{
    try_rewrite_absorb_negation_into_difference_expr, try_rewrite_canonical_difference_product_expr,
};
use cas_math::rationalize_binomial_surd_support::rewrite_rationalize_binomial_surd_expr_with;

// ========== Binomial Conjugate Rationalization (Level 1) ==========
// Transforms: num / (A + B√n) → num * (A - B√n) / (A² - B²·n)
// Only applies when:
// - denominator is a binomial with exactly one numeric surd term
// - A, B are rational, n is a positive integer
// - uses closed-form arithmetic (no calls to general simplifier)

define_rule!(
    RationalizeBinomialSurdRule,
    "Rationalize Binomial Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewritten =
            rewrite_rationalize_binomial_surd_expr_with(ctx, expr, cas_formatter::render_expr)?;
        Some(Rewrite::new(rewritten.0).desc(rewritten.1))
    }
);

// ============================================================================
// R1: Absorb Negation Into Difference Factor
// ============================================================================
// -1/((x-y)*...) → 1/((y-x)*...)
// Absorbs the negative sign by flipping one difference in the denominator.
// Differences can be Sub(x,y) or Add(x, Neg(y)) or Add(x, Mul(-1,y)).

define_rule!(
    AbsorbNegationIntoDifferenceRule,
    "Absorb Negation Into Difference",
    |ctx, expr| {
        let rewrite = try_rewrite_absorb_negation_into_difference_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("Absorb negation into difference factor"))
    }
);

// ============================================================================
// R2: Canonicalize Products of Same-Tail Differences
// ============================================================================
// 1/((p-t)*(q-t)) → 1/((t-p)*(t-q))
// When two difference factors share the same "tail" (right operand),
// flip both to have that common element first.
// Double-flip preserves the overall sign.

define_rule!(
    CanonicalDifferenceProductRule,
    "Canonicalize Difference Product",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_canonical_difference_product_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("Canonicalize same-tail difference product"))
    }
);
