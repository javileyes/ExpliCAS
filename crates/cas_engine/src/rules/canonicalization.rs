use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::exponents_support::try_rewrite_exp_to_epow_expr;
use cas_math::expr_nary::{try_rewrite_canonicalize_add_expr, try_rewrite_canonicalize_mul_expr};
use cas_math::expr_sub_like::{
    try_rewrite_add_negative_constant_to_sub_expr, try_rewrite_cancel_fraction_signs_expr,
    try_rewrite_canonicalize_div_expr, try_rewrite_canonicalize_negation_expr,
    try_rewrite_neg_coeff_flip_binomial_expr, try_rewrite_neg_sub_flip_expr,
    try_rewrite_normalize_binomial_order_expr,
};
use cas_math::root_forms::try_rewrite_canonical_root_expr;

define_rule!(
    CanonicalizeNegationRule,
    "Canonicalize Negation",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_canonicalize_negation_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(CanonicalizeAddRule, "Canonicalize Addition", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let rewrite = try_rewrite_canonicalize_add_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

define_rule!(
    CanonicalizeMulRule,
    "Canonicalize Multiplication",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_canonicalize_mul_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(CanonicalizeDivRule, "Canonicalize Division", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let rewrite = try_rewrite_canonicalize_div_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

define_rule!(CanonicalizeRootRule, "Canonicalize Roots", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let rewrite = try_rewrite_canonical_root_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

define_rule!(NormalizeSignsRule, "Normalize Signs", |ctx, expr| {
    let rewrite = try_rewrite_add_negative_constant_to_sub_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// Normalize binomial order: (b-a) -> -(a-b) when a < b alphabetically
// This ensures consistent representation of binomials like (y-x) vs (x-y)
// so they can be recognized as opposites in fraction simplification.
define_rule!(
    NormalizeBinomialOrderRule,
    "Normalize Binomial Order",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_normalize_binomial_order_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule: -(a - b) → (b - a) ONLY when inner is non-canonical (a > b)
// This prevents the 2-cycle with NormalizeBinomialOrderRule:
// - Normalize: (a-b) → -(b-a) when a > b (produces canonical inner with b < a)
// - Flip: -(a-b) → (b-a) ONLY when a > b (inner is non-canonical)
// Since a > b and b < a are mutually exclusive, no cycle can occur.
define_rule!(
    NegSubFlipRule,
    "Flip Negative Subtraction",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_neg_sub_flip_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(rewrite.desc)
                .local(rewrite.inner, rewrite.rewritten),
        )
    }
);

// Rule: (-A)/(-B) → A/B - Cancel double negation in fractions
// This handles cases like (1-√x)/(1-x) → (√x-1)/(x-1)
// by recognizing that (1-√x) is the negation of (√x-1), etc.
//
// No loop risk: produces canonical order which won't match again.
define_rule!(
    CancelFractionSignsRule,
    "Cancel Fraction Signs",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_cancel_fraction_signs_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule: (-k) * (...) * (a - b) → k * (...) * (b - a) when k > 0
// This produces cleaner output like "1/2 * x * (√2 - 1)" instead of "-1/2 * x * (1 - √2)"
// No loop risk: produces positive coefficient which won't match again
define_rule!(
    NegCoeffFlipBinomialRule,
    "Flip binomial under negative coefficient",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_neg_coeff_flip_binomial_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
/// ExpToEPowRule: Convert exp(x) → e^x
///
/// GATE: Only applies in RealOnly mode.
/// In ComplexEnabled, exp(z) is univalued while e^z (via pow) could imply
/// multivalued logarithm semantics. Keeping exp() as a function preserves
/// the intended univalued semantics in complex domain.
///
/// This allows ExponentialLogRule to match e^(ln(x)) → x patterns.
pub struct ExpToEPowRule;

impl crate::rule::Rule for ExpToEPowRule {
    fn name(&self) -> &str {
        "Convert exp to Power"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::ValueDomain;

        // GATE: Only in RealOnly (exp is univalued; ComplexEnabled needs special handling)
        if parent_ctx.value_domain() != ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_exp_to_epow_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("exp(x) = e^x"))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // RE-ENABLED: Needed for -0 → 0 normalization
    // The non-determinism issue with Sub→Add(Neg) is now handled by canonical ordering
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));

    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeDivRule));
    simplifier.add_rule(Box::new(CancelFractionSignsRule)); // (-A)/(-B) → A/B
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(NormalizeSignsRule));
    // NormalizeBinomialOrderRule DISABLED - causes stack overflow in asin_acos tests
    // even with guarded NegSubFlipRule. The cycle likely involves other rules.
    // EvenPowSubSwapRule handles the specific (x-y)^2 - (y-x)^2 = 0 case safely.
    // simplifier.add_rule(Box::new(NormalizeBinomialOrderRule));
    simplifier.add_rule(Box::new(NegSubFlipRule)); // -(a-b) → (b-a) only when a > b
    simplifier.add_rule(Box::new(NegCoeffFlipBinomialRule)); // (-k)*(a-b) → k*(b-a)

    // exp(x) → e^x (RealOnly only - preserves complex semantics)
    simplifier.add_rule(Box::new(ExpToEPowRule));
}
