//! Rationalization and cancellation rules for fractions.
//!
//! Addition rules (FoldAddIntoFractionRule, AddFractionsRule) have been
//! extracted to `addition_rules.rs`.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::rationalize_diff_squares_support::try_rewrite_cancel_nth_root_binomial_factor_expr;
use cas_math::rationalize_diff_squares_support::try_rewrite_generalized_rationalization_expr;
use cas_math::rationalize_diff_squares_support::try_rewrite_rationalize_denominator_diff_squares_expr;
use cas_math::rationalize_diff_squares_support::try_rewrite_rationalize_nth_root_binomial_expr;
use cas_math::rationalize_diff_squares_support::try_rewrite_sqrt_conjugate_collapse_expr_with;
use cas_math::rationalize_diff_squares_support::SqrtConjugateCollapseGate;

define_rule!(
    RationalizeDenominatorRule,
    "Rationalize Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewrite = try_rewrite_rationalize_denominator_diff_squares_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("Rationalize denominator (diff squares)"))
    }
);

// Rationalize binomial denominators with nth roots (n >= 3) using geometric sum.
// For a^(1/n) - r, multiply by sum_{k=0}^{n-1} a^((n-1-k)/n) * r^k
// This gives denominator a - r^n
define_rule!(
    RationalizeNthRootBinomialRule,
    "Rationalize Nth Root Binomial",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewrite = try_rewrite_rationalize_nth_root_binomial_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Cancel nth root binomial factors: (u ± r^n) / (u^(1/n) ± r) = geometric series
// Example: (x + 1) / (x^(1/3) + 1) = x^(2/3) - x^(1/3) + 1
// Uses identity: a^n - b^n = (a-b)(a^(n-1) + a^(n-2)b + ... + b^(n-1))
//            and: a^n + b^n = (a+b)(a^(n-1) - a^(n-2)b + ... ± b^(n-1)) for odd n
define_rule!(
    CancelNthRootBinomialFactorRule,
    "Cancel Nth Root Binomial Factor",
    None,
    PhaseMask::TRANSFORM | PhaseMask::POST,
    |ctx, expr| {
        let rewrite = try_rewrite_cancel_nth_root_binomial_factor_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Collapse sqrt(A) * B → sqrt(B) when A and B are conjugates with A*B = 1
// Example: sqrt(x + sqrt(x²-1)) * (x - sqrt(x²-1)) → sqrt(x - sqrt(x²-1))
// This works because (p + s)(p - s) = p² - s² = 1 when s = sqrt(p² - 1)
//
// IMPORTANT: This transformation requires `other` (the conjugate being lifted into sqrt)
// to be non-negative (≥ 0), which is an ANALYTIC condition. In Generic mode, this rule
// should be blocked with a hint. In Assume mode, it proceeds with "Assumed: other ≥ 0".
define_rule!(
    SqrtConjugateCollapseRule,
    "Collapse Sqrt Conjugate Product",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, parent_ctx| {
        use crate::semantics::ValueDomain;

        // Guard: Only apply in RealOnly domain (in Complex, sqrt has branch cuts)
        if parent_ctx.value_domain() != ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_sqrt_conjugate_collapse_expr_with(
            ctx,
            expr,
            |ctx, other| {
                // ================================================================
                // Analytic Gate: sqrt(other) requires other ≥ 0 (NonNegative)
                // This is an Analytic condition, blocked in Generic, allowed in Assume
                // ================================================================
                let decision = crate::oracle_allows_with_hint(
                    ctx,
                    parent_ctx.domain_mode(),
                    parent_ctx.value_domain(),
                    &crate::Predicate::NonNegative(other),
                    "Collapse Sqrt Conjugate Product",
                );
                SqrtConjugateCollapseGate {
                    allow: decision.allow,
                    assumed: decision.assumption.is_some(),
                }
            },
        )?;
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> = rewrite
            .assumed_nonnegative_target
            .map(|target| smallvec::smallvec![crate::AssumptionEvent::nonnegative(ctx, target)])
            .unwrap_or_default();

        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("Lift conjugate into sqrt")
                .assume_all(assumption_events),
        )
    }
);

define_rule!(
    GeneralizedRationalizationRule,
    "Generalized Rationalization",
    |ctx, expr| {
        let rewrite = try_rewrite_generalized_rationalization_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
