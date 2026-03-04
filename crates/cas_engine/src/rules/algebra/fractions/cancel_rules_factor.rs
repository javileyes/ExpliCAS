//! Factor-based cancellation and rationalization rules.
//!
//! Contains `CancelCommonFactorsRule` (cancel shared factors in num/den),
//! `RationalizeProductDenominatorRule` (rationalize product denominators),
//! and supporting helper functions.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::fraction_factors::try_rewrite_cancel_common_factors_expr_with;
use cas_math::fraction_factors::CancelCommonFactorsGate;
use cas_math::rationalize_diff_squares_support::try_rewrite_rationalize_product_denominator_expr;

define_rule!(
    RationalizeProductDenominatorRule,
    "Rationalize Product Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewrite = try_rewrite_rationalize_product_denominator_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Predicate;

        // Capture domain mode once at start
        let domain_mode = parent_ctx.domain_mode();
        // NOTE: Pythagorean identity simplification (k - k*sin² → k*cos²) has been
        // extracted to TrigPythagoreanSimplifyRule for pedagogical clarity.
        // CancelCommonFactorsRule now does pure factor cancellation.

        let rewrite = try_rewrite_cancel_common_factors_expr_with(
            ctx,
            expr,
            |ctx, nonzero_base, _emit_assumption| {
                let decision = crate::oracle_allows_with_hint(
                    ctx,
                    domain_mode,
                    parent_ctx.value_domain(),
                    &Predicate::NonZero(nonzero_base),
                    "Cancel Common Factors",
                );
                CancelCommonFactorsGate {
                    allow: decision.allow,
                    assumed: decision.assumption.is_some(),
                }
            },
        )?;
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> =
            rewrite
                .assumed_nonzero_targets
                .into_iter()
                .map(|target| crate::AssumptionEvent::nonzero(ctx, target))
                .collect();

        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("Cancel common factors")
                .local(expr, rewrite.rewritten)
                .assume_all(assumption_events),
        )
    }
);
