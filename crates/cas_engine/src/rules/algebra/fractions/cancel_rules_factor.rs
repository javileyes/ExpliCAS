//! Factor-based cancellation and rationalization rules.
//!
//! Contains `CancelCommonFactorsRule` (cancel shared factors in num/den),
//! `RationalizeProductDenominatorRule` (rationalize product denominators),
//! and supporting helper functions.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::{ChainedRewrite, Rewrite};
use cas_math::fraction_factors::try_rewrite_cancel_common_factors_expr_with;
use cas_math::fraction_factors::CancelCommonFactorsGate;
use cas_math::fraction_power_cancel_support::try_rewrite_cancel_identical_fraction_expr;
use cas_math::rationalize_diff_squares_support::try_rewrite_rationalize_product_denominator_expr;

define_rule!(
    RationalizeProductDenominatorRule,
    "Rationalize Product Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewrite = try_rewrite_rationalize_product_denominator_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("Rationalize product denominator"))
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::ImplicitCondition;
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

        let mut out = Rewrite::new(rewrite.rewritten)
            .desc("Cancel common factors")
            .local(expr, rewrite.rewritten)
            .assume_all(assumption_events);

        // When factor cancellation exposes an identical residual fraction like x/x,
        // finish the closure here so Assume/Generic recover the symbolic NonZero
        // target instead of stopping at the partially cancelled form.
        if let Some(plan) = try_rewrite_cancel_identical_fraction_expr(ctx, rewrite.rewritten) {
            let decision = crate::oracle_allows_with_hint(
                ctx,
                domain_mode,
                parent_ctx.value_domain(),
                &Predicate::NonZero(plan.nonzero_target),
                "Cancel Common Factors",
            );

            if decision.allow {
                let mut cancel = ChainedRewrite::new(plan.rewritten)
                    .desc("Cancel: P/P -> 1")
                    .local(rewrite.rewritten, plan.rewritten)
                    .requires(ImplicitCondition::NonZero(plan.nonzero_target));
                for event in decision.assumption_events(ctx, plan.nonzero_target) {
                    cancel = cancel.assume(event);
                }
                out = out.chain(cancel);
            }
        }

        Some(out)
    }
);
