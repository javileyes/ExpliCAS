//! Fraction combination rules for n-ary Add/Sub chains.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::fraction_combine_policy_support::try_plan_same_denominator_combination_with;

// =============================================================================
// Combine same-denominator fractions in n-ary Add/Sub chains
// =============================================================================
// For expressions like: 1 + (a)/(d) - (b)/(d) + x
// This rule groups fractions with the same denominator and combines them:
// → 1 + (a - b)/(d) + x
//
// Domain Mode Policy:
// - Strict: only combine if prove_nonzero(d) == Proven
// - Assume: combine with domain_assumption warning "Assuming d ≠ 0"
// - Generic: combine unconditionally (educational mode)

define_rule!(
    CombineSameDenominatorFractionsRule,
    "Combine Same Denominator Fractions",
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;
        use crate::Proof;

        let plan = try_plan_same_denominator_combination_with(
            ctx,
            expr,
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            |c, den| prove_nonzero(c, den) == Proof::Proven,
        )?;
        // Note: assumption_events not yet emitted for this rule.
        let _domain_assumption: Option<&str> = if plan.assume_denominator_nonzero {
            Some("Assuming denominator ≠ 0")
        } else {
            None
        };

        Some(
            Rewrite::new(plan.build.result)
                .desc("Combine fractions with same denominator")
                .local(plan.build.focus_before, plan.build.focus_after),
        )
    }
);
