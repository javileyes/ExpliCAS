//! Fraction combination rules for n-ary Add/Sub chains.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{count_nodes, Expr, ExprId};
use cas_math::expr_rewrite::count_div_nodes;
use cas_math::fraction_combine_policy_support::group_fraction_terms_by_denominator;
use std::cmp::Ordering;

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
        use crate::domain::Proof;
        use crate::helpers::prove_nonzero;
        use cas_math::trig_roots_flatten::flatten_add_sub_chain;

        // Only handle Add or Sub at root
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // Flatten the Add/Sub chain into individual terms
        let terms = flatten_add_sub_chain(ctx, expr);
        if terms.len() < 2 {
            return None;
        }

        // Group terms by structurally equivalent denominators.
        let (denom_groups, non_fraction_indices) = group_fraction_terms_by_denominator(ctx, &terms);

        // Find groups with more than one fraction (those can be combined)
        #[allow(clippy::type_complexity)]
        let mut combinable_groups: Vec<(ExprId, Vec<(usize, ExprId, bool)>)> = Vec::new();
        for (den, group) in denom_groups.iter() {
            if group.len() >= 2 {
                combinable_groups.push((*den, group.clone()));
            }
        }

        // If no groups can be combined, nothing to do
        if combinable_groups.is_empty() {
            return None;
        }

        // Take the first combinable group
        let (common_den, group) = &combinable_groups[0];

        // DOMAIN MODE GATE: Check if denominator is provably non-zero
        let den_nonzero = prove_nonzero(ctx, *common_den);
        let policy =
            cas_math::fraction_combine_policy_support::decide_combine_same_denominator_policy(
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
                den_nonzero == Proof::Proven,
            );
        // Note: assumption_events not yet emitted for this rule.
        let _domain_assumption: Option<&str> = match policy {
            cas_math::fraction_combine_policy_support::CombineSameDenominatorPolicy::Block => {
                return None;
            }
            cas_math::fraction_combine_policy_support::CombineSameDenominatorPolicy::Apply {
                assume_denominator_nonzero: true,
            } => Some("Assuming denominator ≠ 0"),
            cas_math::fraction_combine_policy_support::CombineSameDenominatorPolicy::Apply {
                assume_denominator_nonzero: false,
            } => None,
        };

        // Combine numerators: n1 + n2 + ... (handle negation)
        let combined_num_terms: Vec<ExprId> = group
            .iter()
            .map(|(_, num, is_neg)| {
                if *is_neg {
                    ctx.add(Expr::Neg(*num))
                } else {
                    *num
                }
            })
            .collect();

        let combined_num = if combined_num_terms.len() == 1 {
            combined_num_terms[0]
        } else {
            let mut acc = combined_num_terms[0];
            for term in &combined_num_terms[1..] {
                acc = ctx.add(Expr::Add(acc, *term));
            }
            acc
        };

        // Create combined fraction
        let combined_fraction = ctx.add(Expr::Div(combined_num, *common_den));

        // Build new expression: non-combined terms + combined fraction
        let _combined_indices: Vec<usize> = group.iter().map(|(idx, _, _)| *idx).collect();

        let mut new_terms: Vec<ExprId> = Vec::new();

        // Add non-fraction terms
        for &idx in &non_fraction_indices {
            new_terms.push(terms[idx]);
        }

        // Add uncombined fractions (those not in the combined group)
        for (den, single_group) in denom_groups.iter() {
            if single_group.len() == 1
                && crate::ordering::compare_expr(ctx, *den, *common_den) != Ordering::Equal
            {
                let (idx, _, _) = single_group[0];
                new_terms.push(terms[idx]);
            }
        }

        // Add the combined fraction
        new_terms.push(combined_fraction);

        // Build result expression
        if new_terms.is_empty() {
            return None;
        }

        let result = if new_terms.len() == 1 {
            new_terms[0]
        } else {
            let mut acc = new_terms[0];
            for term in &new_terms[1..] {
                acc = ctx.add(Expr::Add(acc, *term));
            }
            acc
        };

        // Avoid no-op
        if count_nodes(ctx, result) >= count_nodes(ctx, expr) {
            // Only proceed if we actually reduced something or combined fractions
            // Check that we did combine (reduced number of Div nodes)
            let old_divs = count_div_nodes(ctx, expr);
            let new_divs = count_div_nodes(ctx, result);
            if new_divs >= old_divs {
                return None;
            }
        }

        // ===== FOCUS CONSTRUCTION =====
        // Capture original fraction terms exactly as they appear (preserving signs)
        // This enables didactic display showing only the combined fractions
        let original_fractions: Vec<ExprId> = group
            .iter()
            .map(|&(idx, _, _)| terms[idx]) // term already has its sign
            .collect();

        // Build focus_before from original fraction terms
        let focus_before = if original_fractions.len() == 1 {
            original_fractions[0]
        } else {
            let mut acc = original_fractions[0];
            for &term in &original_fractions[1..] {
                acc = ctx.add(Expr::Add(acc, term));
            }
            acc
        };

        // focus_after is the combined fraction
        let focus_after = combined_fraction;

        Some(
            Rewrite::new(result)
                .desc("Combine fractions with same denominator")
                .local(focus_before, focus_after),
        )
    }
);
