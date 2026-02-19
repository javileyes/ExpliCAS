//! Fraction combination rules for n-ary Add/Sub chains.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{count_nodes, Context, Expr, ExprId};
use cas_math::expr_rewrite::count_div_nodes;
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
        use crate::helpers::{flatten_add_sub_chain, prove_nonzero};
        use std::collections::HashMap;

        // Only handle Add or Sub at root
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // Flatten the Add/Sub chain into individual terms
        let terms = flatten_add_sub_chain(ctx, expr);
        if terms.len() < 2 {
            return None;
        }

        // Helper to get (num, den, is_negated) from a Div term - does NOT modify ctx
        let get_fraction = |ctx: &Context, term: ExprId| -> Option<(ExprId, ExprId, bool)> {
            match ctx.get(term) {
                Expr::Div(num, den) => Some((*num, *den, false)),
                Expr::Neg(inner) => {
                    if let Expr::Div(num, den) = ctx.get(*inner) {
                        // Neg(Div(n, d)) → mark as negated
                        Some((*num, *den, true))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        // Group terms by denominator
        // Key: ExprId of denominator, Value: list of (term_index, numerator, is_negated)
        let mut denom_groups: HashMap<ExprId, Vec<(usize, ExprId, bool)>> = HashMap::new();
        let mut non_fraction_indices: Vec<usize> = Vec::new();

        for (idx, &term) in terms.iter().enumerate() {
            if let Some((num, den, is_neg)) = get_fraction(ctx, term) {
                // Check if we already have this denominator (by structural equality)
                let mut found_key = None;
                for existing_den in denom_groups.keys() {
                    if crate::ordering::compare_expr(ctx, *existing_den, den) == Ordering::Equal {
                        found_key = Some(*existing_den);
                        break;
                    }
                }

                if let Some(key) = found_key {
                    if let Some(v) = denom_groups.get_mut(&key) {
                        v.push((idx, num, is_neg));
                    } else {
                        // Should be unreachable because `key` comes from the map's keys.
                        denom_groups.insert(key, vec![(idx, num, is_neg)]);
                    }
                } else {
                    denom_groups.insert(den, vec![(idx, num, is_neg)]);
                }
            } else {
                non_fraction_indices.push(idx);
            }
        }

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
        let domain_mode = parent_ctx.domain_mode();

        // Determine if we should proceed and what warning to emit
        // Note: assumption_events not yet emitted for this rule
        let _domain_assumption: Option<&str> = match domain_mode {
            crate::DomainMode::Strict => {
                // Only combine if denominator is provably non-zero
                if den_nonzero != Proof::Proven {
                    return None;
                }
                None
            }
            crate::DomainMode::Assume => {
                // Combine with warning if not proven
                if den_nonzero != Proof::Proven {
                    Some("Assuming denominator ≠ 0")
                } else {
                    None
                }
            }
            crate::DomainMode::Generic => {
                // Educational mode: combine unconditionally
                None
            }
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
