//! Fraction addition rules.
//!
//! This module contains rules for adding terms with fractions:
//! - `FoldAddIntoFractionRule`: k + p/q → (k·q + p)/q
//! - `AddFractionsRule`: a/b + c/d → (ad+bc)/bd

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::expr_classify::is_trig_function;
use cas_math::fraction_add_rewrite_support::{
    plan_add_fraction_rewrite_with, AddFractionRewriteInput,
};
use cas_math::fraction_add_rule_support::{
    try_plan_fold_add_into_fraction_rewrite, try_plan_sub_term_matches_denom_rewrite,
};
use cas_math::fraction_pair_guard_support::{
    should_block_add_fraction_pair, should_block_sub_fraction_pair, AddFractionPairGuardInput,
    SubFractionPairGuardInput,
};
use cas_math::fraction_pair_support::extract_fraction_pair;
use cas_math::fraction_sub_rewrite_support::plan_sub_fraction_rewrite_with;

// =============================================================================
// Fold Add Into Fraction: k + p/q → (k·q + p)/q
// =============================================================================
//
// This rule combines a simple term with a fraction into a single fraction.
// Unlike AddFractionsRule, this always fires when k is "simple enough"
// (Number, Variable, or simple polynomial) to produce canonical rational form.
//
// Examples:
// - 1 + (x+1)/(2x+1) → (3x+2)/(2x+1)
// - x + 1/y → (x·y + 1)/y
// - 2 + 3/x → (2x + 3)/x
//
// Guards:
// - Skip if inside trig arguments (preserve sin(a + pi/9) structure)
// - Skip if k contains functions (preserve arctan(x) + 1/y structure)

define_rule!(
    FoldAddIntoFractionRule,
    "Common Denominator",
    |ctx, expr, parent_ctx| {
        // Match Add(l, r) where one is a fraction and the other is not
        let (l, r) = cas_math::expr_destructure::as_add(ctx, expr)?;

        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        // Guard: Skip if this expression is inside a fraction (numerator OR denominator)
        // Let SimplifyComplexFraction handle nested cases properly
        // This prevents preemptive simplification of 1 + x/(x+1) when it's in a complex fraction
        let inside_fraction = parent_ctx
            .has_ancestor_matching(ctx, |c, node_id| matches!(c.get(node_id), Expr::Div(_, _)));

        let plan =
            try_plan_fold_add_into_fraction_rewrite(ctx, expr, l, r, inside_trig, inside_fraction)?;
        Some(Rewrite::new(plan.rewritten).desc(plan.desc))
    }
);

// =============================================================================
// SubTermMatchesDenomRule: a - b/a → (a² - b)/a
// =============================================================================
//
// When the denominator of a subtracted fraction matches the other term,
// combine them into a single fraction. This pattern always reduces nesting
// and is essential for trig simplification:
//   cos(x) - sin²(x)/cos(x) → (cos²(x) - sin²(x))/cos(x) → cos(2x)/cos(x)
//
// This rule complements FoldAddIntoFractionRule (which handles Add only)
// by specifically targeting the Sub case where the denominator matches.
//
// Guard: Skip inside trig arguments and inside fractions (same as FoldAddIntoFraction).

define_rule!(
    SubTermMatchesDenomRule,
    "Combine Same Denominator Sub",
    |ctx, expr, parent_ctx| {
        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        let plan = try_plan_sub_term_matches_denom_rewrite(ctx, expr, inside_trig)?;
        Some(Rewrite::new(plan.rewritten).desc(plan.desc))
    }
);

define_rule!(
    AddFractionsRule,
    "Add Fractions",
    |ctx, expr, parent_ctx| {
        // Use zero-clone destructuring
        let (l, r) = cas_math::expr_destructure::as_add(ctx, expr)?;

        let parts = extract_fraction_pair(ctx, l, r);
        let (n1, d1, is_frac1) = (parts.n1, parts.d1, parts.is_frac1);
        let (n2, d2, is_frac2) = (parts.n2, parts.d2, parts.is_frac2);

        if should_block_add_fraction_pair(
            ctx,
            AddFractionPairGuardInput {
                l,
                r,
                n1,
                d1,
                is_frac1,
                n2,
                d2,
                is_frac2,
            },
        ) {
            return None;
        }

        // V2.15.8: Detect same-sign fractions for growth allowance
        // (same_sign = both positive or both negative; opposite = one +, one -)
        let same_sign = parts.sign1 == parts.sign2;

        // Context-aware gating: avoid combining symbol + pi-const inside trig functions.
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });

        let plan = plan_add_fraction_rewrite_with(
            ctx,
            AddFractionRewriteInput {
                expr,
                l,
                r,
                n1,
                d1,
                n2,
                d2,
                same_sign,
                inside_trig,
            },
            crate::expand::expand,
        )?;
        Some(Rewrite::new(plan.rewritten).desc(plan.desc()))
    }
);

// =============================================================================
// SubFractionsRule: a/b - c/d → (a·d - c·b) / (b·d)
// =============================================================================
//
// Combines two fractions being subtracted into a single fraction.
// The resulting numerator goes through normal simplification which can prove
// it equals 0 when the fractions were algebraically equal (e.g., different
// representations of the same rational expression).
//
// This handles cases that SubSelfToZeroRule misses because the two fractions
// have structurally different (but algebraically equivalent) numerators/denominators
// from independent simplification paths.
//
// Example: ((u+1)·(u·x+1)+u)/(u·(u+1)) - (u²x+ux+2u+1)/(u²+u)
//        → (cross_product) / (common_den) → 0/den → 0
//
// Guards:
// - Both sides must be fractions (direct Div or FractionParts)
// - Skip if inside trig arguments (preserve sin(a - pi/9) structure)
// - Skip function-containing expressions mixed with constant fractions
// - Same complexity heuristics as AddFractionsRule

define_rule!(
    SubFractionsRule,
    "Subtract Fractions",
    |ctx, expr, parent_ctx| {
        let (l, r) = cas_math::expr_destructure::as_sub(ctx, expr)?;

        let parts = extract_fraction_pair(ctx, l, r);
        let (n1, d1, is_frac1) = (parts.n1, parts.d1, parts.is_frac1);
        let (n2, d2, is_frac2) = (parts.n2, parts.d2, parts.is_frac2);

        // Guard: Skip if inside trig function argument
        let inside_trig = parent_ctx.has_ancestor_matching(ctx, |c, node_id| {
            matches!(c.get(node_id), Expr::Function(fn_id, _) if is_trig_function(c, *fn_id))
        });
        if should_block_sub_fraction_pair(
            ctx,
            SubFractionPairGuardInput {
                l,
                r,
                n1,
                d1,
                is_frac1,
                n2,
                d2,
                is_frac2,
                inside_trig,
            },
        ) {
            return None;
        }

        let plan = plan_sub_fraction_rewrite_with(ctx, n1, n2, d1, d2, crate::expand::expand);
        Some(Rewrite::new(plan.rewritten).desc(plan.desc()))
    }
);
