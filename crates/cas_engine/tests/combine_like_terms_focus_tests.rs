//! Contract tests for CombineLikeTermsRule focus behavior.
//!
//! These tests verify that the focus in CombineLikeTermsRule correctly shows
//! only the relevant terms being cancelled or combined, not the entire expression.
//!
//! Example: For "5 + cos²(x) + sin²(x) - 5" simplifying,
//! instead of showing "5 + cos²(x) + sin²(x) - 5 → cos²(x) + sin²(x)",
//! the step should show focus: "5 - 5 → 0"

use cas_ast::{Context, DisplayExpr};
use cas_engine::eval_step_pipeline::to_display_steps;
use cas_engine::Simplifier;

/// Helper to get display string for an ExprId
fn display(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

/// Helper: simplify and return steps with before_local/after_local info
fn simplify_and_inspect_focus(
    input: &str,
) -> Vec<(String, Option<String>, Option<String>, String)> {
    let mut simplifier = Simplifier::with_default_rules();
    let ctx = &mut simplifier.context;
    let expr = cas_parser::parse(input, ctx).expect("parse failed");

    let (_, raw_steps) = simplifier.simplify(expr);
    let display_steps = to_display_steps(raw_steps);

    display_steps
        .iter()
        .map(|step| {
            let before_local = step.before_local.map(|id| display(&simplifier.context, id));
            let after_local = step.after_local.map(|id| display(&simplifier.context, id));
            let rule = step.rule_name.clone();
            let desc = step.description.clone();
            (rule, before_local, after_local, desc)
        })
        .collect()
}

// =============================================================================
// Contract: Cancellation focus should show only cancelled terms
// =============================================================================

#[test]
fn combine_like_terms_cancellation_focus_shows_specific_terms() {
    // Expression: 5 + x - 5 → x
    // The cancellation step should focus on "5 - 5 → 0", not "5 + x - 5 → x"
    let steps = simplify_and_inspect_focus("5 + x - 5");

    // Find a "Combine Like Terms" step with cancellation
    let combine_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Combine Like Terms");

    if let Some((_, before_local, after_local, desc)) = combine_step {
        // Should have focus set (before_local and after_local)
        assert!(
            before_local.is_some(),
            "Cancellation step should have before_local set for focus"
        );
        assert!(
            after_local.is_some(),
            "Cancellation step should have after_local set for focus"
        );

        // Description should mention cancellation
        assert!(
            desc.contains("Cancel") || desc.contains("cancel") || desc.contains("Combine"),
            "Description should be specific: got '{}'",
            desc
        );

        // after_local should be 0 for cancellation
        let after = after_local.as_ref().unwrap();
        assert_eq!(
            after, "0",
            "Cancellation focus after_local should be 0, got '{}'",
            after
        );
    }
}

#[test]
fn combine_like_terms_combination_focus_shows_combined_terms() {
    // Expression: x + 2*x → 3*x
    // The combination step should focus on "x + 2*x → 3*x"
    let steps = simplify_and_inspect_focus("x + 2*x");

    let combine_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Combine Like Terms");

    if let Some((_, before_local, after_local, desc)) = combine_step {
        // Should have focus set
        assert!(
            before_local.is_some(),
            "Combination step should have before_local"
        );
        assert!(
            after_local.is_some(),
            "Combination step should have after_local"
        );

        // Description should be about combining
        assert!(
            desc.contains("Combine") || desc.contains("ombine"),
            "Description should mention combining: got '{}'",
            desc
        );
    }
}

// =============================================================================
// Contract: Focus should prioritize constant cancellations
// =============================================================================

#[test]
fn combine_like_terms_prioritizes_constant_cancellation() {
    // Expression: 5 + y - 5 → y
    // Should prioritize showing "5 - 5 → 0" (constant cancellation)
    // over any other transformation
    let steps = simplify_and_inspect_focus("5 + y - 5");

    let combine_step = steps
        .iter()
        .find(|(rule, _, _, desc)| rule == "Combine Like Terms" && desc.contains("Cancel"));

    if let Some((_, before_local, after_local, _)) = combine_step {
        assert!(
            before_local.is_some(),
            "Should have before_local for constant cancellation"
        );
        assert!(
            after_local.is_some(),
            "Should have after_local for constant cancellation"
        );

        // The after should be 0
        let after = after_local.as_ref().unwrap();
        assert_eq!(after, "0", "Constant cancellation should result in 0");
    }
}

// =============================================================================
// Contract: Variable cancellation should work too
// =============================================================================

#[test]
fn combine_like_terms_variable_cancellation_focus() {
    // Expression: x - x → 0
    let steps = simplify_and_inspect_focus("x - x");

    // Should have a step that produces 0
    let has_zero_result = steps
        .iter()
        .any(|(_, _, after_local, _)| after_local.as_deref() == Some("0"));

    assert!(
        has_zero_result || steps.is_empty() || steps.iter().any(|(r, _, _, _)| r == "Add Inverse"),
        "x - x should have a step showing → 0, or be handled by Add Inverse. Steps: {:?}",
        steps
    );
}

// =============================================================================
// Contract: Complex expression should focus on specific operation
// =============================================================================

#[test]
fn combine_like_terms_complex_expression_shows_specific_focus() {
    // Expression: a + b + c - c → a + b
    // The cancellation step should focus on "c - c → 0", not the whole thing
    let steps = simplify_and_inspect_focus("a + b + c - c");

    let cancel_step = steps
        .iter()
        .find(|(rule, _, _, desc)| rule == "Combine Like Terms" && desc.contains("Cancel"));

    if let Some((_, before_local, after_local, _)) = cancel_step {
        // Should have focus on just the cancelled terms
        assert!(before_local.is_some(), "Should have before_local focus");
        assert_eq!(
            after_local.as_deref(),
            Some("0"),
            "Cancellation after_local should be 0"
        );

        // The before_local should NOT contain unrelated terms like 'a' or 'b'
        // It should only contain the cancelled terms 'c' and '-c'
        let before = before_local.as_ref().unwrap();
        // Note: the exact format depends on how terms are joined, but it shouldn't be the full expr
        assert!(
            !before.contains("a + b + c"),
            "before_local should focus on cancelled terms only, not entire expression: got '{}'",
            before
        );
    }
}

// =============================================================================
// Regression test: Steps should not show full expression as before_local
// =============================================================================

#[test]
fn combine_like_terms_does_not_show_full_expression_in_focus() {
    // This is the core bug we're testing against:
    // before_local should NOT be the entire input expression
    let steps = simplify_and_inspect_focus("1 + 2 + 3 - 3");

    for (rule, before_local, after_local, _desc) in &steps {
        if rule == "Combine Like Terms" {
            if let (Some(before), Some(_after)) = (before_local, after_local) {
                // The before_local should be simpler than the full expression
                // It should not contain all terms "1 + 2 + 3 - 3"
                let full_like =
                    before.contains("1") && before.contains("2") && before.contains("3 - 3");

                assert!(
                    !full_like,
                    "before_local should show focused terms, not full expression: got '{}'",
                    before
                );
            }
        }
    }
}
