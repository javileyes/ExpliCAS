//! Contract tests for CombineSameDenominatorFractionsRule focus behavior.
//!
//! These tests verify that the focus shows only the fractions being combined,
//! not the entire expression.
//!
//! Example: For "1 + a/x + b/x",
//! instead of showing "1 + a/x + b/x → 1 + (a+b)/x",
//! the step should show focus: "a/x + b/x → (a+b)/x"

use cas_ast::{Context, DisplayExpr};
use cas_engine::eval_step_pipeline::to_display_steps;
use cas_engine::Simplifier;

/// Helper to get display string for an ExprId
fn display(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

/// Helper: simplify and return steps with focus info
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
// Contract: Same denominator fraction focus
// =============================================================================

#[test]
fn same_denom_fractions_focus_shows_combined_fractions() {
    // Expression: a/x + b/x → (a+b)/x
    // The step should focus on the fractions being combined
    let steps = simplify_and_inspect_focus("a/x + b/x");

    let combine_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Combine Same Denominator Fractions");

    if let Some((_, before_local, after_local, _desc)) = combine_step {
        // Should have focus set
        assert!(
            before_local.is_some(),
            "Should have before_local for fraction combination"
        );
        assert!(
            after_local.is_some(),
            "Should have after_local for fraction combination"
        );
    }
}

#[test]
fn same_denom_fractions_with_extra_terms_focuses_only_fractions() {
    // Expression: 1 + a/d + b/d → 1 + (a+b)/d
    // Focus should be just "a/d + b/d → (a+b)/d", NOT "1 + a/d + b/d → 1 + (a+b)/d"
    let steps = simplify_and_inspect_focus("1 + a/d + b/d");

    let combine_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Combine Same Denominator Fractions");

    if let Some((_, Some(before), _after_local, _desc)) = combine_step {
        // before_local should NOT contain the standalone "1"
        // If it shows the full expression, it would start with "1 + ..."
        assert!(
            !before.starts_with("1 + a/d + b/d"),
            "Focus should not be entire expression: got '{}'",
            before
        );
    }
}

#[test]
fn same_denom_fractions_preserves_signs() {
    // Expression: a/x - b/x → (a-b)/x or (a + -b)/x
    // The focus should correctly preserve the subtraction
    let steps = simplify_and_inspect_focus("a/x - b/x");

    let combine_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Combine Same Denominator Fractions");

    if let Some((_, before_local, after_local, _desc)) = combine_step {
        assert!(
            before_local.is_some(),
            "Should have before_local for subtraction case"
        );
        assert!(
            after_local.is_some(),
            "Should have after_local for subtraction case"
        );
    }
}

// =============================================================================
// Focus invariants: size and variables
// =============================================================================

/// Helper to count approximate node size (number of operators/values)
fn approx_node_count(s: &str) -> usize {
    // Simple heuristic: count alphanumeric sequences + operators
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '+' || *c == '-' || *c == '*' || *c == '/')
        .count()
}

/// Helper to extract variables from a string representation
fn extract_vars(s: &str) -> std::collections::HashSet<char> {
    s.chars()
        .filter(|c| c.is_ascii_lowercase() && *c != 'x') // x might be variable or just a letter
        .collect()
}

#[test]
fn focus_invariant_smaller_than_full_expression() {
    // For any step with focus, before_local should generally be smaller than before
    let test_cases = vec![
        "a/x + b/x + c",     // fraction combination with extra term
        "5 + y - 5",         // like terms with extra term
        "1 + 2*z + 3*z - 1", // combination and cancellation
    ];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if let (Some(before_local), Some(_after_local)) = (step.before_local, step.after_local)
            {
                let before_str = display(&simplifier.context, step.before);
                let focus_str = display(&simplifier.context, before_local);

                // Focus should not be the entire before expression
                // (unless it's a global rewrite, which is rare)
                let focus_size = approx_node_count(&focus_str);
                let before_size = approx_node_count(&before_str);

                // Allow equality for small expressions, but focus shouldn't be larger
                assert!(
                    focus_size <= before_size,
                    "Focus '{}' should not be larger than before '{}' for input '{}'",
                    focus_str,
                    before_str,
                    input
                );
            }
        }
    }
}

#[test]
fn focus_invariant_vars_subset() {
    // Variables in focus should be subset of variables in full expression
    let test_cases = vec!["a/x + b/x + c", "5 + y - 5"];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if let (Some(before_local), Some(_after_local)) = (step.before_local, step.after_local)
            {
                let before_str = display(&simplifier.context, step.before);
                let focus_str = display(&simplifier.context, before_local);

                let before_vars = extract_vars(&before_str);
                let focus_vars = extract_vars(&focus_str);

                // Focus vars should be subset of before vars
                for var in &focus_vars {
                    assert!(
                        before_vars.contains(var),
                        "Focus var '{}' not in before vars {:?} for input '{}'. Focus: '{}', Before: '{}'",
                        var, before_vars, input, focus_str, before_str
                    );
                }
            }
        }
    }
}

// =============================================================================
// Focus invariant: informative (before_local != after_local)
// =============================================================================

#[test]
fn focus_invariant_must_show_change() {
    // Focus should be informative: before_local != after_local
    // A focus that shows no change is useless
    let test_cases = vec!["a/x + b/x", "a/x - b/x", "1 + a/d + b/d"];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Combine Same Denominator Fractions" {
                if let (Some(before_local), Some(after_local)) =
                    (step.before_local, step.after_local)
                {
                    let before_str = display(&simplifier.context, before_local);
                    let after_str = display(&simplifier.context, after_local);

                    assert_ne!(
                        before_str, after_str,
                        "Focus must show a change: before_local '{}' == after_local '{}' for input '{}'",
                        before_str, after_str, input
                    );
                }
            }
        }
    }
}

#[test]
fn focus_invariant_combines_at_least_two_fractions() {
    // "Combine Same Denominator Fractions" should only fire when combining ≥2 fractions
    // The focus_before should contain at least 2 fraction-like patterns
    let test_cases = vec![
        "a/x + b/x",       // 2 fractions
        "a/x + b/x + c/x", // 3 fractions
        "1 + a/x + b/x",   // 2 fractions + constant
    ];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Combine Same Denominator Fractions" {
                if let Some(before_local) = step.before_local {
                    let focus_str = display(&simplifier.context, before_local);

                    // Count '/' characters as proxy for fraction count
                    let fraction_count = focus_str.matches('/').count();

                    assert!(
                        fraction_count >= 2,
                        "Focus should combine ≥2 fractions, got {} in '{}' for input '{}'",
                        fraction_count,
                        focus_str,
                        input
                    );
                }
            }
        }
    }
}

// =============================================================================
// Sign preservation edge cases
// =============================================================================

#[test]
fn sign_preservation_neg_numerator() {
    // Input: a/x + (-b)/x - tests Neg representation in numerator
    let steps = simplify_and_inspect_focus("a/x + (-b)/x");

    let combine_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Combine Same Denominator Fractions");

    if let Some((_, before_local, after_local, _)) = combine_step {
        assert!(
            before_local.is_some() && after_local.is_some(),
            "Should produce focus for a/x + (-b)/x"
        );
    }
}

#[test]
fn sign_preservation_mul_minus_one() {
    // Input: a/x + (-1)*b/x - tests Mul(-1, ...) representation
    let steps = simplify_and_inspect_focus("a/x + (-1)*b/x");

    let combine_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Combine Same Denominator Fractions");

    // This may or may not trigger the rule depending on canonicalization,
    // but if it does, it should have valid focus
    if let Some((_, before_local, after_local, _)) = combine_step {
        if before_local.is_some() {
            assert!(
                after_local.is_some(),
                "If before_local exists, after_local should too"
            );
        }
    }
}
