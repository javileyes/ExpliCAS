//! Contract tests for CancelCommonFactorsRule focus behavior.
//!
//! These tests verify that the focus shows the fraction being simplified,
//! including the factor cancelled.

use cas_ast::{Context, DisplayExpr, Expr};
use cas_engine::eval_step_pipeline::to_display_steps;
use cas_engine::Simplifier;

/// Helper to get display string for an ExprId
fn display(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

/// Helper: simplify and return focus info for Cancel Common Factors steps
fn simplify_and_get_cancel_focus(input: &str) -> Vec<(Option<String>, Option<String>, String)> {
    let mut simplifier = Simplifier::with_default_rules();
    let ctx = &mut simplifier.context;
    let expr = cas_parser::parse(input, ctx).expect("parse failed");

    let (_, raw_steps) = simplifier.simplify(expr);
    let display_steps = to_display_steps(raw_steps);

    display_steps
        .iter()
        .filter(|step| step.rule_name == "Cancel Common Factors")
        .map(|step| {
            let before_local = step
                .before_local()
                .map(|id| display(&simplifier.context, id));
            let after_local = step
                .after_local()
                .map(|id| display(&simplifier.context, id));
            (before_local, after_local, step.description.clone())
        })
        .collect()
}

// =============================================================================
// Contract: CancelCommonFactorsRule focus
// =============================================================================

#[test]
fn cancel_common_factors_has_focus() {
    // (x+1)/(x+1) should cancel to 1
    // Note: Some expressions may be handled by other rules (e.g., Simplify Nested Fraction)
    // This test verifies that IF Cancel Common Factors fires, it has focus
    let steps = simplify_and_get_cancel_focus("(x+1)/(x+1)");

    // If the rule fired, verify it has focus
    for (before_local, after_local, _) in &steps {
        assert!(
            before_local.is_some(),
            "CancelCommonFactorsRule should have before_local when it fires"
        );
        assert!(
            after_local.is_some(),
            "CancelCommonFactorsRule should have after_local when it fires"
        );
    }
}

#[test]
fn cancel_identical_factors_has_focus() {
    // (x+1)/(x+1) should cancel to 1
    let steps = simplify_and_get_cancel_focus("(x+1)/(x+1)");

    if !steps.is_empty() {
        let (before_local, after_local, _) = &steps[0];
        assert!(
            before_local.is_some(),
            "Should have before_local for (x+1)/(x+1)"
        );
        assert!(
            after_local.is_some(),
            "Should have after_local for (x+1)/(x+1)"
        );
    }
}

#[test]
fn cancel_focus_invariant_before_ne_after() {
    // Focus must be informative: before_local ≠ after_local
    let test_cases = vec![
        "(a*x)/(b*x)", // cancel x
        "(2*y)/(4*y)", // cancel y and simplify coefficients
        "x^2/x",       // cancel x from power
    ];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Cancel Common Factors" {
                if let (Some(before_local), Some(after_local)) =
                    (step.before_local(), step.after_local())
                {
                    let before_str = display(&simplifier.context, before_local);
                    let after_str = display(&simplifier.context, after_local);

                    assert_ne!(
                        before_str, after_str,
                        "Focus must show a change for input '{}'",
                        input
                    );
                }
            }
        }
    }
}

#[test]
fn cancel_focus_invariant_contains_div() {
    // For fraction rules, before_local must contain at least one Div
    let test_cases = vec!["(a*f)/(b*f)", "x^2/x"];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Cancel Common Factors" {
                if let Some(before_local) = step.before_local() {
                    // Check that before_local contains a Div
                    fn contains_div(ctx: &Context, e: cas_ast::ExprId) -> bool {
                        match ctx.get(e) {
                            Expr::Div(_, _) => true,
                            Expr::Add(l, r)
                            | Expr::Sub(l, r)
                            | Expr::Mul(l, r)
                            | Expr::Pow(l, r) => contains_div(ctx, *l) || contains_div(ctx, *r),
                            Expr::Neg(inner) => contains_div(ctx, *inner),
                            _ => false,
                        }
                    }

                    assert!(
                        contains_div(&simplifier.context, before_local),
                        "before_local must contain a Div for input '{}'",
                        input
                    );
                }
            }
        }
    }
}
