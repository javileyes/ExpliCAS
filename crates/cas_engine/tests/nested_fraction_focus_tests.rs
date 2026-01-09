//! Contract tests for NestedFractionRule focus behavior.
//!
//! These tests verify that the focus shows the complex fraction being simplified.

use cas_ast::{Context, DisplayExpr};
use cas_engine::eval_step_pipeline::to_display_steps;
use cas_engine::Simplifier;

/// Helper to get display string for an ExprId
fn display(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

/// Helper: simplify and return focus info for Simplify Complex Fraction steps
fn simplify_and_get_nested_focus(input: &str) -> Vec<(Option<String>, Option<String>, String)> {
    let mut simplifier = Simplifier::with_default_rules();
    let ctx = &mut simplifier.context;
    let expr = cas_parser::parse(input, ctx).expect("parse failed");

    let (_, raw_steps) = simplifier.simplify(expr);
    let display_steps = to_display_steps(raw_steps);

    display_steps
        .iter()
        .filter(|step| step.rule_name == "Simplify Complex Fraction")
        .map(|step| {
            let before_local = step.before_local.map(|id| display(&simplifier.context, id));
            let after_local = step.after_local.map(|id| display(&simplifier.context, id));
            (before_local, after_local, step.description.clone())
        })
        .collect()
}

// =============================================================================
// Contract: NestedFractionRule focus
// =============================================================================

#[test]
fn nested_fraction_has_focus() {
    // (1/x) / (1/y) = y/x - should show the complex fraction
    let steps = simplify_and_get_nested_focus("(1/x) / (1/y)");

    // If the rule fired, verify it has focus
    for (before_local, after_local, _) in &steps {
        assert!(
            before_local.is_some(),
            "NestedFractionRule should have before_local when it fires"
        );
        assert!(
            after_local.is_some(),
            "NestedFractionRule should have after_local when it fires"
        );
    }
}

#[test]
fn nested_fraction_focus_invariant_before_ne_after() {
    // Focus must be informative: before_local ≠ after_local
    let test_cases = vec![
        "(1/x) / (1/y)",     // complex fraction
        "(a/b) / (c/d)",     // fraction over fraction
        "(1 + 1/x) / (1/y)", // sum with fraction over fraction
    ];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Simplify Complex Fraction" {
                if let (Some(before_local), Some(after_local)) =
                    (step.before_local, step.after_local)
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
fn nested_fraction_focus_invariant_contains_div() {
    // For nested fraction rules, before_local must contain at least one Div
    let test_cases = vec!["(1/x) / (1/y)", "(a/b) / c"];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Simplify Complex Fraction" {
                if let Some(before_local) = step.before_local {
                    use cas_ast::Expr;
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
