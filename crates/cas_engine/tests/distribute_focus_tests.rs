//! Contract tests for DistributeRule focus behavior.
//!
//! These tests verify that the focus shows the product/quotient being distributed,
//! and the resulting sum.

use cas_ast::Context;
use cas_engine::to_display_steps;
use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;

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
            let before_local = step
                .before_local()
                .map(|id| display(&simplifier.context, id));
            let after_local = step
                .after_local()
                .map(|id| display(&simplifier.context, id));
            let rule = step.rule_name.clone();
            let desc = step.description.clone();
            (rule, before_local, after_local, desc)
        })
        .collect()
}

// =============================================================================
// Contract: DistributeRule focus
// =============================================================================

#[test]
fn distribute_mul_sum_has_focus() {
    // 2*(x+y) should show focus on the distribution
    let steps = simplify_and_inspect_focus("2*(x+y)");

    let distribute_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Distributive Property");

    if let Some((_, before_local, after_local, _)) = distribute_step {
        assert!(
            before_local.is_some(),
            "DistributeRule should have before_local"
        );
        assert!(
            after_local.is_some(),
            "DistributeRule should have after_local"
        );
    }
}

#[test]
fn distribute_sum_mul_has_focus() {
    // (x+y)*2 should show focus on the distribution
    let steps = simplify_and_inspect_focus("(x+y)*2");

    let distribute_step = steps
        .iter()
        .find(|(rule, _, _, _)| rule == "Distributive Property");

    if let Some((_, before_local, after_local, _)) = distribute_step {
        assert!(
            before_local.is_some(),
            "DistributeRule should have before_local for (sum)*factor"
        );
        assert!(
            after_local.is_some(),
            "DistributeRule should have after_local for (sum)*factor"
        );
    }
}

#[test]
fn distribute_focus_invariant_before_ne_after() {
    // Focus must be informative: before_local ≠ after_local
    let test_cases = vec!["2*(x+y)", "(a+b)*3", "x^2*(1+x)"];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Distributive Property" {
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
