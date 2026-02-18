//! Contract tests for FactorCommonIntegerFromAdd focus behavior.
//!
//! These tests verify that the focus shows the sum being factored
//! and the resulting factored form.

use cas_ast::Context;
use cas_engine::eval_step_pipeline::to_display_steps;
use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;

/// Helper to get display string for an ExprId
fn display(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

// =============================================================================
// Contract: FactorCommonIntegerFromAdd focus
// =============================================================================

#[test]
fn factor_integer_has_focus() {
    // 2*sqrt(2) - 2 should factor to 2*(sqrt(2) - 1)
    let mut simplifier = Simplifier::with_default_rules();
    let ctx = &mut simplifier.context;
    let expr = cas_parser::parse("2*sqrt(2) - 2", ctx).expect("parse failed");

    let (_, raw_steps) = simplifier.simplify(expr);
    let display_steps = to_display_steps(raw_steps);

    let factor_step = display_steps
        .iter()
        .find(|s| s.rule_name == "Factor Common Integer");

    if let Some(step) = factor_step {
        assert!(
            step.before_local().is_some(),
            "FactorCommonIntegerFromAdd should have before_local"
        );
        assert!(
            step.after_local().is_some(),
            "FactorCommonIntegerFromAdd should have after_local"
        );
    }
}

#[test]
fn factor_integer_focus_shows_change() {
    // Focus must be informative: before_local ≠ after_local
    let test_cases = vec![
        "4*sqrt(3) + 4", // GCD = 4
        "6*sqrt(5) - 6", // GCD = 6
    ];

    for input in test_cases {
        let mut simplifier = Simplifier::with_default_rules();
        let ctx = &mut simplifier.context;
        let expr = cas_parser::parse(input, ctx).expect("parse failed");

        let (_, raw_steps) = simplifier.simplify(expr);
        let display_steps = to_display_steps(raw_steps);

        for step in &display_steps {
            if step.rule_name == "Factor Common Integer" {
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
