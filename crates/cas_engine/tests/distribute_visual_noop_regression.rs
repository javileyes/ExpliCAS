//! Regression test: Distribute steps must produce visible changes
//!
//! Bug: When rationalizing 1/(sqrt(x)-1), the expression contains 1*(a+b)
//! patterns. DistributeRule would apply, producing 1*a + 1*b, but since
//! MulOne simplifies during rendering, Before/After looked identical.
//!
//! Fix: DistributeRule now skips when a factor is 1 (visual no-op).

use cas_engine::eval_step_pipeline::to_display_steps;
use cas_engine::Simplifier;

#[test]
fn distribute_step_must_not_be_visually_noop() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let expr = cas_parser::parse(
        "1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)",
        &mut simplifier.context,
    )
    .unwrap();

    let (_result, raw_steps) = simplifier.simplify(expr);
    let display_steps = to_display_steps(raw_steps);

    // Find any Distribute steps
    let distribute_steps: Vec<_> = display_steps
        .iter()
        .filter(|s| s.rule_name == "Distributive Property")
        .collect();

    // If there are Distribute steps, they must produce visible changes
    for s in &distribute_steps {
        let before_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: s.before
            }
        );
        let after_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: s.after
            }
        );

        assert_ne!(
            before_str, after_str,
            "Distribute step is visually a no-op: before==after=='{}'",
            before_str
        );
    }
}
