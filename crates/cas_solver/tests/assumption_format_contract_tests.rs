use cas_ast::Context;
use cas_solver::api::AssumptionEvent;
use cas_solver_core::assumption_display::{
    format_displayable_assumption_lines_for_step,
    format_displayable_assumption_lines_grouped_for_step,
};
use cas_solver_core::assumption_usage::collect_assumed_conditions_from_steps;

#[test]
fn displayable_assumption_lines_for_step_hides_derived() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut step = cas_solver::runtime::Step::new("demo", "RuleA", x, x, vec![], Some(&ctx));
    step.meta_mut()
        .assumption_events
        .push(AssumptionEvent::nonzero(&ctx, x));
    step.meta_mut()
        .assumption_events
        .push(AssumptionEvent::positive(&ctx, x));

    let lines = format_displayable_assumption_lines_for_step(&step);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with("ℹ️ Requires: "));
    assert!(lines[0].contains("x > 0"));
}

#[test]
fn grouped_displayable_assumption_lines_keep_kind_order() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut step = cas_solver::runtime::Step::new("demo", "RuleB", x, x, vec![], Some(&ctx));
    step.meta_mut()
        .assumption_events
        .push(AssumptionEvent::positive_assumed(&ctx, x));
    step.meta_mut()
        .assumption_events
        .push(AssumptionEvent::inv_trig_principal_range(&ctx, "atan", x));
    step.meta_mut()
        .assumption_events
        .push(AssumptionEvent::positive(&ctx, x));

    let lines = format_displayable_assumption_lines_grouped_for_step(&step);
    assert_eq!(lines.len(), 3);
    assert!(lines[0].starts_with("ℹ️ Requires: "));
    assert!(lines[1].starts_with("🔀 Branch: "));
    assert!(lines[2].starts_with("⚠️ Assumes: "));
}

#[test]
fn collect_assumed_conditions_from_steps_dedupes_by_key() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut step_a = cas_solver::runtime::Step::new("demo", "RuleA", x, x, vec![], Some(&ctx));
    step_a
        .meta_mut()
        .assumption_events
        .push(AssumptionEvent::positive(&ctx, x));

    let mut step_b = cas_solver::runtime::Step::new("demo", "RuleB", x, x, vec![], Some(&ctx));
    step_b
        .meta_mut()
        .assumption_events
        .push(AssumptionEvent::positive(&ctx, x));

    let items = collect_assumed_conditions_from_steps(&[step_a, step_b]);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].0, "x > 0");
    assert_eq!(items[0].1, "RuleA");
}
