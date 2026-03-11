use super::EngineEventCollector;
use cas_ast::Expr;

#[test]
fn collector_records_rule_applied_events() {
    let mut collector = EngineEventCollector::new();
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    cas_solver_core::engine_events::StepListener::on_event(
        &mut collector,
        &cas_solver_core::engine_events::EngineEvent::RuleApplied {
            rule_name: "test".to_string(),
            before: x,
            after: x,
            global_before: None,
            global_after: None,
            is_chained: false,
        },
    );
    assert_eq!(collector.events().len(), 1);
}

#[test]
fn collector_can_be_cloned_for_simplifier_listener() {
    let mut simplifier = crate::runtime::Simplifier::with_default_rules();
    let collector = EngineEventCollector::new();
    simplifier.set_step_listener(Some(Box::new(collector.clone())));

    let x = simplifier.context.var("x");
    let zero = simplifier.context.num(0);
    let expr = simplifier.context.add(Expr::Add(x, zero));
    let _ = simplifier.simplify(expr);

    assert!(
        !collector.events().is_empty(),
        "collector should observe at least one rewrite event"
    );
}
