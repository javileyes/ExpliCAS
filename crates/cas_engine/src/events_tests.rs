use crate::{EngineEvent, Simplifier, StepListener};
use cas_ast::Expr;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct CapturingListener {
    sink: Arc<Mutex<Vec<EngineEvent>>>,
}

impl CapturingListener {
    fn new(sink: Arc<Mutex<Vec<EngineEvent>>>) -> Self {
        Self { sink }
    }
}

impl StepListener for CapturingListener {
    fn on_event(&mut self, event: &EngineEvent) {
        self.sink
            .lock()
            .expect("listener sink poisoned")
            .push(event.clone());
    }
}

#[test]
fn emits_rule_applied_events_when_steps_are_on() {
    let mut simplifier = Simplifier::with_default_rules();
    let sink = Arc::new(Mutex::new(Vec::new()));
    simplifier.set_step_listener(Some(Box::new(CapturingListener::new(sink.clone()))));

    let x = simplifier.context.var("x");
    let zero = simplifier.context.num(0);
    let expr = simplifier.context.add(Expr::Add(x, zero));

    let (_result, _steps) = simplifier.simplify(expr);
    let events = sink.lock().expect("listener sink poisoned");

    assert!(
        events.iter().any(|event| matches!(
            event,
            EngineEvent::RuleApplied {
                rule_name,
                global_before: Some(_),
                global_after: Some(_),
                ..
            } if !rule_name.is_empty()
        )),
        "expected at least one RuleApplied event with global snapshots"
    );
}

#[test]
fn emits_rule_applied_events_when_steps_are_off() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);
    let sink = Arc::new(Mutex::new(Vec::new()));
    simplifier.set_step_listener(Some(Box::new(CapturingListener::new(sink.clone()))));

    let x = simplifier.context.var("x");
    let zero = simplifier.context.num(0);
    let expr = simplifier.context.add(Expr::Add(x, zero));

    let (_result, steps) = simplifier.simplify(expr);
    let events = sink.lock().expect("listener sink poisoned");

    assert!(steps.is_empty(), "steps should remain disabled");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EngineEvent::RuleApplied { .. })),
        "expected RuleApplied events even with steps disabled"
    );
}
