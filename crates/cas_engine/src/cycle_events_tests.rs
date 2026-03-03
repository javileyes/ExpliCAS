#[cfg(test)]
mod tests {
    use crate::cycle_events::{
        clear_cycle_events, register_cycle_event, take_cycle_events, truncate_display, CycleEvent,
        CycleLevel,
    };
    use crate::phase::SimplifyPhase;

    fn make_event(fingerprint: u64, rule: &str, level: CycleLevel) -> CycleEvent {
        CycleEvent {
            phase: SimplifyPhase::Transform,
            period: 2,
            level,
            rule_name: rule.to_string(),
            expr_fingerprint: fingerprint,
            expr_display: "sin(x)^2 + cos(x)^2".to_string(),
            rewrite_step: 5,
        }
    }

    #[test]
    fn test_register_and_take() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(200, "RuleB", CycleLevel::IntraNode));

        let events = take_cycle_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].rule_name, "RuleA");
        assert_eq!(events[1].rule_name, "RuleB");

        let events2 = take_cycle_events();
        assert!(events2.is_empty());
    }

    #[test]
    fn test_dedup() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));

        let events = take_cycle_events();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_dedup_different_level() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(100, "RuleA", CycleLevel::InterIteration));

        let events = take_cycle_events();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_clear() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        clear_cycle_events();

        let events = take_cycle_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_truncate_display() {
        assert_eq!(truncate_display("short", 10), "short");
        assert_eq!(truncate_display("a long string here", 6), "a long…");
    }
}
