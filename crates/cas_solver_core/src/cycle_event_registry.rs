//! Thread-local cycle event registry.

use cas_ast::{Context, ExprId};
use std::cell::RefCell;

thread_local! {
    static CYCLE_EVENTS: RefCell<Vec<crate::cycle_models::CycleEvent>> = const { RefCell::new(Vec::new()) };
}

/// Register one cycle event (deduped by fingerprint/rule/level).
pub fn register_cycle_event(event: crate::cycle_models::CycleEvent) {
    CYCLE_EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        let exists = events.iter().any(|e| {
            e.expr_fingerprint == event.expr_fingerprint
                && e.rule_name == event.rule_name
                && e.level == event.level
        });
        if !exists {
            events.push(event);
        }
    });
}

/// Drain all cycle events and clear registry.
pub fn take_cycle_events() -> Vec<crate::cycle_models::CycleEvent> {
    CYCLE_EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()))
}

/// Clear registry without returning events.
pub fn clear_cycle_events() {
    CYCLE_EVENTS.with(|events| events.borrow_mut().clear());
}

/// Truncate display string with ellipsis.
pub fn truncate_display(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut truncated = s[..max_len].to_string();
        truncated.push('…');
        truncated
    }
}

/// Build a truncated display string for an expression.
pub fn expr_display(ctx: &Context, expr: ExprId, max_len: usize) -> String {
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr
        }
    );
    truncate_display(&rendered, max_len)
}

/// Register a cycle event from an expression id and metadata.
#[allow(clippy::too_many_arguments)]
pub fn register_cycle_event_for_expr(
    ctx: &Context,
    expr: ExprId,
    phase: crate::simplify_phase::SimplifyPhase,
    period: usize,
    level: crate::cycle_models::CycleLevel,
    rule_name: &str,
    expr_fingerprint: u64,
    rewrite_step: usize,
) {
    register_cycle_event(crate::cycle_models::CycleEvent {
        phase,
        period,
        level,
        rule_name: rule_name.to_string(),
        expr_fingerprint,
        expr_display: expr_display(ctx, expr, 120),
        rewrite_step,
    });
}

#[cfg(test)]
mod tests {
    use super::{clear_cycle_events, register_cycle_event, take_cycle_events, truncate_display};
    use crate::cycle_models::{CycleEvent, CycleLevel};

    fn make_event(fingerprint: u64, rule: &str, level: CycleLevel) -> CycleEvent {
        CycleEvent {
            phase: crate::simplify_phase::SimplifyPhase::Transform,
            period: 2,
            level,
            rule_name: rule.to_string(),
            expr_fingerprint: fingerprint,
            expr_display: "sin(x)^2 + cos(x)^2".to_string(),
            rewrite_step: 5,
        }
    }

    #[test]
    fn register_and_take() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(200, "RuleB", CycleLevel::IntraNode));

        let events = take_cycle_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].rule_name, "RuleA");
        assert_eq!(events[1].rule_name, "RuleB");
        assert!(take_cycle_events().is_empty());
    }

    #[test]
    fn dedup_same_fingerprint_rule_and_level() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));

        let events = take_cycle_events();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn dedup_keeps_different_levels() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(100, "RuleA", CycleLevel::InterIteration));

        let events = take_cycle_events();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn clear_registry() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        clear_cycle_events();
        assert!(take_cycle_events().is_empty());
    }

    #[test]
    fn truncate_display_adds_ellipsis() {
        assert_eq!(truncate_display("short", 10), "short");
        assert_eq!(truncate_display("a long string here", 6), "a long…");
    }
}
