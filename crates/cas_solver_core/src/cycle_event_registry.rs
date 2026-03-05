//! Thread-local cycle event registry.

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
