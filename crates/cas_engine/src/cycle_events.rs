//! Cycle event registry for recording detected ping-pong patterns.
//!
//! Thread-local registry (same pattern as `domain.rs` `BlockedHint`) that
//! collects `CycleEvent` structs whenever the engine detects a rule oscillation.
//! Events are collected during simplification and drained at the end of the
//! pipeline for inclusion in `PipelineStats`.
//!
//! # Usage
//!
//! ```ignore
//! // At pipeline start:
//! clear_cycle_events();
//!
//! // When a cycle is detected (in rule_application.rs or orchestrator.rs):
//! register_cycle_event(CycleEvent { ... });
//!
//! // At pipeline end:
//! let events = take_cycle_events();
//! pipeline_stats.cycle_events = events;
//! ```

use std::cell::RefCell;

/// Level at which the cycle was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CycleLevel {
    /// Detected within `apply_rules` (ring buffer in `LocalSimplificationTransformer`).
    /// The rule_name identifies which specific rule triggered the repeated fingerprint.
    IntraNode,
    /// Detected between full-tree iterations in `Orchestrator::run_phase`.
    /// The rule_name is `"(inter-iteration)"` since the specific rule is unknown.
    InterIteration,
}

impl std::fmt::Display for CycleLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CycleLevel::IntraNode => write!(f, "intra-node"),
            CycleLevel::InterIteration => write!(f, "inter-iteration"),
        }
    }
}

/// Record of a detected cycle during simplification.
///
/// Captures enough context to diagnose which rules are conflicting and on
/// which expressions, enabling later targeted fixes to rule priorities or guards.
#[derive(Debug, Clone)]
pub struct CycleEvent {
    /// Which pipeline phase detected the cycle.
    pub phase: crate::phase::SimplifyPhase,
    /// Cycle period (1 = self-loop A→A, 2 = ping-pong A↔B, etc.).
    pub period: usize,
    /// Detection level (intra-node vs inter-iteration).
    pub level: CycleLevel,
    /// Name of the rule that triggered the cycle (or `"(inter-iteration)"`).
    pub rule_name: String,
    /// Structural fingerprint of the expression at detection point.
    pub expr_fingerprint: u64,
    /// Human-readable expression (truncated to ~120 chars).
    pub expr_display: String,
    /// Rewrite step count at detection time.
    pub rewrite_step: usize,
}

impl std::fmt::Display for CycleEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} cycle(period={}) at step {} by '{}': {}",
            self.phase,
            self.level,
            self.period,
            self.rewrite_step,
            self.rule_name,
            self.expr_display
        )
    }
}

// =============================================================================
// Thread-Local Cycle Event Registry
// =============================================================================

thread_local! {
    /// Thread-local storage for cycle events during simplification.
    static CYCLE_EVENTS: RefCell<Vec<CycleEvent>> = const { RefCell::new(Vec::new()) };
}

/// Register a cycle event. Deduplicates by `(fingerprint, rule_name, level)`.
pub fn register_cycle_event(event: CycleEvent) {
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

/// Take all cycle events, clearing the thread-local storage.
/// Called at end of simplification pipeline to retrieve events.
pub fn take_cycle_events() -> Vec<CycleEvent> {
    CYCLE_EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()))
}

/// Clear cycle events without returning them.
/// Called at start of simplification pipeline to reset state.
pub fn clear_cycle_events() {
    CYCLE_EVENTS.with(|events| events.borrow_mut().clear());
}

/// Truncate a string for display (max ~120 chars, appends "…" if truncated).
pub fn truncate_display(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut truncated = s[..max_len].to_string();
        truncated.push('…');
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        // After take, should be empty
        let events2 = take_cycle_events();
        assert!(events2.is_empty());
    }

    #[test]
    fn test_dedup() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode)); // duplicate

        let events = take_cycle_events();
        assert_eq!(events.len(), 1); // deduplicated
    }

    #[test]
    fn test_dedup_different_level() {
        clear_cycle_events();
        register_cycle_event(make_event(100, "RuleA", CycleLevel::IntraNode));
        register_cycle_event(make_event(100, "RuleA", CycleLevel::InterIteration));

        let events = take_cycle_events();
        assert_eq!(events.len(), 2); // different level → not deduped
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
