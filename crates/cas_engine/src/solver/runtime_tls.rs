//! Legacy thread-local runtime collectors for solver diagnostics/output.
//!
//! These TLS cells are intentionally isolated from solver semantics:
//! - assumptions collection (for reporting)
//! - output scope tags (for display transforms)

thread_local! {
    /// Thread-local collector for solver assumptions.
    /// Used to pass assumptions from strategies back to caller without changing return type.
    static SOLVE_ASSUMPTIONS: std::cell::RefCell<Option<crate::assumptions::AssumptionCollector>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard for solver assumption collection.
///
/// Creates a fresh collector on creation, and on drop:
/// - Returns the collected assumptions via `finish()`
/// - Restores any previous collector (for nested solves)
///
/// # Safety against leaks/reentrancy
/// - Drop always clears or restores state
/// - Nested solves get their own collectors (previous is saved)
pub(crate) struct SolveAssumptionsGuard {
    /// The previous collector (if any) that was active before this guard
    previous: Option<crate::assumptions::AssumptionCollector>,
    /// Whether collection is enabled for this guard
    enabled: bool,
}

impl SolveAssumptionsGuard {
    /// Create a new guard that starts assumption collection.
    /// If `enabled` is false, no collection happens (passthrough).
    pub fn new(enabled: bool) -> Self {
        let previous = if enabled {
            // Take any existing collector (for nested solve case)
            let prev = SOLVE_ASSUMPTIONS.with(|c| c.borrow_mut().take());
            // Install fresh collector
            SOLVE_ASSUMPTIONS.with(|c| {
                *c.borrow_mut() = Some(crate::assumptions::AssumptionCollector::new());
            });
            prev
        } else {
            None
        };

        Self { previous, enabled }
    }

    /// Finish collection and return the collected records.
    /// This consumes the guard.
    pub fn finish(self) -> Vec<crate::assumptions::AssumptionRecord> {
        // The Drop impl will restore previous, we just need to take current
        if self.enabled {
            SOLVE_ASSUMPTIONS.with(|c| {
                c.borrow_mut()
                    .take()
                    .map(|collector| collector.finish())
                    .unwrap_or_default()
            })
        } else {
            vec![]
        }
    }
}

impl Drop for SolveAssumptionsGuard {
    fn drop(&mut self) {
        if self.enabled {
            // Restore previous collector (or None if there wasn't one)
            SOLVE_ASSUMPTIONS.with(|c| {
                *c.borrow_mut() = self.previous.take();
            });
        }
    }
}

/// Note an assumption during solver operation (internal use).
pub(crate) fn note_assumption(event: crate::assumptions::AssumptionEvent) {
    SOLVE_ASSUMPTIONS.with(|c| {
        if let Some(ref mut collector) = *c.borrow_mut() {
            collector.note(event);
        }
    });
}
