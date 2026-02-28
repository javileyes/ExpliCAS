//! Re-entrancy guard for expensive ground-evaluation fallback paths.
//!
//! This guard prevents recursive entry into evaluation routines that may invoke
//! each other indirectly (e.g. `prove_nonzero -> simplify -> prove_nonzero`).

thread_local! {
    static IN_GROUND_EVAL: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// RAII guard that decrements the counter on drop.
pub struct GroundEvalGuard;

impl GroundEvalGuard {
    /// Enter the guarded section.
    ///
    /// Returns `None` when already inside a guarded section on this thread.
    pub fn enter() -> Option<Self> {
        IN_GROUND_EVAL.with(|counter| {
            let current = counter.get();
            if current > 0 {
                return None;
            }
            counter.set(current + 1);
            Some(Self)
        })
    }
}

impl Drop for GroundEvalGuard {
    fn drop(&mut self) {
        IN_GROUND_EVAL.with(|counter| {
            counter.set(counter.get().saturating_sub(1));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::GroundEvalGuard;

    #[test]
    fn blocks_nested_entry_and_releases_on_drop() {
        let first = GroundEvalGuard::enter();
        assert!(first.is_some());
        assert!(GroundEvalGuard::enter().is_none());
        drop(first);
        assert!(GroundEvalGuard::enter().is_some());
    }
}
