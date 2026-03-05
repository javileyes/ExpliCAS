//! Shared phase-budget configuration model.

use crate::simplify_phase::SimplifyPhase;

/// Iteration budgets for each phase of the simplification pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PhaseBudgets {
    /// Max iterations for Core phase.
    pub core_iters: usize,
    /// Max iterations for Transform phase.
    pub transform_iters: usize,
    /// Max iterations for Rationalize phase.
    pub rationalize_iters: usize,
    /// Max iterations for PostCleanup phase.
    pub post_iters: usize,
    /// Global safety limit on total rewrites.
    pub max_total_rewrites: usize,
}

impl Default for PhaseBudgets {
    fn default() -> Self {
        Self {
            core_iters: 8,
            transform_iters: 6,
            rationalize_iters: 3,
            post_iters: 4,
            max_total_rewrites: 200,
        }
    }
}

impl PhaseBudgets {
    /// Get the iteration budget for a specific phase.
    pub fn for_phase(&self, phase: SimplifyPhase) -> usize {
        match phase {
            SimplifyPhase::Core => self.core_iters,
            SimplifyPhase::Transform => self.transform_iters,
            SimplifyPhase::Rationalize => self.rationalize_iters,
            SimplifyPhase::PostCleanup => self.post_iters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PhaseBudgets;
    use crate::simplify_phase::SimplifyPhase;

    #[test]
    fn defaults_are_stable() {
        let b = PhaseBudgets::default();
        assert_eq!(b.core_iters, 8);
        assert_eq!(b.transform_iters, 6);
        assert_eq!(b.rationalize_iters, 3);
        assert_eq!(b.post_iters, 4);
        assert_eq!(b.max_total_rewrites, 200);
    }

    #[test]
    fn per_phase_budget_lookup_is_correct() {
        let b = PhaseBudgets {
            core_iters: 11,
            transform_iters: 7,
            rationalize_iters: 5,
            post_iters: 3,
            max_total_rewrites: 999,
        };
        assert_eq!(b.for_phase(SimplifyPhase::Core), 11);
        assert_eq!(b.for_phase(SimplifyPhase::Transform), 7);
        assert_eq!(b.for_phase(SimplifyPhase::Rationalize), 5);
        assert_eq!(b.for_phase(SimplifyPhase::PostCleanup), 3);
    }
}
