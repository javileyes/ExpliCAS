//! Shared simplification pipeline statistics model.

/// Statistics from a single phase execution.
#[derive(Debug, Clone, Default)]
pub struct PhaseStats {
    /// Which phase was executed.
    pub phase: Option<crate::simplify_phase::SimplifyPhase>,
    /// Number of iterations used (out of budget).
    pub iters_used: usize,
    /// Number of rewrites (steps) applied.
    pub rewrites_used: usize,
    /// Whether the expression changed.
    pub changed: bool,
    /// Cycle detected (ping-pong) - if Some, phase was stopped early.
    pub cycle: Option<crate::cycle_models::CycleInfo>,
}

impl PhaseStats {
    pub fn new(phase: crate::simplify_phase::SimplifyPhase) -> Self {
        Self {
            phase: Some(phase),
            iters_used: 0,
            rewrites_used: 0,
            changed: false,
            cycle: None,
        }
    }
}

/// Statistics from the entire pipeline execution.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub core: PhaseStats,
    pub transform: PhaseStats,
    pub rationalize: PhaseStats,
    pub post_cleanup: PhaseStats,
    pub total_rewrites: usize,
    /// Outcome of rationalization attempt (if phase was run).
    pub rationalize_outcome: Option<cas_math::rationalize_policy::RationalizeOutcome>,
    /// The level that was attempted.
    pub rationalize_level: Option<cas_math::rationalize_policy::AutoRationalizeLevel>,
    /// Collected assumptions (deduplicated, with counts).
    pub assumptions: Vec<crate::assumption_model::AssumptionRecord>,
    /// Collected cycle events detected during simplification.
    pub cycle_events: Vec<crate::cycle_models::CycleEvent>,
}
