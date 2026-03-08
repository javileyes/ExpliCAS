use super::ReplCore;
use crate::SessionState;
use cas_engine::{Engine, Simplifier};
use cas_solver_core::simplify_options::SimplifyOptions;

impl ReplCore {
    /// Create a new ReplCore with a pre-configured simplifier.
    pub fn with_simplifier(simplifier: Simplifier) -> Self {
        Self {
            engine: Engine::with_simplifier(simplifier),
            simplify_options: SimplifyOptions::default(),
            debug_mode: false,
            last_stats: None,
            health_enabled: false,
            last_health_report: None,
            state: SessionState::new(),
        }
    }

    /// Create with default rules (for testing or simple use).
    pub fn new() -> Self {
        let simplifier = Simplifier::with_default_rules();
        Self::with_simplifier(simplifier)
    }
}

impl Default for ReplCore {
    fn default() -> Self {
        Self::new()
    }
}
