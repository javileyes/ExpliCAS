//! Simplification phases for the phase-based pipeline.
//!
//! The simplifier executes rules in a fixed phase order:
//! 1. Core - Safe local simplifications
//! 2. Transform - Distribution, expansion, collection
//! 3. Rationalize - Automatic rationalization per policy
//! 4. PostCleanup - Final cleanup without expansion
//!
//! Key invariant: Transform never runs after Rationalize.

pub use cas_solver_core::expand_policy::{ExpandBudget, ExpandPolicy};
pub use cas_solver_core::phase_budgets::PhaseBudgets;
pub use cas_solver_core::phase_stats::{PhaseStats, PipelineStats};
pub use cas_solver_core::simplify_options::{SharedSemanticConfig, SimplifyOptions};
pub use cas_solver_core::simplify_phase::{PhaseMask, SimplifyPhase};
