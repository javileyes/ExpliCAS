//! Local aliases for phase/pipeline runtime types.
//!
//! These remain engine-backed for now but are surfaced from a solver-owned
//! module to make later extraction incremental.

pub type ExpandBudget = cas_solver_core::expand_policy::ExpandBudget;
pub type ExpandPolicy = cas_solver_core::expand_policy::ExpandPolicy;
pub type PhaseBudgets = cas_solver_core::phase_budgets::PhaseBudgets;
pub type PhaseMask = cas_solver_core::simplify_phase::PhaseMask;
pub type PhaseStats = cas_solver_core::phase_stats::PhaseStats;
pub type PipelineStats = cas_solver_core::phase_stats::PipelineStats;
pub type SimplifyPhase = cas_solver_core::simplify_phase::SimplifyPhase;
