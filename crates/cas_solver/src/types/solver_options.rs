#[path = "solver_options/convert.rs"]
mod convert;
#[path = "solver_options/defaults.rs"]
mod defaults;

/// Options for solver operations, containing semantic context.
///
/// `cas_solver` owns this facade type so solver consumers can remain decoupled
/// from `cas_engine` during migration.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// The value domain (RealOnly or ComplexEnabled)
    pub value_domain: crate::ValueDomain,
    /// The domain mode (Strict, Assume, Generic)
    pub domain_mode: crate::DomainMode,
    /// Scope for assumptions (only active if domain_mode=Assume)
    pub assume_scope: crate::AssumeScope,
    /// Budget for conditional branching (anti-explosion)
    pub budget: cas_solver_core::solve_budget::SolveBudget,
    /// If true, generate detailed step narrative (5 atomic steps).
    /// If false, generate compact narrative.
    pub detailed_steps: bool,
}
