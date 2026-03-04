//! Shared type aliases for solver surfaces.
//!
//! These aliases remove repeated long generic signatures in higher-level
//! crates (`cas_engine`, `cas_solver`) while keeping concrete type choices
//! local to each crate.

/// Domain environment alias used by solver contexts.
pub type SolveDomainEnv<ImplicitDomain> = crate::domain_env::SolveDomainEnv<ImplicitDomain>;

/// Shared solver context alias.
pub type SolveCtx<DomainEnv, ImplicitCondition, AssumptionEvent, ScopeTag> =
    crate::solve_context::SolveContext<DomainEnv, ImplicitCondition, AssumptionEvent, ScopeTag>;

/// Wrapper for display-ready solve steps.
pub type DisplaySolveSteps<SolveStep> = crate::display_steps::DisplaySteps<SolveStep>;

/// Shared diagnostics payload alias.
pub type SolveDiagnostics<ImplicitCondition, AssumptionEvent, AssumptionRecord, ScopeTag> =
    crate::solve_types::SolveDiagnostics<
        ImplicitCondition,
        AssumptionEvent,
        AssumptionRecord,
        ScopeTag,
    >;

/// Shared solve sub-step alias.
pub type SolveSubStep<Equation, ImportanceLevel> =
    crate::solve_types::SolveSubStep<Equation, ImportanceLevel>;

/// Shared solve step alias.
pub type SolveStep<Equation, ImportanceLevel, SubStep> =
    crate::solve_types::SolveStep<Equation, ImportanceLevel, SubStep>;
