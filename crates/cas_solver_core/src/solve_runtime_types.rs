//! Concrete runtime solve aliases shared by engine and solver facades.
//!
//! These aliases pin the generic solver context/step shapes used by the
//! current runtime integration layer so upper crates do not need to repeat
//! the full generic signatures.

/// Runtime implicit condition type used by solver orchestration.
pub type RuntimeImplicitCondition = crate::domain_condition::ImplicitCondition;

/// Runtime implicit domain type used by solver orchestration.
pub type RuntimeImplicitDomain = crate::domain_condition::ImplicitDomain;

/// Runtime domain environment for recursive solve contexts.
pub type RuntimeSolveDomainEnv = crate::solve_aliases::SolveDomainEnv<RuntimeImplicitDomain>;

/// Runtime recursive solve context shared by engine and solver facades.
pub type RuntimeSolveCtx = crate::solve_aliases::SolveCtx<
    RuntimeSolveDomainEnv,
    RuntimeImplicitCondition,
    crate::assumption_model::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Runtime solve sub-step payload.
pub type RuntimeSolveSubStep =
    crate::solve_aliases::SolveSubStep<cas_ast::Equation, crate::step_types::ImportanceLevel>;

/// Runtime solve step payload.
pub type RuntimeSolveStep = crate::solve_aliases::SolveStep<
    cas_ast::Equation,
    crate::step_types::ImportanceLevel,
    RuntimeSolveSubStep,
>;

/// Runtime display-step wrapper.
pub type RuntimeDisplaySolveSteps = crate::solve_aliases::DisplaySolveSteps<RuntimeSolveStep>;

/// Runtime diagnostics payload.
pub type RuntimeSolveDiagnostics = crate::solve_aliases::SolveDiagnostics<
    RuntimeImplicitCondition,
    crate::assumption_model::AssumptionEvent,
    crate::assumption_model::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;
