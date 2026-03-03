use cas_ast::Equation;

/// Solver options are shared with `cas_engine` to avoid drift between
/// the compatibility facade and the engine-native solver pipeline.
pub type SolverOptions = cas_engine::SolverOptions;

/// Domain environment for solver operations.
pub type SolveDomainEnv = cas_solver_core::domain_env::SolveDomainEnv<crate::ImplicitDomain>;

/// Solver context for recursive and nested solve flows.
pub type SolveCtx = cas_solver_core::solve_context::SolveContext<
    SolveDomainEnv,
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Display-ready solve steps (post-cleanup).
pub type DisplaySolveSteps = cas_solver_core::display_steps::DisplaySteps<SolveStep>;

/// Diagnostics collected during solve operation.
pub type SolveDiagnostics = cas_solver_core::solve_types::SolveDiagnostics<
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    crate::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Educational sub-step for solver derivations.
pub type SolveSubStep =
    cas_solver_core::solve_types::SolveSubStep<Equation, crate::ImportanceLevel>;

/// Top-level solver step entry.
pub type SolveStep =
    cas_solver_core::solve_types::SolveStep<Equation, crate::ImportanceLevel, SolveSubStep>;
