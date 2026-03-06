//! Shared solver contract and runtime aliases for the engine facade.

pub(crate) type SolverOptions = cas_solver_core::solver_options::SolverOptions;

pub(crate) type SolveDomainEnv =
    cas_solver_core::solve_aliases::SolveDomainEnv<crate::ImplicitDomain>;

pub(crate) type SolveCtx = cas_solver_core::solve_aliases::SolveCtx<
    SolveDomainEnv,
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

pub(crate) type SolveSubStep =
    cas_solver_core::solve_aliases::SolveSubStep<cas_ast::Equation, crate::ImportanceLevel>;

pub(crate) type SolveStep = cas_solver_core::solve_aliases::SolveStep<
    cas_ast::Equation,
    crate::ImportanceLevel,
    SolveSubStep,
>;

pub(crate) type DisplaySolveSteps = cas_solver_core::solve_aliases::DisplaySolveSteps<SolveStep>;

pub(crate) type SolveDiagnostics = cas_solver_core::solve_aliases::SolveDiagnostics<
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    crate::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;
