/// Domain environment for solver operations.
pub type SolveDomainEnv = cas_solver_core::solve_runtime_types::RuntimeSolveDomainEnv;

/// Solver context for recursive and nested solve flows.
pub type SolveCtx = cas_solver_core::solve_runtime_types::RuntimeSolveCtx;

/// Display-ready solve steps (post-cleanup).
pub type DisplaySolveSteps = cas_solver_core::solve_runtime_types::RuntimeDisplaySolveSteps;

/// Diagnostics collected during solve operation.
pub type SolveDiagnostics = cas_solver_core::solve_runtime_types::RuntimeSolveDiagnostics;

/// Educational sub-step for solver derivations.
pub type SolveSubStep = cas_solver_core::solve_runtime_types::RuntimeSolveSubStep;

/// Top-level solver step entry.
pub type SolveStep = cas_solver_core::solve_runtime_types::RuntimeSolveStep;
