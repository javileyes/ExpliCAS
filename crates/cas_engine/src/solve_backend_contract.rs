//! Shared solver contract and runtime aliases for the engine facade.

pub(crate) type CoreSolverOptions = cas_solver_core::solver_options::SolverOptions;
pub(crate) type SolverOptions = CoreSolverOptions;

pub(crate) type SolveCtx = cas_solver_core::solve_runtime_types::RuntimeSolveCtx;
pub(crate) type SolveSubStep = cas_solver_core::solve_runtime_types::RuntimeSolveSubStep;
pub(crate) type SolveStep = cas_solver_core::solve_runtime_types::RuntimeSolveStep;
pub(crate) type DisplaySolveSteps = cas_solver_core::solve_runtime_types::RuntimeDisplaySolveSteps;
pub(crate) type SolveDiagnostics = cas_solver_core::solve_runtime_types::RuntimeSolveDiagnostics;
