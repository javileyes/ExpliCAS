pub mod check;
pub(crate) mod isolation;
pub(crate) mod runtime_adapters;
pub(crate) mod solve_core;
mod types;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::verify_stats;
pub(crate) use runtime_adapters::medium_step;
pub(crate) use types::SolveDomainEnv;
pub use types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveStep, SolveSubStep, SolverOptions,
};

pub use self::solve_core::{solve, solve_with_display_steps};

#[cfg(test)]
mod tests;
