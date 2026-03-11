//! Backward-compatible API facade mirroring former `cas_engine::api::*` usage.

pub use crate::types::{
    DisplaySolveSteps, SolveDiagnostics, SolveStep, SolveSubStep, SolverOptions,
};
pub use cas_ast::{
    BoundType, Case, ConditionPredicate, ConditionSet, Interval, SolutionSet, SolveResult,
};
pub use cas_formatter::{DisplayExpr, LaTeXExpr};
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};

pub use crate::{
    contains_var, infer_solve_variable, solve, solve_with_display_steps, verify_solution,
    verify_solution_set, verify_stats,
};
