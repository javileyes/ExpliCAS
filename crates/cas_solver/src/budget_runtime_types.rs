//! Local aliases for budget/runtime accounting types.
//!
//! Keeping these aliases in `cas_solver` narrows direct `engine_exports`
//! coupling while preserving API compatibility.

pub type Budget = cas_solver_core::budget_model::Budget;
pub type BudgetExceeded = cas_solver_core::budget_model::BudgetExceeded;
pub type Metric = cas_solver_core::budget_model::Metric;
pub type Operation = cas_solver_core::budget_model::Operation;
pub type PassStats = cas_solver_core::budget_model::PassStats;
