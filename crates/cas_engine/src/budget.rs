//! Compatibility facade for unified anti-explosion budget types.
//!
//! Source of truth now lives in `cas_solver_core::budget_model`.

pub use cas_solver_core::budget_model::{
    Budget, BudgetExceeded, BudgetScope, Metric, Operation, PassStats,
};
