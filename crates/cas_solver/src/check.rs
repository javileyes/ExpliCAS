//! Backward-compatible verification facade.
//!
//! Canonical implementation now lives in `cas_engine::solver::check`.

pub use cas_engine::solver::check::{verify_solution, verify_solution_set};
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};
