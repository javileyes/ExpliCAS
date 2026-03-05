//! Backend boundary for solve execution.
//!
//! Keeps `cas_engine` coupling in one place so `cas_solver` API can switch to
//! a local/native backend incrementally during migration.

pub use crate::solve_backend_contract::{CoreSolverOptions, SolveBackend};
