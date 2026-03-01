//! Solver verification facade.
//!
//! During migration, canonical verification implementation remains in
//! `cas_engine::solver::check`. This crate re-exports that API.

pub use cas_engine::solver::check::{
    verify_solution, verify_solution_set, VerifyResult, VerifyStatus, VerifySummary,
};
