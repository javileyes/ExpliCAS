//! Preflight and solve-entry runtime facade extracted from `solve_runtime_flow`.
//!
//! Concrete wrappers are split into focused modules:
//! - `entry`: guards, prepass orchestration, cycle-guard glue.
//! - `analysis`: preflight-domain derivation wrapper.
//! - `prepare`: equation preparation wrappers.

pub use solve_runtime_flow_preflight_analysis::*;
pub use solve_runtime_flow_preflight_entry::*;
pub use solve_runtime_flow_preflight_prepare::*;

#[path = "solve_runtime_flow_preflight_analysis.rs"]
mod solve_runtime_flow_preflight_analysis;
#[path = "solve_runtime_flow_preflight_entry.rs"]
mod solve_runtime_flow_preflight_entry;
#[path = "solve_runtime_flow_preflight_prepare.rs"]
mod solve_runtime_flow_preflight_prepare;
