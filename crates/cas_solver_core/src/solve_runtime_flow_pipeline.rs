//! Strategy-pipeline runtime facade extracted from `solve_runtime_flow`.
//!
//! Concrete wrappers are split into:
//! - `resolution`: var-elimination and discrete-candidate verification helpers.
//! - `execute`: strategy-order pipeline wiring and cycle-guard integration.

pub use solve_runtime_flow_pipeline_execute::*;
pub use solve_runtime_flow_pipeline_resolution::*;

#[path = "solve_runtime_flow_pipeline_execute.rs"]
mod solve_runtime_flow_pipeline_execute;
#[path = "solve_runtime_flow_pipeline_resolution.rs"]
mod solve_runtime_flow_pipeline_resolution;
