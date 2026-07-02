//! Shared solve runtime orchestration facade for integration crates.
//!
//! Runtime crates (`cas_engine`, `cas_solver`) import from this module while
//! concrete implementations live in focused submodules.

pub(crate) use crate::solve_runtime_flow_isolation::*;
pub(crate) use crate::solve_runtime_flow_orchestration::*;
pub(crate) use crate::solve_runtime_flow_pipeline::*;
pub use crate::solve_runtime_flow_preflight::*;
pub(crate) use crate::solve_runtime_flow_strategy::*;
pub(crate) use crate::solve_runtime_flow_strategy_kernels::*;
