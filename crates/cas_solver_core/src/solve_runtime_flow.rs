//! Shared solve runtime orchestration facade for integration crates.
//!
//! Runtime crates (`cas_engine`, `cas_solver`) import from this module while
//! concrete implementations live in focused submodules.

pub use crate::solve_runtime_flow_isolation::*;
pub use crate::solve_runtime_flow_isolation_kernels::*;
pub use crate::solve_runtime_flow_orchestration::*;
pub use crate::solve_runtime_flow_pipeline::*;
pub use crate::solve_runtime_flow_preflight::*;
pub use crate::solve_runtime_flow_strategy::*;
pub use crate::solve_runtime_flow_strategy_kernels::*;
