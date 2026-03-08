//! Strategy-kernel runtime facade extracted from `solve_runtime_flow`.

pub use solve_runtime_flow_strategy_kernels_core::*;
pub use solve_runtime_flow_strategy_kernels_equation::*;

#[path = "solve_runtime_flow_strategy_kernels_core.rs"]
mod solve_runtime_flow_strategy_kernels_core;
#[path = "solve_runtime_flow_strategy_kernels_equation.rs"]
mod solve_runtime_flow_strategy_kernels_equation;
