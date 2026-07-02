//! Strategy-dispatch runtime facade extracted from `solve_runtime_flow_strategy`.

pub(crate) use solve_runtime_flow_strategy_dispatch_default_mappers::apply_strategy_kind_with_default_kernels_and_default_step_and_error_mappers_with_state;

#[path = "solve_runtime_flow_strategy_dispatch_apply_core.rs"]
mod solve_runtime_flow_strategy_dispatch_apply_core;
#[path = "solve_runtime_flow_strategy_dispatch_core.rs"]
mod solve_runtime_flow_strategy_dispatch_core;
#[path = "solve_runtime_flow_strategy_dispatch_default_mappers.rs"]
mod solve_runtime_flow_strategy_dispatch_default_mappers;
