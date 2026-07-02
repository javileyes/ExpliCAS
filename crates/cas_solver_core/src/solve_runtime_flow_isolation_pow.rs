//! Pow-isolation runtime facade extracted from
//! `solve_runtime_flow_isolation_kernels`.

pub(crate) use solve_runtime_flow_isolation_pow_defaults::apply_pow_isolation_with_default_runtime_config_and_recursive_equation_solver_with_state;

#[path = "solve_runtime_flow_isolation_pow_core.rs"]
mod solve_runtime_flow_isolation_pow_core;
#[path = "solve_runtime_flow_isolation_pow_defaults.rs"]
mod solve_runtime_flow_isolation_pow_defaults;
