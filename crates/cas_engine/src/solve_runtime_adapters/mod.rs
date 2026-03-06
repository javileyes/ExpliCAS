//! Shared runtime adapters for solve pipeline internals.

pub(crate) use crate::solve_runtime::{SolveCtx, SolveStep, SolverOptions};
pub(crate) use helper_bridge::*;
pub(crate) use helper_mapping::*;
pub(crate) use helper_state::*;
pub(crate) use isolation::dispatch_isolation_with_default_routes;
pub(crate) use pipeline::{
    apply_strategy, build_solve_preflight_state, execute_strategy_pipeline,
    prepare_equation_for_strategy,
};

mod helper_bridge;
mod helper_mapping;
mod helper_state;
mod isolation;
mod pipeline;
