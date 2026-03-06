//! Shared runtime adapters for solve pipeline internals.

pub(crate) use crate::solve_runtime::{
    solve_with_ctx_and_options, SolveCtx, SolveDomainEnv, SolveStep, SolveSubStep, SolverOptions,
};
pub(crate) use helpers::*;
pub(crate) use isolation::dispatch_isolation_with_default_routes;
pub(crate) use pipeline::{
    apply_strategy, build_solve_preflight_state, execute_strategy_pipeline,
    prepare_equation_for_strategy,
};

mod helpers;
mod isolation;
mod pipeline;
