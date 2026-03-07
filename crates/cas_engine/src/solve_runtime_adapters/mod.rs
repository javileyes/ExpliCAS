//! Shared runtime adapters for solve pipeline internals.

pub(crate) type SolverOptions = crate::SolverOptions;
pub(crate) use crate::{SolveCtx, SolveStep};
pub(crate) use isolation_dispatch::isolate_with_default_depth;
pub(crate) use pipeline_preflight_context::build_solve_preflight_state;
pub(crate) use pipeline_preflight_equation::prepare_equation_for_strategy;
pub(crate) use pipeline_strategy_apply::apply_strategy;
pub(crate) use pipeline_strategy_execute::execute_strategy_pipeline;

mod isolation_dispatch;
mod pipeline_preflight_context;
mod pipeline_preflight_equation;
mod pipeline_strategy_apply;
mod pipeline_strategy_execute;
mod state_impl;
