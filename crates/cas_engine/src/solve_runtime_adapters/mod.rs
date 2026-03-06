//! Shared runtime adapters for solve pipeline internals.

pub(crate) type SolverOptions = crate::SolverOptions;
pub(crate) use crate::{SolveCtx, SolveStep};
pub(crate) use cas_solver_core::solve_runtime_adapter_state_runtime::{
    context_render_expr, simplifier_collect_steps, simplifier_contains_var, simplifier_context,
    simplifier_context_mut, simplifier_expand_expr, simplifier_is_known_negative,
    simplifier_prove_nonzero_status, simplifier_render_expr, simplifier_simplify_expr,
    simplifier_zero_expr, simplify_rhs_with_step_pairs, sym_name_as_string,
};
pub(crate) use cas_solver_core::solve_runtime_mapping::*;
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
