mod adapters;
pub mod cancel_common_terms;
#[cfg(test)]
mod cancel_common_terms_tests;
mod check_helpers;
pub(crate) mod isolation;
mod isolation_arith;
mod isolation_arith_add_sub;
mod isolation_arith_mul_div;
mod isolation_dispatch;
mod isolation_dispatch_routing;
mod isolation_entrypoints;
mod isolation_function;
mod isolation_pow;
mod preflight;
mod preparation;
pub(crate) mod solve_core;
mod solve_entrypoints;
mod strategy_apply;
mod strategy_apply_advanced;
mod strategy_apply_basic;
mod strategy_apply_roots;
mod strategy_apply_subst_quad;
mod strategy_pipeline;
mod strategy_precheck;
mod types;
mod verification;

pub(crate) use self::adapters::{
    context_render_expr, medium_step, simplifier_collect_steps, simplifier_contains_var,
    simplifier_context, simplifier_context_mut, simplifier_expand_expr,
    simplifier_is_known_negative, simplifier_prove_nonzero_status, simplifier_render_expr,
    simplifier_simplify_expr, simplifier_zero_expr,
};
pub use self::verification::{
    verify_solution, verify_solution_set, VerifyResult, VerifyStatus, VerifySummary,
};
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_infer::infer_solve_variable;
pub use cas_solver_core::verify_stats;
pub(crate) use types::SolveDomainEnv;
pub use types::{
    solver_options_from_eval_options, DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveStep,
    SolveSubStep, SolverOptions,
};

pub use self::solve_entrypoints::{solve, solve_with_ctx_and_options, solve_with_display_steps};
