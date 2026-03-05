//! Single compatibility bridge to `cas_engine`.
//!
//! Keeping all direct engine coupling here makes remaining migration work
//! explicit and easy to replace incrementally.

use cas_engine as engine;

pub use engine::rules;
pub use engine::rules::logarithms::LogExpansionRule;
pub use engine::Simplifier;
pub use engine::{Engine, Orchestrator, ParentContext, Rewrite, Rule, RuleProfiler, SimpleRule};

pub fn solve_with_ctx_and_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: crate::solve_backend_contract::CoreSolverOptions,
    ctx: &crate::SolveCtx,
) -> Result<(cas_ast::SolutionSet, Vec<crate::SolveStep>), cas_solver_core::error_model::CasError> {
    engine::solve_with_ctx_and_options(eq, var, simplifier, opts, ctx)
}
