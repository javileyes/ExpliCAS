mod check;
pub(crate) mod isolation;
pub(crate) mod solve_core;
mod types;

use crate::engine::Simplifier;
use cas_ast::{Equation, ExprId};

pub use self::check::{
    verify_solution, verify_solution_set, VerifyResult, VerifyStatus, VerifySummary,
};
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::verify_stats;
pub(crate) use types::SolveDomainEnv;
pub use types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveStep, SolveSubStep, SolverOptions,
};

pub use self::solve_core::{solve, solve_with_display_steps};

pub(crate) fn simplifier_context(simplifier: &mut Simplifier) -> &cas_ast::Context {
    &simplifier.context
}

pub(crate) fn simplifier_context_mut(simplifier: &mut Simplifier) -> &mut cas_ast::Context {
    &mut simplifier.context
}

pub(crate) fn simplifier_contains_var(
    simplifier: &mut Simplifier,
    expr: ExprId,
    var: &str,
) -> bool {
    contains_var(&simplifier.context, expr, var)
}

pub(crate) fn simplifier_simplify_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    simplifier.simplify(expr).0
}

pub(crate) fn simplifier_expand_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    crate::expand::expand(&mut simplifier.context, expr)
}

pub(crate) fn simplifier_render_expr(simplifier: &mut Simplifier, expr: ExprId) -> String {
    cas_formatter::render_expr(&simplifier.context, expr)
}

pub(crate) fn context_render_expr(ctx: &cas_ast::Context, expr: ExprId) -> String {
    cas_formatter::render_expr(ctx, expr)
}

pub(crate) fn simplifier_zero_expr(simplifier: &mut Simplifier) -> ExprId {
    simplifier.context.num(0)
}

pub(crate) fn simplifier_collect_steps(simplifier: &mut Simplifier) -> bool {
    simplifier.collect_steps()
}

pub(crate) fn simplifier_prove_nonzero_status(
    simplifier: &mut Simplifier,
    expr: ExprId,
) -> cas_solver_core::linear_solution::NonZeroStatus {
    cas_solver_core::external_proof::classify_nonzero_status_with_tri_prover(
        &simplifier.context,
        expr,
        crate::helpers::prove_nonzero_core,
    )
}

pub(crate) fn simplifier_is_known_negative(simplifier: &mut Simplifier, expr: ExprId) -> bool {
    cas_solver_core::isolation_utils::is_known_negative(&simplifier.context, expr)
}

pub(crate) fn medium_step(description: String, equation_after: Equation) -> SolveStep {
    SolveStep::new(
        description,
        equation_after,
        crate::step::ImportanceLevel::Medium,
    )
}
