use crate::solve_runtime_adapters::SolverOptions;
use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{Equation, ExprId, SolutionSet};

pub(crate) fn solve_equation_with_solver_ctx(
    simplifier: &mut Simplifier,
    equation: &Equation,
    solve_var: &str,
    opts: SolverOptions,
    solve_ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    crate::solve_core_runtime::solve_inner(equation, solve_var, simplifier, opts, solve_ctx)
}

pub(crate) fn isolate_equation_with_solver_ctx(
    simplifier: &mut Simplifier,
    equation: &Equation,
    solve_var: &str,
    opts: SolverOptions,
    solve_ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    crate::solve_isolation_runtime::isolate(
        equation.lhs,
        equation.rhs,
        equation.op.clone(),
        solve_var,
        simplifier,
        opts,
        solve_ctx,
    )
}

pub(crate) fn classify_log_solve_with_solver_ctx(
    ctx: &cas_ast::Context,
    base: ExprId,
    other_side: ExprId,
    opts: &SolverOptions,
    solve_ctx: &SolveCtx,
) -> cas_solver_core::log_domain::LogSolveDecision {
    cas_solver_core::solve_runtime_flow::classify_log_solve_with_domain_env_and_runtime_positive_prover(
        ctx,
        base,
        other_side,
        opts.value_domain,
        opts.core_domain_mode(),
        &solve_ctx.domain_env,
        crate::proof_runtime::prove_positive,
    )
}

pub(crate) fn note_log_assumption_with_solver_ctx(
    ctx: &cas_ast::Context,
    base: ExprId,
    other_side: ExprId,
    assumption: cas_solver_core::log_domain::LogAssumption,
    solve_ctx: &SolveCtx,
) {
    cas_solver_core::solve_runtime_flow::note_log_assumption_with_runtime_sink(
        ctx,
        base,
        other_side,
        assumption,
        |event| solve_ctx.note_assumption(event),
    );
}

pub(crate) fn note_log_blocked_hint_with_default_sink(
    ctx: &cas_ast::Context,
    hint: cas_solver_core::solve_outcome::LogBlockedHintRecord,
) {
    cas_solver_core::solve_runtime_flow::note_log_blocked_hint_with_runtime_sink(
        ctx,
        hint,
        crate::register_blocked_hint,
    );
}
