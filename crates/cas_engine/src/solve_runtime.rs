//! Public solve runtime entrypoints and type aliases.

use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::SolutionSet;
use cas_solver_core::solve_types::cleanup_display_solve_steps;

pub type SolverOptions = cas_solver_core::solver_options::SolverOptions;

pub(crate) type SolveDomainEnv =
    cas_solver_core::solve_aliases::SolveDomainEnv<crate::ImplicitDomain>;

pub type SolveCtx = cas_solver_core::solve_aliases::SolveCtx<
    SolveDomainEnv,
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

pub type DisplaySolveSteps = cas_solver_core::solve_aliases::DisplaySolveSteps<SolveStep>;

pub type SolveDiagnostics = cas_solver_core::solve_aliases::SolveDiagnostics<
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    crate::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;

pub type SolveSubStep =
    cas_solver_core::solve_aliases::SolveSubStep<cas_ast::Equation, crate::ImportanceLevel>;

pub type SolveStep = cas_solver_core::solve_aliases::SolveStep<
    cas_ast::Equation,
    crate::ImportanceLevel,
    SolveSubStep,
>;

pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_options(eq, var, simplifier, SolverOptions::default())
}

pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    let ctx = SolveCtx::default();
    let result = crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, &ctx);
    cas_solver_core::solve_types::finalize_display_solve_with_ctx(
        &ctx,
        result,
        crate::collect_assumption_records,
        |raw_steps| {
            cleanup_display_solve_steps(
                &mut simplifier.context,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}

pub(crate) fn solve_with_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let ctx = SolveCtx::default();
    crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, &ctx)
}

pub fn solve_with_ctx_and_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, ctx)
}
