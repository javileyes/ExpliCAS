//! Core solve dispatch pipeline.
//!
//! Contains the internal solve pipeline (`solve_inner`).

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::preflight::build_solve_preflight_state;
use crate::solver::preparation::prepare_equation_for_strategy;
use crate::solver::strategy_pipeline::execute_strategy_pipeline;
use crate::solver::strategy_precheck::try_rational_exponent_precheck;
use cas_ast::SolutionSet;
use cas_solver_core::solve_analysis::{
    ensure_solve_entry_for_equation_or_error, guard_solved_result_with_exclusions,
};
use cas_solver_core::solve_budget::MAX_SOLVE_RECURSION_DEPTH;

use super::{SolveStep, SolverOptions};

/// Core solver implementation.
///
/// All public entry points delegate here. `parent_ctx` carries the shared
/// accumulator so that conditions from recursive calls are visible to the
/// top-level caller.
pub(super) fn solve_inner(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    parent_ctx: &super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let current_depth = parent_ctx.depth().saturating_add(1);
    ensure_solve_entry_for_equation_or_error(
        &simplifier.context,
        eq,
        var,
        current_depth,
        MAX_SOLVE_RECURSION_DEPTH,
        || {
            CasError::SolverError(
                "Maximum solver recursion depth exceeded. The equation may be too complex."
                    .to_string(),
            )
        },
        || CasError::VariableNotFound(var.to_string()),
    )?;

    let preflight = build_solve_preflight_state(simplifier, eq, var, opts.value_domain, parent_ctx);
    let domain_exclusions = preflight.domain_exclusions;
    let ctx = preflight.ctx;

    // EARLY CHECK: Handle rational exponent equations BEFORE simplification
    // This prevents x^(3/2) from being simplified to |x|*sqrt(x) which causes loops
    if let Some(result) = try_rational_exponent_precheck(eq, var, simplifier, &opts, &ctx) {
        return guard_solved_result_with_exclusions(result, &domain_exclusions);
    }

    // 2) Pre-strategy equation normalization:
    // simplify var-containing sides, cancel common additive terms, and
    // normalize residual fallbacks before running strategy dispatch.
    let prepared = prepare_equation_for_strategy(simplifier, eq, var);
    let simplified_eq = prepared.equation;
    let diff_simplified = prepared.residual;

    // NOTE: Pre-solve exponent normalization (Div(p,q) → Number(p/q)),
    // nested-pow folding, and additive cancellation are applied above.
    cas_solver_core::solve_analysis::debug_assert_equation_no_top_level_sub(
        &simplifier.context,
        &simplified_eq,
    );

    // 3) Resolve var-eliminated residuals early, otherwise guard cycle + run strategies.
    execute_strategy_pipeline(
        simplifier,
        eq,
        &simplified_eq,
        diff_simplified,
        var,
        opts,
        &ctx,
        &domain_exclusions,
    )
}
