use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::isolation_strategy::execute_rational_exponent_strategy_with_default_kernel_and_accept_all_solutions_and_unified_step_mapper_with_state;
use cas_solver_core::strategy_order::{dispatch_solve_strategy_kind_with_state, SolveStrategyKind};

use super::solve_entrypoints::solve_with_ctx_and_options;
use super::strategy_apply_advanced::{
    apply_quadratic_strategy, apply_rational_roots_strategy, apply_substitution_strategy,
};
use super::strategy_apply_basic::{
    apply_collect_terms_strategy, apply_isolation_strategy, apply_unwrap_strategy,
};
use super::{
    medium_step, simplifier_context_mut, simplifier_simplify_expr, SolveCtx, SolveStep,
    SolverOptions,
};

/// Apply one strategy selected by kind.
pub(super) fn apply_strategy(
    kind: SolveStrategyKind,
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    dispatch_solve_strategy_kind_with_state(
        simplifier,
        kind,
        |simplifier| apply_rational_exponent_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_substitution_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_unwrap_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_quadratic_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_rational_roots_strategy(equation, var, simplifier),
        |simplifier| apply_collect_terms_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_isolation_strategy(equation, var, simplifier, opts, ctx),
    )
}

pub(super) fn apply_rational_exponent_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    execute_rational_exponent_strategy_with_default_kernel_and_accept_all_solutions_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        simplifier_simplify_expr,
        |simplifier, equation, solve_var| {
            solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}
