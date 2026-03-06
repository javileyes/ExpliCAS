use super::pipeline_strategy_apply::apply_strategy;
use crate::solve_runtime_adapters::{SolveCtx, SolveStep, SolverOptions};
use crate::{CasError, Simplifier};
use cas_ast::{Equation, SolutionSet};

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_strategy_pipeline(
    simplifier: &mut Simplifier,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: cas_ast::ExprId,
    var: &str,
    opts: SolverOptions,
    ctx: &SolveCtx,
    domain_exclusions: &[cas_ast::ExprId],
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_pipeline_strategy_execute_bound_runtime::execute_strategy_pipeline_with_runtime_state_and_apply_entrypoint_with_state(
        simplifier,
        original_eq,
        simplified_eq,
        diff_simplified,
        var,
        opts,
        ctx,
        domain_exclusions,
        apply_strategy,
    )
}
