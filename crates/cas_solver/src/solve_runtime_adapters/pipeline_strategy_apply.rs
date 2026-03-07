use crate::solve_runtime_adapters::{
    isolate_with_default_depth, SolveCtx, SolveStep, SolverOptions,
};
use crate::{CasError, Simplifier};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::strategy_order::SolveStrategyKind;

pub(crate) fn apply_strategy(
    kind: SolveStrategyKind,
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    cas_solver_core::solve_runtime_pipeline_strategy_apply_bound_runtime::apply_strategy_with_runtime_state_and_reentrant_entrypoints_and_state(
        simplifier,
        kind,
        equation,
        var,
        *opts,
        ctx,
        crate::solve_core_runtime::solve_inner,
        isolate_with_default_depth,
    )
}
