use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{
    medium_step, simplifier_context_mut, simplifier_expand_expr, simplifier_simplify_expr,
    SolveStep,
};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::rational_roots::execute_rational_roots_strategy_with_default_limits_and_default_root_sorting_and_unified_step_mapper_with_state;

pub(super) fn apply_rational_roots_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    let solved = execute_rational_roots_strategy_with_default_limits_and_default_root_sorting_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        simplifier_simplify_expr,
        simplifier_expand_expr,
        medium_step,
    )?;
    Some(Ok(solved))
}
