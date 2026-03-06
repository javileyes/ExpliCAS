use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute solve strategy dispatch with the default strategy order and
/// per-strategy verification policy.
///
/// Callers provide:
/// - variable-presence check and residual fast-path resolver,
/// - cycle-guard entry,
/// - strategy application for each [`crate::strategy_order::SolveStrategyKind`],
/// - soft-error classifier,
/// - discrete-candidate resolution against `(equation, var)`.
#[allow(clippy::too_many_arguments)]
pub fn execute_default_strategy_order_pipeline_with_state<
    SState,
    S,
    E,
    Guard,
    FContainsVar,
    FResolveVarEliminated,
    FEnterCycle,
    FApplyStrategy,
    FSoftError,
    FResolveDiscreteAgainstEquation,
>(
    state: &mut SState,
    original_equation: &Equation,
    normalized_equation: &Equation,
    residual: ExprId,
    var: &str,
    contains_var: FContainsVar,
    resolve_var_eliminated: FResolveVarEliminated,
    enter_cycle: FEnterCycle,
    mut apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    mut resolve_discrete_against_equation: FResolveDiscreteAgainstEquation,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FResolveVarEliminated: FnMut(&mut SState, ExprId, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FEnterCycle: FnMut(&mut SState, &Equation, &str) -> Result<Guard, E>,
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FSoftError: FnMut(&E) -> bool,
    FResolveDiscreteAgainstEquation:
        FnMut(&mut SState, &Equation, &str, Vec<ExprId>, Vec<S>) -> (SolutionSet, Vec<S>),
{
    let strategy_order = crate::strategy_order::default_solve_strategy_order();
    crate::solve_analysis::execute_prepared_equation_strategy_pipeline_with_state(
        state,
        normalized_equation,
        residual,
        var,
        strategy_order,
        contains_var,
        resolve_var_eliminated,
        enter_cycle,
        |state, strategy_kind| {
            let should_verify = crate::strategy_order::strategy_should_verify(*strategy_kind);
            let attempt = apply_strategy(state, *strategy_kind);
            (attempt, should_verify)
        },
        is_soft_error,
        |state, solutions, steps| {
            resolve_discrete_against_equation(state, original_equation, var, solutions, steps)
        },
        no_solution_error,
    )
}
