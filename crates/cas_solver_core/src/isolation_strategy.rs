//! Isolation-strategy orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep strategy routing orchestration in `cas_solver_core`
//! while callers provide stateful hooks for recursive solving.

use cas_ast::{Equation, SolutionSet};

use crate::strategy_kernels::{solve_isolation_strategy_routing_with, IsolationStrategyRouting};

/// Execute isolation strategy from stateful callbacks:
/// 1) derive side-routing for variable placement,
/// 2) solve or defer according to routing policy.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolation_strategy_with_state<
    T,
    S,
    E,
    FDeriveRouting,
    FSolveEquation,
    FMapSwapStep,
    FVariableNotFoundError,
>(
    state: &mut T,
    eq: &Equation,
    var: &str,
    include_item: bool,
    mut derive_routing: FDeriveRouting,
    mut solve_equation: FSolveEquation,
    map_swap_item_to_step: FMapSwapStep,
    variable_not_found_error: FVariableNotFoundError,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FDeriveRouting: FnMut(&mut T, &Equation, &str) -> IsolationStrategyRouting,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapSwapStep: FnMut(crate::solve_outcome::TermIsolationRewriteExecutionItem) -> S,
    FVariableNotFoundError: FnMut(&str) -> E,
{
    let routing = derive_routing(state, eq, var);
    solve_isolation_strategy_routing_with(
        routing,
        eq,
        var,
        include_item,
        |equation, solve_var| solve_equation(state, equation, solve_var),
        map_swap_item_to_step,
        variable_not_found_error,
    )
}

#[cfg(test)]
mod tests {
    use super::execute_isolation_strategy_with_state;
    use cas_ast::{Equation, RelOp, SolutionSet};

    #[test]
    fn execute_isolation_strategy_with_state_returns_variable_not_found_error() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.num(1);
        let rhs = ctx.num(2);
        let mut state = ();
        let eq = Equation { lhs, rhs, op: RelOp::Eq };
        let out = execute_isolation_strategy_with_state(
            &mut state,
            &eq,
            "x",
            false,
            |_state, _eq, _var| crate::strategy_kernels::IsolationStrategyRouting::VariableNotFound,
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((SolutionSet::AllReals, vec![]))
            },
            |_item| "unused".to_string(),
            |_var| "missing",
        );
        let err = out.expect("must return result").expect_err("must be error");
        assert_eq!(err, "missing");
    }

    #[test]
    fn execute_isolation_strategy_with_state_returns_none_for_both_sides() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.num(1);
        let rhs = ctx.num(2);
        let mut state = ();
        let eq = Equation { lhs, rhs, op: RelOp::Eq };
        let out = execute_isolation_strategy_with_state(
            &mut state,
            &eq,
            "x",
            false,
            |_state, _eq, _var| {
                crate::strategy_kernels::IsolationStrategyRouting::VariableOnBothSides
            },
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((SolutionSet::AllReals, vec![]))
            },
            |_item| "unused".to_string(),
            |_var| "missing",
        );
        assert!(out.is_none());
    }
}
