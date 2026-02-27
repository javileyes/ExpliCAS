//! Isolation-strategy orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep strategy routing orchestration in `cas_solver_core`
//! while callers provide stateful hooks for recursive solving.

use cas_ast::{Equation, SolutionSet};

use crate::strategy_kernels::{solve_isolation_strategy_routing_with, IsolationStrategyRouting};
use crate::unwrap_plan::{
    solve_unwrap_entry_routing_option_with_execution_pipeline_with_item_with_state,
    LogLinearAssumptionRecord, UnwrapEntryRouting, UnwrapExecutionItem,
};

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

/// Execute unwrap strategy orchestration from stateful callbacks:
/// 1) derive unwrap routing (terminal or execution),
/// 2) resolve terminal immediately or execute recursive solve pipeline.
#[allow(clippy::too_many_arguments)]
pub fn execute_unwrap_strategy_with_state<
    T,
    S,
    E,
    FRouteUnwrapEntry,
    FNoteAssumption,
    FSolveEquation,
    FMapExecutionStep,
>(
    state: &mut T,
    eq: &Equation,
    var: &str,
    include_item: bool,
    mut route_unwrap_entry: FRouteUnwrapEntry,
    note_assumption: FNoteAssumption,
    solve_equation: FSolveEquation,
    map_execution_item_to_step: FMapExecutionStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FRouteUnwrapEntry: FnMut(&mut T, &Equation, &str, bool) -> Option<UnwrapEntryRouting<S>>,
    FNoteAssumption: FnMut(&mut T, LogLinearAssumptionRecord),
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapExecutionStep: FnMut(UnwrapExecutionItem) -> S,
{
    let routing = route_unwrap_entry(state, eq, var, include_item);
    solve_unwrap_entry_routing_option_with_execution_pipeline_with_item_with_state(
        state,
        routing,
        var,
        include_item,
        note_assumption,
        solve_equation,
        map_execution_item_to_step,
    )
}

#[cfg(test)]
mod tests {
    use super::{execute_isolation_strategy_with_state, execute_unwrap_strategy_with_state};
    use crate::unwrap_plan::{
        LogLinearAssumptionRecord, UnwrapEntryRouting, UnwrapEquationExecution,
        UnwrapExecutionItem, UnwrapExecutionPlan,
    };
    use cas_ast::{Equation, RelOp, SolutionSet};

    #[test]
    fn execute_isolation_strategy_with_state_returns_variable_not_found_error() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.num(1);
        let rhs = ctx.num(2);
        let mut state = ();
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
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
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
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

    #[test]
    fn execute_unwrap_strategy_with_state_returns_none_when_route_is_none() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.num(1);
        let rhs = ctx.num(2);
        let mut state = ();
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let out = execute_unwrap_strategy_with_state(
            &mut state,
            &eq,
            "x",
            false,
            |_state, _eq, _var, _include_item| None,
            |_state, _record: LogLinearAssumptionRecord| {},
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((SolutionSet::AllReals, vec![]))
            },
            |item: UnwrapExecutionItem| item.description().to_string(),
        );
        assert!(out.is_none());
    }

    #[test]
    fn execute_unwrap_strategy_with_state_routes_execution_and_merges_steps() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let y = ctx.num(2);
        let mut state = ();
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let out = execute_unwrap_strategy_with_state(
            &mut state,
            &eq,
            "x",
            true,
            |_state, equation, _var, _include_item| {
                Some(UnwrapEntryRouting::Execution(UnwrapEquationExecution {
                    execution: UnwrapExecutionPlan {
                        equation: equation.clone(),
                        description: "unwrap".to_string(),
                        assumptions: vec![],
                        log_linear_base: None,
                        items: vec![UnwrapExecutionItem {
                            equation: equation.clone(),
                            description: "unwrap-step".to_string(),
                        }],
                    },
                    other_side: equation.rhs,
                }))
            },
            |_state, _record: LogLinearAssumptionRecord| {},
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((
                    SolutionSet::AllReals,
                    vec!["subsolve".to_string()],
                ))
            },
            |item: UnwrapExecutionItem| item.description().to_string(),
        )
        .expect("route should exist")
        .expect("execution should solve");

        assert!(matches!(out.0, SolutionSet::AllReals));
        assert_eq!(
            out.1,
            vec!["unwrap-step".to_string(), "subsolve".to_string()]
        );
    }
}
