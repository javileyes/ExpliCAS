//! Isolation-strategy orchestration helpers shared by engine-side solver.
//!
//! These wrappers keep strategy routing orchestration in `cas_solver_core`
//! while callers provide stateful hooks for recursive solving.

use cas_ast::{Context, Equation, SolutionSet};

use crate::strategy_kernels::{
    execute_collect_terms_kernel_result_pipeline_for_equation_with_item_with_state,
    execute_rational_exponent_kernel_result_pipeline_with_item_with_state,
    solve_isolation_strategy_routing_with, CollectTermsKernel, IsolationStrategyRouting,
    RationalExponentKernel, StrategyExecutionItem,
};
use crate::unwrap_plan::{
    route_unwrap_entry_with_item,
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

/// Execute isolation strategy using default routing derivation:
/// `derive_isolation_strategy_routing(ctx, equation, var)`.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolation_strategy_with_default_routing_with_state<
    T,
    S,
    E,
    FContextRef,
    FSolveEquation,
    FMapSwapStep,
    FVariableNotFoundError,
>(
    state: &mut T,
    eq: &Equation,
    var: &str,
    include_item: bool,
    context_ref: FContextRef,
    solve_equation: FSolveEquation,
    map_swap_item_to_step: FMapSwapStep,
    variable_not_found_error: FVariableNotFoundError,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FContextRef: Fn(&mut T) -> &Context,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapSwapStep: FnMut(crate::solve_outcome::TermIsolationRewriteExecutionItem) -> S,
    FVariableNotFoundError: FnMut(&str) -> E,
{
    execute_isolation_strategy_with_state(
        state,
        eq,
        var,
        include_item,
        |state, equation, solve_var| {
            crate::strategy_kernels::derive_isolation_strategy_routing(
                context_ref(state),
                equation,
                solve_var,
            )
        },
        solve_equation,
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

/// Execute unwrap strategy using default entry routing:
/// `route_unwrap_entry_with_item(ctx, equation, var, mode, wildcard_scope, residual_suffix, ...)`.
#[allow(clippy::too_many_arguments)]
pub fn execute_unwrap_strategy_with_default_route_with_state<
    T,
    S,
    E,
    FContextMut,
    FClassifyLogSolve,
    FRenderExpr,
    FMapTerminalStep,
    FNoteAssumption,
    FSolveEquation,
    FMapExecutionStep,
>(
    state: &mut T,
    eq: &Equation,
    var: &str,
    include_item: bool,
    context_mut: FContextMut,
    mode: crate::log_domain::DomainModeKind,
    wildcard_scope: bool,
    residual_suffix: &str,
    classify_log_solve: FClassifyLogSolve,
    render_expr: FRenderExpr,
    map_terminal_item_to_step: FMapTerminalStep,
    note_assumption: FNoteAssumption,
    solve_equation: FSolveEquation,
    map_execution_item_to_step: FMapExecutionStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FContextMut: Fn(&mut T) -> &mut Context,
    FClassifyLogSolve:
        FnMut(&Context, cas_ast::ExprId, cas_ast::ExprId) -> crate::log_domain::LogSolveDecision,
    FRenderExpr: FnMut(&Context, cas_ast::ExprId) -> String,
    FMapTerminalStep: FnMut(crate::solve_outcome::TermIsolationExecutionItem) -> S,
    FNoteAssumption: FnMut(&mut T, LogLinearAssumptionRecord),
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapExecutionStep: FnMut(UnwrapExecutionItem) -> S,
{
    let routing = route_unwrap_entry_with_item(
        context_mut(state),
        eq,
        var,
        mode,
        wildcard_scope,
        residual_suffix,
        include_item,
        classify_log_solve,
        render_expr,
        map_terminal_item_to_step,
    );
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

/// Execute collect-terms strategy orchestration from stateful callbacks:
/// 1) derive collect-terms kernel for `(equation, variable)`,
/// 2) run rewrite + recursive solve pipeline.
#[allow(clippy::too_many_arguments)]
pub fn execute_collect_terms_strategy_with_state<
    T,
    S,
    E,
    FDeriveKernel,
    FSimplifyExpr,
    FRenderExpr,
    FSolveEquation,
    FMapStep,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    include_item: bool,
    mut derive_kernel: FDeriveKernel,
    simplify_expr: FSimplifyExpr,
    render_expr: FRenderExpr,
    solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FDeriveKernel: FnMut(&mut T, &Equation, &str) -> Option<CollectTermsKernel>,
    FSimplifyExpr: FnMut(&mut T, cas_ast::ExprId) -> cas_ast::ExprId,
    FRenderExpr: FnMut(&mut T, cas_ast::ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(StrategyExecutionItem) -> S,
{
    execute_collect_terms_kernel_result_pipeline_for_equation_with_item_with_state(
        state,
        |state| derive_kernel(state, equation, var),
        equation,
        var,
        include_item,
        simplify_expr,
        render_expr,
        solve_equation,
        map_item_to_step,
    )
}

/// Execute rational-exponent strategy orchestration from stateful callbacks:
/// 1) derive rational-exponent kernel for `(equation, variable)`,
/// 2) run rewrite + recursive solve + candidate verification pipeline.
#[allow(clippy::too_many_arguments)]
pub fn execute_rational_exponent_strategy_with_state<
    T,
    S,
    E,
    FDeriveKernel,
    FSimplifyExpr,
    FSolveEquation,
    FMapStep,
    FVerifySolution,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    include_item: bool,
    mut derive_kernel: FDeriveKernel,
    simplify_expr: FSimplifyExpr,
    solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
    verify_solution: FVerifySolution,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FDeriveKernel: FnMut(&mut T, &Equation, &str) -> Option<RationalExponentKernel>,
    FSimplifyExpr: FnMut(&mut T, cas_ast::ExprId) -> cas_ast::ExprId,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(StrategyExecutionItem) -> S,
    FVerifySolution: FnMut(&mut T, cas_ast::ExprId) -> bool,
{
    execute_rational_exponent_kernel_result_pipeline_with_item_with_state(
        state,
        |state| derive_kernel(state, equation, var),
        var,
        include_item,
        simplify_expr,
        solve_equation,
        map_item_to_step,
        verify_solution,
    )
}

/// Execute collect-terms strategy using default core kernel derivation:
/// `derive_collect_terms_kernel(ctx, equation, var)`.
#[allow(clippy::too_many_arguments)]
pub fn execute_collect_terms_strategy_with_default_kernel_with_state<
    T,
    S,
    E,
    FContextMut,
    FSimplifyExpr,
    FRenderExpr,
    FSolveEquation,
    FMapStep,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    include_item: bool,
    mut context_mut: FContextMut,
    simplify_expr: FSimplifyExpr,
    render_expr: FRenderExpr,
    solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FContextMut: FnMut(&mut T) -> &mut Context,
    FSimplifyExpr: FnMut(&mut T, cas_ast::ExprId) -> cas_ast::ExprId,
    FRenderExpr: FnMut(&mut T, cas_ast::ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(StrategyExecutionItem) -> S,
{
    execute_collect_terms_strategy_with_state(
        state,
        equation,
        var,
        include_item,
        |state, equation, var| {
            crate::strategy_kernels::derive_collect_terms_kernel(context_mut(state), equation, var)
        },
        simplify_expr,
        render_expr,
        solve_equation,
        map_item_to_step,
    )
}

/// Execute rational-exponent strategy using default core kernel derivation:
/// `derive_rational_exponent_kernel_for_var(ctx, equation, var)`.
#[allow(clippy::too_many_arguments)]
pub fn execute_rational_exponent_strategy_with_default_kernel_with_state<
    T,
    S,
    E,
    FContextMut,
    FSimplifyExpr,
    FSolveEquation,
    FMapStep,
    FVerifySolution,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    include_item: bool,
    mut context_mut: FContextMut,
    simplify_expr: FSimplifyExpr,
    solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
    verify_solution: FVerifySolution,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FContextMut: FnMut(&mut T) -> &mut Context,
    FSimplifyExpr: FnMut(&mut T, cas_ast::ExprId) -> cas_ast::ExprId,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(StrategyExecutionItem) -> S,
    FVerifySolution: FnMut(&mut T, cas_ast::ExprId) -> bool,
{
    execute_rational_exponent_strategy_with_state(
        state,
        equation,
        var,
        include_item,
        |state, equation, var| {
            crate::strategy_kernels::derive_rational_exponent_kernel_for_var(
                context_mut(state),
                equation,
                var,
            )
        },
        simplify_expr,
        solve_equation,
        map_item_to_step,
        verify_solution,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        execute_collect_terms_strategy_with_default_kernel_with_state,
        execute_collect_terms_strategy_with_state, execute_isolation_strategy_with_state,
        execute_rational_exponent_strategy_with_default_kernel_with_state,
        execute_rational_exponent_strategy_with_state, execute_unwrap_strategy_with_state,
    };
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

    #[test]
    fn execute_collect_terms_strategy_with_state_returns_none_when_kernel_missing() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(2);
        let mut state = ();
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let out = execute_collect_terms_strategy_with_state(
            &mut state,
            &eq,
            "x",
            true,
            |_state, _equation, _var| None,
            |_state, expr| expr,
            |_state, _expr| "rhs".to_string(),
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((
                    SolutionSet::AllReals,
                    vec!["subsolve".to_string()],
                ))
            },
            |_item| "item".to_string(),
        );

        assert!(out.is_none());
    }

    #[test]
    fn execute_rational_exponent_strategy_with_state_returns_none_when_kernel_missing() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(2);
        let mut state = ();
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let out = execute_rational_exponent_strategy_with_state(
            &mut state,
            &eq,
            "x",
            true,
            |_state, _equation, _var| None,
            |_state, expr| expr,
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((
                    SolutionSet::AllReals,
                    vec!["subsolve".to_string()],
                ))
            },
            |_item| "item".to_string(),
            |_state, _expr| true,
        );

        assert!(out.is_none());
    }

    #[test]
    fn execute_collect_terms_strategy_with_default_kernel_with_state_returns_none() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(2);
        let mut state = ctx;
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let out = execute_collect_terms_strategy_with_default_kernel_with_state(
            &mut state,
            &eq,
            "x",
            true,
            |state| state,
            |_state, expr| expr,
            |_state, _expr| "rhs".to_string(),
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((SolutionSet::AllReals, vec![]))
            },
            |_item| "item".to_string(),
        );
        assert!(out.is_none());
    }

    #[test]
    fn execute_rational_exponent_strategy_with_default_kernel_with_state_returns_none() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(2);
        let mut state = ctx;
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let out = execute_rational_exponent_strategy_with_default_kernel_with_state(
            &mut state,
            &eq,
            "x",
            true,
            |state| state,
            |_state, expr| expr,
            |_state, _equation, _var| {
                Ok::<(SolutionSet, Vec<String>), &'static str>((SolutionSet::AllReals, vec![]))
            },
            |_item| "item".to_string(),
            |_state, _expr| true,
        );
        assert!(out.is_none());
    }
}
