//! Isolation dispatcher helpers shared by engine-side solver.
//!
//! These helpers keep LHS shape-routing logic in `cas_solver_core`, while
//! callers provide stateful handlers for each isolation branch.

use cas_ast::symbol::SymbolId;
use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};

use crate::solve_outcome::{
    plan_negated_lhs_isolation_step, residual_solution_set, resolve_isolated_variable_outcome,
    solve_isolated_variable_lhs_with_resolver_with_state,
    solve_negated_lhs_isolation_plan_with_and_merge_with_existing_steps,
};

/// Routed LHS shape for equation isolation.
#[derive(Debug, Clone, PartialEq)]
pub enum IsolationDispatchRoute {
    IsolatedVariable,
    Add { left: ExprId, right: ExprId },
    Sub { left: ExprId, right: ExprId },
    Mul { left: ExprId, right: ExprId },
    Div { left: ExprId, right: ExprId },
    Pow { base: ExprId, exponent: ExprId },
    Function { fn_id: SymbolId, args: Vec<ExprId> },
    Neg { inner: ExprId },
    Unsupported { lhs_expr: Expr },
}

/// Derive dispatch route from the current LHS expression and solve variable.
pub fn derive_isolation_dispatch_route(
    ctx: &Context,
    lhs: ExprId,
    var: &str,
) -> IsolationDispatchRoute {
    match ctx.get(lhs).clone() {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => {
            IsolationDispatchRoute::IsolatedVariable
        }
        Expr::Add(left, right) => IsolationDispatchRoute::Add { left, right },
        Expr::Sub(left, right) => IsolationDispatchRoute::Sub { left, right },
        Expr::Mul(left, right) => IsolationDispatchRoute::Mul { left, right },
        Expr::Div(left, right) => IsolationDispatchRoute::Div { left, right },
        Expr::Pow(base, exponent) => IsolationDispatchRoute::Pow { base, exponent },
        Expr::Function(fn_id, args) => IsolationDispatchRoute::Function { fn_id, args },
        Expr::Neg(inner) => IsolationDispatchRoute::Neg { inner },
        lhs_expr => IsolationDispatchRoute::Unsupported { lhs_expr },
    }
}

/// Derive and execute one isolation dispatch route from `(ctx, lhs, var)`
/// with stateful branch handlers.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolation_dispatch_route_for_var_with_state<
    T,
    R,
    E,
    FContext,
    FIsolatedVariable,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FPow,
    FFunction,
    FNeg,
    FUnsupported,
>(
    state: &mut T,
    context: FContext,
    lhs: ExprId,
    var: &str,
    on_isolated_variable: FIsolatedVariable,
    on_add: FAdd,
    on_sub: FSub,
    on_mul: FMul,
    on_div: FDiv,
    on_pow: FPow,
    on_function: FFunction,
    on_neg: FNeg,
    on_unsupported: FUnsupported,
) -> Result<R, E>
where
    FContext: FnMut(&mut T) -> &Context,
    FIsolatedVariable: FnOnce(&mut T) -> Result<R, E>,
    FAdd: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FSub: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FMul: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FDiv: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FPow: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FFunction: FnOnce(&mut T, SymbolId, Vec<ExprId>) -> Result<R, E>,
    FNeg: FnOnce(&mut T, ExprId) -> Result<R, E>,
    FUnsupported: FnOnce(&mut T, Expr) -> Result<R, E>,
{
    let mut context = context;
    let route = derive_isolation_dispatch_route(context(state), lhs, var);
    execute_isolation_dispatch_route_with_state(
        state,
        route,
        on_isolated_variable,
        on_add,
        on_sub,
        on_mul,
        on_div,
        on_pow,
        on_function,
        on_neg,
        on_unsupported,
    )
}

/// Execute one isolation dispatch route with stateful branch handlers.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolation_dispatch_route_with_state<
    T,
    R,
    E,
    FIsolatedVariable,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FPow,
    FFunction,
    FNeg,
    FUnsupported,
>(
    state: &mut T,
    route: IsolationDispatchRoute,
    on_isolated_variable: FIsolatedVariable,
    on_add: FAdd,
    on_sub: FSub,
    on_mul: FMul,
    on_div: FDiv,
    on_pow: FPow,
    on_function: FFunction,
    on_neg: FNeg,
    on_unsupported: FUnsupported,
) -> Result<R, E>
where
    FIsolatedVariable: FnOnce(&mut T) -> Result<R, E>,
    FAdd: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FSub: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FMul: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FDiv: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FPow: FnOnce(&mut T, ExprId, ExprId) -> Result<R, E>,
    FFunction: FnOnce(&mut T, SymbolId, Vec<ExprId>) -> Result<R, E>,
    FNeg: FnOnce(&mut T, ExprId) -> Result<R, E>,
    FUnsupported: FnOnce(&mut T, Expr) -> Result<R, E>,
{
    match route {
        IsolationDispatchRoute::IsolatedVariable => on_isolated_variable(state),
        IsolationDispatchRoute::Add { left, right } => on_add(state, left, right),
        IsolationDispatchRoute::Sub { left, right } => on_sub(state, left, right),
        IsolationDispatchRoute::Mul { left, right } => on_mul(state, left, right),
        IsolationDispatchRoute::Div { left, right } => on_div(state, left, right),
        IsolationDispatchRoute::Pow { base, exponent } => on_pow(state, base, exponent),
        IsolationDispatchRoute::Function { fn_id, args } => on_function(state, fn_id, args),
        IsolationDispatchRoute::Neg { inner } => on_neg(state, inner),
        IsolationDispatchRoute::Unsupported { lhs_expr } => on_unsupported(state, lhs_expr),
    }
}

/// Execute an isolated-variable entry (`x op rhs`) with default core
/// outcome resolution and residual fallback.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolated_variable_entry_with_default_resolution_with_state<
    T,
    S,
    FContextForResolve,
    FContextForResidual,
    FSimplify,
    FTry1,
    FTry2,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    context_for_resolve: FContextForResolve,
    context_for_residual: FContextForResidual,
    simplify_rhs: FSimplify,
    try_linear_collect: FTry1,
    try_linear_collect_v2: FTry2,
) -> (SolutionSet, Vec<S>)
where
    FContextForResolve: FnMut(&mut T) -> &mut Context,
    FContextForResidual: FnMut(&mut T) -> &mut Context,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FTry1: FnMut(&mut T, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FTry2: FnMut(&mut T, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
{
    let mut context_for_resolve = context_for_resolve;
    let mut context_for_residual = context_for_residual;
    solve_isolated_variable_lhs_with_resolver_with_state(
        state,
        lhs,
        rhs,
        op,
        var,
        |state, sim_rhs, rel_op, solve_var| {
            resolve_isolated_variable_outcome(
                context_for_resolve(state),
                sim_rhs,
                rel_op,
                solve_var,
            )
        },
        simplify_rhs,
        try_linear_collect,
        try_linear_collect_v2,
        |state, solve_lhs, solve_rhs, solve_var| {
            residual_solution_set(context_for_residual(state), solve_lhs, solve_rhs, solve_var)
        },
    )
}

/// Convenience variant for isolated-variable entry when the same mutable
/// context accessor is used both for outcome resolution and residual fallback.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolated_variable_entry_with_default_resolution_single_context_with_state<
    T,
    S,
    FContextMut,
    FSimplify,
    FTry1,
    FTry2,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    context_mut: FContextMut,
    simplify_rhs: FSimplify,
    try_linear_collect: FTry1,
    try_linear_collect_v2: FTry2,
) -> (SolutionSet, Vec<S>)
where
    FContextMut: Fn(&mut T) -> &mut Context,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FTry1: FnMut(&mut T, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FTry2: FnMut(&mut T, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
{
    execute_isolated_variable_entry_with_default_resolution_with_state(
        state,
        lhs,
        rhs,
        op,
        var,
        |state| context_mut(state),
        |state| context_mut(state),
        simplify_rhs,
        try_linear_collect,
        try_linear_collect_v2,
    )
}

/// Execute one full isolation-dispatch step with:
/// - default isolated-variable resolution kernels
/// - default negated-LHS rewrite planning and step merge
///
/// while callers provide stateful handlers for arithmetic/function branches.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolation_dispatch_with_default_isolated_and_negated_entries_for_var_with_state<
    T,
    S,
    E,
    FContextRef,
    FContextMut,
    FSimplifyRhs,
    FTryLinearCollect,
    FTryLinearCollectV2,
    FOnAdd,
    FOnSub,
    FOnMul,
    FOnDiv,
    FOnPow,
    FOnFunction,
    FCollectNegatedItem,
    FSolveNegatedRewritten,
    FMapNegatedStep,
    FMapUnsupportedError,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    context_ref: FContextRef,
    context_mut: FContextMut,
    simplify_rhs: FSimplifyRhs,
    try_linear_collect: FTryLinearCollect,
    try_linear_collect_v2: FTryLinearCollectV2,
    on_add: FOnAdd,
    on_sub: FOnSub,
    on_mul: FOnMul,
    on_div: FOnDiv,
    on_pow: FOnPow,
    on_function: FOnFunction,
    collect_negated_item: FCollectNegatedItem,
    solve_negated_rewritten: FSolveNegatedRewritten,
    map_negated_step: FMapNegatedStep,
    map_unsupported_error: FMapUnsupportedError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContextRef: Fn(&mut T) -> &Context,
    FContextMut: Fn(&mut T) -> &mut Context,
    FSimplifyRhs: FnMut(&mut T, ExprId) -> ExprId,
    FTryLinearCollect: FnMut(&mut T, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FTryLinearCollectV2: FnMut(&mut T, ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FOnAdd: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnSub: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnMul: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnDiv: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnPow: FnMut(&mut T, ExprId, ExprId) -> Result<(SolutionSet, Vec<S>), E>,
    FOnFunction: FnMut(&mut T, SymbolId, Vec<ExprId>) -> Result<(SolutionSet, Vec<S>), E>,
    FCollectNegatedItem: FnMut(&mut T) -> bool,
    FSolveNegatedRewritten: FnMut(&mut T, Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMapNegatedStep: FnMut(crate::solve_outcome::TermIsolationRewriteExecutionItem) -> S,
    FMapUnsupportedError: FnMut(&mut T, Expr) -> E,
{
    let context_ref = &context_ref;
    let context_mut = &context_mut;
    let mut simplify_rhs = simplify_rhs;
    let mut try_linear_collect = try_linear_collect;
    let mut try_linear_collect_v2 = try_linear_collect_v2;
    let mut on_add = on_add;
    let mut on_sub = on_sub;
    let mut on_mul = on_mul;
    let mut on_div = on_div;
    let mut on_pow = on_pow;
    let mut on_function = on_function;
    let mut collect_negated_item = collect_negated_item;
    let mut solve_negated_rewritten = solve_negated_rewritten;
    let mut map_negated_step = map_negated_step;
    let mut map_unsupported_error = map_unsupported_error;

    execute_isolation_dispatch_route_for_var_with_state(
        state,
        context_ref,
        lhs,
        var,
        |state| {
            let solved =
                execute_isolated_variable_entry_with_default_resolution_single_context_with_state(
                    state,
                    lhs,
                    rhs,
                    op.clone(),
                    var,
                    context_mut,
                    &mut simplify_rhs,
                    &mut try_linear_collect,
                    &mut try_linear_collect_v2,
                );
            Ok(solved)
        },
        &mut on_add,
        &mut on_sub,
        &mut on_mul,
        &mut on_div,
        &mut on_pow,
        &mut on_function,
        |state, inner| {
            let include_item = collect_negated_item(state);
            execute_negated_lhs_entry_with_default_plan_and_merge_with_existing_steps_with_state(
                state,
                inner,
                rhs,
                op.clone(),
                var,
                include_item,
                Vec::new(),
                context_mut,
                &mut solve_negated_rewritten,
                &mut map_negated_step,
            )
        },
        |state, lhs_expr| Err(map_unsupported_error(state, lhs_expr)),
    )
}

/// Execute a negated-LHS entry (`-A op rhs`) with default core negation rewrite
/// planning, then delegate solving of rewritten equation via callback.
#[allow(clippy::too_many_arguments)]
pub fn execute_negated_lhs_entry_with_default_plan_and_merge_with_existing_steps_with_state<
    T,
    E,
    S,
    FContextMut,
    FSolveRewritten,
    FStep,
>(
    state: &mut T,
    inner: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    include_item: bool,
    existing_steps: Vec<S>,
    mut context_mut: FContextMut,
    mut solve_rewritten: FSolveRewritten,
    map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContextMut: FnMut(&mut T) -> &mut Context,
    FSolveRewritten: FnMut(&mut T, Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(crate::solve_outcome::TermIsolationRewriteExecutionItem) -> S,
{
    let rewrite = plan_negated_lhs_isolation_step(context_mut(state), inner, rhs, op);
    solve_negated_lhs_isolation_plan_with_and_merge_with_existing_steps(
        rewrite,
        var,
        include_item,
        existing_steps,
        |equation, solve_var| solve_rewritten(state, equation, solve_var),
        map_item_to_step,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        derive_isolation_dispatch_route,
        execute_isolated_variable_entry_with_default_resolution_with_state,
        execute_isolation_dispatch_route_for_var_with_state,
        execute_isolation_dispatch_route_with_state,
        execute_isolation_dispatch_with_default_isolated_and_negated_entries_for_var_with_state,
        execute_negated_lhs_entry_with_default_plan_and_merge_with_existing_steps_with_state,
        IsolationDispatchRoute,
    };
    use cas_ast::{Context, Expr, RelOp, SolutionSet};

    #[test]
    fn derive_isolation_dispatch_route_detects_isolated_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let route = derive_isolation_dispatch_route(&ctx, x, "x");
        assert_eq!(route, IsolationDispatchRoute::IsolatedVariable);
    }

    #[test]
    fn execute_isolation_dispatch_route_with_state_calls_matching_branch() {
        let mut ctx = Context::new();
        let inner = ctx.var("x");
        let mut state = 0usize;
        let out = execute_isolation_dispatch_route_with_state(
            &mut state,
            IsolationDispatchRoute::Neg { inner },
            |_state| Ok::<_, &'static str>("iso"),
            |_state, _l, _r| Ok("add"),
            |_state, _l, _r| Ok("sub"),
            |_state, _l, _r| Ok("mul"),
            |_state, _l, _r| Ok("div"),
            |_state, _b, _e| Ok("pow"),
            |_state, _fn_id, _args| Ok("fn"),
            |state, _inner| {
                *state += 1;
                Ok("neg")
            },
            |_state, _expr| Ok("unsupported"),
        )
        .expect("route should resolve");
        assert_eq!(out, "neg");
        assert_eq!(state, 1);
    }

    #[test]
    fn execute_isolation_dispatch_route_for_var_with_state_derives_and_dispatches() {
        let mut state = Context::new();
        let x = state.var("x");
        let one = state.num(1);
        let add = state.add(Expr::Add(x, one));
        let mut hits = 0usize;

        let out = execute_isolation_dispatch_route_for_var_with_state(
            &mut state,
            |ctx| ctx,
            add,
            "x",
            |_ctx| Ok::<_, &'static str>("iso"),
            |_, _, _| Ok("add"),
            |_, _, _| Ok("sub"),
            |_, _, _| Ok("mul"),
            |_, _, _| Ok("div"),
            |_, _, _| Ok("pow"),
            |_, _, _| Ok("fn"),
            |_, _| Ok("neg"),
            |_, _| {
                hits += 1;
                Ok("unsupported")
            },
        )
        .expect("derived route should dispatch");

        assert_eq!(out, "add");
        assert_eq!(hits, 0);
    }

    #[test]
    fn execute_isolation_dispatch_with_default_isolated_and_negated_entries_routes_add_branch() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let add = ctx.add(Expr::Add(x, one));

        let solved =
            execute_isolation_dispatch_with_default_isolated_and_negated_entries_for_var_with_state(
                &mut ctx,
                add,
                two,
                RelOp::Eq,
                "x",
                |ctx| ctx,
                |ctx| ctx,
                |_ctx, expr| expr,
                |_ctx, _lhs, _rhs, _var| None::<(SolutionSet, Vec<String>)>,
                |_ctx, _lhs, _rhs, _var| None::<(SolutionSet, Vec<String>)>,
                |_ctx, left, right| {
                    Ok::<(SolutionSet, Vec<String>), &'static str>((
                        SolutionSet::Discrete(vec![left, right]),
                        vec!["add".to_string()],
                    ))
                },
                |_ctx, _left, _right| Err("unexpected-sub"),
                |_ctx, _left, _right| Err("unexpected-mul"),
                |_ctx, _left, _right| Err("unexpected-div"),
                |_ctx, _base, _exp| Err("unexpected-pow"),
                |_ctx, _fn_id, _args| Err("unexpected-fn"),
                |_ctx| true,
                |_ctx, _eq, _var| Err("unexpected-neg"),
                |_item| "neg-step".to_string(),
                |_ctx, _lhs_expr| "unexpected-unsupported",
            )
            .expect("add branch should resolve");

        let (solutions, steps) = solved;
        assert_eq!(steps, vec!["add".to_string()]);
        assert!(matches!(solutions, SolutionSet::Discrete(_)));
    }

    #[test]
    fn derive_isolation_dispatch_route_marks_unsupported_shape() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let matrix = ctx.add(Expr::Matrix {
            rows: 1,
            cols: 1,
            data: vec![one],
        });
        let route = derive_isolation_dispatch_route(&ctx, matrix, "x");
        match route {
            IsolationDispatchRoute::Unsupported { lhs_expr } => {
                assert!(matches!(lhs_expr, Expr::Matrix { .. }));
            }
            other => panic!("expected unsupported route, got {other:?}"),
        }
    }

    #[derive(Default)]
    struct IsolatedState {
        context: Context,
    }

    #[test]
    fn execute_isolated_variable_entry_with_default_resolution_with_state_solves_eq() {
        let mut state = IsolatedState::default();
        let x = state.context.var("x");
        let two = state.context.num(2);

        let (set, steps) = execute_isolated_variable_entry_with_default_resolution_with_state(
            &mut state,
            x,
            two,
            RelOp::Eq,
            "x",
            |state| &mut state.context,
            |state| &mut state.context,
            |_state, expr| expr,
            |_state, _lhs, _rhs, _var| None::<(SolutionSet, Vec<String>)>,
            |_state, _lhs, _rhs, _var| None::<(SolutionSet, Vec<String>)>,
        );

        assert!(steps.is_empty());
        match set {
            SolutionSet::Discrete(solutions) => assert_eq!(solutions, vec![two]),
            other => panic!("expected discrete solution set, got {other:?}"),
        }
    }

    #[test]
    fn execute_negated_lhs_entry_with_default_plan_and_merge_with_existing_steps_with_state_merges_steps(
    ) {
        let mut state = IsolatedState::default();
        let x = state.context.var("x");
        let y = state.context.num(2);

        let solved =
            execute_negated_lhs_entry_with_default_plan_and_merge_with_existing_steps_with_state(
                &mut state,
                x,
                y,
                RelOp::Eq,
                "x",
                true,
                vec!["existing".to_string()],
                |state| &mut state.context,
                |_state, _equation, _var| {
                    Ok::<(SolutionSet, Vec<String>), &'static str>((
                        SolutionSet::AllReals,
                        vec!["subsolve".to_string()],
                    ))
                },
                |item| item.description,
            )
            .expect("negated entry should solve");

        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert!(solved.1.contains(&"subsolve".to_string()));
        assert_eq!(solved.1.last(), Some(&"existing".to_string()));
    }
}
