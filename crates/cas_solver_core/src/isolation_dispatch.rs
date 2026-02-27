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
        execute_isolation_dispatch_route_with_state,
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
