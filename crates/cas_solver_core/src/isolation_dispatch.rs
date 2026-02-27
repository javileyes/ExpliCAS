//! Isolation dispatcher helpers shared by engine-side solver.
//!
//! These helpers keep LHS shape-routing logic in `cas_solver_core`, while
//! callers provide stateful handlers for each isolation branch.

use cas_ast::symbol::SymbolId;
use cas_ast::{Context, Expr, ExprId};

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

#[cfg(test)]
mod tests {
    use super::{
        derive_isolation_dispatch_route, execute_isolation_dispatch_route_with_state,
        IsolationDispatchRoute,
    };
    use cas_ast::{Context, Expr};

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
}
