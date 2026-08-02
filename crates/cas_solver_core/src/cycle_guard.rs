use cas_ast::{Context, ExprId};
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    /// Fingerprints seen in the current solve call stack.
    static SOLVE_SEEN: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
}

/// RAII guard that removes a fingerprint from the cycle set on drop.
pub struct CycleGuard {
    fp: u64,
}

impl Drop for CycleGuard {
    fn drop(&mut self) {
        SOLVE_SEEN.with(|s| {
            s.borrow_mut().remove(&self.fp);
        });
    }
}

/// Try entering a fingerprinted solve state.
///
/// Returns `None` when the fingerprint is already active in the current
/// call stack (cycle detected). Returns a guard otherwise; dropping the guard
/// removes the fingerprint.
pub(crate) fn try_enter(fp: u64) -> Option<CycleGuard> {
    let inserted = SOLVE_SEEN.with(|s| s.borrow_mut().insert(fp));
    if inserted {
        Some(CycleGuard { fp })
    } else {
        None
    }
}

/// Try entering solve state for a full equation fingerprint.
///
/// Returns `None` when the same `(var, lhs, rhs, op)` shape is already
/// active in the current solve call stack.
pub(crate) fn try_enter_equation_fingerprint(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    op: &cas_ast::RelOp,
) -> Option<CycleGuard> {
    let fp = crate::fingerprint::equation_fingerprint(ctx, lhs, rhs, var, op);
    try_enter(fp)
}

/// True when the exact `(var, lhs, rhs, op)` fingerprint is already active
/// in the current solve stack — re-entering it is guaranteed to be reported
/// as a cycle. A rewrite that would DELEGATE to an identical equation (e.g.
/// the `!=` zero-product reorientation when the equation is already in
/// normal orientation) must check this first and keep its fallback path
/// instead: self-delegation can never make progress.
pub(crate) fn equation_fingerprint_active(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    op: &cas_ast::RelOp,
) -> bool {
    let fp = crate::fingerprint::equation_fingerprint(ctx, lhs, rhs, var, op);
    SOLVE_SEEN.with(|s| s.borrow().contains(&fp))
}

#[cfg(test)]
mod tests {
    use super::{try_enter, try_enter_equation_fingerprint};
    use cas_ast::{Context, Expr};

    #[test]
    fn detects_reentry_until_guard_drops() {
        let guard = try_enter(123).expect("first insert should succeed");
        assert!(try_enter(123).is_none(), "second insert must detect cycle");
        drop(guard);
        assert!(
            try_enter(123).is_some(),
            "insert should succeed again after guard drop"
        );
    }

    #[test]
    fn detects_equation_reentry_until_guard_drops() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = one;

        let guard = try_enter_equation_fingerprint(&ctx, lhs, rhs, "x", &cas_ast::RelOp::Eq)
            .expect("first equation insert should succeed");
        assert!(
            try_enter_equation_fingerprint(&ctx, lhs, rhs, "x", &cas_ast::RelOp::Eq).is_none(),
            "second equivalent equation insert must detect cycle"
        );
        drop(guard);
        assert!(
            try_enter_equation_fingerprint(&ctx, lhs, rhs, "x", &cas_ast::RelOp::Eq).is_some(),
            "equation insert should succeed again after guard drop"
        );
    }

    #[test]
    fn associated_equation_with_different_op_is_not_a_cycle() {
        // The `!=` owners solve the associated `= 0` equation with the SAME
        // (lhs, rhs, var): that delegation must enter cleanly while the
        // outer `!=` fingerprint is still active.
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.num(0);

        let outer = try_enter_equation_fingerprint(&ctx, lhs, rhs, "x", &cas_ast::RelOp::Neq)
            .expect("outer != insert should succeed");
        let inner = try_enter_equation_fingerprint(&ctx, lhs, rhs, "x", &cas_ast::RelOp::Eq);
        assert!(
            inner.is_some(),
            "associated = equation must not be reported as a cycle"
        );
        drop(inner);
        drop(outer);
    }
}
