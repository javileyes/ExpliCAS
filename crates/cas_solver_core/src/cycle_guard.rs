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
pub fn try_enter(fp: u64) -> Option<CycleGuard> {
    let inserted = SOLVE_SEEN.with(|s| s.borrow_mut().insert(fp));
    if inserted {
        Some(CycleGuard { fp })
    } else {
        None
    }
}

/// Try entering solve state for a full equation fingerprint.
///
/// Returns `None` when the same `(var, lhs, rhs)` shape is already active
/// in the current solve call stack.
pub fn try_enter_equation_fingerprint(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<CycleGuard> {
    let fp = crate::fingerprint::equation_fingerprint(ctx, lhs, rhs, var);
    try_enter(fp)
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

        let guard = try_enter_equation_fingerprint(&ctx, lhs, rhs, "x")
            .expect("first equation insert should succeed");
        assert!(
            try_enter_equation_fingerprint(&ctx, lhs, rhs, "x").is_none(),
            "second equivalent equation insert must detect cycle"
        );
        drop(guard);
        assert!(
            try_enter_equation_fingerprint(&ctx, lhs, rhs, "x").is_some(),
            "equation insert should succeed again after guard drop"
        );
    }
}
