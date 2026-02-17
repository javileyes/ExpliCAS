//! Ground expression evaluation for predicate proofs.
//!
//! When `prove_nonzero` encounters a variable-free ("ground") expression it
//! can't structurally analyse (e.g. `cos(π/3)` or `2^(1/2) + 3^(1/2)`),
//! this module provides a fallback: clone the context into a lightweight
//! simplifier, simplify with `DomainMode::Generic`, and inspect the result.
//!
//! # Re-entrancy guard
//!
//! Because `simplify` itself may call `prove_nonzero` (via `has_undefined_risk`),
//! a `thread_local!` counter prevents infinite recursion.  When the counter is
//! non-zero, the fallback returns `None` immediately.

use cas_ast::{Context, Expr, ExprId};

use crate::domain::Proof;

// Thread-local re-entrancy guard.
//
// Incremented on entry to `try_ground_nonzero`, decremented on exit.
// If > 0 when entering, the fallback is skipped.
thread_local! {
    static IN_GROUND_EVAL: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// RAII guard that decrements the counter on drop, even on panic.
pub(crate) struct GroundEvalGuard;

impl GroundEvalGuard {
    pub(crate) fn enter() -> Option<Self> {
        IN_GROUND_EVAL.with(|c| {
            let current = c.get();
            if current > 0 {
                return None; // Already inside a ground eval — prevent re-entrancy
            }
            c.set(current + 1);
            Some(GroundEvalGuard)
        })
    }
}

impl Drop for GroundEvalGuard {
    fn drop(&mut self) {
        IN_GROUND_EVAL.with(|c| {
            c.set(c.get().saturating_sub(1));
        });
    }
}

/// Attempt to prove non-zero by simplifying a ground expression.
///
/// Returns `Some(Proven)` if the expression simplifies to a non-zero `Number`,
/// `Some(Disproven)` if it simplifies to 0, and `None` if it can't determine
/// (expression doesn't fully reduce, or re-entrancy guard fires).
///
/// # Safety invariants
/// - Caller MUST ensure `!contains_variable(ctx, expr)`.
/// - Re-entrancy guard prevents `prove_nonzero → simplify → prove_nonzero` cycles.
pub(crate) fn try_ground_nonzero(ctx: &Context, expr: ExprId) -> Option<Proof> {
    // Re-entrancy guard: if already inside a ground eval, bail
    let _guard = GroundEvalGuard::enter()?;

    // Clone context into a temporary simplifier with minimal options
    let mut simplifier = crate::engine::Simplifier::with_context(ctx.clone());
    simplifier.set_collect_steps(false);

    // Use Generic mode (allows cancellations without domain assumptions on constants)
    // with tight budgets to keep this cheap
    let opts = crate::phase::SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig {
                domain_mode: crate::domain::DomainMode::Generic,
                ..Default::default()
            },
            ..Default::default()
        },
        budgets: crate::phase::PhaseBudgets {
            core_iters: 4,
            transform_iters: 2,
            rationalize_iters: 0,
            post_iters: 2,
            max_total_rewrites: 50,
        },
        ..Default::default()
    };

    let (result, _, _) = simplifier.simplify_with_stats(expr, opts);

    match simplifier.context.get(result) {
        Expr::Number(n) => {
            if num_traits::Zero::is_zero(n) {
                Some(Proof::Disproven) // expression = 0
            } else {
                Some(Proof::Proven) // expression ≠ 0
            }
        }
        _ => {
            // Didn't reduce to a number — try structural analysis on the
            // simplified form.  E.g. sqrt(2) → Pow(2, 1/2) which the
            // Pow arm in prove_nonzero_depth can handle (positive base).
            // The re-entrancy guard is still held, so any ground eval
            // attempt inside this call will short-circuit to None.
            let proof = super::predicates::prove_nonzero_depth(
                &simplifier.context,
                result,
                8, // shallow depth budget
            );
            if proof == Proof::Proven || proof == Proof::Disproven {
                Some(proof)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Constant;

    fn make_ctx() -> Context {
        Context::new()
    }

    #[test]
    fn test_ground_nonzero_constant_sqrt2() {
        // sqrt(2) = 2^(1/2) is non-zero
        let mut ctx = make_ctx();
        let two = ctx.num(2);
        let half = ctx.rational(1, 2);
        let sqrt2 = ctx.add(Expr::Pow(two, half));

        let result = try_ground_nonzero(&ctx, sqrt2);
        assert_eq!(result, Some(Proof::Proven));
    }

    #[test]
    fn test_ground_nonzero_zero() {
        // 0 should be disproven
        let mut ctx = make_ctx();
        let zero = ctx.num(0);

        let result = try_ground_nonzero(&ctx, zero);
        assert_eq!(result, Some(Proof::Disproven));
    }

    #[test]
    fn test_ground_nonzero_pi() {
        // π is non-zero
        let mut ctx = make_ctx();
        let pi = ctx.add(Expr::Constant(Constant::Pi));

        let result = try_ground_nonzero(&ctx, pi);
        assert_eq!(result, Some(Proof::Proven));
    }

    #[test]
    fn test_ground_nonzero_sum_of_constants() {
        // 1 + 2 = 3 is non-zero
        let mut ctx = make_ctx();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let sum = ctx.add(Expr::Add(one, two));

        let result = try_ground_nonzero(&ctx, sum);
        assert_eq!(result, Some(Proof::Proven));
    }

    #[test]
    fn test_ground_nonzero_diff_equal() {
        // 5 - 5 = 0 should be disproven
        let mut ctx = make_ctx();
        let five_a = ctx.num(5);
        let five_b = ctx.num(5);
        let diff = ctx.add(Expr::Sub(five_a, five_b));

        let result = try_ground_nonzero(&ctx, diff);
        assert_eq!(result, Some(Proof::Disproven));
    }

    #[test]
    fn test_reentrancy_guard_prevents_recursion() {
        // Manually hold the guard and verify that enter() returns None
        let _guard = GroundEvalGuard::enter();
        assert!(_guard.is_some(), "First enter should succeed");

        let second = GroundEvalGuard::enter();
        assert!(second.is_none(), "Second enter should be blocked");
    }
}
