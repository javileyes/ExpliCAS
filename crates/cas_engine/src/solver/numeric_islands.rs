//! Numeric island folding for verification.
//!
//! When a Strict-mode residual still contains variables but also has variable-free
//! subtrees ("numeric islands") that use different representational forms of the
//! same constant (e.g. `2^(-1/2)` vs `sqrt(2)/2`), those islands can block
//! cancellation because Strict mode won't simplify them (they involve division or
//! function calls where `prove_nonzero` returns `Unknown`).
//!
//! **`fold_numeric_islands`** identifies such ground subtrees and canonicalizes them
//! by simplifying each with `Generic` mode in a temporary simplifier. Only "benign"
//! results (numbers, constants, or strictly smaller ground expressions) are accepted.
//!
//! ## Safety invariants
//!
//! - Only fires on **variable-free** subtrees — can't erase parametric conditions.
//! - Uses the `GroundEvalGuard` re-entrancy guard (shared with `ground_eval.rs`)
//!   to prevent `simplify → prove_nonzero → ground_eval → simplify` loops.
//! - Limits folding to islands with `≤ MAX_ISLAND_NODES` deduped nodes and
//!   `≤ MAX_ISLAND_DEPTH` depth to avoid expensive simplifications.
//! - Rejects folds that produce undefined results (`Div(_, 0)` etc.) or that
//!   increase expression size.

use cas_ast::{Context, ExprId};
use cas_solver_core::numeric_islands::{
    fold_numeric_islands_with, is_benign_fold_result, transplant_expr,
};

use crate::helpers::ground_eval::GroundEvalGuard;

/// Maximum dedup node count for an island to be eligible for folding.
const MAX_ISLAND_NODES: usize = 80;

/// Maximum depth for an island to be eligible for folding.
const MAX_ISLAND_DEPTH: usize = 4;

/// Fold numeric islands in an expression tree.
///
/// Performs a bottom-up (post-order) traversal. For each subtree that is:
/// - variable-free (ground)
/// - not a simple leaf (Number, Constant)
/// - within size/depth limits
///
/// ...creates a temporary simplifier, simplifies with Generic mode, and replaces
/// the subtree if the result is "benign" (simpler or a leaf).
///
/// Returns the new root ExprId (may be unchanged if no folding occurred).
pub(crate) fn fold_numeric_islands(ctx: &mut Context, root: ExprId) -> ExprId {
    // Acquire re-entrancy guard — if already inside ground_eval, skip entirely
    let _guard = match GroundEvalGuard::enter() {
        Some(g) => g,
        None => return root,
    };

    fold_numeric_islands_with(
        ctx,
        root,
        MAX_ISLAND_NODES,
        MAX_ISLAND_DEPTH,
        cas_solver_core::verify_stats::record_skipped_limits,
        fold_one_island_candidate,
    )
}

/// Fold one pre-checked eligible island.
fn fold_one_island_candidate(ctx: &mut Context, id: ExprId, node_count: usize) -> ExprId {
    // Try simplifying with Generic mode in a temporary simplifier
    let mut tmp = crate::engine::Simplifier::with_context(ctx.clone());
    tmp.set_collect_steps(false);

    let opts = crate::phase::SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig {
                domain_mode: crate::domain::DomainMode::Generic,
                value_domain: crate::semantics::ValueDomain::RealOnly,
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

    let (result, _, _) = tmp.simplify_with_stats(id, opts);

    // Acceptance filter: only accept if result is "benign"
    if !is_benign_fold_result(&tmp.context, result, node_count) {
        return id;
    }

    // Transplant the result back into the original context.
    // For simple leaves, we can create directly. For more complex ground
    // results that are smaller, we need to deep-copy the subtree.
    transplant_expr(&tmp.context, result, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_math::expr_predicates::contains_variable;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_ctx() -> Context {
        Context::new()
    }

    /// Build `sqrt(expr)` as `Pow(expr, 1/2)`.
    fn sqrt(ctx: &mut Context, expr: ExprId) -> ExprId {
        let half = ctx.rational(1, 2);
        ctx.add(Expr::Pow(expr, half))
    }

    // -----------------------------------------------------------------------
    // Contract tests — fold_numeric_islands behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn fold_ground_island_to_number() {
        // sqrt(4) → should fold to 2
        let mut ctx = make_ctx();
        let four = ctx.num(4);
        let sqrt4 = sqrt(&mut ctx, four);

        let folded = fold_numeric_islands(&mut ctx, sqrt4);

        // After folding, should become Number(2)
        match ctx.get(folded) {
            Expr::Number(n) => {
                assert_eq!(*n, num_rational::BigRational::from_integer(2.into()));
            }
            other => panic!("expected Number(2), got {:?}", other),
        }
    }

    #[test]
    fn fold_sqrt2_div_sqrt2_to_one() {
        // sqrt(2) / sqrt(2) → should fold to 1
        let mut ctx = make_ctx();
        let two = ctx.num(2);
        let s2a = sqrt(&mut ctx, two);
        let s2b = sqrt(&mut ctx, two); // same node due to dedup
        let div = ctx.add(Expr::Div(s2a, s2b));

        let folded = fold_numeric_islands(&mut ctx, div);

        match ctx.get(folded) {
            Expr::Number(n) => {
                assert!(
                    n == &num_rational::BigRational::from_integer(1.into()),
                    "expected 1, got {}",
                    n
                );
            }
            other => panic!("expected Number(1), got {:?}", other),
        }
    }

    #[test]
    fn fold_does_not_touch_variables() {
        // x + y should be unchanged (no numeric islands to fold)
        let mut ctx = make_ctx();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x, y));

        let folded = fold_numeric_islands(&mut ctx, sum);
        assert_eq!(folded, sum, "Pure variable expression should not change");
    }

    #[test]
    fn fold_does_not_touch_simple_numbers() {
        // Number(42) should be unchanged (already a number)
        let mut ctx = make_ctx();
        let n = ctx.num(42);

        let folded = fold_numeric_islands(&mut ctx, n);
        assert_eq!(folded, n, "Simple number should not change");
    }

    #[test]
    fn fold_mixed_var_with_ground_island() {
        // x + sqrt(4) → x + 2  (folds the ground island sqrt(4))
        let mut ctx = make_ctx();
        let x = ctx.var("x");
        let four = ctx.num(4);
        let sqrt4 = sqrt(&mut ctx, four);
        let expr = ctx.add(Expr::Add(x, sqrt4));

        let folded = fold_numeric_islands(&mut ctx, expr);

        // The result should differ from the original (sqrt(4) folded to 2)
        // We verify the folded tree no longer contains Pow (the sqrt)
        fn has_pow(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Pow(_, _) => true,
                Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                    has_pow(ctx, *a) || has_pow(ctx, *b)
                }
                Expr::Neg(e) => has_pow(ctx, *e),
                _ => false,
            }
        }
        assert!(
            !has_pow(&ctx, folded),
            "Folded expression should not contain Pow (sqrt was folded)"
        );
    }

    #[test]
    fn fold_preserves_guard_against_oversized_islands() {
        // Build a ground expression that exceeds MAX_ISLAND_NODES.
        // Key invariant: folding should not panic or produce incorrect results.
        let mut ctx = make_ctx();
        let mut expr = ctx.num(1);
        for i in 2..=(MAX_ISLAND_NODES as i64 + 10) {
            let n = ctx.num(i);
            expr = ctx.add(Expr::Add(expr, n));
        }

        let folded = fold_numeric_islands(&mut ctx, expr);

        // Verify no panic and result is still ground
        assert!(
            !contains_variable(&ctx, folded),
            "Folded result should still be ground"
        );
    }

    // -----------------------------------------------------------------------
    // Micro-benchmark tests — Phase 1.5 value demonstration
    // -----------------------------------------------------------------------

    /// Simplify with Strict-only and return whether it verified to 0.
    fn verify_strict_only(simplifier: &mut crate::engine::Simplifier, expr: ExprId) -> bool {
        let strict_opts = crate::SimplifyOptions {
            shared: crate::phase::SharedSemanticConfig {
                semantics: crate::semantics::EvalConfig {
                    domain_mode: crate::domain::DomainMode::Strict,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let (result, _, _) = simplifier.simplify_with_stats(expr, strict_opts);
        matches!(simplifier.context.get(result), Expr::Number(n) if num_traits::Zero::is_zero(n))
    }

    /// Simplify with Strict + Phase 1.5 (island fold then re-Strict).
    fn verify_strict_with_island_fold(
        simplifier: &mut crate::engine::Simplifier,
        expr: ExprId,
    ) -> bool {
        let strict_opts = crate::SimplifyOptions {
            shared: crate::phase::SharedSemanticConfig {
                semantics: crate::semantics::EvalConfig {
                    domain_mode: crate::domain::DomainMode::Strict,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };

        // Phase 1: Strict
        let (strict_result, _, _) = simplifier.simplify_with_stats(expr, strict_opts.clone());
        if matches!(simplifier.context.get(strict_result), Expr::Number(n) if num_traits::Zero::is_zero(n))
        {
            return true;
        }

        // Phase 1.5: fold numeric islands and re-Strict
        if contains_variable(&simplifier.context, strict_result) {
            let folded = fold_numeric_islands(&mut simplifier.context, strict_result);
            if folded != strict_result {
                let (folded_result, _, _) = simplifier.simplify_with_stats(folded, strict_opts);
                if matches!(simplifier.context.get(folded_result), Expr::Number(n) if num_traits::Zero::is_zero(n))
                {
                    return true;
                }
            }
        }

        false
    }

    #[test]
    fn phase15_enables_verification_x_sqrt2_div_sqrt2() {
        // Expression: x - x * sqrt(2) / sqrt(2)
        // sqrt(2)/sqrt(2) blocks Strict because prove_nonzero can't handle sqrt(2).
        // Phase 1.5 folds sqrt(2)/sqrt(2) → 1, then Strict sees x - x*1 = 0.
        let mut s = crate::engine::Simplifier::with_default_rules();
        let x = s.context.var("x");
        let two = s.context.num(2);
        let s2a = sqrt(&mut s.context, two);
        let s2b = sqrt(&mut s.context, two);

        // Build: x * sqrt(2) / sqrt(2)
        let x_times_s2 = s.context.add(Expr::Mul(x, s2a));
        let frac = s.context.add(Expr::Div(x_times_s2, s2b));
        // diff = x - frac
        let diff = s.context.add(Expr::Sub(x, frac));

        let strict_only = verify_strict_only(&mut s, diff);

        let mut s2 = crate::engine::Simplifier::with_default_rules();
        let x2 = s2.context.var("x");
        let two2 = s2.context.num(2);
        let s2a2 = sqrt(&mut s2.context, two2);
        let s2b2 = sqrt(&mut s2.context, two2);
        let x_times_s2_2 = s2.context.add(Expr::Mul(x2, s2a2));
        let frac2 = s2.context.add(Expr::Div(x_times_s2_2, s2b2));
        let diff2 = s2.context.add(Expr::Sub(x2, frac2));

        let with_phase15 = verify_strict_with_island_fold(&mut s2, diff2);

        // Phase 1.5 should verify this
        assert!(
            with_phase15,
            "Phase 1.5 should verify x - x*sqrt(2)/sqrt(2) = 0"
        );

        eprintln!(
            "  strict_only={}, with_phase15={}",
            strict_only, with_phase15
        );
    }

    #[test]
    fn phase15_enables_verification_x_sqrt2_squared() {
        // Expression: x * sqrt(2) * sqrt(2) - 2 * x
        // sqrt(2)*sqrt(2) should be 2, making the expression 0.
        let mut s = crate::engine::Simplifier::with_default_rules();
        let x = s.context.var("x");
        let two = s.context.num(2);
        let s2a = sqrt(&mut s.context, two);
        let s2b = sqrt(&mut s.context, two);

        let x_s2 = s.context.add(Expr::Mul(x, s2a));
        let x_s2_s2 = s.context.add(Expr::Mul(x_s2, s2b));
        let two_x = s.context.add(Expr::Mul(two, x));
        let diff = s.context.add(Expr::Sub(x_s2_s2, two_x));

        let with_phase15 = verify_strict_with_island_fold(&mut s, diff);

        assert!(
            with_phase15,
            "Phase 1.5 should verify x*sqrt(2)*sqrt(2) - 2*x = 0"
        );
    }
}
