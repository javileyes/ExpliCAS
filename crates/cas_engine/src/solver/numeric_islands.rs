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

use std::collections::{HashMap, HashSet};

use cas_ast::{Context, Expr, ExprId};

use crate::helpers::ground_eval::GroundEvalGuard;
use crate::implicit_domain::contains_variable;

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

    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    fold_recursive(ctx, root, &mut memo)
}

/// Recursive post-order fold with memoization.
///
/// Works bottom-up: first fold children, then check if the resulting node
/// is a foldable ground island.
fn fold_recursive(ctx: &mut Context, id: ExprId, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
    // Memoize: if we've already folded this node, return cached result
    if let Some(&cached) = memo.get(&id) {
        return cached;
    }

    // Leaves never need folding
    let node = ctx.get(id).clone();
    let result = match node {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => id,

        Expr::Add(a, b) => {
            let fa = fold_recursive(ctx, a, memo);
            let fb = fold_recursive(ctx, b, memo);
            let new_id = if fa == a && fb == b {
                id
            } else {
                ctx.add(Expr::Add(fa, fb))
            };
            try_fold_island(ctx, new_id)
        }
        Expr::Sub(a, b) => {
            let fa = fold_recursive(ctx, a, memo);
            let fb = fold_recursive(ctx, b, memo);
            let new_id = if fa == a && fb == b {
                id
            } else {
                ctx.add(Expr::Sub(fa, fb))
            };
            try_fold_island(ctx, new_id)
        }
        Expr::Mul(a, b) => {
            let fa = fold_recursive(ctx, a, memo);
            let fb = fold_recursive(ctx, b, memo);
            let new_id = if fa == a && fb == b {
                id
            } else {
                ctx.add(Expr::Mul(fa, fb))
            };
            try_fold_island(ctx, new_id)
        }
        Expr::Div(a, b) => {
            let fa = fold_recursive(ctx, a, memo);
            let fb = fold_recursive(ctx, b, memo);
            let new_id = if fa == a && fb == b {
                id
            } else {
                ctx.add(Expr::Div(fa, fb))
            };
            try_fold_island(ctx, new_id)
        }
        Expr::Pow(a, b) => {
            let fa = fold_recursive(ctx, a, memo);
            let fb = fold_recursive(ctx, b, memo);
            let new_id = if fa == a && fb == b {
                id
            } else {
                ctx.add(Expr::Pow(fa, fb))
            };
            try_fold_island(ctx, new_id)
        }
        Expr::Neg(a) => {
            let fa = fold_recursive(ctx, a, memo);
            let new_id = if fa == a { id } else { ctx.add(Expr::Neg(fa)) };
            try_fold_island(ctx, new_id)
        }
        Expr::Function(name, ref args) => {
            let mut changed = false;
            let folded_args: Vec<ExprId> = args
                .iter()
                .map(|&arg| {
                    let fa = fold_recursive(ctx, arg, memo);
                    if fa != arg {
                        changed = true;
                    }
                    fa
                })
                .collect();
            let new_id = if changed {
                ctx.add(Expr::Function(name, folded_args))
            } else {
                id
            };
            try_fold_island(ctx, new_id)
        }
        Expr::Hold(inner) => {
            // Don't fold inside Hold — it's user-protected
            let fi = fold_recursive(ctx, inner, memo);
            if fi == inner {
                id
            } else {
                ctx.add(Expr::Hold(fi))
            }
        }
        Expr::Matrix {
            rows,
            cols,
            ref data,
        } => {
            let mut changed = false;
            let folded_data: Vec<ExprId> = data
                .iter()
                .map(|&elem| {
                    let fe = fold_recursive(ctx, elem, memo);
                    if fe != elem {
                        changed = true;
                    }
                    fe
                })
                .collect();
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: folded_data,
                })
            } else {
                id
            }
        }
    };

    memo.insert(id, result);
    result
}

/// Attempt to fold a single node if it's a ground island within limits.
///
/// Returns the folded ExprId if successful, otherwise the original.
fn try_fold_island(ctx: &mut Context, id: ExprId) -> ExprId {
    // Skip if this subtree contains variables — not a ground island
    if contains_variable(ctx, id) {
        return id;
    }

    // Skip if already a simple leaf
    match ctx.get(id) {
        Expr::Number(_) | Expr::Constant(_) => return id,
        _ => {}
    }

    // Check size/depth limits (dedup count)
    let (node_count, depth) = count_nodes_dedup(ctx, id);
    if node_count > MAX_ISLAND_NODES || depth > MAX_ISLAND_DEPTH {
        super::verify_stats::record_skipped_limits();
        return id;
    }

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
    let result_node = tmp.context.get(result).clone();

    // Acceptance filter: only accept if result is "benign"
    if !is_benign_result(&tmp.context, result, node_count) {
        return id;
    }

    // Transplant the result back into the original context.
    // For simple leaves, we can create directly. For more complex ground
    // results that are smaller, we need to deep-copy the subtree.
    transplant_expr(&tmp.context, result, &result_node, ctx)
}

/// Check if a simplification result is "benign" (safe to accept).
///
/// Accepts:
/// - `Number(_)` (ideal — fully evaluated)
/// - `Constant(_)` (already canonical)
/// - Any ground result that is strictly smaller (fewer dedup nodes)
///   AND does not contain `Div(_, 0)` or undefined forms
fn is_benign_result(ctx: &Context, result: ExprId, original_node_count: usize) -> bool {
    match ctx.get(result) {
        // Fully evaluated — always accept
        Expr::Number(_) | Expr::Constant(_) => true,

        // For non-leaf results, accept only if strictly smaller and safe
        _ => {
            let (result_count, _) = count_nodes_dedup(ctx, result);
            if result_count >= original_node_count {
                return false; // Didn't shrink — not helpful
            }
            // Reject results with division by zero
            !has_zero_denominator(ctx, result)
        }
    }
}

/// Check if expression contains any Div(_, 0) nodes.
fn has_zero_denominator(ctx: &Context, id: ExprId) -> bool {
    let mut stack = vec![id];
    while let Some(node_id) = stack.pop() {
        match ctx.get(node_id) {
            Expr::Div(_, den) => {
                if matches!(ctx.get(*den), Expr::Number(n) if num_traits::Zero::is_zero(n)) {
                    return true;
                }
                stack.push(*den);
                // Also check numerator
                if let Expr::Div(num, _) = ctx.get(node_id) {
                    stack.push(*num);
                }
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            _ => {}
        }
    }
    false
}

/// Count unique nodes and max depth (dedup by ExprId).
fn count_nodes_dedup(ctx: &Context, root: ExprId) -> (usize, usize) {
    let mut seen = HashSet::new();
    let mut max_depth = 0;
    let mut stack: Vec<(ExprId, usize)> = vec![(root, 0)];

    while let Some((id, depth)) = stack.pop() {
        if !seen.insert(id) {
            continue; // Already visited (DAG sharing)
        }
        max_depth = max_depth.max(depth);

        let child_depth = depth + 1;
        match ctx.get(id) {
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push((*a, child_depth));
                stack.push((*b, child_depth));
            }
            Expr::Neg(e) | Expr::Hold(e) => stack.push((*e, child_depth)),
            Expr::Function(_, args) => {
                for &arg in args {
                    stack.push((arg, child_depth));
                }
            }
            Expr::Matrix { data, .. } => {
                for &elem in data {
                    stack.push((elem, child_depth));
                }
            }
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    (seen.len(), max_depth)
}

/// Deep-copy an expression subtree from `src` context into `dst` context.
///
/// For simple leaves (Number, Constant), creates them directly in `dst`.
/// For composite expressions, recursively transplants children first.
fn transplant_expr(src: &Context, _id: ExprId, node: &Expr, dst: &mut Context) -> ExprId {
    match node {
        Expr::Number(n) => dst.add(Expr::Number(n.clone())),
        Expr::Constant(c) => dst.add(Expr::Constant(c.clone())),
        Expr::Variable(sym) => {
            // Preserve variable name
            let name = src.sym_name(*sym);
            dst.var(name)
        }
        Expr::SessionRef(r) => dst.add(Expr::SessionRef(*r)),
        Expr::Add(a, b) => {
            let ta = transplant_expr(src, *a, src.get(*a), dst);
            let tb = transplant_expr(src, *b, src.get(*b), dst);
            dst.add(Expr::Add(ta, tb))
        }
        Expr::Sub(a, b) => {
            let ta = transplant_expr(src, *a, src.get(*a), dst);
            let tb = transplant_expr(src, *b, src.get(*b), dst);
            dst.add(Expr::Sub(ta, tb))
        }
        Expr::Mul(a, b) => {
            let ta = transplant_expr(src, *a, src.get(*a), dst);
            let tb = transplant_expr(src, *b, src.get(*b), dst);
            dst.add(Expr::Mul(ta, tb))
        }
        Expr::Div(a, b) => {
            let ta = transplant_expr(src, *a, src.get(*a), dst);
            let tb = transplant_expr(src, *b, src.get(*b), dst);
            dst.add(Expr::Div(ta, tb))
        }
        Expr::Pow(a, b) => {
            let ta = transplant_expr(src, *a, src.get(*a), dst);
            let tb = transplant_expr(src, *b, src.get(*b), dst);
            dst.add(Expr::Pow(ta, tb))
        }
        Expr::Neg(inner) => {
            let ti = transplant_expr(src, *inner, src.get(*inner), dst);
            dst.add(Expr::Neg(ti))
        }
        Expr::Function(name, args) => {
            let targs: Vec<ExprId> = args
                .iter()
                .map(|&arg| transplant_expr(src, arg, src.get(arg), dst))
                .collect();
            dst.add(Expr::Function(*name, targs))
        }
        Expr::Hold(inner) => {
            let ti = transplant_expr(src, *inner, src.get(*inner), dst);
            dst.add(Expr::Hold(ti))
        }
        Expr::Matrix { rows, cols, data } => {
            let tdata: Vec<ExprId> = data
                .iter()
                .map(|&elem| transplant_expr(src, elem, src.get(elem), dst))
                .collect();
            dst.add(Expr::Matrix {
                rows: *rows,
                cols: *cols,
                data: tdata,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

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
