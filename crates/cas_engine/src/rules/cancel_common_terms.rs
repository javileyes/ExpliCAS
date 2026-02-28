//! Equation-level additive term cancellation utilities.
//!
//! Provides `cancel_common_additive_terms()`, which compares terms from
//! two expression trees (LHS and RHS of an equation) and cancels matching
//! pairs using structural comparison.
//!
//! # Why this is NOT a simplifier rule
//!
//! The simplifier's `CanonicalizeNegationRule` converts `Sub(a, b)` to
//! `Add(a, Neg(b))` very early in the pipeline. A rule targeting `Sub`
//! nodes would never fire because the Sub is already gone.
//!
//! More fundamentally, equation-level cancellation is a **relational
//! operation** between two expression trees (LHS vs RHS of `lhs = rhs`).
//! A simplifier rule only sees a single expression node. These are
//! different abstractions and belong in different layers.
//!
//! # Preconditions
//!
//! For maximum effectiveness, both `lhs` and `rhs` should be in
//! canonical form before calling:
//! - Rational exponents: `Div(Number(p), Number(q))` → `Number(p/q)`
//!   (via CanonicalizeRationalDivRule)
//! - Flattened Add/Mul chains with stable ordering
//!   (via standard canonicalization rules)
//!
//! # Guarantees
//!
//! - **Strictly reductive**: total term count always decreases
//! - **Sound**: uses `compare_expr` for structural equality (no numeric
//!   approximation)
//! - **Deterministic**: order of cancellation is fixed (rhs terms scanned
//!   in order, matched against lhs terms in order)

use crate::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use cas_math::cancel_expand_support::try_expand_for_cancel_with;
use cas_math::cancel_normalization_support::{normalize_for_cancel, OriginSafety};
use cas_math::cancel_support::{
    collect_additive_terms_signed as collect_additive_terms,
    rebuild_from_signed_terms as rebuild_from_terms,
    structural_expr_fingerprint as term_fingerprint,
};
use std::cmp::Ordering;
use std::collections::HashSet;

/// Result of equation-level additive term cancellation.
#[allow(dead_code)]
pub struct CancelResult {
    /// New LHS with cancelled terms removed.
    pub new_lhs: ExprId,
    /// New RHS with cancelled terms removed.
    pub new_rhs: ExprId,
    /// Number of term pairs that were cancelled.
    pub cancelled_count: usize,
}

/// Cancel common additive terms between two expression trees.
///
/// Flattens both sides into multisets of `(term, is_positive)`, matches
/// pairs by structural equality (same term, same sign polarity), and
/// rebuilds both sides without the matched pairs.
///
/// Returns `None` if no terms could be cancelled.
///
/// # Example
/// ```text
/// lhs = Add(x², Add(x^(5/6), 1))    rhs = x^(5/6)
/// → lhs_terms = [(x², +), (x^(5/6), +), (1, +)]
/// → rhs_terms = [(x^(5/6), +)]
/// → cancel x^(5/6) pair
/// → new_lhs = Add(x², 1), new_rhs = 0, cancelled = 1
/// ```
pub fn cancel_common_additive_terms(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_additive_terms(ctx, lhs, true, &mut lhs_terms);
    collect_additive_terms(ctx, rhs, true, &mut rhs_terms);

    let mut lhs_used = vec![false; lhs_terms.len()];
    let mut rhs_used = vec![false; rhs_terms.len()];
    let mut cancelled = 0;

    for (ri, (rt, rp)) in rhs_terms.iter().enumerate() {
        if rhs_used[ri] {
            continue;
        }
        for (li, (lt, lp)) in lhs_terms.iter().enumerate() {
            if lhs_used[li] {
                continue;
            }
            if lp == rp && compare_expr(ctx, *lt, *rt) == Ordering::Equal {
                lhs_used[li] = true;
                rhs_used[ri] = true;
                cancelled += 1;
                break;
            }
        }
    }

    if cancelled == 0 {
        return None;
    }

    let new_lhs_terms: Vec<_> = lhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !lhs_used[*i])
        .map(|(_, t)| t)
        .collect();
    let new_rhs_terms: Vec<_> = rhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !rhs_used[*i])
        .map(|(_, t)| t)
        .collect();

    Some(CancelResult {
        new_lhs: rebuild_from_terms(ctx, &new_lhs_terms),
        new_rhs: rebuild_from_terms(ctx, &new_rhs_terms),
        cancelled_count: cancelled,
    })
}

/// Semantic fallback for equation-level term cancellation.
///
/// **2-phase candidate+proof pattern for soundness:**
///
/// 1. *Candidate generation* (Generic domain): normalize both sides
///    (split fractions, distribute scalar multiplication, split log
///    products/sqrt), then simplify each term with Generic domain to
///    expose structure (allows `x/x → 1`, etc.).
///
/// 2. *Pair proof* (Strict domain): for each candidate pair, verify
///    `simplify_strict(l - r) == 0` before cancelling. This ensures
///    the cancellation doesn't depend on unproven assumptions (e.g.,
///    `x ≠ 0` for `x/x → 1`). The Generic simplification only exposes
///    candidates; Strict validates them.
///
/// # Guards
/// - ≤ `MAX_TERMS` terms per side (avoids O(n²) explosion)
/// - ≤ `MAX_NODES` nodes per term (avoids expensive simplification)
pub fn cancel_additive_terms_semantic(
    simplifier: &mut crate::Simplifier,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    use cas_ast::traversal::count_nodes_matching;
    use num_traits::Zero;

    const MAX_TERMS: usize = 12; // raised from 8: normalization may expand terms
    const MAX_NODES: usize = 200;

    // Pre-normalize: split Div(sum, D) and ln(product) into finer additive terms.
    // Track OriginSafety: terms from ln-normalizations are NeedsAnalyticConditions.
    let (norm_lhs, lhs_safety) = normalize_for_cancel(&mut simplifier.context, lhs, 0);
    let (norm_rhs, rhs_safety) = normalize_for_cancel(&mut simplifier.context, rhs, 0);

    let mut lhs_terms: Vec<(ExprId, bool, OriginSafety)> = Vec::new();
    let mut rhs_terms: Vec<(ExprId, bool, OriginSafety)> = Vec::new();
    {
        let mut raw_lhs = Vec::new();
        collect_additive_terms(&simplifier.context, norm_lhs, true, &mut raw_lhs);
        for (t, p) in raw_lhs {
            lhs_terms.push((t, p, lhs_safety));
        }
    }
    {
        let mut raw_rhs = Vec::new();
        collect_additive_terms(&simplifier.context, norm_rhs, true, &mut raw_rhs);
        for (t, p) in raw_rhs {
            rhs_terms.push((t, p, rhs_safety));
        }
    }

    // Simplify each term individually for canonical comparison.
    // Uses default domain (Generic) to allow x/x → 1 (essential for
    // Div(Mul(x, ln(..)), x) → ln(..) after fraction splitting).
    // Strict would block it — it requires proof of x ≠ 0.
    // Safe: this is purely for normalization, doesn't modify the solution set.
    // Phase 1: candidate generation — Generic allows x/x → 1, etc.
    let candidate_opts = crate::SimplifyOptions {
        collect_steps: false,
        ..Default::default()
    };
    // Phase 2: pair proof — Strict prevents domain expansion
    let strict_proof_opts = crate::SimplifyOptions {
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig::strict(),
            ..Default::default()
        },
        collect_steps: false,
        ..Default::default()
    };
    for (term, _, _) in &mut lhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }
    for (term, _, _) in &mut rhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }

    // Second normalize pass: simplification may expose new ln(product) terms
    // (e.g., Div(Mul(x, ln(a*b)), x) → ln(a*b) after the first simplify).
    // Re-normalize and re-collect if any term expanded.
    let re_normalize_terms = |ctx: &mut Context,
                              terms: &[(ExprId, bool, OriginSafety)]|
     -> Vec<(ExprId, bool, OriginSafety)> {
        let mut out = Vec::new();
        for &(t, p, s) in terms {
            let (n, ns) = normalize_for_cancel(ctx, t, 0);
            let merged = s.merge(ns);
            if n == t {
                out.push((t, p, merged));
            } else {
                let mut raw = Vec::new();
                collect_additive_terms(ctx, n, p, &mut raw);
                for (rt, rp) in raw {
                    out.push((rt, rp, merged));
                }
            }
        }
        out
    };
    let lhs_terms2 = re_normalize_terms(&mut simplifier.context, &lhs_terms);
    let rhs_terms2 = re_normalize_terms(&mut simplifier.context, &rhs_terms);
    let mut lhs_terms = lhs_terms2;
    let mut rhs_terms = rhs_terms2;

    // Re-simplify any newly created terms
    for (term, _, _) in &mut lhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }
    for (term, _, _) in &mut rhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }

    // Final re-flatten: simplification may turn a single term into Add(a, b),
    // e.g., Div(Mul(Add(ln(y),2),2), 2) → Add(ln(y), 2). Re-collect to split.
    let re_flatten = |terms: Vec<(ExprId, bool, OriginSafety)>,
                      ctx: &Context|
     -> Vec<(ExprId, bool, OriginSafety)> {
        let mut out = Vec::new();
        for (t, p, s) in terms {
            let mut raw = Vec::new();
            collect_additive_terms(ctx, t, p, &mut raw);
            for (rt, rp) in raw {
                out.push((rt, rp, s));
            }
        }
        out
    };
    let mut lhs_terms = re_flatten(lhs_terms, &simplifier.context);
    let mut rhs_terms = re_flatten(rhs_terms, &simplifier.context);

    // Context-aware expansion phase: expand Pow(Add,n) terms only
    // when expanded terms overlap with the opposing side's terms.
    // This enables cancellation of (x+1)^2 - (x² + 2x + 1) → 0
    // without requiring expand_mode in the global simplifier.
    {
        let rhs_fps: HashSet<u64> = rhs_terms
            .iter()
            .map(|(t, _, _)| term_fingerprint(&simplifier.context, *t))
            .collect();
        let lhs_fps: HashSet<u64> = lhs_terms
            .iter()
            .map(|(t, _, _)| term_fingerprint(&simplifier.context, *t))
            .collect();

        let lhs_expanded = try_expand_for_cancel_with(
            &mut simplifier.context,
            &mut lhs_terms,
            &rhs_fps,
            crate::expand::expand,
        );
        let rhs_expanded = try_expand_for_cancel_with(
            &mut simplifier.context,
            &mut rhs_terms,
            &lhs_fps,
            crate::expand::expand,
        );

        // If any expansion happened, re-simplify and re-flatten the new terms
        if lhs_expanded || rhs_expanded {
            for (term, _, _) in &mut lhs_terms {
                let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
                *term = s;
            }
            for (term, _, _) in &mut rhs_terms {
                let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
                *term = s;
            }
            lhs_terms = re_flatten(lhs_terms, &simplifier.context);
            rhs_terms = re_flatten(rhs_terms, &simplifier.context);
        }
    }

    // Guard: skip if too many terms
    if lhs_terms.len() > MAX_TERMS || rhs_terms.len() > MAX_TERMS {
        return None;
    }

    let mut lhs_used = vec![false; lhs_terms.len()];
    let mut rhs_used = vec![false; rhs_terms.len()];
    let mut cancelled = 0;

    // First pass: structural match — definability-sound, no proof needed.
    //
    // When compare_expr confirms l == r structurally, the cancellation is
    // t − t = 0, which is a tautology at every point where t is defined.
    // Since both terms originated from the equation, they ARE defined in
    // the equation's domain. We treat equation validity over the
    // definability domain of its terms; cancelling identical terms does
    // not widen domain.
    //
    // Note: this is NOT the same as rewriting t to something else (like
    // x/x → 1, which requires x≠0). We are only asserting t − t = 0,
    // which holds universally when t is defined.
    //
    // SAFETY GATE: only cancel without proof if BOTH terms are
    // DefinabilityPreserving. If either came from a log-normalizer
    // (NeedsAnalyticConditions), skip — the structural equality was
    // "manufactured" by normalize_for_cancel and may not hold without
    // analytic assumptions (e.g., positivity for ln).
    for (ri, (rt, rp, rs)) in rhs_terms.iter().enumerate() {
        if rhs_used[ri] {
            continue;
        }
        for (li, (lt, lp, ls)) in lhs_terms.iter().enumerate() {
            if lhs_used[li] {
                continue;
            }
            if lp == rp
                && *ls == OriginSafety::DefinabilityPreserving
                && *rs == OriginSafety::DefinabilityPreserving
                && compare_expr(&simplifier.context, *lt, *rt) == Ordering::Equal
            {
                lhs_used[li] = true;
                rhs_used[ri] = true;
                cancelled += 1;
                break;
            }
        }
    }

    // Second pass: semantic match (simplify(l - r) == 0) with Strict proof.
    // This covers both DefinabilityPreserving terms that didn't match
    // structurally and NeedsAnalyticConditions terms.
    for (li, (lt, lp, _ls)) in lhs_terms.iter().enumerate() {
        if lhs_used[li] {
            continue;
        }
        // Guard: skip large terms
        let l_nodes = count_nodes_matching(&simplifier.context, *lt, |_| true);
        if l_nodes > MAX_NODES {
            continue;
        }

        for (ri, (rt, rp, _rs)) in rhs_terms.iter().enumerate() {
            if rhs_used[ri] {
                continue;
            }
            if lp != rp {
                continue;
            }
            // Guard: skip large terms
            let r_nodes = count_nodes_matching(&simplifier.context, *rt, |_| true);
            if r_nodes > MAX_NODES {
                continue;
            }

            // Pair proof with Strict domain: verify l - r == 0
            // without unproven domain assumptions.
            let diff = simplifier.context.add(Expr::Sub(*lt, *rt));
            let (simplified_diff, _, _) =
                simplifier.simplify_with_stats(diff, strict_proof_opts.clone());

            // Check if result is zero
            let is_zero = if let Expr::Number(n) = simplifier.context.get(simplified_diff) {
                n.is_zero()
            } else {
                false
            };
            if is_zero {
                lhs_used[li] = true;
                rhs_used[ri] = true;
                cancelled += 1;
                break;
            }
        }
    }

    if cancelled == 0 {
        return None;
    }

    let new_lhs_terms: Vec<(ExprId, bool)> = lhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !lhs_used[*i])
        .map(|(_, (t, p, _))| (t, p))
        .collect();
    let new_rhs_terms: Vec<(ExprId, bool)> = rhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !rhs_used[*i])
        .map(|(_, (t, p, _))| (t, p))
        .collect();

    Some(CancelResult {
        new_lhs: rebuild_from_terms(&mut simplifier.context, &new_lhs_terms),
        new_rhs: rebuild_from_terms(&mut simplifier.context, &new_rhs_terms),
        cancelled_count: cancelled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};

    #[test]
    fn test_cancel_simple() {
        // (x^2 + y) - y → x^2 (on LHS), 0 (on RHS)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x2 = ctx.add(Expr::Pow(x, two));
        let y = ctx.var("y");
        let lhs = ctx.add(Expr::Add(x2, y));
        let rhs = y;
        let result = cancel_common_additive_terms(&mut ctx, lhs, rhs).unwrap();
        assert_eq!(result.cancelled_count, 1);
        assert!(matches!(ctx.get(result.new_lhs), Expr::Pow(_, _)));
        assert!(matches!(ctx.get(result.new_rhs), Expr::Number(_))); // 0
    }

    #[test]
    fn test_no_cancel_different_terms() {
        // (x + y) vs z → no cancellation
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let lhs = ctx.add(Expr::Add(x, y));
        assert!(cancel_common_additive_terms(&mut ctx, lhs, z).is_none());
    }

    #[test]
    fn test_cancel_with_duplicates() {
        // (a + b + b) vs b → cancels one b, leaves (a + b) vs 0
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let b2 = ctx.var("b");
        let ab = ctx.add(Expr::Add(a, b));
        let lhs = ctx.add(Expr::Add(ab, b2));
        let result = cancel_common_additive_terms(&mut ctx, lhs, b).unwrap();
        assert_eq!(result.cancelled_count, 1);
    }

    #[test]
    fn test_cancel_symmetric() {
        // (a + b + c) vs (b + c) → a vs 0, cancelled=2
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let ab = ctx.add(Expr::Add(a, b));
        let lhs = ctx.add(Expr::Add(ab, c));
        let rhs = ctx.add(Expr::Add(b, c));
        let result = cancel_common_additive_terms(&mut ctx, lhs, rhs).unwrap();
        assert_eq!(result.cancelled_count, 2);
        // new_lhs should be a, new_rhs should be 0
        assert!(matches!(ctx.get(result.new_lhs), Expr::Variable(_)));
        assert!(matches!(ctx.get(result.new_rhs), Expr::Number(_)));
    }
}
