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
use std::cmp::Ordering;

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

/// Pre-normalize expression for cancel: split fraction numerators and log
/// products to expose hidden additive terms.
///
/// Applied ONLY within the semantic cancel pipeline, not globally.
/// - `Div(Add(a,b), D)` → `Add(Div(a,D), Div(b,D))`
/// - `ln(a*b)` → `ln(a) + ln(b)`
/// - `ln(x^r)` → `r * ln(x)`
///
/// Guard: max recursion depth 3, max 6 numerator terms.
fn normalize_for_cancel(ctx: &mut Context, id: ExprId, depth: usize) -> ExprId {
    if depth > 3 {
        return id;
    }

    // Phase 1: classify expression (immutable borrow released before phase 2)
    enum Action {
        Pass,
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Neg(ExprId),
        SplitDiv(ExprId, ExprId),       // (numerator, denominator)
        LnOfMul(usize, ExprId, ExprId), // (ln_name, a, b)
        LnOfPow(usize, ExprId, ExprId), // (ln_name, base, exp)
        LnOfSqrt(usize, ExprId),        // (ln_name, sqrt_arg) — ln(sqrt(x)) → ½·ln(x)
        DistributeMul(ExprId, ExprId),  // (scalar, additive) — k*(a+b) → k*a + k*b
    }

    let action = match ctx.get(id) {
        Expr::Add(l, r) => Action::Add(*l, *r),
        Expr::Sub(l, r) => Action::Sub(*l, *r),
        Expr::Neg(inner) => Action::Neg(*inner),
        Expr::Div(num, den) => {
            // Only split if numerator has additive structure
            let has_add = matches!(ctx.get(*num), Expr::Add(_, _) | Expr::Sub(_, _));
            if has_add {
                Action::SplitDiv(*num, *den)
            } else {
                Action::Pass
            }
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let name_id = *name;
            let arg = args[0];
            let is_ln = ctx
                .builtin_of(name_id)
                .map(|b| b.name() == "ln")
                .unwrap_or(false);
            if !is_ln {
                Action::Pass
            } else {
                match ctx.get(arg) {
                    Expr::Mul(a, b) => Action::LnOfMul(name_id, *a, *b),
                    Expr::Pow(base, exp) => Action::LnOfPow(name_id, *base, *exp),
                    Expr::Function(inner_name, inner_args) if inner_args.len() == 1 => {
                        let is_sqrt = ctx
                            .builtin_of(*inner_name)
                            .map(|b| b.name() == "sqrt")
                            .unwrap_or(false);
                        if is_sqrt {
                            Action::LnOfSqrt(name_id, inner_args[0])
                        } else {
                            Action::Pass
                        }
                    }
                    _ => Action::Pass,
                }
            }
        }
        Expr::Mul(a, b) => {
            // k·(sum) → distribute, where k is a non-additive factor
            let a_add = matches!(ctx.get(*a), Expr::Add(_, _) | Expr::Sub(_, _));
            let b_add = matches!(ctx.get(*b), Expr::Add(_, _) | Expr::Sub(_, _));
            if a_add && !b_add {
                Action::DistributeMul(*b, *a) // scalar=b, additive=a
            } else if b_add && !a_add {
                Action::DistributeMul(*a, *b) // scalar=a, additive=b
            } else {
                Action::Pass
            }
        }
        _ => Action::Pass,
    };

    // Phase 2: build normalized expression (mutable borrow)
    match action {
        Action::Pass => id,
        Action::Add(l, r) => {
            let nl = normalize_for_cancel(ctx, l, depth);
            let nr = normalize_for_cancel(ctx, r, depth);
            if nl == l && nr == r {
                id
            } else {
                ctx.add(Expr::Add(nl, nr))
            }
        }
        Action::Sub(l, r) => {
            let nl = normalize_for_cancel(ctx, l, depth);
            let nr = normalize_for_cancel(ctx, r, depth);
            if nl == l && nr == r {
                id
            } else {
                ctx.add(Expr::Sub(nl, nr))
            }
        }
        Action::Neg(inner) => {
            let ni = normalize_for_cancel(ctx, inner, depth);
            if ni == inner {
                id
            } else {
                ctx.add(Expr::Neg(ni))
            }
        }
        Action::SplitDiv(num, den) => {
            // Collect additive terms of numerator, wrap each in Div(_,den)
            let mut num_terms = Vec::new();
            collect_additive_terms(ctx, num, true, &mut num_terms);
            if num_terms.len() <= 1 || num_terms.len() > 6 {
                return id;
            }
            let split: Vec<(ExprId, bool)> = num_terms
                .iter()
                .map(|(t, p)| (ctx.add(Expr::Div(*t, den)), *p))
                .collect();
            rebuild_from_terms(ctx, &split)
        }
        Action::LnOfMul(name, a, b) => {
            // ln(a*b) → ln(a) + ln(b)
            let ln_a = ctx.add(Expr::Function(name, vec![a]));
            let ln_b = ctx.add(Expr::Function(name, vec![b]));
            let nla = normalize_for_cancel(ctx, ln_a, depth + 1);
            let nlb = normalize_for_cancel(ctx, ln_b, depth + 1);
            ctx.add(Expr::Add(nla, nlb))
        }
        Action::LnOfPow(name, base, exp) => {
            // ln(x^r) → r * ln(x)
            let ln_base = ctx.add(Expr::Function(name, vec![base]));
            ctx.add(Expr::Mul(exp, ln_base))
        }
        Action::LnOfSqrt(name, sqrt_arg) => {
            // ln(sqrt(x)) → (1/2) * ln(x)
            let half = ctx.add(Expr::Number(num_rational::BigRational::new(
                num_bigint::BigInt::from(1),
                num_bigint::BigInt::from(2),
            )));
            let ln_arg = ctx.add(Expr::Function(name, vec![sqrt_arg]));
            ctx.add(Expr::Mul(half, ln_arg))
        }
        Action::DistributeMul(scalar, additive) => {
            // k*(a+b) → k*a + k*b  (distribute scalar over additive terms)
            let mut terms = Vec::new();
            collect_additive_terms(ctx, additive, true, &mut terms);
            if terms.len() <= 1 || terms.len() > 6 {
                return id;
            }
            let split: Vec<(ExprId, bool)> = terms
                .iter()
                .map(|(t, p)| (ctx.add(Expr::Mul(scalar, *t)), *p))
                .collect();
            rebuild_from_terms(ctx, &split)
        }
    }
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

    // Pre-normalize: split Div(sum, D) and ln(product) into finer additive terms
    let norm_lhs = normalize_for_cancel(&mut simplifier.context, lhs, 0);
    let norm_rhs = normalize_for_cancel(&mut simplifier.context, rhs, 0);

    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_additive_terms(&simplifier.context, norm_lhs, true, &mut lhs_terms);
    collect_additive_terms(&simplifier.context, norm_rhs, true, &mut rhs_terms);

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
    for (term, _) in &mut lhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }
    for (term, _) in &mut rhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }

    // Second normalize pass: simplification may expose new ln(product) terms
    // (e.g., Div(Mul(x, ln(a*b)), x) → ln(a*b) after the first simplify).
    // Re-normalize and re-collect if any term expanded.
    let re_normalize_terms = |ctx: &mut Context, terms: &[(ExprId, bool)]| -> Vec<(ExprId, bool)> {
        let mut out = Vec::new();
        for &(t, p) in terms {
            let n = normalize_for_cancel(ctx, t, 0);
            if n == t {
                out.push((t, p));
            } else {
                collect_additive_terms(ctx, n, p, &mut out);
            }
        }
        out
    };
    let lhs_terms2 = re_normalize_terms(&mut simplifier.context, &lhs_terms);
    let rhs_terms2 = re_normalize_terms(&mut simplifier.context, &rhs_terms);
    let mut lhs_terms = lhs_terms2;
    let mut rhs_terms = rhs_terms2;

    // Re-simplify any newly created terms
    for (term, _) in &mut lhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }
    for (term, _) in &mut rhs_terms {
        let (s, _, _) = simplifier.simplify_with_stats(*term, candidate_opts.clone());
        *term = s;
    }

    // Final re-flatten: simplification may turn a single term into Add(a, b),
    // e.g., Div(Mul(Add(ln(y),2),2), 2) → Add(ln(y), 2). Re-collect to split.
    let re_flatten = |terms: Vec<(ExprId, bool)>, ctx: &Context| -> Vec<(ExprId, bool)> {
        let mut out = Vec::new();
        for (t, p) in terms {
            collect_additive_terms(ctx, t, p, &mut out);
        }
        out
    };
    let lhs_terms = re_flatten(lhs_terms, &simplifier.context);
    let rhs_terms = re_flatten(rhs_terms, &simplifier.context);

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
    for (ri, (rt, rp)) in rhs_terms.iter().enumerate() {
        if rhs_used[ri] {
            continue;
        }
        for (li, (lt, lp)) in lhs_terms.iter().enumerate() {
            if lhs_used[li] {
                continue;
            }
            if lp == rp && compare_expr(&simplifier.context, *lt, *rt) == Ordering::Equal {
                lhs_used[li] = true;
                rhs_used[ri] = true;
                cancelled += 1;
                break;
            }
        }
    }

    // Second pass: semantic match (simplify(l - r) == 0)
    for (li, (lt, lp)) in lhs_terms.iter().enumerate() {
        if lhs_used[li] {
            continue;
        }
        // Guard: skip large terms
        let l_nodes = count_nodes_matching(&simplifier.context, *lt, |_| true);
        if l_nodes > MAX_NODES {
            continue;
        }

        for (ri, (rt, rp)) in rhs_terms.iter().enumerate() {
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
        new_lhs: rebuild_from_terms(&mut simplifier.context, &new_lhs_terms),
        new_rhs: rebuild_from_terms(&mut simplifier.context, &new_rhs_terms),
        cancelled_count: cancelled,
    })
}

/// Collect additive terms from an expression, flattening Add/Sub/Neg.
/// Each term is `(ExprId, is_positive)`.
pub(crate) fn collect_additive_terms(
    ctx: &Context,
    id: ExprId,
    positive: bool,
    out: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_additive_terms(ctx, *l, positive, out);
            collect_additive_terms(ctx, *r, positive, out);
        }
        Expr::Sub(l, r) => {
            collect_additive_terms(ctx, *l, positive, out);
            collect_additive_terms(ctx, *r, !positive, out);
        }
        Expr::Neg(inner) => {
            collect_additive_terms(ctx, *inner, !positive, out);
        }
        _ => {
            out.push((id, positive));
        }
    }
}

/// Rebuild an expression from a list of `(term, is_positive)` pairs.
/// Returns `Number(0)` for an empty list.
pub(crate) fn rebuild_from_terms(ctx: &mut Context, terms: &[(ExprId, bool)]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }

    let mut result = if terms[0].1 {
        terms[0].0
    } else {
        ctx.add(Expr::Neg(terms[0].0))
    };

    for &(term, positive) in &terms[1..] {
        if positive {
            result = ctx.add(Expr::Add(result, term));
        } else {
            result = ctx.add(Expr::Sub(result, term));
        }
    }

    result
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
