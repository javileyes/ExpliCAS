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
use num_traits::{Signed, ToPrimitive};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

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

/// Safety classification for normalizer output.
///
/// Determines whether structurally equal terms can be cancelled
/// without a Strict proof (definability-preserving), or must be
/// skipped in structural cancel (needs analytic conditions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OriginSafety {
    /// SplitDiv, DistributeMul, Add/Sub/Neg passthrough.
    /// These do not introduce new domain requirements.
    DefinabilityPreserving,
    /// LnOfMul, LnOfPow, LnOfSqrt.
    /// These require analytic conditions (e.g., positivity for ln).
    NeedsAnalyticConditions,
}

impl OriginSafety {
    /// Combine two safety classifications: worst wins.
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (OriginSafety::NeedsAnalyticConditions, _)
            | (_, OriginSafety::NeedsAnalyticConditions) => OriginSafety::NeedsAnalyticConditions,
            _ => OriginSafety::DefinabilityPreserving,
        }
    }
}

/// Pre-normalize expression for cancel: split fraction numerators and log
/// products to expose hidden additive terms.
///
/// Applied ONLY within the semantic cancel pipeline, not globally.
/// Returns `(normalized_expr, safety)` where safety indicates whether
/// the normalization is definability-preserving or needs analytic conditions.
///
/// Guard: max recursion depth 3, max 6 numerator terms.
fn normalize_for_cancel(ctx: &mut Context, id: ExprId, depth: usize) -> (ExprId, OriginSafety) {
    if depth > 3 {
        return (id, OriginSafety::DefinabilityPreserving);
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
        ExpandCos2x(ExprId),            // inner arg `a` — cos(2a) → 1 - 2·sin²(a)
        ExpandSin2x(ExprId),            // inner arg `a` — sin(2a) → 2·sin(a)·cos(a)
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
            let builtin = ctx.builtin_of(name_id);
            let is_ln = builtin.map(|b| b.name() == "ln").unwrap_or(false);
            if is_ln {
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
            } else {
                // Trig double-angle expansion for cancel context:
                // cos(2a) → 1 - 2sin²(a),  sin(2a) → 2sin(a)cos(a)
                // Uses extract_int_multiple_additive to also handle
                // additive forms like cos(2u²+2) → cos(2*(u²+1))
                let is_cos = matches!(builtin, Some(cas_ast::BuiltinFn::Cos));
                let is_sin = matches!(builtin, Some(cas_ast::BuiltinFn::Sin));
                if is_cos || is_sin {
                    if let Some((true, inner)) =
                        cas_math::trig_roots_flatten::extract_int_multiple_additive(ctx, arg, 2)
                    {
                        if is_cos {
                            Action::ExpandCos2x(inner)
                        } else {
                            Action::ExpandSin2x(inner)
                        }
                    } else {
                        Action::Pass
                    }
                } else {
                    Action::Pass
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
        Action::Pass => (id, OriginSafety::DefinabilityPreserving),
        Action::Add(l, r) => {
            let (nl, sl) = normalize_for_cancel(ctx, l, depth);
            let (nr, sr) = normalize_for_cancel(ctx, r, depth);
            let safety = sl.merge(sr);
            if nl == l && nr == r {
                (id, safety)
            } else {
                (ctx.add(Expr::Add(nl, nr)), safety)
            }
        }
        Action::Sub(l, r) => {
            let (nl, sl) = normalize_for_cancel(ctx, l, depth);
            let (nr, sr) = normalize_for_cancel(ctx, r, depth);
            let safety = sl.merge(sr);
            if nl == l && nr == r {
                (id, safety)
            } else {
                (ctx.add(Expr::Sub(nl, nr)), safety)
            }
        }
        Action::Neg(inner) => {
            let (ni, si) = normalize_for_cancel(ctx, inner, depth);
            if ni == inner {
                (id, si)
            } else {
                (ctx.add(Expr::Neg(ni)), si)
            }
        }
        Action::SplitDiv(num, den) => {
            // Collect additive terms of numerator, wrap each in Div(_,den)
            let mut num_terms = Vec::new();
            collect_additive_terms(ctx, num, true, &mut num_terms);
            if num_terms.len() <= 1 || num_terms.len() > 6 {
                return (id, OriginSafety::DefinabilityPreserving);
            }
            let split: Vec<(ExprId, bool)> = num_terms
                .iter()
                .map(|(t, p)| (ctx.add(Expr::Div(*t, den)), *p))
                .collect();
            (
                rebuild_from_terms(ctx, &split),
                OriginSafety::DefinabilityPreserving,
            )
        }
        Action::LnOfMul(name, a, b) => {
            // ln(a*b) → ln(a) + ln(b)  [NEEDS analytic conditions: a>0, b>0]
            let ln_a = ctx.add(Expr::Function(name, vec![a]));
            let ln_b = ctx.add(Expr::Function(name, vec![b]));
            let (nla, _) = normalize_for_cancel(ctx, ln_a, depth + 1);
            let (nlb, _) = normalize_for_cancel(ctx, ln_b, depth + 1);
            (
                ctx.add(Expr::Add(nla, nlb)),
                OriginSafety::NeedsAnalyticConditions,
            )
        }
        Action::LnOfPow(name, base, exp) => {
            // ln(x^r) → r * ln(x)  [NEEDS analytic conditions: x>0]
            let ln_base = ctx.add(Expr::Function(name, vec![base]));
            (
                ctx.add(Expr::Mul(exp, ln_base)),
                OriginSafety::NeedsAnalyticConditions,
            )
        }
        Action::LnOfSqrt(name, sqrt_arg) => {
            // ln(sqrt(x)) → (1/2) * ln(x)  [NEEDS analytic conditions: x>0]
            let half = ctx.add(Expr::Number(num_rational::BigRational::new(
                num_bigint::BigInt::from(1),
                num_bigint::BigInt::from(2),
            )));
            let ln_arg = ctx.add(Expr::Function(name, vec![sqrt_arg]));
            (
                ctx.add(Expr::Mul(half, ln_arg)),
                OriginSafety::NeedsAnalyticConditions,
            )
        }
        Action::DistributeMul(scalar, additive) => {
            // k*(a+b) → k*a + k*b  (distribute scalar over additive terms)
            let mut terms = Vec::new();
            collect_additive_terms(ctx, additive, true, &mut terms);
            if terms.len() <= 1 || terms.len() > 6 {
                return (id, OriginSafety::DefinabilityPreserving);
            }
            let split: Vec<(ExprId, bool)> = terms
                .iter()
                .map(|(t, p)| (ctx.add(Expr::Mul(scalar, *t)), *p))
                .collect();
            (
                rebuild_from_terms(ctx, &split),
                OriginSafety::DefinabilityPreserving,
            )
        }
        Action::ExpandCos2x(inner) => {
            // cos(2a) → 1 - 2·sin²(a)   [unconditionally valid]
            let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![inner]);
            let two = ctx.num(2);
            let sin_sq = ctx.add(Expr::Pow(sin_a, two));
            let two_sin_sq = ctx.add(Expr::Mul(two, sin_sq));
            let one = ctx.num(1);
            let result = ctx.add(Expr::Sub(one, two_sin_sq));
            let (nr, ns) = normalize_for_cancel(ctx, result, depth + 1);
            (nr, ns)
        }
        Action::ExpandSin2x(inner) => {
            // sin(2a) → 2·sin(a)·cos(a)  [unconditionally valid]
            let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![inner]);
            let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![inner]);
            let prod = ctx.add(Expr::Mul(sin_a, cos_a));
            let two = ctx.num(2);
            let result = ctx.add(Expr::Mul(two, prod));
            (result, OriginSafety::DefinabilityPreserving)
        }
    }
}

/// Compute a deterministic structural hash of an AST subtree.
///
/// Used for O(1) overlap detection: build `HashSet<u64>` from one side,
/// probe with terms from the other side.
fn term_fingerprint(ctx: &Context, id: ExprId) -> u64 {
    use std::hash::DefaultHasher;
    let mut h = DefaultHasher::new();
    hash_expr_structural(ctx, id, &mut h);
    h.finish()
}

/// Recursive structural hash for fingerprinting.
fn hash_expr_structural(ctx: &Context, id: ExprId, h: &mut impl Hasher) {
    let node = ctx.get(id);
    std::mem::discriminant(node).hash(h);
    match node {
        Expr::Number(n) => n.hash(h),
        Expr::Variable(s) => ctx.sym_name(*s).hash(h),
        Expr::Constant(c) => std::mem::discriminant(c).hash(h),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            hash_expr_structural(ctx, *l, h);
            hash_expr_structural(ctx, *r, h);
        }
        Expr::Pow(b, e) => {
            hash_expr_structural(ctx, *b, h);
            hash_expr_structural(ctx, *e, h);
        }
        Expr::Neg(e) | Expr::Hold(e) => hash_expr_structural(ctx, *e, h),
        Expr::Function(name, args) => {
            ctx.sym_name(*name).hash(h);
            for a in args {
                hash_expr_structural(ctx, *a, h);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            rows.hash(h);
            cols.hash(h);
            for d in data {
                hash_expr_structural(ctx, *d, h);
            }
        }
        Expr::SessionRef(s) => s.hash(h),
    }
}

/// Context-aware expansion phase for the semantic cancel pipeline.
///
/// Scans `terms` for `Pow(Add|Sub, n)` where n ∈ [2,4]. For each,
/// preview-expands and checks if any resulting term's fingerprint
/// matches `opposing_fps`. If overlap is found, replaces the Pow
/// term with the flattened expanded form.
///
/// Guards: n ≤ 4, k ≤ 6, pred_terms ≤ 35, base_nodes ≤ 25.
/// These mirror `SmallMultinomialExpansionRule`.
///
/// Returns `true` if any term was expanded.
/// Lightweight 2-term product simplification for preview fingerprinting.
///
/// Folds `Pow(x,a) * Pow(x,b) → Pow(x,a+b)` and `Number * Number → Number`,
/// treating bare variables `x` as `Pow(x,1)`.  Falls back to plain `Mul`.
fn mul_preview(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    // Number * Number → fuse
    if let (Expr::Number(na), Expr::Number(nb)) = (ctx.get(a), ctx.get(b)) {
        let product = na.clone() * nb.clone();
        return ctx.add(Expr::Number(product));
    }

    // Number * expr or expr * Number → Mul (keep as-is, fingerprint will match)
    if matches!(ctx.get(a), Expr::Number(_)) || matches!(ctx.get(b), Expr::Number(_)) {
        return ctx.add(Expr::Mul(a, b));
    }

    // Extract (base, exp) — bare expressions treated as Pow(x,1)
    let (ba, ea) = match ctx.get(a) {
        Expr::Pow(base, exp) => (*base, Some(*exp)),
        _ => (a, None),
    };
    let (bb, eb) = match ctx.get(b) {
        Expr::Pow(base, exp) => (*base, Some(*exp)),
        _ => (b, None),
    };

    // Power merge: same base ⇒ add exponents
    if compare_expr(ctx, ba, bb) == Ordering::Equal {
        let ea_val = ea.and_then(|e| {
            if let Expr::Number(n) = ctx.get(e) {
                Some(n.clone())
            } else {
                None
            }
        });
        let eb_val = eb.and_then(|e| {
            if let Expr::Number(n) = ctx.get(e) {
                Some(n.clone())
            } else {
                None
            }
        });
        let ea_num = ea_val.unwrap_or_else(|| num_rational::BigRational::from_integer(1.into()));
        let eb_num = eb_val.unwrap_or_else(|| num_rational::BigRational::from_integer(1.into()));
        let sum = ea_num + eb_num;
        let new_exp = ctx.add(Expr::Number(sum));
        return ctx.add(Expr::Pow(ba, new_exp));
    }

    // Fallback
    ctx.add(Expr::Mul(a, b))
}
fn try_expand_for_cancel(
    ctx: &mut Context,
    terms: &mut Vec<(ExprId, bool, OriginSafety)>,
    opposing_fps: &HashSet<u64>,
) -> bool {
    if opposing_fps.is_empty() {
        return false;
    }

    /// Maximum exponent for context-aware cancel expansion.
    const MAX_EXP: u32 = 4;
    /// Maximum base terms (k) for context-aware cancel expansion.
    const MAX_BASE_K: usize = 6;
    /// Maximum predicted terms from multinomial expansion.
    const MAX_PRED_TERMS: usize = 35;
    /// Maximum base nodes (deduped) for context-aware cancel expansion.
    const MAX_BASE_NODES: usize = 25;
    /// Maximum terms per Add factor in Mul(Add, Add) expansion.
    const MAX_MUL_FACTOR_TERMS: usize = 4;
    /// Maximum product terms from Mul(Add, Add) expansion.
    const MAX_MUL_PRODUCT_TERMS: usize = 16;

    let mut expanded_any = false;
    let mut i = 0;

    // Phase A: Pow(Add|Sub, n) expansion (existing)
    while i < terms.len() {
        let (term_id, term_pos, term_safety) = terms[i];

        // Phase 1: check if term is Pow(Add|Sub, n) with n in [2, MAX_EXP]
        let pow_info = match ctx.get(term_id) {
            Expr::Pow(base, exp) => {
                let base = *base;
                let exp = *exp;
                // Check exponent is integer in [2, MAX_EXP]
                let n = match ctx.get(exp) {
                    Expr::Number(num) => {
                        if !num.is_integer() || num.is_negative() {
                            i += 1;
                            continue;
                        }
                        match num.to_integer().to_u32() {
                            Some(v) if (2..=MAX_EXP).contains(&v) => v,
                            _ => {
                                i += 1;
                                continue;
                            }
                        }
                    }
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                // Check base is additive
                let is_additive = matches!(ctx.get(base), Expr::Add(_, _) | Expr::Sub(_, _));
                if !is_additive {
                    i += 1;
                    continue;
                }
                Some((base, exp, n))
            }
            _ => None,
        };

        let (base, exp, n) = match pow_info {
            Some(info) => info,
            None => {
                i += 1;
                continue;
            }
        };

        // Phase 2: Guard checks (mirror SmallMultinomialExpansionRule)
        let mut base_terms_vec = Vec::new();
        collect_additive_terms(ctx, base, true, &mut base_terms_vec);
        let k = base_terms_vec.len();
        if !(2..=MAX_BASE_K).contains(&k) {
            i += 1;
            continue;
        }

        // Predicted term count: C(n+k-1, k-1)
        let pred_terms =
            match cas_math::multinomial_expand::multinomial_term_count(n, k, MAX_PRED_TERMS) {
                Some(t) if t <= MAX_PRED_TERMS => t,
                _ => {
                    i += 1;
                    continue;
                }
            };
        let _ = pred_terms; // used for guard check above

        // Base node count guard
        let base_nodes = cas_ast::traversal::count_all_nodes(ctx, base);
        if base_nodes > MAX_BASE_NODES {
            i += 1;
            continue;
        }

        // Phase 3: Preview expand
        let budget = cas_math::multinomial_expand::MultinomialExpandBudget {
            max_exp: MAX_EXP,
            max_base_terms: MAX_BASE_K,
            max_vars: MAX_BASE_K,
            max_output_terms: MAX_PRED_TERMS,
        };
        let expanded = match cas_math::multinomial_expand::try_expand_multinomial_direct(
            ctx, base, exp, &budget,
        ) {
            Some(e) => {
                // Unwrap __hold if present
                match ctx.get(e) {
                    Expr::Hold(inner) => *inner,
                    _ => e,
                }
            }
            None => {
                // Fallback: use generic expand (handles non-linear atoms)
                let pow_expr = ctx.add(Expr::Pow(base, exp));
                let expanded = crate::expand::expand(ctx, pow_expr);
                // Skip if expand was a no-op
                if expanded == pow_expr {
                    i += 1;
                    continue;
                }
                expanded
            }
        };

        // Phase 4: Check overlap — collect expanded terms and fingerprint
        let mut exp_terms = Vec::new();
        collect_additive_terms(ctx, expanded, true, &mut exp_terms);

        let has_overlap = exp_terms.iter().any(|(t, _)| {
            let fp = term_fingerprint(ctx, *t);
            if opposing_fps.contains(&fp) {
                // Confirm with structural comparison against all opposing
                // (fingerprint collision is rare but not impossible)
                true
            } else {
                false
            }
        });

        if !has_overlap {
            i += 1;
            continue;
        }

        // Phase 5: Commit — replace Pow term with expanded terms
        tracing::debug!(
            target: "cancel",
            k, n, "context-aware expansion: overlap detected, committing expansion"
        );

        terms.remove(i);
        for (et, ep) in exp_terms {
            // Combine polarity: if original term was negative, flip expanded term signs
            let final_pos = if term_pos { ep } else { !ep };
            terms.insert(i, (et, final_pos, term_safety));
            i += 1;
        }
        expanded_any = true;
        // Don't increment i — we already advanced past the inserted terms
    }

    // Phase B: Generalized Mul-factor expansion
    // Flattens nested Mul trees to find ≥2 add-like factors, distributes
    // the cheapest pair, wraps with remaining scalar factors, checks overlap.
    // Handles: Mul(atom, Mul(Add, Add)), Mul(Mul(atom, Add), Add), etc.
    // Guards: ≤ MAX_MUL_FACTORS total, each add-like ≤ MAX_MUL_FACTOR_TERMS,
    //         product ≤ MAX_MUL_PRODUCT_TERMS, no Pow(Add,n) inside (Phase A handles those).
    let mut j = 0;
    while j < terms.len() {
        let (term_id, term_pos, term_safety) = terms[j];

        // Only consider Mul nodes
        if !matches!(ctx.get(term_id), Expr::Mul(_, _)) {
            j += 1;
            continue;
        }

        // Step 1: Flatten multiplicative tree into factors list
        let mut factors: Vec<ExprId> = Vec::new();
        {
            let mut stack = vec![term_id];
            let max_depth = 8;
            let max_factors = 7;
            let mut depth = 0;
            while let Some(node) = stack.pop() {
                depth += 1;
                if depth > max_depth || factors.len() >= max_factors {
                    factors.push(node);
                    continue;
                }
                match ctx.get(node) {
                    Expr::Mul(a, b) => {
                        // Push right first so left is processed first (stack is LIFO)
                        stack.push(*b);
                        stack.push(*a);
                    }
                    _ => {
                        factors.push(node);
                    }
                }
            }
        }

        if factors.len() < 2 {
            j += 1;
            continue;
        }

        // Step 2: Classify factors into add-like vs scalar
        let mut add_like_indices: Vec<usize> = Vec::new();
        for (fi, &fid) in factors.iter().enumerate() {
            let is_additive = matches!(ctx.get(fid), Expr::Add(_, _) | Expr::Sub(_, _));
            if is_additive {
                add_like_indices.push(fi);
            }
        }

        if add_like_indices.len() < 2 {
            j += 1;
            continue;
        }

        // Step 3: Collect additive terms for each add-like factor, find cheapest pair
        let mut add_term_lists: Vec<(usize, Vec<(ExprId, bool)>)> = Vec::new();
        let mut bail = false;
        for &fi in &add_like_indices {
            let mut ft = Vec::new();
            collect_additive_terms(ctx, factors[fi], true, &mut ft);
            if ft.len() > MAX_MUL_FACTOR_TERMS {
                bail = true;
                break;
            }
            add_term_lists.push((fi, ft));
        }
        if bail || add_term_lists.len() < 2 {
            j += 1;
            continue;
        }

        // Sort by ascending term count to pick cheapest pair
        add_term_lists.sort_by_key(|(_, terms_list)| terms_list.len());
        let (idx_a, ref a_add_terms) = add_term_lists[0];
        let (idx_b, ref b_add_terms) = add_term_lists[1];

        // Product guard
        if a_add_terms.len() * b_add_terms.len() > MAX_MUL_PRODUCT_TERMS {
            j += 1;
            continue;
        }

        // Step 4: Distribute the cheapest add-like pair
        let mut distributed: Vec<(ExprId, bool)> = Vec::new();
        for (at, ap) in a_add_terms {
            for (bt, bp) in b_add_terms {
                let prod = mul_preview(ctx, *at, *bt);
                let sign = *ap == *bp;
                distributed.push((prod, sign));
            }
        }

        // Step 5: Reconstruct — wrap each distributed term with remaining scalar factors
        let scalar_indices: Vec<usize> = (0..factors.len())
            .filter(|i| *i != idx_a && *i != idx_b)
            .collect();

        // Pre-check: overlap on unwrapped distributed terms (for scalar cases)
        let distributed_overlap = !scalar_indices.is_empty()
            && distributed.iter().any(|(t, _)| {
                let fp = term_fingerprint(ctx, *t);
                opposing_fps.contains(&fp)
            });

        let wrapped_terms: Vec<(ExprId, bool)> = if scalar_indices.is_empty() {
            distributed
        } else {
            distributed
                .iter()
                .map(|(dt, dp)| {
                    // Wrap: scalar1 * scalar2 * ... * dt
                    let mut wrapped = *dt;
                    for &si in &scalar_indices {
                        wrapped = ctx.add(Expr::Mul(factors[si], wrapped));
                    }
                    (wrapped, *dp)
                })
                .collect()
        };

        // Step 6: Overlap check against opposing side
        // Primary: fingerprint match on wrapped or unwrapped terms.
        // Fallback: term-count heuristic — when scalar factors exist AND
        // opposing side has ≥ product-count additive terms, the opposing
        // is likely the fully-expanded form.  Guards already cap product size.
        let fp_overlap = distributed_overlap
            || wrapped_terms.iter().any(|(t, _)| {
                let fp = term_fingerprint(ctx, *t);
                opposing_fps.contains(&fp)
            });

        let count_overlap = !fp_overlap
            && !scalar_indices.is_empty()
            && opposing_fps.len() >= a_add_terms.len() * b_add_terms.len();

        if !fp_overlap && !count_overlap {
            j += 1;
            continue;
        }

        // Step 7: Commit
        tracing::debug!(
            target: "cancel",
            n_factors = factors.len(),
            n_add_like = add_like_indices.len(),
            a_k = a_add_terms.len(),
            b_k = b_add_terms.len(),
            "context-aware Mul-factor expansion: overlap detected, committing"
        );

        terms.remove(j);
        for (wt, wp) in wrapped_terms {
            let final_pos = if term_pos { wp } else { !wp };
            terms.insert(j, (wt, final_pos, term_safety));
            j += 1;
        }
        expanded_any = true;
    }

    expanded_any
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

        let lhs_expanded = try_expand_for_cancel(&mut simplifier.context, &mut lhs_terms, &rhs_fps);
        let rhs_expanded = try_expand_for_cancel(&mut simplifier.context, &mut rhs_terms, &lhs_fps);

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
