//! Support routines for semantic additive cancellation pipelines.
//!
//! This module keeps pair-matching and term-selection logic independent from
//! engine orchestration and simplifier internals.

use crate::cancel_expand_support::try_expand_for_cancel_with;
use crate::cancel_normalization_support::OriginSafety;
use crate::cancel_support::{
    collect_additive_terms_signed as collect_additive_terms, rebuild_from_signed_terms,
    structural_expr_fingerprint as term_fingerprint,
};
use cas_ast::ordering::compare_expr;
use cas_ast::traversal::count_nodes_matching;
use cas_ast::{Context, ExprId};
use std::cmp::Ordering;
use std::collections::HashSet;

/// Signed additive term with origin-safety metadata.
pub type SignedSafetyTerm = (ExprId, bool, OriginSafety);

/// Collect additive signed terms from one expression and tag each with the
/// same origin safety classification.
pub fn collect_signed_terms_with_uniform_safety(
    ctx: &Context,
    expr: ExprId,
    safety: OriginSafety,
) -> Vec<SignedSafetyTerm> {
    let mut raw = Vec::new();
    collect_additive_terms(ctx, expr, true, &mut raw);
    raw.into_iter().map(|(t, p)| (t, p, safety)).collect()
}

/// Build a set of structural fingerprints for signed terms.
pub fn signed_term_fingerprints(ctx: &Context, terms: &[SignedSafetyTerm]) -> HashSet<u64> {
    terms
        .iter()
        .map(|(t, _, _)| term_fingerprint(ctx, *t))
        .collect()
}

/// Compute per-term node counts using a caller-provided counting callback.
pub fn signed_term_node_counts<FCount>(terms: &[SignedSafetyTerm], mut count: FCount) -> Vec<usize>
where
    FCount: FnMut(ExprId) -> usize,
{
    terms.iter().map(|(t, _, _)| count(*t)).collect()
}

/// Apply an ExprId-to-ExprId mapping in place to each signed term.
pub fn map_signed_terms_in_place<FMap>(terms: &mut [SignedSafetyTerm], mut map: FMap)
where
    FMap: FnMut(ExprId) -> ExprId,
{
    for (term, _, _) in terms.iter_mut() {
        *term = map(*term);
    }
}

/// Try overlap-aware expansion in both directions:
/// - expand `lhs_terms` against fingerprints from `rhs_terms`,
/// - expand `rhs_terms` against fingerprints from `lhs_terms`.
///
/// Returns `true` when at least one side expanded.
pub fn try_bidirectional_overlap_expansion_with(
    ctx: &mut Context,
    lhs_terms: &mut Vec<SignedSafetyTerm>,
    rhs_terms: &mut Vec<SignedSafetyTerm>,
    fallback_expand: fn(&mut Context, ExprId) -> ExprId,
) -> bool {
    let rhs_fps = signed_term_fingerprints(ctx, rhs_terms);
    let lhs_fps = signed_term_fingerprints(ctx, lhs_terms);
    let lhs_expanded = try_expand_for_cancel_with(ctx, lhs_terms, &rhs_fps, fallback_expand);
    let rhs_expanded = try_expand_for_cancel_with(ctx, rhs_terms, &lhs_fps, fallback_expand);
    lhs_expanded || rhs_expanded
}

/// Usage masks and cancellation counter for pair matching.
#[derive(Debug, Clone)]
pub struct CancelUsage {
    pub lhs_used: Vec<bool>,
    pub rhs_used: Vec<bool>,
    pub cancelled: usize,
}

/// First-pass structural matching.
///
/// Matches pairs only when:
/// - signs are equal,
/// - both terms are `DefinabilityPreserving`,
/// - structural comparison is equal.
pub fn match_structural_definability_pairs(
    ctx: &Context,
    lhs_terms: &[SignedSafetyTerm],
    rhs_terms: &[SignedSafetyTerm],
) -> CancelUsage {
    let mut usage = CancelUsage {
        lhs_used: vec![false; lhs_terms.len()],
        rhs_used: vec![false; rhs_terms.len()],
        cancelled: 0,
    };

    for (ri, (rt, rp, rs)) in rhs_terms.iter().enumerate() {
        if usage.rhs_used[ri] {
            continue;
        }
        for (li, (lt, lp, ls)) in lhs_terms.iter().enumerate() {
            if usage.lhs_used[li] {
                continue;
            }
            if lp == rp
                && *ls == OriginSafety::DefinabilityPreserving
                && *rs == OriginSafety::DefinabilityPreserving
                && compare_expr(ctx, *lt, *rt) == Ordering::Equal
            {
                usage.lhs_used[li] = true;
                usage.rhs_used[ri] = true;
                usage.cancelled += 1;
                break;
            }
        }
    }

    usage
}

/// Second-pass semantic matching using a caller-provided proof callback.
///
/// `lhs_node_counts` / `rhs_node_counts` are precomputed node counts for the
/// corresponding term vectors, used to enforce the max-node guard.
pub fn match_semantic_pairs_with_usage_and_node_counts<FProof>(
    lhs_terms: &[SignedSafetyTerm],
    rhs_terms: &[SignedSafetyTerm],
    lhs_node_counts: &[usize],
    rhs_node_counts: &[usize],
    usage: &mut CancelUsage,
    max_nodes: usize,
    mut prove_equal: FProof,
) where
    FProof: FnMut(ExprId, ExprId) -> bool,
{
    for (li, (lt, lp, _)) in lhs_terms.iter().enumerate() {
        if usage.lhs_used[li] {
            continue;
        }
        if lhs_node_counts[li] > max_nodes {
            continue;
        }

        for (ri, (rt, rp, _)) in rhs_terms.iter().enumerate() {
            if usage.rhs_used[ri] {
                continue;
            }
            if lp != rp {
                continue;
            }
            if rhs_node_counts[ri] > max_nodes {
                continue;
            }

            if prove_equal(*lt, *rt) {
                usage.lhs_used[li] = true;
                usage.rhs_used[ri] = true;
                usage.cancelled += 1;
                break;
            }
        }
    }
}

/// Collect unmatched signed terms from a term list and usage mask.
pub fn collect_unmatched_signed_terms(
    terms: &[SignedSafetyTerm],
    used: &[bool],
) -> Vec<(ExprId, bool)> {
    terms
        .iter()
        .enumerate()
        .filter(|(i, _)| !used[*i])
        .map(|(_, (t, p, _))| (*t, *p))
        .collect()
}

/// Configuration for semantic additive cancellation.
#[derive(Debug, Clone, Copy)]
pub struct SemanticCancelConfig {
    pub max_terms: usize,
    pub max_nodes: usize,
}

/// Result payload produced by semantic additive cancellation.
#[derive(Debug, Clone)]
pub struct SemanticCancelResult {
    pub new_lhs: ExprId,
    pub new_rhs: ExprId,
    pub cancelled_count: usize,
}

/// Generic semantic additive-cancel pipeline that keeps engine-specific
/// simplifier policy behind callbacks.
///
/// This performs normalization, candidate simplification, overlap-aware expansion,
/// structural+semantic matching and rebuild.
#[allow(clippy::too_many_arguments)]
pub fn try_cancel_additive_terms_semantic_with_state<
    S,
    FGetContext,
    FGetContextMut,
    FSimplifyCandidate,
    FProveEqual,
>(
    state: &mut S,
    lhs: ExprId,
    rhs: ExprId,
    config: SemanticCancelConfig,
    mut get_context: FGetContext,
    mut get_context_mut: FGetContextMut,
    mut simplify_candidate: FSimplifyCandidate,
    fallback_expand: fn(&mut Context, ExprId) -> ExprId,
    mut prove_equal: FProveEqual,
) -> Option<SemanticCancelResult>
where
    FGetContext: FnMut(&S) -> &Context,
    FGetContextMut: FnMut(&mut S) -> &mut Context,
    FSimplifyCandidate: FnMut(&mut S, ExprId) -> ExprId,
    FProveEqual: FnMut(&mut S, ExprId, ExprId) -> bool,
{
    let (norm_lhs, lhs_safety) = {
        let ctx = get_context_mut(state);
        crate::cancel_normalization_support::normalize_for_cancel(ctx, lhs, 0)
    };
    let (norm_rhs, rhs_safety) = {
        let ctx = get_context_mut(state);
        crate::cancel_normalization_support::normalize_for_cancel(ctx, rhs, 0)
    };

    let mut lhs_terms = {
        let ctx = get_context(state);
        collect_signed_terms_with_uniform_safety(ctx, norm_lhs, lhs_safety)
    };
    let mut rhs_terms = {
        let ctx = get_context(state);
        collect_signed_terms_with_uniform_safety(ctx, norm_rhs, rhs_safety)
    };

    for (term, _, _) in lhs_terms.iter_mut() {
        *term = simplify_candidate(state, *term);
    }
    for (term, _, _) in rhs_terms.iter_mut() {
        *term = simplify_candidate(state, *term);
    }

    lhs_terms = {
        let ctx = get_context_mut(state);
        crate::cancel_normalization_support::renormalize_signed_terms_for_cancel(ctx, &lhs_terms)
    };
    rhs_terms = {
        let ctx = get_context_mut(state);
        crate::cancel_normalization_support::renormalize_signed_terms_for_cancel(ctx, &rhs_terms)
    };

    for (term, _, _) in lhs_terms.iter_mut() {
        *term = simplify_candidate(state, *term);
    }
    for (term, _, _) in rhs_terms.iter_mut() {
        *term = simplify_candidate(state, *term);
    }

    lhs_terms = {
        let ctx = get_context(state);
        crate::cancel_normalization_support::reflatten_signed_terms_for_cancel(ctx, lhs_terms)
    };
    rhs_terms = {
        let ctx = get_context(state);
        crate::cancel_normalization_support::reflatten_signed_terms_for_cancel(ctx, rhs_terms)
    };

    let expanded = {
        let ctx = get_context_mut(state);
        try_bidirectional_overlap_expansion_with(
            ctx,
            &mut lhs_terms,
            &mut rhs_terms,
            fallback_expand,
        )
    };
    if expanded {
        for (term, _, _) in lhs_terms.iter_mut() {
            *term = simplify_candidate(state, *term);
        }
        for (term, _, _) in rhs_terms.iter_mut() {
            *term = simplify_candidate(state, *term);
        }
        lhs_terms = {
            let ctx = get_context(state);
            crate::cancel_normalization_support::reflatten_signed_terms_for_cancel(ctx, lhs_terms)
        };
        rhs_terms = {
            let ctx = get_context(state);
            crate::cancel_normalization_support::reflatten_signed_terms_for_cancel(ctx, rhs_terms)
        };
    }

    if lhs_terms.len() > config.max_terms || rhs_terms.len() > config.max_terms {
        return None;
    }

    let mut usage = {
        let ctx = get_context(state);
        match_structural_definability_pairs(ctx, &lhs_terms, &rhs_terms)
    };

    let lhs_node_counts = {
        let ctx = get_context(state);
        signed_term_node_counts(&lhs_terms, |t| count_nodes_matching(ctx, t, |_| true))
    };
    let rhs_node_counts = {
        let ctx = get_context(state);
        signed_term_node_counts(&rhs_terms, |t| count_nodes_matching(ctx, t, |_| true))
    };

    match_semantic_pairs_with_usage_and_node_counts(
        &lhs_terms,
        &rhs_terms,
        &lhs_node_counts,
        &rhs_node_counts,
        &mut usage,
        config.max_nodes,
        |lt, rt| prove_equal(state, lt, rt),
    );

    if usage.cancelled == 0 {
        return None;
    }

    let new_lhs_terms = collect_unmatched_signed_terms(&lhs_terms, &usage.lhs_used);
    let new_rhs_terms = collect_unmatched_signed_terms(&rhs_terms, &usage.rhs_used);

    let (new_lhs, new_rhs) = {
        let ctx = get_context_mut(state);
        (
            rebuild_from_signed_terms(ctx, &new_lhs_terms),
            rebuild_from_signed_terms(ctx, &new_rhs_terms),
        )
    };

    Some(SemanticCancelResult {
        new_lhs,
        new_rhs,
        cancelled_count: usage.cancelled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;
    use cas_formatter::render_expr;
    use cas_parser::parse;

    #[test]
    fn structural_match_marks_expected_pair() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let y = parse("y", &mut ctx).expect("parse");
        let lhs = vec![
            (x, true, OriginSafety::DefinabilityPreserving),
            (y, true, OriginSafety::DefinabilityPreserving),
        ];
        let rhs = vec![(y, true, OriginSafety::DefinabilityPreserving)];
        let usage = match_structural_definability_pairs(&ctx, &lhs, &rhs);
        assert_eq!(usage.cancelled, 1);
        assert_eq!(usage.lhs_used, vec![false, true]);
        assert_eq!(usage.rhs_used, vec![true]);
    }

    #[test]
    fn collect_signed_terms_with_uniform_safety_applies_tag() {
        let mut ctx = Context::new();
        let expr = parse("a - b", &mut ctx).expect("parse");
        let terms = collect_signed_terms_with_uniform_safety(
            &ctx,
            expr,
            OriginSafety::NeedsAnalyticConditions,
        );
        assert_eq!(terms.len(), 2);
        assert!(terms
            .iter()
            .all(|(_, _, s)| *s == OriginSafety::NeedsAnalyticConditions));
    }

    #[test]
    fn signed_term_node_counts_uses_callback() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let y = parse("y", &mut ctx).expect("parse");
        let terms = vec![
            (x, true, OriginSafety::DefinabilityPreserving),
            (y, true, OriginSafety::DefinabilityPreserving),
        ];
        let counts = signed_term_node_counts(&terms, |_id| 7usize);
        assert_eq!(counts, vec![7, 7]);
    }

    #[test]
    fn map_signed_terms_in_place_updates_ids() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let y = parse("y", &mut ctx).expect("parse");
        let mut terms = vec![(x, true, OriginSafety::DefinabilityPreserving)];
        map_signed_terms_in_place(&mut terms, |_old| y);
        assert_eq!(terms[0].0, y);
    }

    #[test]
    fn bidirectional_overlap_expansion_noop_without_overlap() {
        let mut ctx = Context::new();
        let lhs = parse("(x+1)^2", &mut ctx).expect("parse");
        let rhs = parse("z", &mut ctx).expect("parse");
        let mut lhs_terms = vec![(lhs, true, OriginSafety::DefinabilityPreserving)];
        let mut rhs_terms = vec![(rhs, true, OriginSafety::DefinabilityPreserving)];
        let changed = try_bidirectional_overlap_expansion_with(
            &mut ctx,
            &mut lhs_terms,
            &mut rhs_terms,
            |_c, e| e,
        );
        assert!(!changed);
    }

    #[test]
    fn semantic_match_uses_callback_for_remaining_terms() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let lhs = vec![(x, true, OriginSafety::NeedsAnalyticConditions)];
        let rhs = vec![(x, true, OriginSafety::NeedsAnalyticConditions)];
        let mut usage = CancelUsage {
            lhs_used: vec![false],
            rhs_used: vec![false],
            cancelled: 0,
        };
        let lhs_counts = vec![1];
        let rhs_counts = vec![1];
        match_semantic_pairs_with_usage_and_node_counts(
            &lhs,
            &rhs,
            &lhs_counts,
            &rhs_counts,
            &mut usage,
            10,
            |_l, _r| true,
        );
        assert_eq!(usage.cancelled, 1);
    }

    #[test]
    fn semantic_cancel_pipeline_with_state_cancels_structural_pair() {
        struct DummyState {
            context: Context,
        }

        let mut state = DummyState {
            context: Context::new(),
        };
        let lhs = parse("x + y", &mut state.context).expect("parse lhs");
        let rhs = parse("y", &mut state.context).expect("parse rhs");

        let result = try_cancel_additive_terms_semantic_with_state(
            &mut state,
            lhs,
            rhs,
            SemanticCancelConfig {
                max_terms: 12,
                max_nodes: 64,
            },
            |s| &s.context,
            |s| &mut s.context,
            |_s, term| term,
            |_ctx, expr| expr,
            |_s, _lt, _rt| false,
        )
        .expect("expected structural cancellation");

        assert_eq!(result.cancelled_count, 1);
        assert_eq!(render_expr(&state.context, result.new_lhs), "x");
        assert_eq!(render_expr(&state.context, result.new_rhs), "0");
    }
}
