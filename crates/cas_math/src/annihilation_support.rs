//! Structural helpers for additive annihilation rewrites.

use crate::expr_relations::poly_equal;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoTermAnnihilationMatch {
    pub left_term: ExprId,
    pub right_term: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnihilationRewriteKind {
    TwoTerm,
    HoldSum,
}

fn collect_additive_term_signs_preserving_hold(ctx: &Context, expr: ExprId) -> Vec<(ExprId, bool)> {
    let mut out = Vec::new();
    let mut stack = vec![(expr, false)];

    while let Some((id, is_neg)) = stack.pop() {
        match ctx.get(id) {
            Expr::Add(left, right) => {
                stack.push((*right, is_neg));
                stack.push((*left, is_neg));
            }
            Expr::Sub(left, right) => {
                stack.push((*right, !is_neg));
                stack.push((*left, is_neg));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, !is_neg));
            }
            _ => out.push((id, is_neg)),
        }
    }

    out
}

/// Detect `X - X` style annihilation on two-term additive expressions.
///
/// Returns both original terms so callers can apply policy gates
/// (e.g. domain/definedness checks) before committing to `0`.
pub fn find_two_term_annihilation_match(
    ctx: &Context,
    expr: ExprId,
) -> Option<TwoTermAnnihilationMatch> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let terms = collect_additive_term_signs_preserving_hold(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let (left_term, left_neg) = terms[0];
    let (right_term, right_neg) = terms[1];
    if left_neg == right_neg {
        return None;
    }

    let left_unwrapped = cas_ast::hold::unwrap_hold(ctx, left_term);
    let right_unwrapped = cas_ast::hold::unwrap_hold(ctx, right_term);
    if !poly_equal(ctx, left_unwrapped, right_unwrapped) {
        return None;
    }

    Some(TwoTermAnnihilationMatch {
        left_term,
        right_term,
    })
}

/// Determine whether an additive expression should be rewritten to `0`
/// by annihilation (`X-X` or `__hold(sum)-sum`), with optional strict-domain
/// blocking for two-term matches.
pub fn should_rewrite_annihilation_to_zero_with(
    ctx: &Context,
    expr: ExprId,
    strict_domain: bool,
    has_undefined_risk: impl Fn(&Context, ExprId) -> bool,
) -> Option<AnnihilationRewriteKind> {
    if let Some(matched) = find_two_term_annihilation_match(ctx, expr) {
        if strict_domain
            && (has_undefined_risk(ctx, matched.left_term)
                || has_undefined_risk(ctx, matched.right_term))
        {
            return None;
        }
        return Some(AnnihilationRewriteKind::TwoTerm);
    }

    if is_hold_sum_annihilation(ctx, expr) {
        return Some(AnnihilationRewriteKind::HoldSum);
    }

    None
}

/// Mode-flags adapter for [`should_rewrite_annihilation_to_zero_with`].
///
/// `assume_mode` takes precedence over `strict_mode` when both are set.
pub fn should_rewrite_annihilation_to_zero_with_mode_flags(
    ctx: &Context,
    expr: ExprId,
    assume_mode: bool,
    strict_mode: bool,
    has_undefined_risk: impl Fn(&Context, ExprId) -> bool,
) -> Option<AnnihilationRewriteKind> {
    let strict_domain = if assume_mode { false } else { strict_mode };
    should_rewrite_annihilation_to_zero_with(ctx, expr, strict_domain, has_undefined_risk)
}

fn terms_cancel(
    ctx: &Context,
    held_term: ExprId,
    held_neg: bool,
    other_term: ExprId,
    other_neg: bool,
) -> bool {
    if other_neg != held_neg {
        return poly_equal(ctx, held_term, other_term);
    }

    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(held_term), ctx.get(other_term)) {
        return n1 == &-n2.clone();
    }
    false
}

/// Detect `__hold(sum) - sum` annihilation patterns in additive expressions.
pub fn is_hold_sum_annihilation(ctx: &Context, expr: ExprId) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let terms = collect_additive_term_signs_preserving_hold(ctx, expr);
    if terms.len() < 2 {
        return false;
    }

    for (idx, (term, is_neg)) in terms.iter().enumerate() {
        if *is_neg || !cas_ast::hold::is_hold(ctx, *term) {
            continue;
        }

        let held_content = cas_ast::hold::unwrap_hold(ctx, *term);
        let held_terms = collect_additive_term_signs_preserving_hold(ctx, held_content);
        let other_terms: Vec<(ExprId, bool)> = terms
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, pair)| *pair)
            .collect();

        if held_terms.len() != other_terms.len() {
            continue;
        }

        let mut used = vec![false; other_terms.len()];
        let mut all_cancel = true;

        for (held_term, held_neg) in held_terms {
            let mut found = false;

            for (j, (other_term, other_neg)) in other_terms.iter().enumerate() {
                if used[j] {
                    continue;
                }
                if terms_cancel(ctx, held_term, held_neg, *other_term, *other_neg) {
                    used[j] = true;
                    found = true;
                    break;
                }
            }

            if !found {
                all_cancel = false;
                break;
            }
        }

        if all_cancel && used.iter().all(|u| *u) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::{
        find_two_term_annihilation_match, is_hold_sum_annihilation,
        should_rewrite_annihilation_to_zero_with,
        should_rewrite_annihilation_to_zero_with_mode_flags, AnnihilationRewriteKind,
    };
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn finds_two_term_annihilation_match_for_sub() {
        let mut ctx = Context::new();
        let expr = parse("x - x", &mut ctx).expect("parse");
        let matched = find_two_term_annihilation_match(&ctx, expr);
        assert!(matched.is_some());
    }

    #[test]
    fn rejects_two_term_when_signs_do_not_cancel() {
        let mut ctx = Context::new();
        let expr = parse("x + x", &mut ctx).expect("parse");
        assert!(find_two_term_annihilation_match(&ctx, expr).is_none());
    }

    #[test]
    fn detects_hold_sum_annihilation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x, y));
        let held = cas_ast::hold::wrap_hold(&mut ctx, sum);
        let neg_x = ctx.add(Expr::Neg(x));
        let neg_y = ctx.add(Expr::Neg(y));
        let rhs = ctx.add(Expr::Add(neg_x, neg_y));
        let expr = ctx.add(Expr::Add(held, rhs));

        assert!(is_hold_sum_annihilation(&ctx, expr));
    }

    #[test]
    fn rejects_non_matching_hold_sum() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let sum = ctx.add(Expr::Add(x, y));
        let held = cas_ast::hold::wrap_hold(&mut ctx, sum);
        let neg_x = ctx.add(Expr::Neg(x));
        let neg_z = ctx.add(Expr::Neg(z));
        let rhs = ctx.add(Expr::Add(neg_x, neg_z));
        let expr = ctx.add(Expr::Add(held, rhs));

        assert!(!is_hold_sum_annihilation(&ctx, expr));
    }

    #[test]
    fn strict_domain_blocks_two_term_when_risky() {
        let mut ctx = Context::new();
        let expr = parse("x - x", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse x");

        let blocked =
            should_rewrite_annihilation_to_zero_with(&ctx, expr, true, |_ctx, term| term == x);
        assert!(blocked.is_none());
    }

    #[test]
    fn non_strict_allows_two_term_when_risky() {
        let mut ctx = Context::new();
        let expr = parse("x - x", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse x");

        let kind =
            should_rewrite_annihilation_to_zero_with(&ctx, expr, false, |_ctx, term| term == x);
        assert_eq!(kind, Some(AnnihilationRewriteKind::TwoTerm));
    }

    #[test]
    fn hold_sum_always_rewrites() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x, y));
        let held = cas_ast::hold::wrap_hold(&mut ctx, sum);
        let neg_x = ctx.add(Expr::Neg(x));
        let neg_y = ctx.add(Expr::Neg(y));
        let rhs = ctx.add(Expr::Add(neg_x, neg_y));
        let expr = ctx.add(Expr::Add(held, rhs));

        let kind = should_rewrite_annihilation_to_zero_with(&ctx, expr, true, |_ctx, _| true);
        assert_eq!(kind, Some(AnnihilationRewriteKind::HoldSum));
    }

    #[test]
    fn mode_flags_prioritize_assume_over_strict() {
        let mut ctx = Context::new();
        let expr = parse("x - x", &mut ctx).expect("parse");

        let out = should_rewrite_annihilation_to_zero_with_mode_flags(
            &ctx,
            expr,
            true,
            true,
            |_core_ctx, _term| true,
        );
        assert_eq!(out, Some(AnnihilationRewriteKind::TwoTerm));
    }
}
