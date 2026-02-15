use crate::define_rule;
use crate::helpers::as_sub;
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use cas_ast::Expr;
use std::cmp::Ordering;

// ──────────────────────────────────────────────────────────────────────
// CancelCommonAdditiveTermsRule: Sub(Add(..., T, ...), T) → Add(...)
//
// Implements the "algebra 101" identity: when subtracting an expression
// from a sum that contains a structurally-identical term, cancel it.
//
// This is strictly reductive (always decreases term count) — NO loop risk.
//
// Patterns handled:
//   Sub(Add(A, B), B)           → A
//   Sub(Add(A, B, C), Add(B, C))→ A
//   Sub(term, Add(term, X))     → Neg(X)
//
// Uses structural comparison (compare_expr) for term matching, which
// requires prior canonicalization (e.g. CanonicalizeRationalDivRule)
// to be effective across different representations of the same value.
// ──────────────────────────────────────────────────────────────────────
define_rule!(
    CancelCommonAdditiveTermsRule,
    "Cancel Common Additive Terms",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let (lhs, rhs) = as_sub(ctx, expr)?;

        // Collect flattened additive terms from both sides
        let mut lhs_terms = Vec::new();
        let mut rhs_terms = Vec::new();
        collect_additive_terms(ctx, lhs, true, &mut lhs_terms);
        collect_additive_terms(ctx, rhs, true, &mut rhs_terms);

        // Need at least one term on each side to cancel
        if lhs_terms.is_empty() || rhs_terms.is_empty() {
            return None;
        }

        // Mark which terms have been cancelled
        let mut lhs_used = vec![false; lhs_terms.len()];
        let mut rhs_used = vec![false; rhs_terms.len()];
        let mut cancelled = 0;

        // O(n*m) matching — fine for typical term counts
        for (ri, (r_term, r_pos)) in rhs_terms.iter().enumerate() {
            if rhs_used[ri] {
                continue;
            }
            for (li, (l_term, l_pos)) in lhs_terms.iter().enumerate() {
                if lhs_used[li] {
                    continue;
                }
                // Both must have the same sign polarity AND match structurally
                if r_pos == l_pos
                    && compare_expr(ctx, *l_term, *r_term) == Ordering::Equal
                {
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

        // Rebuild both sides without cancelled terms
        let new_lhs_terms: Vec<(cas_ast::ExprId, bool)> = lhs_terms
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !lhs_used[*i])
            .map(|(_, t)| t)
            .collect();
        let new_rhs_terms: Vec<(cas_ast::ExprId, bool)> = rhs_terms
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !rhs_used[*i])
            .map(|(_, t)| t)
            .collect();

        let new_lhs = rebuild_from_terms(ctx, &new_lhs_terms);
        let new_rhs = rebuild_from_terms(ctx, &new_rhs_terms);
        let new_expr = ctx.add(Expr::Sub(new_lhs, new_rhs));

        Some(
            Rewrite::new(new_expr)
                .desc("Cancel common terms from both sides"),
        )
    }
);

/// Collect additive terms from an expression, flattening Add/Sub.
/// Each term is (ExprId, is_positive).
pub(crate) fn collect_additive_terms(
    ctx: &cas_ast::Context,
    id: cas_ast::ExprId,
    positive: bool,
    out: &mut Vec<(cas_ast::ExprId, bool)>,
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

/// Rebuild an expression from a list of (term, is_positive) pairs.
pub(crate) fn rebuild_from_terms(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, bool)],
) -> cas_ast::ExprId {
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

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CancelCommonAdditiveTermsRule));
}

// ──────────────────────────────────────────────────────────────────────
// Equation-level cancellation (used by solver pre-solve pipeline)
// ──────────────────────────────────────────────────────────────────────
// The CancelCommonAdditiveTermsRule above targets Sub nodes, but in the
// simplifier CanonicalizeNegationRule converts Sub → Add(Neg) first,
// so the rule rarely fires during normal simplification.
//
// The solver needs equation-level cancellation that compares terms from
// LHS and RHS as a *pair*, which is fundamentally different from a
// single-expression rule. This public function provides that.

/// Cancel common additive terms between two expression trees.
/// Returns `Some((new_lhs, new_rhs))` if any terms were cancelled.
pub(crate) fn cancel_common_additive_terms(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
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

    Some((
        rebuild_from_terms(ctx, &new_lhs_terms),
        rebuild_from_terms(ctx, &new_rhs_terms),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, Expr};

    #[test]
    fn test_cancel_simple() {
        // (x^2 + y) - y → x^2
        let mut ctx = Context::new();
        let rule = CancelCommonAdditiveTermsRule;
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x2 = ctx.add(Expr::Pow(x, two));
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x2, y));
        let expr = ctx.add(Expr::Sub(sum, y));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Result should be Sub(x^2, 0) which simplifies to x^2
        // But at the rule level, it's x^2 - 0 or just x^2
        // The rebuilt LHS is x^2, RHS is 0
        if let Expr::Sub(l, r) = ctx.get(rewrite.new_expr) {
            assert!(matches!(ctx.get(*l), Expr::Pow(_, _)));
            assert!(matches!(ctx.get(*r), Expr::Number(_)));
        }
    }

    #[test]
    fn test_no_cancel_different_terms() {
        // (x + y) - z → no cancellation
        let mut ctx = Context::new();
        let rule = CancelCommonAdditiveTermsRule;
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let sum = ctx.add(Expr::Add(x, y));
        let expr = ctx.add(Expr::Sub(sum, z));
        assert!(rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .is_none());
    }

    #[test]
    fn test_cancel_with_duplicates() {
        // (a + b + b) - b → a + b
        let mut ctx = Context::new();
        let rule = CancelCommonAdditiveTermsRule;
        let a = ctx.var("a");
        let b = ctx.var("b");
        let b2 = ctx.var("b"); // Another b node
        let ab = ctx.add(Expr::Add(a, b));
        let abb = ctx.add(Expr::Add(ab, b2));
        let expr = ctx.add(Expr::Sub(abb, b));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should cancel one b, leaving a + b
        assert!(rewrite.new_expr != expr);
    }
}
