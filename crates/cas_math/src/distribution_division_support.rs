//! Support for selective division distribution rewrites.
//!
//! Rewrites `(a + b + ...)/d` into `a/d + b/d + ...` only when a reduction
//! heuristic predicts simplification opportunity.

use crate::distribution_guard_support::estimate_division_distribution_simplification_reduction;
use crate::expr_destructure::as_div;
use crate::expr_nary::{build_balanced_add, AddView, Sign};
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivisionDistributionRewrite {
    pub rewritten: ExprId,
}

/// Plan division distribution rewrite when heuristics predict simplification.
///
/// Pattern:
/// - `(t1 + t2 + ...)/d -> t1/d + t2/d + ...`
///
/// Guards:
/// - numerator must be additive (>= 2 terms),
/// - at least one term must show simplification reduction potential,
/// - post-rewrite complexity must be bounded by predicted reduction.
pub fn try_rewrite_div_distribution_simplifying_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DivisionDistributionRewrite> {
    let (numer, denom) = as_div(ctx, expr)?;
    let num_view = AddView::from_expr(ctx, numer);

    if num_view.terms.len() <= 1 {
        return None;
    }

    let mut total_reduction: usize = 0;
    let mut any_simplifies = false;
    for &(term, _sign) in &num_view.terms {
        let red = estimate_division_distribution_simplification_reduction(ctx, term, denom);
        if red > 0 {
            any_simplifies = true;
            total_reduction += red;
        }
    }

    if !any_simplifies {
        return None;
    }

    let new_terms: Vec<ExprId> = num_view
        .terms
        .iter()
        .map(|&(term, sign)| {
            let div_term = ctx.add(Expr::Div(term, denom));
            match sign {
                Sign::Pos => div_term,
                Sign::Neg => ctx.add(Expr::Neg(div_term)),
            }
        })
        .collect();
    let rewritten = build_balanced_add(ctx, &new_terms);

    let old_complexity = cas_ast::count_nodes(ctx, expr);
    let new_complexity = cas_ast::count_nodes(ctx, rewritten);
    if new_complexity > old_complexity + total_reduction {
        return None;
    }

    Some(DivisionDistributionRewrite { rewritten })
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_div_distribution_simplifying_expr;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn rewrites_when_term_matches_denominator() {
        let mut ctx = Context::new();
        let expr = parse("(2*x + 4)/2", &mut ctx).expect("parse");
        let out = try_rewrite_div_distribution_simplifying_expr(&mut ctx, expr);
        assert!(out.is_some());
    }

    #[test]
    fn skips_non_additive_numerator() {
        let mut ctx = Context::new();
        let expr = parse("x/y", &mut ctx).expect("parse");
        let out = try_rewrite_div_distribution_simplifying_expr(&mut ctx, expr);
        assert!(out.is_none());
    }
}
