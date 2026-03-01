use cas_ast::{Context, ExprId};

use crate::expr_classify::is_trig_function;
use crate::expr_predicates::is_constant_expr;
use crate::fold_add_build_support::try_build_fold_add_fraction_rewrite;
use crate::fold_add_fraction_support::extract_fold_add_operands;
use crate::fold_add_guard_support::should_block_fold_add_rewrite;
use crate::fraction_sub_term_match_support::try_rewrite_sub_term_matches_denom_expr;

/// Planned rewrite for `k + p/q -> (k*q + p)/q` or `p/q + k`.
#[derive(Clone, Debug)]
pub struct FoldAddIntoFractionPlan {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Planned rewrite for `a - b/a -> (a^2 - b)/a`.
#[derive(Clone, Debug)]
pub struct SubTermMatchesDenomPlan {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Try planning fold-add into fraction rewrite for one `Add(l, r)` node.
pub fn try_plan_fold_add_into_fraction_rewrite(
    ctx: &mut Context,
    expr: ExprId,
    l: ExprId,
    r: ExprId,
    inside_trig: bool,
    inside_fraction: bool,
) -> Option<FoldAddIntoFractionPlan> {
    let ops = extract_fold_add_operands(ctx, l, r)?;
    let term = ops.term;
    let p = ops.numerator;
    let q = ops.denominator;

    if should_block_fold_add_rewrite(
        ctx,
        term,
        q,
        is_constant_expr(ctx, p),
        inside_trig,
        inside_fraction,
    ) {
        return None;
    }

    let rewritten = try_build_fold_add_fraction_rewrite(ctx, expr, term, p, q)?;
    let desc = if ops.swapped {
        "Common denominator: p/q + k → (p + k·q)/q"
    } else {
        "Common denominator: k + p/q → (k·q + p)/q"
    };
    Some(FoldAddIntoFractionPlan { rewritten, desc })
}

/// Try planning `a - b/a` denominator-match rewrite.
pub fn try_plan_sub_term_matches_denom_rewrite(
    ctx: &mut Context,
    expr: ExprId,
    inside_trig: bool,
) -> Option<SubTermMatchesDenomPlan> {
    if inside_trig {
        return None;
    }
    let rewrite = try_rewrite_sub_term_matches_denom_expr(ctx, expr)?;
    Some(SubTermMatchesDenomPlan {
        rewritten: rewrite.rewritten,
        desc: rewrite.desc,
    })
}

/// Compute whether parent ancestry indicates a trig-function argument context.
pub fn is_inside_trig_ancestor_with<F>(ctx: &Context, mut has_ancestor_matching: F) -> bool
where
    F: FnMut(&Context, &mut dyn FnMut(&Context, ExprId) -> bool) -> bool,
{
    has_ancestor_matching(
        ctx,
        &mut |c, node_id| matches!(c.get(node_id), cas_ast::Expr::Function(fn_id, _) if is_trig_function(c, *fn_id)),
    )
}

#[cfg(test)]
mod tests {
    use super::{try_plan_fold_add_into_fraction_rewrite, try_plan_sub_term_matches_denom_rewrite};
    use crate::expr_destructure::as_div;
    use crate::poly_compare::poly_eq;
    use cas_ast::ordering::compare_expr;
    use cas_ast::{Context, Expr};
    use std::cmp::Ordering;

    #[test]
    fn plans_fold_add_into_fraction() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let frac = ctx.add(Expr::Div(one, y));
        let add = ctx.add(Expr::Add(x, frac));

        let plan = try_plan_fold_add_into_fraction_rewrite(&mut ctx, add, x, frac, false, false)
            .expect("plan");
        assert_eq!(plan.desc, "Common denominator: k + p/q → (k·q + p)/q");
    }

    #[test]
    fn plans_sub_term_matches_denom() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let frac = ctx.add(Expr::Div(y, x));
        let expr = ctx.add(Expr::Sub(x, frac));

        let plan = try_plan_sub_term_matches_denom_rewrite(&mut ctx, expr, false).expect("plan");

        let (_num_out, den_out) = as_div(&ctx, plan.rewritten).expect("must stay as fraction");
        assert!(poly_eq(&ctx, den_out, x));
        assert_ne!(compare_expr(&ctx, plan.rewritten, expr), Ordering::Equal);
    }
}
