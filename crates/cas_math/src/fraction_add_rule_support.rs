use cas_ast::ordering::compare_expr;
use cas_ast::views::FractionParts;
use cas_ast::{Context, Expr, ExprId};

use crate::expr_classify::is_trig_function;
use crate::expr_predicates::is_constant_expr;
use crate::fold_add_build_support::try_build_fold_add_fraction_rewrite;
use crate::fold_add_fraction_support::extract_fold_add_operands;
use crate::fold_add_guard_support::should_block_fold_add_rewrite;
use crate::fraction_sub_term_match_support::try_rewrite_sub_term_matches_denom_expr;
use crate::semantic_equality::SemanticEqualityChecker;

/// Planned rewrite for `k + p/q -> (k*q + p)/q` or `p/q + k`.
#[derive(Clone, Debug)]
pub struct FoldAddIntoFractionPlan {
    pub rewritten: ExprId,
    pub swapped: bool,
}

/// Planned rewrite for `a - b/a -> (a^2 - b)/a`.
#[derive(Clone, Debug)]
pub struct SubTermMatchesDenomPlan {
    pub rewritten: ExprId,
}

/// Planned rewrite for `1/(a-1) + 1/(a+1) -> 2*a/(a^2-1)`.
#[derive(Clone, Debug)]
pub struct SymmetricReciprocalSumPlan {
    pub rewritten: ExprId,
}

fn extract_unit_shift_base(ctx: &mut Context, den: ExprId) -> Option<(ExprId, i8)> {
    let one = ctx.num(1);
    match ctx.get(den) {
        Expr::Sub(base, rhs) if compare_expr(ctx, *rhs, one) == std::cmp::Ordering::Equal => {
            Some((*base, -1))
        }
        Expr::Add(lhs, rhs) if compare_expr(ctx, *lhs, one) == std::cmp::Ordering::Equal => {
            Some((*rhs, 1))
        }
        Expr::Add(lhs, rhs) if compare_expr(ctx, *rhs, one) == std::cmp::Ordering::Equal => {
            Some((*lhs, 1))
        }
        _ => None,
    }
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
    Some(FoldAddIntoFractionPlan {
        rewritten,
        swapped: ops.swapped,
    })
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
    })
}

/// Try planning `1/(a-1) + 1/(a+1) -> 2*a/(a^2-1)`.
pub fn try_plan_symmetric_reciprocal_sum_rewrite(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SymmetricReciprocalSumPlan> {
    let (l, r) = crate::expr_destructure::as_add(ctx, expr)?;

    let fp_l = FractionParts::from(&*ctx, l);
    let fp_r = FractionParts::from(&*ctx, r);
    if fp_l.sign != 1 || fp_r.sign != 1 {
        return None;
    }

    let (n1, d1, is_frac1) = fp_l.to_num_den(ctx);
    let (n2, d2, is_frac2) = fp_r.to_num_den(ctx);
    if !is_frac1 || !is_frac2 {
        return None;
    }

    let one = ctx.num(1);
    if compare_expr(ctx, n1, one) != std::cmp::Ordering::Equal
        || compare_expr(ctx, n2, one) != std::cmp::Ordering::Equal
    {
        return None;
    }

    let (base1, shift1) = extract_unit_shift_base(ctx, d1)?;
    let (base2, shift2) = extract_unit_shift_base(ctx, d2)?;
    if shift1 == shift2 {
        return None;
    }

    let checker = SemanticEqualityChecker::new(ctx);
    if !checker.are_equal(base1, base2) {
        return None;
    }

    let two = ctx.num(2);
    let base_sq = ctx.add(Expr::Pow(base1, two));
    let numerator = ctx.add(Expr::Mul(two, base1));
    let denominator = ctx.add(Expr::Sub(base_sq, one));
    let rewritten = ctx.add(Expr::Div(numerator, denominator));
    Some(SymmetricReciprocalSumPlan { rewritten })
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
    use super::{
        try_plan_fold_add_into_fraction_rewrite, try_plan_sub_term_matches_denom_rewrite,
        try_plan_symmetric_reciprocal_sum_rewrite,
    };
    use crate::expr_destructure::as_div;
    use crate::poly_compare::poly_eq;
    use crate::semantic_equality::SemanticEqualityChecker;
    use cas_ast::ordering::compare_expr;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;
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
        assert!(!plan.swapped);
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

    #[test]
    fn plans_symmetric_reciprocal_sum() {
        let mut ctx = Context::new();
        let expr = parse("1/(x-1) + 1/(x+1)", &mut ctx).expect("parse");
        let expected = parse("2*x/(x^2 - 1)", &mut ctx).expect("expected");

        let plan = try_plan_symmetric_reciprocal_sum_rewrite(&mut ctx, expr).expect("plan");
        let checker = SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(plan.rewritten, expected));
    }

    #[test]
    fn plans_symmetric_reciprocal_sum_with_fractional_base() {
        let mut ctx = Context::new();
        let expr = parse("1/((u/(u+1))-1) + 1/((u/(u+1))+1)", &mut ctx).expect("parse");
        let expected = parse("2*(u/(u+1))/((u/(u+1))^2 - 1)", &mut ctx).expect("expected");

        let plan = try_plan_symmetric_reciprocal_sum_rewrite(&mut ctx, expr).expect("plan");
        let checker = SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(plan.rewritten, expected));
    }
}
