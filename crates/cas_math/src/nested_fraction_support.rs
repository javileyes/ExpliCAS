//! Structural support for nested-fraction simplification.
//!
//! Rewrites complex fraction layouts by multiplying numerator and denominator
//! by a common product of discovered denominator atoms.

use crate::build::mul2_raw;
use crate::expr_rewrite::{collect_denominators, count_div_nodes, distribute};
use cas_ast::views::RationalFnView;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct NestedFractionRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Try to simplify nested fractions by clearing inner denominators.
pub fn try_rewrite_simplify_nested_fraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<NestedFractionRewrite> {
    let view = RationalFnView::from(ctx, expr)?;
    let (num, den) = (view.num, view.den);

    let num_denoms = collect_denominators(ctx, num);
    let den_denoms = collect_denominators(ctx, den);
    if num_denoms.is_empty() && den_denoms.is_empty() {
        return None;
    }

    let mut all_denoms = Vec::new();
    all_denoms.extend(num_denoms);
    all_denoms.extend(den_denoms);
    if all_denoms.is_empty() {
        return None;
    }

    let mut unique_denoms: Vec<ExprId> = Vec::new();
    for d in all_denoms {
        if !unique_denoms.contains(&d) {
            unique_denoms.push(d);
        }
    }
    if unique_denoms.is_empty() {
        return None;
    }

    let (&first, rest) = unique_denoms.split_first()?;
    let multiplier = rest
        .iter()
        .copied()
        .fold(first, |acc, d| mul2_raw(ctx, acc, d));

    let new_num = distribute(ctx, num, multiplier);
    let new_den = distribute(ctx, den, multiplier);
    let new_expr = ctx.add(Expr::Div(new_num, new_den));
    if new_expr == expr {
        return None;
    }

    let old_divs = count_div_nodes(ctx, expr);
    let new_divs = count_div_nodes(ctx, new_expr);
    if new_divs >= old_divs {
        return None;
    }

    Some(NestedFractionRewrite {
        rewritten: new_expr,
        desc: "Simplify nested fraction",
    })
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_simplify_nested_fraction_expr;
    use crate::expr_rewrite::count_div_nodes;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn rewrites_simple_nested_fraction() {
        let mut ctx = Context::new();
        let expr = parse("(1/x)/(1/y)", &mut ctx).expect("parse");
        let rw = try_rewrite_simplify_nested_fraction_expr(&mut ctx, expr).expect("rewrite");
        let old_divs = count_div_nodes(&ctx, expr);
        let new_divs = count_div_nodes(&ctx, rw.rewritten);
        assert!(new_divs < old_divs);
    }

    #[test]
    fn rejects_non_nested_fraction() {
        let mut ctx = Context::new();
        let expr = parse("x/y", &mut ctx).expect("parse");
        assert!(try_rewrite_simplify_nested_fraction_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn rejects_no_improvement_rewrite() {
        let mut ctx = Context::new();
        let expr = parse("(a*b)/(c*d)", &mut ctx).expect("parse");
        assert!(try_rewrite_simplify_nested_fraction_expr(&mut ctx, expr).is_none());
    }
}
