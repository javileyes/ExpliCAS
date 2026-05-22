//! Structural support for nested-fraction simplification.
//!
//! Rewrites complex fraction layouts by multiplying numerator and denominator
//! by a common product of discovered denominator atoms.

use crate::build::mul2_raw;
use crate::expr_rewrite::{collect_denominators, count_div_nodes, distribute, smart_mul};
use cas_ast::views::RationalFnView;
use cas_ast::{Context, Expr, ExprId};
use num_traits::Signed;

#[derive(Debug, Clone, Copy)]
pub struct NestedFractionRewrite {
    pub rewritten: ExprId,
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
    let num_has_inner_denoms = !num_denoms.is_empty() || contains_reciprocal_power(ctx, num);
    let den_has_inner_denoms = !den_denoms.is_empty() || contains_reciprocal_power(ctx, den);

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

    let new_num = if num_has_inner_denoms {
        distribute(ctx, num, multiplier)
    } else {
        smart_mul(ctx, num, multiplier)
    };
    let new_den = if den_has_inner_denoms {
        distribute(ctx, den, multiplier)
    } else {
        smart_mul(ctx, den, multiplier)
    };
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
    })
}

fn contains_reciprocal_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            matches!(ctx.get(*exp), Expr::Number(n) if n.is_negative())
                || contains_reciprocal_power(ctx, *base)
                || contains_reciprocal_power(ctx, *exp)
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            contains_reciprocal_power(ctx, *left) || contains_reciprocal_power(ctx, *right)
        }
        Expr::Div(_, _) => true,
        Expr::Neg(inner) | Expr::Hold(inner) => contains_reciprocal_power(ctx, *inner),
        Expr::Function(_, args) => args.iter().any(|arg| contains_reciprocal_power(ctx, *arg)),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| contains_reciprocal_power(ctx, *entry)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_simplify_nested_fraction_expr;
    use crate::expr_rewrite::count_div_nodes;
    use cas_ast::Context;
    use cas_ast::Expr;
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

    #[test]
    fn preserves_clean_outer_denominator_as_product() {
        let mut ctx = Context::new();
        let expr = parse("(1/(x+2))/(x+3)", &mut ctx).expect("parse");
        let rw = try_rewrite_simplify_nested_fraction_expr(&mut ctx, expr).expect("rewrite");
        let Expr::Div(_, den) = ctx.get(rw.rewritten).clone() else {
            panic!("expected fraction result");
        };
        assert!(
            matches!(ctx.get(den), Expr::Mul(_, _)),
            "clean outer denominator should remain factored, got {:?}",
            ctx.get(den)
        );
    }

    #[test]
    fn distributes_reciprocal_power_outer_denominator() {
        let mut ctx = Context::new();
        let expr = parse("(2/sqrt(u) + (u+1)/u)/(u^(-1/2)+1)", &mut ctx).expect("parse");
        let rw = try_rewrite_simplify_nested_fraction_expr(&mut ctx, expr).expect("rewrite");
        let Expr::Div(_, den) = ctx.get(rw.rewritten).clone() else {
            panic!("expected fraction result");
        };
        assert!(
            matches!(ctx.get(den), Expr::Add(_, _)),
            "reciprocal-power denominator should be distributed, got {:?}",
            ctx.get(den)
        );
    }
}
