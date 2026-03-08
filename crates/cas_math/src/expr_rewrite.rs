//! AST rewrite helpers for algebraic distribution and denominator analysis.

use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Signed};

/// Count `Div(_, _)` nodes in an expression tree.
pub fn count_div_nodes(ctx: &Context, expr: ExprId) -> usize {
    cas_ast::traversal::count_nodes_matching(ctx, expr, |node| matches!(node, Expr::Div(_, _)))
}

/// Build `Mul(a, b)` while collapsing trivial multiplication by `1`.
pub fn smart_mul(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if is_one_expr(ctx, a) {
        return b;
    }
    if is_one_expr(ctx, b) {
        return a;
    }
    ctx.add_raw(Expr::Mul(a, b))
}

/// Distribute `multiplier` over `target`, favoring denominator-clearing paths.
pub fn distribute(ctx: &mut Context, target: ExprId, multiplier: ExprId) -> ExprId {
    enum Shape {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Mul(ExprId, ExprId),
        Div(ExprId, ExprId),
        Other,
    }

    let shape = match ctx.get(target) {
        Expr::Add(l, r) => Shape::Add(*l, *r),
        Expr::Sub(l, r) => Shape::Sub(*l, *r),
        Expr::Mul(l, r) => Shape::Mul(*l, *r),
        Expr::Div(l, r) => Shape::Div(*l, *r),
        _ => Shape::Other,
    };

    match shape {
        Shape::Add(l, r) => {
            let dl = distribute(ctx, l, multiplier);
            let dr = distribute(ctx, r, multiplier);
            ctx.add(Expr::Add(dl, dr))
        }
        Shape::Sub(l, r) => {
            let dl = distribute(ctx, l, multiplier);
            let dr = distribute(ctx, r, multiplier);
            ctx.add(Expr::Sub(dl, dr))
        }
        Shape::Mul(l, r) => {
            let l_denoms = collect_denominators(ctx, l);
            if !l_denoms.is_empty() {
                let dl = distribute(ctx, l, multiplier);
                return distribute(ctx, r, dl);
            }
            let r_denoms = collect_denominators(ctx, r);
            if !r_denoms.is_empty() {
                let dr = distribute(ctx, r, multiplier);
                return distribute(ctx, l, dr);
            }
            smart_mul(ctx, target, multiplier)
        }
        Shape::Div(l, r) => {
            if let Some(quotient) = get_quotient(ctx, multiplier, r) {
                return smart_mul(ctx, l, quotient);
            }
            let div_expr = ctx.add(Expr::Div(l, r));
            smart_mul(ctx, div_expr, multiplier)
        }
        Shape::Other => smart_mul(ctx, target, multiplier),
    }
}

/// Collect denominator candidates from explicit `Div` nodes under `expr`.
pub fn collect_denominators(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut denoms = Vec::new();
    match ctx.get(expr) {
        Expr::Div(_, den) => {
            denoms.push(*den);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            denoms.extend(collect_denominators(ctx, *l));
            denoms.extend(collect_denominators(ctx, *r));
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_negative() {
                    // b^-k = 1/b^k, but this helper currently tracks only explicit Div nodes.
                }
            }
            denoms.extend(collect_denominators(ctx, *base));
        }
        _ => {}
    }
    denoms
}

fn is_one_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn get_quotient(ctx: &mut Context, dividend: ExprId, divisor: ExprId) -> Option<ExprId> {
    if dividend == divisor {
        return Some(ctx.num(1));
    }

    let mul_parts = match ctx.get(dividend) {
        Expr::Mul(l, r) => Some((*l, *r)),
        _ => None,
    };

    if let Some((l, r)) = mul_parts {
        if let Some(q) = get_quotient(ctx, l, divisor) {
            return Some(smart_mul(ctx, q, r));
        }
        if let Some(q) = get_quotient(ctx, r, divisor) {
            return Some(smart_mul(ctx, l, q));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::ordering::compare_expr;
    use std::cmp::Ordering;

    #[test]
    fn count_div_nodes_counts_nested_divisions() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let d = ctx.var("d");

        let left = ctx.add(Expr::Div(a, b));
        let right = ctx.add(Expr::Div(c, d));
        let expr = ctx.add(Expr::Add(left, right));

        assert_eq!(count_div_nodes(&ctx, expr), 2);
    }

    #[test]
    fn collect_denominators_extracts_divisors() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let d = ctx.var("d");

        let left = ctx.add(Expr::Div(a, b));
        let right = ctx.add(Expr::Div(c, d));
        let expr = ctx.add(Expr::Add(left, right));
        let denoms = collect_denominators(&ctx, expr);

        assert_eq!(denoms.len(), 2);
        assert_eq!(denoms[0], b);
        assert_eq!(denoms[1], d);
    }

    #[test]
    fn distribute_over_add_builds_two_products() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let m = ctx.var("m");
        let sum = ctx.add(Expr::Add(a, b));

        let out = distribute(&mut ctx, sum, m);
        match ctx.get(out) {
            Expr::Add(l, r) => {
                assert!(matches!(ctx.get(*l), Expr::Mul(_, _)));
                assert!(matches!(ctx.get(*r), Expr::Mul(_, _)));
            }
            _ => panic!("expected additive distributed form"),
        }
    }

    #[test]
    fn distribute_division_cancels_when_multiplier_contains_denominator() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let frac = ctx.add(Expr::Div(a, b));
        let multiplier = ctx.add_raw(Expr::Mul(b, c));
        let out = distribute(&mut ctx, frac, multiplier);
        let expected = ctx.add_raw(Expr::Mul(a, c));

        assert_eq!(compare_expr(&ctx, out, expected), Ordering::Equal);
    }
}
