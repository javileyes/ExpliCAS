//! Pattern detector for small expandable structures.

use cas_ast::{Context, Expr, ExprId};

/// Check whether an expression tree contains an expandable product or power:
/// - `Mul(_, Add/Sub)` or `Mul(Add/Sub, _)`
/// - `Pow(Add/Sub, integer >= 2)`
///
/// Search depth is capped to keep this predicate cheap.
pub fn contains_expandable_small_depth(ctx: &Context, expr: ExprId) -> bool {
    contains_expandable_depth(ctx, expr, 0)
}

fn contains_expandable_depth(ctx: &Context, expr: ExprId, depth: usize) -> bool {
    if depth > 3 {
        return false;
    }
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            let l_is_sum = matches!(ctx.get(*l), Expr::Add(_, _) | Expr::Sub(_, _));
            let r_is_sum = matches!(ctx.get(*r), Expr::Add(_, _) | Expr::Sub(_, _));
            if l_is_sum || r_is_sum {
                return true;
            }
            contains_expandable_depth(ctx, *l, depth + 1)
                || contains_expandable_depth(ctx, *r, depth + 1)
        }
        Expr::Pow(b, e) => {
            if matches!(ctx.get(*b), Expr::Add(_, _) | Expr::Sub(_, _)) {
                if let Expr::Number(n) = ctx.get(*e) {
                    if n.is_integer() && *n >= num_rational::BigRational::from_integer(2.into()) {
                        return true;
                    }
                }
            }
            contains_expandable_depth(ctx, *b, depth + 1)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Div(l, r) => {
            contains_expandable_depth(ctx, *l, depth + 1)
                || contains_expandable_depth(ctx, *r, depth + 1)
        }
        Expr::Neg(e) => contains_expandable_depth(ctx, *e, depth + 1),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::contains_expandable_small_depth;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_mul_over_sum() {
        let mut ctx = Context::new();
        let expr = parse("a*(b+c)", &mut ctx).expect("parse");
        assert!(contains_expandable_small_depth(&ctx, expr));
    }

    #[test]
    fn detects_pow_of_sum_with_integer_exp() {
        let mut ctx = Context::new();
        let expr = parse("(x+y)^2", &mut ctx).expect("parse");
        assert!(contains_expandable_small_depth(&ctx, expr));
    }
}
