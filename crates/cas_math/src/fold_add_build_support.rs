//! Builders for folding `k + p/q` into a single fraction.

use crate::build::mul2_raw;
use cas_ast::{count_nodes, Context, Expr, ExprId};

/// Build `(term*q + p)/q` and enforce the same anti-growth guard used by engine.
pub fn try_build_fold_add_fraction_rewrite(
    ctx: &mut Context,
    original_expr: ExprId,
    term: ExprId,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<ExprId> {
    let term_times_den = mul2_raw(ctx, term, denominator);
    let new_num = ctx.add(Expr::Add(term_times_den, numerator));
    let new_expr = ctx.add(Expr::Div(new_num, denominator));

    let old_nodes = count_nodes(ctx, original_expr);
    let new_nodes = count_nodes(ctx, new_expr);
    if new_nodes > old_nodes * 3 / 2 + 2 {
        return None;
    }

    Some(new_expr)
}

#[cfg(test)]
mod tests {
    use super::try_build_fold_add_fraction_rewrite;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_ast::Expr;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn builds_fold_add_fraction_when_growth_is_reasonable() {
        let mut ctx = Context::new();
        let expr = parse("x + a/b", &mut ctx).expect("parse");
        let term = parse("x", &mut ctx).expect("parse");
        let p = parse("a", &mut ctx).expect("parse");
        let q = parse("b", &mut ctx).expect("parse");
        let got = try_build_fold_add_fraction_rewrite(&mut ctx, expr, term, p, q).expect("rewrite");
        let b = parse("b", &mut ctx).expect("parse");
        if let Expr::Div(_, den) = ctx.get(got) {
            assert_eq!(compare_expr(&ctx, *den, b), Ordering::Equal);
        } else {
            panic!("expected division rewrite");
        }
    }
}
