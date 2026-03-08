//! Support for pulling constant factors out of fraction numerators.

use crate::build::mul2_raw;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy)]
pub enum PullConstantFromFractionKind {
    PullConstant,
    PullNegation,
}

#[derive(Debug, Clone, Copy)]
pub struct PullConstantFromFractionRewrite {
    pub rewritten: ExprId,
    pub kind: PullConstantFromFractionKind,
}

/// Rewrite:
/// - `(c*x)/y -> c*(x/y)`
/// - `(x*c)/y -> c*(x/y)`
/// - `(-x)/y -> -1*(x/y)`
pub fn try_rewrite_pull_constant_from_fraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PullConstantFromFractionRewrite> {
    let (n, d) = if let Expr::Div(n, d) = ctx.get(expr) {
        (*n, *d)
    } else {
        return None;
    };

    enum NumShape {
        Mul(ExprId, ExprId),
        Neg(ExprId),
        Other,
    }
    let num_shape = match ctx.get(n) {
        Expr::Mul(l, r) => NumShape::Mul(*l, *r),
        Expr::Neg(inner) => NumShape::Neg(*inner),
        _ => NumShape::Other,
    };

    if let NumShape::Mul(l, r) = num_shape {
        let l_is_const = matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_));
        let r_is_const = matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_));
        if l_is_const {
            let div = ctx.add(Expr::Div(r, d));
            return Some(PullConstantFromFractionRewrite {
                rewritten: mul2_raw(ctx, l, div),
                kind: PullConstantFromFractionKind::PullConstant,
            });
        }
        if r_is_const {
            let div = ctx.add(Expr::Div(l, d));
            return Some(PullConstantFromFractionRewrite {
                rewritten: mul2_raw(ctx, r, div),
                kind: PullConstantFromFractionKind::PullConstant,
            });
        }
    }

    if let NumShape::Neg(inner) = num_shape {
        let minus_one = ctx.num(-1);
        let div = ctx.add(Expr::Div(inner, d));
        return Some(PullConstantFromFractionRewrite {
            rewritten: mul2_raw(ctx, minus_one, div),
            kind: PullConstantFromFractionKind::PullNegation,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_pull_constant_from_fraction_expr;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn rewrites_const_times_term_over_den() {
        let mut ctx = Context::new();
        let expr = parse("(2*x)/y", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_pull_constant_from_fraction_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("2*(x/y)", &mut ctx).expect("expected");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_negated_numerator() {
        let mut ctx = Context::new();
        let expr = parse("(-x)/y", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_pull_constant_from_fraction_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("(-1)*(x/y)", &mut ctx).expect("expected");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }
}
