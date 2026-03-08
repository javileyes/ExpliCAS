//! Canonicalization support for reciprocal square-root forms.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalSqrtCanonRewrite {
    pub rewritten: ExprId,
}

fn contains_symbol(ctx: &Context, e: ExprId) -> bool {
    match ctx.get(e) {
        Expr::Variable(_) => true,
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        Expr::Neg(a) | Expr::Hold(a) => contains_symbol(ctx, *a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            contains_symbol(ctx, *a) || contains_symbol(ctx, *b)
        }
        Expr::Function(_, args) => args.iter().any(|&a| contains_symbol(ctx, a)),
        Expr::Matrix { data, .. } => data.iter().any(|&a| contains_symbol(ctx, a)),
    }
}

/// Canonicalize reciprocal-square-root forms:
///
/// - `1/sqrt(x) -> x^(-1/2)`
/// - `sqrt(x)/x -> x^(-1/2)`
///
/// Guarded to symbolic bases only (skip pure numeric bases).
pub fn try_rewrite_reciprocal_sqrt_canon_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ReciprocalSqrtCanonRewrite> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    // Pattern 1: 1/sqrt(x) -> x^(-1/2)
    if let Expr::Number(n) = ctx.get(num) {
        if n.is_one() {
            if let Some(base) = crate::root_forms::extract_square_root_base(ctx, den) {
                if !contains_symbol(ctx, base) {
                    return None;
                }
                let exp = ctx.add(Expr::Number(num_rational::BigRational::new(
                    (-1).into(),
                    2.into(),
                )));
                let rewritten = ctx.add(Expr::Pow(base, exp));
                return Some(ReciprocalSqrtCanonRewrite { rewritten });
            }
        }
    }

    // Pattern 2: sqrt(x)/x -> x^(-1/2)
    if let Some(sqrt_base) = crate::root_forms::extract_square_root_base(ctx, num) {
        if compare_expr(ctx, sqrt_base, den) == std::cmp::Ordering::Equal {
            if !contains_symbol(ctx, sqrt_base) {
                return None;
            }
            let exp = ctx.add(Expr::Number(num_rational::BigRational::new(
                (-1).into(),
                2.into(),
            )));
            let rewritten = ctx.add(Expr::Pow(sqrt_base, exp));
            return Some(ReciprocalSqrtCanonRewrite { rewritten });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_reciprocal_sqrt_canon_expr;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn rewrites_one_over_sqrt_symbolic() {
        let mut ctx = Context::new();
        let expr = parse("1/sqrt(x)", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse x");
        let rewrite = try_rewrite_reciprocal_sqrt_canon_expr(&mut ctx, expr).expect("rewrite");
        match ctx.get(rewrite.rewritten) {
            Expr::Pow(base, exp) => {
                assert_eq!(
                    cas_ast::ordering::compare_expr(&ctx, *base, x),
                    std::cmp::Ordering::Equal
                );
                match ctx.get(*exp) {
                    Expr::Number(n) => {
                        assert_eq!(*n.numer(), (-1).into());
                        assert_eq!(*n.denom(), 2.into());
                    }
                    other => panic!("expected numeric exponent, got {:?}", other),
                }
            }
            other => panic!("expected Pow rewrite, got {:?}", other),
        }
    }

    #[test]
    fn rewrites_sqrt_over_base_symbolic() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)/x", &mut ctx).expect("parse");
        let rewrite = try_rewrite_reciprocal_sqrt_canon_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn skips_numeric_bases() {
        let mut ctx = Context::new();
        let expr = parse("1/sqrt(2)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_reciprocal_sqrt_canon_expr(&mut ctx, expr);
        assert!(rewrite.is_none());
    }
}
