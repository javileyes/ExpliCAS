//! Structural detection for square-root-of-square patterns.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqrtSquarePattern {
    PowSquare { arg: ExprId },
    RepeatedMul { arg: ExprId },
}

/// Detect base patterns that justify `(base)^(1/2) -> abs(arg)`.
///
/// Recognized forms:
/// - `base = arg^2`
/// - `base = arg * arg`
pub fn detect_sqrt_square_pattern(ctx: &Context, base: ExprId) -> Option<SqrtSquarePattern> {
    match ctx.get(base) {
        Expr::Pow(inner_base, inner_exp) => {
            if crate::expr_predicates::is_two_expr(ctx, *inner_exp) {
                Some(SqrtSquarePattern::PowSquare { arg: *inner_base })
            } else {
                None
            }
        }
        Expr::Mul(left, right) => {
            if compare_expr(ctx, *left, *right) == Ordering::Equal {
                Some(SqrtSquarePattern::RepeatedMul { arg: *left })
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{detect_sqrt_square_pattern, SqrtSquarePattern};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_pow_square_pattern() {
        let mut ctx = Context::new();
        let base = parse("x^2", &mut ctx).expect("parse");
        let pat = detect_sqrt_square_pattern(&ctx, base);
        assert!(matches!(pat, Some(SqrtSquarePattern::PowSquare { .. })));
    }

    #[test]
    fn detects_repeated_mul_pattern() {
        let mut ctx = Context::new();
        let base = parse("x*x", &mut ctx).expect("parse");
        let pat = detect_sqrt_square_pattern(&ctx, base);
        assert!(matches!(pat, Some(SqrtSquarePattern::RepeatedMul { .. })));
    }

    #[test]
    fn rejects_non_square_pattern() {
        let mut ctx = Context::new();
        let base = parse("x*y", &mut ctx).expect("parse");
        assert!(detect_sqrt_square_pattern(&ctx, base).is_none());
    }
}
