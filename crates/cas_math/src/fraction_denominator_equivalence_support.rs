//! Helpers to detect algebraic equivalence between fraction denominators.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

/// Return true when two denominator expressions are algebraically equivalent.
///
/// Strategy:
/// - Fast structural equality check.
/// - Optional expansion-based comparison for polynomial-looking terms.
/// - Numeric polynomial zero probe on `expand(d1 - d2)` as fallback.
pub fn are_denominators_algebraically_equal_with<FExpand>(
    ctx: &mut Context,
    d1: ExprId,
    d2: ExprId,
    mut expand: FExpand,
) -> bool
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    if d1 == d2 || compare_expr(ctx, d1, d2) == Ordering::Equal {
        return true;
    }

    let worth_expanding = |id: ExprId| {
        matches!(
            ctx.get(id),
            Expr::Mul(_, _) | Expr::Add(_, _) | Expr::Sub(_, _)
        )
    };

    if !(worth_expanding(d1) || worth_expanding(d2)) {
        return false;
    }

    let d1_exp = expand(ctx, d1);
    let d2_exp = expand(ctx, d2);
    if compare_expr(ctx, d1_exp, d2_exp) == Ordering::Equal {
        return true;
    }

    let diff = ctx.add(Expr::Sub(d1, d2));
    let diff_exp = expand(ctx, diff);
    crate::numeric_eval::numeric_poly_zero_check(ctx, diff_exp)
}

#[cfg(test)]
mod tests {
    use super::are_denominators_algebraically_equal_with;
    use crate::expand_ops::expand;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_structural_equality() {
        let mut ctx = Context::new();
        let d = parse("x+1", &mut ctx).expect("parse");
        assert!(are_denominators_algebraically_equal_with(
            &mut ctx, d, d, expand
        ));
    }

    #[test]
    fn detects_algebraic_equality_after_expand() {
        let mut ctx = Context::new();
        let d1 = parse("u*(u+2)", &mut ctx).expect("parse");
        let d2 = parse("u^2+2*u", &mut ctx).expect("parse");
        assert!(are_denominators_algebraically_equal_with(
            &mut ctx, d1, d2, expand
        ));
    }

    #[test]
    fn rejects_non_equivalent_denominators() {
        let mut ctx = Context::new();
        let d1 = parse("x+1", &mut ctx).expect("parse");
        let d2 = parse("x+2", &mut ctx).expect("parse");
        assert!(!are_denominators_algebraically_equal_with(
            &mut ctx, d1, d2, expand
        ));
    }
}
