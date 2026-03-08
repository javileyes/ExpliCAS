//! Structural extraction helpers for folding `k + p/q` patterns.

use crate::expr_predicates::{is_minus_one_expr, is_one_expr};
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct FoldAddOperands {
    pub term: ExprId,
    pub numerator: ExprId,
    pub denominator: ExprId,
    pub swapped: bool,
}

/// Extract `(term, p, q)` from `term + p/q` or `p/q + term`.
///
/// Returns `None` when both operands are fractions or none is a direct `Div`.
pub fn extract_fold_add_operands(ctx: &Context, l: ExprId, r: ExprId) -> Option<FoldAddOperands> {
    if let Expr::Div(p, q) = ctx.get(r) {
        if matches!(ctx.get(l), Expr::Div(_, _)) {
            return None;
        }
        return Some(FoldAddOperands {
            term: l,
            numerator: *p,
            denominator: *q,
            swapped: false,
        });
    }

    if let Expr::Div(p, q) = ctx.get(l) {
        if matches!(ctx.get(r), Expr::Div(_, _)) {
            return None;
        }
        return Some(FoldAddOperands {
            term: r,
            numerator: *p,
            denominator: *q,
            swapped: true,
        });
    }

    None
}

/// Guard policy for `FoldAddIntoFractionRule` unit-term cases.
///
/// Returns true when the external term is `1` or `-1` and the fraction numerator
/// is constant-like, which tends to interact poorly with cancellation cycles.
pub fn should_block_fold_add_unit_constant_term(
    ctx: &Context,
    term: ExprId,
    numerator_is_constant: bool,
) -> bool {
    if !numerator_is_constant {
        return false;
    }
    is_one_expr(ctx, term) || is_minus_one_expr(ctx, term)
}

#[cfg(test)]
mod tests {
    use super::{extract_fold_add_operands, should_block_fold_add_unit_constant_term};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn extracts_term_plus_fraction() {
        let mut ctx = Context::new();
        let l = parse("x", &mut ctx).expect("parse");
        let r = parse("a/b", &mut ctx).expect("parse");
        let got = extract_fold_add_operands(&ctx, l, r).expect("match");
        assert_eq!(got.term, l);
        assert!(!got.swapped);
    }

    #[test]
    fn extracts_fraction_plus_term_swapped() {
        let mut ctx = Context::new();
        let l = parse("a/b", &mut ctx).expect("parse");
        let r = parse("x", &mut ctx).expect("parse");
        let got = extract_fold_add_operands(&ctx, l, r).expect("match");
        assert_eq!(got.term, r);
        assert!(got.swapped);
    }

    #[test]
    fn rejects_two_fractions() {
        let mut ctx = Context::new();
        let l = parse("a/b", &mut ctx).expect("parse");
        let r = parse("c/d", &mut ctx).expect("parse");
        assert!(extract_fold_add_operands(&ctx, l, r).is_none());
    }

    #[test]
    fn blocks_unit_term_with_constant_numerator() {
        let mut ctx = Context::new();
        let one = parse("1", &mut ctx).expect("parse");
        assert!(should_block_fold_add_unit_constant_term(&ctx, one, true));
    }

    #[test]
    fn allows_non_unit_or_nonconstant_numerator() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let one = parse("1", &mut ctx).expect("parse");
        assert!(!should_block_fold_add_unit_constant_term(&ctx, x, true));
        assert!(!should_block_fold_add_unit_constant_term(&ctx, one, false));
    }
}
