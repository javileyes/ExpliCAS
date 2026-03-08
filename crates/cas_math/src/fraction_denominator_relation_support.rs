//! Denominator-relation helpers for fraction add/sub rewrites.

use crate::fraction_denominator_equivalence_support::are_denominators_algebraically_equal_with;
use crate::fraction_forms::are_denominators_opposite;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenominatorRelation {
    Same,
    Opposite,
    Different,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddDenominatorResolution {
    pub adjusted_n2: ExprId,
    pub adjusted_d2: ExprId,
    pub relation: DenominatorRelation,
}

/// Check whether two denominators should be treated as the same denominator
/// for subtraction/addition paths that do not use opposite-denominator folding.
pub fn denominators_are_same_with<FExpand>(
    ctx: &mut Context,
    d1: ExprId,
    d2: ExprId,
    expand: FExpand,
) -> bool
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    are_denominators_algebraically_equal_with(ctx, d1, d2, expand)
}

/// Resolve denominator relation for `a/d1 + b/d2` style rewrites.
///
/// If denominators are opposite, this normalizes to common denominator `d1`
/// and flips `n2` sign accordingly.
pub fn resolve_add_denominator_relation_with<FExpand>(
    ctx: &mut Context,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
    expand: FExpand,
) -> AddDenominatorResolution
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    if denominators_are_same_with(ctx, d1, d2, expand) {
        return AddDenominatorResolution {
            adjusted_n2: n2,
            adjusted_d2: d2,
            relation: DenominatorRelation::Same,
        };
    }

    if are_denominators_opposite(ctx, d1, d2) {
        let minus_n2 = ctx.add(Expr::Neg(n2));
        return AddDenominatorResolution {
            adjusted_n2: minus_n2,
            adjusted_d2: d1,
            relation: DenominatorRelation::Opposite,
        };
    }

    AddDenominatorResolution {
        adjusted_n2: n2,
        adjusted_d2: d2,
        relation: DenominatorRelation::Different,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        denominators_are_same_with, resolve_add_denominator_relation_with, DenominatorRelation,
    };
    use crate::expand_ops::expand;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn same_relation_detected_structurally() {
        let mut ctx = Context::new();
        let d = parse("x+1", &mut ctx).expect("parse");
        let n2 = parse("a", &mut ctx).expect("parse");
        let rel = resolve_add_denominator_relation_with(&mut ctx, n2, d, d, expand);
        assert_eq!(rel.relation, DenominatorRelation::Same);
        assert_eq!(rel.adjusted_n2, n2);
        assert_eq!(rel.adjusted_d2, d);
    }

    #[test]
    fn opposite_relation_flips_sign_and_denominator() {
        let mut ctx = Context::new();
        let n2 = parse("a", &mut ctx).expect("parse");
        let d1 = parse("x-1", &mut ctx).expect("parse");
        let d2 = parse("1-x", &mut ctx).expect("parse");
        let rel = resolve_add_denominator_relation_with(&mut ctx, n2, d1, d2, expand);
        assert_eq!(rel.relation, DenominatorRelation::Opposite);
        assert_eq!(rel.adjusted_d2, d1);
        assert!(matches!(ctx.get(rel.adjusted_n2), Expr::Neg(_)));
    }

    #[test]
    fn different_relation_preserves_inputs() {
        let mut ctx = Context::new();
        let n2 = parse("a", &mut ctx).expect("parse");
        let d1 = parse("x+1", &mut ctx).expect("parse");
        let d2 = parse("x+2", &mut ctx).expect("parse");
        let rel = resolve_add_denominator_relation_with(&mut ctx, n2, d1, d2, expand);
        assert_eq!(rel.relation, DenominatorRelation::Different);
        assert_eq!(rel.adjusted_n2, n2);
        assert_eq!(rel.adjusted_d2, d2);
    }

    #[test]
    fn same_relation_detected_algebraically() {
        let mut ctx = Context::new();
        let d1 = parse("u*(u+2)", &mut ctx).expect("parse");
        let d2 = parse("u^2+2*u", &mut ctx).expect("parse");
        assert!(denominators_are_same_with(&mut ctx, d1, d2, expand));
    }
}
