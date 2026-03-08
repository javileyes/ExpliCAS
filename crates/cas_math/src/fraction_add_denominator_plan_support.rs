//! Denominator planning helpers for fraction addition.

use crate::fraction_denominator_relation_support::{
    resolve_add_denominator_relation_with, DenominatorRelation,
};
use crate::fraction_forms::check_divisible_denominators;
use cas_ast::{Context, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct AddDenominatorPlan {
    pub n1: ExprId,
    pub n2: ExprId,
    pub d1: ExprId,
    pub d2: ExprId,
    pub common_den: ExprId,
    pub opposite_denom: bool,
    pub same_denom: bool,
}

/// Plan denominator relation for `(n1/d1) + (n2/d2)`.
///
/// Handles:
/// - same/opposite denominator relation (with algebraic check)
/// - divisible denominator normalization for common-denominator construction.
pub fn plan_add_denominator_with<FExpand>(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
    mut expand: FExpand,
) -> AddDenominatorPlan
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    let rel = resolve_add_denominator_relation_with(ctx, n2, d1, d2, &mut expand);
    let opposite_denom = matches!(rel.relation, DenominatorRelation::Opposite);
    let same_relation = matches!(rel.relation, DenominatorRelation::Same);

    let (n1, n2, common_den, divisible_denom) =
        check_divisible_denominators(ctx, n1, rel.adjusted_n2, d1, rel.adjusted_d2);

    AddDenominatorPlan {
        n1,
        n2,
        d1,
        d2: rel.adjusted_d2,
        common_den,
        opposite_denom,
        same_denom: same_relation || divisible_denom,
    }
}

#[cfg(test)]
mod tests {
    use super::plan_add_denominator_with;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_same_denominator_plan() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse");
        let n2 = parse("b", &mut ctx).expect("parse");
        let d1 = parse("x+1", &mut ctx).expect("parse");
        let d2 = parse("x+1", &mut ctx).expect("parse");
        let plan = plan_add_denominator_with(&mut ctx, n1, n2, d1, d2, |_, e| e);
        assert!(plan.same_denom);
        assert!(!plan.opposite_denom);
    }

    #[test]
    fn keeps_unrelated_denominators_as_different() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse");
        let n2 = parse("b", &mut ctx).expect("parse");
        let d1 = parse("x+1", &mut ctx).expect("parse");
        let d2 = parse("x+2", &mut ctx).expect("parse");
        let plan = plan_add_denominator_with(&mut ctx, n1, n2, d1, d2, |_, e| e);
        assert!(!plan.same_denom);
        assert!(!plan.opposite_denom);
    }
}
