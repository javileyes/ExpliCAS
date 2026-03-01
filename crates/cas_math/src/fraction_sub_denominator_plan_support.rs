//! Denominator planning helpers for fraction subtraction.

use crate::fraction_denominator_relation_support::denominators_are_same_with;
use crate::fraction_forms::check_divisible_denominators;
use cas_ast::{Context, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct SubDenominatorPlan {
    pub n1: ExprId,
    pub n2: ExprId,
    pub common_den: ExprId,
    pub same_denom: bool,
}

/// Plan denominator relation for `(n1/d1) - (n2/d2)`.
///
/// Handles:
/// - semantic/algebraic same-denominator detection
/// - divisible denominator normalization via `check_divisible_denominators`
pub fn plan_sub_denominator_with<FExpand>(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
    mut expand: FExpand,
) -> SubDenominatorPlan
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    let same_denom = denominators_are_same_with(ctx, d1, d2, &mut expand);
    let (n1, n2, common_den, divisible_denom) = check_divisible_denominators(ctx, n1, n2, d1, d2);
    SubDenominatorPlan {
        n1,
        n2,
        common_den,
        same_denom: same_denom || divisible_denom,
    }
}

#[cfg(test)]
mod tests {
    use super::plan_sub_denominator_with;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_same_denominator() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse");
        let n2 = parse("b", &mut ctx).expect("parse");
        let d1 = parse("x+1", &mut ctx).expect("parse");
        let d2 = parse("x+1", &mut ctx).expect("parse");
        let plan = plan_sub_denominator_with(&mut ctx, n1, n2, d1, d2, |_, e| e);
        assert!(plan.same_denom);
    }

    #[test]
    fn keeps_different_denominators_as_not_same() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse");
        let n2 = parse("b", &mut ctx).expect("parse");
        let d1 = parse("x+1", &mut ctx).expect("parse");
        let d2 = parse("x+2", &mut ctx).expect("parse");
        let plan = plan_sub_denominator_with(&mut ctx, n1, n2, d1, d2, |_, e| e);
        assert!(!plan.same_denom);
    }
}
