//! End-to-end planner for subtraction of fraction pairs.

use crate::fraction_sub_build_support::build_sub_fraction_rewrite;
use crate::fraction_sub_denominator_plan_support::plan_sub_denominator_with;
use crate::fraction_zero_numerator_support::{
    build_zero_or_zero_over_den, numerator_simplifies_to_zero_with,
};
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFractionRewriteKind {
    ZeroNumerator,
    NumericDenominators,
    General,
}

#[derive(Debug, Clone, Copy)]
pub struct SubFractionRewritePlan {
    pub rewritten: ExprId,
    pub kind: SubFractionRewriteKind,
}

impl SubFractionRewritePlan {
    pub fn desc(self) -> &'static str {
        match self.kind {
            SubFractionRewriteKind::ZeroNumerator => "Subtract fractions: numerator cancels to 0",
            SubFractionRewriteKind::NumericDenominators => "Subtract numeric fractions",
            SubFractionRewriteKind::General => "Subtract fractions: a/b - c/d -> (ad-bc)/bd",
        }
    }
}

/// Plan rewrite for `(n1/d1) - (n2/d2)`.
///
/// Includes:
/// - denominator relation planning,
/// - cross-product or same-denominator build,
/// - early zero-numerator collapse to `0` or `0/den`,
/// - numeric-denominator classification for description selection.
pub fn plan_sub_fraction_rewrite_with<FExpand>(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
    mut expand: FExpand,
) -> SubFractionRewritePlan
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    let den_plan = plan_sub_denominator_with(ctx, n1, n2, d1, d2, &mut expand);
    let built = build_sub_fraction_rewrite(
        ctx,
        den_plan.n1,
        den_plan.n2,
        d1,
        d2,
        den_plan.common_den,
        den_plan.same_denom,
    );

    if numerator_simplifies_to_zero_with(ctx, built.numerator, &mut expand) {
        return SubFractionRewritePlan {
            rewritten: build_zero_or_zero_over_den(ctx, built.denominator),
            kind: SubFractionRewriteKind::ZeroNumerator,
        };
    }

    let rewritten = ctx.add(Expr::Div(built.numerator, built.denominator));
    let is_numeric = |e: ExprId| matches!(ctx.get(e), Expr::Number(_));
    let kind = if is_numeric(d1) && is_numeric(d2) {
        SubFractionRewriteKind::NumericDenominators
    } else {
        SubFractionRewriteKind::General
    };
    SubFractionRewritePlan { rewritten, kind }
}

#[cfg(test)]
mod tests {
    use super::{plan_sub_fraction_rewrite_with, SubFractionRewriteKind};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn plans_numeric_denominator_subtraction() {
        let mut ctx = Context::new();
        let n1 = parse("1", &mut ctx).expect("parse");
        let d1 = parse("2", &mut ctx).expect("parse");
        let n2 = parse("1", &mut ctx).expect("parse");
        let d2 = parse("3", &mut ctx).expect("parse");
        let plan = plan_sub_fraction_rewrite_with(&mut ctx, n1, n2, d1, d2, |_, e| e);
        assert_eq!(plan.kind, SubFractionRewriteKind::NumericDenominators);
    }

    #[test]
    fn plans_zero_numerator_collapse() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse");
        let d1 = parse("d", &mut ctx).expect("parse");
        let n2 = parse("a", &mut ctx).expect("parse");
        let d2 = parse("d", &mut ctx).expect("parse");
        let plan =
            plan_sub_fraction_rewrite_with(&mut ctx, n1, n2, d1, d2, crate::expand_ops::expand);
        assert_eq!(plan.kind, SubFractionRewriteKind::ZeroNumerator);
    }
}
