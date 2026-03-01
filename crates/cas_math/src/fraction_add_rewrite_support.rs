//! End-to-end planner for addition of fraction pairs.

use crate::fraction_add_build_support::build_add_fraction_rewrite;
use crate::fraction_add_denominator_plan_support::plan_add_denominator_with;
use crate::fraction_add_heuristics_support::{
    assess_fraction_add_simplification, should_accept_fraction_add_rewrite,
    FractionAddAcceptanceInput,
};
use crate::fraction_trig_add_policy_support::try_plan_numeric_fraction_add_desc;
use crate::fraction_zero_numerator_support::{
    build_zero_or_zero_over_den, numerator_simplifies_to_zero_with,
};
use cas_ast::{count_nodes, Context, Expr, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct AddFractionRewriteInput {
    pub expr: ExprId,
    pub l: ExprId,
    pub r: ExprId,
    pub n1: ExprId,
    pub d1: ExprId,
    pub n2: ExprId,
    pub d2: ExprId,
    pub same_sign: bool,
    pub inside_trig: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddFractionRewriteKind {
    ZeroNumerator,
    NumericDenominators,
    General,
}

#[derive(Debug, Clone, Copy)]
pub struct AddFractionRewritePlan {
    pub rewritten: ExprId,
    pub kind: AddFractionRewriteKind,
}

impl AddFractionRewritePlan {
    pub fn desc(self) -> &'static str {
        match self.kind {
            AddFractionRewriteKind::ZeroNumerator => "Add fractions: numerator cancels to 0",
            AddFractionRewriteKind::NumericDenominators => "Add numeric fractions",
            AddFractionRewriteKind::General => "Add fractions: a/b + c/d -> (ad+bc)/bd",
        }
    }
}

/// Plan rewrite for `(n1/d1) + (n2/d2)`.
///
/// Includes:
/// - denominator relation planning,
/// - fraction build,
/// - early zero-numerator collapse to `0` or `0/den`,
/// - numeric/trig fast path policy,
/// - complexity-driven general acceptance heuristic.
pub fn plan_add_fraction_rewrite_with<FExpand>(
    ctx: &mut Context,
    input: AddFractionRewriteInput,
    mut expand: FExpand,
) -> Option<AddFractionRewritePlan>
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    let den_plan =
        plan_add_denominator_with(ctx, input.n1, input.n2, input.d1, input.d2, &mut expand);
    let (n1, n2, d1, d2, common_den, opposite_denom, same_denom) = (
        den_plan.n1,
        den_plan.n2,
        den_plan.d1,
        den_plan.d2,
        den_plan.common_den,
        den_plan.opposite_denom,
        den_plan.same_denom,
    );

    let old_complexity = count_nodes(ctx, input.expr);
    let built = build_add_fraction_rewrite(
        ctx,
        n1,
        n2,
        d1,
        d2,
        common_den,
        opposite_denom || same_denom,
    );
    let new_num = built.numerator;
    let common_den = built.denominator;

    if numerator_simplifies_to_zero_with(ctx, new_num, &mut expand) {
        return Some(AddFractionRewritePlan {
            rewritten: build_zero_or_zero_over_den(ctx, common_den),
            kind: AddFractionRewriteKind::ZeroNumerator,
        });
    }

    let rewritten = ctx.add(Expr::Div(new_num, common_den));
    let new_complexity = count_nodes(ctx, rewritten);

    if matches!(ctx.get(d1), Expr::Number(_)) && matches!(ctx.get(d2), Expr::Number(_)) {
        let desc =
            try_plan_numeric_fraction_add_desc(ctx, input.l, input.r, d1, d2, input.inside_trig)?;
        if desc == "Add numeric fractions" {
            return Some(AddFractionRewritePlan {
                rewritten,
                kind: AddFractionRewriteKind::NumericDenominators,
            });
        }
    }

    let (does_simplify, is_proper) = assess_fraction_add_simplification(ctx, new_num, common_den);
    if should_accept_fraction_add_rewrite(
        ctx,
        FractionAddAcceptanceInput {
            n1,
            n2,
            old_complexity,
            new_complexity,
            opposite_denom,
            same_denom,
            does_simplify,
            is_proper,
            same_sign: input.same_sign,
        },
    ) {
        return Some(AddFractionRewritePlan {
            rewritten,
            kind: AddFractionRewriteKind::General,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{plan_add_fraction_rewrite_with, AddFractionRewriteInput, AddFractionRewriteKind};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn plans_zero_numerator_collapse() {
        let mut ctx = Context::new();
        let expr = parse("a/d + (-a)/d", &mut ctx).expect("parse");
        let l = parse("a/d", &mut ctx).expect("parse");
        let r = parse("(-a)/d", &mut ctx).expect("parse");
        let n1 = parse("a", &mut ctx).expect("parse");
        let d1 = parse("d", &mut ctx).expect("parse");
        let n2 = parse("-a", &mut ctx).expect("parse");
        let d2 = parse("d", &mut ctx).expect("parse");
        let plan = plan_add_fraction_rewrite_with(
            &mut ctx,
            AddFractionRewriteInput {
                expr,
                l,
                r,
                n1,
                d1,
                n2,
                d2,
                same_sign: false,
                inside_trig: false,
            },
            crate::expand_ops::expand,
        )
        .expect("plan");
        assert_eq!(plan.kind, AddFractionRewriteKind::ZeroNumerator);
    }

    #[test]
    fn plans_numeric_denominator_case_outside_trig() {
        let mut ctx = Context::new();
        let expr = parse("1/2 + 1/3", &mut ctx).expect("parse");
        let l = parse("1/2", &mut ctx).expect("parse");
        let r = parse("1/3", &mut ctx).expect("parse");
        let n1 = parse("1", &mut ctx).expect("parse");
        let d1 = parse("2", &mut ctx).expect("parse");
        let n2 = parse("1", &mut ctx).expect("parse");
        let d2 = parse("3", &mut ctx).expect("parse");
        let plan = plan_add_fraction_rewrite_with(
            &mut ctx,
            AddFractionRewriteInput {
                expr,
                l,
                r,
                n1,
                d1,
                n2,
                d2,
                same_sign: true,
                inside_trig: false,
            },
            crate::expand_ops::expand,
        )
        .expect("plan");
        assert_eq!(plan.kind, AddFractionRewriteKind::NumericDenominators);
    }

    #[test]
    fn blocks_numeric_symbol_plus_pi_inside_trig() {
        let mut ctx = Context::new();
        let expr = parse("x + pi/6", &mut ctx).expect("parse");
        let l = parse("x", &mut ctx).expect("parse");
        let r = parse("pi/6", &mut ctx).expect("parse");
        let n1 = parse("x", &mut ctx).expect("parse");
        let d1 = parse("1", &mut ctx).expect("parse");
        let n2 = parse("pi", &mut ctx).expect("parse");
        let d2 = parse("6", &mut ctx).expect("parse");
        let plan = plan_add_fraction_rewrite_with(
            &mut ctx,
            AddFractionRewriteInput {
                expr,
                l,
                r,
                n1,
                d1,
                n2,
                d2,
                same_sign: true,
                inside_trig: true,
            },
            crate::expand_ops::expand,
        );
        assert!(plan.is_none());
    }
}
