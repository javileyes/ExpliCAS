use super::{find_fraction_in_add, NestedFractionPattern};
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_division_like_term;
use num_bigint::BigInt;

/// Classify a nested fraction expression.
pub(crate) fn classify_nested_fraction(
    ctx: &Context,
    expr: ExprId,
) -> Option<NestedFractionPattern> {
    let is_one = |id: ExprId| -> bool {
        matches!(ctx.get(id), Expr::Number(n) if n.is_integer() && *n.numer() == BigInt::from(1))
    };

    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Some(inner_frac) = find_fraction_in_add(ctx, *den) {
            if is_one(*num) {
                if let Expr::Div(n, _) = ctx.get(inner_frac) {
                    if is_one(*n) {
                        return Some(NestedFractionPattern::OneOverSumWithUnitFraction);
                    }
                }
                return Some(NestedFractionPattern::OneOverSumWithFraction);
            }
            return Some(NestedFractionPattern::FractionOverSumWithFraction);
        }

        if contains_division_like_term(ctx, *num) && !contains_division_like_term(ctx, *den) {
            return Some(NestedFractionPattern::SumWithFractionOverScalar);
        }

        if contains_division_like_term(ctx, *den) {
            return Some(NestedFractionPattern::General);
        }
    }

    None
}
