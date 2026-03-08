mod classify;
mod combined;
mod search;

use cas_ast::{Context, Expr, ExprId};

/// Pattern classification for nested fractions.
#[derive(Debug)]
pub(crate) enum NestedFractionPattern {
    /// P1: 1/(a + 1/b) -> b/(a*b + 1)
    OneOverSumWithUnitFraction,
    /// P2: 1/(a + b/c) -> c/(a*c + b)
    OneOverSumWithFraction,
    /// P3: A/(B + C/D) -> A*D/(B*D + C)
    FractionOverSumWithFraction,
    /// P4: (A + 1/B)/C -> (A*B + 1)/(B*C)
    SumWithFractionOverScalar,
    /// Fallback for complex patterns.
    General,
}

pub(crate) use self::classify::classify_nested_fraction;
pub(crate) use self::combined::extract_combined_fraction_str;
pub(crate) use self::search::find_div_in_expr;

pub(crate) fn find_fraction_in_add(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            if matches!(ctx.get(*l), Expr::Div(_, _)) {
                Some(*l)
            } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
                Some(*r)
            } else {
                None
            }
        }
        _ => None,
    }
}
