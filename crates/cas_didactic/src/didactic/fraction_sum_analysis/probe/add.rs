use super::FractionSumInfo;
use crate::didactic::{collect_add_terms, try_as_fraction, IsOne};
use cas_ast::{Context, ExprId};
use num_rational::BigRational;

pub(super) fn find_fraction_sum_in_add(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    let mut terms = Vec::new();
    collect_add_terms(ctx, expr, &mut terms);

    let mut fractions = Vec::new();
    for term in &terms {
        if let Some(frac) = try_as_fraction(ctx, *term) {
            fractions.push(frac);
        } else {
            return None;
        }
    }

    if fractions.len() < 2 {
        return None;
    }

    let has_actual_fraction = fractions.iter().any(|f| !f.denom().is_one());
    if !has_actual_fraction {
        return None;
    }

    let result: BigRational = fractions.iter().cloned().sum();
    Some(FractionSumInfo { fractions, result })
}
