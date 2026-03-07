mod dedupe;
mod select;

use super::FractionSumInfo;
use cas_ast::{Context, ExprId};

pub(super) fn collect_primary_fraction_sums(
    ctx: &Context,
    original_expr: ExprId,
    find_all_fraction_sums: fn(&Context, ExprId) -> Vec<FractionSumInfo>,
) -> Vec<FractionSumInfo> {
    let all_fraction_sums = find_all_fraction_sums(ctx, original_expr);
    if all_fraction_sums.is_empty() {
        return Vec::new();
    }

    let max_fractions = select::max_fraction_count(&all_fraction_sums);
    dedupe::collect_unique_primary_fraction_sums(all_fraction_sums, max_fractions)
}
