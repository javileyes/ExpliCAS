mod children;
mod current;

use super::{find_fraction_sum_in_expr, FractionSumInfo};
use cas_ast::{Context, ExprId};

pub(super) fn find_all_fraction_sums_recursive(
    ctx: &Context,
    expr: ExprId,
    results: &mut Vec<FractionSumInfo>,
) {
    current::push_fraction_sum_if_present(find_fraction_sum_in_expr(ctx, expr), results);
    children::scan_fraction_sum_children(ctx, expr, results, find_all_fraction_sums_recursive);
}
