mod add;
mod search;

use super::FractionSumInfo;
use cas_ast::{Context, ExprId};

pub(super) fn find_fraction_sum_in_expr(ctx: &Context, expr: ExprId) -> Option<FractionSumInfo> {
    search::find_fraction_sum_in_expr(ctx, expr)
}
