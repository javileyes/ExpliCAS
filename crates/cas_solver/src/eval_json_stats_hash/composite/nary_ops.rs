use std::hash::Hasher;

use cas_ast::{symbol::SymbolId, Context, ExprId};

use super::nary;

pub(in crate::eval_json_stats_hash) fn hash_function<H: Hasher>(
    ctx: &Context,
    name: &SymbolId,
    args: &[ExprId],
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    nary::hash_function(ctx, name, args, hasher, recur);
}

pub(in crate::eval_json_stats_hash) fn hash_matrix<H: Hasher>(
    ctx: &Context,
    rows: &usize,
    cols: &usize,
    data: &[ExprId],
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    nary::hash_matrix(ctx, rows, cols, data, hasher, recur);
}
