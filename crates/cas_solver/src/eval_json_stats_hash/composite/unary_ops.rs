use std::hash::Hasher;

use cas_ast::{Context, ExprId};

use super::unary;

pub(in crate::eval_json_stats_hash) fn hash_hold<H: Hasher>(
    ctx: &Context,
    inner: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    unary::hash_hold(ctx, inner, hasher, recur);
}

pub(in crate::eval_json_stats_hash) fn hash_neg<H: Hasher>(
    ctx: &Context,
    inner: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    unary::hash_neg(ctx, inner, hasher, recur);
}
