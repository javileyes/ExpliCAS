use std::hash::Hasher;

use cas_ast::{Context, ExprId};

use super::binary;

pub(in crate::eval_json_stats_hash) fn hash_add<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    binary::hash_add(ctx, l, r, hasher, recur);
}

pub(in crate::eval_json_stats_hash) fn hash_sub<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    binary::hash_sub(ctx, l, r, hasher, recur);
}

pub(in crate::eval_json_stats_hash) fn hash_mul<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    binary::hash_mul(ctx, l, r, hasher, recur);
}

pub(in crate::eval_json_stats_hash) fn hash_div<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    binary::hash_div(ctx, l, r, hasher, recur);
}

pub(in crate::eval_json_stats_hash) fn hash_pow<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    binary::hash_pow(ctx, l, r, hasher, recur);
}
