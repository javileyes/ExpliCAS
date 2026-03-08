use std::hash::{Hash, Hasher};

use cas_ast::{Context, ExprId};

pub(super) fn hash_add<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    hash_binary(3u8, ctx, l, r, hasher, recur);
}

pub(super) fn hash_sub<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    hash_binary(4u8, ctx, l, r, hasher, recur);
}

pub(super) fn hash_mul<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    hash_binary(5u8, ctx, l, r, hasher, recur);
}

pub(super) fn hash_div<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    hash_binary(6u8, ctx, l, r, hasher, recur);
}

pub(super) fn hash_pow<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    hash_binary(7u8, ctx, l, r, hasher, recur);
}

fn hash_binary<H: Hasher>(
    tag: u8,
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    tag.hash(hasher);
    recur(ctx, l, hasher);
    recur(ctx, r, hasher);
}
