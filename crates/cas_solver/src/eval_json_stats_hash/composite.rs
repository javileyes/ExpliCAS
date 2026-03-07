use std::hash::{Hash, Hasher};

use cas_ast::{symbol::SymbolId, Context, ExprId};

pub(super) fn hash_hold<H: Hasher>(
    ctx: &Context,
    inner: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    12u8.hash(hasher);
    recur(ctx, inner, hasher);
}

pub(super) fn hash_add<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    3u8.hash(hasher);
    recur(ctx, l, hasher);
    recur(ctx, r, hasher);
}

pub(super) fn hash_sub<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    4u8.hash(hasher);
    recur(ctx, l, hasher);
    recur(ctx, r, hasher);
}

pub(super) fn hash_mul<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    5u8.hash(hasher);
    recur(ctx, l, hasher);
    recur(ctx, r, hasher);
}

pub(super) fn hash_div<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    6u8.hash(hasher);
    recur(ctx, l, hasher);
    recur(ctx, r, hasher);
}

pub(super) fn hash_pow<H: Hasher>(
    ctx: &Context,
    l: ExprId,
    r: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    7u8.hash(hasher);
    recur(ctx, l, hasher);
    recur(ctx, r, hasher);
}

pub(super) fn hash_neg<H: Hasher>(
    ctx: &Context,
    inner: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    8u8.hash(hasher);
    recur(ctx, inner, hasher);
}

pub(super) fn hash_function<H: Hasher>(
    ctx: &Context,
    name: &SymbolId,
    args: &[ExprId],
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    9u8.hash(hasher);
    name.hash(hasher);
    for arg in args {
        recur(ctx, *arg, hasher);
    }
}

pub(super) fn hash_matrix<H: Hasher>(
    ctx: &Context,
    rows: &usize,
    cols: &usize,
    data: &[ExprId],
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    10u8.hash(hasher);
    rows.hash(hasher);
    cols.hash(hasher);
    for elem in data {
        recur(ctx, *elem, hasher);
    }
}
