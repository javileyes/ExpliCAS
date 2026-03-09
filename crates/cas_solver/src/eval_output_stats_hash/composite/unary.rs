use std::hash::{Hash, Hasher};

use cas_ast::{Context, ExprId};

pub(super) fn hash_hold<H: Hasher>(
    ctx: &Context,
    inner: ExprId,
    hasher: &mut H,
    recur: fn(&Context, ExprId, &mut H),
) {
    12u8.hash(hasher);
    recur(ctx, inner, hasher);
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
