use std::hash::{Hash, Hasher};

use cas_ast::{symbol::SymbolId, Context, ExprId};

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
