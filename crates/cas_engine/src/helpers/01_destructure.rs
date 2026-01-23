// =============================================================================
// Expression Destructuring Helpers (Zero-Clone Pattern)
// =============================================================================
//
// These helpers extract child ExprIds without cloning the Expr enum.
// Use them instead of `ctx.get(id).clone()` to avoid unnecessary allocations.
//
// Pattern: Extract IDs first (in a short scope), then mutate ctx.

/// Destruct Add(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_add(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Add(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Sub(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_sub(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Sub(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Mul(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_mul(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Mul(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Div(l, r) -> Some((l, r)), else None
#[inline]
pub fn as_div(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Div(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct Pow(base, exp) -> Some((base, exp)), else None
#[inline]
pub fn as_pow(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Pow(b, e) => Some((*b, *e)),
        _ => None,
    }
}

/// Destruct Neg(inner) -> Some(inner), else None
#[inline]
pub fn as_neg(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Neg(inner) => Some(*inner),
        _ => None,
    }
}

/// Extract function name and args without cloning String or Vec.
/// Returns references with the Context's lifetime.
#[inline]
pub fn fn_name_args(ctx: &Context, id: ExprId) -> Option<(&str, &[ExprId])> {
    match ctx.get(id) {
        Expr::Function(name, args) => Some((name.as_str(), args.as_slice())),
        _ => None,
    }
}

/// Check if expression matches a 1-arg function with the given name.
/// Returns the argument ExprId if matched.
#[inline]
pub fn as_fn1(ctx: &Context, id: ExprId, name: &str) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Function(fn_name, args) if fn_name == name && args.len() == 1 => Some(args[0]),
        _ => None,
    }
}

/// Check if expression matches a 2-arg function with the given name.
/// Returns the argument ExprIds if matched.
#[inline]
pub fn as_fn2(ctx: &Context, id: ExprId, name: &str) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Function(fn_name, args) if fn_name == name && args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    }
}

