// =============================================================================
// Numerical Helpers (Zero-Clone for Number inspection)
// =============================================================================

/// Get a reference to the Number without cloning.
/// Use for inspection only; the reference is tied to the Context's lifetime.
#[inline]
pub fn as_number(ctx: &Context, id: ExprId) -> Option<&num_rational::BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n),
        _ => None,
    }
}

/// Check if expression is a negative number (without cloning).
#[inline]
pub fn is_negative_number(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Number(n) if n.is_negative())
}

/// Check if expression is a positive integer (without cloning).
#[inline]
pub fn is_positive_integer(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Number(n) if n.is_integer() && n.is_positive())
}

/// Try to extract an i64 value from a Number expression (without cloning).
/// Returns None if not a Number, not an integer, or doesn't fit in i64.
#[inline]
pub fn as_i64(ctx: &Context, id: ExprId) -> Option<i64> {
    match ctx.get(id) {
        Expr::Number(n) if n.is_integer() => n.to_integer().to_i64(),
        _ => None,
    }
}

/// Alias for as_div - useful for fraction-specific code.
#[inline]
pub fn as_frac(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    as_div(ctx, id)
}

/// Check if expression is a fraction (Div node).
#[inline]
pub fn is_frac(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Div(_, _))
}

