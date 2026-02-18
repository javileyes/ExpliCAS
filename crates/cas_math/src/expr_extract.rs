//! Shared helpers to parse primitive values from AST expressions.

use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

/// Extract a non-negative integer as `u64` from an expression.
pub fn extract_u64_integer(ctx: &Context, expr: ExprId) -> Option<u64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return n.to_integer().to_u64();
        }
    }
    None
}

/// Extract a non-negative integer as `usize` from an expression.
pub fn extract_usize_integer(ctx: &Context, expr: ExprId) -> Option<usize> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return n.to_integer().to_usize();
        }
    }
    None
}

/// Extract a symbol token from an expression (represented as `Variable`).
pub fn extract_symbol_name(ctx: &Context, expr: ExprId) -> Option<&str> {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        return Some(ctx.sym_name(*sym_id));
    }
    None
}
