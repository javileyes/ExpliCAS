//! Canonical helpers for internal equation wrapper `__eq__`.
//!
//! The `__eq__` function is an internal wrapper representing equations
//! in the form `lhs = rhs`. It is NOT the same as `Equal` (symbolic comparison).
//!
//! # Usage
//!
//! ```rust,ignore
//! use cas_ast::eq::{eq_name, is_eq_name};
//!
//! // Instead of: name == "__eq__"
//! // Use:
//! if is_eq_name(name) { ... }
//!
//! // Instead of: ctx.call("__eq__", vec![lhs, rhs])
//! // Use:
//! let eq_expr = wrap_eq(ctx, lhs, rhs);
//! ```
//!
//! # See also
//!
//! - `BuiltinFn::Eq` in `builtin.rs`
//! - `docs/builtin_guidelines.md` for the integration pattern

use crate::{BuiltinFn, Context, Expr, ExprId};

/// Canonical name for the internal equation wrapper.
#[inline]
pub const fn eq_name() -> &'static str {
    BuiltinFn::Eq.name()
}

/// Check if a function name is the internal equation wrapper.
#[inline]
pub fn is_eq_name(name: &str) -> bool {
    name == eq_name()
}

/// Wrap two expressions as an equation `lhs = rhs`.
///
/// Returns `__eq__(lhs, rhs)` which represents the equation internally.
/// Uses BuiltinFn::Eq for O(1) symbol lookup.
#[inline]
pub fn wrap_eq(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> ExprId {
    ctx.call_builtin(BuiltinFn::Eq, vec![lhs, rhs])
}

/// Try to unwrap an equation into its (lhs, rhs) components.
///
/// Returns `Some((lhs, rhs))` if the expression is `__eq__(lhs, rhs)`,
/// `None` otherwise.
pub fn unwrap_eq(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Function(fn_id, args) = ctx.get(id) {
        if is_eq_name(ctx.sym_name(*fn_id)) && args.len() == 2 {
            return Some((args[0], args[1]));
        }
    }
    None
}

/// Check if an expression is an equation wrapper.
#[inline]
pub fn is_eq(ctx: &Context, id: ExprId) -> bool {
    unwrap_eq(ctx, id).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_name() {
        assert_eq!(eq_name(), "__eq__");
    }

    #[test]
    fn test_is_eq_name() {
        assert!(is_eq_name("__eq__"));
        assert!(!is_eq_name("Equal"));
        assert!(!is_eq_name("eq"));
    }

    #[test]
    fn test_wrap_unwrap_eq() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);

        let eq_expr = wrap_eq(&mut ctx, x, one);

        assert!(is_eq(&ctx, eq_expr));
        let (lhs, rhs) = unwrap_eq(&ctx, eq_expr).unwrap();
        assert_eq!(lhs, x);
        assert_eq!(rhs, one);
    }

    #[test]
    fn test_non_eq_returns_none() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        assert!(!is_eq(&ctx, x));
        assert!(unwrap_eq(&ctx, x).is_none());
    }
}
