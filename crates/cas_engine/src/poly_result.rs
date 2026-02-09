//! Poly Result Utilities
//!
//! `poly_result(id)` is an internal wrapper that references stored polynomials
//! by their PolyId without materializing the full AST.
//!
//! # Contract
//!
//! 1. `poly_result(Number(id))` is the canonical form
//! 2. All checks for "is this a poly_result?" MUST use these helpers
//! 3. Do NOT compare strings directly - use `is_poly_result` instead
//!
//! # Single Source of Truth
//!
//! This module is the ONLY place where the string "poly_result" should appear.
//! External code should use the helpers below.

use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

use crate::poly_store::PolyId;

/// Canonical function name for poly_result wrapper.
/// This constant is private - external code should use helpers.
const POLY_RESULT_FN: &str = "poly_result";

/// Check if expression is a poly_result wrapper.
///
/// # Example
/// ```ignore
/// if is_poly_result(ctx, expr) {
///     // Handle poly_result case
/// }
/// ```
#[inline]
pub fn is_poly_result(ctx: &Context, id: ExprId) -> bool {
    matches!(
        ctx.get(id),
        Expr::Function(fn_id, args)
            if ctx.sym_name(*fn_id) == POLY_RESULT_FN && args.len() == 1
    )
}

/// Extract the raw argument from a poly_result wrapper.
/// Returns `None` if not a poly_result.
///
/// Use this when you need access to the argument expression
/// but don't need to parse it as a PolyId yet.
#[inline]
pub fn poly_result_arg(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Function(fn_id, args)
            if ctx.sym_name(*fn_id) == POLY_RESULT_FN && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

/// Parse a poly_result expression and extract the PolyId.
/// Returns `None` if:
/// - Not a poly_result wrapper
/// - Argument is not a Number
/// - Number is not a valid non-negative integer
/// - Number doesn't fit in PolyId (u32)
///
/// # Example
/// ```ignore
/// if let Some(poly_id) = parse_poly_result_id(ctx, expr) {
///     // Use poly_id with PolyStore
/// }
/// ```
pub fn parse_poly_result_id(ctx: &Context, id: ExprId) -> Option<PolyId> {
    let arg = poly_result_arg(ctx, id)?;

    match ctx.get(arg) {
        Expr::Number(n) => {
            // Must be a non-negative integer that fits in PolyId
            if n.is_integer() && n.numer().sign() != num_bigint::Sign::Minus {
                n.to_integer().to_u32()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Create a poly_result wrapper expression.
///
/// # Example
/// ```ignore
/// let poly_id = store.insert(meta, poly);
/// let wrapped = wrap_poly_result(ctx, poly_id);
/// ```
pub fn wrap_poly_result(ctx: &mut Context, poly_id: PolyId) -> ExprId {
    let id_expr = ctx.num(poly_id as i64);
    ctx.call(POLY_RESULT_FN, vec![id_expr])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_poly_result() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let id_expr = ctx.num(42);
        let poly_result = ctx.call_builtin(cas_ast::BuiltinFn::PolyResult, vec![id_expr]);

        assert!(!is_poly_result(&ctx, x));
        assert!(!is_poly_result(&ctx, id_expr));
        assert!(is_poly_result(&ctx, poly_result));
    }

    #[test]
    fn test_poly_result_arg() {
        let mut ctx = Context::new();
        let id_expr = ctx.num(42);
        let poly_result = ctx.call_builtin(cas_ast::BuiltinFn::PolyResult, vec![id_expr]);

        assert_eq!(poly_result_arg(&ctx, poly_result), Some(id_expr));
        assert_eq!(poly_result_arg(&ctx, id_expr), None);
    }

    #[test]
    fn test_parse_poly_result_id() {
        let mut ctx = Context::new();
        let id_expr = ctx.num(42);
        let poly_result = ctx.call_builtin(cas_ast::BuiltinFn::PolyResult, vec![id_expr]);

        assert_eq!(parse_poly_result_id(&ctx, poly_result), Some(42));

        // Not a poly_result
        assert_eq!(parse_poly_result_id(&ctx, id_expr), None);

        // Negative number
        let neg = ctx.num(-1);
        let bad_poly_result = ctx.call_builtin(cas_ast::BuiltinFn::PolyResult, vec![neg]);
        assert_eq!(parse_poly_result_id(&ctx, bad_poly_result), None);
    }

    #[test]
    fn test_wrap_poly_result() {
        let mut ctx = Context::new();
        let wrapped = wrap_poly_result(&mut ctx, 123);

        assert!(is_poly_result(&ctx, wrapped));
        assert_eq!(parse_poly_result_id(&ctx, wrapped), Some(123));
    }

    #[test]
    fn test_roundtrip() {
        let mut ctx = Context::new();
        let original_id: PolyId = 999;

        let wrapped = wrap_poly_result(&mut ctx, original_id);
        let parsed = parse_poly_result_id(&ctx, wrapped);

        assert_eq!(parsed, Some(original_id));
    }
}
