//! Compatibility adapter for `poly_result(id)` AST helpers.
//!
//! Canonical implementation lives in `cas_math::poly_result`.

use crate::poly_store::PolyId;
use cas_ast::{Context, ExprId};

/// Check whether `id` is a `poly_result(...)` wrapper.
#[inline]
pub fn is_poly_result(ctx: &Context, id: ExprId) -> bool {
    cas_math::poly_result::is_poly_result(ctx, id)
}

/// Return the wrapped argument of `poly_result(arg)`.
#[inline]
#[allow(dead_code)] // Kept for parity with previous internal API surface.
pub fn poly_result_arg(ctx: &Context, id: ExprId) -> Option<ExprId> {
    cas_math::poly_result::poly_result_arg(ctx, id)
}

/// Parse a `poly_result(id)` wrapper and return the engine `PolyId`.
#[inline]
pub fn parse_poly_result_id(ctx: &Context, id: ExprId) -> Option<PolyId> {
    cas_math::poly_result::parse_poly_result_id(ctx, id)
}

/// Wrap an engine `PolyId` into `poly_result(id)`.
#[inline]
pub fn wrap_poly_result(ctx: &mut Context, poly_id: PolyId) -> ExprId {
    cas_math::poly_result::wrap_poly_result(ctx, poly_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_engine_poly_id() {
        let mut ctx = Context::new();
        let poly_id: PolyId = 42;

        let wrapped = wrap_poly_result(&mut ctx, poly_id);
        assert!(is_poly_result(&ctx, wrapped));
        assert_eq!(parse_poly_result_id(&ctx, wrapped), Some(poly_id));
    }
}
