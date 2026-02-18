//! Compatibility adapter for Expr -> MultiPolyModP conversion.
//!
//! Canonical converter lives in `cas_math::poly_modp_conv`.
//! This layer preserves the engine API shape where `expr_to_poly_modp`
//! resolves `poly_result(id)` from the thread-local store.

use cas_ast::{Context, ExprId};
use cas_math::multipoly_modp::MultiPolyModP;

pub use cas_math::poly_modp_conv::{
    strip_hold, PolyConvError, PolyModpBudget, VarTable, DEFAULT_PRIME,
};

/// Convert Expr to MultiPolyModP.
///
/// Engine compatibility behavior: resolves `poly_result(id)` through
/// the thread-local poly store.
pub fn expr_to_poly_modp(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
) -> Result<MultiPolyModP, PolyConvError> {
    cas_math::poly_modp_conv::expr_to_poly_modp_with_store(ctx, expr, p, budget, vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_simple_var() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x", &mut ctx).unwrap();
        let mut vars = VarTable::new();
        let poly =
            expr_to_poly_modp(&ctx, expr, 17, &PolyModpBudget::default(), &mut vars).unwrap();
        assert_eq!(poly.num_terms(), 1);
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn test_linear_sum() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("1 + 2*x + 3*y", &mut ctx).unwrap();
        let mut vars = VarTable::new();
        let poly =
            expr_to_poly_modp(&ctx, expr, 17, &PolyModpBudget::default(), &mut vars).unwrap();
        assert_eq!(poly.num_terms(), 3);
    }

    #[test]
    fn test_pow_fast_path() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("(1 + x)^3", &mut ctx).unwrap();
        let mut vars = VarTable::new();
        let poly =
            expr_to_poly_modp(&ctx, expr, 17, &PolyModpBudget::default(), &mut vars).unwrap();
        assert_eq!(poly.num_terms(), 4);
    }
}
