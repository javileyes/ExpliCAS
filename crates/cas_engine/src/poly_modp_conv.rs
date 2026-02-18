//! Compatibility adapter for Expr -> MultiPolyModP conversion.
//!
//! Canonical converter lives in `cas_math::poly_modp_conv`.
//! This adapter adds engine-only support for resolving `poly_result(id)`
//! from the thread-local `PolyStore`.

use cas_ast::{Context, Expr, ExprId};
use cas_math::multipoly_modp::MultiPolyModP;
use cas_math::poly_modp_conv::{expr_to_poly_modp_with_resolver, PolyResultResolver};
use num_traits::ToPrimitive;

pub use cas_math::poly_modp_conv::{
    strip_hold, PolyConvError, PolyModpBudget, VarTable, DEFAULT_PRIME,
};

struct EnginePolyResultResolver;

impl PolyResultResolver for EnginePolyResultResolver {
    fn resolve_poly_result(
        &self,
        ctx: &Context,
        p: u64,
        vars: &mut VarTable,
        id_expr: ExprId,
    ) -> Result<MultiPolyModP, PolyConvError> {
        use crate::poly_store::{thread_local_get_for_materialize, PolyId};

        let id_u32: u32 = match ctx.get(id_expr) {
            Expr::Number(n) => n.to_integer().to_u32().ok_or_else(|| {
                PolyConvError::UnsupportedExpr("poly_result id not valid integer".into())
            })?,
            _ => {
                return Err(PolyConvError::UnsupportedExpr(
                    "poly_result arg must be integer".into(),
                ))
            }
        };

        let poly_id: PolyId = id_u32;

        let (meta, poly) = thread_local_get_for_materialize(poly_id).ok_or_else(|| {
            PolyConvError::UnsupportedExpr(format!("invalid poly_result({})", poly_id))
        })?;

        if poly.p != p {
            return Err(PolyConvError::BadPrime(format!(
                "poly_result modulus {} differs from requested {}",
                poly.p, p
            )));
        }

        for name in &meta.var_names {
            vars.get_or_insert(name)
                .ok_or(PolyConvError::TooManyVariables)?;
        }

        let remap: Vec<usize> = meta
            .var_names
            .iter()
            .map(|name| {
                vars.get_index(name).ok_or_else(|| {
                    PolyConvError::UnsupportedExpr(format!(
                        "VarTable missing variable '{}' after insert",
                        name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(poly.remap(&remap, vars.len()))
    }
}

/// Convert Expr to MultiPolyModP.
///
/// Includes engine-specific `poly_result(id)` resolution through `PolyStore`.
pub fn expr_to_poly_modp(
    ctx: &Context,
    expr: ExprId,
    p: u64,
    budget: &PolyModpBudget,
    vars: &mut VarTable,
) -> Result<MultiPolyModP, PolyConvError> {
    expr_to_poly_modp_with_resolver(ctx, expr, p, budget, vars, &EnginePolyResultResolver)
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
