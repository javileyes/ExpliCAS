//! Helpers for AST -> `MultiPoly` conversion with additional guards.

use crate::multipoly::{multipoly_from_expr, MultiPoly, PolyBudget, PolyError};
use cas_ast::{Context, ExprId};

/// Convert an expression to `MultiPoly` and enforce a max variable count.
pub fn multipoly_from_expr_with_var_limit(
    ctx: &Context,
    expr: ExprId,
    budget: &PolyBudget,
    max_vars: usize,
) -> Result<MultiPoly, PolyError> {
    let poly = multipoly_from_expr(ctx, expr, budget)?;
    if poly.vars.len() > max_vars {
        return Err(PolyError::BudgetExceeded);
    }
    Ok(poly)
}

/// Fallible conversion helper returning `None` on failure.
pub fn try_multipoly_from_expr_with_var_limit(
    ctx: &Context,
    expr: ExprId,
    budget: &PolyBudget,
    max_vars: usize,
) -> Option<MultiPoly> {
    multipoly_from_expr_with_var_limit(ctx, expr, budget, max_vars).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn var_limit_accepts_small_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("x + y + 1", &mut ctx).expect("parse");
        let budget = PolyBudget {
            max_terms: 32,
            max_total_degree: 4,
            max_pow_exp: 4,
        };

        let poly = multipoly_from_expr_with_var_limit(&ctx, expr, &budget, 2).expect("poly");
        assert_eq!(poly.vars.len(), 2);
    }

    #[test]
    fn var_limit_rejects_large_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("x + y + z", &mut ctx).expect("parse");
        let budget = PolyBudget {
            max_terms: 32,
            max_total_degree: 4,
            max_pow_exp: 4,
        };

        let poly = try_multipoly_from_expr_with_var_limit(&ctx, expr, &budget, 2);
        assert!(poly.is_none());
    }
}
