use cas_ast::{Context, ExprId};
use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

use super::super::LinearSystemError;

pub(super) fn build_linear_poly(
    ctx: &Context,
    expr: ExprId,
    budget: PolyBudget,
) -> Result<cas_math::multipoly::MultiPoly, LinearSystemError> {
    let poly =
        multipoly_from_expr(ctx, expr, &budget).map_err(LinearSystemError::PolyConversion)?;

    if poly.total_degree() > 1 {
        return Err(LinearSystemError::NotLinear(
            "degree > 1 in the system".to_string(),
        ));
    }

    Ok(poly)
}

pub(super) fn non_linear_degree_error(total_exp: u32) -> LinearSystemError {
    LinearSystemError::NotLinear(format!("non-linear term with degree {}", total_exp))
}

pub(super) fn unexpected_variable_error(name: &str) -> LinearSystemError {
    LinearSystemError::NotLinear(format!("unexpected variable '{}'", name))
}
