mod extract;

use cas_ast::{Context, ExprId};
use num_rational::BigRational;

/// Linear coefficients for equation `a*x + b*y + c = 0` (2 variables).
#[derive(Debug, Clone)]
pub(crate) struct LinearCoeffs {
    pub(crate) a: BigRational,
    pub(crate) b: BigRational,
    pub(crate) c: BigRational,
}

pub(crate) fn extract_linear_coeffs(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<LinearCoeffs, super::super::types::LinearSystemError> {
    extract::extract_linear_coeffs(ctx, expr, var_x, var_y)
}
