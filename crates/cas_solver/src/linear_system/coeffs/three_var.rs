mod extract;

use cas_ast::{Context, ExprId};

use super::super::LinearSystemError;
use num_rational::BigRational;

/// Linear coefficients for equation `a*x + b*y + c*z + d = 0` (3 variables).
#[derive(Debug, Clone)]
pub(crate) struct LinearCoeffs3 {
    pub(crate) a: BigRational,
    pub(crate) b: BigRational,
    pub(crate) c: BigRational,
    pub(crate) d: BigRational,
}

pub(crate) fn extract_linear_coeffs_3(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
    var_z: &str,
) -> Result<LinearCoeffs3, LinearSystemError> {
    extract::extract_linear_coeffs_3(ctx, expr, var_x, var_y, var_z)
}
