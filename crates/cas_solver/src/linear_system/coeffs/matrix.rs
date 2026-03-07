mod row;

use cas_ast::{Context, ExprId};
use num_rational::BigRational;

use super::super::types::LinearSystemError;

pub(crate) fn build_augmented_matrix(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<Vec<Vec<BigRational>>, LinearSystemError> {
    let n = vars.len();
    let m = exprs.len();
    let mut matrix = Vec::with_capacity(m);

    for (i, &expr) in exprs.iter().enumerate() {
        let (coeffs, b) = row::extract_linear_row(ctx, expr, vars).map_err(|e| match e {
            LinearSystemError::NotLinear(msg) => {
                LinearSystemError::NotLinear(format!("equation {}: {}", i + 1, msg))
            }
            other => other,
        })?;

        if coeffs.len() != n {
            return Err(LinearSystemError::NotLinear(format!(
                "equation {} has wrong number of coefficients",
                i + 1
            )));
        }

        let mut row = coeffs;
        row.push(b);
        matrix.push(row);
    }

    Ok(matrix)
}
