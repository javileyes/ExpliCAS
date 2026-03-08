use cas_ast::{Context, ExprId};
use num_rational::BigRational;

mod coeffs;
mod gauss;
mod solve2;
mod solve3;
mod types;

use self::coeffs::{extract_linear_coeffs, extract_linear_coeffs_3};
use self::gauss::solve_nxn_gauss;
use self::solve2::solve_2x2_cramer;
use self::solve3::solve_3x3_cramer;
use self::types::with_equation_index;

pub use self::types::{LinSolveResult, LinearSystemError};

/// Solve 2x2 linear system from normalized equations `lhs - rhs = 0`.
pub fn solve_2x2_linear_system(
    ctx: &Context,
    expr1: ExprId,
    expr2: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<(BigRational, BigRational), LinearSystemError> {
    let coeffs1 =
        extract_linear_coeffs(ctx, expr1, var_x, var_y).map_err(|e| with_equation_index(e, 1))?;
    let coeffs2 =
        extract_linear_coeffs(ctx, expr2, var_x, var_y).map_err(|e| with_equation_index(e, 2))?;
    solve_2x2_cramer(&coeffs1, &coeffs2)
}

/// Solve 3x3 linear system from normalized equations `lhs - rhs = 0`.
pub fn solve_3x3_linear_system(
    ctx: &Context,
    expr1: ExprId,
    expr2: ExprId,
    expr3: ExprId,
    var_x: &str,
    var_y: &str,
    var_z: &str,
) -> Result<(BigRational, BigRational, BigRational), LinearSystemError> {
    let coeffs1 = extract_linear_coeffs_3(ctx, expr1, var_x, var_y, var_z)
        .map_err(|e| with_equation_index(e, 1))?;
    let coeffs2 = extract_linear_coeffs_3(ctx, expr2, var_x, var_y, var_z)
        .map_err(|e| with_equation_index(e, 2))?;
    let coeffs3 = extract_linear_coeffs_3(ctx, expr3, var_x, var_y, var_z)
        .map_err(|e| with_equation_index(e, 3))?;
    solve_3x3_cramer(&coeffs1, &coeffs2, &coeffs3)
}

/// Solve n×n linear system from normalized equations `lhs - rhs = 0`.
pub fn solve_nxn_linear_system(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<LinSolveResult, LinearSystemError> {
    solve_nxn_gauss(ctx, exprs, vars)
}
