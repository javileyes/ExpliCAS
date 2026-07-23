use cas_ast::{Context, ExprId};
use cas_math::multipoly::PolyError;
use num_rational::BigRational;

mod coeffs;
mod gauss;
mod solve2;
mod solve3;
mod symbolic2;

use self::coeffs::{extract_linear_coeffs, extract_linear_coeffs_3};
use self::gauss::solve_nxn_gauss;
use self::solve2::solve_2x2_cramer;
use self::solve3::solve_3x3_cramer;
pub(crate) use self::symbolic2::{
    poly_determinant, solve_2x2_symbolic, solve_nxn_symbolic, Symbolic2x2Outcome,
};

/// Error type for linear system solving.
#[derive(Debug)]
pub enum LinearSystemError {
    NotLinear(String),
    /// System has infinitely many solutions (dependent equations).
    InfiniteSolutions,
    /// System has no solution (inconsistent equations).
    NoSolution,
    PolyConversion(PolyError),
}

impl std::fmt::Display for LinearSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearSystemError::NotLinear(msg) => write!(f, "non-linear term: {}", msg),
            LinearSystemError::InfiniteSolutions => {
                write!(
                    f,
                    "system has infinitely many solutions (dependent equations)"
                )
            }
            LinearSystemError::NoSolution => {
                write!(f, "system has no solution (inconsistent equations)")
            }
            LinearSystemError::PolyConversion(e) => write!(f, "polynomial conversion: {}", e),
        }
    }
}

/// Result of solving a linear system.
#[derive(Debug)]
pub enum LinSolveResult {
    /// Unique solution: values for each variable in order.
    Unique(Vec<BigRational>),
    /// Unique solution with symbolic parameter coefficients: expression values
    /// plus the `expr ≠ 0` requirements (determinants) they are valid under.
    UniqueExpr {
        values: Vec<ExprId>,
        nonzero_conditions: Vec<ExprId>,
    },
    /// Finite solution PAIRS of a nonlinear system (each inner vec aligned
    /// with the requested unknown order); every pair verified exactly against
    /// both original equations before emission.
    SolutionPairs(Vec<Vec<ExprId>>),
    /// Infinitely many solutions (dependent equations).
    Infinite,
    /// No solution (inconsistent equations).
    Inconsistent,
}

pub(crate) fn with_equation_index(error: LinearSystemError, index: usize) -> LinearSystemError {
    match error {
        LinearSystemError::NotLinear(message) => {
            LinearSystemError::NotLinear(format!("equation {}: {}", index, message))
        }
        other => other,
    }
}

/// Solve 2x2 linear system from normalized equations `lhs - rhs = 0`.
pub(crate) fn solve_2x2_linear_system(
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
pub(crate) fn solve_3x3_linear_system(
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
pub(crate) fn solve_nxn_linear_system(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<LinSolveResult, LinearSystemError> {
    solve_nxn_gauss(ctx, exprs, vars)
}
