mod back_substitute;
mod classify;
mod eliminate;

use cas_ast::{Context, ExprId};
use num_rational::BigRational;

use super::coeffs::build_augmented_matrix;
use super::{LinSolveResult, LinearSystemError};

#[allow(clippy::needless_range_loop)]
fn gauss_solve(mut matrix: Vec<Vec<BigRational>>, n: usize) -> LinSolveResult {
    let pivot_cols = eliminate::reduce_to_row_echelon(&mut matrix, n);

    if let Some(result) = classify::classify_reduced_system(&matrix, n, &pivot_cols) {
        return result;
    }

    LinSolveResult::Unique(back_substitute::solve_unique_solution(
        &matrix,
        n,
        &pivot_cols,
    ))
}

pub(crate) fn solve_nxn_gauss(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[&str],
) -> Result<LinSolveResult, LinearSystemError> {
    let n = vars.len();
    let matrix = build_augmented_matrix(ctx, exprs, vars)?;
    Ok(gauss_solve(matrix, n))
}
