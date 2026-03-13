use num_rational::BigRational;
use num_traits::Zero;

use super::super::LinSolveResult;

pub(super) fn classify_reduced_system(
    matrix: &[Vec<BigRational>],
    n: usize,
    pivot_cols: &[usize],
) -> Option<LinSolveResult> {
    let rank = pivot_cols.len();

    for mat_row in matrix.iter().skip(rank) {
        let all_zero = (0..n).all(|j| mat_row[j].is_zero());
        if all_zero && !mat_row[n].is_zero() {
            return Some(LinSolveResult::Inconsistent);
        }
    }

    if rank < n {
        return Some(LinSolveResult::Infinite);
    }

    None
}
