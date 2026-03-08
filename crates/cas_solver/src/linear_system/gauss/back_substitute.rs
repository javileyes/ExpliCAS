use num_rational::BigRational;
use num_traits::Zero;

pub(super) fn solve_unique_solution(
    matrix: &[Vec<BigRational>],
    n: usize,
    pivot_cols: &[usize],
) -> Vec<BigRational> {
    let rank = pivot_cols.len();
    let mut solution = vec![BigRational::zero(); n];

    for i in (0..rank).rev() {
        let col = pivot_cols[i];
        let mut val = matrix[i][n].clone();

        for j in (col + 1)..n {
            val -= &matrix[i][j] * &solution[j];
        }
        solution[col] = val;
    }

    solution
}
