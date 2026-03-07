use num_rational::BigRational;
use num_traits::Zero;

#[allow(clippy::needless_range_loop)]
pub(super) fn reduce_to_row_echelon(matrix: &mut [Vec<BigRational>], n: usize) -> Vec<usize> {
    let m = matrix.len();
    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();

    for col in 0..n {
        let mut pivot_found = None;
        for (row, mat_row) in matrix.iter().enumerate().skip(pivot_row) {
            if !mat_row[col].is_zero() {
                pivot_found = Some(row);
                break;
            }
        }

        let Some(pivot_idx) = pivot_found else {
            continue;
        };

        if pivot_idx != pivot_row {
            matrix.swap(pivot_row, pivot_idx);
        }

        pivot_cols.push(col);

        let pivot_val = matrix[pivot_row][col].clone();
        for cell in matrix[pivot_row].iter_mut().take(n + 1) {
            *cell = &*cell / &pivot_val;
        }

        for row in (pivot_row + 1)..m {
            if !matrix[row][col].is_zero() {
                let factor = matrix[row][col].clone();
                for j in 0..=n {
                    let subtrahend = &factor * &matrix[pivot_row][j];
                    matrix[row][j] -= subtrahend;
                }
            }
        }

        pivot_row += 1;
        if pivot_row >= m {
            break;
        }
    }

    pivot_cols
}
