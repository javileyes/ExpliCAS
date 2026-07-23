use num_rational::BigRational;

use super::super::coeffs::LinearCoeffs3;
use super::super::{gauss, LinSolveResult, LinearSystemError};

pub(super) fn classify_degenerate_3x3(
    c1: &LinearCoeffs3,
    c2: &LinearCoeffs3,
    c3: &LinearCoeffs3,
    e1: &BigRational,
    e2: &BigRational,
    e3: &BigRational,
) -> LinearSystemError {
    let matrix = vec![
        vec![c1.a.clone(), c1.b.clone(), c1.c.clone(), e1.clone()],
        vec![c2.a.clone(), c2.b.clone(), c2.c.clone(), e2.clone()],
        vec![c3.a.clone(), c3.b.clone(), c3.c.clone(), e3.clone()],
    ];

    match gauss::classify_augmented_matrix(matrix, 3) {
        LinSolveResult::Infinite => LinearSystemError::InfiniteSolutions,
        LinSolveResult::Inconsistent => LinearSystemError::NoSolution,
        LinSolveResult::Unique(_) | LinSolveResult::UniqueExpr { .. } => {
            debug_assert!(
                false,
                "3x3 Gaussian fallback reported a unique solution despite zero determinant"
            );
            LinearSystemError::NoSolution
        }
    }
}
