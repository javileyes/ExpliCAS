use num_rational::BigRational;

use super::super::LinearSystemError;

pub(super) fn classify_degenerate_2x2(
    a1: &BigRational,
    b1: &BigRational,
    d1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    d2: &BigRational,
) -> LinearSystemError {
    let lhs_consistent = d1 * b2 == d2 * b1;
    let rhs_consistent = d1 * a2 == d2 * a1;

    if lhs_consistent && rhs_consistent {
        LinearSystemError::InfiniteSolutions
    } else {
        LinearSystemError::NoSolution
    }
}
