use num_rational::BigRational;

use super::super::coeffs::LinearCoeffs3;
use super::super::types::LinearSystemError;

pub(super) fn classify_degenerate_3x3(
    c1: &LinearCoeffs3,
    c2: &LinearCoeffs3,
    c3: &LinearCoeffs3,
    e1: &BigRational,
    e2: &BigRational,
    e3: &BigRational,
) -> LinearSystemError {
    let pair1_consistent = check_proportional_3(&c1.a, &c1.b, &c1.c, e1, &c2.a, &c2.b, &c2.c, e2);
    let pair2_consistent = check_proportional_3(&c1.a, &c1.b, &c1.c, e1, &c3.a, &c3.b, &c3.c, e3);
    let pair3_consistent = check_proportional_3(&c2.a, &c2.b, &c2.c, e2, &c3.a, &c3.b, &c3.c, e3);

    if pair1_consistent && pair2_consistent && pair3_consistent {
        LinearSystemError::InfiniteSolutions
    } else {
        LinearSystemError::NoSolution
    }
}

#[allow(clippy::too_many_arguments)]
fn check_proportional_3(
    a1: &BigRational,
    b1: &BigRational,
    c1: &BigRational,
    e1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    c2: &BigRational,
    e2: &BigRational,
) -> bool {
    let ab = a1 * b2 == a2 * b1;
    let ac = a1 * c2 == a2 * c1;
    let ae = a1 * e2 == a2 * e1;
    let bc = b1 * c2 == b2 * c1;
    let be = b1 * e2 == b2 * e1;
    let ce = c1 * e2 == c2 * e1;

    ab && ac && ae && bc && be && ce
}
