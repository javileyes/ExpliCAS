mod classify;

use num_rational::BigRational;
use num_traits::Zero;

use super::coeffs::LinearCoeffs;
use super::LinearSystemError;

pub(crate) fn solve_2x2_cramer(
    coeffs1: &LinearCoeffs,
    coeffs2: &LinearCoeffs,
) -> Result<(BigRational, BigRational), LinearSystemError> {
    let a1 = &coeffs1.a;
    let b1 = &coeffs1.b;
    let d1 = -coeffs1.c.clone();

    let a2 = &coeffs2.a;
    let b2 = &coeffs2.b;
    let d2 = -coeffs2.c.clone();

    let det = a1 * b2 - a2 * b1;
    if det.is_zero() {
        return Err(classify::classify_degenerate_2x2(a1, b1, &d1, a2, b2, &d2));
    }

    let x = (&d1 * b2 - b1 * &d2) / &det;
    let y = (a1 * &d2 - &d1 * a2) / &det;
    Ok((x, y))
}
