mod classify;
mod det;

use num_rational::BigRational;
use num_traits::Zero;

use super::coeffs::LinearCoeffs3;
use super::LinearSystemError;

pub(crate) fn solve_3x3_cramer(
    c1: &LinearCoeffs3,
    c2: &LinearCoeffs3,
    c3: &LinearCoeffs3,
) -> Result<(BigRational, BigRational, BigRational), LinearSystemError> {
    let e1 = -c1.d.clone();
    let e2 = -c2.d.clone();
    let e3 = -c3.d.clone();

    let det_a = det::det3x3(
        &c1.a, &c1.b, &c1.c, &c2.a, &c2.b, &c2.c, &c3.a, &c3.b, &c3.c,
    );

    if det_a.is_zero() {
        return Err(classify::classify_degenerate_3x3(c1, c2, c3, &e1, &e2, &e3));
    }

    let det_x = det::det3x3(&e1, &c1.b, &c1.c, &e2, &c2.b, &c2.c, &e3, &c3.b, &c3.c);
    let det_y = det::det3x3(&c1.a, &e1, &c1.c, &c2.a, &e2, &c2.c, &c3.a, &e3, &c3.c);
    let det_z = det::det3x3(&c1.a, &c1.b, &e1, &c2.a, &c2.b, &e2, &c3.a, &c3.b, &e3);

    let x = det_x / &det_a;
    let y = det_y / &det_a;
    let z = det_z / &det_a;
    Ok((x, y, z))
}
