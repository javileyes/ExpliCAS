use num_rational::BigRational;

#[allow(clippy::too_many_arguments)]
pub(super) fn det3x3(
    a1: &BigRational,
    b1: &BigRational,
    c1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    c2: &BigRational,
    a3: &BigRational,
    b3: &BigRational,
    c3: &BigRational,
) -> BigRational {
    a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2)
}
