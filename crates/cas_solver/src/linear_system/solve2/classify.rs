use num_rational::BigRational;
use num_traits::Zero;

use super::super::LinearSystemError;

pub(super) fn classify_degenerate_2x2(
    a1: &BigRational,
    b1: &BigRational,
    d1: &BigRational,
    a2: &BigRational,
    b2: &BigRational,
    d2: &BigRational,
) -> LinearSystemError {
    // A row `0·x + 0·y = d` with `d ≠ 0` is an outright contradiction — no (x, y) satisfies it — so
    // the whole system is INCONSISTENT (no solution). The proportionality cross-products below cannot
    // detect this when BOTH coefficient rows are zero: `d1·b2 == d2·b1` and `d1·a2 == d2·a1` both
    // reduce to `0 == 0` regardless of the right-hand sides, so an all-zero coefficient matrix with a
    // nonzero RHS (`0 = 1`) was wrongly reported as InfiniteSolutions. Check the contradiction first.
    let row1_contradiction = a1.is_zero() && b1.is_zero() && !d1.is_zero();
    let row2_contradiction = a2.is_zero() && b2.is_zero() && !d2.is_zero();
    if row1_contradiction || row2_contradiction {
        return LinearSystemError::NoSolution;
    }

    let lhs_consistent = d1 * b2 == d2 * b1;
    let rhs_consistent = d1 * a2 == d2 * a1;

    if lhs_consistent && rhs_consistent {
        LinearSystemError::InfiniteSolutions
    } else {
        LinearSystemError::NoSolution
    }
}
