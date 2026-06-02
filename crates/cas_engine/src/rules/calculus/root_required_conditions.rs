use super::polynomial_support::polynomial_is_strictly_positive_everywhere;
use cas_ast::ExprId;
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::Signed;

pub(super) fn shifted_sqrt_product_required_conditions(
    radicand: ExprId,
    shift: &BigRational,
    shifted_sqrt: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let mut required_conditions = vec![crate::ImplicitCondition::Positive(radicand)];
    if !shift.is_positive() {
        required_conditions.push(crate::ImplicitCondition::NonZero(shifted_sqrt));
    }
    required_conditions
}

pub(super) fn positive_polynomial_radicand_required_conditions(
    radicand: ExprId,
    radicand_poly: &Polynomial,
) -> Vec<crate::ImplicitCondition> {
    if polynomial_is_strictly_positive_everywhere(radicand_poly) {
        Vec::new()
    } else {
        vec![crate::ImplicitCondition::Positive(radicand)]
    }
}

pub(super) fn positive_polynomial_radicand_and_nonzero_required_conditions(
    radicand: ExprId,
    radicand_poly: &Polynomial,
    nonzero_witness: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let mut required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, radicand_poly);
    required_conditions.push(crate::ImplicitCondition::NonZero(nonzero_witness));
    required_conditions
}
