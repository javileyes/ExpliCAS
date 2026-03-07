use super::super::classification::SubStepClassification;

pub(super) fn should_dedupe_once(classification: SubStepClassification) -> bool {
    classification.has_fraction_sum
        && !classification.has_nested_fraction
        && !classification.has_factorization
        && !classification.has_polynomial_identity
}
