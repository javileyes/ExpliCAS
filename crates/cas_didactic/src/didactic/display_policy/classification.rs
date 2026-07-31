use super::SubStep;

/// Stable classification of didactic sub-steps used by CLI and timeline renderers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubStepClassification {
    pub has_fraction_sum: bool,
    pub has_factorization: bool,
    pub has_nested_fraction: bool,
    pub has_polynomial_identity: bool,
}

/// The fraction-sum generator's own sub-steps, recognized by KEY. Their Spanish
/// descriptions contain "denominador"/"fracciones", which the substring sniff
/// below would misread as nested-fraction narration — the key is exact.
fn is_fraction_sum_key(sub_step: &SubStep) -> bool {
    matches!(
        sub_step.desc_key,
        Some("fraction.find_common_denominator" | "fraction.sum_fractions")
    )
}

/// Classify a sub-step block by its didactic content.
pub(crate) fn classify_sub_steps(sub_steps: &[SubStep]) -> SubStepClassification {
    let descriptions: Vec<String> = sub_steps
        .iter()
        .map(|s| s.description.to_lowercase())
        .collect();

    SubStepClassification {
        has_fraction_sum: sub_steps.iter().any(is_fraction_sum_key)
            || descriptions.iter().any(|s| {
                s.contains("common denominator")
                    || s.contains("sum the fractions")
                    || s.contains("denominador común")
                    || s.contains("sumar fracciones")
            }),
        has_factorization: descriptions
            .iter()
            .any(|s| s.contains("cancel common factor") || s.contains("factor")),
        has_nested_fraction: sub_steps.iter().any(|s| {
            if is_fraction_sum_key(s) {
                return false;
            }
            let d = s.description.to_lowercase();
            d.contains("combinar términos")
                || d.contains("invertir la fracción")
                || d.contains("invertirla")
                || d.contains("recíproco")
                || d.contains("fracción")
                || d.contains("denominador")
        }),
        has_polynomial_identity: descriptions.iter().any(|s| {
            s.contains("forma normal polinómica") || s.contains("cancelar términos semejantes")
        }),
    }
}
