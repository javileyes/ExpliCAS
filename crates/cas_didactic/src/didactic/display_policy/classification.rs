use super::SubStep;

/// Stable classification of didactic sub-steps used by CLI and timeline renderers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubStepClassification {
    pub has_fraction_sum: bool,
    pub has_factorization: bool,
    pub has_nested_fraction: bool,
    pub has_polynomial_identity: bool,
}

/// Classify a sub-step block by its didactic content.
pub(crate) fn classify_sub_steps(sub_steps: &[SubStep]) -> SubStepClassification {
    let descriptions: Vec<String> = sub_steps
        .iter()
        .map(|s| s.description.to_lowercase())
        .collect();

    SubStepClassification {
        has_fraction_sum: descriptions.iter().any(|s| {
            s.contains("common denominator")
                || s.contains("sum the fractions")
                || s.contains("denominador común")
                || s.contains("sumar fracciones")
        }),
        has_factorization: descriptions
            .iter()
            .any(|s| s.contains("cancel common factor") || s.contains("factor")),
        has_nested_fraction: descriptions.iter().any(|s| {
            s.contains("combinar términos")
                || s.contains("invertir la fracción")
                || s.contains("invertirla")
                || s.contains("recíproco")
                || s.contains("fracción")
                || s.contains("denominador")
        }),
        has_polynomial_identity: descriptions.iter().any(|s| {
            s.contains("forma normal polinómica") || s.contains("cancelar términos semejantes")
        }),
    }
}
