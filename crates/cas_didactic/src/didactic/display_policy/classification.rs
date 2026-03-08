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
pub fn classify_sub_steps(sub_steps: &[SubStep]) -> SubStepClassification {
    SubStepClassification {
        has_fraction_sum: sub_steps.iter().any(|s| {
            s.description.contains("common denominator")
                || s.description.contains("Sum the fractions")
        }),
        has_factorization: sub_steps.iter().any(|s| {
            s.description.contains("Cancel common factor") || s.description.contains("Factor")
        }),
        has_nested_fraction: sub_steps.iter().any(|s| {
            s.description.contains("Combinar términos")
                || s.description.contains("Invertir la fracción")
                || s.description.contains("denominadores internos")
                || s.description.contains("Invertir")
                || s.description.contains("denominador")
        }),
        has_polynomial_identity: sub_steps.iter().any(|s| {
            s.description.contains("forma normal polinómica")
                || s.description.contains("Cancelar términos semejantes")
        }),
    }
}
