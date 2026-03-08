use super::super::super::SubStep;

pub(super) fn build_combine_numerator_substep(num_str: &str) -> SubStep {
    SubStep {
        description: "Combinar términos del numerador (denominador común)".to_string(),
        before_expr: num_str.to_string(),
        after_expr: "(numerador combinado) / B".to_string(),
        before_latex: None,
        after_latex: None,
    }
}
