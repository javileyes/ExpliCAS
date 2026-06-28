use super::super::SubStep;

pub(super) fn build_nested_fraction_default_substeps(
    before_str: &str,
    after_str: &str,
) -> Vec<SubStep> {
    vec![SubStep {
        description: "Simplificar fracción anidada".to_string(),
        before_expr: before_str.to_string(),
        after_expr: after_str.to_string(),
        before_latex: None,
        after_latex: None,
        desc_key: None,
        desc_args: Vec::new(),
    }]
}

pub(super) fn build_general_expression_default_substeps(
    before_str: &str,
    after_str: &str,
) -> Vec<SubStep> {
    vec![SubStep {
        description: "Simplificar expresión".to_string(),
        before_expr: before_str.to_string(),
        after_expr: after_str.to_string(),
        before_latex: None,
        after_latex: None,
        desc_key: None,
        desc_args: Vec::new(),
    }]
}
