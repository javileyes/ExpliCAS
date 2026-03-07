use crate::didactic::SubStep;

pub(super) fn build_cancellation_substep() -> SubStep {
    SubStep {
        description: "Todos los términos se cancelan".to_string(),
        before_expr: "Expandir y agrupar".to_string(),
        after_expr: "= 0".to_string(),
        before_latex: Some("\\text{Expandir y agrupar}".to_string()),
        after_latex: Some("= 0".to_string()),
    }
}
