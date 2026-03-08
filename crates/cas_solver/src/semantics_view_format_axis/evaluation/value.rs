use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_value_axis_lines(state: &SemanticsViewState) -> Vec<String> {
    let current = match state.value_domain {
        crate::ValueDomain::RealOnly => "real",
        crate::ValueDomain::ComplexEnabled => "complex",
    };
    vec![
        format!("value: {}", current),
        "  Values: real | complex".to_string(),
        "  real:    ℝ only (sqrt(-1) undefined)".to_string(),
        "  complex: ℂ enabled (sqrt(-1) = i)".to_string(),
    ]
}
