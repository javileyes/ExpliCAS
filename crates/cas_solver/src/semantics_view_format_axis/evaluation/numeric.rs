use crate::SemanticsViewState;

pub(super) fn format_numeric_axis_lines(state: &SemanticsViewState) -> Vec<String> {
    let current = match state.numeric_display {
        crate::NumericDisplayMode::Exact => "exact",
        crate::NumericDisplayMode::Decimal => "decimal",
    };
    vec![
        format!("numeric: {}", current),
        "  Values: exact | decimal".to_string(),
        "  exact:   Exact fractions and radicals (1/3 + 1/3 → 2/3)".to_string(),
        "  decimal: Approximate results at the output boundary".to_string(),
        "           (internally the engine stays exact and symbolic)".to_string(),
    ]
}
