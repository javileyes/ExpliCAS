use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_const_fold_axis_lines(state: &SemanticsViewState) -> Vec<String> {
    let current = match state.const_fold {
        crate::ConstFoldMode::Off => "off",
        crate::ConstFoldMode::Safe => "safe",
    };
    vec![
        format!("const_fold: {}", current),
        "  Values: off | safe".to_string(),
        "  off:  No constant folding (defer semantic decisions)".to_string(),
        "  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)".to_string(),
    ]
}
