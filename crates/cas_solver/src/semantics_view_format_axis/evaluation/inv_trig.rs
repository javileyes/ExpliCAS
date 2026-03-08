use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_inv_trig_axis_lines(state: &SemanticsViewState) -> Vec<String> {
    let current = match state.inv_trig {
        crate::InverseTrigPolicy::Strict => "strict",
        crate::InverseTrigPolicy::PrincipalValue => "principal",
    };
    vec![
        format!("inv_trig: {}", current),
        "  Values: strict | principal".to_string(),
        "  strict:    arctan(tan(x)) unchanged".to_string(),
        "  principal: arctan(tan(x)) → x with warning".to_string(),
    ]
}
