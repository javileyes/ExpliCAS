use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_branch_axis_lines(state: &SemanticsViewState) -> Vec<String> {
    let current = match state.branch {
        crate::BranchPolicy::Principal => "principal",
    };
    let inactive = state.value_domain == crate::ValueDomain::RealOnly;
    let mut lines = Vec::new();
    if inactive {
        lines.push(format!("branch: {} (inactive: value=real)", current));
    } else {
        lines.push(format!("branch: {}", current));
    }
    lines.push("  Values: principal".to_string());
    lines.push("  principal: Use principal branch for multi-valued functions".to_string());
    if inactive {
        lines.push("  Note: Only active when value=complex".to_string());
    }
    lines
}
