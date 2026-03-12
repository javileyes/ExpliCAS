use cas_solver_core::semantics_view_types::SemanticsViewState;

pub(super) fn format_domain_axis_lines(state: &SemanticsViewState) -> Vec<String> {
    let current = match state.domain_mode {
        crate::DomainMode::Strict => "strict",
        crate::DomainMode::Assume => "assume",
        crate::DomainMode::Generic => "generic",
    };
    vec![
        format!("domain: {}", current),
        "  Values: strict | generic | assume".to_string(),
        "  strict:  No domain assumptions (x/x stays x/x)".to_string(),
        "  generic: Classic CAS 'almost everywhere' algebra".to_string(),
        "  assume:  Use assumptions with warnings".to_string(),
    ]
}
