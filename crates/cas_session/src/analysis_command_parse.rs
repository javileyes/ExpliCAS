//! Parsing helpers for analysis command invocations.

/// Extract tail of `equiv ...` input.
pub fn extract_equiv_command_tail(line: &str) -> &str {
    line.strip_prefix("equiv").unwrap_or(line).trim()
}

/// Extract tail of `subst ...` input.
pub fn extract_substitute_command_tail(line: &str) -> &str {
    line.strip_prefix("subst").unwrap_or(line).trim()
}

/// Extract tail of `explain ...` input.
pub fn extract_explain_command_tail(line: &str) -> &str {
    line.strip_prefix("explain").unwrap_or(line).trim()
}

/// Extract tail of `visualize ...` / `viz ...` input.
pub fn extract_visualize_command_tail(line: &str) -> &str {
    line.strip_prefix("visualize ")
        .or_else(|| line.strip_prefix("viz "))
        .unwrap_or(line)
        .trim()
}
