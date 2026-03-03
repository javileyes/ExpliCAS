use crate::substitute_subcommand_types::SubstituteCommandMode;

pub(crate) fn substitute_command_mode_str(mode: SubstituteCommandMode) -> &'static str {
    match mode {
        SubstituteCommandMode::Exact => "exact",
        SubstituteCommandMode::Power => "power",
    }
}

/// Evaluate substitute subcommand in canonical JSON mode.
///
/// Uses session-level JSON bridge as canonical serializer.
pub fn evaluate_substitute_subcommand_json_canonical(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteCommandMode,
    steps_enabled: bool,
) -> String {
    let mode = substitute_command_mode_str(mode);
    let opts = format!(
        "{{\"mode\":\"{}\",\"steps\":{},\"pretty\":true}}",
        mode, steps_enabled
    );
    crate::evaluate_substitute_json_canonical(expr, target, replacement, Some(&opts))
}
