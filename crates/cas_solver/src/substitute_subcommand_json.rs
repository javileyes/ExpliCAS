use crate::substitute_subcommand_types::SubstituteCommandMode;

pub(crate) fn substitute_command_mode_str(mode: SubstituteCommandMode) -> &'static str {
    match mode {
        SubstituteCommandMode::Exact => "exact",
        SubstituteCommandMode::Power => "power",
    }
}

/// Evaluate substitute subcommand in canonical JSON mode.
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
    crate::substitute_str_to_json(expr, target, replacement, Some(&opts))
}
