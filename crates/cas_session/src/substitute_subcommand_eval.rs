use crate::substitute_subcommand_json::{
    evaluate_substitute_subcommand_json_canonical, substitute_command_mode_str,
};
use crate::substitute_subcommand_text::parse_substitute_json_text_lines;
use crate::substitute_subcommand_types::{SubstituteCommandMode, SubstituteSubcommandOutput};

/// Evaluate substitute subcommand and map solver contracts to session-layer output.
pub fn evaluate_substitute_subcommand(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteCommandMode,
    steps_enabled: bool,
    json_output: bool,
) -> Result<SubstituteSubcommandOutput, String> {
    if json_output {
        let out = evaluate_substitute_subcommand_json_canonical(
            expr,
            target,
            replacement,
            mode,
            steps_enabled,
        );
        return Ok(SubstituteSubcommandOutput::Json(out));
    }

    let mode = substitute_command_mode_str(mode);
    let opts = format!(
        "{{\"mode\":\"{}\",\"steps\":{},\"pretty\":false}}",
        mode, steps_enabled
    );
    let payload = crate::evaluate_substitute_json_canonical(expr, target, replacement, Some(&opts));
    let lines = parse_substitute_json_text_lines(&payload, steps_enabled)?;
    Ok(SubstituteSubcommandOutput::TextLines(lines))
}
