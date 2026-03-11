use crate::substitute_subcommand_text::parse_substitute_wire_text_lines;
use crate::substitute_subcommand_types::{SubstituteCommandMode, SubstituteSubcommandOutput};
use crate::substitute_subcommand_wire::{
    evaluate_substitute_subcommand_wire, substitute_command_mode_str,
};

/// Evaluate substitute subcommand and map canonical wire/text contracts to CLI output.
pub fn evaluate_substitute_subcommand(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteCommandMode,
    steps_enabled: bool,
    wire_output: bool,
) -> Result<SubstituteSubcommandOutput, String> {
    if wire_output {
        let out =
            evaluate_substitute_subcommand_wire(expr, target, replacement, mode, steps_enabled);
        return Ok(SubstituteSubcommandOutput::Wire(out));
    }

    let mode = substitute_command_mode_str(mode);
    let opts = format!(
        "{{\"mode\":\"{}\",\"steps\":{},\"pretty\":false}}",
        mode, steps_enabled
    );
    let payload = crate::wire::substitute_str_to_wire(expr, target, replacement, Some(&opts));
    let lines = parse_substitute_wire_text_lines(&payload, steps_enabled)?;
    Ok(SubstituteSubcommandOutput::TextLines(lines))
}
