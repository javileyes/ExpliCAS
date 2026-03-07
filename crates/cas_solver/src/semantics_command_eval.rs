mod mutate;
mod view;

use crate::{parse_semantics_command_input, SemanticsCommandInput, SemanticsCommandOutput};

/// Evaluate `semantics ...` command line and apply runtime option changes.
pub fn evaluate_semantics_command_line(
    line: &str,
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> SemanticsCommandOutput {
    match parse_semantics_command_input(line) {
        SemanticsCommandInput::Show => view::show_semantics_output(simplify_options, eval_options),
        SemanticsCommandInput::Help => view::help_semantics_output(),
        SemanticsCommandInput::Set { args } => {
            mutate::set_semantics_output(args, simplify_options, eval_options)
        }
        SemanticsCommandInput::Axis { axis } => {
            view::axis_semantics_output(simplify_options, eval_options, &axis)
        }
        SemanticsCommandInput::Preset { args } => {
            mutate::preset_semantics_output(args, simplify_options, eval_options)
        }
        SemanticsCommandInput::Unknown { subcommand } => {
            view::unknown_semantics_output(&subcommand)
        }
    }
}
