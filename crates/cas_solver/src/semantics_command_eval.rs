use crate::{parse_semantics_command_input, SemanticsCommandInput, SemanticsCommandOutput};

/// Evaluate `semantics ...` command line and apply runtime option changes.
pub fn evaluate_semantics_command_line(
    line: &str,
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> SemanticsCommandOutput {
    match parse_semantics_command_input(line) {
        SemanticsCommandInput::Show => {
            let state = crate::semantics_view_state_from_options(simplify_options, eval_options);
            SemanticsCommandOutput {
                lines: crate::format_semantics_overview_lines(&state),
                sync_simplifier: false,
            }
        }
        SemanticsCommandInput::Help => SemanticsCommandOutput {
            lines: vec![crate::semantics_help_message().to_string()],
            sync_simplifier: false,
        },
        SemanticsCommandInput::Set { args } => {
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            match crate::evaluate_semantics_set_args_to_overview_lines(
                &refs,
                simplify_options,
                eval_options,
            ) {
                Ok(lines) => SemanticsCommandOutput {
                    lines,
                    sync_simplifier: true,
                },
                Err(error) => SemanticsCommandOutput {
                    lines: vec![error],
                    sync_simplifier: false,
                },
            }
        }
        SemanticsCommandInput::Axis { axis } => {
            let state = crate::semantics_view_state_from_options(simplify_options, eval_options);
            SemanticsCommandOutput {
                lines: crate::format_semantics_axis_lines(&state, &axis),
                sync_simplifier: false,
            }
        }
        SemanticsCommandInput::Preset { args } => {
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            let out = crate::evaluate_semantics_preset_args_to_options(
                &refs,
                simplify_options,
                eval_options,
            );
            SemanticsCommandOutput {
                lines: out.lines,
                sync_simplifier: out.applied,
            }
        }
        SemanticsCommandInput::Unknown { subcommand } => SemanticsCommandOutput {
            lines: vec![crate::format_semantics_unknown_subcommand_message(
                &subcommand,
            )],
            sync_simplifier: false,
        },
    }
}
