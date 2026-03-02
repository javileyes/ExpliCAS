/// Parsed top-level `semantics ...` command shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticsCommandInput {
    Show,
    Help,
    Set { args: Vec<String> },
    Axis { axis: String },
    Preset { args: Vec<String> },
    Unknown { subcommand: String },
}

/// Evaluated output for a full `semantics ...` command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsCommandOutput {
    pub lines: Vec<String>,
    pub sync_simplifier: bool,
}

/// Parse a full `semantics ...` command line.
pub fn parse_semantics_command_input(line: &str) -> SemanticsCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => SemanticsCommandInput::Show,
        Some(&"help") => SemanticsCommandInput::Help,
        Some(&"set") => SemanticsCommandInput::Set {
            args: args[2..].iter().map(|s| (*s).to_string()).collect(),
        },
        Some(&"preset") => SemanticsCommandInput::Preset {
            args: args[2..].iter().map(|s| (*s).to_string()).collect(),
        },
        Some(&"domain")
        | Some(&"value")
        | Some(&"branch")
        | Some(&"inv_trig")
        | Some(&"const_fold")
        | Some(&"assumptions")
        | Some(&"assume_scope")
        | Some(&"requires") => SemanticsCommandInput::Axis {
            axis: args[1].to_string(),
        },
        Some(other) => SemanticsCommandInput::Unknown {
            subcommand: (*other).to_string(),
        },
    }
}

/// Evaluate `semantics ...` command line and apply runtime option changes.
pub fn evaluate_semantics_command_line(
    line: &str,
    simplify_options: &mut cas_engine::SimplifyOptions,
    eval_options: &mut cas_engine::EvalOptions,
) -> SemanticsCommandOutput {
    match parse_semantics_command_input(line) {
        SemanticsCommandInput::Show => {
            let state = crate::semantics_display::semantics_view_state_from_options(
                simplify_options,
                eval_options,
            );
            SemanticsCommandOutput {
                lines: crate::semantics_display::format_semantics_overview_lines(&state),
                sync_simplifier: false,
            }
        }
        SemanticsCommandInput::Help => SemanticsCommandOutput {
            lines: vec![crate::semantics_display::semantics_help_message().to_string()],
            sync_simplifier: false,
        },
        SemanticsCommandInput::Set { args } => {
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            match crate::semantics_set::evaluate_semantics_set_args_to_overview_lines(
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
            let state = crate::semantics_display::semantics_view_state_from_options(
                simplify_options,
                eval_options,
            );
            SemanticsCommandOutput {
                lines: crate::semantics_display::format_semantics_axis_lines(&state, &axis),
                sync_simplifier: false,
            }
        }
        SemanticsCommandInput::Preset { args } => {
            let refs: Vec<&str> = args.iter().map(String::as_str).collect();
            let out = crate::semantics_presets::evaluate_semantics_preset_args_to_options(
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
            lines: vec![
                crate::semantics_display::format_semantics_unknown_subcommand_message(&subcommand),
            ],
            sync_simplifier: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_semantics_command_line, parse_semantics_command_input, SemanticsCommandInput,
    };

    #[test]
    fn parse_semantics_command_input_detects_set() {
        let input = parse_semantics_command_input("semantics set domain strict");
        assert_eq!(
            input,
            SemanticsCommandInput::Set {
                args: vec!["domain".to_string(), "strict".to_string()]
            }
        );
    }

    #[test]
    fn parse_semantics_command_input_detects_axis() {
        let input = parse_semantics_command_input("semantics assumptions");
        assert_eq!(
            input,
            SemanticsCommandInput::Axis {
                axis: "assumptions".to_string()
            }
        );
    }

    #[test]
    fn parse_semantics_command_input_detects_unknown() {
        let input = parse_semantics_command_input("semantics weird");
        assert_eq!(
            input,
            SemanticsCommandInput::Unknown {
                subcommand: "weird".to_string()
            }
        );
    }

    #[test]
    fn evaluate_semantics_command_line_show_formats_overview() {
        let mut simplify_options = cas_engine::SimplifyOptions::default();
        let mut eval_options = cas_engine::EvalOptions::default();
        let out =
            evaluate_semantics_command_line("semantics", &mut simplify_options, &mut eval_options);
        assert!(!out.sync_simplifier);
        assert!(out.lines.iter().any(|line| line.contains("domain_mode")));
    }

    #[test]
    fn evaluate_semantics_command_line_set_applies_and_requests_sync() {
        let mut simplify_options = cas_engine::SimplifyOptions::default();
        let mut eval_options = cas_engine::EvalOptions::default();
        let out = evaluate_semantics_command_line(
            "semantics set domain assume",
            &mut simplify_options,
            &mut eval_options,
        );
        assert!(out.sync_simplifier);
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            cas_engine::DomainMode::Assume
        );
    }
}
