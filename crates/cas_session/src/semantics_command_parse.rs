use crate::semantics_command_types::SemanticsCommandInput;

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
