use crate::semantics_command_parse::axis::is_axis_subcommand;
use crate::semantics_command_types::SemanticsCommandInput;

pub(super) fn parse_semantics_command_args(args: &[&str]) -> SemanticsCommandInput {
    match args.get(1) {
        None => SemanticsCommandInput::Show,
        Some(&"help") => SemanticsCommandInput::Help,
        Some(&"set") => SemanticsCommandInput::Set {
            args: args[2..].iter().map(|s| (*s).to_string()).collect(),
        },
        Some(&"preset") => SemanticsCommandInput::Preset {
            args: args[2..].iter().map(|s| (*s).to_string()).collect(),
        },
        Some(subcommand) if is_axis_subcommand(subcommand) => SemanticsCommandInput::Axis {
            axis: (*subcommand).to_string(),
        },
        Some(other) => SemanticsCommandInput::Unknown {
            subcommand: (*other).to_string(),
        },
    }
}
