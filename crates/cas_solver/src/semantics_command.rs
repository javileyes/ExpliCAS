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

#[cfg(test)]
mod tests {
    use super::{parse_semantics_command_input, SemanticsCommandInput};

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
}
