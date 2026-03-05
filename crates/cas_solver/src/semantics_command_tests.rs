#[cfg(test)]
mod tests {
    use crate::{
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
        let mut simplify_options = crate::SimplifyOptions::default();
        let mut eval_options = crate::EvalOptions::default();
        let out =
            evaluate_semantics_command_line("semantics", &mut simplify_options, &mut eval_options);
        assert!(!out.sync_simplifier);
        assert!(out.lines.iter().any(|line| line.contains("domain_mode")));
    }

    #[test]
    fn evaluate_semantics_command_line_set_applies_and_requests_sync() {
        let mut simplify_options = crate::SimplifyOptions::default();
        let mut eval_options = crate::EvalOptions::default();
        let out = evaluate_semantics_command_line(
            "semantics set domain assume",
            &mut simplify_options,
            &mut eval_options,
        );
        assert!(out.sync_simplifier);
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            crate::DomainMode::Assume
        );
    }
}
