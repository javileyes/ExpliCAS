#![allow(unused_imports)]

pub use crate::substitute_subcommand_eval::evaluate_substitute_subcommand;
pub use crate::substitute_subcommand_json::evaluate_substitute_subcommand_json_canonical;
pub use crate::substitute_subcommand_text::parse_substitute_json_text_lines;
pub use crate::substitute_subcommand_types::{SubstituteCommandMode, SubstituteSubcommandOutput};

#[cfg(test)]
mod tests {
    use super::{
        evaluate_substitute_subcommand, parse_substitute_json_text_lines, SubstituteCommandMode,
        SubstituteSubcommandOutput,
    };

    #[test]
    fn evaluate_substitute_subcommand_json_contract() {
        let out = evaluate_substitute_subcommand(
            "x^2+1",
            "x",
            "y",
            SubstituteCommandMode::Exact,
            false,
            true,
        )
        .expect("substitute json");

        match out {
            SubstituteSubcommandOutput::Json(payload) => {
                assert!(payload.contains("\"ok\""));
            }
            _ => panic!("expected json output"),
        }
    }

    #[test]
    fn evaluate_substitute_subcommand_text_contract() {
        let out = evaluate_substitute_subcommand(
            "x^2+1",
            "x",
            "y",
            SubstituteCommandMode::Exact,
            true,
            false,
        )
        .expect("substitute text");

        match out {
            SubstituteSubcommandOutput::TextLines(lines) => {
                assert!(!lines.is_empty());
                assert!(lines.iter().any(|line| line.contains('y')));
            }
            _ => panic!("expected text output"),
        }
    }

    #[test]
    fn parse_substitute_json_text_lines_maps_error_message() {
        let payload = r#"{"ok":false,"error":{"message":"Parse error in target: bad token"}}"#;
        let err = parse_substitute_json_text_lines(payload, false).expect_err("should fail");
        assert_eq!(err, "Parse error in target: bad token");
    }
}
