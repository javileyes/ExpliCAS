#[cfg(test)]
mod tests {
    use crate::command_api::substitute::{
        evaluate_substitute_subcommand, SubstituteCommandMode, SubstituteSubcommandOutput,
    };

    #[test]
    fn evaluate_substitute_subcommand_wire_contract() {
        let out = evaluate_substitute_subcommand(
            "x^2+1",
            "x",
            "y",
            SubstituteCommandMode::Exact,
            false,
            true,
        )
        .expect("substitute wire");

        match out {
            SubstituteSubcommandOutput::Wire(payload) => {
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
}
