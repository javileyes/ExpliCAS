#[cfg(test)]
mod tests {
    use cas_api_models::{
        LimitCommandApproach, LimitCommandEvalError, LimitSubcommandEvalError,
        LimitSubcommandEvalOutput,
    };
    use cas_math::limit_types::{Approach, PreSimplifyMode};

    use crate::limit_command_core::{
        evaluate_limit_command_input, evaluate_limit_subcommand_output,
        format_limit_subcommand_error,
    };
    use crate::limit_command_parse::parse_limit_command_input;

    #[test]
    fn parse_limit_command_input_defaults_var_and_direction() {
        let parsed = parse_limit_command_input("x^2");
        assert_eq!(parsed.expr, "x^2");
        assert_eq!(parsed.var, "x");
        assert_eq!(parsed.approach, Approach::PosInfinity);
        assert_eq!(parsed.presimplify, PreSimplifyMode::Off);
    }

    #[test]
    fn parse_limit_command_input_reads_neg_infinity_and_safe() {
        let parsed = parse_limit_command_input("(x^2+1)/(2*x^2-3), t, -infinity, safe");
        assert_eq!(parsed.expr, "(x^2+1)/(2*x^2-3)");
        assert_eq!(parsed.var, "t");
        assert_eq!(parsed.approach, Approach::NegInfinity);
        assert_eq!(parsed.presimplify, PreSimplifyMode::Safe);
    }

    #[test]
    fn evaluate_limit_command_input_rejects_empty_input() {
        let err = evaluate_limit_command_input("  ").expect_err("expected empty-input error");
        assert_eq!(err, LimitCommandEvalError::EmptyInput);
    }

    #[test]
    fn evaluate_limit_command_input_computes_basic_limit() {
        let out = evaluate_limit_command_input("x^2, x, infinity").expect("limit eval");
        assert_eq!(out.var, "x");
        assert_eq!(out.approach, LimitCommandApproach::Infinity);
        assert!(!out.result.is_empty());
    }

    #[test]
    fn evaluate_limit_subcommand_output_wire_mode_returns_payload() {
        let out = evaluate_limit_subcommand_output(
            "(x^2+1)/(2*x^2-3)",
            "x",
            Approach::PosInfinity,
            PreSimplifyMode::Off,
            true,
        )
        .expect("wire output");

        match out {
            LimitSubcommandEvalOutput::Wire(payload) => {
                let json: serde_json::Value = serde_json::from_str(&payload).expect("json");
                assert_eq!(json["ok"], true);
            }
            _ => panic!("expected wire payload"),
        }
    }

    #[test]
    fn evaluate_limit_subcommand_output_parse_error_is_typed() {
        let err = evaluate_limit_subcommand_output(
            "sin(",
            "x",
            Approach::PosInfinity,
            PreSimplifyMode::Off,
            false,
        )
        .expect_err("parse error");

        match err {
            LimitSubcommandEvalError::Parse(message) => {
                assert!(message.starts_with("Parse error:"));
            }
            _ => panic!("expected parse error"),
        }
    }

    #[test]
    fn format_limit_subcommand_error_matches_cli_contract() {
        let parse = format_limit_subcommand_error(&LimitSubcommandEvalError::Parse(
            "Parse error: bad input".to_string(),
        ));
        assert_eq!(parse, "Parse error: bad input");

        let limit = format_limit_subcommand_error(&LimitSubcommandEvalError::Limit(
            "cannot compute".to_string(),
        ));
        assert_eq!(limit, "Error: cannot compute");
    }
}
