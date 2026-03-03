#[cfg(test)]
mod tests {
    use crate::limit_command_eval::{
        evaluate_limit_command_input, evaluate_limit_subcommand_output,
        format_limit_subcommand_error, parse_limit_command_input,
    };
    use crate::limit_command_types::{
        LimitCommandEvalError, LimitSubcommandEvalError, LimitSubcommandEvalOutput,
    };

    #[test]
    fn parse_limit_command_input_defaults_var_and_direction() {
        let parsed = parse_limit_command_input("x^2");
        assert_eq!(parsed.expr, "x^2");
        assert_eq!(parsed.var, "x");
        assert_eq!(parsed.approach, cas_solver::Approach::PosInfinity);
        assert_eq!(parsed.presimplify, cas_solver::PreSimplifyMode::Off);
    }

    #[test]
    fn parse_limit_command_input_reads_neg_infinity_and_safe() {
        let parsed = parse_limit_command_input("(x^2+1)/(2*x^2-3), t, -infinity, safe");
        assert_eq!(parsed.expr, "(x^2+1)/(2*x^2-3)");
        assert_eq!(parsed.var, "t");
        assert_eq!(parsed.approach, cas_solver::Approach::NegInfinity);
        assert_eq!(parsed.presimplify, cas_solver::PreSimplifyMode::Safe);
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
        assert_eq!(out.approach, cas_solver::Approach::PosInfinity);
        assert!(!out.result.is_empty());
    }

    #[test]
    fn evaluate_limit_subcommand_output_json_mode_returns_payload() {
        let out = evaluate_limit_subcommand_output(
            "(x^2+1)/(2*x^2-3)",
            "x",
            cas_solver::Approach::PosInfinity,
            cas_solver::PreSimplifyMode::Off,
            true,
        )
        .expect("json output");

        match out {
            LimitSubcommandEvalOutput::Json(payload) => {
                let json: serde_json::Value = serde_json::from_str(&payload).expect("json");
                assert_eq!(json["ok"], true);
            }
            _ => panic!("expected json payload"),
        }
    }

    #[test]
    fn evaluate_limit_subcommand_output_parse_error_is_typed() {
        let err = evaluate_limit_subcommand_output(
            "sin(",
            "x",
            cas_solver::Approach::PosInfinity,
            cas_solver::PreSimplifyMode::Off,
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
