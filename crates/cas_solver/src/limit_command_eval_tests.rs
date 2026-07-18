#[cfg(test)]
mod tests {
    use cas_api_models::{
        LimitCommandApproach, LimitCommandEvalError, LimitSubcommandEvalError,
        LimitSubcommandEvalOutput,
    };
    use cas_math::limit_types::{Approach, PreSimplifyMode};

    use crate::limit_command_core::{
        evaluate_limit_command_input_in_domain, evaluate_limit_subcommand_output,
        format_limit_subcommand_error,
    };

    // Real-domain view of the command entry (the pre-F0b legacy signature).
    fn evaluate_limit_command_input(
        input: &str,
    ) -> Result<cas_api_models::LimitCommandEvalOutput, cas_api_models::LimitCommandEvalError> {
        evaluate_limit_command_input_in_domain(input, false)
    }
    use crate::limit_command_parse::parse_limit_command_input;

    #[test]
    fn parse_limit_command_input_defaults_var_and_direction() {
        let parsed = parse_limit_command_input("x^2").expect("parse");
        assert_eq!(parsed.expr, "x^2");
        assert_eq!(parsed.var, "x");
        assert_eq!(parsed.approach, Approach::PosInfinity);
        assert_eq!(parsed.presimplify, PreSimplifyMode::Off);
    }

    #[test]
    fn parse_limit_command_input_reads_neg_infinity_and_safe() {
        let parsed =
            parse_limit_command_input("(x^2+1)/(2*x^2-3), t, -infinity, safe").expect("parse");
        assert_eq!(parsed.expr, "(x^2+1)/(2*x^2-3)");
        assert_eq!(parsed.var, "t");
        assert_eq!(parsed.approach, Approach::NegInfinity);
        assert_eq!(parsed.presimplify, PreSimplifyMode::Safe);
    }

    #[test]
    fn parse_limit_command_input_rejects_unsupported_finite_direction() {
        let err = parse_limit_command_input("x, x, 0").expect_err("finite point unsupported");
        assert!(err.contains("Unsupported limit direction `0`"));
        assert!(err.contains("Finite point limits are not supported yet"));
        assert!(err.contains("use infinity or -infinity"));
    }

    #[test]
    fn evaluate_limit_command_input_rejects_empty_input() {
        let err = evaluate_limit_command_input("  ").expect_err("expected empty-input error");
        assert_eq!(err, LimitCommandEvalError::EmptyInput);
    }

    #[test]
    fn evaluate_limit_command_input_rejects_unsupported_finite_direction() {
        let err = evaluate_limit_command_input("x, x, 0").expect_err("finite point unsupported");
        match err {
            LimitCommandEvalError::Parse(message) => {
                assert!(message.contains("Unsupported limit direction `0`"));
                assert!(message.contains("Finite point limits are not supported yet"));
            }
            other => panic!("expected parse error, got {other:?}"),
        }
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
    fn limit_command_complex_domain_declines_to_residual() {
        // F0b (Fase 3): la superficie del comando REPL heredaba el motor con
        // complex_enabled=false SIEMPRE — con la sesión en value complex emitía
        // valores de orden real (`lim e^(-x) = 0`). El eje enhebrado debe
        // declinar TODO límite a residual con el warning del kill-switch.
        for input in [
            "e^(-x), x, infinity",
            "atan(x), x, infinity",
            "1/x, x, -infinity",
        ] {
            let out = evaluate_limit_command_input_in_domain(input, true).expect("limit eval");
            assert!(
                out.result.starts_with("limit("),
                "`{input}` debe declinar a residual bajo complex, got: {}",
                out.result
            );
            let warning = out.warning.expect("complex decline carries a warning");
            assert!(
                warning.contains("complex value domain"),
                "`{input}` debe llevar el motivo del kill-switch, got: {warning}"
            );
        }
        // Pin: la vista real (flag false y la firma legacy) sigue computando.
        let real = evaluate_limit_command_input_in_domain("e^(-x), x, infinity", false)
            .expect("limit eval");
        assert_eq!(real.result, "0");
        assert!(real.warning.is_none());
        let legacy = evaluate_limit_command_input("e^(-x), x, infinity").expect("limit eval");
        assert_eq!(legacy.result, "0");
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
