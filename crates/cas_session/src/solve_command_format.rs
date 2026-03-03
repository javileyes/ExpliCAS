pub use crate::solve_command_format_errors::{
    format_solve_command_error_message, format_solve_prepare_error_message,
    format_verify_summary_lines,
};
pub use crate::solve_command_weierstrass::{
    evaluate_weierstrass_command_lines, evaluate_weierstrass_invocation_lines,
    evaluate_weierstrass_invocation_message, parse_weierstrass_invocation_input,
    weierstrass_usage_message,
};

#[cfg(test)]
mod tests {
    #[test]
    fn parse_weierstrass_invocation_input_reads_tail() {
        assert_eq!(
            super::parse_weierstrass_invocation_input("weierstrass sin(x)+cos(x)"),
            Some("sin(x)+cos(x)".to_string())
        );
    }

    #[test]
    fn format_solve_command_error_message_prepare() {
        let msg = super::format_solve_command_error_message(
            &crate::SolveCommandEvalError::Prepare(crate::SolvePrepareError::NoVariable),
        );
        assert!(msg.contains("no variable"));
    }

    #[test]
    fn evaluate_weierstrass_command_lines_runs() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let lines = super::evaluate_weierstrass_command_lines(&mut simplifier, "sin(x)+cos(x)")
            .expect("weierstrass eval");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_weierstrass_invocation_lines_requires_input() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let err = super::evaluate_weierstrass_invocation_lines(&mut simplifier, "weierstrass")
            .expect_err("usage");
        assert!(err.contains("Usage: weierstrass"));
    }

    #[test]
    fn evaluate_weierstrass_invocation_message_joins_lines() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let message =
            super::evaluate_weierstrass_invocation_message(&mut simplifier, "weierstrass sin(x)")
                .expect("weierstrass");
        assert!(message.contains("Result:"));
    }
}
