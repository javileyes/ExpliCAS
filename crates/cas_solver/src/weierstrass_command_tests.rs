#[cfg(test)]
mod tests {
    use crate::solve_command_errors::format_solve_command_error_message;
    use crate::weierstrass_command::evaluate_weierstrass_invocation_message;
    use cas_api_models::{SolveCommandEvalError, SolvePrepareError};

    #[test]
    fn format_solve_command_error_message_prepare() {
        let msg = format_solve_command_error_message(&SolveCommandEvalError::Prepare(
            SolvePrepareError::NoVariable,
        ));
        assert!(msg.contains("no variable"));
    }

    #[test]
    fn evaluate_weierstrass_invocation_message_joins_lines() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let message =
            evaluate_weierstrass_invocation_message(&mut simplifier, "weierstrass sin(x)")
                .expect("weierstrass");
        assert!(message.contains("Result:"));
    }
}
