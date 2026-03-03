#[cfg(test)]
mod tests {
    use crate::full_simplify_eval::{evaluate_full_simplify_input, FullSimplifyEvalError};

    #[test]
    fn evaluate_full_simplify_input_parse_error_is_typed() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let session = crate::SessionState::new();
        let err = evaluate_full_simplify_input(&mut simplifier, &session, "x+", true)
            .expect_err("parse error");
        assert!(matches!(err, FullSimplifyEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_full_simplify_input_runs() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let session = crate::SessionState::new();
        let out = evaluate_full_simplify_input(&mut simplifier, &session, "x + 0", true)
            .expect("full simplify");
        let shown = cas_formatter::render_expr(&simplifier.context, out.simplified_expr);
        assert_eq!(shown, "x");
    }
}
