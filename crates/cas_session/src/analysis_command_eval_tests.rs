#[cfg(test)]
mod tests {
    use crate::analysis_command_eval::{
        evaluate_equiv_command_lines, evaluate_equiv_command_message,
        evaluate_equiv_invocation_message, evaluate_explain_command_lines,
        evaluate_explain_command_message, evaluate_explain_invocation_message,
        evaluate_visualize_command_dot, evaluate_visualize_command_output,
        evaluate_visualize_invocation_output,
    };
    use crate::analysis_command_types::VisualizeEvalError;

    #[test]
    fn evaluate_equiv_command_lines_true() {
        let mut simplifier = cas_solver::Simplifier::new();
        let lines = evaluate_equiv_command_lines(&mut simplifier, "x+1, 1+x")
            .expect("equiv should evaluate");
        assert!(lines.iter().any(|line| line.contains("True")));
    }

    #[test]
    fn evaluate_visualize_command_dot_parse_error() {
        let mut ctx = cas_ast::Context::new();
        let err = evaluate_visualize_command_dot(&mut ctx, "x+").expect_err("expected parse");
        assert!(matches!(err, VisualizeEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_visualize_command_output_sets_file_and_hints() {
        let mut ctx = cas_ast::Context::new();
        let out =
            evaluate_visualize_command_output(&mut ctx, "x+1").expect("visualize should evaluate");
        assert_eq!(out.file_name, "ast.dot");
        assert!(out.dot_source.contains("digraph"));
        assert_eq!(out.hint_lines.len(), 2);
        assert!(out.hint_lines[0].contains("dot -Tsvg"));
    }

    #[test]
    fn evaluate_explain_command_lines_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let lines =
            evaluate_explain_command_lines(&mut ctx, "gcd(8, 6)").expect("explain should evaluate");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_equiv_command_message_joins_lines() {
        let mut simplifier = cas_solver::Simplifier::new();
        let message = evaluate_equiv_command_message(&mut simplifier, "x+1,1+x")
            .expect("equiv should evaluate");
        assert!(message.contains("True"));
    }

    #[test]
    fn evaluate_equiv_invocation_message_formats_parse_error() {
        let mut simplifier = cas_solver::Simplifier::new();
        let message =
            evaluate_equiv_invocation_message(&mut simplifier, "equiv x+1").expect_err("parse");
        assert!(message.contains("equiv"));
    }

    #[test]
    fn evaluate_explain_command_message_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let message = evaluate_explain_command_message(&mut ctx, "gcd(8, 6)")
            .expect("explain should evaluate");
        assert!(message.contains("Result:"));
    }

    #[test]
    fn evaluate_explain_invocation_message_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let message = evaluate_explain_invocation_message(&mut ctx, "explain gcd(8, 6)")
            .expect("explain should evaluate");
        assert!(message.contains("Result:"));
    }

    #[test]
    fn evaluate_visualize_invocation_output_sets_file() {
        let mut ctx = cas_ast::Context::new();
        let out = evaluate_visualize_invocation_output(&mut ctx, "visualize x+1")
            .expect("visualize should evaluate");
        assert_eq!(out.file_name, "ast.dot");
    }
}
