#[cfg(test)]
mod tests {
    use crate::solver_exports::{
        apply_assignment, format_assignment_error_message, format_assignment_success_message,
        format_let_assignment_parse_error_message, parse_let_assignment_input, AssignmentError,
        LetAssignmentParseError, ParsedLetAssignment,
    };
    use crate::SessionState;

    #[test]
    fn apply_assignment_validates_name() {
        let mut state = SessionState::new();
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let err = apply_assignment(&mut state, &mut simplifier, "", "1", false).expect_err("err");
        assert_eq!(err, AssignmentError::EmptyName);
    }

    #[test]
    fn apply_assignment_stores_eager_value() {
        let mut state = SessionState::new();
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let id = apply_assignment(&mut state, &mut simplifier, "a", "x + x", false).expect("ok");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id
            }
        );
        assert_eq!(rendered.replace(' ', ""), "2*x");
    }

    #[test]
    fn apply_assignment_stores_lazy_formula() {
        let mut state = SessionState::new();
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let id = apply_assignment(&mut state, &mut simplifier, "a", "x + x", true).expect("ok");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id
            }
        );
        assert_eq!(rendered, "x + x");
    }

    #[test]
    fn parse_let_assignment_input_supports_lazy_and_eager() {
        let lazy = parse_let_assignment_input("a := x + 1").expect("lazy");
        assert_eq!(
            lazy,
            ParsedLetAssignment {
                name: "a",
                expr: "x + 1",
                lazy: true
            }
        );

        let eager = parse_let_assignment_input("b = x + 2").expect("eager");
        assert_eq!(
            eager,
            ParsedLetAssignment {
                name: "b",
                expr: "x + 2",
                lazy: false
            }
        );
    }

    #[test]
    fn parse_let_assignment_input_requires_operator() {
        let err = parse_let_assignment_input("abc").expect_err("missing operator");
        assert_eq!(err, LetAssignmentParseError::MissingAssignmentOperator);
    }

    #[test]
    fn format_let_assignment_parse_error_message_returns_usage() {
        let msg = format_let_assignment_parse_error_message(
            &LetAssignmentParseError::MissingAssignmentOperator,
        );
        assert!(msg.contains("Usage: let <name> = <expr>"));
    }

    #[test]
    fn format_assignment_error_message_reserved_name() {
        let msg = format_assignment_error_message(&AssignmentError::ReservedName("pi".to_string()));
        assert_eq!(
            msg,
            "Error: 'pi' is a reserved name and cannot be assigned".to_string()
        );
    }

    #[test]
    fn format_assignment_success_message_supports_eager_and_lazy() {
        assert_eq!(
            format_assignment_success_message("a", "2 * x", false),
            "a = 2 * x"
        );
        assert_eq!(
            format_assignment_success_message("a", "x + x", true),
            "a := x + x"
        );
    }
}
