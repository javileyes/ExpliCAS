#[cfg(test)]
mod tests {
    use num_rational::BigRational;

    use crate::evaluate_linear_system_command_message;
    use crate::linear_system_format::display_linear_system_solution;
    use crate::linear_system_parse::parse_linear_system_invocation_input;

    #[test]
    fn evaluate_linear_system_command_message_solves_2x2() {
        let mut ctx = cas_ast::Context::new();
        let shown =
            evaluate_linear_system_command_message(&mut ctx, "solve_system(x+y=3; x-y=1; x; y)");
        assert_eq!(shown, "{ x = 2, y = 1 }");
    }

    #[test]
    fn evaluate_linear_system_command_message_reports_usage() {
        let mut ctx = cas_ast::Context::new();
        let shown = evaluate_linear_system_command_message(&mut ctx, "solve_system x+y=3");
        assert!(shown.contains("Usage:"));
    }

    #[test]
    fn display_linear_system_solution_formats_pairs() {
        let mut ctx = cas_ast::Context::new();
        let vars = vec!["x".to_string(), "y".to_string()];
        let values = vec![
            BigRational::from_integer(2.into()),
            BigRational::from_integer(1.into()),
        ];
        let shown = display_linear_system_solution(&mut ctx, &vars, &values);
        assert_eq!(shown, "{ x = 2, y = 1 }");
    }

    #[test]
    fn parse_linear_system_invocation_input_accepts_parenthesized_form() {
        assert_eq!(
            parse_linear_system_invocation_input("solve_system(x+y=3; x-y=1; x; y)"),
            "x+y=3; x-y=1; x; y".to_string()
        );
    }

    #[test]
    fn parse_linear_system_invocation_input_accepts_space_form() {
        assert_eq!(
            parse_linear_system_invocation_input("solve_system x+y=3; x-y=1; x; y"),
            "x+y=3; x-y=1; x; y".to_string()
        );
    }
}
