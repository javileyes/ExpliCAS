#[cfg(test)]
mod tests {
    use crate::repl_command_parse::parse_repl_command_input;
    use crate::repl_command_preprocess::{preprocess_repl_function_syntax, split_repl_statements};
    use crate::repl_command_types::ReplCommandInput;

    #[test]
    fn preprocess_repl_function_syntax_maps_simplify_form() {
        assert_eq!(
            preprocess_repl_function_syntax("simplify(x^2 + 1)"),
            "simplify x^2 + 1"
        );
    }

    #[test]
    fn preprocess_repl_function_syntax_maps_solve_form() {
        assert_eq!(
            preprocess_repl_function_syntax("solve(x+1=2,x)"),
            "solve x+1=2,x"
        );
    }

    #[test]
    fn preprocess_repl_function_syntax_trims_and_preserves_other_inputs() {
        assert_eq!(
            preprocess_repl_function_syntax("  x + x  "),
            "x + x".to_string()
        );
    }

    #[test]
    fn parses_help_prefix() {
        assert_eq!(
            parse_repl_command_input("help solve"),
            ReplCommandInput::Help("help solve")
        );
    }

    #[test]
    fn parses_lazy_assignment_before_eval() {
        assert_eq!(
            parse_repl_command_input("x := y + 1"),
            ReplCommandInput::Assignment {
                name: "x",
                expr: "y + 1",
                lazy: true,
            }
        );
    }

    #[test]
    fn parses_solve_system_before_solve() {
        assert_eq!(
            parse_repl_command_input("solve_system(x+y=1; x; y)"),
            ReplCommandInput::SolveSystem("solve_system(x+y=1; x; y)")
        );
    }

    #[test]
    fn parses_expand_log_before_expand() {
        assert_eq!(
            parse_repl_command_input("expand_log ln(x*y)"),
            ReplCommandInput::ExpandLog("expand_log ln(x*y)")
        );
    }

    #[test]
    fn parses_bare_commands_instead_of_falling_back_to_eval() {
        assert_eq!(
            parse_repl_command_input("equiv"),
            ReplCommandInput::Equiv("equiv")
        );
        assert_eq!(
            parse_repl_command_input("subst"),
            ReplCommandInput::Subst("subst")
        );
        assert_eq!(
            parse_repl_command_input("solve"),
            ReplCommandInput::Solve("solve")
        );
        assert_eq!(
            parse_repl_command_input("simplify"),
            ReplCommandInput::Simplify("simplify")
        );
        assert_eq!(
            parse_repl_command_input("telescope"),
            ReplCommandInput::Telescope("telescope")
        );
        assert_eq!(
            parse_repl_command_input("weierstrass"),
            ReplCommandInput::Weierstrass("weierstrass")
        );
        assert_eq!(
            parse_repl_command_input("rationalize"),
            ReplCommandInput::Rationalize("rationalize")
        );
        assert_eq!(
            parse_repl_command_input("limit"),
            ReplCommandInput::Limit("limit")
        );
    }

    #[test]
    fn parses_visualize_alias_viz() {
        assert_eq!(
            parse_repl_command_input("viz x + 1"),
            ReplCommandInput::Visualize("viz x + 1")
        );
    }

    #[test]
    fn falls_back_to_eval() {
        assert_eq!(
            parse_repl_command_input("x + x"),
            ReplCommandInput::Eval("x + x")
        );
    }

    #[test]
    fn split_repl_statements_preserves_solve_system_line() {
        let parts = split_repl_statements("solve_system x+y=3; x-y=1; x; y");
        assert_eq!(parts, vec!["solve_system x+y=3; x-y=1; x; y"]);
    }

    #[test]
    fn split_repl_statements_splits_regular_semicolons() {
        let parts = split_repl_statements("let a = 1; let b = 2; a + b");
        assert_eq!(parts, vec!["let a = 1", "let b = 2", "a + b"]);
    }
}
