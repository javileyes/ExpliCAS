pub use crate::solve_command_eval::{evaluate_solve_command_lines, evaluate_solve_command_message};

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_solve_command_lines_reports_ambiguous_variables() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let out = super::evaluate_solve_command_lines(
            &mut engine,
            &mut session,
            "solve x+y=0",
            crate::SetDisplayMode::Normal,
            false,
        )
        .expect_err("expected ambiguous-variable error");
        assert!(out.contains("ambiguous variables"));
    }

    #[test]
    fn evaluate_solve_command_message_joins_lines() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let message = super::evaluate_solve_command_message(
            &mut engine,
            &mut session,
            "solve x+2=5",
            crate::SetDisplayMode::Normal,
            false,
        )
        .expect("solve should succeed");
        assert!(message.contains("x"));
        assert!(message.contains("3"));
    }
}
