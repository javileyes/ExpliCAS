#[cfg(test)]
mod tests {
    use crate::{evaluate_show_command_lines, SessionState};

    #[test]
    fn evaluate_show_command_lines_invalid_id_reports_error() {
        let mut engine = cas_solver::Engine::new();
        let mut session = SessionState::new();
        let err = evaluate_show_command_lines(&mut session, &mut engine, "show nope")
            .expect_err("expected invalid id error");
        assert!(err.contains("Invalid entry ID"));
    }
}
