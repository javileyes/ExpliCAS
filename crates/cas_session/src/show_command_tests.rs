#[cfg(test)]
mod tests {
    use crate::state_core::SessionState;
    #[allow(unused_imports)]
    use cas_solver::session_api::{
        formatting::*, options::*, runtime::*, session_support::*, symbolic_commands::*, types::*,
    };

    #[test]
    fn evaluate_show_command_lines_invalid_id_reports_error() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();
        let err = evaluate_show_command_lines(&mut session, &mut engine, "show nope")
            .expect_err("expected invalid id error");
        assert!(err.contains("Invalid entry ID"));
    }
}
