//! Linear system solving command routing.

use super::output::{reply_output, ReplReply};
use super::Repl;

impl Repl {
    /// Core: handle `solve_system` command (routing only, no direct I/O).
    pub(crate) fn handle_solve_system_core(&mut self, line: &str) -> ReplReply {
        let message =
            cas_solver::session_api::runtime::evaluate_linear_system_command_message_on_repl_core(
                &mut self.core,
                line,
            );
        reply_output(message)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_linear_system_command_message_accepts_parenthesized_form() {
        let mut ctx = cas_ast::Context::new();
        let shown = cas_solver::session_api::runtime::evaluate_linear_system_command_message(
            &mut ctx,
            "solve_system(x+y=3; x-y=1; x; y)",
        );
        assert_eq!(shown, "{ x = 2, y = 1 }");
    }

    #[test]
    fn evaluate_linear_system_command_message_accepts_space_form() {
        let mut ctx = cas_ast::Context::new();
        let shown = cas_solver::session_api::runtime::evaluate_linear_system_command_message(
            &mut ctx,
            "solve_system x+y=3; x-y=1; x; y",
        );
        assert_eq!(shown, "{ x = 2, y = 1 }");
    }
}
