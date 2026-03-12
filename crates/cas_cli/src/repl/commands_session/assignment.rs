use super::*;

impl Repl {
    pub(crate) fn handle_budget_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(
            cas_solver::session_api::budget::evaluate_solve_budget_command_message_on_repl_core(
                &mut self.core,
                line,
            ),
        )
    }

    /// Handle "let <name> = <expr>" (eager) or "let <name> := <expr>" (lazy) command
    pub(crate) fn handle_let_command_core(&mut self, rest: &str) -> ReplReply {
        match cas_solver::session_api::bindings::evaluate_let_assignment_command_message_on_repl_core(
            &mut self.core,
            rest,
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }

    /// Handle variable assignment (from "let" or ":=")
    /// - eager=false (=): evaluate then store (unwrap __hold)
    /// - eager=true (:=): store formula without evaluating
    pub(crate) fn handle_assignment_core(
        &mut self,
        name: &str,
        expr_str: &str,
        lazy: bool,
    ) -> ReplReply {
        match cas_solver::session_api::bindings::evaluate_assignment_command_message_on_repl_core(
            &mut self.core,
            name,
            expr_str,
            lazy,
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }
}
