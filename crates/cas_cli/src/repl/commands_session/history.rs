use super::*;

impl Repl {
    /// Handle "vars" command - list all variable bindings
    pub(crate) fn handle_vars_command_core(&self) -> ReplReply {
        reply_output(
            cas_solver::session_api::runtime::evaluate_vars_command_message_on_repl_core(
                &self.core,
            ),
        )
    }

    /// Handle "history" or "list" command - show session history
    pub(crate) fn handle_history_command_core(&self) -> ReplReply {
        reply_output(
            cas_solver::session_api::runtime::evaluate_history_command_message_on_repl_core(
                &self.core,
            ),
        )
    }

    /// Handle "show #id" command - show details of a specific entry
    pub(crate) fn handle_show_command_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::session_api::runtime::evaluate_show_command_lines_on_repl_core(
            &mut self.core,
            line,
        ) {
            Ok(lines) => output::reply_output_lines(lines),
            Err(message) => reply_output(message),
        }
    }

    /// Handle "del #id [#id...]" command - delete session entries
    pub(crate) fn handle_del_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(
            cas_solver::session_api::runtime::evaluate_delete_history_command_message_on_repl_core(
                &mut self.core,
                line,
            ),
        )
    }

    /// Handle "clear" or "clear <names>" command
    pub(crate) fn handle_clear_command_core(&mut self, line: &str) -> ReplReply {
        output::reply_output_lines(
            cas_solver::session_api::runtime::evaluate_clear_command_lines_on_repl_core(
                &mut self.core,
                line,
            ),
        )
    }
}
