use super::super::*;

impl Repl {
    /// Handle "context" command - show or switch context mode
    pub(crate) fn handle_context_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(cas_session::repl::evaluate_context_command_on_repl(
            line,
            &mut self.core,
            &self.config,
        ))
    }

    /// Handle "autoexpand" command - show or switch auto-expand policy
    pub(crate) fn handle_autoexpand_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(cas_session::repl::evaluate_autoexpand_command_on_repl(
            line,
            &mut self.core,
            &self.config,
        ))
    }
}
