use super::*;

impl Repl {
    /// Handle "semantics" command - unified control for semantic axes
    pub(crate) fn handle_semantics_core(&mut self, line: &str) -> ReplReply {
        reply_output(cas_session::repl::evaluate_semantics_command_on_repl(
            line,
            &mut self.core,
            &self.config,
        ))
    }
}
