use super::*;

impl Repl {
    /// Handle "reset" command - reset entire session
    pub(crate) fn handle_reset_command_core(&mut self) -> ReplReply {
        cas_session::reset_repl_core_with_config(&mut self.core, &self.config);
        reply_output("Session reset. Environment and context cleared.")
    }

    /// Handle "reset full" command - reset session AND clear profile cache
    pub(crate) fn handle_reset_full_command_core(&mut self) -> ReplReply {
        cas_session::reset_repl_core_full_with_config(&mut self.core, &self.config);
        reply_output(
            "Session reset. Environment and context cleared.\n\
             Profile cache cleared (will rebuild on next eval).",
        )
    }
}
