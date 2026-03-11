use super::*;

impl Repl {
    /// Handle "cache" command - show status or clear cache
    pub(crate) fn handle_cache_command_core(&mut self, line: &str) -> ReplReply {
        output::reply_output_lines(
            cas_session::solver_exports::evaluate_profile_cache_command_lines_on_repl_core(
                &mut self.core,
                line,
            ),
        )
    }

    /// Handle "profile" command - profiler status/toggles.
    pub(crate) fn handle_profile_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(
            cas_session::solver_exports::evaluate_profile_command_message_on_repl_core(
                &mut self.core,
                line,
            ),
        )
    }
}
