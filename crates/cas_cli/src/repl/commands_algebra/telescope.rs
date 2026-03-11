use super::*;

impl Repl {
    /// Handle the 'telescope' command for proving telescoping identities like Dirichlet kernel
    pub(crate) fn handle_telescope_core(&mut self, line: &str) -> ReplReply {
        let message =
            match cas_solver::session_api::runtime::evaluate_telescope_invocation_message_on_repl_core(
                &mut self.core,
                line,
            ) {
                Ok(message) => message,
                Err(error) => return reply_output(error),
            };
        reply_output(message)
    }
}
