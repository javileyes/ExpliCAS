use super::*;

impl Repl {
    pub(crate) fn handle_health_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::session_api::runtime::evaluate_health_command_message_on_repl_core(
            &mut self.core,
            line,
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }
}
