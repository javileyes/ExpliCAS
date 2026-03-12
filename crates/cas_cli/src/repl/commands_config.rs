use super::output::{reply_output, ReplReply};
use super::*;

impl Repl {
    pub(crate) fn handle_config_core(&mut self, line: &str) -> ReplReply {
        reply_output(
            cas_session::repl_api::evaluate_and_apply_config_command_on_repl(
                line,
                &mut self.config,
                &mut self.core,
            ),
        )
    }
}
