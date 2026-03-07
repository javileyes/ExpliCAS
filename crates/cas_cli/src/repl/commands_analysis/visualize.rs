use super::*;

impl Repl {
    pub(crate) fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        let output = match cas_session::evaluate_visualize_invocation_output_on_repl_core(
            &mut self.core,
            line,
        ) {
            Ok(output) => output,
            Err(message) => return reply_output(message),
        };
        output::visualize_output_to_reply(output)
    }
}
