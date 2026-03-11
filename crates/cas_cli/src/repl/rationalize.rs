use super::*;

impl Repl {
    pub(crate) fn handle_rationalize_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::session_api::runtime::evaluate_rationalize_command_lines_on_repl_core(
            &mut self.core,
            line,
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }
}
