use super::*;

impl Repl {
    pub(crate) fn handle_limit_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_limit_command_lines(line) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }
}
