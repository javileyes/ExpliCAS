use super::*;

impl Repl {
    pub(crate) fn handle_rationalize_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_rationalize_command_lines_with_engine(
            &mut self.core.engine,
            line,
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }
}
