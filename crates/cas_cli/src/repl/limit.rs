use super::*;

impl Repl {
    pub(crate) fn handle_limit_core(&mut self, line: &str) -> ReplReply {
        let complex_enabled = self.core.value_domain_is_complex();
        match cas_solver::command_api::limit::evaluate_limit_command_lines_in_domain(
            line,
            complex_enabled,
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }
}
