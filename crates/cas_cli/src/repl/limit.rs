use super::*;

impl Repl {
    pub(crate) fn handle_limit(&mut self, line: &str) {
        let reply = self.handle_limit_core(line);
        self.print_reply(reply);
    }

    fn handle_limit_core(&mut self, line: &str) -> ReplReply {
        let rest = cas_solver::extract_limit_command_tail(line);
        if rest.is_empty() {
            return reply_output(cas_solver::limit_usage_message());
        }

        match cas_solver::evaluate_limit_command_input(rest) {
            Ok(limit_result) => {
                reply_output(cas_solver::format_limit_command_eval_lines(&limit_result).join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_limit_command_error_message(&e)),
        }
    }
}
