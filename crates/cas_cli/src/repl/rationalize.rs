use super::*;

impl Repl {
    pub(crate) fn handle_rationalize(&mut self, line: &str) {
        let reply = self.handle_rationalize_core(line);
        self.print_reply(reply);
    }

    fn handle_rationalize_core(&mut self, line: &str) -> ReplReply {
        let rest = match cas_solver::parse_rationalize_command_input(line) {
            cas_solver::RationalizeCommandInput::MissingInput => {
                return reply_output(cas_solver::rationalize_usage_message());
            }
            cas_solver::RationalizeCommandInput::Expr(rest) => rest,
        };

        match cas_solver::evaluate_rationalize_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => reply_output(
                cas_solver::format_rationalize_eval_lines(
                    &self.core.engine.simplifier.context,
                    &out,
                )
                .join("\n"),
            ),
            Err(e) => reply_output(cas_solver::format_rationalize_eval_error_message(&e)),
        }
    }
}
