use super::*;

impl Repl {
    /// Handle the 'expand' command for aggressive polynomial expansion
    /// Uses the engine `expand()` path which distributes without educational guards
    pub(crate) fn handle_expand_core(&mut self, line: &str) -> ReplReply {
        match cas_session::solver_exports::evaluate_expand_command_render_plan_on_repl_core(
            &mut self.core,
            line,
            self.verbosity == Verbosity::None,
        ) {
            Ok(plan) => self.render_eval_plan_to_reply(plan),
            Err(message) => reply_output(message),
        }
    }

    /// Handle the 'expand_log' command for explicit logarithm expansion
    /// Expands ln(xy) → ln(x) + ln(y), ln(x/y) → ln(x) - ln(y), ln(x^n) → n*ln(x)
    pub(crate) fn handle_expand_log_core(&mut self, line: &str) -> ReplReply {
        let message =
            match cas_session::solver_exports::evaluate_expand_log_invocation_message_on_repl_core(
                &mut self.core,
                line,
            ) {
                Ok(message) => message,
                Err(error) => return reply_output(error),
            };
        reply_output(message)
    }
}
