use super::*;

impl Repl {
    pub(crate) fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::session_api::analysis::evaluate_equiv_invocation_message_on_repl_core(
            &mut self.core,
            line,
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_subst_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let message = match cas_solver::session_api::substitute::evaluate_substitute_invocation_user_message_on_repl_core(
            &mut self.core,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(message) => message,
            Err(message) => return reply_output(message),
        };
        reply_output(message)
    }

    pub(crate) fn handle_explain_core(&mut self, line: &str) -> ReplReply {
        let message =
            match cas_solver::session_api::analysis::evaluate_explain_invocation_message_on_repl_core(
                &mut self.core,
                line,
            ) {
                Ok(message) => message,
                Err(message) => return reply_output(message),
            };
        reply_output(message)
    }
}
