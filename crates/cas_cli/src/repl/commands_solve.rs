use super::*;

impl Repl {
    /// Handle the 'weierstrass' command for applying Weierstrass substitution
    /// Transforms sin(x), cos(x), tan(x) into rational expressions in t = tan(x/2)
    pub(crate) fn handle_weierstrass_core(&mut self, line: &str) -> ReplReply {
        let message =
            match cas_solver::session_api::runtime::evaluate_weierstrass_invocation_message_on_repl_core(
                &mut self.core,
                line,
            ) {
                Ok(message) => message,
                Err(error) => return reply_output(error),
            };
        reply_output(message)
    }

    pub(crate) fn handle_solve_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let message =
            match cas_solver::session_api::runtime::evaluate_solve_command_message_on_repl_core(
                &mut self.core,
                line,
                Self::set_display_mode_from_verbosity(verbosity),
            ) {
                Ok(message) => message,
                Err(error) => return reply_output(error),
            };

        reply_output(message)
    }
}
