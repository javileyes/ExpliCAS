use super::*;

impl Repl {
    /// Handle the 'weierstrass' command for applying Weierstrass substitution
    /// Transforms sin(x), cos(x), tan(x) into rational expressions in t = tan(x/2)
    pub(crate) fn handle_weierstrass_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_weierstrass_command_lines_with_engine(&mut self.core.engine, line)
        {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_solve_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let lines = cas_solver::evaluate_solve_command_lines_with_session_options(
            &mut self.core.engine,
            &mut self.core.state,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
            self.core.debug_mode,
        );

        reply_output(lines.join("\n"))
    }
}
