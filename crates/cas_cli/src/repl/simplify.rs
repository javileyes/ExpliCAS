use super::*;

impl Repl {
    pub(crate) fn handle_full_simplify_core(
        &mut self,
        line: &str,
        verbosity: Verbosity,
    ) -> ReplReply {
        let lines = match cas_solver::evaluate_full_simplify_command_lines(
            &mut self.core.engine,
            &self.core.state,
            line,
            verbosity != Verbosity::None,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(lines) => lines,
            Err(message) => return reply_output(message),
        };

        // Store health report for the `health` command (if health tracking is enabled)
        if self.core.health_enabled {
            self.core.last_health_report =
                Some(self.core.engine.simplifier.profiler.health_report());
        }

        reply_output(lines.join("\n"))
    }
}
