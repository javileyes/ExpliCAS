use super::*;

impl Repl {
    pub(crate) fn handle_full_simplify_core(
        &mut self,
        line: &str,
        verbosity: Verbosity,
    ) -> ReplReply {
        let lines = match cas_solver::evaluate_full_simplify_command_lines_for_display_mode(
            &mut self.core.engine,
            &self.core.state,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(lines) => lines,
            Err(message) => return reply_output(message),
        };

        // Store health report for the `health` command (if health tracking is enabled)
        self.core.last_health_report =
            cas_solver::capture_health_report_if_enabled(&self.core.engine, self.core.health_enabled);

        reply_output(lines.join("\n"))
    }
}
