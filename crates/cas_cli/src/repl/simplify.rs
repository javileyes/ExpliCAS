use super::*;

impl Repl {
    pub(crate) fn handle_full_simplify_core(
        &mut self,
        line: &str,
        verbosity: Verbosity,
    ) -> ReplReply {
        let lines =
            match cas_solver::session_api::solve::evaluate_full_simplify_command_lines_on_repl_core(
                &mut self.core,
                line,
                Self::set_display_mode_from_verbosity(verbosity),
            ) {
                Ok(lines) => lines,
                Err(error) => return reply_output(error),
            };

        // Store health report for the `health` command (if health tracking is enabled).
        cas_solver::session_api::health::update_health_report_on_repl_core(&mut self.core);

        reply_output(lines.join("\n"))
    }
}
