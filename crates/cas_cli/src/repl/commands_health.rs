use super::*;

impl Repl {
    pub(crate) fn handle_health_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_health_command_with_engine(
            &mut self.core.engine,
            line,
            self.core.last_stats.as_ref(),
            self.core.last_health_report.as_deref(),
        ) {
            Ok(out) => {
                if let Some(enabled) = out.set_enabled {
                    self.core.health_enabled = enabled;
                }
                if out.clear_last_report {
                    self.core.last_health_report = None;
                }
                reply_output(out.lines.join("\n"))
            }
            Err(message) => reply_output(message),
        }
    }
}
