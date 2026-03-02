use super::output::{reply_output, ReplReply};
use super::*;

impl Repl {
    pub(crate) fn handle_config_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_config_command(line, self.config_as_solver_toggle()) {
            cas_solver::ConfigCommandResult::ShowList { message } => reply_output(message),
            cas_solver::ConfigCommandResult::SaveRequested => match self.config.save() {
                Ok(_) => reply_output("Configuration saved to cas_config.toml"),
                Err(e) => reply_output(format!("Error saving configuration: {}", e)),
            },
            cas_solver::ConfigCommandResult::RestoreRequested => {
                self.config = CasConfig::restore();
                self.sync_config_to_simplifier();
                reply_output("Configuration restored to defaults.")
            }
            cas_solver::ConfigCommandResult::ApplyToggleConfig { toggles, message } => {
                self.set_config_from_solver_toggle(toggles);
                self.sync_config_to_simplifier();
                reply_output(message)
            }
            cas_solver::ConfigCommandResult::Error { message } => reply_output(message),
        }
    }
}
