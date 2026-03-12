use super::super::*;

impl Repl {
    pub(crate) fn verbosity_to_steps_display_mode(
        verbosity: Verbosity,
    ) -> cas_solver::session_api::settings::StepsDisplayMode {
        match verbosity {
            Verbosity::None => cas_solver::session_api::settings::StepsDisplayMode::None,
            Verbosity::Succinct => cas_solver::session_api::settings::StepsDisplayMode::Succinct,
            Verbosity::Normal => cas_solver::session_api::settings::StepsDisplayMode::Normal,
            Verbosity::Verbose => cas_solver::session_api::settings::StepsDisplayMode::Verbose,
        }
    }

    fn steps_display_mode_to_verbosity(
        mode: cas_solver::session_api::settings::StepsDisplayMode,
    ) -> Verbosity {
        match mode {
            cas_solver::session_api::settings::StepsDisplayMode::None => Verbosity::None,
            cas_solver::session_api::settings::StepsDisplayMode::Succinct => Verbosity::Succinct,
            cas_solver::session_api::settings::StepsDisplayMode::Normal => Verbosity::Normal,
            cas_solver::session_api::settings::StepsDisplayMode::Verbose => Verbosity::Verbose,
        }
    }

    fn steps_command_state(&self) -> cas_solver::session_api::settings::StepsCommandState {
        cas_solver::session_api::settings::steps_command_state_for_repl_core(
            &self.core,
            Self::verbosity_to_steps_display_mode(self.verbosity),
        )
    }

    fn apply_steps_command_result_update(
        &mut self,
        set_steps_mode: Option<cas_solver::runtime::StepsMode>,
        set_display_mode: Option<cas_solver::session_api::settings::StepsDisplayMode>,
    ) {
        let effects = cas_solver::session_api::settings::apply_steps_command_update_on_repl_core(
            &mut self.core,
            set_steps_mode,
            set_display_mode,
        );
        if let Some(display_mode) = effects.set_display_mode {
            self.verbosity = Self::steps_display_mode_to_verbosity(display_mode);
        }
    }

    /// Handle "steps" command - show or switch steps collection mode AND display verbosity
    /// Collection: on, off, compact (controls StepsMode in engine)
    /// Display: verbose, succinct, normal, none (controls Verbosity in CLI)
    pub(crate) fn handle_steps_command_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::session_api::settings::evaluate_steps_command_input(
            line,
            self.steps_command_state(),
        ) {
            cas_solver::session_api::settings::StepsCommandResult::ShowCurrent { message } => {
                reply_output(message)
            }
            cas_solver::session_api::settings::StepsCommandResult::Update {
                set_steps_mode,
                set_display_mode,
                message,
            } => {
                self.apply_steps_command_result_update(set_steps_mode, set_display_mode);
                reply_output(message)
            }
            cas_solver::session_api::settings::StepsCommandResult::Invalid { message } => {
                reply_output(message)
            }
        }
    }
}
