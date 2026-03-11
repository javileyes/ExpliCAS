use super::super::*;

impl Repl {
    pub(crate) fn verbosity_to_steps_display_mode(
        verbosity: Verbosity,
    ) -> cas_session::solver_exports::StepsDisplayMode {
        match verbosity {
            Verbosity::None => cas_session::solver_exports::StepsDisplayMode::None,
            Verbosity::Succinct => cas_session::solver_exports::StepsDisplayMode::Succinct,
            Verbosity::Normal => cas_session::solver_exports::StepsDisplayMode::Normal,
            Verbosity::Verbose => cas_session::solver_exports::StepsDisplayMode::Verbose,
        }
    }

    fn steps_display_mode_to_verbosity(
        mode: cas_session::solver_exports::StepsDisplayMode,
    ) -> Verbosity {
        match mode {
            cas_session::solver_exports::StepsDisplayMode::None => Verbosity::None,
            cas_session::solver_exports::StepsDisplayMode::Succinct => Verbosity::Succinct,
            cas_session::solver_exports::StepsDisplayMode::Normal => Verbosity::Normal,
            cas_session::solver_exports::StepsDisplayMode::Verbose => Verbosity::Verbose,
        }
    }

    fn steps_command_state(&self) -> cas_session::solver_exports::StepsCommandState {
        cas_session::solver_exports::steps_command_state_for_repl_core(
            &self.core,
            Self::verbosity_to_steps_display_mode(self.verbosity),
        )
    }

    fn apply_steps_command_result_update(
        &mut self,
        set_steps_mode: Option<cas_session::StepsMode>,
        set_display_mode: Option<cas_session::solver_exports::StepsDisplayMode>,
    ) {
        let effects = cas_session::solver_exports::apply_steps_command_update_on_repl_core(
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
        match cas_session::solver_exports::evaluate_steps_command_input(
            line,
            self.steps_command_state(),
        ) {
            cas_session::solver_exports::StepsCommandResult::ShowCurrent { message } => {
                reply_output(message)
            }
            cas_session::solver_exports::StepsCommandResult::Update {
                set_steps_mode,
                set_display_mode,
                message,
            } => {
                self.apply_steps_command_result_update(set_steps_mode, set_display_mode);
                reply_output(message)
            }
            cas_session::solver_exports::StepsCommandResult::Invalid { message } => {
                reply_output(message)
            }
        }
    }
}
