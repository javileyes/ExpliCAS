use super::*;

impl Repl {
    fn verbosity_to_steps_display_mode(verbosity: Verbosity) -> cas_session::StepsDisplayMode {
        match verbosity {
            Verbosity::None => cas_session::StepsDisplayMode::None,
            Verbosity::Succinct => cas_session::StepsDisplayMode::Succinct,
            Verbosity::Normal => cas_session::StepsDisplayMode::Normal,
            Verbosity::Verbose => cas_session::StepsDisplayMode::Verbose,
        }
    }

    fn steps_display_mode_to_verbosity(mode: cas_session::StepsDisplayMode) -> Verbosity {
        match mode {
            cas_session::StepsDisplayMode::None => Verbosity::None,
            cas_session::StepsDisplayMode::Succinct => Verbosity::Succinct,
            cas_session::StepsDisplayMode::Normal => Verbosity::Normal,
            cas_session::StepsDisplayMode::Verbose => Verbosity::Verbose,
        }
    }

    /// Handle "context" command - show or switch context mode
    pub(crate) fn handle_context_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(cas_session::evaluate_context_command_on_repl(
            line,
            &mut self.core,
            &self.config,
        ))
    }

    fn steps_command_state(&self) -> cas_session::StepsCommandState {
        cas_session::steps_command_state_for_repl_core(
            &self.core,
            Self::verbosity_to_steps_display_mode(self.verbosity),
        )
    }

    fn apply_steps_command_result_update(
        &mut self,
        set_steps_mode: Option<cas_session::StepsMode>,
        set_display_mode: Option<cas_session::StepsDisplayMode>,
    ) {
        let effects = cas_session::apply_steps_command_update_on_repl_core(
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
        match cas_session::evaluate_steps_command_input(line, self.steps_command_state()) {
            cas_session::StepsCommandResult::ShowCurrent { message } => reply_output(message),
            cas_session::StepsCommandResult::Update {
                set_steps_mode,
                set_display_mode,
                message,
            } => {
                self.apply_steps_command_result_update(set_steps_mode, set_display_mode);
                reply_output(message)
            }
            cas_session::StepsCommandResult::Invalid { message } => reply_output(message),
        }
    }

    /// Handle "autoexpand" command - show or switch auto-expand policy
    pub(crate) fn handle_autoexpand_command_core(&mut self, line: &str) -> ReplReply {
        reply_output(cas_session::evaluate_autoexpand_command_on_repl(
            line,
            &mut self.core,
            &self.config,
        ))
    }
}
