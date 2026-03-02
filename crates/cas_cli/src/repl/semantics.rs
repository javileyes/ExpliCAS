use super::*;

impl Repl {
    fn verbosity_to_steps_display_mode(
        verbosity: Verbosity,
    ) -> super::steps_command::StepsDisplayMode {
        match verbosity {
            Verbosity::None => super::steps_command::StepsDisplayMode::None,
            Verbosity::Succinct => super::steps_command::StepsDisplayMode::Succinct,
            Verbosity::Normal => super::steps_command::StepsDisplayMode::Normal,
            Verbosity::Verbose => super::steps_command::StepsDisplayMode::Verbose,
        }
    }

    fn steps_display_mode_to_verbosity(mode: super::steps_command::StepsDisplayMode) -> Verbosity {
        match mode {
            super::steps_command::StepsDisplayMode::None => Verbosity::None,
            super::steps_command::StepsDisplayMode::Succinct => Verbosity::Succinct,
            super::steps_command::StepsDisplayMode::Normal => Verbosity::Normal,
            super::steps_command::StepsDisplayMode::Verbose => Verbosity::Verbose,
        }
    }

    /// Handle "context" command - show or switch context mode
    pub(crate) fn handle_context_command_core(&mut self, line: &str) -> ReplReply {
        let applied =
            cas_session::evaluate_and_apply_context_command(line, self.core.state.options_mut());
        if applied.rebuild_simplifier {
            self.core.engine.simplifier =
                cas_solver::Simplifier::with_profile(self.core.state.options());
            self.sync_config_to_simplifier();
        }
        reply_output(applied.message)
    }

    fn steps_command_state(&self) -> super::steps_command::StepsCommandState {
        super::steps_command::StepsCommandState {
            steps_mode: self.core.state.options().steps_mode,
            display_mode: Self::verbosity_to_steps_display_mode(self.verbosity),
        }
    }

    fn apply_steps_command_result_update(
        &mut self,
        set_steps_mode: Option<cas_solver::StepsMode>,
        set_display_mode: Option<super::steps_command::StepsDisplayMode>,
    ) {
        let effects = super::steps_command::apply_steps_command_update(
            set_steps_mode,
            set_display_mode,
            self.core.state.options_mut(),
        );
        if let Some(mode) = effects.set_steps_mode {
            self.core.engine.simplifier.set_steps_mode(mode);
        }
        if let Some(display_mode) = effects.set_display_mode {
            self.verbosity = Self::steps_display_mode_to_verbosity(display_mode);
        }
    }

    /// Handle "steps" command - show or switch steps collection mode AND display verbosity
    /// Collection: on, off, compact (controls StepsMode in engine)
    /// Display: verbose, succinct, normal, none (controls Verbosity in CLI)
    pub(crate) fn handle_steps_command_core(&mut self, line: &str) -> ReplReply {
        match super::steps_command::evaluate_steps_command_input(line, self.steps_command_state()) {
            super::steps_command::StepsCommandResult::ShowCurrent { message } => {
                reply_output(message)
            }
            super::steps_command::StepsCommandResult::Update {
                set_steps_mode,
                set_display_mode,
                message,
            } => {
                self.apply_steps_command_result_update(set_steps_mode, set_display_mode);
                reply_output(message)
            }
            super::steps_command::StepsCommandResult::Invalid { message } => reply_output(message),
        }
    }

    /// Handle "autoexpand" command - show or switch auto-expand policy
    pub(crate) fn handle_autoexpand_command_core(&mut self, line: &str) -> ReplReply {
        let applied =
            cas_session::evaluate_and_apply_autoexpand_command(line, self.core.state.options_mut());
        if applied.rebuild_simplifier {
            self.core.engine.simplifier =
                cas_solver::Simplifier::with_profile(self.core.state.options());
            self.sync_config_to_simplifier();
        }
        reply_output(applied.message)
    }
}
