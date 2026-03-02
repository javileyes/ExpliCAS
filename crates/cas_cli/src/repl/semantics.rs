use super::*;

impl Repl {
    fn verbosity_to_steps_display_mode(verbosity: Verbosity) -> cas_solver::StepsDisplayMode {
        match verbosity {
            Verbosity::None => cas_solver::StepsDisplayMode::None,
            Verbosity::Succinct => cas_solver::StepsDisplayMode::Succinct,
            Verbosity::Normal => cas_solver::StepsDisplayMode::Normal,
            Verbosity::Verbose => cas_solver::StepsDisplayMode::Verbose,
        }
    }

    fn steps_display_mode_to_verbosity(mode: cas_solver::StepsDisplayMode) -> Verbosity {
        match mode {
            cas_solver::StepsDisplayMode::None => Verbosity::None,
            cas_solver::StepsDisplayMode::Succinct => Verbosity::Succinct,
            cas_solver::StepsDisplayMode::Normal => Verbosity::Normal,
            cas_solver::StepsDisplayMode::Verbose => Verbosity::Verbose,
        }
    }

    fn semantics_set_state(&self) -> cas_solver::SemanticsSetState {
        cas_solver::semantics_set_state_from_options(
            &self.core.simplify_options,
            self.core.state.options(),
        )
    }

    fn apply_semantics_set_state(&mut self, next: cas_solver::SemanticsSetState) {
        cas_solver::apply_semantics_set_state_to_options(
            next,
            &mut self.core.simplify_options,
            self.core.state.options_mut(),
        );
    }

    pub(crate) fn parse_semantics_set_core(&mut self, args: &[&str]) -> ReplReply {
        let current = self.semantics_set_state();
        let next = match cas_solver::evaluate_semantics_set_args(args, current) {
            Ok(next) => next,
            Err(error) => return reply_output(error),
        };

        self.apply_semantics_set_state(next);
        self.sync_config_to_simplifier();
        self.print_semantics_core()
    }

    /// Handle "context" command - show or switch context mode
    pub(crate) fn handle_context_command_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_context_command_input(
            line,
            self.core.state.options().shared.context_mode,
        ) {
            cas_solver::ContextCommandResult::ShowCurrent { message } => reply_output(message),
            cas_solver::ContextCommandResult::SetMode { mode, message } => {
                self.core.state.options_mut().shared.context_mode = mode;
                self.core.engine.simplifier =
                    cas_solver::Simplifier::with_profile(self.core.state.options());
                self.sync_config_to_simplifier();
                reply_output(message)
            }
            cas_solver::ContextCommandResult::Invalid { message } => reply_output(message),
        }
    }

    fn steps_command_state(&self) -> cas_solver::StepsCommandState {
        cas_solver::StepsCommandState {
            steps_mode: self.core.state.options().steps_mode,
            display_mode: Self::verbosity_to_steps_display_mode(self.verbosity),
        }
    }

    fn apply_steps_command_result_update(
        &mut self,
        set_steps_mode: Option<cas_solver::StepsMode>,
        set_display_mode: Option<cas_solver::StepsDisplayMode>,
    ) {
        if let Some(mode) = set_steps_mode {
            self.core.state.options_mut().steps_mode = mode;
            self.core.engine.simplifier.set_steps_mode(mode);
        }
        if let Some(display_mode) = set_display_mode {
            self.verbosity = Self::steps_display_mode_to_verbosity(display_mode);
        }
    }

    fn autoexpand_command_state(&self) -> cas_solver::AutoexpandCommandState {
        cas_solver::AutoexpandCommandState {
            policy: self.core.state.options().shared.expand_policy,
            budget: cas_solver::autoexpand_budget_view_from_options(self.core.state.options()),
        }
    }

    fn apply_autoexpand_policy(&mut self, policy: cas_solver::ExpandPolicy) {
        if self.core.state.options().shared.expand_policy != policy {
            self.core.state.options_mut().shared.expand_policy = policy;
            self.core.engine.simplifier =
                cas_solver::Simplifier::with_profile(self.core.state.options());
            self.sync_config_to_simplifier();
        }
    }

    /// Handle "steps" command - show or switch steps collection mode AND display verbosity
    /// Collection: on, off, compact (controls StepsMode in engine)
    /// Display: verbose, succinct, normal, none (controls Verbosity in CLI)
    pub(crate) fn handle_steps_command_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_steps_command_input(line, self.steps_command_state()) {
            cas_solver::StepsCommandResult::ShowCurrent { message } => reply_output(message),
            cas_solver::StepsCommandResult::Update {
                set_steps_mode,
                set_display_mode,
                message,
            } => {
                self.apply_steps_command_result_update(set_steps_mode, set_display_mode);
                reply_output(message)
            }
            cas_solver::StepsCommandResult::Invalid { message } => reply_output(message),
        }
    }

    /// Handle "autoexpand" command - show or switch auto-expand policy
    pub(crate) fn handle_autoexpand_command_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_autoexpand_command_input(line, self.autoexpand_command_state()) {
            cas_solver::AutoexpandCommandResult::ShowCurrent { message } => reply_output(message),
            cas_solver::AutoexpandCommandResult::SetPolicy { policy, message } => {
                self.apply_autoexpand_policy(policy);
                reply_output(message)
            }
            cas_solver::AutoexpandCommandResult::Invalid { message } => reply_output(message),
        }
    }
}
